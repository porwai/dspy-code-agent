#!/usr/bin/env python3
"""Upload MLflow DSPy traces to Docent with SWE-bench results integration."""

from pathlib import Path
import json
import sys
import os
import typer

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

# Ensure project root is on sys.path so top-level "integrations" imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Load .env from repo root if available
repo_root = Path(__file__).resolve().parents[1]
env_path = repo_root / ".env"
if load_dotenv and env_path.exists():
    load_dotenv(dotenv_path=str(env_path))

from docent import Docent
from integrations.docent import load_mlflow_traces, load_dspy_trace_to_docent

app = typer.Typer(help="Upload MLflow DSPy traces to Docent with SWE-bench results integration")


def load_preds_json(preds_path: Path) -> dict[str, dict]:
    """Load preds.json and return as dict mapping instance_id to prediction data.
    
    Args:
        preds_path: Path to preds.json file
        
    Returns:
        Dict mapping instance_id -> {model_patch, model_name_or_path, ...}
    """
    if not preds_path.exists():
        raise FileNotFoundError(f"preds.json not found: {preds_path}")
    
    with open(preds_path, "r") as f:
        return json.load(f)


def load_evaluation_output(eval_path: Path) -> dict[str, list[str]]:
    """Load evaluation_output JSON and extract resolved/unresolved instance IDs.
    
    Args:
        eval_path: Path to evaluation_output JSON file
        
    Returns:
        Dict with keys: 'resolved_ids', 'unresolved_ids', 'submitted_ids'
    """
    if not eval_path.exists():
        raise FileNotFoundError(f"evaluation_output not found: {eval_path}")
    
    with open(eval_path, "r") as f:
        data = json.load(f)
    
    return {
        "resolved_ids": set(data.get("resolved_ids", [])),
        "unresolved_ids": set(data.get("unresolved_ids", [])),
        "submitted_ids": set(data.get("submitted_ids", [])),
    }


def enhance_metadata_with_swebench(
    trace,
    preds_data: dict[str, dict] | None,
    eval_data: dict[str, set[str]] | None,
) -> dict:
    """Enhance trace metadata with SWE-bench results (model_patch and pass/fail status).
    
    Args:
        trace: MLflow Trace object
        preds_data: Dict mapping instance_id to prediction data (from preds.json)
        eval_data: Dict with resolved_ids, unresolved_ids, submitted_ids sets
        
    Returns:
        Enhanced metadata dict with model_patch and swebench_status fields
    """
    metadata = {}
    
    # Extract instance_id from trace tags
    tags = trace.info.tags if hasattr(trace.info, "tags") and isinstance(trace.info.tags, dict) else {}
    instance_id = tags.get("instance_id")
    
    if not instance_id:
        return metadata
    
    # Lookup model_patch from preds.json
    if preds_data and instance_id in preds_data:
        pred_entry = preds_data[instance_id]
        metadata["model_patch"] = pred_entry.get("model_patch")
        metadata["model_name_or_path"] = pred_entry.get("model_name_or_path")
    else:
        metadata["model_patch"] = None
    
    # Determine SWE-bench status from evaluation_output
    if eval_data:
        if instance_id in eval_data["resolved_ids"]:
            metadata["swebench_status"] = "passed"
            metadata["swebench_resolved"] = True
        elif instance_id in eval_data["unresolved_ids"]:
            metadata["swebench_status"] = "failed"
            metadata["swebench_resolved"] = False
        elif instance_id in eval_data["submitted_ids"]:
            metadata["swebench_status"] = "not_resolved"
            metadata["swebench_resolved"] = False
        else:
            metadata["swebench_status"] = "not_evaluated"
            metadata["swebench_resolved"] = None
    else:
        metadata["swebench_status"] = None
        metadata["swebench_resolved"] = None
    
    return metadata


@app.command()
def main(
    experiment_name: str = typer.Argument(..., help="MLflow experiment name"),
    preds_json: Path = typer.Option(None, "--preds", "-p", help="Path to preds.json file"),
    evaluation_output: Path = typer.Option(None, "--evaluation", "-e", help="Path to evaluation_output JSON file"),
    tracking_uri: str = typer.Option("http://127.0.0.1:5000", "--tracking-uri", "-u", help="MLflow tracking URI"),
    collection_name: str = typer.Option(None, "--collection", "-c", help="Docent collection name (default: 'SWE-bench {experiment_name}')"),
    max_results: int = typer.Option(100, "--max-results", "-m", help="Maximum number of traces to load"),
    filter_string: str = typer.Option(None, "--filter", "-f", help="MLflow filter string (e.g., \"status = 'OK'\")"),
) -> None:
    """Upload MLflow DSPy traces to Docent with SWE-bench results integration.
    
    Matches traces with preds.json (to get model_patch) and evaluation_output.json
    (to determine pass/fail status) using instance_id from trace tags.
    """
    # Load preds.json if provided
    preds_data = None
    if preds_json:
        try:
            preds_data = load_preds_json(preds_json)
            print(f"Loaded {len(preds_data)} entries from preds.json")
        except Exception as e:
            print(f"Warning: Failed to load preds.json: {e}", file=sys.stderr)
            preds_data = None
    
    # Load evaluation_output.json if provided
    eval_data = None
    if evaluation_output:
        try:
            eval_data = load_evaluation_output(evaluation_output)
            total_eval = len(eval_data["resolved_ids"]) + len(eval_data["unresolved_ids"]) + len(eval_data["submitted_ids"])
            print(f"Loaded evaluation data: {len(eval_data['resolved_ids'])} resolved, {len(eval_data['unresolved_ids'])} unresolved, {total_eval} total")
        except Exception as e:
            print(f"Warning: Failed to load evaluation_output: {e}", file=sys.stderr)
            eval_data = None
    
    # Load MLflow traces
    print(f"Loading traces from experiment '{experiment_name}'...")
    try:
        traces = load_mlflow_traces(
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            filter_string=filter_string,
            order_by=["timestamp_ms DESC"],
            max_results=max_results,
            return_type="list",
        )
        print(f"Found {len(traces)} traces")
    except Exception as e:
        print(f"Error loading traces: {e}", file=sys.stderr)
        raise typer.Exit(1)
    
    # Convert traces to AgentRun objects with enhanced metadata
    agent_runs = []
    matched_count = 0
    unmatched_count = 0
    
    for trace in traces:
        # Extract instance_id for matching
        tags = trace.info.tags if hasattr(trace.info, "tags") and isinstance(trace.info.tags, dict) else {}
        instance_id = tags.get("instance_id")
        
        # Enhance metadata with SWE-bench results
        enhanced_metadata = enhance_metadata_with_swebench(trace, preds_data, eval_data)
        
        if instance_id:
            if (preds_data and instance_id in preds_data) or (eval_data and instance_id in (eval_data["resolved_ids"] | eval_data["unresolved_ids"] | eval_data["submitted_ids"])):
                matched_count += 1
        else:
            unmatched_count += 1
        
        # Build final metadata (merge enhanced metadata with trace_id)
        metadata = {
            "task_id": instance_id or trace.info.trace_id,
            "trace_id": trace.info.trace_id,
            **enhanced_metadata,
        }
        
        # Convert to AgentRun
        try:
            agent_run = load_dspy_trace_to_docent(trace, metadata=metadata)
            agent_runs.append(agent_run)
        except Exception as e:
            print(f"Warning: Failed to convert trace {trace.info.trace_id}: {e}", file=sys.stderr)
    
    print(f"Converted {len(agent_runs)} traces to AgentRun objects ({matched_count} matched with SWE-bench data, {unmatched_count} without instance_id)")
    
    # Create Docent client and upload
    api_key = os.getenv("DOCENT_API_KEY")
    if not api_key:
        print("Error: DOCENT_API_KEY environment variable not set", file=sys.stderr)
        raise typer.Exit(1)
    
    client = Docent(api_key=api_key)
    
    # Determine collection name
    if collection_name is None:
        collection_name = f"SWE-bench {experiment_name}"
    
    collection_id = client.create_collection(
        name=collection_name,
        description=f"Agent runs from MLflow DSPy traces (experiment: {experiment_name})",
    )
    
    print(f"Created/using Docent collection: {collection_name} (id: {collection_id})")
    print(f"Uploading {len(agent_runs)} agent runs...")
    
    client.add_agent_runs(collection_id, agent_runs)
    
    print(f"Successfully uploaded {len(agent_runs)} agent runs to Docent")


if __name__ == "__main__":
    app()
