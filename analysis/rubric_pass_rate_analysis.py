"""
rubric_pass_rate_analysis.py

Compute pass rates for agent runs based on Docent rubric labels.
Given:
- A collection_id
- A rubric_id

Outputs:
- # total runs
- # runs labeled match / non-match
- pass rate for each group
"""

import json
from pathlib import Path
import os
import typer

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

# Load .env from repo root if available
repo_root = Path(__file__).resolve().parents[1]
env_path = repo_root / ".env"
if load_dotenv and env_path.exists():
    load_dotenv(dotenv_path=str(env_path))

from docent import Docent

app = typer.Typer(help="Compute pass rates for agent runs based on Docent rubric labels")


# ---------------------------------------------------------
# Dump raw data for analysis
# ---------------------------------------------------------

def dump_raw_data(client: Docent, collection_id: str, rubric_id: str) -> None:
    """Dump raw API responses to JSON files in the same directory as this script."""
    script_dir = Path(__file__).parent
    
    # Try to get rubric run state - check what methods are available
    print("Checking available client methods...")
    client_methods = [m for m in dir(client) if not m.startswith("_")]
    print(f"Available methods: {', '.join(sorted(client_methods))}")
    print()
    
    # Try to fetch rubric run state
    print("Fetching rubric run state...")
    try:
        # Try different possible method names
        if hasattr(client, "get_rubric_run_state"):
            rubric_state = client.get_rubric_run_state(collection_id, rubric_id)
        elif hasattr(client, "get_rubric_state"):
            rubric_state = client.get_rubric_state(collection_id, rubric_id)
        else:
            print("Warning: Could not find rubric state method. Trying to inspect collection...")
            rubric_state = None
    except Exception as e:
        print(f"Error fetching rubric state: {e}")
        rubric_state = None
    
    if rubric_state is not None:
        rubric_file = script_dir / "rubric_run_state.json"
        with open(rubric_file, "w") as f:
            json.dump(rubric_state, f, indent=2, default=str)
        print(f"Saved rubric run state to: {rubric_file}")
        results_count = len(rubric_state.get("results", [])) if isinstance(rubric_state, dict) else 0
        print(f"Found {results_count} rubric results")
    else:
        print("Could not fetch rubric state - check available methods above")
    
    # Try to get agent runs
    print("\nFetching agent runs...")
    try:
        # Try different possible method names
        if hasattr(client, "get_agent_runs"):
            runs = client.get_agent_runs(collection_id)
        elif hasattr(client, "list_agent_runs"):
            runs = client.list_agent_runs(collection_id)
        elif hasattr(client, "get_collection"):
            collection = client.get_collection(collection_id)
            runs = collection.get("agent_runs", []) if isinstance(collection, dict) else []
        else:
            print("Warning: Could not find agent runs method.")
            runs = None
    except Exception as e:
        print(f"Error fetching agent runs: {e}")
        runs = None
    
    if runs is not None:
        runs_file = script_dir / "agent_runs.json"
        with open(runs_file, "w") as f:
            json.dump(runs, f, indent=2, default=str)
        print(f"Saved agent runs to: {runs_file}")
        runs_count = len(runs) if isinstance(runs, list) else 0
        print(f"Found {runs_count} agent runs")
    else:
        print("Could not fetch agent runs - check available methods above")
    
    print(f"\nFiles saved in: {script_dir}")


# ---------------------------------------------------------
# Extract rubric labels from rubric state
# ---------------------------------------------------------

def extract_rubric_labels(rubric_state: dict) -> dict[str, str]:
    """Extract agent_run_id -> label mapping from rubric state.
    
    Returns dict mapping agent_run_id to "match" or "non-match" (or other label).
    """
    rubric_labels = {}
    
    if not isinstance(rubric_state, dict) or "results" not in rubric_state:
        return rubric_labels
    
    for result_group in rubric_state["results"]:
        if not isinstance(result_group, dict):
            continue
        
        agent_run_id = result_group.get("agent_run_id")
        if not agent_run_id:
            continue
        
        # Skip if we already have a label for this run
        if agent_run_id in rubric_labels:
            continue
        
        # Look for label in nested results array
        results = result_group.get("results", [])
        for result in results:
            if not isinstance(result, dict):
                continue
            
            # Try to get label from output
            output = result.get("output")
            if isinstance(output, dict):
                label = output.get("label")
                if label:
                    rubric_labels[agent_run_id] = label
                    break
            
            # Fall back to result_metadata
            if agent_run_id not in rubric_labels:
                result_metadata = result.get("result_metadata", {})
                final_results = result_metadata.get("final_results", [])
                if final_results and isinstance(final_results[0], dict):
                    label = final_results[0].get("label")
                    if label:
                        rubric_labels[agent_run_id] = label
                        break
    
    return rubric_labels


# ---------------------------------------------------------
# Get individual agent run status
# ---------------------------------------------------------

def get_agent_run_status(client: Docent, collection_id: str, agent_run_id: str) -> bool | None:
    """Fetch individual agent run and check if swebench_resolved is true.
    
    Args:
        client: Docent client instance
        collection_id: Collection ID
        agent_run_id: Agent run ID to fetch
    
    Returns:
        True if swebench_resolved == true, False if false, None if cannot determine
    """
    try:
        # get_agent_run requires both collection_id and agent_run_id
        run = client.get_agent_run(collection_id, agent_run_id)
        
        if run is None:
            return None
        
        # get_agent_run returns an AgentRun object, not a dict
        # Access metadata attribute
        metadata = getattr(run, "metadata", None)
        if metadata is None:
            return None
        
        # metadata might be a dict or an object with attributes
        if isinstance(metadata, dict):
            swebench_resolved = metadata.get("swebench_resolved")
        else:
            swebench_resolved = getattr(metadata, "swebench_resolved", None)
        
        # Return True if explicitly True, False if explicitly False, None otherwise
        if swebench_resolved is True:
            return True
        elif swebench_resolved is False:
            return False
        else:
            return None
            
    except Exception as e:
        print(f"  Warning: Could not fetch run {agent_run_id}: {e}")
        return None


# ---------------------------------------------------------
# Compute pass rates
# ---------------------------------------------------------

def compute_passrate(run_ids: list[str], run_scores: dict[str, bool | None]) -> tuple[float, int, int]:
    """Compute resolved rate for a list of run IDs.
    
    Returns:
        (resolved_rate, resolved_count, total_count) tuple
        resolved_rate = P(resolved | label) = resolved_count / total_count
    """
    if not run_ids:
        return (0.0, 0, 0)
    # Count resolved runs (True) vs all runs in the group
    # Runs without status (None) are treated as not resolved
    resolved = sum(1 for run_id in run_ids if run_scores.get(run_id) is True)
    total = len(run_ids)
    return (resolved / total if total > 0 else 0.0, resolved, total)


@app.command()
def dump(
    collection_id: str = typer.Option(
        os.getenv("DOCENT_COLLECTION_ID", ""),
        "--collection-id",
        "-c",
        help="Docent collection ID (or set DOCENT_COLLECTION_ID env var)",
    ),
    rubric_id: str = typer.Option(
        os.getenv("DOCENT_RUBRIC_ID", ""),
        "--rubric-id",
        "-r",
        help="Docent rubric ID (or set DOCENT_RUBRIC_ID env var)",
    ),
    api_key: str = typer.Option(
        os.getenv("DOCENT_API_KEY", ""),
        "--api-key",
        "-k",
        help="Docent API key (or set DOCENT_API_KEY env var, can be omitted if in .env)",
    ),
) -> None:
    """Dump raw API responses to JSON files for analysis."""
    if not api_key:
        api_key = os.getenv("DOCENT_API_KEY", "")
    if not api_key:
        raise typer.BadParameter("DOCENT_API_KEY must be provided via env var or --api-key")
    if not collection_id:
        raise typer.BadParameter("Collection ID must be provided via env var DOCENT_COLLECTION_ID or --collection-id")
    if not rubric_id:
        raise typer.BadParameter("Rubric ID must be provided via env var DOCENT_RUBRIC_ID or --rubric-id")

    client = Docent(api_key=api_key)
    dump_raw_data(client, collection_id, rubric_id)


@app.command()
def main(
    collection_id: str = typer.Option(
        os.getenv("DOCENT_COLLECTION_ID", ""),
        "--collection-id",
        "-c",
        help="Docent collection ID (or set DOCENT_COLLECTION_ID env var)",
    ),
    rubric_id: str = typer.Option(
        os.getenv("DOCENT_RUBRIC_ID", ""),
        "--rubric-id",
        "-r",
        help="Docent rubric ID (or set DOCENT_RUBRIC_ID env var)",
    ),
    api_key: str = typer.Option(
        os.getenv("DOCENT_API_KEY", ""),
        "--api-key",
        "-k",
        help="Docent API key (or set DOCENT_API_KEY env var, can be omitted if in .env)",
    ),
    dump_data: bool = typer.Option(
        False,
        "--dump",
        "-d",
        help="Dump raw API responses to JSON files before analysis",
    ),
) -> None:
    """Compute pass rates for agent runs based on Docent rubric labels."""
    if not api_key:
        api_key = os.getenv("DOCENT_API_KEY", "")
    if not api_key:
        raise typer.BadParameter("DOCENT_API_KEY must be provided via env var or --api-key")
    if not collection_id:
        raise typer.BadParameter("Collection ID must be provided via env var DOCENT_COLLECTION_ID or --collection-id")
    if not rubric_id:
        raise typer.BadParameter("Rubric ID must be provided via env var DOCENT_RUBRIC_ID or --rubric-id")

    script_dir = Path(__file__).parent
    client = Docent(api_key=api_key)

    if dump_data:
        dump_raw_data(client, collection_id, rubric_id)
        print("\n" + "=" * 60 + "\n")

    # Load rubric state (from file if available, otherwise fetch)
    rubric_file = script_dir / "rubric_run_state.json"
    if rubric_file.exists():
        print(f"Loading rubric data from file: {rubric_file}")
        with open(rubric_file, "r") as f:
            rubric_state = json.load(f)
    else:
        print("Fetching rubric data from API...")
        try:
            if hasattr(client, "get_rubric_run_state"):
                rubric_state = client.get_rubric_run_state(collection_id, rubric_id)
            elif hasattr(client, "get_rubric_state"):
                rubric_state = client.get_rubric_state(collection_id, rubric_id)
            else:
                raise typer.BadParameter("Could not find method to get rubric state")
        except Exception as e:
            raise typer.BadParameter(f"Error fetching rubric state: {e}")
    
    # Extract labels from rubric state
    print("Extracting rubric labels...")
    rubric_labels = extract_rubric_labels(rubric_state)
    print(f"Found {len(rubric_labels)} agent runs with rubric labels")
    
    # Split by label
    match_runs = [run_id for run_id, label in rubric_labels.items() if label == "match"]
    nonmatch_runs = [run_id for run_id, label in rubric_labels.items() if label != "match"]
    
    print(f"Runs labeled MATCH: {len(match_runs)}")
    print(f"Runs labeled NON-MATCH: {len(nonmatch_runs)}")
    
    # Fetch status for each agent run
    print("\nFetching agent run statuses (checking swebench_resolved)...")
    run_scores: dict[str, bool | None] = {}
    all_run_ids = list(rubric_labels.keys())
    
    for i, run_id in enumerate(all_run_ids, 1):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(all_run_ids)}")
        status = get_agent_run_status(client, collection_id, run_id)
        run_scores[run_id] = status
    
    # Count how many we successfully determined
    determined = sum(1 for s in run_scores.values() if s is not None)
    print(f"\nSuccessfully determined status for {determined}/{len(all_run_ids)} runs")
    
    # Compute resolved rates
    match_rate, match_resolved, match_total = compute_passrate(match_runs, run_scores)
    nonmatch_rate, nonmatch_resolved, nonmatch_total = compute_passrate(nonmatch_runs, run_scores)
    
    # Count runs with status for reporting
    match_with_status = sum(1 for run_id in match_runs if run_scores.get(run_id) is not None)
    nonmatch_with_status = sum(1 for run_id in nonmatch_runs if run_scores.get(run_id) is not None)
    
    # Print report
    print("\n" + "=" * 60)
    print(" Rubric Evaluation Summary")
    print("=" * 60)
    print(f"Total runs evaluated: {len(rubric_labels)}")
    print(f"Runs labeled MATCH:     {len(match_runs)}")
    print(f"Runs labeled NON-MATCH: {len(nonmatch_runs)}")
    print("-" * 60)
    print(f"MATCH runs with status: {match_with_status}/{len(match_runs)}")
    print(f"NON-MATCH runs with status: {nonmatch_with_status}/{len(nonmatch_runs)}")
    print("-" * 60)
    print(f"Resolved rate | MATCH:     {match_rate:.3f} ({match_resolved}/{match_total} resolved)")
    print(f"Resolved rate | NON-MATCH: {nonmatch_rate:.3f} ({nonmatch_resolved}/{nonmatch_total} resolved)")
    print("=" * 60)


if __name__ == "__main__":
    app()