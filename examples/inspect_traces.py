#!/usr/bin/env python3
"""Diagnostic script to inspect MLflow traces and extract their tags/metadata."""

from pathlib import Path
import json
import sys
import typer

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from integrations.docent import load_mlflow_traces

app = typer.Typer(help="Inspect MLflow traces and extract tags/metadata to JSON")


@app.command()
def main(
    experiment_name: str = typer.Argument(..., help="MLflow experiment name"),
    output_file: Path = typer.Option("trace_inspection.json", "-o", "--output", help="Output JSON file path"),
    tracking_uri: str = typer.Option("http://127.0.0.1:5000", "--tracking-uri", "-u", help="MLflow tracking URI"),
    max_results: int = typer.Option(10, "--max-results", "-m", help="Maximum number of traces to inspect"),
    filter_string: str = typer.Option(None, "--filter", "-f", help="MLflow filter string"),
) -> None:
    """Inspect MLflow traces and extract all tags, metadata, and task information to JSON."""
    
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
    
    # Extract information from each trace
    trace_data = []
    
    for i, trace in enumerate(traces):
        trace_info = trace.info
        tags = trace_info.tags if hasattr(trace_info, "tags") and isinstance(trace_info.tags, dict) else {}
        
        # Extract task/prediction from spans if possible
        task_description = None
        final_prediction = None
        
        def extract_task_from_spans(spans, depth=0):
            """Recursively extract task description and final prediction from spans."""
            nonlocal task_description, final_prediction
            
            for span in spans:
                span_name = getattr(span, "name", "unknown")
                span_inputs = getattr(span, "inputs", None)
                span_outputs = getattr(span, "outputs", None)
                span_attributes = getattr(span, "attributes", {})
                
                # Normalize inputs/outputs from attributes if needed
                if span_inputs in (None, {}, []):
                    span_inputs = span_attributes.get("mlflow.spanInputs") if isinstance(span_attributes, dict) else None
                if span_outputs in (None, {}, []):
                    span_outputs = span_attributes.get("mlflow.spanOutputs") if isinstance(span_attributes, dict) else None
                
                # Look for task description in inputs
                if span_inputs and not task_description:
                    if isinstance(span_inputs, dict):
                        # Try common keys for task description
                        for key in ["task_description", "task", "question", "query", "prompt", "input"]:
                            if key in span_inputs:
                                val = span_inputs[key]
                                if isinstance(val, str) and val.strip():
                                    task_description = val[:1000]  # Limit length
                                    break
                
                # Look for final prediction in outputs
                if span_outputs and not final_prediction:
                    if isinstance(span_outputs, dict):
                        for key in ["solution", "answer", "completion", "result", "output"]:
                            if key in span_outputs:
                                val = span_outputs[key]
                                if isinstance(val, str) and val.strip():
                                    final_prediction = val[:1000]  # Limit length
                                    break
                
                # Recurse into child spans
                if hasattr(span, "child_spans") and span.child_spans:
                    extract_task_from_spans(span.child_spans, depth + 1)
        
        # Extract from spans if available
        if hasattr(trace, "data") and hasattr(trace.data, "spans"):
            extract_task_from_spans(trace.data.spans)
        
        trace_entry = {
            "trace_id": getattr(trace_info, "trace_id", None),
            "status": getattr(trace_info, "status", None),
            "execution_time_ms": getattr(trace_info, "execution_time_ms", None),
            "timestamp_ms": getattr(trace_info, "timestamp_ms", None),
            "experiment_id": getattr(trace_info, "experiment_id", None),
            "request_id": getattr(trace_info, "request_id", None),
            "tags": dict(tags) if tags else {},
            "task_description": task_description,
            "final_prediction": final_prediction,
            "tag_keys": list(tags.keys()) if tags else [],
            "has_instance_id": "instance_id" in tags if tags else False,
            "instance_id": tags.get("instance_id") if tags else None,
        }
        
        trace_data.append(trace_entry)
        
        # Print summary
        print(f"\nTrace {i+1}: {trace_entry['trace_id']}")
        print(f"  Status: {trace_entry['status']}")
        print(f"  Tags: {len(trace_entry['tags'])} tags")
        print(f"  Has instance_id: {trace_entry['has_instance_id']}")
        if trace_entry['instance_id']:
            print(f"  instance_id: {trace_entry['instance_id']}")
        if trace_entry['task_description']:
            print(f"  Task (first 100 chars): {trace_entry['task_description'][:100]}...")
        if trace_entry['final_prediction']:
            print(f"  Prediction (first 100 chars): {trace_entry['final_prediction'][:100]}...")
    
    # Save to JSON file
    output_data = {
        "experiment_name": experiment_name,
        "tracking_uri": tracking_uri,
        "total_traces": len(trace_data),
        "traces_with_instance_id": sum(1 for t in trace_data if t["has_instance_id"]),
        "traces": trace_data,
    }
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total traces: {len(trace_data)}")
    print(f"  Traces with instance_id tag: {output_data['traces_with_instance_id']}")
    print(f"  Traces without instance_id tag: {len(trace_data) - output_data['traces_with_instance_id']}")
    print(f"\nDetailed inspection saved to: {output_file}")


if __name__ == "__main__":
    app()

