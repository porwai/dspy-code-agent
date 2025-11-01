from pathlib import Path
import json
import sys
import mlflow
from mlflow.tracking import MlflowClient

def view_trace_tags(trace_id: str, tracking_uri: str = "http://127.0.0.1:5000", output_dir: Path | None = None) -> dict:
    """Get tags for a specific trace ID from MLflow and optionally save to JSON.
    
    Args:
        trace_id: The trace ID to query (e.g., "tr-b7e99acaa97065e18e46d534c88a618e")
        tracking_uri: MLflow tracking URI (default: http://127.0.0.1:5000)
        output_dir: Optional directory to save JSON output (default: examples/output)
        
    Returns:
        Dictionary containing trace tags and metadata
    """
    # Set tracking URI before creating client
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    
    # Get trace by ID (use trace_id parameter instead of deprecated request_id)
    trace = client.get_trace(trace_id=trace_id)
    
    # Extract tags from trace info
    tags = getattr(trace.info, "tags", {}) or {}
    
    # Build result dictionary with trace metadata and tags
    result = {
        "trace_id": getattr(trace.info, "trace_id", trace_id),
        "status": getattr(trace.info, "status", None),
        "execution_time_ms": getattr(trace.info, "execution_time_ms", None),
        "experiment_id": getattr(trace.info, "experiment_id", None),
        "timestamp_ms": getattr(trace.info, "timestamp_ms", None),
        "request_id": getattr(trace.info, "request_id", None),
        "tags": tags,
    }
    
    # Print to console
    print(json.dumps(result, indent=2, default=str))
    
    # Save to JSON file
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"trace_tags_{trace_id}.json"
    output_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nSaved tags to: {output_path}")
    
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python view_tags.py <trace_id> [tracking_uri]")
        print("Example: python view_tags.py tr-b7e99acaa97065e18e46d534c88a618e")
        sys.exit(1)
    
    trace_id = sys.argv[1]
    tracking_uri = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:5000"
    
    view_trace_tags(trace_id, tracking_uri)