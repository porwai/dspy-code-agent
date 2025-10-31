from pathlib import Path
import sys

# Ensure project root is on sys.path so top-level "integrations" imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from integrations.docent import load_mlflow_traces, load_dspy_trace_to_docent

# 1) Pull traces from MLflow
traces = load_mlflow_traces(
    tracking_uri="http://127.0.0.1:5000",
    experiment_name="DSPy",
    filter_string="status = 'OK'",
    order_by=["timestamp_ms DESC"],
    max_results=5,
    return_type="list",
)
assert traces, "No MLflow traces found"

# 2) Convert the newest trace to a Docent AgentRun, preserving metadata
trace = traces[1]
agent_run = load_dspy_trace_to_docent(
    trace,
    metadata={
        "task_id": trace.info.trace_id,   # optional extra metadata you want to attach
        "model": "dspy-agent",            # optional override
    }
)

# 3) Print full transcript and metadata
import json

print("Trace ID:", trace.info.trace_id)
print("Messages:", len(agent_run.transcripts[0].messages))
print("\n=== Metadata ===")
print(json.dumps(agent_run.metadata, indent=2, default=str))

print("\n=== Transcript ===")
for i, m in enumerate(agent_run.transcripts[0].messages):
    print(f"\n[{i}] role={getattr(m, 'role', 'unknown')}")
    if getattr(m, 'name', None):
        print(f"name={m.name}")
    if getattr(m, 'tool_call_id', None):
        print(f"tool_call_id={m.tool_call_id}")
    tool_calls = getattr(m, 'tool_calls', None)
    if tool_calls:
        calls = [{"id": tc.id, "function": tc.function, "arguments": tc.arguments} for tc in tool_calls]
        print("tool_calls=", json.dumps(calls, indent=2, default=str))
    content = getattr(m, 'content', '')
    print(content if content is not None else "")

# 4) Store to examples/output
out_dir = Path(__file__).resolve().parent / "output"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"agent_run_{trace.info.trace_id}.json"

def message_to_dict(msg):
    d = {
        "role": getattr(msg, "role", None),
        "content": getattr(msg, "content", None),
    }
    if getattr(msg, "name", None):
        d["name"] = msg.name
    if getattr(msg, "tool_call_id", None):
        d["tool_call_id"] = msg.tool_call_id
    tcs = getattr(msg, "tool_calls", None)
    if tcs:
        d["tool_calls"] = [
            {"id": tc.id, "function": tc.function, "arguments": tc.arguments}
            for tc in tcs
        ]
    return d

payload = {
    "trace_id": trace.info.trace_id,
    "metadata": agent_run.metadata,
    "messages": [message_to_dict(m) for m in agent_run.transcripts[0].messages],
}

out_path.write_text(json.dumps(payload, indent=2, default=str))
print(f"\nSaved full run to: {out_path}")

# 5) Store raw MLflow trace structure for reference
def span_to_dict(span):
    return {
        "name": getattr(span, "name", None),
        "attributes": getattr(span, "attributes", {}) or {},
        "inputs": getattr(span, "inputs", {}) or {},
        "outputs": getattr(span, "outputs", {}) or {},
        "child_spans": [span_to_dict(s) for s in getattr(span, "child_spans", []) or []],
    }

trace_dict = {
    "info": {
        "trace_id": getattr(trace.info, "trace_id", None),
        "status": getattr(trace.info, "status", None),
        "execution_time_ms": getattr(trace.info, "execution_time_ms", None),
        "tags": getattr(trace.info, "tags", {}) or {},
        "experiment_id": getattr(trace.info, "experiment_id", None),
        "timestamp_ms": getattr(trace.info, "timestamp_ms", None),
        "request_id": getattr(trace.info, "request_id", None),
    },
    "data": {
        "spans": [span_to_dict(s) for s in (getattr(getattr(trace, "data", object()), "spans", []) or [])]
    },
}

trace_out_path = out_dir / f"mlflow_trace_{trace.info.trace_id}.json"
trace_out_path.write_text(json.dumps(trace_dict, indent=2, default=str))
print(f"Saved raw MLflow trace to: {trace_out_path}")