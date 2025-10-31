from pathlib import Path
import sys
import os
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

# 1) Convert MLflow traces → AgentRun(s)
traces = load_mlflow_traces(
    tracking_uri="http://127.0.0.1:5000",
    experiment_name="DSPy",
    filter_string="status = 'OK'",
    order_by=["timestamp_ms DESC"],
    max_results=5,
    return_type="list",
)
agent_runs = [
    load_dspy_trace_to_docent(t, metadata={"task_id": t.info.trace_id})
    for t in traces
]

# 2) Create client + collection, then ingest
api_key = os.getenv("DOCENT_API_KEY")
client = Docent(api_key=api_key)
collection_id = client.create_collection(
    name="DSPy MLflow Traces",
    description="Agent runs from MLflow DSPy traces",
)
client.add_agent_runs(collection_id, agent_runs)
