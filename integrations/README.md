# Docent Integration

This module provides integration between MLflow DSPy traces and Docent for visualization and analysis.

## Requirements

- **Python 3.11+** (required by `docent-python`)
- `docent-python` package
- `mlflow` package

## Installation

```bash
# Note: Requires Python 3.11 or higher
pip install docent-python mlflow
```

## Python Version Issue

If you're using Python 3.10 or earlier, you'll need to:

1. **Upgrade Python** (recommended):
   ```bash
   # Create a new virtual environment with Python 3.11+
   python3.11 -m venv .venv311
   source .venv311/bin/activate  # or .venv311\Scripts\activate on Windows
   pip install docent-python mlflow
   ```

2. **Or use trajectory files**: The integration can also work with saved trajectory JSON files without requiring Docent, though you won't be able to ingest into Docent UI.

## Usage

See `examples/docent_integration.py` for complete examples of:
- Loading traces from MLflow
- Loading from trajectory JSON files
- Converting to Docent format
- Ingesting into Docent

## Quick Start

```python
from integrations.docent import load_mlflow_traces, load_dspy_trace_to_docent, ingest_to_docent
from docent.client import DocentClient

# Load traces from MLflow
traces = load_mlflow_traces(
    tracking_uri="http://127.0.0.1:5000",
    experiment_name="DSPy",
    return_type="list",
)

# Convert to Docent format
agent_runs = [load_dspy_trace_to_docent(trace) for trace in traces]

# Ingest into Docent
client = DocentClient()
collection = client.create_collection(name="My Collection")
ingest_to_docent(agent_runs, collection.collection_id, client)
```

