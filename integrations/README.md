# Docent Integration for SWE-bench Traces

This guide explains how to run SWE-bench evaluations with the DSPy agent, evaluate predictions, and upload traces to Docent with full metadata (including model patches and pass/fail status).

## Overview

The complete workflow consists of three main steps:

1. **Run the DSPy agent** on SWE-bench instances (generates `preds.json` and MLflow traces)
2. **Evaluate the predictions** using SWE-bench harness (generates `evaluation_output.json`)
3. **Upload traces to Docent** with enhanced metadata (`upload_traces.py`)

## Prerequisites

1. **MLflow tracking server running**:
   ```bash
   mlflow ui --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlartifacts
   ```

2. **Environment variables set** (in `.env` file or exported):
   ```bash
   MLFLOW_DSPY_ENABLE=1
   MLFLOW_TRACKING_URI=http://127.0.0.1:5000
   MLFLOW_EXPERIMENT=your_experiment_name
   DOCENT_API_KEY=your_docent_api_key  # For upload_traces.py
   ```

3. **Required packages**:
   ```bash
   pip install docent-python mlflow swebench
   ```

## Step 1: Run the DSPy Agent on SWE-bench

### Batch Mode (Recommended)

Run the agent on multiple SWE-bench instances with MLflow tracing enabled:

```bash
python -m minisweagent.run.extra.swebench_dspy_batch \
  --subset lite \
  --split dev \
  --slice 0:5 \
  --output outputs/my_run \
  --config src/minisweagent/config/extra/swebench.yaml \
  --workers 1
```

**Important Configuration:**

Make sure your config file (`src/minisweagent/config/extra/swebench.yaml` or custom) has:

```yaml
agent:
  mlflow_enable: true
  mlflow_experiment: "your_experiment_name"  # Must match MLFLOW_EXPERIMENT
  mlflow_tracking_uri: "http://127.0.0.1:5000"
```

**Output:**
- `outputs/my_run/preds.json` - Model predictions/patches
- `outputs/my_run/{instance_id}/{instance_id}.traj.json` - Agent trajectories
- `outputs/my_run/{instance_id}/{instance_id}.dspy.traj.json` - DSPy-specific trajectories
- MLflow traces in your MLflow tracking server (with tags including `instance_id`)

### Single Instance Mode

For testing a single instance:

```bash
python -m minisweagent.run.extra.swebench_dspy \
  --subset lite \
  --split dev \
  --instance django__django-16139 \
  --output outputs/test_instance \
  --config src/minisweagent/config/extra/swebench.yaml
```

## Step 2: Evaluate Predictions

Use the SWE-bench evaluation harness to determine which instances passed/failed:

```bash
python scripts/evaluate_swebench.py \
  --preds outputs/my_run/preds.json \
  --dataset princeton-nlp/SWE-Bench_Lite \
  --run-id my_evaluation_run
```

**Output:**
- `evaluation_output/{model_name}.{run_id}.json` - Contains:
  - `resolved_ids`: List of instance IDs that passed
  - `unresolved_ids`: List of instance IDs that failed
  - `submitted_ids`: List of instance IDs that were submitted
  - Statistics and metrics

**Note:** The evaluation script requires `swebench` package. If you get import errors:
```bash
pip install swebench
```

## Step 3: Upload Traces to Docent

Upload MLflow traces to Docent with enhanced metadata (model patches and pass/fail status):

```bash
python integrations/upload_traces.py "your_experiment_name" \
  --preds outputs/my_run/preds.json \
  --evaluation evaluation_output/openrouter__qwen__qwen3-coder-30b-a3b-instruct.my_evaluation_run.json \
  --collection "My SWE-bench Collection" \
  --max-results 100
```

**Arguments:**
- `experiment_name` (required): MLflow experiment name (must match what you used in Step 1)
- `--preds` / `-p`: Path to `preds.json` from Step 1
- `--evaluation` / `-e`: Path to evaluation output JSON from Step 2
- `--collection` / `-c`: Docent collection name (default: `"SWE-bench {experiment_name}"`)
- `--tracking-uri` / `-u`: MLflow tracking URI (default: `http://127.0.0.1:5000`)
- `--max-results` / `-m`: Maximum traces to upload (default: 100)
- `--filter` / `-f`: MLflow filter string (e.g., `"status = 'OK'"`)

**What It Does:**

1. Loads traces from MLflow experiment
2. Matches each trace with `preds.json` using `instance_id` tag to get `model_patch`
3. Matches each trace with evaluation results to determine pass/fail status
4. Enhances AgentRun metadata with:
   - `model_patch`: The actual code patch generated
   - `swebench_status`: `"passed"`, `"failed"`, `"not_resolved"`, or `"not_evaluated"`
   - `swebench_resolved`: Boolean indicating if the instance passed
5. Uploads to Docent collection for visualization

**Output:**
- Collection created/updated in Docent with all traces
- Each trace now includes model patch and evaluation status in metadata

## Diagnostic Tools

### Inspect Traces

Check what tags are actually present in MLflow traces:

```bash
python examples/inspect_traces.py "your_experiment_name" \
  -o trace_inspection.json \
  --max-results 10
```

This will show:
- All tags on each trace
- Whether `instance_id` tags are present
- Task descriptions and predictions extracted from spans

### Test Tag Setting

Verify that MLflow tag setting works in your environment:

```bash
python examples/test_tag_setting.py
```

This tests both methods of setting tags and shows which ones work.

## Troubleshooting

### Tags Not Appearing in Traces

1. **Verify MLflow is enabled**: Check that `MLFLOW_DSPY_ENABLE=1` is set
2. **Check experiment name matches**: The experiment name in config must match what you query
3. **Run diagnostic tools**: Use `inspect_traces.py` to see what tags are actually present
4. **Check logs**: Look for messages like `"Got trace ID from get_last_active_trace_id()"` or `"Set X/Y tags on trace ..."`

### Matching Fails in upload_traces.py

1. **Verify instance_id tags exist**: Use `inspect_traces.py` to confirm traces have `instance_id` tags
2. **Check preds.json format**: Should be `{instance_id: {model_patch: "...", ...}}`
3. **Check evaluation_output format**: Should have `resolved_ids`, `unresolved_ids`, `submitted_ids` lists
4. **Verify instance_ids match**: The IDs in traces, preds.json, and evaluation_output must match exactly

### Trace Not Found

1. **Wait for trace commit**: Traces may take a moment to be persisted; the code includes a 0.5s delay
2. **Check experiment name**: Ensure you're querying the correct experiment
3. **Verify MLflow server**: Make sure MLflow UI is accessible and traces are visible there

## Example Complete Workflow

```bash
# 1. Run agent on 5 instances
python -m minisweagent.run.extra.swebench_dspy_batch \
  --subset lite \
  --split dev \
  --slice 0:5 \
  --output outputs/test_run \
  --config src/minisweagent/config/extra/swebench.yaml

# 2. Evaluate predictions
python scripts/evaluate_swebench.py \
  --preds outputs/test_run/preds.json \
  --dataset princeton-nlp/SWE-Bench_Lite \
  --run-id test_eval

# 3. Inspect traces (optional, for verification)
python examples/inspect_traces.py "my_experiment" -o trace_inspection.json

# 4. Upload to Docent
python integrations/upload_traces.py "my_experiment" \
  --preds outputs/test_run/preds.json \
  --evaluation evaluation_output/openrouter__qwen__qwen3-coder-30b-a3b-instruct.test_eval.json \
  --collection "Test SWE-bench Run"
```

## Additional Resources

- MLflow Tracing Setup: See `MLFLOW_TRACING_SETUP.md` in project root
- Docent Integration API: See `integrations/docent.py` for low-level functions
- Examples: See `examples/` directory for more usage examples
