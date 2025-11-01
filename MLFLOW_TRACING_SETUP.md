# MLflow Tracing Setup Guide

This guide explains how to enable MLflow tracing for DSPy agent runs.

## Quick Start

### 1. Enable Tracing via Environment Variable (Recommended)

Add to your `.env` file in the repo root:

```bash
MLFLOW_DSPY_ENABLE=1
MLFLOW_TRACKING_URI=http://127.0.0.1:5000  # Optional, defaults to this
MLFLOW_EXPERIMENT=DSPy  # Optional, defaults to "DSPy"
```

The `.env` file is automatically loaded from the repo root when running DSPy agents.

### 2. Enable Tracing via Config File

Edit `src/minisweagent/config/dspy.yaml` (or your custom config):

```yaml
agent:
  mlflow_enable: true
  mlflow_tracking_uri: "http://127.0.0.1:5000"  # Optional
  mlflow_experiment: "DSPy"  # Optional
```

### 3. Ensure MLflow Server is Running

Start MLflow tracking server (if not already running):

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlartifacts
```

Or use a different backend (PostgreSQL, etc.).

## Environment Variable Priority

The code checks in this order:
1. Config file: `agent.mlflow_enable`
2. Environment variable: `MLFLOW_DSPY_ENABLE` (set to "1", "true", or "yes")

For tracking URI and experiment:
1. Config file values
2. Environment variables (`MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT`)
3. Defaults (http://127.0.0.1:5000, "DSPy")

## Verification

After enabling, you should see:
- MLflow DSPy autolog messages in logs
- Debug logs showing which experiment is being used: `MLflow experiment set to: YourExperimentName`
- Traces appearing in MLflow UI at http://127.0.0.1:5000 under your configured experiment (not "DSPy")
- Tags and metadata in trace info

## Important: Experiment Name

The experiment is set **before each trace is created** to ensure traces go to the correct experiment. Make sure:

1. Set `MLFLOW_EXPERIMENT=YourExperimentName` in `.env`, OR
2. Set `mlflow_experiment: "YourExperimentName"` in your config YAML

The experiment name is resolved in this order:
1. Config file: `agent.mlflow_experiment`
2. Environment variable: `MLFLOW_EXPERIMENT`
3. Default: `"DSPy"`

**If all traces are going to "DSPy" instead of your experiment, check:**
- Your `.env` file has `MLFLOW_EXPERIMENT=YourExperimentName` (no quotes)
- Your config file has `mlflow_experiment: "YourExperimentName"` (with quotes in YAML)
- The experiment is set before the agent runs (which now happens automatically in `run()`)

## Troubleshooting

**Traces not appearing:**
- Check MLflow server is running: `curl http://127.0.0.1:5000/health`
- Verify environment variable: `echo $MLFLOW_DSPY_ENABLE`
- Check logs for "MLflow DSPy autolog enabled" message

**.env file not loading:**
- Ensure `.env` is in repo root (same level as `src/`)
- Check file has correct format: `KEY=value` (no quotes, no spaces around `=`)
- Restart your terminal/session after creating .env

## Example .env File

```bash
# API Keys
OPENROUTER_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here  # if needed

# MLflow Tracing
MLFLOW_DSPY_ENABLE=1
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
MLFLOW_EXPERIMENT=DSPy

# Other config...
```

