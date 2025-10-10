# DSPy Agent for mini-SWE-agent

This directory contains a DSPy-based agent that integrates with the mini-SWE-agent framework.

## Overview

The DSPy agent uses the DSPy framework for reasoning and tool usage, providing a different approach to software engineering tasks compared to the default agent.

## Files

- `src/minisweagent/agents/dspy.py` - Main DSPy agent implementation
- `src/minisweagent/config/dspy.yaml` - Configuration file for the DSPy agent
- `src/minisweagent/run/dspy_agent.py` - Command-line runner for the DSPy agent
- `examples/dspy_example.py` - Example usage of the DSPy agent

## Usage

### 1. Basic Task Execution

```bash
# Run with default configuration
python -m minisweagent.run.dspy_agent -t "Your task here"

# Run with custom model
python -m minisweagent.run.dspy_agent -t "Your task" -m "gpt-4o"

# Run with custom configuration
python -m minisweagent.run.dspy_agent -t "Your task" -c path/to/config.yaml
```

### 2. SWE-bench Evaluation

The DSPy agent supports evaluation on SWE-bench datasets using the specialized runner:

#### Single Instance Evaluation

```bash
# Run on a single SWE-bench instance
python -m minisweagent.run.extra.swebench_dspy \
    --subset lite \
    --split dev \
    --instance 0 \
    --model gpt-4o \
    --output outputs/dspy_test.traj.json

# Run on a specific instance by ID
python -m minisweagent.run.extra.swebench_dspy \
    --subset lite \
    --split dev \
    --instance sqlfluff__sqlfluff-1625 \
    --model gpt-4o \
    --output outputs/dspy_sqlfluff.traj.json
```

#### Batch Evaluation

```bash
# Run on multiple instances (parallel processing)
python -m minisweagent.run.extra.swebench \
    --subset lite \
    --split dev \
    --slice 0:5 \
    --model gpt-4o \
    --workers 4 \
    --output outputs/swebench_dspy/
```

#### Available Datasets

- `lite`: SWE-Bench Lite (smaller, faster evaluation)
- `verified`: SWE-Bench Verified (verified instances)
- `full`: Full SWE-Bench dataset
- `multimodal`: SWE-Bench Multimodal
- `multilingual`: SWE-Bench Multilingual
- `smith`: SWE-smith dataset

### 3. Evaluation with SWE-bench Harness

After running the DSPy agent on SWE-bench instances, you can evaluate the results using the official SWE-bench evaluation harness:

#### Using the Evaluation Script

```bash
# Evaluate predictions using the provided script
python scripts/evaluate_swebench.py \
    --preds outputs/swebench_dspy/preds.json \
    --dataset princeton-nlp/SWE-Bench_Lite \
    --run-id dspy_test_run
```

#### Direct SWE-bench Harness Evaluation

```bash
# Install SWE-bench if not already installed
pip install swebench

# Run evaluation directly
python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-Bench_Lite \
    --predictions_path outputs/swebench_dspy/preds.jsonl \
    --max_workers 4 \
    --run_id dspy_evaluation
```

#### Cloud-based Evaluation (Recommended)

For faster evaluation, you can use the cloud-based evaluation service:

```bash
# Install sb-cli
pip install sb-cli

# Submit for cloud evaluation (free!)
sb-cli submit swe-bench_lite test \
    --predictions_path outputs/swebench_dspy/preds.json \
    --run_id dspy_cloud_eval
```

### 4. Using in Python Code

```python
from minisweagent.agents.dspy import DSPyAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_model import LitellmModel

# Initialize components
model = LitellmModel(model_name="gpt-4o")
env = LocalEnvironment()
agent = DSPyAgent(model=model, env=env)

# Run agent
exit_status, result = agent.run("Your task here")
```

### 5. Configuration

The DSPy agent can be configured through YAML files. See `src/minisweagent/config/dspy.yaml` for the default configuration.

Key configuration options:
- `step_limit`: Maximum number of reasoning steps
- `cost_limit`: Maximum cost for the agent
- `model_name`: Language model to use
- `temperature`: Model temperature setting

## Output Files and Results

### Trajectory Files

The DSPy agent generates several output files:

- **`.traj.json`**: Standard mini-SWE-agent trajectory format
- **`.dspy.traj.json`**: DSPy-specific trajectory with detailed reasoning steps
- **`preds.json`**: SWE-bench compatible predictions file

### Understanding Results

#### Trajectory Analysis

The `.dspy.traj.json` file contains:
- **`dspy_result._store.trajectory`**: Complete reasoning trajectory
- **Tool calls**: Each step shows `thought_X`, `tool_name_X`, `tool_args_X`, `observation_X`
- **Submit work output**: When the agent uses `submit_work` tool, the observation contains the final patch

#### SWE-bench Evaluation Results

After running evaluation, you'll get:
- **Pass rate**: Percentage of instances solved correctly
- **Detailed results**: Per-instance success/failure information
- **Patches**: Generated code changes for each instance

### Example Workflow

```bash
# 1. Run DSPy agent on SWE-bench instances
python -m minisweagent.run.extra.swebench_dspy \
    --subset lite \
    --split dev \
    --slice 0:3 \
    --model gpt-4o \
    --output outputs/dspy_test/

# 2. Check generated predictions
cat outputs/dspy_test/preds.json

# 3. Evaluate with SWE-bench harness
python scripts/evaluate_swebench.py \
    --preds outputs/dspy_test/preds.json \
    --dataset princeton-nlp/SWE-Bench_Lite \
    --run-id dspy_test

# 4. View detailed trajectory
cat outputs/dspy_test/run_*/test.dspy.traj.json
```

## Implementation Notes

### Current Status

The DSPy agent is **fully functional** and includes:

1. **Complete Tool Integration**: Uses mini-SWE-agent's comprehensive tool suite
2. **SWE-bench Compatibility**: Generates proper prediction files for evaluation
3. **Trajectory Logging**: Captures detailed reasoning steps and tool usage
4. **Submit Work Detection**: Automatically extracts final patches from `submit_work` tool calls

### Framework Integration

The DSPy agent follows the mini-SWE-agent framework conventions:

- **`run(task)`**: Main entry point that returns `(exit_status, result)`
- **`step()`**: Single step execution (required by framework)
- **`add_message()`**: Message handling (required by framework)
- **Exception handling**: Uses framework exception types
- **Tool wrapping**: Automatically logs tool calls and results

### Key Features

1. **DSPy ReAct Integration**: Uses DSPy's ReAct framework for reasoning
2. **Tool Call Logging**: All tool calls are logged with inputs and outputs
3. **Submit Work Detection**: Automatically prioritizes `submit_work` tool output
4. **Trajectory Serialization**: Converts DSPy trajectories to JSON-safe format
5. **SWE-bench Evaluation**: Generates compatible prediction files

## Dependencies

Make sure you have the required dependencies:

```bash
# Core dependencies
pip install dspy-ai
pip install minisweagent

# For SWE-bench evaluation
pip install swebench
pip install sb-cli  # For cloud evaluation (optional)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install dspy-ai minisweagent swebench
   ```

2. **Model Configuration**: Verify your API keys and model settings
   ```bash
   # Check environment variables
   echo $OPENAI_API_KEY
   echo $ANTHROPIC_API_KEY
   ```

3. **Docker Environment**: Ensure Docker is running for SWE-bench evaluation
   ```bash
   docker --version
   docker ps
   ```

4. **Trajectory Files**: Check that output directories exist and are writable
   ```bash
   mkdir -p outputs/swebench_dspy
   ```

### Debugging

1. **Check Trajectory Files**: Examine the `.dspy.traj.json` files for detailed reasoning steps
2. **Verify Tool Calls**: Look for `submit_work` tool calls in the trajectory
3. **Model Output**: Check if the model is generating proper responses
4. **Environment Issues**: Ensure the Docker environment is properly set up

## Performance Tips

1. **Use Cloud Evaluation**: For faster results, use `sb-cli` for cloud-based evaluation
2. **Parallel Processing**: Use `--workers` flag for batch evaluation
3. **Model Selection**: Choose appropriate models based on your needs and budget
4. **Cost Limits**: Set appropriate cost limits to avoid unexpected charges

## Contributing

The DSPy agent is fully functional and ready for use. To extend or modify:

1. **Tool Integration**: The agent already uses mini-SWE-agent's comprehensive tool suite
2. **Custom Models**: Add new model classes in `src/minisweagent/models/`
3. **Configuration**: Modify `src/minisweagent/config/dspy.yaml` for default settings
4. **Evaluation**: Use the provided evaluation scripts and SWE-bench harness
