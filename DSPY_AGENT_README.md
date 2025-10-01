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

### 1. Using the Command Line Runner

```bash
# Run with default configuration
python -m minisweagent.run.dspy_agent -t "Your task here"

# Run with custom model
python -m minisweagent.run.dspy_agent -t "Your task" -m "gpt-4o"

# Run with custom configuration
python -m minisweagent.run.dspy_agent -t "Your task" -c path/to/config.yaml
```

### 2. Using in Python Code

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

### 3. Configuration

The DSPy agent can be configured through YAML files. See `src/minisweagent/config/dspy.yaml` for the default configuration.

Key configuration options:
- `step_limit`: Maximum number of reasoning steps
- `cost_limit`: Maximum cost for the agent
- `model_name`: Language model to use
- `temperature`: Model temperature setting

## Implementation Notes

### Current Status

The current implementation is a **basic framework integration**. You need to:

1. **Implement Tools**: The `_get_tools()` method currently returns an empty list. You need to implement or import your actual DSPy tools.

2. **Tool Integration**: Connect your DSPy tools with the mini-SWE-agent environment system.

3. **Error Handling**: Enhance error handling for DSPy-specific exceptions.

### Framework Integration

The DSPy agent follows the mini-SWE-agent framework conventions:

- **`run(task)`**: Main entry point that returns `(exit_status, result)`
- **`step()`**: Single step execution (required by framework)
- **`add_message()`**: Message handling (required by framework)
- **Exception handling**: Uses framework exception types

### Next Steps

To make this agent fully functional:

1. **Implement Tools**: Add your actual DSPy tools to the `_get_tools()` method
2. **Environment Integration**: Connect tools with the `self.env` environment
3. **Testing**: Test with various software engineering tasks
4. **SWE-bench Integration**: Adapt for SWE-bench evaluation if needed

## Example Tools Structure

Here's how you might structure your tools:

```python
def _get_tools(self):
    """Get tools for DSPy agent."""
    return [
        # File operations
        self._create_file_tool,
        self._read_file_tool,
        self._edit_file_tool,
        
        # Code analysis
        self._search_code_tool,
        self._analyze_structure_tool,
        
        # Execution
        self._run_command_tool,
        self._test_code_tool,
    ]

def _create_file_tool(self, filename: str, content: str) -> str:
    """Create a file with given content."""
    # Implementation using self.env
    pass
```

## Dependencies

Make sure you have the required dependencies:

```bash
pip install dspy-ai
pip install minisweagent
```

## Troubleshooting

1. **Import Errors**: Ensure all imports are correct and dependencies are installed
2. **Tool Errors**: Check that your tools are properly implemented
3. **Model Errors**: Verify your model configuration and API keys
4. **Environment Errors**: Ensure the environment is properly initialized

## Contributing

To extend this agent:

1. Add new tools to the `_get_tools()` method
2. Implement tool functions that use `self.env` for environment interaction
3. Update configuration as needed
4. Add tests for new functionality
