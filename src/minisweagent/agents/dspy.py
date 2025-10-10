"""DSPy-based agent for mini-SWE-agent framework."""

import dspy
from dataclasses import dataclass
from typing import Callable
from dataclasses import fields
from typing import Any
import os
import json
from dotenv import load_dotenv

from minisweagent import Environment, Model
from minisweagent.tools import search as search_tools
from minisweagent.tools import read_write as rw_tools
from minisweagent.tools import tree as tree_tools
from minisweagent.tools import edit as edit_tools
from minisweagent.tools.environment_tools import create_environment_tools
from minisweagent.agents.default import (
    NonTerminatingException,
    TerminatingException,
    Submitted,
    LimitsExceeded,
)
from minisweagent.dspy_modules import DSPySoftwareEngineeringAgent
from minisweagent.utils.log import logger


@dataclass
class DSPyAgentConfig:
    """Configuration for DSPy agent."""
    system_template: str = "You are a DSPy coding agent for SWE-bench tasks."
    step_limit: int = 6
    cost_limit: float = 3.0

# DSPy LM configuration will be set dynamically based on the model passed to the agent

class DSPyAgent:
    """DSPy-based agent that integrates with mini-SWE-agent framework."""
    
    def __init__(self, model: Model, env: Environment, *, config_class: Callable = DSPyAgentConfig, **kwargs):
        cfg_field_names = {f.name for f in fields(config_class)}
        cfg_kwargs = {k: v for k, v in kwargs.items() if k in cfg_field_names}
        self.config = config_class(**cfg_kwargs)
        self.model = model
        self.env = env
        self.messages: list[dict] = []
        self.extra_template_vars = {}
        self.dspy_trajectory: list = []
        self.dspy_result: dict = {}

        # Configure DSPy LM based on the model
        self._configure_dspy_lm()

        # Initialize DSPy ReAct agent with basic tools
        # Note: You'll need to implement or import your actual tools
        self.dspy_agent = dspy.ReAct(
            DSPySoftwareEngineeringAgent,
            tools=self._get_tools(),
            max_iters=self.config.step_limit,
        )

    def _configure_dspy_lm(self):
        """Configure DSPy LM based on the model provider and settings."""
        try:
            model_name = getattr(self.model.config, 'model_name', 'gpt-4o-mini')
            model_kwargs = getattr(self.model.config, 'model_kwargs', {})
            
            # Extract API key from model kwargs or environment
            api_key = model_kwargs.get('api_key')
            if not api_key:
                # Try to get from environment variables
                api_key = (os.getenv('OPENAI_API_KEY') or 
                          os.getenv('ANTHROPIC_API_KEY') or 
                          os.getenv('OPENROUTER_API_KEY'))
            
            # Determine provider and configure accordingly
            model_type = str(type(self.model)).lower()
            
            # Handle different model providers
            if 'anthropic' in model_name.lower() or 'claude' in model_name.lower() or 'anthropic' in model_type:
                # Anthropic models
                dspy_lm_name = f"anthropic/{model_name}"
                api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
            elif 'openrouter' in model_type:
                # OpenRouter models
                dspy_lm_name = f"openai/{model_name}"  # OpenRouter uses OpenAI-compatible API
                api_key = api_key or os.getenv('OPENROUTER_API_KEY')
                api_base = "https://openrouter.ai/api/v1"
                lm = dspy.LM(dspy_lm_name, api_key=api_key, api_base=api_base)
                dspy.configure(lm=lm)
                logger.info(f"Configured DSPy with OpenRouter model: {model_name}")
                return
            elif 'litellm' in model_type:
                # LiteLLM models - use the model name as-is
                dspy_lm_name = model_name
            else:
                # Default to OpenAI format
                dspy_lm_name = f"openai/{model_name}"
                api_key = api_key or os.getenv('OPENAI_API_KEY')
            
            # Configure DSPy with the determined LM
            if api_key:
                lm = dspy.LM(dspy_lm_name, api_key=api_key)
                dspy.configure(lm=lm)
                logger.info(f"Configured DSPy with model: {dspy_lm_name}")
            else:
                # Fallback configuration
                logger.warning(f"No API key found for model {model_name}, using default DSPy configuration")
                # DSPy will use its default configuration
                
        except Exception as e:
            logger.error(f"Error configuring DSPy LM: {e}")
            logger.warning("Falling back to default DSPy configuration")
            # DSPy will use its default configuration

    def _wrap_tool(self, tool: Any) -> Any:
        """Wrap a dspy.Tool to record calls and results into self.messages."""
        original_func = getattr(tool, "func", None) or getattr(tool, "_func", None)

        def wrapped_func(**kwargs):
            try:
                # log invocation
                self.messages.append({
                    "role": "assistant",
                    "content": f"TOOL {getattr(tool, 'name', '<tool>')}({kwargs})",
                })
                result = original_func(**kwargs)
                # log result
                self.messages.append({
                    "role": "user",
                    "content": str(result),
                })
                return result
            except Exception as e:
                err = f"[Error] {type(e).__name__}: {e}"
                self.messages.append({
                    "role": "user",
                    "content": err,
                })
                return err

        # Recreate a Tool with same schema but wrapped function
        return dspy.Tool(
            func=wrapped_func,
            name=getattr(tool, "name", None) or "tool",
            desc=getattr(tool, "desc", None) or "",
            args=getattr(tool, "args", None) or {},
            arg_types=getattr(tool, "arg_types", None) or {},
            arg_desc=getattr(tool, "arg_desc", None) or {},
        )

    def _get_tools(self):
        """Get tools for DSPy agent with trajectory logging wrappers."""
        # Prefer environment-aware tools whenever the environment can execute commands
        if hasattr(self.env, 'execute'):
            base_tools = create_environment_tools(self.env)
        else:
            # Fallback to local filesystem tools for non-environment runs
            from minisweagent.tools import run as run_tools
            base_tools = [
                # execution tools
                run_tools.execute_command_tool,
                run_tools.run_tests_tool,
                # search tools
                search_tools.search_code_tool,
                search_tools.regex_search_tool,
                search_tools.relevant_files_tool,
                # file read/write tools
                rw_tools.cat_tool,
                rw_tools.rm_tool,
                rw_tools.mv_tool,
                # tree tools
                tree_tools.tree_tool,
                tree_tools.find_file_tool,
                # edit tools
                edit_tools.create_file_tool,
                edit_tools.update_file_tool,
                edit_tools.update_file_regex_tool,
            ]
        return [self._wrap_tool(t) for t in base_tools]

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run the agent on a task. Returns exit status and result."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.dspy_trajectory = []
        self.dspy_result = {}
        
        try:
            # Use DSPy agent to solve the task
            result = self.dspy_agent(task_description=task)
            # Capture DSPy trajectory if present
            trajectory = getattr(result, "trajectory", None)
            if trajectory is not None:
                self.dspy_trajectory = self._serialize_trajectory(trajectory)
            # Capture full DSPy result (JSON-safe) using simple serializer
            self.dspy_result = self._serialize_response(result)
            
            # First, check if there's a submit_work tool call in the trajectory
            submit_work_output = self._extract_submit_work_output()
            
            if submit_work_output:
                # Use the submit_work tool output as the final result
                final_output = submit_work_output
            else:
                # Fallback to extracting solution from DSPySoftwareEngineeringAgent signature
                solution_text = (
                    getattr(result, "solution", None)
                    or getattr(result, "answer", None)
                    or getattr(result, "completion", None)
                    or (result.get("solution") if isinstance(result, dict) else None)
                    or (result.get("answer") if isinstance(result, dict) else None)
                    or (result.get("completion") if isinstance(result, dict) else None)
                    or str(result)
                )
                
                # Check if the solution contains a git diff (from submit_work tool)
                if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in solution_text:
                    # The solution already contains the proper submission format
                    final_output = solution_text
                else:
                    # Fallback: wrap the solution in the expected format
                    final_output = f"COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n{solution_text}"
            
            raise Submitted(final_output)
            
        except TerminatingException as e:
            return type(e).__name__, str(e)
        except NonTerminatingException as e:
            return type(e).__name__, str(e)
        except Exception as e:
            return "Error", str(e)

    def step(self) -> dict:
        """Single step execution (required by framework)."""
        # DSPy handles its own step logic internally
        # This is a placeholder for framework compatibility
        return {"status": "completed"}

    def add_message(self, role: str, content: str, **kwargs):
        """Add message to conversation (required by framework)."""
        self.messages.append({"role": role, "content": content, **kwargs})

    def _serialize_trajectory(self, trajectory) -> list:
        """Best-effort serialization of DSPy trajectory for saving to JSON."""
        serialized: list = []
        try:
            for step in list(trajectory):  # ensure it's iterable and materialize
                # Try common representations
                if hasattr(step, "to_dict"):
                    serialized.append(step.to_dict())
                elif hasattr(step, "dict"):
                    try:
                        serialized.append(step.dict())
                    except Exception:
                        serialized.append(str(step))
                elif hasattr(step, "__dict__") and step.__dict__:
                    serialized.append({k: v for k, v in step.__dict__.items() if not k.startswith("_")})
                else:
                    serialized.append(str(step))
        except Exception:
            # Fallback to stringifying whole trajectory
            try:
                serialized = [str(trajectory)]
            except Exception:
                serialized = []
        return serialized

    def _serialize_value(self, value):
        from collections.abc import Mapping, Sequence
        basic = (str, int, float, bool, type(None))
        if isinstance(value, basic):
            return value
        if isinstance(value, Mapping):
            return {str(k): self._serialize_value(v) for k, v in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [self._serialize_value(v) for v in list(value)]
        # try common attr dumps
        for attr in ("to_dict", "dict", "model_dump"):
            f = getattr(value, attr, None)
            if callable(f):
                try:
                    return self._serialize_value(f())
                except Exception:
                    pass
        try:
            return vars(value)
        except Exception:
            return str(value)

    def _extract_submit_work_output(self) -> str | None:
        """Extract submit_work tool output from trajectory if available."""
        # Check if dspy_result has the trajectory in _store
        if not self.dspy_result or "_store" not in self.dspy_result:
            return None
            
        store = self.dspy_result["_store"]
        if "trajectory" not in store:
            return None
            
        trajectory = store["trajectory"]
        if not isinstance(trajectory, dict):
            return None
            
        # Search through trajectory for submit_work tool calls
        for key, value in trajectory.items():
            if key.startswith("tool_name_") and value == "submit_work":
                # Find the corresponding observation
                step_num = key.split("_")[-1]
                observation_key = f"observation_{step_num}"
                if observation_key in trajectory:
                    return trajectory[observation_key]
        
        return None

    def _serialize_response(self, response: Any):
        """Convert DSPy response to JSON-safe structure (simple and robust)."""
        try:
            # If it has a dict of attributes, serialize values safely
            if hasattr(response, "__dict__"):
                out = {}
                for k, v in response.__dict__.items():
                    if isinstance(v, (dict, list, str, int, float, bool, type(None))):
                        out[k] = v
                    else:
                        try:
                            out[k] = json.loads(str(v))
                        except Exception:
                            out[k] = str(v)
                return out
            # Try parseable string first
            try:
                return json.loads(str(response))
            except Exception:
                return str(response)
        except Exception:
            return str(response)
