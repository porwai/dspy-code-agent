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


@dataclass
class DSPyAgentConfig:
    """Configuration for DSPy agent."""
    system_template: str = "You are a DSPy coding agent for SWE-bench tasks."
    step_limit: int = 6
    cost_limit: float = 3.0

'''
@dataclass
class AgentConfig:
    # The default settings are the bare minimum to run the agent. Take a look at the config files for improved settings.
    system_template: str = "You are a helpful assistant that can do anything."
    instance_template: str = (
        "Your task: {{task}}. Please reply with a single shell command in triple backticks. "
        "To finish, the first line of the output of the shell command must be 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'."
    )
    timeout_template: str = (
        "The last command <command>{{action['action']}}</command> timed out and has been killed.\n"
        "The output of the command was:\n <output>\n{{output}}\n</output>\n"
        "Please try another command and make sure to avoid those requiring interactive input."
    )
    format_error_template: str = "Please always provide EXACTLY ONE action in triple backticks."
    action_observation_template: str = "Observation: {{output}}"
    step_limit: int = 0
    cost_limit: float = 3.0
'''

lm = dspy.LM('openai/gpt-4o-mini', api_key=os.environ["OPENAI_API_KEY"])# Configure DSPy to use OpenAI
dspy.configure(lm=lm)

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

        # Initialize DSPy ReAct agent with basic tools
        # Note: You'll need to implement or import your actual tools
        self.dspy_agent = dspy.ReAct(
            signature="instruction -> answer",
            tools=self._get_tools(),
            max_iters=self.config.step_limit,
        )

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
        # Check if we're in a SWE-bench environment (Docker/containerized)
        if hasattr(self.env, 'execute') and hasattr(self.env, 'image'):
            # Use environment-aware tools for SWE-bench
            base_tools = create_environment_tools(self.env)
        else:
            # Use local filesystem tools for local development
            base_tools = [
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
            result = self.dspy_agent(instruction=task)
            # Capture DSPy trajectory if present
            trajectory = getattr(result, "trajectory", None)
            if trajectory is not None:
                self.dspy_trajectory = self._serialize_trajectory(trajectory)
            # Capture full DSPy result (JSON-safe) using simple serializer
            self.dspy_result = self._serialize_response(result)
            
            # Format result for framework compatibility
            answer_text = (
                getattr(result, "answer", None)
                or getattr(result, "completion", None)
                or (result.get("answer") if isinstance(result, dict) else None)
                or (result.get("completion") if isinstance(result, dict) else None)
                or str(result)
            )
            final_output = f"COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n{answer_text}"
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
