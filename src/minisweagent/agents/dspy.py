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


@dataclass
class DSPyAgentConfig:
    """Configuration for DSPy agent."""
    system_template: str = "You are a DSPy coding agent for SWE-bench tasks."
    step_limit: int = 6
    cost_limit: float = 3.0

# lm = dspy.LM('openai/gpt-4o-mini', api_key=os.environ["OPENAI_API_KEY"])# Configure DSPy to use OpenAI
lm = dspy.LM('openrouter/qwen/qwen3-coder-30b-a3b-instruct', api_key=os.environ["OPENROUTER_API_KEY"], max_tokens=32000, temperature=0.7) # Configure DSPy to use OpenRouter Qwen
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
            DSPySoftwareEngineeringAgent,
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
        # Prefer environment-aware tools whenever the environment can execute commands
        if hasattr(self.env, 'execute'):
            base_tools = create_environment_tools(self.env)
        else:
            # Fallback to local filesystem tools for non-environment runs
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
            result = self.dspy_agent(task_description=task)
            # Capture DSPy trajectory if present
            trajectory = getattr(result, "trajectory", None)
            if trajectory is not None:
                self.dspy_trajectory = self._serialize_trajectory(trajectory)
            # Capture full DSPy result (JSON-safe) using simple serializer
            self.dspy_result = self._serialize_response(result)
            
            # First try to extract submit_work output from trajectory
            submit_work_output = self._extract_submit_work_output(self.dspy_trajectory, self.dspy_result)
            if submit_work_output:
                # Use the raw git diff as the final output
                final_output = submit_work_output
            else:
                # Fallback: extract the solution from DSPySoftwareEngineeringAgent signature
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

    def _extract_submit_work_output(self, trajectory: list, dspy_result: dict) -> str | None:
        """Extract submit_work tool output from DSPy trajectory."""
        # First try to extract from the dspy_result._store.trajectory (flat structure)
        if isinstance(dspy_result, dict) and "_store" in dspy_result:
            store = dspy_result["_store"]
            if isinstance(store, dict) and "trajectory" in store:
                traj_data = store["trajectory"]
                if isinstance(traj_data, dict):
                    # Look for submit_work tool calls in the flat trajectory structure
                    for key, value in traj_data.items():
                        if key.startswith("tool_name_") and value == "submit_work":
                            # Find the corresponding observation
                            step_num = key.split("_")[-1]
                            observation_key = f"observation_{step_num}"
                            if observation_key in traj_data:
                                observation = traj_data[observation_key]
                                # Return the raw observation (git diff) from submit_work tool
                                return observation
        
        # Fallback: try the serialized trajectory list
        if trajectory:
            # The trajectory is a list of dictionaries, each containing trajectory data
            # Look for submit_work tool calls in the trajectory
            for step in trajectory:
                if isinstance(step, dict):
                    # Check if this step contains tool calls
                    for key, value in step.items():
                        if key.startswith("tool_name_") and value == "submit_work":
                            # Find the corresponding observation
                            step_num = key.split("_")[-1]
                            observation_key = f"observation_{step_num}"
                            if observation_key in step:
                                observation = step[observation_key]
                                # Return the raw observation (git diff) from submit_work tool
                                return observation
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
