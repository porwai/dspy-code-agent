"""DSPy-based agent for mini-SWE-agent framework."""

import dspy
from dataclasses import dataclass
from typing import Callable
from dataclasses import fields
from typing import Any
import os
import json
from pathlib import Path
from dotenv import load_dotenv

import mlflow

# Load .env from repo root if available (in addition to global config)
_repo_root = Path(__file__).resolve().parents[3]  # src/minisweagent/agents/dspy.py -> repo root
_env_path = _repo_root / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=str(_env_path), override=False)  # Don't override existing env vars 

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
    # MLflow tracing (optional)
    mlflow_enable: bool = False
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "DSPy"

# lm = dspy.LM('openai/gpt-4o-mini', api_key=os.environ["OPENAI_API_KEY"])# Configure DSPy to use OpenAI
lm = dspy.LM('openrouter/qwen/qwen3-coder-30b-a3b-instruct', api_key=os.environ["OPENROUTER_API_KEY"], max_tokens=32000, temperature=0.7) # Configure DSPy to use OpenRouter Qwen
dspy.configure(lm=lm)

class DSPyAgent:
    """DSPy-based agent that integrates with mini-SWE-agent framework."""
    
    def __init__(self, model: Model, env: Environment, *, config_class: Callable = DSPyAgentConfig, **kwargs):
        cfg_field_names = {f.name for f in fields(config_class)}
        cfg_kwargs = {k: v for k, v in kwargs.items() if k in cfg_field_names}
        self.config = config_class(**cfg_kwargs)
        
        # Override mlflow_experiment from env var if config didn't explicitly set it
        # This ensures env var takes precedence over default "DSPy" value
        if "mlflow_experiment" not in cfg_kwargs and os.getenv("MLFLOW_EXPERIMENT"):
            self.config.mlflow_experiment = os.getenv("MLFLOW_EXPERIMENT")
        self.model = model
        self.env = env
        self.messages: list[dict] = []
        self.extra_template_vars = {}
        self.dspy_trajectory: list = []
        self.dspy_result: dict = {}

        # Optionally enable MLflow DSPy autologging for tracing/observability
        self._maybe_enable_mlflow_tracing()

        # Initialize DSPy ReAct agent with basic tools
        # Note: You'll need to implement or import your actual tools
        self.dspy_agent = dspy.ReAct(
            DSPySoftwareEngineeringAgent,
            tools=self._get_tools(),
            max_iters=self.config.step_limit,
        )

    def _maybe_enable_mlflow_tracing(self) -> None:
        """Configure MLflow tracking URI/experiment if enabled. 
        
        Note: mlflow.dspy.autolog() should be called once from the main thread,
        not from worker threads. This method only sets the tracking URI/experiment.
        
        Honors, in order of precedence:
        - self.config.mlflow_enable
        - ENV: MLFLOW_DSPY_ENABLE in {"1","true","yes"}
        """
        import threading
        enabled = self.config.mlflow_enable or str(os.getenv("MLFLOW_DSPY_ENABLE", "")).lower() in {"1", "true", "yes"}
        if not enabled:
            return
        tracking_uri = self.config.mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or "http://127.0.0.1:5000"
        experiment = (self.config.mlflow_experiment or os.getenv("MLFLOW_EXPERIMENT") or "DSPy")
        mlflow.set_tracking_uri(tracking_uri)
        
        # Ensure experiment exists and is set before autolog (critical!)
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient(tracking_uri=tracking_uri)
            exp = client.get_experiment_by_name(experiment)
            if exp is None:
                client.create_experiment(experiment)
        except Exception:
            pass  # Fallback to set_experiment if client fails
        
        mlflow.set_experiment(experiment)
        
        # Only call autolog from main thread (should already be done by batch script)
        # Worker threads should not call mlflow.dspy.autolog() as it modifies dspy.settings
        if threading.current_thread() is threading.main_thread():
            if hasattr(mlflow, "dspy") and hasattr(mlflow.dspy, "autolog"):
                try:
                    mlflow.dspy.autolog()
                except RuntimeError:
                    # Already configured, ignore
                    pass

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

    def _collect_trace_tags(self, task: str, **kwargs) -> dict[str, str]:
        """Collect metadata tags for MLflow trace."""
        tags: dict[str, str] = {}
        
        # Agent type
        tags["agent_type"] = "dspy"
        
        # Task description
        tags["task"] = task[:500] if len(task) > 500 else task  # Limit length
        
        # Config settings
        tags["step_limit"] = str(self.config.step_limit)
        tags["cost_limit"] = str(self.config.cost_limit)
        if self.config.mlflow_experiment:
            tags["experiment"] = self.config.mlflow_experiment
        
        # Model information
        model_name = "unknown"
        try:
            # Try to get model name from DSPy settings
            if hasattr(dspy, "settings") and hasattr(dspy.settings, "lm"):
                dspy_lm = dspy.settings.lm
                if hasattr(dspy_lm, "model_name"):
                    model_name = str(dspy_lm.model_name)
                elif hasattr(dspy_lm, "model"):
                    model_name = str(dspy_lm.model)
                elif hasattr(dspy_lm, "name"):
                    model_name = str(dspy_lm.name)
            # Also try module-level lm variable
            if model_name == "unknown" and hasattr(lm, "model_name"):
                model_name = str(lm.model_name)
            elif model_name == "unknown" and hasattr(lm, "model"):
                model_name = str(lm.model)
            # Fallback to self.model if available
            if model_name == "unknown" and hasattr(self.model, "config"):
                if hasattr(self.model.config, "model_name"):
                    model_name = str(self.model.config.model_name)
        except Exception:
            pass
        tags["model"] = model_name
        tags["model_name"] = model_name
        
        # Model class name
        model_class_name = type(self.model).__name__
        tags["model_class"] = model_class_name
        
        # Environment information
        env_class_name = type(self.env).__name__
        tags["environment_class"] = env_class_name
        if hasattr(self.env, "config"):
            if hasattr(self.env.config, "__dict__"):
                env_config_dict = {k: str(v) for k, v in self.env.config.__dict__.items() if not k.startswith("_")}
                for k, v in list(env_config_dict.items())[:5]:  # Limit to first 5 keys
                    tags[f"env_{k}"] = v[:200]  # Limit value length
        
        # SWE-bench specific metadata (recognize standard fields)
        swebench_fields = {
            "instance_id": "instance_id",
            "dataset": "dataset",
            "subset": "subset", 
            "split": "split",
            "repo": "repo",
            "base_commit": "base_commit",
            "patch": "patch",
            "test_patch": "test_patch",
        }
        for field_name, tag_key in swebench_fields.items():
            if field_name in kwargs and kwargs[field_name] is not None:
                val = str(kwargs[field_name])
                tags[tag_key] = val[:200] if len(val) > 200 else val
        
        # Additional template vars from kwargs (exclude already-processed SWE-bench fields)
        swebench_keys = set(swebench_fields.keys())
        other_kwargs = {k: v for k, v in kwargs.items() if k not in swebench_keys}
        if other_kwargs:
            for k, v in list(other_kwargs.items())[:10]:  # Limit to first 10 kwargs
                if isinstance(v, (str, int, float, bool)):
                    tags[f"extra_{k}"] = str(v)[:200]
                elif v is None:
                    tags[f"extra_{k}"] = "None"
        
        # System/environment context
        import platform
        tags["platform"] = platform.system()
        tags["python_version"] = platform.python_version()
        
        return tags

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run the agent on a task. Returns exit status and result."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.dspy_trajectory = []
        self.dspy_result = {}
        
        # Ensure MLflow experiment is set before creating traces (important for correct experiment)
        try:
            if self.config.mlflow_enable or str(os.getenv("MLFLOW_DSPY_ENABLE", "")).lower() in {"1", "true", "yes"}:
                # Set experiment explicitly before trace creation
                experiment = (self.config.mlflow_experiment or os.getenv("MLFLOW_EXPERIMENT") or "DSPy")
                tracking_uri = self.config.mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or "http://127.0.0.1:5000"
                mlflow.set_tracking_uri(tracking_uri)
                
                # Get experiment ID (create if doesn't exist)
                from mlflow.tracking import MlflowClient
                client = MlflowClient(tracking_uri=tracking_uri)
                try:
                    exp = client.get_experiment_by_name(experiment)
                    if exp is None:
                        exp_id = client.create_experiment(experiment)
                        from minisweagent.utils.log import logger
                        logger.info(f"Created MLflow experiment: {experiment} (id: {exp_id})")
                    else:
                        exp_id = exp.experiment_id
                except Exception as e:
                    from minisweagent.utils.log import logger
                    logger.warning(f"Failed to get/create experiment {experiment}: {e}, using set_experiment()")
                    mlflow.set_experiment(experiment)
                    exp_id = None
                
                # Explicitly set experiment (must be done before trace creation)
                mlflow.set_experiment(experiment)
                
                # Verify experiment is actually set (critical for debugging)
                from minisweagent.utils.log import logger
                try:
                    from mlflow.tracking import MlflowClient
                    verify_client = MlflowClient(tracking_uri=tracking_uri)
                    current_exp = verify_client.get_experiment_by_name(experiment)
                    if current_exp:
                        logger.info(f"MLflow experiment verified: {experiment} (id: {current_exp.experiment_id}, tracking_uri: {tracking_uri})")
                    else:
                        logger.warning(f"MLflow experiment {experiment} not found after setting!")
                except Exception as verify_e:
                    logger.info(f"MLflow experiment set to: {experiment} (tracking_uri: {tracking_uri}) (verification failed: {verify_e})")
                
                # Prepare tags (will be applied after trace is created)
                tags = self._collect_trace_tags(task, **kwargs)
        except Exception as e:
            # If MLflow is not available or not in a trace context, silently continue
            from minisweagent.utils.log import logger
            logger.debug(f"MLflow experiment setup failed (non-critical): {e}")
            tags = None
        
        try:
            # Use DSPy agent to solve the task
            result = self.dspy_agent(task_description=task)
            # Capture DSPy trajectory if present
            trajectory = getattr(result, "trajectory", None)
            if trajectory is not None:
                self.dspy_trajectory = self._serialize_trajectory(trajectory)
            # Capture full DSPy result (JSON-safe) using simple serializer
            self.dspy_result = self._serialize_response(result)
            
            # Update MLflow trace tags AFTER the trace is created (by DSPy autolog)
            if tags and (self.config.mlflow_enable or str(os.getenv("MLFLOW_DSPY_ENABLE", "")).lower() in {"1", "true", "yes"}):
                from minisweagent.utils.log import logger
                tracking_uri = self.config.mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or "http://127.0.0.1:5000"
                experiment = (self.config.mlflow_experiment or os.getenv("MLFLOW_EXPERIMENT") or "DSPy")
                
                trace_id = None
                try:
                    # Method 1: Try to get last active trace ID (if available)
                    if hasattr(mlflow, "get_last_active_trace_id"):
                        trace_id = mlflow.get_last_active_trace_id()
                        if trace_id:
                            logger.info(f"Got trace ID from get_last_active_trace_id(): {trace_id}")
                except Exception as e:
                    logger.debug(f"get_last_active_trace_id() failed: {e}")
                
                # Method 2: Try to update current trace (if still in trace context and we don't have trace_id)
                tags_set_via_update = False
                if not trace_id:
                    try:
                        mlflow.update_current_trace(tags=tags)
                        logger.info(f"Successfully updated current trace with {len(tags)} tags via update_current_trace()")
                        tags_set_via_update = True
                    except Exception as e1:
                        logger.debug(f"update_current_trace() failed: {e1}, will try to find trace by search...")
                
                # Method 3: If update_current_trace failed, or if we already have trace_id, set tags via client API
                if not tags_set_via_update:
                    try:
                        from mlflow.tracking import MlflowClient
                        client = MlflowClient(tracking_uri=tracking_uri)
                        
                        # If we don't have trace_id yet, search for it
                        if not trace_id:
                            exp = client.get_experiment_by_name(experiment)
                            if exp:
                                # Add small delay to ensure trace is committed
                                import time
                                time.sleep(0.5)  # Brief delay to ensure trace is persisted
                                
                                traces = mlflow.search_traces(
                                    experiment_ids=[exp.experiment_id],
                                    order_by=["timestamp_ms DESC"],
                                    max_results=1,
                                    return_type="list",
                                )
                                if traces:
                                    trace_id = traces[0].info.trace_id
                                    logger.info(f"Found trace via search: {trace_id}")
                        
                        # Set tags using client API if we have trace_id
                        if trace_id:
                            tags_set = 0
                            for key, value in tags.items():
                                if value:  # Only set non-empty values
                                    try:
                                        # MLflow client API for setting trace tags
                                        client.set_trace_tag(trace_id, key, str(value))
                                        tags_set += 1
                                    except Exception as tag_error:
                                        logger.debug(f"Failed to set tag '{key}': {tag_error}")
                            
                            if tags_set > 0:
                                logger.info(f"Set {tags_set}/{len(tags)} tags on trace {trace_id} via client API")
                            else:
                                logger.warning(f"No tags were set on trace {trace_id}")
                        else:
                            logger.warning(f"No traces found in experiment {experiment} to update tags")
                    except Exception as e2:
                        logger.warning(f"Failed to update trace tags via client: {e2}")
                
            
            # Extract the solution from DSPySoftwareEngineeringAgent signature
            solution_text = (
                getattr(result, "solution", None)
                or getattr(result, "answer", None)
                or getattr(result, "completion", None)
                or (result.get("solution") if isinstance(result, dict) else None)
                or (result.get("answer") if isinstance(result, dict) else None)
                or (result.get("completion") if isinstance(result, dict) else None)
                or str(result)
            )
            
            # Ensure solution_text always ends with a newline
            if solution_text and not solution_text.endswith("\n"):
                solution_text = solution_text + "\n"
                        
            raise Submitted(solution_text)
            
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
