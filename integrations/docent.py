"""Integration module to convert MLflow DSPy traces to Docent format.

This module provides functions to:
- Load traces from MLflow backend (SQLite)
- Convert trajectory JSON files to Docent format
- Parse MLflow trace spans and map to Docent ChatMessage format
- Ingest AgentRun objects into Docent for visualization
"""

import json
import os
from pathlib import Path
from typing import Any

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ImportError:
    mlflow = None
    MlflowClient = None

try:
    from docent.data_models import AgentRun, Transcript
    from docent.data_models.chat import ChatMessage, ToolCall, parse_chat_message
    DOCENT_AVAILABLE = True
    DOCENT_ERROR = None
except ImportError as e:
    AgentRun = None
    Transcript = None
    ChatMessage = None
    ToolCall = None
    parse_chat_message = None
    DOCENT_AVAILABLE = False
    DOCENT_ERROR = str(e)


def load_mlflow_traces(
    tracking_uri: str = "http://127.0.0.1:5000",
    experiment_name: str = "DSPy",
    experiment_ids: list[str] | None = None,
    filter_string: str | None = None,
    order_by: list[str] | None = None,
    max_results: int = 100,
    return_type: str = "list",
) -> list:
    """Load traces from MLflow backend using search_traces API.
    
    Args:
        tracking_uri: MLflow tracking URI (default: http://127.0.0.1:5000)
        experiment_name: Name of the MLflow experiment (used if experiment_ids not provided)
        experiment_ids: Optional list of experiment IDs to search
        filter_string: Optional MLflow filter string (e.g., "status = 'ERROR'")
        order_by: Optional list of columns to order by (e.g., ["timestamp_ms DESC"])
        max_results: Maximum number of traces to return
        return_type: Either "list" (Trace objects) or "dataframe" (pandas DataFrame)
        
    Returns:
        List of Trace objects or pandas DataFrame (depending on return_type)
    """
    if mlflow is None:
        raise ImportError("mlflow is required. Install with: pip install mlflow")
    
    mlflow.set_tracking_uri(tracking_uri)
    
    # Get experiment ID if not provided
    if experiment_ids is None:
        try:
            client = MlflowClient(tracking_uri=tracking_uri)
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                if return_type == "list":
                    return []
                else:
                    import pandas as pd
                    return pd.DataFrame()  # Return empty DataFrame
            experiment_ids = [experiment.experiment_id]
        except Exception as e:
            raise ValueError(f"Could not find experiment '{experiment_name}': {e}")
    
    # Query traces
    try:
        traces = mlflow.search_traces(
            experiment_ids=experiment_ids,
            filter_string=filter_string,
            order_by=order_by,
            max_results=max_results,
            return_type=return_type,
        )
        return traces
    except Exception as e:
        raise RuntimeError(f"Error querying traces: {e}")


def load_trace_from_file(traj_path: Path) -> dict:
    """Load trajectory data from a saved JSON file.
    
    Args:
        traj_path: Path to trajectory JSON file (typically .dspy.traj.json)
        
    Returns:
        Dictionary with trajectory data
    """
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")
    
    with open(traj_path, "r") as f:
        return json.load(f)


def _extract_tool_calls_from_text(input_str: str) -> list[dict] | None:
    """Extract tool calls embedded in assistant message text.

    Supports patterns like:
      - "Calling tools: [ { 'id': '...', 'function': { 'name': 'tool', 'arguments': {...} } } ]"

    Returns a list of dicts with keys similar to Docent ToolCall expectation.
    """
    import re
    import ast

    # Flexible match for a list after the phrase "Calling tools:"
    match = re.search(r"Calling\s+tools:\s*(\[.*\])", input_str, re.DOTALL | re.MULTILINE)
    if not match:
        return None
    try:
        parsed = ast.literal_eval(match.group(1))
        return parsed if isinstance(parsed, list) else None
    except (ValueError, SyntaxError):
        return None


def load_dspy_trace_to_docent(
    trace_data: dict | str | Path,
    metadata: dict | None = None,
) -> Any:
    """Convert DSPy trace/trajectory data to Docent AgentRun format.
    
    Args:
        trace_data: Either a dict (loaded trajectory JSON), Path to JSON file, 
                   or MLflow Trace object
        metadata: Optional additional metadata to merge
        
    Returns:
        AgentRun object compatible with Docent
        
    Raises:
        ImportError: If docent is not installed or Python version < 3.11
    """
    if not DOCENT_AVAILABLE:
        error_msg = "docent is required. Install with: pip install docent-python"
        if DOCENT_ERROR:
            error_msg += f"\n\nImport error details: {DOCENT_ERROR}"
            if "requires" in DOCENT_ERROR.lower() and "3.11" in DOCENT_ERROR:
                error_msg += "\n\nNote: docent-python requires Python 3.11 or higher."
                error_msg += "\nYour current Python version may be incompatible."
        raise ImportError(error_msg)
    
    # Load data if Path or string
    if isinstance(trace_data, (str, Path)):
        trace_data = load_trace_from_file(Path(trace_data))
    
    # Handle MLflow Trace object (from search_traces with return_type="list")
    if hasattr(trace_data, "info") and hasattr(trace_data, "data"):
        return _convert_mlflow_trace_to_docent(trace_data, metadata)
    
    # Handle trajectory JSON format (from save_traj_dspy)
    elif isinstance(trace_data, dict):
        if "messages" in trace_data and "info" in trace_data:
            return _convert_trajectory_to_docent(trace_data, metadata)
        elif "trace_info" in trace_data or "run_id" in trace_data:
            # Old format - try to convert
            raise ValueError("MLflow trace dict format not yet supported. Use Trace objects from search_traces()")
        else:
            raise ValueError(f"Unrecognized trace data format: {list(trace_data.keys())}")
    
    else:
        raise ValueError(f"Unsupported trace data type: {type(trace_data)}")


def _convert_trajectory_to_docent(data: dict, metadata: dict | None = None) -> Any:
    """Convert trajectory JSON format to Docent AgentRun.
    
    Trajectory format has:
    - messages: list of {role, content} dicts
    - info: dict with exit_status, submission, etc.
    - dspy_trajectory: optional DSPy-specific trajectory
    - dspy_result: optional DSPy result
    """
    import ast
    import uuid
    
    messages: list[Any] = []
    traj = data.get("messages", [])
    info = data.get("info", {})
    
    tool_call_counter = 0
    pending_tool_call_id = None
    
    for i, msg in enumerate(traj):
        role = msg.get("role")
        content = msg.get("content", "")
        
        # Create message data dict
        message_data = {"role": role, "content": content}
        
        # Handle assistant messages with tool calls (DSPy TOOL format)
        if role == "assistant" and content.startswith("TOOL "):
            # Parse tool call from content like "TOOL tool_name({'arg': 'value'})"
            try:
                tool_call_content = content[5:].strip()  # Remove "TOOL "
                # Extract tool name and arguments
                paren_idx = tool_call_content.find("(")
                if paren_idx > 0:
                    tool_name = tool_call_content[:paren_idx].strip()
                    tool_args_str = tool_call_content[paren_idx + 1:]
                    # Remove trailing closing paren
                    if tool_args_str.endswith(")"):
                        tool_args_str = tool_args_str[:-1]
                    
                    # Parse arguments
                    try:
                        tool_args = ast.literal_eval(tool_args_str) if tool_args_str else {}
                    except (ValueError, SyntaxError):
                        # Fallback: try to parse as dict string
                        tool_args = {}
                        # Could try regex parsing here if needed
                    
                    # Generate tool call ID
                    tool_call_id = f"call_{tool_call_counter}"
                    tool_call_counter += 1
                    pending_tool_call_id = tool_call_id
                    
                    tool_call = ToolCall(
                        id=tool_call_id,
                        function=tool_name,
                        arguments=tool_args if isinstance(tool_args, dict) else {},
                        type="function",
                        parse_error=None,
                    )
                    message_data["tool_calls"] = [tool_call]
                else:
                    # No arguments - just tool name
                    tool_name = tool_call_content.strip()
                    tool_call_id = f"call_{tool_call_counter}"
                    tool_call_counter += 1
                    pending_tool_call_id = tool_call_id
                    
                    tool_call = ToolCall(
                        id=tool_call_id,
                        function=tool_name,
                        arguments={},
                        type="function",
                        parse_error=None,
                    )
                    message_data["tool_calls"] = [tool_call]
            except Exception as e:
                # If parsing fails, just keep as regular message
                pass

        # Handle assistant messages that embed a tool-call list (Docent-style logs)
        elif role == "assistant":
            embedded_calls = _extract_tool_calls_from_text(content)
            if embedded_calls:
                parsed_tool_calls: list[ToolCall] = []
                for tc in embedded_calls:
                    fn = (tc.get("function") or {}).get("name")
                    args = (tc.get("function") or {}).get("arguments", {})
                    tool_call_id = tc.get("id") or f"call_{tool_call_counter}"
                    tool_call_counter += 1
                    pending_tool_call_id = tool_call_id
                    parsed_tool_calls.append(
                        ToolCall(
                            id=tool_call_id,
                            function=fn or "tool",
                            arguments=args if isinstance(args, dict) else {},
                            type="function",
                            parse_error=None,
                        )
                    )
                if parsed_tool_calls:
                    message_data["tool_calls"] = parsed_tool_calls
        
        # Handle user messages that follow tool calls - convert to tool messages
        elif role == "user" and pending_tool_call_id is not None:
            # This is a tool response
            message_data["role"] = "tool"
            message_data["name"] = messages[-1].tool_calls[0].function if messages and hasattr(messages[-1], "tool_calls") else "unknown"
            message_data["tool_call_id"] = pending_tool_call_id
            pending_tool_call_id = None  # Reset
        
        # Parse message
        try:
            chat_message = parse_chat_message(message_data)
            messages.append(chat_message)
        except Exception as e:
            # If parsing fails, create a basic message
            if role == "assistant":
                from docent.data_models.chat import AssistantMessage
                chat_message = AssistantMessage(role="assistant", content=content)
            elif role == "user":
                from docent.data_models.chat import UserMessage
                chat_message = UserMessage(role="user", content=content)
            else:
                from docent.data_models.chat import SystemMessage
                chat_message = SystemMessage(role=role, content=content)
            messages.append(chat_message)
    
    # Build metadata
    instance_id = info.get("exit_status", "unknown")
    task_id = metadata.get("task_id") if metadata else None
    if task_id is None:
        # Try to extract from file path or other sources
        task_id = instance_id
    
    scores = {}
    if info.get("submission"):
        # Could derive score from submission quality, etc.
        pass
    
    metadata_dict = {
        "benchmark_id": task_id,
        "task_id": task_id,
        "model": info.get("model_name", "unknown"),
        "scores": scores,
        "exit_status": info.get("exit_status"),
        "submission": info.get("submission"),
        "additional_metadata": info,
    }
    
    if metadata:
        metadata_dict.update(metadata)
    
    # Create transcript and AgentRun
    transcript = Transcript(messages=messages, metadata=metadata_dict)
    agent_run = AgentRun(transcripts=[transcript], metadata=metadata_dict)
    
    return agent_run


def _convert_mlflow_trace_to_docent(trace_obj: Any, metadata: dict | None = None) -> Any:
    """Convert MLflow Trace object to Docent AgentRun.
    
    MLflow traces have a tree structure of spans:
    - Root span represents the entire trace
    - Child spans represent individual operations (modules, tools, LM calls)
    - Each span has input/output that can be mapped to messages
    """
    messages: list[Any] = []
    
    # Extract trace info
    trace_info = trace_obj.info
    trace_id = trace_info.trace_id
    status = trace_info.status if hasattr(trace_info, "status") else "OK"
    execution_time = trace_info.execution_time_ms if hasattr(trace_info, "execution_time_ms") else None
    
    # Helpers to normalize span IO and emit messages robustly
    def _norm_io(field: Any, attrs: dict, attr_key: str) -> Any:
        if field not in (None, {}, []):
            return field
        if isinstance(attrs, dict) and attr_key in attrs:
            return attrs.get(attr_key)
        return {}

    def _emit_list_as_messages(candidate: Any) -> None:
        if candidate is None:
            return
        if isinstance(candidate, list):
            for item in candidate:
                if isinstance(item, dict) and item.get("role") and item.get("content") is not None:
                    messages.append(parse_chat_message({
                        "role": item.get("role"),
                        "content": item.get("content", ""),
                    }))
                elif isinstance(item, str):
                    messages.append(parse_chat_message({"role": "assistant", "content": item}))
        elif isinstance(candidate, dict) and candidate.get("role") and candidate.get("content") is not None:
            messages.append(parse_chat_message({
                "role": candidate.get("role"),
                "content": candidate.get("content", ""),
            }))
        elif isinstance(candidate, str):
            messages.append(parse_chat_message({"role": "assistant", "content": candidate}))

    def _maybe_emit_simple(role: str, container: Any) -> None:
        if isinstance(container, dict):
            for key in ("task_description", "task", "question", "query", "prompt", "input", "content"):
                if key in container and container.get(key) not in (None, ""):
                    val = container.get(key)
                    if isinstance(val, (dict, list)):
                        _emit_list_as_messages(val)
                    else:
                        messages.append(parse_chat_message({"role": role, "content": str(val)}))
                    return
        elif isinstance(container, list):
            _emit_list_as_messages(container)
        elif isinstance(container, str) and container.strip():
            messages.append(parse_chat_message({"role": role, "content": container}))

    # Find and extract system prompt from "ChatAdapter.format_1" span
    def find_and_extract_system_prompt(spans: list) -> str | None:
        """Recursively find the first span with name 'ChatAdapter.format_1' and extract system prompt."""
        for span in spans:
            span_name = span.name if hasattr(span, "name") else "unknown"
            if span_name == "ChatAdapter.format_1":
                span_attributes = span.attributes if hasattr(span, "attributes") else {}
                raw_inputs = span.inputs if hasattr(span, "inputs") else None
                span_inputs = _norm_io(raw_inputs, span_attributes, "mlflow.spanInputs")
                if span_inputs and isinstance(span_inputs, dict) and "signature" in span_inputs:
                    signature = span_inputs.get("signature")
                    if signature:
                        return str(signature)
            # Recurse into child spans
            if hasattr(span, "child_spans") and span.child_spans:
                result = find_and_extract_system_prompt(span.child_spans)
                if result:
                    return result
        return None

    # Find and process only "ReAct.forward" spans
    def find_react_forward_span(spans: list) -> Any | None:
        """Recursively find the first span with name 'ReAct.forward'."""
        for span in spans:
            span_name = span.name if hasattr(span, "name") else "unknown"
            if span_name == "ReAct.forward":
                return span
            # Recurse into child spans
            if hasattr(span, "child_spans") and span.child_spans:
                result = find_react_forward_span(span.child_spans)
                if result:
                    return result
        return None

    def process_react_forward_span(span: Any) -> None:
        """Process a ReAct.forward span: extract task_description and trajectory."""
        span_attributes = span.attributes if hasattr(span, "attributes") else {}
        raw_inputs = span.inputs if hasattr(span, "inputs") else None
        raw_outputs = span.outputs if hasattr(span, "outputs") else None
        
        # Normalize inputs/outputs (check attributes if empty)
        span_inputs = _norm_io(raw_inputs, span_attributes, "mlflow.spanInputs")
        span_outputs = _norm_io(raw_outputs, span_attributes, "mlflow.spanOutputs")
        
        # Extract task_description from span_inputs
        if span_inputs and isinstance(span_inputs, dict) and "task_description" in span_inputs:
            task_desc = span_inputs.get("task_description")
            if task_desc:
                messages.append(parse_chat_message({
                    "role": "user",
                    "content": str(task_desc),
                }))
        
        # Extract trajectory from span_outputs
        if span_outputs and isinstance(span_outputs, dict) and "trajectory" in span_outputs:
            trajectory = span_outputs.get("trajectory")
            if isinstance(trajectory, dict):
                # Parse trajectory: thought_N, tool_name_N, tool_args_N, observation_N
                # Find all step numbers from any of the keys (thought, tool_name, observation)
                step_numbers = set()
                for key in trajectory.keys():
                    if key.startswith("thought_") or key.startswith("tool_name_") or key.startswith("observation_") or key.startswith("tool_args_"):
                        try:
                            # Extract number from key like "thought_0" -> 0, "tool_name_1" -> 1
                            # Split by underscore and take the last part as the step number
                            parts = key.rsplit("_", 1)  # Split from right, max 1 split
                            if len(parts) == 2:
                                step_num = int(parts[1])
                                step_numbers.add(step_num)
                        except (ValueError, IndexError):
                            pass
                
                # Process each step in order
                for step_num in sorted(step_numbers):
                    # Extract thought
                    thought_key = f"thought_{step_num}"
                    thought = trajectory.get(thought_key, "")
                    
                    # Extract tool info
                    tool_name_key = f"tool_name_{step_num}"
                    tool_args_key = f"tool_args_{step_num}"
                    observation_key = f"observation_{step_num}"
                    
                    tool_name = trajectory.get(tool_name_key)
                    tool_args = trajectory.get(tool_args_key)
                    if tool_args is None:
                        tool_args = {}
                    observation = trajectory.get(observation_key, "")
                    
                    # Skip if this step has nothing
                    if not thought and not tool_name and not observation:
                        continue
                    
                    # Ensure tool_args is a dict
                    if not isinstance(tool_args, dict):
                        tool_args = {}
                    
                    # Create assistant message with thought
                    if thought:
                        # If there's a tool call, include it
                        if tool_name:
                            tool_call_id = f"call_{step_num}"
                            tool_call = ToolCall(
                                id=tool_call_id,
                                function=tool_name,
                                arguments=tool_args,
                                type="function",
                                parse_error=None,
                            )
                            messages.append(parse_chat_message({
                                "role": "assistant",
                                "content": thought,
                                "tool_calls": [tool_call],
                            }))
                            
                            # Add tool response message
                            if observation:
                                messages.append(parse_chat_message({
                                    "role": "tool",
                                    "name": tool_name,
                                    "tool_call_id": tool_call_id,
                                    "content": str(observation),
                                }))
                        else:
                            # Just a thought without tool call
                            messages.append(parse_chat_message({
                                "role": "assistant",
                                "content": thought,
                            }))
                    elif tool_name:
                        # Tool call without explicit thought
                        tool_call_id = f"call_{step_num}"
                        tool_call = ToolCall(
                            id=tool_call_id,
                            function=tool_name,
                            arguments=tool_args,
                            type="function",
                            parse_error=None,
                        )
                        messages.append(parse_chat_message({
                            "role": "assistant",
                            "content": f"Calling tool: {tool_name}",
                            "tool_calls": [tool_call],
                        }))
                        
                        if observation:
                            messages.append(parse_chat_message({
                                "role": "tool",
                                "name": tool_name,
                                "tool_call_id": tool_call_id,
                                "content": str(observation),
                            }))
                
                # Extract final reasoning and solution if present
                if "reasoning" in trajectory:
                    reasoning = trajectory.get("reasoning")
                    if reasoning:
                        messages.append(parse_chat_message({
                            "role": "assistant",
                            "content": f"Reasoning: {reasoning}",
                        }))
                
                if "solution" in trajectory:
                    solution = trajectory.get("solution")
                    if solution:
                        messages.append(parse_chat_message({
                            "role": "assistant",
                            "content": f"Solution: {solution}",
                        }))
    
    # Extract final prediction/result from trace spans
    all_spans = trace_obj.data.spans if hasattr(trace_obj, "data") and hasattr(trace_obj.data, "spans") else []
    final_prediction = None
    
    def extract_final_result(span: Any) -> str | None:
        """Extract final prediction/result from a span's outputs."""
        if not span:
            return None
        
        span_outputs = span.outputs if hasattr(span, "outputs") else None
        span_attributes = span.attributes if hasattr(span, "attributes") else {}
        
        # Normalize outputs (check attributes if outputs is empty)
        if span_outputs in (None, {}, []):
            span_outputs = span_attributes.get("mlflow.spanOutputs") if isinstance(span_attributes, dict) else None
        
        if span_outputs:
            # Look for solution/answer/completion in outputs
            if isinstance(span_outputs, dict):
                # First check trajectory for solution
                if "trajectory" in span_outputs and isinstance(span_outputs.get("trajectory"), dict):
                    trajectory = span_outputs.get("trajectory")
                    if "solution" in trajectory:
                        solution = trajectory.get("solution")
                        if solution:
                            return str(solution)
                
                # Fall back to other keys
                result = (
                    span_outputs.get("solution")
                    or span_outputs.get("answer")
                    or span_outputs.get("completion")
                    or span_outputs.get("result")
                    or span_outputs.get("output")
                )
                if result:
                    return str(result)
            elif isinstance(span_outputs, str) and span_outputs.strip():
                # If outputs is directly a string
                return span_outputs
        
        # Recurse into child spans (depth-first, prefer deeper results)
        if hasattr(span, "child_spans") and span.child_spans:
            for child in span.child_spans:
                result = extract_final_result(child)
                if result:
                    return result
        
        return None
    
    if all_spans:
        # Extract system prompt from "ChatAdapter.format_1" span
        system_prompt = find_and_extract_system_prompt(all_spans)
        if system_prompt:
            messages.insert(0, parse_chat_message({
                "role": "system",
                "content": system_prompt,
            }))
        
        # Find and process only "ReAct.forward" span
        react_forward_span = find_react_forward_span(all_spans)
        if react_forward_span:
            process_react_forward_span(react_forward_span)
            # Extract final prediction from the ReAct.forward span
            result = extract_final_result(react_forward_span)
            if result:
                final_prediction = result
    
    # If no messages extracted, create a basic message from trace
    if not messages:
        messages.append(parse_chat_message({
            "role": "assistant",
            "content": f"Trace {trace_id} completed with status: {status}",
        }))
    
    # Build metadata (preserve and merge MLflow metadata as much as possible)
    task_id = metadata.get("task_id") if metadata else trace_id
    tags = trace_info.tags if hasattr(trace_info, "tags") and isinstance(trace_info.tags, dict) else {}
    model_name = tags.get("model") or tags.get("model_name") or "unknown"

    # Try to capture additional info safely
    def safe_get(obj: Any, attr: str, default: Any = None) -> Any:
        return getattr(obj, attr, default)

    metadata_dict = {
        "benchmark_id": task_id,
        "task_id": task_id,
        "model": model_name,
        "scores": {},
        "trace_id": trace_id,
        "status": status,
        "execution_time_ms": execution_time,
        # Extract commonly used tags to top level for easier access
        "agent_type": tags.get("agent_type"),
        "model_class": tags.get("model_class"),
        "environment_class": tags.get("environment_class"),
        "step_limit": tags.get("step_limit"),
        "cost_limit": tags.get("cost_limit"),
        "platform": tags.get("platform"),
        "python_version": tags.get("python_version"),
        # SWE-bench specific tags (if available)
        "instance_id": tags.get("instance_id"),
        "dataset": tags.get("dataset"),
        "subset": tags.get("subset"),
        "split": tags.get("split"),
        "repo": tags.get("repo"),
        "base_commit": tags.get("base_commit"),
        # Final prediction/result extracted from trace
        "prediction": final_prediction,
        "mlflow": {
            "tags": tags,  # All tags preserved here
            "experiment_id": safe_get(trace_info, "experiment_id"),
            "timestamp_ms": safe_get(trace_info, "timestamp_ms"),
            "request_id": safe_get(trace_info, "request_id"),
        },
    }
    
    # Use instance_id as task_id if available (more specific than trace_id)
    if tags.get("instance_id"):
        metadata_dict["task_id"] = tags.get("instance_id")
        metadata_dict["benchmark_id"] = tags.get("instance_id")
    
    # Remove None values from top-level metadata
    metadata_dict = {k: v for k, v in metadata_dict.items() if v is not None}

    # Merge caller-provided metadata last to allow overrides
    if metadata:
        metadata_dict.update(metadata)
    
    # Create transcript and AgentRun
    transcript = Transcript(messages=messages, metadata=metadata_dict)
    agent_run = AgentRun(transcripts=[transcript], metadata=metadata_dict)
    
    return agent_run


"""
Note: Docent client ingestion utilities have been intentionally removed from this
integration module. This module focuses solely on converting DSPy/MLflow traces
into Docent-compatible data models (AgentRun, Transcript, ChatMessage).
"""

