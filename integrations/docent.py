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

    # Track and dedupe system messages (only include the first one)
    system_message_added = False

    # Recursively extract messages from spans
    def extract_from_spans(spans: list, parent_type: str | None = None):
        """Recursively extract messages from trace spans."""
        for span in spans:
            span_name = span.name if hasattr(span, "name") else "unknown"
            span_attributes = span.attributes if hasattr(span, "attributes") else {}
            raw_inputs = span.inputs if hasattr(span, "inputs") else None
            raw_outputs = span.outputs if hasattr(span, "outputs") else None
            # MLflow sometimes stores IO in attributes under these keys
            span_inputs = _norm_io(raw_inputs, span_attributes, "mlflow.spanInputs")
            span_outputs = _norm_io(raw_outputs, span_attributes, "mlflow.spanOutputs")

            # Helper: emit chat-style messages from a messages list
            def emit_messages(obj: Any) -> None:
                try:
                    if isinstance(obj, list):
                        for m in obj:
                            if isinstance(m, dict) and m.get("role") and m.get("content") is not None:
                                nonlocal system_message_added
                                role_val = m.get("role")
                                if role_val == "system":
                                    if system_message_added:
                                        continue
                                    system_message_added = True
                                messages.append(parse_chat_message({
                                    "role": role_val,
                                    "content": m.get("content", ""),
                                }))
                except Exception:
                    pass

            # Helper: try common keys for user/assistant text
            def maybe_emit_simple(role: str, container: dict) -> None:
                _maybe_emit_simple(role, container)

            # Map span types to message roles
            # DSPy spans typically have names like "dspy.ReAct", "dspy.LM", "tool_name", etc.
            if "ReAct" in span_name or "agent" in span_name.lower():
                # Skip emitting initial input (task_description) and final output (solution/answer)
                # Only include embedded chat-style messages, if present
                if span_inputs and "messages" in span_inputs:
                    emit_messages(span_inputs.get("messages"))

            elif "lm" in span_name.lower() or "language_model" in span_name.lower():
                # LLM call - capture prompts/responses if surfaced as chat messages
                if span_inputs:
                    if isinstance(span_inputs, dict) and isinstance(span_inputs.get("messages"), list):
                        emit_messages(span_inputs.get("messages"))
                    elif isinstance(span_inputs, dict) and span_inputs.get("prompt") is not None:
                        messages.append(parse_chat_message({
                            "role": "user",
                            "content": str(span_inputs.get("prompt")),
                        }))
                    else:
                        maybe_emit_simple("user", span_inputs)
                if span_outputs:
                    llm_msg = None
                    if isinstance(span_outputs, dict):
                        llm_msg = span_outputs.get("message") or span_outputs.get("response") or span_outputs.get("output")
                    if llm_msg is not None:
                        _emit_list_as_messages(llm_msg)
                    else:
                        maybe_emit_simple("assistant", span_outputs)

            elif (
                "tool" in span_name.lower()
                or any(token in span_name for token in ["Tool", "execute"])
                or span_name.lower() in {"finish", "finalize", "submit", "finalize_submission", "write_submission"}
            ):
                # Tool call span
                tool_name = (span_attributes.get("tool.name") if isinstance(span_attributes, dict) else None) or span_name.split(".")[-1]
                tool_inputs = span_inputs if isinstance(span_inputs, dict) else {}

                tool_call_id = f"call_{len([m for m in messages if hasattr(m, 'tool_calls')])}"
                tool_call = ToolCall(
                    id=tool_call_id,
                    function=tool_name,
                    arguments=tool_inputs if isinstance(tool_inputs, dict) else {},
                    type="function",
                    parse_error=None,
                )

                messages.append(parse_chat_message({
                    "role": "assistant",
                    "content": f"Calling tool: {tool_name}",
                    "tool_calls": [tool_call],
                }))

                tool_output = None
                if isinstance(span_outputs, dict):
                    tool_output = span_outputs.get("output") or span_outputs.get("result") or span_outputs.get("response") or span_outputs.get("content")
                if tool_output is None:
                    tool_output = span_outputs
                if isinstance(tool_output, (dict, list)):
                    tool_output_str = json.dumps(tool_output, default=str)
                else:
                    tool_output_str = str(tool_output) if tool_output is not None else ""
                messages.append(parse_chat_message({
                    "role": "tool",
                    "name": tool_name,
                    "tool_call_id": tool_call_id,
                    "content": tool_output_str,
                }))

            # Recurse into child spans
            if hasattr(span, "child_spans") and span.child_spans:
                extract_from_spans(span.child_spans, span_name)
    
    # Extract messages from all spans (not only root) to maximize recall
    all_spans = trace_obj.data.spans if hasattr(trace_obj, "data") and hasattr(trace_obj.data, "spans") else []
    if all_spans:
        extract_from_spans(all_spans)
    
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
        "mlflow": {
            "tags": tags,
            "experiment_id": safe_get(trace_info, "experiment_id"),
            "timestamp_ms": safe_get(trace_info, "timestamp_ms"),
            "request_id": safe_get(trace_info, "request_id"),
        },
    }

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

