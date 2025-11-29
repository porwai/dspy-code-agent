import json
from pathlib import Path
from typing import Any

from minisweagent.utils.log import logger


def save_traj_dspy(
    agent: Any | None,
    path: Path,
    *,
    exit_status: str | None = None,
    result: str | None = None,
    extra_info: dict | None = None,
) -> None:
    """Save a minimal trajectory compatible with SWE-bench style outputs.

    Writes a small JSON with agent metadata, messages if available, and result.
    """
    data = {
        "info": {
            "exit_status": exit_status,
            "submission": result,
        },
        "messages": getattr(agent, "messages", []),
        "dspy_trajectory": getattr(agent, "dspy_trajectory", []),
        "dspy_result": getattr(agent, "dspy_result", {}),
    }
    if extra_info:
        data["info"].update(extra_info)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    logger.info(f"Saved DSPy trajectory to '{path}'")


def update_preds_json(output_dir: Path, instance_id: str | None, model_name: str, result: str) -> None:
    """Update preds.json similar to SWE-bench batch runner.

    If instance_id is None, skip writing preds.json (non-batch runs).
    """
    if instance_id is None:
        return
    preds_path = output_dir / "preds.json"
    try:
        # Ensure result always ends with a newline
        if result is not None:
            result = str(result)
            if not result.endswith("\n"):
                result = result + "\n"
        else:
            result = "\n"
        
        preds = {}
        if preds_path.exists():
            preds = json.loads(preds_path.read_text())
        preds[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": result,
        }
        preds_path.write_text(json.dumps(preds, indent=2))
        logger.info(f"Updated preds.json at '{preds_path}' for instance {instance_id}")
    except Exception as e:
        logger.error(f"Failed to update preds.json: {e}")


