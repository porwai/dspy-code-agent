#!/usr/bin/env python3

"""Run DSPy agent for software engineering tasks."""

import os
import traceback
from pathlib import Path

import typer
import yaml

from minisweagent import global_config_dir
from minisweagent.agents.dspy import DSPyAgent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models import get_model
from minisweagent.run.utils.save import save_traj
from minisweagent.run.utils.save_dspy import save_traj_dspy, update_preds_json
from minisweagent.utils.log import logger

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

DEFAULT_CONFIG = builtin_config_dir / "dspy.yaml"
DEFAULT_OUTPUT = global_config_dir / "last_dspy_run.traj.json"

_HELP_TEXT = """Run DSPy-based agent for software engineering tasks.

[not dim]
This agent uses DSPy framework for reasoning and tool usage.
[/not dim]
"""


@app.command(help=_HELP_TEXT)
def main(
    task: str = typer.Option(..., "-t", "--task", help="Task/problem statement", show_default=False),
    model_name: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use", rich_help_panel="Advanced"),
    config_spec: Path = typer.Option(DEFAULT_CONFIG, "-c", "--config", help="Path to config file", rich_help_panel="Basic"),
    output: Path = typer.Option(DEFAULT_OUTPUT, "-o", "--output", help="Output trajectory file", rich_help_panel="Basic"),
    step_limit: int | None = typer.Option(None, "--step-limit", help="Maximum number of steps", rich_help_panel="Advanced"),
    cost_limit: float | None = typer.Option(None, "--cost-limit", help="Maximum cost limit", rich_help_panel="Advanced"),
) -> DSPyAgent:
    """Run DSPy agent on a software engineering task."""
    
    # Load configuration
    config_path = get_config_path(config_spec)
    logger.info(f"Loading DSPy agent config from '{config_path}'")
    config = yaml.safe_load(config_path.read_text())
    
    # Override config with command line arguments
    if model_name is not None:
        config.setdefault("model", {})["model_name"] = model_name
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class
    if step_limit is not None:
        config.setdefault("agent", {})["step_limit"] = step_limit
    if cost_limit is not None:
        config.setdefault("agent", {})["cost_limit"] = cost_limit
    
    # Initialize model and environment
    model = get_model(model_name, config.get("model", {}))
    env = LocalEnvironment(**config.get("environment", {}))
    
    # Create DSPy agent
    agent = DSPyAgent(
        model,
        env,
        **config.get("agent", {}),
    )
    
    exit_status, result, extra_info = None, None, None
    
    try:
        logger.info(f"Running DSPy agent on task: {task}")
        exit_status, result = agent.run(task)
        logger.info(f"Agent completed with status: {exit_status}")
        logger.info(f"Result: {result}")
        
    except Exception as e:
        logger.error(f"Error running DSPy agent: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}
    
    finally:
        # Save both the default mini trajectory and a DSPy/SWE-bench compatible minimal file
        if output:
            save_traj(agent, output, exit_status=exit_status, result=result, extra_info=extra_info)
            save_traj_dspy(agent, output.with_suffix(".dspy.traj.json"), exit_status=exit_status, result=result, extra_info=extra_info)
            logger.info(f"Trajectories saved to: {output} and {output.with_suffix('.dspy.traj.json')}")
    
    return agent


if __name__ == "__main__":
    app()
