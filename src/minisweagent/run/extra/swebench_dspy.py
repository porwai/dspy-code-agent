#!/usr/bin/env python3

"""Run DSPy agent on SWE-bench instances."""

import os
import traceback
from datetime import datetime
from pathlib import Path

import typer
import yaml
from datasets import load_dataset

from minisweagent import global_config_dir
from minisweagent.agents.dspy import DSPyAgent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.extra.swebench import (
    DATASET_MAPPING,
    get_swebench_docker_image_name,
    update_preds_file,
)
from minisweagent.run.utils.save import save_traj
from minisweagent.run.utils.save_dspy import save_traj_dspy, update_preds_json
from minisweagent.utils.log import logger

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

DEFAULT_CONFIG = builtin_config_dir / "extra" / "swebench.yaml"
# Default to project-local outputs directory with timestamp
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
OUTPUT_BASE = Path(os.getenv("MSWEA_OUTPUT_DIR", PROJECT_ROOT / "outputs"))
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_OUTPUT = OUTPUT_BASE / "swebench_dspy" / f"run_{TIMESTAMP}" / "traj.json"

_HELP_TEXT = """Run DSPy agent on SWE-bench instances.

[not dim]
This uses DSPy ReAct agent with SWE-bench Docker environments.
[/not dim]
"""


@app.command(help=_HELP_TEXT)
def main(
    subset: str = typer.Option("lite", "--subset", help="SWEBench subset to use", rich_help_panel="Data selection"),
    split: str = typer.Option("dev", "--split", help="Dataset split", rich_help_panel="Data selection"),
    instance_spec: str = typer.Option(0, "-i", "--instance", help="SWE-Bench instance ID or index", rich_help_panel="Data selection"),
    model_name: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use", rich_help_panel="Advanced"),
    config_path: Path = typer.Option(DEFAULT_CONFIG, "-c", "--config", help="Path to config file", rich_help_panel="Basic"),
    environment_class: str | None = typer.Option("docker", "--environment-class", help="Environment type (docker/singularity)", rich_help_panel="Advanced"),
    output: Path = typer.Option(DEFAULT_OUTPUT, "-o", "--output", help="Output trajectory file", rich_help_panel="Basic"),
    exit_immediately: bool = typer.Option(False, "--exit-immediately", help="Exit immediately when done", rich_help_panel="Basic"),
) -> None:
    """Run DSPy agent on a single SWE-bench instance."""
    
    # Load dataset
    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset from {dataset_path}, split {split}...")
    instances = {
        inst["instance_id"]: inst  # type: ignore
        for inst in load_dataset(dataset_path, split=split)
    }
    
    if instance_spec.isnumeric():
        instance_spec = sorted(instances.keys())[int(instance_spec)]
    instance: dict = instances[instance_spec]  # type: ignore

    # Load config
    config_path = get_config_path(config_path)
    logger.info(f"Loading agent config from '{config_path}'")
    config = yaml.safe_load(config_path.read_text())
    
    if environment_class is not None:
        config.setdefault("environment", {})["environment_class"] = environment_class
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class
    if exit_immediately:
        config.setdefault("agent", {})["confirm_exit"] = False

    # Get SWE-bench environment
    env_config = config.setdefault("environment", {})
    env_config["environment_class"] = env_config.get("environment_class", "docker")
    image_name = get_swebench_docker_image_name(instance)
    if env_config["environment_class"] == "docker":
        env_config["image"] = image_name
    elif env_config["environment_class"] == "singularity":
        env_config["image"] = "docker://" + image_name
    
    env = get_environment(env_config)
    model = get_model(model_name, config.get("model", {}))
    
    # Create DSPy agent with SWE-bench environment
    agent = DSPyAgent(
        model,
        env,
        **config.get("agent", {}),
    )

    exit_status, result, extra_info = None, None, None
    try:
        logger.info(f"Running DSPy agent on SWE-bench instance: {instance_spec}")
        exit_status, result = agent.run(instance["problem_statement"])  # type: ignore[arg-type]
        logger.info(f"Agent completed with status: {exit_status}")
        
    except Exception as e:
        logger.error(f"Error processing instance {instance_spec}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}
    
    finally:
        # Save trajectories
        if output:
            save_traj(agent, output, exit_status=exit_status, result=result, extra_info=extra_info)  # type: ignore[arg-type]
            save_traj_dspy(agent, output.with_suffix(".dspy.traj.json"), exit_status=exit_status, result=result, extra_info=extra_info)
            logger.info(f"Trajectories saved to: {output} and {output.with_suffix('.dspy.traj.json')}")
        
        # If possible, capture a unified diff from the repo as the submission
        try:
            # Ensure we diff within the repo working dir
            diff_out = env.execute("git -C /testbed diff")
            diff_text = (diff_out.get("output") or "").strip()
            if diff_out.get("returncode") == 0 and diff_text:
                result = diff_text
        except Exception:
            pass

        # Update preds.json for SWE-bench evaluation
        output_dir = output.parent if output else Path(".")
        update_preds_json(output_dir, instance_spec, model.config.model_name, result)


if __name__ == "__main__":
    app()
