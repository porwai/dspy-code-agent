#!/usr/bin/env python3

"""Example of using DSPy agent with mini-SWE-agent framework."""

import sys
from pathlib import Path

# Add the src directory to the path so we can import minisweagent
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from minisweagent.agents.dspy import DSPyAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_model import LitellmModel


def main():
    """Example usage of DSPy agent."""
    
    # Example task
    task = """
    Create a simple Python script that:
    1. Takes a list of numbers as input
    2. Calculates the sum, average, and maximum
    3. Prints the results in a formatted way
    
    Save the script as 'calculator.py' and test it with some sample data.
    """
    
    print("Running DSPy Agent Example")
    print("=" * 50)
    print(f"Task: {task}")
    print("=" * 50)
    
    # Initialize model (you'll need to set your API key)
    model = LitellmModel(model_name="gpt-4o")
    
    # Initialize environment
    env = LocalEnvironment()
    
    # Create DSPy agent
    agent = DSPyAgent(
        model=model,
        env=env,
        step_limit=5,
        cost_limit=2.0
    )
    
    try:
        # Run the agent
        exit_status, result = agent.run(task)
        
        print(f"\nAgent completed with status: {exit_status}")
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"Error running agent: {e}")


if __name__ == "__main__":
    main()
