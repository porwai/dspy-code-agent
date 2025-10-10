#!/usr/bin/env python3
"""Simple batch runner for DSPy agent on SWE-bench instances."""

import subprocess
import sys
from pathlib import Path

def run_dspy_batch(start_idx: int = 0, end_idx: int = 14, model: str = "gpt-4o", output_dir: str = "outputs/dspy_batch"):
    """Run DSPy agent on multiple SWE-bench instances."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Running DSPy agent on instances {start_idx} to {end_idx}")
    print(f"Output directory: {output_path}")
    
    for i in range(start_idx, end_idx + 1):
        print(f"\n--- Running instance {i} ---")
        
        # Run single DSPy instance
        cmd = [
            sys.executable, "-m", "minisweagent.run.extra.swebench_dspy",
            "--subset", "lite",
            "--split", "dev", 
            "--instance", str(i),
            "--model", model,
            "--output", str(output_path / f"instance_{i}.traj.json")
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✓ Instance {i} completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Instance {i} failed: {e}")
            print(f"Error output: {e.stderr}")
    
    print(f"\nBatch completed! Results saved to {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run DSPy agent on multiple SWE-bench instances")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=14, help="End index") 
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model to use")
    parser.add_argument("--output", type=str, default="outputs/dspy_batch", help="Output directory")
    
    args = parser.parse_args()
    
    run_dspy_batch(args.start, args.end, args.model, args.output)
