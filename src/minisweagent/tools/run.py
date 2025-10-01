import subprocess
import dspy

# -------- core --------
def execute_command(command: str, cwd: str = ".") -> str:
    try:
        p = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30
        )
        out = p.stdout.strip()
        err = p.stderr.strip()
        if p.returncode != 0:
            return f"[Error] Command failed ({p.returncode}): {err or out}"
        return out or "[Success] Command executed (no output)"
    except subprocess.TimeoutExpired:
        return f"[Error] Command timed out: {command}"
    except Exception as e:
        return f"[Error] Could not execute command: {e}"

def run_tests(test_command: str = "pytest -q", cwd: str = ".") -> str:
    return execute_command(test_command, cwd=cwd)

# -------- tools --------
execute_command_tool = dspy.Tool(
    func=execute_command,
    name="execute_command",
    desc="Execute a shell command and return stdout/stderr.",
    args={
        "command": {"type": "string"},
        "cwd": {"type": "string", "default": "."}
    },
    arg_types={"command": str, "cwd": str},
    arg_desc={
        "command": "Shell command to run.",
        "cwd": "Working directory (default: current)."
    },
)

run_tests_tool = dspy.Tool(
    func=run_tests,
    name="run_tests",
    desc="Run the project test suite (default pytest).",
    args={
        "test_command": {"type": "string", "default": "pytest -q"},
        "cwd": {"type": "string", "default": "."}
    },
    arg_types={"test_command": str, "cwd": str},
    arg_desc={
        "test_command": "Test runner command (e.g., 'pytest -q').",
        "cwd": "Working directory to run tests in."
    },
)
