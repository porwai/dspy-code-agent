"""DSPy tools that work with mini-SWE-agent environments (Docker/local)."""

import dspy
from typing import Any


def create_environment_tools(env: Any) -> list[dspy.Tool]:
    """Create DSPy tools that work with the given environment."""
    
    def execute_command(cmd: str) -> str:
        """Execute a shell command in the environment."""
        try:
            result = env.execute(cmd)
            return f"Return code: {result.get('returncode', 'unknown')}\nOutput:\n{result.get('output', '')}"
        except Exception as e:
            return f"Error executing command '{cmd}': {e}"

    def read_file(path: str) -> str:
        """Read a file from the environment."""
        try:
            result = env.execute(f"cat {path}")
            if result.get("returncode") == 0:
                return result.get("output", "")
            else:
                return f"Error reading file '{path}': {result.get('output', 'Unknown error')}"
        except Exception as e:
            return f"Error reading file '{path}': {e}"

    def write_file(path: str, content: str) -> str:
        """Write content to a file in the environment."""
        try:
            # Use heredoc to write content safely
            cmd = f"cat > {path} << 'EOF'\n{content}\nEOF"
            result = env.execute(cmd)
            if result.get("returncode") == 0:
                return f"Successfully wrote to {path}"
            else:
                return f"Error writing to '{path}': {result.get('output', 'Unknown error')}"
        except Exception as e:
            return f"Error writing to '{path}': {e}"

    def list_files(path: str = ".") -> str:
        """List files in a directory."""
        try:
            # Ignore noisy virtualenv and site-packages to keep results relevant
            result = env.execute(
                f"find {path} -type f \
                   -not -path '*/.*' \
                   -not -path '*/.venv/*' \
                   -not -path '*/venv/*' \
                   -not -path '*/site-packages/*' | head -50"
            )
            if result.get("returncode") == 0:
                return result.get("output", "No files found")
            else:
                return f"Error listing files in '{path}': {result.get('output', 'Unknown error')}"
        except Exception as e:
            return f"Error listing files in '{path}': {e}"

    def search_code(query: str, path: str = ".") -> str:
        """Search for text in files."""
        try:
            # Exclude vendor and venv directories to avoid noise and timeouts
            result = env.execute(
                "bash -lc "
                + repr(
                    "grep -r --binary-files=without-match "
                    "--exclude-dir='.git' --exclude-dir='.venv' --exclude-dir='venv' --exclude-dir='site-packages' "
                    "--include='*.py' --include='*.js' --include='*.ts' --include='*.java' --include='*.cpp' --include='*.c' --include='*.h' "
                    f"'{query}' {path} | head -20"
                )
            )
            if result.get("returncode") == 0:
                output = result.get("output", "")
                return output if output else f"No matches found for '{query}'"
            else:
                return f"Search completed (no matches or error): {result.get('output', '')}"
        except Exception as e:
            return f"Error searching for '{query}': {e}"

    def run_tests() -> str:
        """Run tests in the environment."""
        try:
            # Try common test commands
            test_commands = ["python -m pytest", "python -m unittest", "npm test", "make test"]
            for cmd in test_commands:
                result = env.execute(cmd)
                if result.get("returncode") == 0:
                    return f"Tests passed with '{cmd}':\n{result.get('output', '')}"
                elif "not found" not in result.get("output", ""):
                    return f"Test command '{cmd}' failed:\n{result.get('output', '')}"
            return "No test commands found or all failed"
        except Exception as e:
            return f"Error running tests: {e}"

    def submit_work() -> str:
        """Submit work by staging all changes and returning git diff. Use this when confident the task is complete."""
        try:
            # Stage all changes
            add_result = env.execute("git add -A")
            if add_result.get("returncode") != 0:
                return f"Error staging changes: {add_result.get('output', 'Unknown error')}"
            
            # Get the staged diff
            diff_result = env.execute("git diff --cached")
            if diff_result.get("returncode") != 0:
                return f"Error getting diff: {diff_result.get('output', 'Unknown error')}"
            
            diff_output = diff_result.get("output", "").strip()
            if not diff_output:
                return "No changes to submit. Make sure you have made modifications to the codebase."
            
            # Return the submission format similar to default agent
            return f"{diff_output}"
        except Exception as e:
            return f"Error submitting work: {e}"

    # Create DSPy tools
    return [
        dspy.Tool(
            func=execute_command,
            name="execute_command",
            desc="Execute a shell command in the environment",
            args={"cmd": {"type": "string"}},
            arg_types={"cmd": str},
            arg_desc={"cmd": "Shell command to execute"},
        ),
        dspy.Tool(
            func=read_file,
            name="read_file",
            desc="Read contents of a file",
            args={"path": {"type": "string"}},
            arg_types={"path": str},
            arg_desc={"path": "Path to file to read"},
        ),
        dspy.Tool(
            func=write_file,
            name="write_file",
            desc="Write content to a file",
            args={"path": {"type": "string"}, "content": {"type": "string"}},
            arg_types={"path": str, "content": str},
            arg_desc={"path": "Path to file to write", "content": "Content to write"},
        ),
        dspy.Tool(
            func=list_files,
            name="list_files",
            desc="List files in a directory",
            args={"path": {"type": "string", "default": "."}},
            arg_types={"path": str},
            arg_desc={"path": "Directory to list files from"},
        ),
        dspy.Tool(
            func=search_code,
            name="search_code",
            desc="Search for text in code files",
            args={"query": {"type": "string"}, "path": {"type": "string", "default": "."}},
            arg_types={"query": str, "path": str},
            arg_desc={"query": "Text to search for", "path": "Directory to search in"},
        ),
        dspy.Tool(
            func=run_tests,
            name="run_tests",
            desc="Run tests in the environment",
            args={},
            arg_types={},
            arg_desc={},
        ),
        dspy.Tool(
            func=submit_work,
            name="submit_work",
            desc="Submit work by staging all changes and returning git diff. Use this when confident the task is complete.",
            args={},
            arg_types={},
            arg_desc={},
        ),
    ]
