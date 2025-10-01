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
            result = env.execute(f"find {path} -type f -not -path '*/.*' | head -50")
            if result.get("returncode") == 0:
                return result.get("output", "No files found")
            else:
                return f"Error listing files in '{path}': {result.get('output', 'Unknown error')}"
        except Exception as e:
            return f"Error listing files in '{path}': {e}"

    def search_code(query: str, path: str = ".") -> str:
        """Search for text in files."""
        try:
            result = env.execute(f"grep -r '{query}' {path} --include='*.py' --include='*.js' --include='*.ts' --include='*.java' --include='*.cpp' --include='*.c' --include='*.h' | head -20")
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
    ]
