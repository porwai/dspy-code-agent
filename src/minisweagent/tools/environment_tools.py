"""DSPy tools that work with mini-SWE-agent environments (Docker/local)."""

import dspy
from typing import Any


def create_environment_tools(env: Any) -> list[dspy.Tool]:
    """Create DSPy tools that work with the given environment."""
    
    def execute_command(cmd: str) -> str:
        """Execute a shell command in the environment."""
        try:
            result = env.execute(cmd)
            returncode = result.get('returncode', 0)
            output = result.get('output', '')
            return f"<returncode>{returncode}</returncode>\n<output>{output}</output>"
        except Exception as e:
            return f"<returncode>1</returncode>\n<output>Error executing command '{cmd}': {e}</output>"

    def read_file(path: str) -> str:
        """Read a file from the environment."""
        try:
            result = env.execute(f"cat {path}")
            returncode = result.get("returncode", 0)
            output = result.get("output", "")
            if returncode == 0:
                return f"<returncode>0</returncode>\n<output>{output}</output>"
            else:
                return f"<returncode>{returncode}</returncode>\n<output>Error reading file '{path}': {output}</output>"
        except Exception as e:
            return f"<returncode>1</returncode>\n<output>Error reading file '{path}': {e}</output>"

    def write_file(path: str, content: str) -> str:
        """Write content to a file in the environment."""
        try:
            # Use unique delimiter to avoid EOF conflicts
            import uuid
            delimiter = f"EOF_{uuid.uuid4().hex[:8]}"
            cmd = f"cat > {path} << '{delimiter}'\n{content}\n{delimiter}"
            result = env.execute(cmd)
            if result.get("returncode") == 0:
                return f"<returncode>0</returncode>\n<output>Successfully wrote to {path}</output>"
            else:
                return f"<returncode>{result.get('returncode', 1)}</returncode>\n<output>Error writing to '{path}': {result.get('output', 'Unknown error')}</output>"
        except Exception as e:
            return f"<returncode>1</returncode>\n<output>Error writing to '{path}': {e}</output>"

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
            returncode = result.get("returncode", 0)
            output = result.get("output", "No files found")
            if returncode == 0:
                return f"<returncode>0</returncode>\n<output>{output}</output>"
            else:
                return f"<returncode>{returncode}</returncode>\n<output>Error listing files in '{path}': {output}</output>"
        except Exception as e:
            return f"<returncode>1</returncode>\n<output>Error listing files in '{path}': {e}</output>"

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
            returncode = result.get("returncode", 0)
            output = result.get("output", "")
            if returncode == 0:
                if output:
                    return f"<returncode>0</returncode>\n<output>{output}</output>"
                else:
                    return f"<returncode>0</returncode>\n<output>No matches found for '{query}'</output>"
            else:
                return f"<returncode>{returncode}</returncode>\n<output>Search completed (no matches or error): {output}</output>"
        except Exception as e:
            return f"<returncode>1</returncode>\n<output>Error searching for '{query}': {e}</output>"

    def run_tests() -> str:
        """Run tests in the environment."""
        try:
            # Try common test commands
            test_commands = ["python -m pytest", "python -m unittest", "npm test", "make test"]
            for cmd in test_commands:
                result = env.execute(cmd)
                returncode = result.get("returncode", 0)
                output = result.get("output", "")
                if returncode == 0:
                    return f"<returncode>0</returncode>\n<output>Tests passed with '{cmd}':\n{output}</output>"
                elif "not found" not in output:
                    return f"<returncode>{returncode}</returncode>\n<output>Test command '{cmd}' failed:\n{output}</output>"
            return f"<returncode>1</returncode>\n<output>No test commands found or all failed</output>"
        except Exception as e:
            return f"<returncode>1</returncode>\n<output>Error running tests: {e}</output>"

    def git_diff() -> str:
        """Show current changes as git diff."""
        try:
            result = env.execute("git diff")
            if result.get("returncode") == 0:
                output = result.get("output", "No changes")
                return f"<returncode>0</returncode>\n<output>Current changes:\n{output}</output>"
            else:
                return f"<returncode>{result.get('returncode', 1)}</returncode>\n<output>Error getting diff: {result.get('output', 'Unknown error')}</output>"
        except Exception as e:
            return f"<returncode>1</returncode>\n<output>Error getting diff: {e}</output>"

    def edit_file_lines(path: str, start_line: int, end_line: int, new_content: str) -> str:
        """Edit specific lines in a file using sed."""
        try:
            # Use sed to replace lines start_line to end_line
            # Escape newlines in content for sed
            escaped_content = new_content.replace('\n', '\\n').replace('/', '\\/')
            cmd = f"sed -i '{start_line},{end_line}c\\{escaped_content}' {path}"
            result = env.execute(cmd)
            if result.get("returncode") == 0:
                return f"<returncode>0</returncode>\n<output>Successfully edited lines {start_line}-{end_line} in {path}</output>"
            else:
                return f"<returncode>{result.get('returncode', 1)}</returncode>\n<output>Error editing lines in '{path}': {result.get('output', 'Unknown error')}</output>"
        except Exception as e:
            return f"<returncode>1</returncode>\n<output>Error editing lines in '{path}': {e}</output>"

    def replace_in_file(path: str, pattern: str, replacement: str) -> str:
        """Replace text in a file using sed regex."""
        try:
            # Escape special characters for sed
            escaped_pattern = pattern.replace('/', '\\/')
            escaped_replacement = replacement.replace('/', '\\/')
            cmd = f"sed -i 's/{escaped_pattern}/{escaped_replacement}/g' {path}"
            result = env.execute(cmd)
            if result.get("returncode") == 0:
                return f"<returncode>0</returncode>\n<output>Successfully replaced pattern in {path}</output>"
            else:
                return f"<returncode>{result.get('returncode', 1)}</returncode>\n<output>Error replacing in '{path}': {result.get('output', 'Unknown error')}</output>"
        except Exception as e:
            return f"<returncode>1</returncode>\n<output>Error replacing in '{path}': {e}</output>"

    def view_lines(path: str, start_line: int, end_line: int) -> str:
        """View specific lines from a file."""
        try:
            cmd = f"sed -n '{start_line},{end_line}p' {path}"
            result = env.execute(cmd)
            if result.get("returncode") == 0:
                output = result.get("output", "")
                return f"<returncode>0</returncode>\n<output>Lines {start_line}-{end_line} of {path}:\n{output}</output>"
            else:
                return f"<returncode>{result.get('returncode', 1)}</returncode>\n<output>Error viewing lines from '{path}': {result.get('output', 'Unknown error')}</output>"
        except Exception as e:
            return f"<returncode>1</returncode>\n<output>Error viewing lines from '{path}': {e}</output>"

    def append_file(path: str, content: str) -> str:
        """Append content to a file."""
        try:
            # Use echo to append content
            import uuid
            delimiter = f"EOF_{uuid.uuid4().hex[:8]}"
            cmd = f"cat >> {path} << '{delimiter}'\n{content}\n{delimiter}"
            result = env.execute(cmd)
            if result.get("returncode") == 0:
                return f"<returncode>0</returncode>\n<output>Successfully appended to {path}</output>"
            else:
                return f"<returncode>{result.get('returncode', 1)}</returncode>\n<output>Error appending to '{path}': {result.get('output', 'Unknown error')}</output>"
        except Exception as e:
            return f"<returncode>1</returncode>\n<output>Error appending to '{path}': {e}</output>"

    def submit_work() -> str:
        """Submit work by staging all changes and returning git diff. Use this when confident the task is complete."""
        try:
            # Stage all changes
            add_result = env.execute("git add -A")
            if add_result.get("returncode") != 0:
                return f"<returncode>{add_result.get('returncode', 1)}</returncode>\n<output>Error staging changes: {add_result.get('output', 'Unknown error')}</output>"
            
            # Get the staged diff
            diff_result = env.execute("git diff --cached")
            if diff_result.get("returncode") != 0:
                return f"<returncode>{diff_result.get('returncode', 1)}</returncode>\n<output>Error getting diff: {diff_result.get('output', 'Unknown error')}</output>"
            
            diff_output = diff_result.get("output", "")
            if not diff_output.endswith("\n"):
                diff_output += "\n"
            # Return the submission format similar to default agent
            return f"<returncode>0</returncode>\n<output>{diff_output}</output>"    
        except Exception as e:
            return f"<returncode>1</returncode>\n<output>Error submitting work: {e}</output>"

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
            func=git_diff,
            name="git_diff",
            desc="Show current changes as git diff",
            args={},
            arg_types={},
            arg_desc={},
        ),
        dspy.Tool(
            func=edit_file_lines,
            name="edit_file_lines",
            desc="Edit specific lines in a file",
            args={"path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}, "new_content": {"type": "string"}},
            arg_types={"path": str, "start_line": int, "end_line": int, "new_content": str},
            arg_desc={"path": "Path to file", "start_line": "Start line number", "end_line": "End line number", "new_content": "New content to replace lines"},
        ),
        dspy.Tool(
            func=replace_in_file,
            name="replace_in_file",
            desc="Replace text in a file using regex",
            args={"path": {"type": "string"}, "pattern": {"type": "string"}, "replacement": {"type": "string"}},
            arg_types={"path": str, "pattern": str, "replacement": str},
            arg_desc={"path": "Path to file", "pattern": "Regex pattern to find", "replacement": "Replacement text"},
        ),
        dspy.Tool(
            func=view_lines,
            name="view_lines",
            desc="View specific lines from a file",
            args={"path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}},
            arg_types={"path": str, "start_line": int, "end_line": int},
            arg_desc={"path": "Path to file", "start_line": "Start line number", "end_line": "End line number"},
        ),
        dspy.Tool(
            func=append_file,
            name="append_file",
            desc="Append content to a file",
            args={"path": {"type": "string"}, "content": {"type": "string"}},
            arg_types={"path": str, "content": str},
            arg_desc={"path": "Path to file", "content": "Content to append"},
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
