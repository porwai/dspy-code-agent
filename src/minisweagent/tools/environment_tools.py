""""DSPy tools that work with mini-SWE-agent environments (Docker/local)."""

import dspy
from typing import Any
import os
import re
import shlex
import time


def create_environment_tools(env: Any) -> list[dspy.Tool]:
    """Create simple, intuitive DSPy tools for software engineering tasks.

    Design goals:
    - Keep tools simple and focused
    - Clear, actionable descriptions
    - Minimal complexity for the agent
    """

    # -------------------------
    # Helpers
    # -------------------------
    def _exec(cmd: str) -> dict:
        """Execute a shell command in the environment and return {'returncode', 'output'}."""
        try:
            return env.execute(cmd)  # expected to return {'returncode': int, 'output': str}
        except Exception as e:
            return {"returncode": 1, "output": f"[internal-error] {type(e).__name__}: {e}"}

    def _atomic_write(path: str, content: str) -> tuple[bool, str]:
        """Write file atomically with better Docker compatibility."""
        try:
            # Create directory if it doesn't exist
            dir_path = os.path.dirname(path)
            if dir_path:
                mkdir_cmd = f"mkdir -p {shlex.quote(dir_path)}"
                _exec(mkdir_cmd)
            
            # Try multiple approaches for better Docker compatibility
            approaches = [
                # Approach 1: Direct echo with proper escaping
                f"echo {shlex.quote(content)} > {shlex.quote(path)}",
                # Approach 2: Using printf for better control
                f"printf '%s' {shlex.quote(content)} > {shlex.quote(path)}",
                # Approach 3: Using cat with here-doc (original approach)
                f"cat > {shlex.quote(path)} << 'EOF'\n{content}\nEOF",
            ]
            
            for i, cmd in enumerate(approaches, 1):
                res = _exec(cmd)
                if res.get("returncode") == 0:
                    # Verify the file was actually written
                    verify_cmd = f"test -f {shlex.quote(path)} && wc -l {shlex.quote(path)}"
                    verify_res = _exec(verify_cmd)
                    if verify_res.get("returncode") == 0:
                        return True, f"Successfully wrote to {path} (approach {i})"
                    else:
                        # Try next approach
                        continue
                else:
                    # Try next approach
                    continue
            
            return False, f"All write approaches failed for '{path}': {res.get('output', 'Unknown error')}"
        except Exception as e:
            return False, f"Error writing to '{path}': {e}"

    # -------------------------
    # Simple Core Tools
    # -------------------------
    def run_command(cmd: str) -> str:
        """Run any shell command. Use this for everything - file operations, running scripts, etc."""
        # Special handling for Python script execution
        if "python" in cmd and ".py" in cmd:
            # First check if the file exists
            script_path = cmd.split()[-1]  # Get the last argument (script path)
            check_cmd = f"test -f {shlex.quote(script_path)} && echo 'File exists' || echo 'File not found'"
            check_res = _exec(check_cmd)
            if "File not found" in check_res.get("output", ""):
                return f"ERROR: Script file '{script_path}' does not exist. Cannot execute: {cmd}"
        
        res = _exec(cmd)
        if res.get("returncode") == 0:
            return f"SUCCESS: {res.get('output', '')}"
        else:
            return f"ERROR: {res.get('output', '')}"

    def read_file(path: str) -> str:
        """Read the contents of a file."""
        res = _exec(f"cat {shlex.quote(path)}")
        if res.get("returncode") == 0:
            return f"FILE CONTENTS:\n{res.get('output', '')}"
        return f"ERROR: Could not read file '{path}': {res.get('output', 'Unknown error')}"

    def write_file(path: str, content: str) -> str:
        """Write content to a file (creates new file or overwrites existing)."""
        ok, msg = _atomic_write(path, content)
        if ok:
            return f"SUCCESS: {msg}"
        return f"ERROR: {msg}"

    def list_files(path: str = ".") -> str:
        """List all files in a directory."""
        cmd = f"ls -la {shlex.quote(path)}"
        res = _exec(cmd)
        if res.get("returncode") == 0:
            return f"FILES:\n{res.get('output', '')}"
        return f"ERROR: Could not list files: {res.get('output', '')}"

    def search_files(query: str, path: str = ".") -> str:
        """Search for text in files."""
        cmd = f"grep -r '{query}' {shlex.quote(path)} | head -20"
        res = _exec(cmd)
        if res.get("returncode") == 0:
            out = res.get("output", "").strip()
            return f"SEARCH RESULTS:\n{out}" if out else f"No matches found for '{query}'"
        return f"ERROR: Search failed: {res.get('output', '')}"

    # -------------------------
    # Simple File Editing
    # -------------------------
    def edit_file(path: str, old_text: str, new_text: str) -> str:
        """Replace text in a file. Use this to make changes to files."""
        # Read the file first
        res = _exec(f"cat {shlex.quote(path)}")
        if res.get("returncode") != 0:
            return f"ERROR: Could not read file '{path}': {res.get('output', 'Unknown error')}"
        
        content = res.get("output", "")
        if old_text not in content:
            return f"ERROR: Text '{old_text}' not found in file '{path}'"
        
        # Replace the text
        new_content = content.replace(old_text, new_text)
        
        # Write back to file
        ok, msg = _atomic_write(path, new_content)
        if ok:
            return f"SUCCESS: Updated file {path}"
        return f"ERROR: {msg}"

    # -------------------------
    # Testing and Submission
    # -------------------------
    def run_tests() -> str:
        """Run tests to check if your changes work."""
        cmd = "python -m pytest -q || python -m unittest || pytest || echo 'No tests found'"
        res = _exec(cmd)
        if res.get("returncode") == 0:
            return f"TEST RESULTS:\n{res.get('output', '')}"
        return f"TEST ERROR:\n{res.get('output', '')}"

    def submit_work() -> str:
        """Submit your work when you're done. This creates a patch of all your changes."""
        try:
            add_result = env.execute("git add -A")
            if add_result.get("returncode") != 0:
                return f"ERROR: Could not stage changes: {add_result.get('output', 'Unknown error')}"

            # Keep git's output exact; avoid color/externals; include binary diffs just in case
            diff_result = env.execute("git diff --cached --no-color --no-ext-diff --binary")
            if diff_result.get("returncode") != 0:
                return f"ERROR: Could not get diff: {diff_result.get('output', 'Unknown error')}"

            diff_raw = diff_result.get("output") or ""

            # Use .strip() ONLY for the emptiness check, but return the raw text unmodified
            if not diff_raw.strip():
                return "ERROR: No changes to submit. Make sure you have made modifications to the codebase."

            # Ensure the patch ends with a newline (don’t remove anything)
            if not diff_raw.endswith("\n"):
                diff_raw += "\n"

            return diff_raw
        except Exception as e:
            return f"ERROR: Could not submit work: {e}"
    # -------------------------
    # Simple Tool Registration
    # -------------------------
    return [
        # Essential tools only
        dspy.Tool(
            func=run_command,
            name="run_command",
            desc="Run any shell command. Use this for everything - file operations, running scripts, etc.",
            args={"cmd": {"type": "string"}},
            arg_types={"cmd": str},
            arg_desc={"cmd": "The shell command to run"},
        ),
        dspy.Tool(
            func=read_file,
            name="read_file",
            desc="Read the contents of a file",
            args={"path": {"type": "string"}},
            arg_types={"path": str},
            arg_desc={"path": "Path to the file to read"},
        ),
        dspy.Tool(
            func=write_file,
            name="write_file",
            desc="Write content to a file (creates new file or overwrites existing)",
            args={"path": {"type": "string"}, "content": {"type": "string"}},
            arg_types={"path": str, "content": str},
            arg_desc={"path": "Path to the file to write", "content": "Content to write to the file"},
        ),
        dspy.Tool(
            func=list_files,
            name="list_files",
            desc="List all files in a directory",
            args={"path": {"type": "string", "default": "."}},
            arg_types={"path": str},
            arg_desc={"path": "Directory to list files from (default: current directory)"},
        ),
        dspy.Tool(
            func=search_files,
            name="search_files",
            desc="Search for text in files",
            args={"query": {"type": "string"}, "path": {"type": "string", "default": "."}},
            arg_types={"query": str, "path": str},
            arg_desc={"query": "Text to search for", "path": "Directory to search in (default: current directory)"},
        ),
        dspy.Tool(
            func=edit_file,
            name="edit_file",
            desc="Replace text in a file. Use this to make changes to files",
            args={"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}},
            arg_types={"path": str, "old_text": str, "new_text": str},
            arg_desc={"path": "Path to the file to edit", "old_text": "Text to replace", "new_text": "New text to replace with"},
        ),
        dspy.Tool(
            func=run_tests,
            name="run_tests",
            desc="Run tests to check if your changes work",
            args={},
            arg_types={},
            arg_desc={},
        ),
        dspy.Tool(
            func=submit_work,
            name="submit_work",
            desc="Submit your work when you're done. This creates a patch of all your changes",
            args={},
            arg_types={},
            arg_desc={},
        ),
    ]