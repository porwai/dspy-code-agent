"""Minimal DSPy tools for SWE-bench-like environments with advanced edit and search."""

import dspy
import os, re, shlex
from pathlib import Path
from typing import Any, Optional


def create_environment_tools(env: Any) -> list[dspy.Tool]:
    """Create minimal yet powerful DSPy tools for software-engineering agents."""

    def _exec(cmd: str) -> dict:
        """Execute a shell command in the environment and return standardized result.
        
        Args:
            cmd: Shell command to execute
            
        Returns:
            Dictionary with 'returncode' and 'output' keys. On error, returncode is 1
            and output contains error information.
        """
        try:
            return env.execute(cmd)
        except Exception as e:
            return {"returncode": 1, "output": f"[internal-error] {type(e).__name__}: {e}"}

    def _atomic_write(path: str, content: str) -> tuple[bool, str]:
        """Perform atomic file write operation safe for Docker environments.
        
        Creates parent directories if needed and writes content atomically using
        shell redirection to avoid partial writes.
        
        Args:
            path: Target file path
            content: Content to write to file
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            dir_path = os.path.dirname(path)
            if dir_path:
                _exec(f"mkdir -p {shlex.quote(dir_path)}")
            res = _exec(f"cat > {shlex.quote(path)} << 'EOF'\n{content}\nEOF")
            if res.get("returncode") == 0:
                return True, f"Wrote {path}"
            return False, res.get("output", "")
        except Exception as e:
            return False, str(e)

    # -------------------------
    # Core Commands
    # -------------------------
    def run_command(cmd: str) -> str:
        """Execute a shell command and return formatted result with status.
        
        Args:
            cmd: Shell command to execute
            
        Returns:
            Formatted string with SUCCESS/ERROR status and command output
        """
        res = _exec(cmd)
        status = "SUCCESS" if res.get("returncode") == 0 else "ERROR"
        return f"{status}: {res.get('output','').strip()}"

    def read_file(path: str, max_lines: int = 200) -> str:
        """Read the first N lines of a text file with line numbers.
        
        Args:
            path: Path to the file to read
            max_lines: Maximum number of lines to read (default: 200)
            
        Returns:
            Formatted file content with line numbers, or error message if file not found
        """
        if _exec(f"test -f {shlex.quote(path)} && echo exists").get("output","").strip() != "exists":
            return f"ERROR: File '{path}' not found."
        res = _exec(f"head -n {max_lines} {shlex.quote(path)}")
        lines = res.get("output","").splitlines()
        out = "\n".join(f"{i+1:04d}: {line}" for i, line in enumerate(lines))
        if len(lines) >= max_lines:
            out += f"\n[... truncated after {max_lines} lines ...]"
        return f"FILE: {path}\n---\n{out}"

    def read_file_range(path: str, start_line: int, end_line: int) -> str:
        """Read a specific range of lines from a file with line numbers.
        
        Args:
            path: Path to the file to read
            start_line: Starting line number (1-indexed)
            end_line: Ending line number (1-indexed)
            
        Returns:
            Formatted file content for the specified range with line numbers,
            or error message if invalid range or file read error
        """
        if start_line < 1 or end_line < start_line:
            return f"ERROR: Invalid range {start_line}-{end_line}"
        res = _exec(f"sed -n '{start_line},{end_line}p' {shlex.quote(path)}")
        if res.get("returncode") != 0:
            return f"ERROR reading {path}:{start_line}-{end_line}"
        lines = res.get("output","").splitlines()
        return "\n".join(f"{i+start_line:04d}: {line}" for i,line in enumerate(lines))

    def write_file(path: str, content: str) -> str:
        """Write or overwrite a file with the given content.
        
        Args:
            path: Path to the file to write
            content: Content to write to the file
            
        Returns:
            SUCCESS message with file path or ERROR message with details
        """
        ok, msg = _atomic_write(path, content)
        return f"SUCCESS: {msg}" if ok else f"ERROR: {msg}"

    def list_files(path: str = ".", show_hidden: bool = False, max_items: int = 100) -> str:
        """List files and directories in the specified path.
        
        Args:
            path: Directory path to list (default: current directory)
            show_hidden: Whether to include hidden files/directories (default: False)
            max_items: Maximum number of items to display (default: 100)
            
        Returns:
            Formatted directory listing with file sizes and types, or error message
        """
        try:
            # Check if path exists in environment
            res = _exec(f"test -d {shlex.quote(path)} && echo 'dir' || test -f {shlex.quote(path)} && echo 'file' || echo 'not_found'")
            path_type = res.get("output", "").strip()
            if path_type == "not_found":
                return f"ERROR: Path '{path}' does not exist"
            
            # Build ls command with options
            cmd_parts = ["ls", "-la" if show_hidden else "-l"]
            if not show_hidden:
                cmd_parts.append("--ignore=.*")  # Hide hidden files when show_hidden=False
            
            cmd = " ".join(cmd_parts) + f" {shlex.quote(path)}"
            res = _exec(cmd)
            
            if res.get("returncode") != 0:
                return f"ERROR: Failed to list directory '{path}': {res.get('output', '')}"
            
            lines = res.get("output", "").strip().splitlines()
            
            # Skip total line and limit results
            if lines and lines[0].startswith("total "):
                lines = lines[1:]
            
            if len(lines) > max_items:
                lines = lines[:max_items]
                truncated = True
            else:
                truncated = False
            
            # Format output
            result = f"Directory listing for: {path}\n"
            result += "-" * 50 + "\n"
            
            for line in lines:
                if line.strip():  # Skip empty lines
                    result += line + "\n"
            
            if truncated:
                result += f"\n[... truncated after {max_items} items ...]"
            
            return result
            
        except Exception as e:
            return f"ERROR: Failed to list directory '{path}': {e}"

    # -------------------------
    # Edit File (multi-mode)
    # -------------------------
    def edit_file(command: str, path: str,
                  content: Optional[str] = None,
                  line_number: Optional[int] = None,
                  old_str: Optional[str] = None,
                  new_str: Optional[str] = None) -> str:
        """Perform flexible file edits with multiple operation modes.
        
        Supports various file operations: view, create, replace, insert, delete.
        
        Args:
            command: Operation to perform ('view', 'create', 'str_replace', 'insert', 'delete')
            path: Path to the file to operate on
            content: Content for create/insert operations
            line_number: Line number for insert/delete operations (1-indexed)
            old_str: String to replace in str_replace operation
            new_str: Replacement string for str_replace operation
            
        Returns:
            SUCCESS message with operation details or ERROR message with details
        """
        try:
            if command == "view":
                # Check if file exists in environment
                res = _exec(f"test -f {shlex.quote(path)} && echo 'file' || test -d {shlex.quote(path)} && echo 'dir' || echo 'not_found'")
                file_type = res.get("output", "").strip()
                if file_type == "not_found":
                    return f"Path {path} does not exist"
                if file_type == "dir":
                    return f"Path {path} is a directory"
                # File exists, read it
                res = _exec(f"head -c 5000 {shlex.quote(path)}")
                out = res.get("output","")
                return out + ("..." if len(out) >= 5000 else "")

            elif command == "create":
                # Check if file already exists in environment
                res = _exec(f"test -e {shlex.quote(path)} && echo 'exists' || echo 'not_exists'")
                if res.get("output", "").strip() == "exists":
                    return f"Error: {path} already exists"
                # Create parent directory and file
                _exec(f"mkdir -p {shlex.quote(os.path.dirname(path))}")
                ok, msg = _atomic_write(path, content or "")
                return f"SUCCESS: {msg}" if ok else f"ERROR: {msg}"

            elif command == "str_replace":
                # Check if file exists in environment
                res = _exec(f"test -f {shlex.quote(path)} && echo 'file' || echo 'not_file'")
                if res.get("output", "").strip() != "file":
                    return f"Error: {path} is not a file"
                res = _exec(f"cat {shlex.quote(path)}")
                text = res.get("output","")
                if old_str not in text:
                    return f"Error: Could not find exact match for replacement"
                new_text = text.replace(old_str, new_str or "")
                ok, msg = _atomic_write(path, new_text)
                return f"SUCCESS: {msg}" if ok else f"ERROR: {msg}"

            elif command == "insert":
                # Check if file exists in environment
                res = _exec(f"test -f {shlex.quote(path)} && echo 'file' || echo 'not_file'")
                if res.get("output", "").strip() != "file":
                    return f"Error: {path} is not a file"
                if line_number is None:
                    return "Error: Line number required for insert"
                res = _exec(f"cat {shlex.quote(path)}")
                lines = res.get("output","").splitlines(keepends=True)
                if not (1 <= line_number <= len(lines)+1):
                    return f"Error: Invalid line number {line_number}"
                lines.insert(line_number-1, (content or "") + "\n")
                ok, msg = _atomic_write(path, "".join(lines))
                return f"SUCCESS: Inserted at line {line_number}" if ok else f"ERROR: {msg}"

            elif command == "delete":
                # Check if file exists in environment
                res = _exec(f"test -f {shlex.quote(path)} && echo 'file' || echo 'not_file'")
                if res.get("output", "").strip() != "file":
                    return f"Error: {path} is not a file"
                if line_number is None:
                    return "Error: Line number required for delete"
                res = _exec(f"cat {shlex.quote(path)}")
                lines = res.get("output","").splitlines(keepends=True)
                if not (1 <= line_number <= len(lines)):
                    return f"Error: Invalid line {line_number}"
                del lines[line_number-1]
                ok, msg = _atomic_write(path, "".join(lines))
                return f"SUCCESS: Deleted line {line_number}" if ok else f"ERROR: {msg}"

            else:
                return f"Error: Unknown command '{command}'"

        except Exception as e:
            return f"Error performing {command}: {e}"
    
    # -------------------------
    # Content Search (semantic grep)
    # -------------------------
    def file_content_search(query: str,
                            exclude_pattern: Optional[str] = "*.pyc,*.git*,__pycache__,*.bin,*.exe,*.dll,*.so") -> str:
        """Search text content within files for a pattern (regex or plain text).

        Efficiently searches across files in the environment using `grep`,
        skipping binary/cache files and limiting context for readability.

        Args:
            query: The search term or regex pattern to look for
            exclude_pattern: Comma-separated glob patterns to exclude from search

        Returns:
            Formatted matches with file paths, line numbers, and context.
        """
        if not query.strip():
            return "Error: Empty search query."

        try:
            context_lines = 3
            max_matches = 10
            max_files = 50
            matches_found = 0
            results = []

            # Build find command with exclusions
            exclude_patterns = [p.strip() for p in exclude_pattern.split(',')] if exclude_pattern else []
            find_cmd = "find . -type f"
            for pattern in exclude_patterns:
                if pattern:
                    find_cmd += f" -not -path '*/{pattern}' -not -name '{pattern}'"

            # Exclude known large/unhelpful files
            find_cmd += " -not -name 'input.json' -not -name 'agent_args.json' -not -name 'steps.json' -not -name 'main.py'"

            # Execute find
            res = _exec(find_cmd)
            if res.get("returncode") != 0:
                return f"Error listing files: {res.get('output', '')}"

            file_paths = [line.strip() for line in res.get("output", "").splitlines() if line.strip()]
            file_paths = file_paths[:max_files]

            for file_path in file_paths:
                if matches_found >= max_matches:
                    break

                # Grep with context
                grep_cmd = f"grep -i -n -C {context_lines} {shlex.quote(query)} {shlex.quote(file_path)}"
                grep_res = _exec(grep_cmd)

                if grep_res.get("returncode") == 0 and grep_res.get("output", "").strip():
                    output = grep_res["output"].strip()
                    # Truncate overly long outputs
                    if len(output) > 2000:
                        output = output[:1000] + "\n... (truncated) ...\n" + output[-1000:]

                    results.append(f"File: {file_path}\n{output}\n---")
                    matches_found += 1

            if not results:
                return f"No matches found for '{query}' in searched files."

            summary = f"Found {matches_found} matches for '{query}' across {len(file_paths)} files.\n\n"
            return summary + "\n".join(results)

        except Exception as e:
            return f"Error searching files: {type(e).__name__}: {e}"
    
    # -------------------------
    # Minimal Test + Submit
    # -------------------------
    def run_tests() -> str:
        """Run repository tests to verify code changes.
        
        Attempts to run tests using pytest first, then falls back to unittest,
        or reports if no tests are found.
        
        Returns:
            Test output or message indicating no tests found
        """
        res = _exec("pytest -q || python -m unittest -q || echo 'No tests found'")
        return res.get("output", "(no output)")

    def finish() -> str:
        """Submit work by generating git patch diff and terminating the session.
        
        Stages all changes and generates a git diff showing the modifications
        made during the current session. This tool should be called when the task
        is complete and should terminate the agent execution.
        
        Returns:
            Git diff output showing all staged changes, or error message if no diff available
        """
        _exec("git add -A")
        # Use proper git diff format for patches
        diff = _exec("git diff --cached --no-color --no-ext-diff")
        result = diff.get("output", "ERROR: No diff output")
        # Ensure the diff ends with a newline; some patch tools error if missing
        if result and not result.endswith("\n"):
            result += "\n"
        
        # Add termination signal to make it clear this should end the session
        
        return result

    def grep_search(pattern: str, path: str = ".", options: str = "-n -i") -> str:
        """Search for text patterns in files using grep.
        
        Args:
            pattern: Text pattern to search for
            path: Directory or file path to search in (default: current directory)
            options: Grep options (default: "-n -i" for line numbers and case-insensitive)
            
        Returns:
            Grep output with matching lines and line numbers, or error message
        """
        try:
            # Build grep command
            cmd = f"grep {options} {shlex.quote(pattern)} {shlex.quote(path)}"
            res = _exec(cmd)
            
            if res.get("returncode") == 0:
                output = res.get("output", "").strip()
                if output:
                    return f"Found matches:\n{output}"
                else:
                    return f"No matches found for '{pattern}' in {path}"
            else:
                # Grep returns non-zero for no matches, but also for errors
                output = res.get("output", "").strip()
                if "No such file or directory" in output:
                    return f"Error: Path '{path}' does not exist"
                elif "Is a directory" in output:
                    return f"Error: '{path}' is a directory, use recursive search with -r option"
                else:
                    return f"No matches found for '{pattern}' in {path}"
                    
        except Exception as e:
            return f"Error running grep: {e}"

    # -------------------------
    # Register tools
    # -------------------------
    return [
        dspy.Tool(func=run_command, name="run_command",
                  desc="""Execute a shell command and return formatted result with status.
        
        Args:
            cmd: Shell command to execute
            
        Returns:
            Formatted string with SUCCESS/ERROR status and command output""",
                  args={"cmd":{"type":"string"}}),
        dspy.Tool(func=read_file, name="read_file",
                  desc="""Read the first N lines of a text file with line numbers.
        
        Args:
            path: Path to the file to read
            max_lines: Maximum number of lines to read (default: 200)
            
        Returns:
            Formatted file content with line numbers, or error message if file not found""",
                  args={"path":{"type":"string"},
                        "max_lines":{"type":"integer","default":200}}),
        dspy.Tool(func=read_file_range, name="read_file_range",
                  desc="""Read a specific range of lines from a file with line numbers.
        
        Args:
            path: Path to the file to read
            start_line: Starting line number (1-indexed)
            end_line: Ending line number (1-indexed)
            
        Returns:
            Formatted file content for the specified range with line numbers,
            or error message if invalid range or file read error
                  """,
                  args={"path":{"type":"string"},
                        "start_line":{"type":"integer"},
                        "end_line":{"type":"integer"}}),
        dspy.Tool(func=write_file, name="write_file",
                  desc="""Write or overwrite a file with the given content.
         
         Args:
             path: Path to the file to write
             content: Content to write to the file
             
         Returns:
             SUCCESS message with file path or ERROR message with details
                   """,
                  args={"path":{"type":"string"},
                        "content":{"type":"string"}}),
        dspy.Tool(func=list_files, name="list_files",
                  desc="""List files and directories in the specified path.
         
         Args:
             path: Directory path to list (default: current directory)
             show_hidden: Whether to include hidden files/directories (default: False)
             max_items: Maximum number of items to display (default: 100)
             
         Returns:
             Formatted directory listing with file sizes and types, or error message
                   """,
                  args={"path":{"type":"string","default":"."},
                        "show_hidden":{"type":"boolean","default":False},
                        "max_items":{"type":"integer","default":100}}),
        dspy.Tool(func=edit_file, name="edit_file",
                  desc="""Perform flexible file edits with multiple operation modes.
        
        Supports various file operations: view, create, replace, insert, delete.
        
        Args:
            command: Operation to perform ('view', 'create', 'str_replace', 'insert', 'delete')
            path: Path to the file to operate on
            content: Content for create/insert operations
            line_number: Line number for insert/delete operations (1-indexed)
            old_str: String to replace in str_replace operation
            new_str: Replacement string for str_replace operation
            
        Returns:
            SUCCESS message with operation details or ERROR message with details
                  """,
                  args={"command":{"type":"string"},
                        "path":{"type":"string"},
                        "content":{"type":"string","optional":True},
                        "line_number":{"type":"integer","optional":True},
                        "old_str":{"type":"string","optional":True},
                        "new_str":{"type":"string","optional":True}}),
        dspy.Tool(func=grep_search, name="grep_search",
                  desc="""Search for text patterns in files using grep command.
         
         Simple and reliable text search using standard grep command. Supports
         all standard grep options and patterns.
         
         Args:
             pattern: Text pattern to search for
             path: Directory or file path to search in (default: current directory)
             options: Grep options (default: "-n -i" for line numbers and case-insensitive)
             
         Returns:
             Grep output with matching lines and line numbers, or error message
                   """,
                  args={"pattern":{"type":"string"},
                        "path":{"type":"string","default":"."},
                        "options":{"type":"string","default":"-n -i"}}),
        dspy.Tool(func=finish, name="finish",
                  desc="""Generate a git patch diff of all changes and terminate the session.
         
         This tool should ONLY be called when the task is completely finished.
         It stages all modifications and provides formatted diff output for submission.
         After calling this tool, the agent should terminate as the task is complete.
         
         Returns:
             Git diff output showing all staged changes with termination signal"""),
        dspy.Tool(func=file_content_search, name="file_content_search",
                  desc="""Search text content within files for a pattern (regex or plain text).

        Efficiently searches across files in the environment using `grep`,
        skipping binary/cache files and limiting context for readability.

        Args:
            query: The search term or regex pattern to look for
            exclude_pattern: Comma-separated glob patterns to exclude from search

        Returns:
            Formatted matches with file paths, line numbers, and context.""",
            args={"query":{"type":"string"},
                        "exclude_pattern":{"type":"string","optional":True}}),
    ]
