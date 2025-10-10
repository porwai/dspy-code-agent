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
    # Line-Precise Editing Tools
    # -------------------------

    def _read_text(path: str) -> tuple[bool, str]:
        res = _exec(f"cat {shlex.quote(path)}")
        if res.get("returncode") != 0:
            return False, f"ERROR: Could not read file '{path}': {res.get('output', 'Unknown error')}"
        return True, res.get("output", "")

    def _show_diff(path: str, max_lines: int = 200) -> str:
        """Return a short diff snippet for the given path (helps agent see what changed)."""
        diff = _exec(f"git diff --no-color -- {shlex.quote(path)} | head -n {int(max_lines)}")
        return diff.get("output", "").strip()

    def _ensure_trailing_newline(s: str) -> str:
        return s if s.endswith("\n") else (s + "\n")

    def replace_lines(path: str, start_line: int, end_line: int, new_text: str) -> str:
        """
        Replace lines [start_line, end_line] (1-based, inclusive) with new_text.
        If you want to replace a single line, set start_line == end_line.
        """
        ok, content = _read_text(path)
        if not ok:
            return content

        # Keep original line endings
        lines = content.splitlines(keepends=True)
        n = len(lines)
        if start_line < 1 or end_line < start_line or end_line > n:
            return f"ERROR: Invalid line range [{start_line}, {end_line}] in file with {n} lines."

        new_block = new_text.splitlines(keepends=False)
        # Preserve line endings; ensure newline at end of each inserted line
        new_block = [ln + ("\n" if not ln.endswith("\n") else "") for ln in new_block]
        # Special case: if new_text is empty, we still keep it as an empty list (deletion)
        lines[start_line - 1 : end_line] = new_block

        new_content = "".join(lines)
        ok, msg = _atomic_write(path, new_content)
        if not ok:
            return f"ERROR: {msg}"

        snippet = _show_diff(path)
        return f"SUCCESS: Replaced lines {start_line}-{end_line} in {path}\n{('DIFF:\\n' + snippet) if snippet else ''}"

    def insert_lines(path: str, line: int, new_text: str, position: str = "after") -> str:
        """
        Insert new_text before or after a given 1-based line number.
        position ∈ {'before','after'}
        - If line == 0 and position == 'before', inserts at file start.
        - If line == number_of_lines and position == 'after', appends at end.
        """
        ok, content = _read_text(path)
        if not ok:
            return content

        lines = content.splitlines(keepends=True)
        n = len(lines)
        if line < 0 or line > n:
            return f"ERROR: Invalid target line {line} for file with {n} lines."

        ins = new_text.splitlines(keepends=False)
        ins = [ln + ("\n" if not ln.endswith("\n") else "") for ln in ins]

        if position not in ("before", "after"):
            return "ERROR: position must be 'before' or 'after'."

        idx = line - 1 if position == "before" else line
        if idx < 0:
            idx = 0
        if idx > n:
            idx = n

        lines[idx:idx] = ins
        new_content = "".join(lines)

        ok, msg = _atomic_write(path, new_content)
        if not ok:
            return f"ERROR: {msg}"

        snippet = _show_diff(path)
        return f"SUCCESS: Inserted {len(ins)} line(s) {position} line {line} in {path}\n{('DIFF:\\n' + snippet) if snippet else ''}"

    def delete_lines(path: str, start_line: int, end_line: int) -> str:
        """Delete lines [start_line, end_line] (1-based, inclusive)."""
        ok, content = _read_text(path)
        if not ok:
            return content

        lines = content.splitlines(keepends=True)
        n = len(lines)
        if start_line < 1 or end_line < start_line or end_line > n:
            return f"ERROR: Invalid line range [{start_line}, {end_line}] in file with {n} lines."

        del lines[start_line - 1 : end_line]
        new_content = "".join(lines)

        ok, msg = _atomic_write(path, new_content)
        if not ok:
            return f"ERROR: {msg}"

        snippet = _show_diff(path)
        return f"SUCCESS: Deleted lines {start_line}-{end_line} in {path}\n{('DIFF:\\n' + snippet) if snippet else ''}"

    def regex_replace(path: str, pattern: str, replacement: str, flags: str = "") -> str:
        """
        Regex (sed-like) replacement over the whole file.
        flags: combination of 'i' (IGNORECASE), 'm' (MULTILINE), 's' (DOTALL), 'g' (global).
        If 'g' is omitted, only the first match is replaced.
        Supports backrefs like \\1 in 'replacement'.
        """
        ok, content = _read_text(path)
        if not ok:
            return content

        re_flags = 0
        import re as _re
        if "i" in flags:
            re_flags |= _re.IGNORECASE
        if "m" in flags:
            re_flags |= _re.MULTILINE
        if "s" in flags:
            re_flags |= _re.DOTALL

        count = 0 if "g" in flags else 1
        try:
            new_content, subs = _re.subn(pattern, replacement, content, count=count, flags=re_flags)
        except Exception as e:
            return f"ERROR: Invalid regex or replacement: {type(e).__name__}: {e}"

        if subs == 0:
            return "ERROR: No matches found; file unchanged."

        # Keep a trailing newline if the file had one originally
        if content.endswith("\n") and not new_content.endswith("\n"):
            new_content += "\n"

        ok, msg = _atomic_write(path, new_content)
        if not ok:
            return f"ERROR: {msg}"

        snippet = _show_diff(path)
        return f"SUCCESS: Performed {subs} substitution(s) in {path}\n{('DIFF:\\n' + snippet) if snippet else ''}"

    def replace_between(path: str, start_pattern: str, end_pattern: str, new_text: str, include_markers: bool = False, flags: str = "") -> str:
        """
        Replace the text between the first match of start_pattern and the next match of end_pattern.
        - start_pattern, end_pattern are regexes.
        - If include_markers=True, the markers themselves are replaced, otherwise only the content between them.
        - flags like regex_replace: 'i', 'm', 's'.
        """
        ok, content = _read_text(path)
        if not ok:
            return content

        import re as _re
        re_flags = 0
        if "i" in flags:
            re_flags |= _re.IGNORECASE
        if "m" in flags:
            re_flags |= _re.MULTILINE
        if "s" in flags:
            re_flags |= _re.DOTALL

        s_match = _re.search(start_pattern, content, flags=re_flags)
        if not s_match:
            return "ERROR: start_pattern not found."

        e_match = _re.search(end_pattern, content[s_match.end():], flags=re_flags)
        if not e_match:
            return "ERROR: end_pattern not found after start_pattern."

        start_idx = s_match.start() if include_markers else s_match.end()
        end_idx = s_match.end() + e_match.end() if include_markers else s_match.end() + e_match.start()

        # Normalize new_text block with newline handling
        block = new_text
        if not block.endswith("\n"):
            block += "\n"

        new_content = content[:start_idx] + block + content[end_idx:]

        ok, msg = _atomic_write(path, new_content)
        if not ok:
            return f"ERROR: {msg}"

        snippet = _show_diff(path)
        return f"SUCCESS: Replaced content between patterns in {path}\n{('DIFF:\\n' + snippet) if snippet else ''}"

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
        dspy.Tool(
            func=replace_lines,
            name="replace_lines",
            desc="Replace a range of lines [start_line, end_line] (1-based, inclusive) with new_text.",
            args={"path": {"type": "string"}, "start_line": {"type": "number"}, "end_line": {"type": "number"}, "new_text": {"type": "string"}},
            arg_types={"path": str, "start_line": int, "end_line": int, "new_text": str},
            arg_desc={"path": "File path", "start_line": "Start line (1-based)", "end_line": "End line (inclusive, 1-based)", "new_text": "Replacement text (can be multiple lines)"},
        ),
        dspy.Tool(
            func=insert_lines,
            name="insert_lines",
            desc="Insert new_text before or after a specific 1-based line number.",
            args={"path": {"type": "string"}, "line": {"type": "number"}, "new_text": {"type": "string"}, "position": {"type": "string", "default": "after"}},
            arg_types={"path": str, "line": int, "new_text": str, "position": str},
            arg_desc={"path": "File path", "line": "Target line (1-based)", "new_text": "Text to insert", "position": "before|after (default: after)"},
        ),
        dspy.Tool(
            func=delete_lines,
            name="delete_lines",
            desc="Delete a range of lines [start_line, end_line] (1-based, inclusive).",
            args={"path": {"type": "string"}, "start_line": {"type": "number"}, "end_line": {"type": "number"}},
            arg_types={"path": str, "start_line": int, "end_line": int},
            arg_desc={"path": "File path", "start_line": "Start line (1-based)", "end_line": "End line (inclusive, 1-based)"},
        ),
        dspy.Tool(
            func=regex_replace,
            name="regex_replace",
            desc="Regex (sed-like) substitution across the whole file. flags supports i,m,s,g.",
            args={"path": {"type": "string"}, "pattern": {"type": "string"}, "replacement": {"type": "string"}, "flags": {"type": "string", "default": ""}},
            arg_types={"path": str, "pattern": str, "replacement": str, "flags": str},
            arg_desc={"path": "File path", "pattern": "Python regex", "replacement": "Replacement string (\\1 etc.)", "flags": "Any of i,m,s,g"},
        ),
        dspy.Tool(
            func=replace_between,
            name="replace_between",
            desc="Replace content between start_pattern and end_pattern (regex). Use include_markers=True to replace markers too.",
            args={"path": {"type": "string"}, "start_pattern": {"type": "string"}, "end_pattern": {"type": "string"}, "new_text": {"type": "string"}, "include_markers": {"type": "boolean", "default": False}, "flags": {"type": "string", "default": ""}},
            arg_types={"path": str, "start_pattern": str, "end_pattern": str, "new_text": str, "include_markers": bool, "flags": str},
            arg_desc={
                "path": "File path",
                "start_pattern": "Regex for start marker",
                "end_pattern": "Regex for end marker (must occur after start)",
                "new_text": "Replacement block",
                "include_markers": "If True, also replaces the markers themselves",
                "flags": "Any of i,m,s"
            },
        ),
    ]