import os
import re
import dspy

def create_file(path: str, content: str) -> str:
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"[Success] Created file: {path}"
    except Exception as e:
        return f"[Error] Could not create file {path}: {e}"

def update_file(path: str, new_content: str) -> str:
    if not os.path.exists(path):
        return f"[Error] File not found: {path}"
    if not os.path.isfile(path):
        return f"[Error] Path is not a file: {path}"
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return f"[Success] Updated file: {path}"
    except Exception as e:
        return f"[Error] Could not update file {path}: {e}"

def update_file_regex(path: str, pattern: str, replacement: str) -> str:
    if not os.path.exists(path):
        return f"[Error] File not found: {path}"
    if not os.path.isfile(path):
        return f"[Error] Path is not a file: {path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        new_content, count = re.subn(pattern, replacement, content)
        if count == 0:
            return f"[Warning] No matches for '{pattern}' in {path}"
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return f"[Success] Applied {count} replacements in {path}"
    except Exception as e:
        return f"[Error] Could not apply regex update to {path}: {e}"

create_file_tool = dspy.Tool(
    func=create_file,
    name="create_file",
    desc="Create a new file with provided content (overwrites if exists).",
    args={"path": {"type": "string"}, "content": {"type": "string"}},
    arg_types={"path": str, "content": str},
    arg_desc={"path": "File path to create.", "content": "Initial file contents."},
)

update_file_tool = dspy.Tool(
    func=update_file,
    name="update_file",
    desc="Replace the entire contents of a file.",
    args={"path": {"type": "string"}, "new_content": {"type": "string"}},
    arg_types={"path": str, "new_content": str},
    arg_desc={"path": "Target file path.", "new_content": "New contents to write."},
)

update_file_regex_tool = dspy.Tool(
    func=update_file_regex,
    name="update_file_regex",
    desc="Apply a regex substitution (pattern → replacement) to a file.",
    args={
        "path": {"type": "string"},
        "pattern": {"type": "string"},
        "replacement": {"type": "string"},
    },
    arg_types={"path": str, "pattern": str, "replacement": str},
    arg_desc={
        "path": "Target file path.",
        "pattern": "Regex pattern to find.",
        "replacement": "Replacement string (supports backrefs).",
    },
)