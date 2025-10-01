import os
import shutil
import dspy

def cat(path: str) -> str:
    if not os.path.exists(path):
        return f"[Error] File not found: {path}"
    if not os.path.isfile(path):
        return f"[Error] Path is not a file: {path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"[Error] Could not read file {path}: {e}"

def rm(path: str) -> str:
    if not os.path.exists(path):
        return f"[Error] File not found: {path}"
    if not os.path.isfile(path):
        return f"[Error] Path is not a file: {path}"
    try:
        os.remove(path)
        return f"[Success] Removed file: {path}"
    except Exception as e:
        return f"[Error] Could not remove file {path}: {e}"

def mv(src: str, dest: str) -> str:
    if not os.path.exists(src):
        return f"[Error] Source file not found: {src}"
    if not os.path.isfile(src):
        return f"[Error] Source path is not a file: {src}"
    try:
        shutil.move(src, dest)
        return f"[Success] Moved {src} → {dest}"
    except Exception as e:
        return f"[Error] Could not move {src} → {dest}: {e}"

cat_tool = dspy.Tool(
    func=cat,
    name="cat",
    desc="Read and return the contents of a file.",
    args={"path": {"type": "string"}},
    arg_types={"path": str},
    arg_desc={"path": "Path to a file."},
)

rm_tool = dspy.Tool(
    func=rm,
    name="rm",
    desc="Remove a file.",
    args={"path": {"type": "string"}},
    arg_types={"path": str},
    arg_desc={"path": "Path to a file to remove."},
)

mv_tool = dspy.Tool(
    func=mv,
    name="mv",
    desc="Move or rename a file.",
    args={
        "src": {"type": "string"},
        "dest": {"type": "string"}
    },
    arg_types={"src": str, "dest": str},
    arg_desc={"src": "Source file path.", "dest": "Destination path."},
)