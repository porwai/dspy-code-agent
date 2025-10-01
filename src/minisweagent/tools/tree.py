import os
import dspy

# -------- core --------
def list_repo_tree(path: str = ".") -> str:
    tree = []
    for root, dirs, files in os.walk(path):
        # Skip common hidden metadata dirs
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        files = [f for f in files if not f.startswith(".")]

        level = root.replace(path, "").count(os.sep)
        indent = " " * 2 * level
        tree.append(f"{indent}{os.path.basename(root) or root}/")
        subindent = " " * 2 * (level + 1)
        for f in files:
            tree.append(f"{subindent}{f}")
    return "\n".join(tree)

def find_file_by_name(path: str = ".", pattern: str = "") -> str:
    matches = []
    for root, _, files in os.walk(path):
        for f in files:
            if pattern.lower() in f.lower():
                matches.append(os.path.join(root, f))
    return "\n".join(matches) if matches else f"[Warning] No files match '{pattern}'"

# -------- tools --------
tree_tool = dspy.Tool(
    func=list_repo_tree,
    name="list_repo_tree",
    desc="List files/folders as a tree from a path.",
    args={"path": {"type": "string", "default": "."}},
    arg_types={"path": str},
    arg_desc={"path": "Directory to start walking from (default: current)."},
)

find_file_tool = dspy.Tool(
    func=find_file_by_name,
    name="find_file_by_name",
    desc="Find files whose names contain a substring.",
    args={
        "path": {"type": "string", "default": "."},
        "pattern": {"type": "string"}
    },
    arg_types={"path": str, "pattern": str},
    arg_desc={
        "path": "Directory to search within.",
        "pattern": "Case-insensitive substring to match in filenames."
    },
)
