import os
import re
import dspy
from collections import Counter

# -------------------------
# Core Functions
# -------------------------

def search_code(query: str, path: str = ".") -> str:
    """Search for a substring in all text files under path."""
    results = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.startswith("."):  # skip hidden
                continue
            file_path = os.path.join(root, f)
            try:
                with open(file_path, "r", encoding="utf-8") as fh:
                    for i, line in enumerate(fh, 1):
                        if query in line:
                            results.append(f"{file_path}:{i}: {line.strip()}")
            except Exception:
                continue
    return "\n".join(results) if results else f"[Warning] No matches for '{query}'"


def regex_search(pattern: str, path: str = ".") -> str:
    """Search for a regex pattern in all text files under path."""
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"[Error] Invalid regex: {e}"

    results = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.startswith("."):
                continue
            file_path = os.path.join(root, f)
            try:
                with open(file_path, "r", encoding="utf-8") as fh:
                    for i, line in enumerate(fh, 1):
                        if regex.search(line):
                            results.append(f"{file_path}:{i}: {line.strip()}")
            except Exception:
                continue
    return "\n".join(results) if results else f"[Warning] No matches for '{pattern}'"


def relevant_files(query: str, k: int = 10) -> str:
    """Rank files by frequency of query tokens (heuristic relevance)."""
    tokens = query.lower().split()
    scores = Counter()

    for root, _, files in os.walk("."):
        for f in files:
            if f.startswith("."):
                continue
            file_path = os.path.join(root, f)
            try:
                with open(file_path, "r", encoding="utf-8") as fh:
                    text = fh.read().lower()
                score = sum(text.count(tok) for tok in tokens)
                if score > 0:
                    scores[file_path] = score
            except Exception:
                continue

    if not scores:
        return f"[Warning] No relevant files found for '{query}'"
    ranked = scores.most_common(k)
    return "\n".join(f"{f} (score={s})" for f, s in ranked)

# -------------------------
# DSPy Tool Wrappers
# -------------------------

search_code_tool = dspy.Tool(
    func=search_code,
    name="search_code",
    desc="Search code files for a substring and return matching lines with file paths.",
    args={
        "query": {"type": "string"},
        "path": {"type": "string", "default": "."},
    },
    arg_types={"query": str, "path": str},
    arg_desc={
        "query": "Substring to search for in code files.",
        "path": "Root directory to search (default: .).",
    },
)

regex_search_tool = dspy.Tool(
    func=regex_search,
    name="regex_search",
    desc="Regex search across codebase; returns file:line matches.",
    args={
        "pattern": {"type": "string"},
        "path": {"type": "string", "default": "."},
    },
    arg_types={"pattern": str, "path": str},
    arg_desc={
        "pattern": "Regex pattern to search for.",
        "path": "Root directory to search (default: .).",
    },
)

relevant_files_tool = dspy.Tool(
    func=relevant_files,
    name="relevant_files",
    desc="Heuristically rank files by relevance to a query.",
    args={
        "query": {"type": "string"},
        "k": {"type": "number", "default": 10},
    },
    arg_types={"query": str, "k": int},
    arg_desc={
        "query": "Query string to find relevant files.",
        "k": "Number of top files to return (default: 10).",
    },
)
