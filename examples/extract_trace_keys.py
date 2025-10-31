from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Iterable


def _merge_sets(items: Iterable[set[str]]) -> set[str]:
    s: set[str] = set()
    for it in items:
        s.update(it)
    return s


def extract_key_paths(obj: Any, prefix: str = "", sample_list_items: int = 50, with_types: bool = False) -> set[str]:
    def add(path: str, val: Any) -> str:
        return f"{path} :: {type(val).__name__}" if with_types else path

    if isinstance(obj, dict):
        if not obj:
            return {add(prefix or "$", obj)}
        paths: set[str] = set()
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            paths.update(extract_key_paths(v, p, sample_list_items, with_types))
        return paths

    if isinstance(obj, list):
        if not obj:
            return {add(f"{prefix}[*]", obj)}
        head = add(f"{prefix}[*]", obj)
        sampled = obj[: sample_list_items if sample_list_items > 0 else None]
        return {head} | _merge_sets(
            extract_key_paths(v, f"{prefix}[*]", sample_list_items, with_types) for v in sampled
        )

    return {add(prefix or "$", obj)}


def main(input_path: str, output_path: str | None = None, with_types: bool = False) -> Path:
    src = Path(input_path)
    data = json.loads(src.read_text())
    keys = sorted(extract_key_paths(data, with_types=with_types))

    out = Path(output_path) if output_path else src.with_suffix("")
    if out.is_dir() or output_path is None:
        out = out.parent / f"{src.stem}.keys.txt"

    out.write_text("\n".join(keys))
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract dot-path keys from a JSON trace")
    parser.add_argument("input", help="Path to JSON trace (e.g., examples/output/mlflow_trace_<id>.json)")
    parser.add_argument("--output", help="Optional output path (.txt)")
    parser.add_argument("--with-types", action="store_true", help="Include Python value types next to keys")
    args = parser.parse_args()

    outp = main(args.input, args.output, args.with_types)
    print(f"Saved keys to: {outp}")


