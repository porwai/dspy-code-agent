import argparse
import json
import sys
from pathlib import Path
from subprocess import run, PIPE


def _ensure_jsonl(preds_path: Path) -> Path:
    """If given a JSON mapping {instance_id: obj}, convert to JSONL and return new path.
    If already .jsonl, return as-is.
    """
    if preds_path.suffix.lower() == ".jsonl":
        return preds_path
    try:
        data = json.loads(preds_path.read_text())
    except Exception as e:
        raise SystemExit(f"Failed to read predictions JSON at {preds_path}: {e}")
    if not isinstance(data, dict):
        raise SystemExit("Predictions JSON must be an object mapping instance_id -> record")
    out_path = preds_path.with_suffix(".jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for _, obj in data.items():
            f.write(json.dumps(obj) + "\n")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate SWE-bench predictions via Python module entrypoint")
    parser.add_argument("--preds", default="outputs/swebench_dspy/run_manual/preds.json", help="Path to preds.json or preds.jsonl")
    parser.add_argument("--dataset", default="princeton-nlp/SWE-Bench_Lite", help="Dataset name for evaluation")
    parser.add_argument("--max-workers", type=int, default=1, dest="max_workers", help="Parallel workers")
    parser.add_argument("--run-id", default="local_test", dest="run_id", help="Run identifier")
    args = parser.parse_args()

    preds = Path(args.preds).resolve()
    if not preds.exists():
        raise SystemExit(f"Missing predictions file at {preds}")

    preds_jsonl = _ensure_jsonl(preds)

    cmd = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        args.dataset,
        "--predictions_path",
        str(preds_jsonl),
        "--max_workers",
        str(args.max_workers),
        "--run_id",
        args.run_id,
    ]

    result = run(cmd, stdout=PIPE, stderr=PIPE, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        if "No module named swebench.harness.run_evaluation" in result.stderr or "No module named swebench" in result.stderr:
            print("Hint: pip install swebench", file=sys.stderr)
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()