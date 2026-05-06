"""Iterate results/<tag>/*/metrics.json one file at a time and pretty-print
selected keys. Never concatenates JSON.

Usage:
    python -m finite_response_sail.inspect_results <tag>
        [--results-dir DIR]
        [--split {train,val,test}]
        [--keys key1 key2 ...]
"""
import argparse
import json
from pathlib import Path
from typing import Iterable, List


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.5f}"
    return json.dumps(v)


def _print_run(metrics_path: Path, split: str, keys: List[str]) -> None:
    data = json.loads(metrics_path.read_text())
    run_id = data.get("run_id", metrics_path.parent.name)
    block = data.get("final", {}).get(split, {})
    print(f"=== {run_id}  (split={split}) ===")
    if not keys:
        for k in sorted(block.keys()):
            print(f"  {k}: {_fmt(block[k])}")
    else:
        for k in keys:
            if k in block:
                print(f"  {k}: {_fmt(block[k])}")
            else:
                print(f"  {k}: <missing>")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tag")
    parser.add_argument("--results-dir", default="finite_response_sail/results")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--keys", nargs="*", default=[])
    args = parser.parse_args()

    tag_dir = Path(args.results_dir).resolve() / args.tag
    if not tag_dir.is_dir():
        raise SystemExit(f"[error] no results dir at {tag_dir}")
    metric_files = sorted(tag_dir.glob("*/metrics.json"))
    if not metric_files:
        raise SystemExit(f"[error] no metrics.json under {tag_dir}")
    for p in metric_files:
        _print_run(p, args.split, args.keys)


if __name__ == "__main__":
    main()
