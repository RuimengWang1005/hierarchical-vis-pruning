#!/usr/bin/env python3
"""Print a metric-by-metric GAZE versus Mask-VIS comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def newest_result(root: Path, condition: str, budget: str) -> Path:
    candidates = sorted((root / condition / f"budget_{budget}").rglob("*_results.json"))
    if not candidates:
        raise FileNotFoundError(f"No result JSON: condition={condition}, budget={budget}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def numeric_metrics(payload: dict[str, Any]) -> dict[tuple[str, str], float]:
    rows: dict[tuple[str, str], float] = {}
    for task, metrics in payload.get("results", {}).items():
        if not isinstance(metrics, dict):
            continue
        for metric, value in metrics.items():
            if metric == "alias" or "stderr" in metric or not isinstance(value, (int, float)):
                continue
            rows[(task, metric)] = float(value)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/mask_vis_20k_online/general_eval"),
    )
    parser.add_argument("--budgets", nargs="+", default=["0.1", "0.2", "0.3"])
    args = parser.parse_args()

    print("| Budget | Task | Metric | GAZE | Mask-VIS | Mask-GAZE |")
    print("|---:|---|---|---:|---:|---:|")
    for budget in args.budgets:
        gaze_path = newest_result(args.root, "gaze", budget)
        mask_path = newest_result(args.root, "mask", budget)
        gaze = numeric_metrics(json.loads(gaze_path.read_text(encoding="utf-8")))
        mask = numeric_metrics(json.loads(mask_path.read_text(encoding="utf-8")))
        for task_metric in sorted(gaze.keys() & mask.keys()):
            task, metric = task_metric
            gaze_value = gaze[task_metric]
            mask_value = mask[task_metric]
            print(
                f"| {budget} | {task} | {metric} | {gaze_value:.6f} | "
                f"{mask_value:.6f} | {mask_value - gaze_value:+.6f} |"
            )


if __name__ == "__main__":
    main()

