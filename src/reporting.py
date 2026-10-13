from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

from src.utils import atomic_write_json, load_config, read_jsonl


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": median(ordered),
        "mean": mean(ordered),
        "max": ordered[-1],
    }


def build_summary(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    output_dir = Path(config["paths"]["output_dir"])
    samples = list(read_jsonl(output_dir / "dataset.jsonl")) if (output_dir / "dataset.jsonl").exists() else []
    failures = list(read_jsonl(output_dir / "failures.jsonl")) if (output_dir / "failures.jsonl").exists() else []
    target = int(config["project"]["sample_count"])
    source_counts = Counter(sample["source"] for sample in samples)
    detail_counts = Counter(sum(bool(candidate.get("is_detail")) for candidate in sample["candidates"]) for sample in samples)
    raw_counts = [int(sample["raw_candidate_count"]) for sample in samples]
    filtered_counts = [int(sample["filtered_candidate_count"]) for sample in samples]
    filter_reasons: Counter[str] = Counter()
    for sample in samples:
        filter_reasons.update(sample.get("candidate_filter_reasons", {}))
    detail_areas = [float(candidate["area_ratio"]) for sample in samples for candidate in sample["candidates"] if candidate.get("is_detail")]
    semantic_scores = [float(candidate["semantic_score"]) for sample in samples for candidate in sample["candidates"] if candidate.get("semantic_score") is not None]
    counterfactual_scores_values = [float(candidate["counterfactual_score"]) for sample in samples for candidate in sample["candidates"] if candidate.get("counterfactual_score") is not None]
    token_counts: Counter[int] = Counter(label for sample in samples for label in sample["token_labels"])
    total_tokens = sum(token_counts.values())
    mapping_ok = sum(
        len(sample["token_labels"]) == int(sample["visual_token_count"])
        and len(sample["token_confidence"]) == int(sample["visual_token_count"])
        for sample in samples
    )
    timing_keys = ("sam_subject_seconds", "sam_candidates_seconds", "qwen_seconds")
    timings = {
        key: _distribution([float(sample.get("timing", {}).get(key, 0.0)) for sample in samples])
        for key in timing_keys
    }
    fallback_count = sum(bool(sample.get("subject_mask_fallback")) for sample in samples)
    visualization_count = len(list((output_dir / "visualizations").glob("*.png"))) if (output_dir / "visualizations").exists() else 0
    peak_memory = max((float(sample.get("metadata", {}).get("gpu_peak_memory_mib", 0.0)) for sample in samples), default=0.0)
    summary: dict[str, Any] = {
        "requested_samples": target,
        "successful_samples": len(samples),
        "failed_samples": target - len(samples),
        "success_rate": len(samples) / target if target else 0.0,
        "recorded_failures": len(failures),
        "failure_reasons": dict(sorted(Counter(f"{failure.get('stage')}:{failure.get('reason')}" for failure in failures).items())),
        "source_counts": dict(sorted(source_counts.items())),
        "raw_candidate_count": _distribution(raw_counts),
        "filtered_candidate_count": _distribution(filtered_counts),
        "candidate_filter_reasons": dict(sorted(filter_reasons.items())),
        "detail_mask_count_distribution": {str(count): detail_counts.get(count, 0) for count in range(4)},
        "detail_mask_count_proportion": {str(count): detail_counts.get(count, 0) / len(samples) if samples else 0.0 for count in range(4)},
        "subject_mask_fallback_count": fallback_count,
        "subject_mask_fallback_rate": fallback_count / len(samples) if samples else 0.0,
        "detail_mask_area_ratio": _distribution(detail_areas),
        "semantic_score": _distribution(semantic_scores),
        "counterfactual_score": _distribution(counterfactual_scores_values),
        "token_counts": {str(label): token_counts.get(label, 0) for label in (-1, 0, 1, 2)},
        "token_proportions": {str(label): token_counts.get(label, 0) / total_tokens if total_tokens else 0.0 for label in (-1, 0, 1, 2)},
        "total_visual_tokens": total_tokens,
        "token_mapping_length_checks_passed": mapping_ok,
        "token_mapping_length_checks_failed": len(samples) - mapping_ok,
        "timing_seconds": timings,
        "gpu_peak_memory_mib": peak_memory,
        "visualization_count": visualization_count,
        "overview_path": str((output_dir / "visualizations" / "overview.html").resolve()),
        "counterfactual_methods": dict(sorted(Counter(sample.get("metadata", {}).get("counterfactual_method", "unknown") for sample in samples).items())),
    }
    atomic_write_json(output_dir / "summary.json", summary)
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "value"))
        for key, value in summary.items():
            writer.writerow((key, json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build JSON and CSV aggregate reports")
    parser.add_argument("--config", default="configs/pilot_200.yaml")
    args = parser.parse_args()
    print(json.dumps(build_summary(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

