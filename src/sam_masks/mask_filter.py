from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from src.masks import decode_rle, mask_iou
from src.utils import append_jsonl, load_config, read_jsonl


def filter_candidates(candidates: list[dict[str, Any]], sam_config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rejected: Counter[str] = Counter()
    eligible: list[dict[str, Any]] = []
    for original in candidates:
        candidate = dict(original)
        reasons: list[str] = []
        if candidate["inside_ratio"] < float(sam_config["min_inside_ratio"]):
            reasons.append("outside_subject")
        if candidate["area_ratio"] < float(sam_config["min_area_ratio"]):
            reasons.append("too_small")
        if candidate["area_ratio"] > float(sam_config["max_area_ratio"]):
            reasons.append("too_large")
        if candidate["sam_pred_iou"] < float(sam_config["min_pred_iou"]):
            reasons.append("low_pred_iou")
        if candidate["stability"] < float(sam_config["min_stability"]):
            reasons.append("low_stability")
        if candidate["connected_components"] > int(sam_config["max_connected_components"]):
            reasons.append("fragmented")
        candidate["filter_reasons"] = reasons
        if reasons:
            rejected.update(reasons)
        else:
            eligible.append(candidate)

    eligible.sort(key=lambda item: (-float(item["sam_quality"]), item["candidate_id"]))
    retained: list[dict[str, Any]] = []
    decoded: list[Any] = []
    for candidate in eligible:
        candidate_mask = decode_rle(candidate["mask_rle"])
        duplicate = False
        for kept, kept_mask in zip(retained, decoded):
            overlap = mask_iou(candidate_mask, kept_mask)
            area_similarity = min(candidate["area"], kept["area"]) / max(candidate["area"], kept["area"])
            if overlap >= float(sam_config["duplicate_iou"]) and area_similarity >= float(sam_config["duplicate_area_similarity"]):
                duplicate = True
                break
        if duplicate:
            rejected["duplicate"] += 1
            continue
        retained.append(candidate)
        decoded.append(candidate_mask)
        if len(retained) >= int(sam_config["max_filtered_candidates"]):
            break
    if len(eligible) > len(retained):
        overflow = max(0, len(eligible) - len(retained) - rejected["duplicate"])
        rejected["candidate_cap"] += overflow
    return retained, dict(sorted(rejected.items()))


def run_filter_stage(config_path: str | Path, *, overwrite: bool = False) -> None:
    config = load_config(config_path)
    output_dir = Path(config["paths"]["output_dir"])
    stage_path = output_dir / "filtered_candidates.jsonl"
    if overwrite:
        stage_path.unlink(missing_ok=True)
    completed = {row["sample_id"] for row in read_jsonl(stage_path)} if stage_path.exists() else set()
    for index, sample in enumerate(read_jsonl(output_dir / "raw_candidates.jsonl"), start=1):
        if sample["sample_id"] in completed:
            continue
        filtered, reasons = filter_candidates(sample["raw_candidates"], config["sam"])
        record = {key: value for key, value in sample.items() if key != "raw_candidates"}
        record.update(
            {
                "filtered_candidate_count": len(filtered),
                "candidate_filter_reasons": reasons,
                "candidates": filtered,
            }
        )
        append_jsonl(stage_path, record)
        print(f"[filter {index:03d}] {sample['sample_id']} {sample['raw_candidate_count']} -> {len(filtered)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter and deduplicate local SAM candidates")
    parser.add_argument("--config", default="configs/pilot_200.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_filter_stage(args.config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

