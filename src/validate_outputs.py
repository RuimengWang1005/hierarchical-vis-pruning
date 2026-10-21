from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.masks import decode_rle
from src.utils import atomic_write_json, load_config, read_jsonl


def validate_outputs(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    output_dir = Path(config["paths"]["output_dir"])
    records = list(read_jsonl(output_dir / "dataset.jsonl"))
    expected = int(config["project"]["sample_count"])
    errors: list[str] = []
    sample_ids = [record.get("sample_id") for record in records]
    if len(records) != expected:
        errors.append(f"dataset cardinality {len(records)} != {expected}")
    if len(set(sample_ids)) != len(sample_ids):
        errors.append("sample_id values are not unique")

    label_counts: Counter[int] = Counter()
    require_soft_targets = bool(config["token_labels"].get("require_soft_targets", False))
    for record in records:
        sample_id = record["sample_id"]
        token_count = int(record["visual_token_count"])
        labels = record["token_labels"]
        confidence = record["token_confidence"]
        if len(labels) != token_count or len(confidence) != token_count:
            errors.append(f"{sample_id}: token length mismatch")
        if not set(labels).issubset({-1, 0, 1, 2}):
            errors.append(f"{sample_id}: unexpected token label")
        if any(not 0.0 <= float(value) <= 1.0 for value in confidence):
            errors.append(f"{sample_id}: confidence outside [0, 1]")
        soft_targets = record.get("token_soft_targets")
        if require_soft_targets and soft_targets is None:
            errors.append(f"{sample_id}: missing token_soft_targets")
        if soft_targets is not None:
            if len(soft_targets) != token_count:
                errors.append(f"{sample_id}: soft-target length mismatch")
            for token_index, target in enumerate(soft_targets):
                if len(target) != 3:
                    errors.append(f"{sample_id}: soft target {token_index} is not length 3")
                    break
                values = [float(value) for value in target]
                if any(value < 0.0 or value > 1.0 for value in values) or abs(sum(values) - 1.0) > 1e-5:
                    errors.append(f"{sample_id}: invalid soft target {token_index}")
                    break
            if record.get("token_soft_target_class_order") != ["background", "foreground", "detail"]:
                errors.append(f"{sample_id}: unexpected soft-target class order")
        if require_soft_targets and not any(label in (1, 2) for label in labels):
            errors.append(f"{sample_id}: no positive hard token")
        selected_details = sum(bool(candidate.get("is_detail")) for candidate in record["candidates"])
        if require_soft_targets and selected_details and 2 not in labels:
            errors.append(f"{sample_id}: selected detail masks but no detail hard token")
        detail_positive_indices = record.get("token_label_assignment", {}).get("detail_mask_positive_indices", [])
        if require_soft_targets and len(detail_positive_indices) != selected_details:
            errors.append(f"{sample_id}: not every selected detail mask has a positive token")
        if any(index < 0 or index >= token_count or labels[index] != 2 for index in detail_positive_indices):
            errors.append(f"{sample_id}: invalid detail-mask positive-token index")
        label_counts.update(labels)
        candidates = record["candidates"]
        if len(candidates) > int(config["sam"]["max_filtered_candidates"]):
            errors.append(f"{sample_id}: too many filtered candidates")
        details = sum(bool(candidate.get("is_detail")) for candidate in candidates)
        if details > int(config["teacher_score"]["max_detail_masks"]):
            errors.append(f"{sample_id}: too many detail masks")
        if sum(bool(candidate.get("selected_for_counterfactual")) for candidate in candidates) > int(config["teacher_score"]["semantic_top_k"]):
            errors.append(f"{sample_id}: too many counterfactual candidates")
        expected_shape = (int(record["height"]), int(record["width"]))
        if decode_rle(record["subject_mask_rle"]).shape != expected_shape:
            errors.append(f"{sample_id}: subject RLE shape mismatch")
        for candidate in candidates:
            if decode_rle(candidate["mask_rle"]).shape != expected_shape:
                errors.append(f"{sample_id}/{candidate['candidate_id']}: candidate RLE shape mismatch")

    visualizations = list((output_dir / "visualizations").glob("*.png"))
    if len(visualizations) != len(records):
        errors.append(f"visualization count {len(visualizations)} != dataset count {len(records)}")
    required_outputs = ["dataset.jsonl", "summary.json", "summary.csv", "visualizations/overview.html"]
    if "token_source_jsonl" not in config["paths"]:
        required_outputs.extend(("manifest.jsonl", "failures.jsonl"))
    for required in required_outputs:
        if not (output_dir / required).exists():
            errors.append(f"missing required output: {required}")

    report = {
        "strict_ok": not errors,
        "errors": errors,
        "dataset_records": len(records),
        "unique_sample_ids": len(set(sample_ids)),
        "visualizations": len(visualizations),
        "token_label_counts": {str(label): label_counts.get(label, 0) for label in (-1, 0, 1, 2)},
        "all_token_lengths_match": not any("token length mismatch" in error for error in errors),
        "all_rle_shapes_match": not any("RLE shape mismatch" in error for error in errors),
    }
    atomic_write_json(output_dir / "validation_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Strictly validate all final pilot outputs")
    parser.add_argument("--config", default="configs/pilot_200.yaml")
    args = parser.parse_args()
    report = validate_outputs(args.config)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["strict_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

