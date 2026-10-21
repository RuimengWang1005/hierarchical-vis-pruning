from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.masks import decode_rle
from src.utils import append_jsonl, config_hash, load_config, read_jsonl


def token_grid_shape(image_grid_thw: list[int] | tuple[int, int, int], spatial_merge_size: int) -> tuple[int, int, int]:
    temporal, patch_height, patch_width = [int(value) for value in image_grid_thw]
    if temporal <= 0 or patch_height <= 0 or patch_width <= 0:
        raise ValueError(f"Invalid image_grid_thw: {image_grid_thw}")
    if patch_height % spatial_merge_size or patch_width % spatial_merge_size:
        raise ValueError(f"Grid {image_grid_thw} is not divisible by merge size {spatial_merge_size}")
    return temporal, patch_height // spatial_merge_size, patch_width // spatial_merge_size


def mask_to_token_coverage(
    mask: np.ndarray,
    image_grid_thw: list[int] | tuple[int, int, int],
    *,
    patch_size: int,
    spatial_merge_size: int,
) -> np.ndarray:
    """Map an original-pixel mask through Qwen's exact resize and spatial merge grid."""
    temporal, token_height, token_width = token_grid_shape(image_grid_thw, spatial_merge_size)
    patch_height = int(image_grid_thw[1])
    patch_width = int(image_grid_thw[2])
    resized_height = patch_height * patch_size
    resized_width = patch_width * patch_size
    resized = cv2.resize(mask.astype(np.float32), (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    cell = patch_size * spatial_merge_size
    coverage_2d = resized.reshape(token_height, cell, token_width, cell).mean(axis=(1, 3))
    coverage = np.tile(coverage_2d.reshape(-1), temporal)
    expected = temporal * token_height * token_width
    if coverage.size != expected:
        raise AssertionError(f"Coverage length {coverage.size} != expected visual tokens {expected}")
    return np.clip(coverage.astype(np.float32), 0.0, 1.0)


def assign_token_labels(
    subject_coverage: np.ndarray,
    detail_coverages: list[np.ndarray],
    label_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Create hard compatibility labels and loss-ready three-class soft targets.

    Soft-target columns are ordered as background, ordinary foreground, detail.
    Hard labels deliberately use permissive evidence thresholds and deterministic
    positive fallbacks; they are intended for visualization, evaluation, and
    training code that cannot yet consume soft targets.
    """
    subject = np.clip(np.asarray(subject_coverage, dtype=np.float32), 0.0, 1.0)
    if detail_coverages:
        raw_details = [np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0) for value in detail_coverages]
        if any(value.shape != subject.shape for value in raw_details):
            raise ValueError("Every detail coverage array must match subject_coverage")
        detail = np.maximum.reduce(raw_details)
    else:
        raw_details = []
        detail = np.zeros_like(subject, dtype=np.float32)

    # Detail is a subclass of subject. Clipping suppresses the small amount of
    # candidate-mask leakage that can remain after SAM filtering/resizing.
    detail = np.minimum(detail, subject)
    soft_targets = np.stack(
        (1.0 - subject, np.maximum(subject - detail, 0.0), detail),
        axis=-1,
    ).astype(np.float32)

    labels = np.full(subject.shape, int(label_config["ignore_label"]), dtype=np.int8)

    background = subject <= float(label_config["background_max_subject_coverage"])
    foreground = subject >= float(label_config["foreground_min_coverage"])
    detail_tokens = detail >= float(label_config["detail_min_coverage"])

    labels[background] = int(label_config["background_label"])
    labels[foreground] = int(label_config["foreground_label"])
    labels[detail_tokens] = int(label_config["detail_label"])

    diagnostics: dict[str, Any] = {
        "subject_positive_fallback": False,
        "detail_positive_fallback_count": 0,
        "detail_mask_positive_indices": [],
    }

    if bool(label_config.get("ensure_each_detail_positive", True)):
        for raw_detail in raw_details:
            evidence = np.minimum(raw_detail, subject)
            if not evidence.size or float(evidence.max()) <= 0.0:
                continue
            represented = np.flatnonzero(
                (labels == int(label_config["detail_label"])) & (evidence > 0.0)
            )
            if represented.size:
                token_index = int(represented[np.argmax(evidence[represented])])
            else:
                token_index = int(np.argmax(evidence))
                labels[token_index] = int(label_config["detail_label"])
                diagnostics["detail_positive_fallback_count"] += 1
            diagnostics["detail_mask_positive_indices"].append(token_index)

    positive = (labels == int(label_config["foreground_label"])) | (labels == int(label_config["detail_label"]))
    if bool(label_config.get("ensure_subject_positive", True)) and subject.size and float(subject.max()) > 0.0 and not positive.any():
        labels[int(np.argmax(subject))] = int(label_config["foreground_label"])
        diagnostics["subject_positive_fallback"] = True

    class_to_column = {
        int(label_config["background_label"]): 0,
        int(label_config["foreground_label"]): 1,
        int(label_config["detail_label"]): 2,
    }
    confidence = soft_targets.max(axis=1)
    for label, column in class_to_column.items():
        selected = labels == label
        confidence[selected] = soft_targets[selected, column]
    ignored = labels == int(label_config["ignore_label"])
    if ignored.any():
        confidence[ignored] = soft_targets[ignored].max(axis=1)
    return labels, np.clip(confidence, 0.0, 1.0), detail, soft_targets, diagnostics


def label_sample(sample: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    qwen_config = config["qwen"]
    grid = sample["image_grid_thw"]
    subject = decode_rle(sample["subject_mask_rle"])
    subject_coverage = mask_to_token_coverage(
        subject,
        grid,
        patch_size=int(qwen_config["patch_size"]),
        spatial_merge_size=int(qwen_config["spatial_merge_size"]),
    )
    detail_coverages = [
        mask_to_token_coverage(
            decode_rle(candidate["mask_rle"]),
            grid,
            patch_size=int(qwen_config["patch_size"]),
            spatial_merge_size=int(qwen_config["spatial_merge_size"]),
        )
        for candidate in sample["candidates"]
        if candidate.get("is_detail")
    ]
    labels, confidence, detail, soft_targets, assignment = assign_token_labels(
        subject_coverage, detail_coverages, config["token_labels"]
    )
    expected = int(sample["visual_token_count"])
    if len(labels) != expected:
        raise AssertionError(f"{sample['sample_id']}: token_labels length {len(labels)} != Qwen visual tokens {expected}")
    result = dict(sample)
    result.update(
        {
            "token_labels": labels.astype(int).tolist(),
            "token_confidence": confidence.astype(float).tolist(),
            "token_soft_targets": soft_targets.astype(float).tolist(),
            "token_soft_target_class_order": ["background", "foreground", "detail"],
            "subject_token_coverage": subject_coverage.astype(float).tolist(),
            "detail_token_coverage": detail.astype(float).tolist(),
            "token_label_assignment": assignment,
            "metadata": {
                **sample.get("metadata", {}),
                "upstream_config_hash": sample.get("metadata", {}).get("config_hash"),
                "config_hash": config_hash(config),
                "token_mapping": "qwen_resize_then_area_pool_over_merged_patch_cells",
                "token_label_schema": "v2_soft_targets_with_positive_hard_fallbacks",
                "token_mapping_length_ok": True,
            },
        }
    )
    return result


def run_token_label_stage(config_path: str | Path, *, overwrite: bool = False) -> None:
    config = load_config(config_path)
    output_dir = Path(config["paths"]["output_dir"])
    stage_path = output_dir / "dataset.jsonl"
    if overwrite:
        stage_path.unlink(missing_ok=True)
    completed = {row["sample_id"] for row in read_jsonl(stage_path)} if stage_path.exists() else set()
    source_path = Path(config["paths"].get("token_source_jsonl", output_dir / "teacher_scores.jsonl"))
    if not source_path.exists():
        raise FileNotFoundError(f"Token-label input does not exist: {source_path}")
    for index, sample in enumerate(read_jsonl(source_path), start=1):
        if sample["sample_id"] in completed:
            continue
        result = label_sample(sample, config)
        append_jsonl(stage_path, result)
        print(f"[tokens {index:03d}] {sample['sample_id']} n={len(result['token_labels'])}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Map subject/detail masks to Qwen visual-token labels")
    parser.add_argument("--config", default="configs/pilot_200.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_token_label_stage(args.config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

