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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if detail_coverages:
        detail = np.maximum.reduce(detail_coverages)
    else:
        detail = np.zeros_like(subject_coverage, dtype=np.float32)
    labels = np.full(subject_coverage.shape, int(label_config["ignore_label"]), dtype=np.int8)
    confidence = np.zeros(subject_coverage.shape, dtype=np.float32)

    background = subject_coverage <= float(label_config["background_max_subject_coverage"])
    foreground = (subject_coverage >= float(label_config["foreground_min_coverage"])) & (
        detail < float(label_config["conflict_min_coverage"])
    )
    detail_tokens = detail >= float(label_config["detail_min_coverage"])

    labels[background] = int(label_config["background_label"])
    confidence[background] = 1.0 - subject_coverage[background]
    labels[foreground] = int(label_config["foreground_label"])
    confidence[foreground] = subject_coverage[foreground] * (1.0 - detail[foreground])
    labels[detail_tokens] = int(label_config["detail_label"])
    confidence[detail_tokens] = detail[detail_tokens]

    ignored = labels == int(label_config["ignore_label"])
    confidence[ignored] = np.maximum.reduce(
        [subject_coverage[ignored], 1.0 - subject_coverage[ignored], detail[ignored]]
    ) if ignored.any() else confidence[ignored]
    return labels, np.clip(confidence, 0.0, 1.0), detail


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
    labels, confidence, detail = assign_token_labels(subject_coverage, detail_coverages, config["token_labels"])
    expected = int(sample["visual_token_count"])
    if len(labels) != expected:
        raise AssertionError(f"{sample['sample_id']}: token_labels length {len(labels)} != Qwen visual tokens {expected}")
    result = dict(sample)
    result.update(
        {
            "token_labels": labels.astype(int).tolist(),
            "token_confidence": confidence.astype(float).tolist(),
            "subject_token_coverage": subject_coverage.astype(float).tolist(),
            "detail_token_coverage": detail.astype(float).tolist(),
            "metadata": {
                **sample.get("metadata", {}),
                "config_hash": config_hash(config),
                "token_mapping": "qwen_resize_then_area_pool_over_merged_patch_cells",
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
    for index, sample in enumerate(read_jsonl(output_dir / "teacher_scores.jsonl"), start=1):
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

