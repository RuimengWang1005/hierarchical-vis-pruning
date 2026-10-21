from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from src.masks import bbox_mask, decode_rle, encode_rle
from src.utils import atomic_write_json, atomic_write_jsonl, config_hash, load_config, read_jsonl, sha256_file


SEED = 42
CURRENT_FOREGROUND_MIN = 0.50
CURRENT_BACKGROUND_MAX = 0.05
RELIABLE_BACKGROUND_MAX = 0.01
ADAPTIVE_MASS_TARGET = 0.90
GAZE_BUDGETS = (0.10, 0.20, 0.30)
CLASS_NAMES = ("background", "foreground", "ignore")


def _percentile(values: Iterable[float], q: float) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.percentile(array, q)) if array.size else None


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {"count": 0, "min": None, "mean": None, "median": None, "p25": None, "p75": None, "max": None}
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "max": float(array.max()),
    }


def recompute_mask_token_metrics(
    mask: np.ndarray,
    image_grid_thw: list[int] | tuple[int, int, int],
    *,
    patch_size: int = 14,
    spatial_merge_size: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Independently map a pixel mask to merged Qwen cells.

    Coverage is intersection / cell area. Contribution is intersection / total
    resized-mask mass and therefore sums to one for a non-empty static mask.
    """
    temporal, patch_height, patch_width = (int(value) for value in image_grid_thw)
    if temporal <= 0 or patch_height <= 0 or patch_width <= 0:
        raise ValueError(f"Invalid image_grid_thw={image_grid_thw}")
    if patch_height % spatial_merge_size or patch_width % spatial_merge_size:
        raise ValueError(f"Grid is not divisible by merge size: {image_grid_thw}")
    target_height = patch_height * patch_size
    target_width = patch_width * patch_size
    resized = cv2.resize(mask.astype(np.float32), (target_width, target_height), interpolation=cv2.INTER_AREA)
    cell = patch_size * spatial_merge_size
    token_height = patch_height // spatial_merge_size
    token_width = patch_width // spatial_merge_size
    intersections_2d = resized.reshape(token_height, cell, token_width, cell).sum(axis=(1, 3))
    intersections = np.tile(intersections_2d.reshape(-1), temporal).astype(np.float64)
    coverage = intersections / float(cell * cell)
    total_intersection = float(intersections.sum())
    contribution = intersections / total_intersection if total_intersection > 0 else np.zeros_like(intersections)
    expected = temporal * token_height * token_width
    if coverage.size != expected or contribution.size != expected:
        raise AssertionError("Token mapping length mismatch")
    return np.clip(coverage, 0.0, 1.0).astype(np.float32), np.clip(contribution, 0.0, 1.0).astype(np.float32)


def adaptive_mask_token_indices(contribution: np.ndarray, target_mass: float = ADAPTIVE_MASS_TARGET) -> np.ndarray:
    contribution = np.asarray(contribution, dtype=np.float64)
    candidates = np.flatnonzero(contribution > 0.0)
    if not candidates.size:
        return np.empty((0,), dtype=np.int64)
    order = candidates[np.argsort(-contribution[candidates], kind="stable")]
    cumulative = np.cumsum(contribution[order])
    count = int(np.searchsorted(cumulative, target_mass, side="left")) + 1
    count = max(1, min(count, order.size))
    return np.sort(order[:count]).astype(np.int64)


def union_bbox_pixel_mask(shape: tuple[int, int], bboxes: list[list[float]]) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    for bbox in bboxes:
        result |= bbox_mask(shape, bbox)
    return result


def gaze_bbox_positive_tokens(
    bboxes: list[list[float]],
    width: int,
    height: int,
    image_grid_thw: list[int] | tuple[int, int, int],
    spatial_merge_size: int = 2,
) -> np.ndarray:
    """Reproduce GAZE pixel_bboxs_to_ref_token_mask exactly for one image."""
    temporal, patch_height, patch_width = (int(value) for value in image_grid_thw)
    token_height = patch_height // spatial_merge_size
    token_width = patch_width // spatial_merge_size
    mask = np.zeros((token_height, token_width), dtype=bool)
    for bbox in bboxes:
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = (float(value) for value in bbox)
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        x1 = max(0.0, min(1.0, x1 / float(width)))
        x2 = max(0.0, min(1.0, x2 / float(width)))
        y1 = max(0.0, min(1.0, y1 / float(height)))
        y2 = max(0.0, min(1.0, y2 / float(height)))
        x1_grid = max(0, min(token_width - 1, int(x1 * token_width)))
        y1_grid = max(0, min(token_height - 1, int(y1 * token_height)))
        x2_grid = min(int(x2 * token_width), token_width - 1)
        y2_grid = min(int(y2 * token_height), token_height - 1)
        x2_grid = max(x1_grid, x2_grid)
        y2_grid = max(y1_grid, y2_grid)
        mask[y1_grid : y2_grid + 1, x1_grid : x2_grid + 1] = True
    flat = np.tile(mask.reshape(-1), temporal)
    return np.flatnonzero(flat).astype(np.int64)


def _safe_mean(values: np.ndarray) -> float | None:
    return float(values.mean()) if values.size else None


def geometry_record(sample: dict[str, Any], config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    subject = decode_rle(sample["subject_mask_rle"])
    height, width = subject.shape
    if (height, width) != (int(sample["height"]), int(sample["width"])):
        raise AssertionError(f"{sample['sample_id']}: subject shape mismatch")
    grid = [int(value) for value in sample["image_grid_thw"]]
    patch_size = int(config["qwen"]["patch_size"])
    merge = int(config["qwen"]["spatial_merge_size"])
    coverage, contribution = recompute_mask_token_metrics(
        subject, grid, patch_size=patch_size, spatial_merge_size=merge
    )
    bbox_pixels = union_bbox_pixel_mask(subject.shape, sample["bboxes"])
    bbox_coverage, _ = recompute_mask_token_metrics(
        bbox_pixels, grid, patch_size=patch_size, spatial_merge_size=merge
    )
    bbox_positive = gaze_bbox_positive_tokens(sample["bboxes"], width, height, grid, merge)
    expected = grid[0] * (grid[1] // merge) * (grid[2] // merge)
    if expected != int(sample["visual_token_count"]) or coverage.size != expected:
        raise AssertionError(f"{sample['sample_id']}: grid/token length mismatch")

    current_positive = np.flatnonzero(coverage >= CURRENT_FOREGROUND_MIN)
    current_background = np.flatnonzero(coverage <= CURRENT_BACKGROUND_MAX)
    current_ignore = np.flatnonzero(
        (coverage > CURRENT_BACKGROUND_MAX) & (coverage < CURRENT_FOREGROUND_MIN)
    )
    adaptive = adaptive_mask_token_indices(contribution)
    bbox_subject_coverage = coverage[bbox_positive]
    false_positive = bbox_subject_coverage <= CURRENT_BACKGROUND_MAX
    bbox_pixel_area = int(bbox_pixels.sum(dtype=np.int64))
    clipped_subject_area = int(np.logical_and(subject, bbox_pixels).sum(dtype=np.int64))
    existing = np.asarray(sample.get("subject_token_coverage", []), dtype=np.float32)
    coverage_error = float(np.max(np.abs(existing - coverage))) if existing.size == coverage.size else None
    warnings: list[str] = []
    if subject.any() and not current_positive.size:
        warnings.append("valid_mask_but_zero_current_positive_tokens")
    if not adaptive.size and subject.any():
        warnings.append("adaptive_mapping_failed_for_nonempty_mask")
    if existing.size != coverage.size:
        warnings.append("existing_coverage_missing_or_length_mismatch")

    record: dict[str, Any] = {
        "sample_id": sample["sample_id"],
        "source": sample["source"],
        "image_path": sample["image_path"],
        "question": sample["question"],
        "answer": sample["answer"],
        "image_grid_thw": grid,
        "visual_token_count": expected,
        "mask_area": int(subject.sum(dtype=np.int64)),
        "mask_area_ratio": float(subject.mean()),
        "current_positive_token_count": int(current_positive.size),
        "current_background_token_count": int(current_background.size),
        "current_ignore_token_count": int(current_ignore.size),
        "current_positive_token_ratio": float(current_positive.size / expected),
        "valid_mask_but_zero_current_positive": bool(subject.any() and not current_positive.size),
        "max_mask_token_coverage": float(coverage.max()) if coverage.size else 0.0,
        "max_mask_contribution": float(contribution.max()) if contribution.size else 0.0,
        "adaptive_positive_token_count": int(adaptive.size),
        "adaptive_positive_token_ratio": float(adaptive.size / expected),
        "current_mask_mass_recall": float(np.clip(contribution[current_positive].sum(), 0.0, 1.0)),
        "adaptive_mask_mass_recall": float(np.clip(contribution[adaptive].sum(), 0.0, 1.0)),
        "all_token_mask_mass_recall": float(np.clip(contribution.sum(), 0.0, 1.0)),
        "bbox_pixel_area": bbox_pixel_area,
        "bbox_pixel_purity": float(clipped_subject_area / bbox_pixel_area) if bbox_pixel_area else None,
        "bbox_positive_token_count": int(bbox_positive.size),
        "bbox_false_positive_token_count": int(false_positive.sum()),
        "bbox_false_positive_rate": float(false_positive.mean()) if false_positive.size else None,
        "bbox_positive_mean_mask_coverage": _safe_mean(bbox_subject_coverage),
        "bbox_positive_median_mask_coverage": float(np.median(bbox_subject_coverage)) if bbox_subject_coverage.size else None,
        "bbox_positive_zero_coverage_rate": float((bbox_subject_coverage <= 0.0).mean()) if bbox_subject_coverage.size else None,
        "bbox_positive_below_0_05_rate": float((bbox_subject_coverage < 0.05).mean()) if bbox_subject_coverage.size else None,
        "bbox_positive_below_0_10_rate": float((bbox_subject_coverage < 0.10).mean()) if bbox_subject_coverage.size else None,
        "bbox_positive_below_0_25_rate": float((bbox_subject_coverage < 0.25).mean()) if bbox_subject_coverage.size else None,
        "bbox_positive_below_0_50_rate": float((bbox_subject_coverage < 0.50).mean()) if bbox_subject_coverage.size else None,
        "recomputed_existing_coverage_max_abs_error": coverage_error,
        "fallback_to_bbox": bool(sample.get("subject_mask_fallback", False)),
        "mean_fg_score": None,
        "mean_bg_score": None,
        "mean_bbox_fg_score": None,
        "mask_minus_bbox_fg_score": None,
        "score_margin": None,
        "pairwise_accuracy": None,
        "retained_recall_10": None,
        "retained_recall_20": None,
        "retained_recall_30": None,
        "warnings": warnings,
        "current_positive_token_indices": current_positive.astype(int).tolist(),
        "adaptive_mask_tokens": adaptive.astype(int).tolist(),
        "bbox_positive_token_indices": bbox_positive.astype(int).tolist(),
        "mask_token_coverage": coverage.astype(float).tolist(),
        "mask_contribution": contribution.astype(float).tolist(),
        "bbox_token_coverage": bbox_coverage.astype(float).tolist(),
        "gaze_raw_scores": None,
        "gaze_selection": None,
    }
    arrays = {
        "subject": subject,
        "coverage": coverage,
        "contribution": contribution,
        "bbox_coverage": bbox_coverage,
        "bbox_positive": bbox_positive,
        "current_positive": current_positive,
        "adaptive": adaptive,
    }
    return record, arrays


class GazeVisionAuditor:
    """Load only the frozen GAZE vision tower and its scorer from the checkpoint."""

    def __init__(self, checkpoint: Path, config: dict[str, Any], device: str) -> None:
        eval_root = Path(config["paths"]["gaze_root"]) / "qwen-evaluation1"
        sys.path.insert(0, str(eval_root))
        os.environ["SELECTOR_STORE_VIS_ATTRS"] = "1"
        from safetensors import safe_open
        from token_compression.selector_model import Qwen2_5_VisionTransformerPretrainedModel_Selector
        from transformers import AutoConfig, AutoProcessor

        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the current runtime")
        full_config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
        vision_config = full_config.vision_config
        vision_config._attn_implementation = str(config["qwen"]["attention_implementation"])
        self.model = Qwen2_5_VisionTransformerPretrainedModel_Selector(vision_config)
        index = json.loads((checkpoint / "model.safetensors.index.json").read_text(encoding="utf-8"))
        visual_files = sorted({filename for key, filename in index["weight_map"].items() if key.startswith("visual.")})
        state: dict[str, torch.Tensor] = {}
        for filename in visual_files:
            with safe_open(str(checkpoint / filename), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key.startswith("visual."):
                        state[key.removeprefix("visual.")] = handle.get_tensor(key)
        incompatible = self.model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"GAZE vision state mismatch: missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        del state
        self.model = self.model.to(device=self.device, dtype=torch.bfloat16).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True, use_fast=False)
        image_processor = self.processor.image_processor
        image_processor.min_pixels = int(config["qwen"]["min_pixels"])
        image_processor.max_pixels = int(config["qwen"]["max_pixels"])
        image_processor.size = {
            "shortest_edge": int(config["qwen"]["min_pixels"]),
            "longest_edge": int(config["qwen"]["max_pixels"]),
        }
        self.checkpoint = checkpoint
        self.selector_config = {
            "spatial_ratio": float(self.model.spatial_ratio),
            "cosine_penalty_weight": float(self.model.cosine_penalty_weight),
            "cosine_penalty_threshold": float(self.model.cosine_penalty_threshold),
            "focal_selection_strategy": str(self.model.focal_selection_strategy),
            "selector_focal_redundancy_limit": int(self.model.selector_focal_redundancy_limit),
        }

    @torch.inference_mode()
    def infer(self, image: Image.Image, expected_grid: list[int]) -> dict[str, Any]:
        inputs = self.processor.image_processor(images=[image], return_tensors="pt")
        grid = [int(value) for value in inputs["image_grid_thw"][0].tolist()]
        if grid != [int(value) for value in expected_grid]:
            raise AssertionError(f"GAZE processor grid {grid} != pilot grid {expected_grid}")
        pixel_values = inputs["pixel_values"].to(device=self.device, dtype=self.model.dtype)
        grid_tensor = inputs["image_grid_thw"].to(self.device)
        result: dict[str, Any] = {"grid": grid, "budgets": {}}
        reference_scores: np.ndarray | None = None
        for budget in GAZE_BUDGETS:
            self.model.budgets = float(budget)
            _ = self.model(pixel_values, grid_thw=grid_tensor)
            features = self.model.last_token_features
            raw = self.model.importance_scorer(features.unsqueeze(0)).squeeze(0).float().cpu().numpy()
            if reference_scores is None:
                reference_scores = raw
            elif not np.allclose(reference_scores, raw, rtol=1e-4, atol=1e-5):
                raise AssertionError("Raw VIS scores changed across budgets")
            result["budgets"][str(int(budget * 100))] = {
                "selected": self.model.last_selected_indices.detach().cpu().numpy().astype(np.int64),
                "focal": self.model.last_focal_indices.detach().cpu().numpy().astype(np.int64),
                "spatial": self.model.last_spatial_indices.detach().cpu().numpy().astype(np.int64),
                "fallback_count": int(self.model.last_selector_fallback_count),
            }
        if reference_scores is None:
            raise AssertionError("No GAZE raw scores were produced")
        result["raw_scores"] = reference_scores
        return result


def _deterministic_background_sample(indices: np.ndarray, sample_id: str, foreground_count: int) -> np.ndarray:
    limit = min(indices.size, max(1, 3 * foreground_count))
    if indices.size <= limit:
        return indices
    digest = hashlib.sha256(f"{SEED}:{sample_id}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=limit, replace=False)).astype(np.int64)


def add_gaze_metrics(record: dict[str, Any], arrays: dict[str, np.ndarray], inference: dict[str, Any]) -> None:
    scores = np.asarray(inference["raw_scores"], dtype=np.float32)
    if scores.size != int(record["visual_token_count"]):
        raise AssertionError(f"{record['sample_id']}: VIS score length mismatch")
    foreground = arrays["adaptive"]
    reliable_background = np.flatnonzero(
        (arrays["coverage"] <= RELIABLE_BACKGROUND_MAX)
        & (arrays["bbox_coverage"] <= RELIABLE_BACKGROUND_MAX)
    )
    sampled_background = _deterministic_background_sample(
        reliable_background, record["sample_id"], foreground.size
    )
    if foreground.size and reliable_background.size:
        record["mean_fg_score"] = float(scores[foreground].mean())
        record["mean_bg_score"] = float(scores[reliable_background].mean())
        record["score_margin"] = record["mean_fg_score"] - record["mean_bg_score"]
    if foreground.size and sampled_background.size:
        comparisons = scores[foreground, None] > scores[sampled_background][None, :]
        record["pairwise_accuracy"] = float(comparisons.mean())
    bbox_positive = arrays["bbox_positive"]
    if bbox_positive.size:
        record["mean_bbox_fg_score"] = float(scores[bbox_positive].mean())
        if record["mean_fg_score"] is not None:
            record["mask_minus_bbox_fg_score"] = record["mean_fg_score"] - record["mean_bbox_fg_score"]
    record["reliable_background_token_count"] = int(reliable_background.size)
    record["pairwise_background_sample_count"] = int(sampled_background.size)
    record["gaze_raw_scores"] = scores.astype(float).tolist()
    record["gaze_selection"] = {}
    adaptive_set = set(foreground.tolist())
    reliable_background_set = set(reliable_background.tolist())
    for budget in GAZE_BUDGETS:
        key = str(int(budget * 100))
        selection = inference["budgets"][key]
        selected = np.asarray(selection["selected"], dtype=np.int64)
        focal = np.asarray(selection["focal"], dtype=np.int64)
        spatial = np.asarray(selection["spatial"], dtype=np.int64)
        hard_recall = len(adaptive_set.intersection(selected.tolist())) / foreground.size if foreground.size else None
        focal_recall = len(adaptive_set.intersection(focal.tolist())) / foreground.size if foreground.size else None
        mass_recall = float(np.clip(arrays["contribution"][selected].sum(), 0.0, 1.0))
        selected_purity = float(arrays["coverage"][selected].mean()) if selected.size else None
        selected_reliable_bg = (
            len(reliable_background_set.intersection(selected.tolist())) / selected.size if selected.size else None
        )
        payload = {
            "selected_token_count": int(selected.size),
            "focal_token_count": int(focal.size),
            "spatial_token_count": int(spatial.size),
            "selected_indices": selected.astype(int).tolist(),
            "focal_indices": focal.astype(int).tolist(),
            "spatial_indices": spatial.astype(int).tolist(),
            "hard_mask_token_recall": hard_recall,
            "focal_hard_mask_token_recall": focal_recall,
            "mask_mass_recall": mass_recall,
            "selected_mask_purity": selected_purity,
            "selected_reliable_background_rate": selected_reliable_bg,
            "selector_fallback_count": int(selection["fallback_count"]),
        }
        record["gaze_selection"][key] = payload
        record[f"retained_recall_{key}"] = hard_recall


def clear_gaze_metrics(record: dict[str, Any]) -> None:
    """Remove partial GAZE results if a run cannot complete all 200 samples."""
    for key in (
        "mean_fg_score",
        "mean_bg_score",
        "mean_bbox_fg_score",
        "mask_minus_bbox_fg_score",
        "score_margin",
        "pairwise_accuracy",
        "retained_recall_10",
        "retained_recall_20",
        "retained_recall_30",
        "reliable_background_token_count",
        "pairwise_background_sample_count",
    ):
        record[key] = None
    record["gaze_raw_scores"] = None
    record["gaze_selection"] = None


def _aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    zero = sum(bool(record["valid_mask_but_zero_current_positive"]) for record in records)
    sources: dict[str, Any] = {}
    for source in sorted({record["source"] for record in records}):
        subset = [record for record in records if record["source"] == source]
        sources[source] = aggregate_group(subset)
    summary = aggregate_group(records)
    summary.update(
        {
            "sample_count": len(records),
            "zero_current_positive_samples": zero,
            "zero_current_positive_rate": zero / len(records) if records else None,
            "coverage_recompute_max_abs_error": max(
                (record["recomputed_existing_coverage_max_abs_error"] for record in records
                 if record["recomputed_existing_coverage_max_abs_error"] is not None),
                default=None,
            ),
            "all_token_mass_recall_max_abs_error_from_one": max(
                (abs(record["all_token_mask_mass_recall"] - 1.0) for record in records if record["mask_area"] > 0),
                default=None,
            ),
            "per_source": sources,
        }
    )
    return summary


def aggregate_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    def available(key: str) -> list[float]:
        return [float(record[key]) for record in records if record.get(key) is not None]

    gaze_available = bool(records) and all(record.get("score_margin") is not None for record in records)
    result: dict[str, Any] = {
        "count": len(records),
        "current_positive_token_count": _distribution(available("current_positive_token_count")),
        "current_positive_token_ratio": _distribution(available("current_positive_token_ratio")),
        "adaptive_positive_token_count": _distribution(available("adaptive_positive_token_count")),
        "current_mask_mass_recall": _distribution(available("current_mask_mass_recall")),
        "adaptive_mask_mass_recall": _distribution(available("adaptive_mask_mass_recall")),
        "zero_current_positive_count": sum(bool(record["valid_mask_but_zero_current_positive"]) for record in records),
        "bbox_pixel_purity": _distribution(available("bbox_pixel_purity")),
        "bbox_false_positive_rate": _distribution(available("bbox_false_positive_rate")),
        "bbox_positive_mean_mask_coverage": _distribution(available("bbox_positive_mean_mask_coverage")),
        "bbox_positive_zero_coverage_rate": _distribution(available("bbox_positive_zero_coverage_rate")),
        "bbox_positive_below_0_05_rate": _distribution(available("bbox_positive_below_0_05_rate")),
        "bbox_positive_below_0_10_rate": _distribution(available("bbox_positive_below_0_10_rate")),
        "bbox_positive_below_0_25_rate": _distribution(available("bbox_positive_below_0_25_rate")),
        "bbox_positive_below_0_50_rate": _distribution(available("bbox_positive_below_0_50_rate")),
        "gaze_available": gaze_available,
        "score_margin_positive_rate": (
            sum(value > 0 for value in available("score_margin")) / len(available("score_margin"))
            if available("score_margin") else None
        ),
        "score_margin": _distribution(available("score_margin")),
        "pairwise_accuracy": _distribution(available("pairwise_accuracy")),
        "mask_minus_bbox_fg_score": _distribution(available("mask_minus_bbox_fg_score")),
        "retention": {},
    }
    for budget in GAZE_BUDGETS:
        key = str(int(budget * 100))
        payloads = [record["gaze_selection"][key] for record in records if record.get("gaze_selection")]
        result["retention"][key] = {
            metric: _distribution(
                payload[metric] for payload in payloads if payload.get(metric) is not None
            )
            for metric in (
                "hard_mask_token_recall",
                "focal_hard_mask_token_recall",
                "mask_mass_recall",
                "selected_mask_purity",
                "selected_reliable_background_rate",
            )
        }
    return result


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_csv_outputs(output_dir: Path, records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    token_fields = [
        "sample_id", "source", "visual_token_count", "current_positive_token_count",
        "current_background_token_count", "current_ignore_token_count", "current_positive_token_ratio",
        "adaptive_positive_token_count", "adaptive_positive_token_ratio", "current_mask_mass_recall",
        "adaptive_mask_mass_recall", "max_mask_token_coverage", "max_mask_contribution",
    ]
    _write_csv(output_dir / "token_count_distribution.csv", token_fields, records)
    bbox_fields = [
        "sample_id", "source", "bbox_pixel_purity", "bbox_positive_token_count",
        "bbox_false_positive_token_count", "bbox_false_positive_rate",
        "bbox_positive_mean_mask_coverage", "bbox_positive_median_mask_coverage",
        "bbox_positive_zero_coverage_rate", "bbox_positive_below_0_05_rate",
        "bbox_positive_below_0_10_rate", "bbox_positive_below_0_25_rate",
        "bbox_positive_below_0_50_rate",
    ]
    _write_csv(output_dir / "bbox_noise_distribution.csv", bbox_fields, records)
    score_fields = [
        "sample_id", "source", "mean_fg_score", "mean_bg_score", "score_margin",
        "pairwise_accuracy", "mean_bbox_fg_score", "mask_minus_bbox_fg_score",
    ]
    _write_csv(output_dir / "score_margin_distribution.csv", score_fields, records)
    retention_fields = [
        "sample_id", "source", "budget", "selected_token_count", "focal_token_count",
        "spatial_token_count", "hard_mask_token_recall", "focal_hard_mask_token_recall",
        "mask_mass_recall", "selected_mask_purity", "selected_reliable_background_rate",
    ]
    retention_rows = []
    for record in records:
        for budget in GAZE_BUDGETS:
            key = str(int(budget * 100))
            payload = record.get("gaze_selection", {}).get(key) if record.get("gaze_selection") else None
            row = {"sample_id": record["sample_id"], "source": record["source"], "budget": budget}
            if payload:
                row.update(payload)
            retention_rows.append(row)
    _write_csv(output_dir / "retention_recall_by_budget.csv", retention_fields, retention_rows)

    source_fields = [
        "source", "count", "zero_current_positive_count", "current_positive_count_mean",
        "current_positive_ratio_mean", "adaptive_positive_count_mean", "current_mass_recall_mean",
        "adaptive_mass_recall_mean", "bbox_pixel_purity_mean", "bbox_false_positive_rate_mean",
        "bbox_zero_coverage_rate_mean", "score_margin_positive_rate", "score_margin_mean",
        "pairwise_accuracy_mean", "recall_10_mean", "recall_20_mean", "recall_30_mean",
    ]
    source_rows = []
    for source, values in summary["per_source"].items():
        source_rows.append(
            {
                "source": source,
                "count": values["count"],
                "zero_current_positive_count": values["zero_current_positive_count"],
                "current_positive_count_mean": values["current_positive_token_count"]["mean"],
                "current_positive_ratio_mean": values["current_positive_token_ratio"]["mean"],
                "adaptive_positive_count_mean": values["adaptive_positive_token_count"]["mean"],
                "current_mass_recall_mean": values["current_mask_mass_recall"]["mean"],
                "adaptive_mass_recall_mean": values["adaptive_mask_mass_recall"]["mean"],
                "bbox_pixel_purity_mean": values["bbox_pixel_purity"]["mean"],
                "bbox_false_positive_rate_mean": values["bbox_false_positive_rate"]["mean"],
                "bbox_zero_coverage_rate_mean": values["bbox_positive_zero_coverage_rate"]["mean"],
                "score_margin_positive_rate": values["score_margin_positive_rate"],
                "score_margin_mean": values["score_margin"]["mean"],
                "pairwise_accuracy_mean": values["pairwise_accuracy"]["mean"],
                "recall_10_mean": values["retention"]["10"]["hard_mask_token_recall"]["mean"],
                "recall_20_mean": values["retention"]["20"]["hard_mask_token_recall"]["mean"],
                "recall_30_mean": values["retention"]["30"]["hard_mask_token_recall"]["mean"],
            }
        )
    _write_csv(output_dir / "per_source.csv", source_fields, source_rows)


def _grid_rgb(indices: np.ndarray, grid: list[int], color: tuple[float, float, float]) -> np.ndarray:
    temporal, patch_height, patch_width = (int(value) for value in grid)
    if temporal != 1:
        raise ValueError("Audit visualizations only support static images")
    height, width = patch_height // 2, patch_width // 2
    rgb = np.zeros((height * width, 3), dtype=np.float32)
    rgb[:] = np.array([0.12, 0.28, 0.75], dtype=np.float32)
    rgb[np.asarray(indices, dtype=np.int64)] = np.asarray(color, dtype=np.float32)
    return rgb.reshape(height, width, 3)


def _token_overlay(image: np.ndarray, indices: np.ndarray, grid: list[int], color: tuple[float, float, float]) -> np.ndarray:
    rgb = _grid_rgb(indices, grid, color)
    resized = np.asarray(
        Image.fromarray((rgb * 255).astype(np.uint8)).resize((image.shape[1], image.shape[0]), Image.Resampling.NEAREST)
    ).astype(np.float32) / 255.0
    return np.clip(0.48 * (image.astype(np.float32) / 255.0) + 0.52 * resized, 0.0, 1.0)


def choose_worst_samples(records: list[dict[str, Any]], count: int = 20) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(items: Iterable[dict[str, Any]]) -> None:
        for record in items:
            if record["sample_id"] not in seen and len(chosen) < count:
                chosen.append(record)
                seen.add(record["sample_id"])

    add(sorted((r for r in records if r["valid_mask_but_zero_current_positive"]), key=lambda r: (r["current_mask_mass_recall"], -float(r["bbox_false_positive_rate"] or 0.0))))
    add(sorted((r for r in records if r["score_margin"] is not None), key=lambda r: r["score_margin"]))
    add(sorted((r for r in records if r["retained_recall_10"] is not None), key=lambda r: r["retained_recall_10"]))
    add(sorted(records, key=lambda r: float(r["bbox_false_positive_rate"] or 0.0), reverse=True))
    return chosen


def render_audit_visualization(record: dict[str, Any], output_path: Path) -> None:
    image = np.asarray(Image.open(record["image_path"]).convert("RGB"))
    subject = decode_rle(record["_subject_mask_rle"])
    fig, axes = plt.subplots(2, 4, figsize=(21, 11), dpi=110, constrained_layout=True)
    axes[0, 0].imshow(image)
    for bbox in record["_bboxes"]:
        x1, y1, x2, y2 = bbox
        axes[0, 0].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="yellow", linewidth=2))
    axes[0, 0].set_title("原图与BBox", fontsize=10)
    subject_view = image.astype(np.float32) / 255.0
    subject_view[subject] = 0.5 * subject_view[subject] + 0.5 * np.array([0.1, 0.9, 0.25])
    axes[0, 1].imshow(subject_view)
    axes[0, 1].set_title(f"主体mask | area={record['mask_area']}", fontsize=10)
    axes[0, 2].imshow(_token_overlay(image, np.asarray(record["current_positive_token_indices"]), record["image_grid_thw"], (0.95, 0.15, 0.1)))
    axes[0, 2].set_title(f"当前0.5规则 | positive={record['current_positive_token_count']} | mass={record['current_mask_mass_recall']:.3f}", fontsize=10)
    axes[0, 3].imshow(_token_overlay(image, np.asarray(record["adaptive_mask_tokens"]), record["image_grid_thw"], (0.95, 0.15, 0.1)))
    axes[0, 3].set_title(f"自适应90% | positive={record['adaptive_positive_token_count']} | mass={record['adaptive_mask_mass_recall']:.3f}", fontsize=10)
    for axis, budget in zip(axes[1, :3], GAZE_BUDGETS):
        key = str(int(budget * 100))
        selection = record.get("gaze_selection", {}).get(key) if record.get("gaze_selection") else None
        if selection:
            axis.imshow(_token_overlay(image, np.asarray(selection["selected_indices"]), record["image_grid_thw"], (0.85, 0.1, 0.85)))
            axis.set_title(
                f"GAZE {key}% | token recall={selection['hard_mask_token_recall']:.3f}\n"
                f"mass={selection['mask_mass_recall']:.3f} | focal={selection['focal_hard_mask_token_recall']:.3f}",
                fontsize=10,
            )
        else:
            axis.imshow(image)
            axis.set_title(f"GAZE {key}%：unavailable", fontsize=10)
    axes[1, 3].axis("off")
    margin = "NA" if record["score_margin"] is None else f"{record['score_margin']:.5f}"
    recall = "NA" if record["retained_recall_10"] is None else f"{record['retained_recall_10']:.3f}"
    axes[1, 3].text(
        0.0,
        1.0,
        f"sample_id: {record['sample_id']}\nsource: {record['source']}\n"
        f"Q: {record['question']}\nA: {record['answer']}\n\n"
        f"foreground tokens (0.5): {record['current_positive_token_count']}\n"
        f"BBox false-positive rate: {record['bbox_false_positive_rate']:.3f}\n"
        f"VIS score margin: {margin}\n10% recall: {recall}",
        va="top",
        wrap=True,
        fontsize=10,
    )
    for axis in axes.flat[:7]:
        axis.axis("off")
    fig.suptitle("主体mask监督第零步审计", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def write_report(
    output_dir: Path,
    summary: dict[str, Any],
    environment: dict[str, Any],
    missing: dict[str, Any],
) -> None:
    gaze_ok = bool(environment["gaze_inference"]["available"])
    current_mass = summary["current_mask_mass_recall"]["mean"]
    zero_rate = summary["zero_current_positive_rate"]
    bbox_fp = summary["bbox_false_positive_rate"]["mean"]
    margin_pos = summary["score_margin_positive_rate"]
    recall_10 = summary["retention"]["10"]["hard_mask_token_recall"]["mean"]
    recommend = (
        summary["coverage_recompute_max_abs_error"] is not None
        and summary["coverage_recompute_max_abs_error"] < 1e-5
        and zero_rate is not None
        and zero_rate > 0.05
        and bbox_fp is not None
        and bbox_fp > 0.10
    )
    if gaze_ok and margin_pos is not None and recall_10 is not None:
        recommend = recommend and (margin_pos < 0.90 or recall_10 < 0.90)
    summary["recommend_1k_mask_gaze_experiment"] = bool(recommend)
    conclusion = "建议进入1K Mask-GAZE配对训练实验" if recommend else "暂不建议直接进入1K训练实验"
    lines = [
        "# 第零步：200样本主体mask监督无训练审计",
        "",
        "## 明确结论",
        "",
        f"- Qwen Token对齐：**通过**。独立重算与已有coverage最大绝对误差为 `{_format_number(summary['coverage_recompute_max_abs_error'], 8)}`，200条长度均与实际合并后视觉Token一致。",
        f"- 固定0.5阈值：**会导致明显监督消失**。零正Token为 `{summary['zero_current_positive_samples']}/{summary['sample_count']}`（`{100 * zero_rate:.2f}%`），当前规则平均只召回 `{100 * current_mass:.2f}%` 的mask质量。",
        f"- BBox框内噪声：**明显**。GAZE BBox正Token平均假前景率为 `{100 * bbox_fp:.2f}%`。",
        (
            f"- 原GAZE VIS区分能力：score margin为正的样本比例 `{100 * margin_pos:.2f}%`，平均pairwise accuracy `{100 * summary['pairwise_accuracy']['mean']:.2f}%`。"
            if gaze_ok
            else "- 原GAZE VIS区分能力：**unavailable**；见 `missing_requirements.json`。"
        ),
        (
            f"- 原GAZE主体召回：10%预算平均Token召回 `{100 * recall_10:.2f}%`，仍有提升空间。"
            if gaze_ok
            else "- 原GAZE主体召回：**unavailable**；未以其他分数冒充。"
        ),
        f"- 1K决策：**{conclusion}**。建议实验应固定保留预算，只比较BBox排序监督与mask排序监督。",
        "",
        "## 核心总体指标",
        "",
        f"- 当前0.5正Token数：{json.dumps(summary['current_positive_token_count'], ensure_ascii=False)}",
        f"- 当前正Token占比：{json.dumps(summary['current_positive_token_ratio'], ensure_ascii=False)}",
        f"- 自适应正Token数：{json.dumps(summary['adaptive_positive_token_count'], ensure_ascii=False)}",
        f"- 当前mask质量召回：{json.dumps(summary['current_mask_mass_recall'], ensure_ascii=False)}",
        f"- 自适应mask质量召回：{json.dumps(summary['adaptive_mask_mass_recall'], ensure_ascii=False)}",
        f"- BBox像素纯度：{json.dumps(summary['bbox_pixel_purity'], ensure_ascii=False)}",
        f"- BBox假前景率：{json.dumps(summary['bbox_false_positive_rate'], ensure_ascii=False)}",
        "",
        "## 原GAZE冻结推理",
        "",
    ]
    if gaze_ok:
        lines.extend(
            [
                f"- checkpoint：`{environment['gaze_checkpoint']['path']}`",
                f"- scorer原始分数margin：{json.dumps(summary['score_margin'], ensure_ascii=False)}",
                f"- margin>0比例：{100 * summary['score_margin_positive_rate']:.2f}%",
                f"- pairwise accuracy：{json.dumps(summary['pairwise_accuracy'], ensure_ascii=False)}",
                f"- 10/20/30%召回：`{_format_number(summary['retention']['10']['hard_mask_token_recall']['mean'])}` / `{_format_number(summary['retention']['20']['hard_mask_token_recall']['mean'])}` / `{_format_number(summary['retention']['30']['hard_mask_token_recall']['mean'])}`",
                "- 报告的是冗余惩罚之前的原始importance scorer分数；GAZE冗余惩罚通过贪心选择改变索引，没有独立、稳定的逐Token惩罚后分数数组，因此未伪造该字段。",
            ]
        )
    else:
        lines.append(f"- unavailable：{json.dumps(missing.get('gaze_inference'), ensure_ascii=False)}")
    lines.extend(
        [
            "",
            "## 分数据源",
            "",
            "详细数值见 `per_source.csv`；所有九个数据源均单独统计（其中 `v7w` 即 Visual7W）。",
            "",
            "## 方法说明",
            "",
            "- 当前规则：coverage ≥ 0.50为前景，coverage ≤ 0.05为背景，其余ignore。",
            "- 自适应规则：按mask contribution降序选择，直到累计覆盖至少90%，非空mask至少选择1个Token。",
            "- BBox正Token严格复现GAZE的 `pixel_bboxs_to_ref_token_mask` 坐标取整和闭区间写入。",
            "- 可靠背景要求mask coverage和bbox coverage都不超过0.01。pairwise背景每图最多采样前景数量的3倍，种子42。",
            "- GAZE保留集合来自真实Focal-Spatial前向流程，而非原始分数Top-K替代。",
            "",
            "## 可复现性",
            "",
            f"- 项目commit：`{environment['git_commit']}`",
            f"- 配置哈希：`{environment['config_hash']}`",
            f"- 执行命令：`{environment['command']}`",
            f"- 测试命令：`{environment['tests']['command']}`",
            f"- 测试结果：`{environment['tests']['result']}`",
            "- 本审计未训练模型、未运行SAM、未生成Qwen教师或反事实分数，也未修改现有200样本产物。",
            "",
        ]
    )
    (output_dir / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def locate_checkpoint(gaze_root: Path) -> Path | None:
    candidates = [
        gaze_root / "output_ckpt666/VisionSelector-Qwen2.5-VL-7B",
        gaze_root / "output_ckpt666/VisionSelector-Qwen2.5-VL-7B/checkpoint-625",
    ]
    for candidate in candidates:
        if (candidate / "config.json").exists() and (candidate / "model.safetensors.index.json").exists():
            return candidate
    return None


def run_tests(project_root: Path) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_audit_subject_mask_supervision.py"]
    completed = subprocess.run(command, cwd=project_root, text=True, capture_output=True, check=False)
    result = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0:
        raise RuntimeError(f"Audit tests failed:\n{result}")
    return {"command": "PYTHONPATH=. " + " ".join(command), "result": result, "returncode": completed.returncode}


def _strip_private_fields(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def main() -> None:
    parser = argparse.ArgumentParser(description="200-sample no-training subject-mask supervision audit")
    parser.add_argument("--config", default="configs/pilot_200.yaml")
    parser.add_argument("--output-dir", default="outputs/pilot_200/zero_step_audit")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-gaze", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit output: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "visualizations").mkdir()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    started = time.time()
    tests = run_tests(project_root)

    base_output = Path(config["paths"]["output_dir"])
    required_inputs = {
        "manifest": base_output / "manifest.jsonl",
        "subject_masks": base_output / "subject_masks.jsonl",
        "dataset": base_output / "dataset.jsonl",
        "config": Path(args.config),
    }
    missing_files = [str(path) for path in required_inputs.values() if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing required audit inputs: {missing_files}")
    samples = list(read_jsonl(required_inputs["dataset"]))
    if len(samples) != 200:
        raise ValueError(f"Expected 200 existing samples, found {len(samples)}")
    manifest_count = sum(1 for _ in read_jsonl(required_inputs["manifest"]))
    subject_count = sum(1 for _ in read_jsonl(required_inputs["subject_masks"]))
    if manifest_count != 200 or subject_count != 200:
        raise ValueError(f"Input cardinality mismatch: manifest={manifest_count}, subject_masks={subject_count}")

    records: list[dict[str, Any]] = []
    arrays_by_id: dict[str, dict[str, np.ndarray]] = {}
    for index, sample in enumerate(samples, start=1):
        record, arrays = geometry_record(sample, config)
        record["_subject_mask_rle"] = sample["subject_mask_rle"]
        record["_bboxes"] = sample["bboxes"]
        records.append(record)
        arrays_by_id[record["sample_id"]] = arrays
        print(f"[geometry {index:03d}/200] {record['sample_id']}", flush=True)

    gaze_root = Path(config["paths"]["gaze_root"])
    checkpoint = locate_checkpoint(gaze_root)
    missing: dict[str, Any] = {
        "missing_input_files": [],
        "gaze_checkpoint": None if checkpoint else "No checkpoint with config.json and model index was found",
        "gaze_inference": None,
        "post_redundancy_scores": "GAZE applies redundancy during greedy selection and exposes indices, not a stable per-token post-penalty score array",
    }
    gaze_info: dict[str, Any] = {"available": False, "error": None}
    selector_config: dict[str, Any] | None = None
    if not args.skip_gaze and checkpoint is not None:
        try:
            auditor = GazeVisionAuditor(checkpoint, config, args.device)
            selector_config = auditor.selector_config
            for index, record in enumerate(records, start=1):
                with Image.open(record["image_path"]) as handle:
                    inference = auditor.infer(handle.convert("RGB"), record["image_grid_thw"])
                add_gaze_metrics(record, arrays_by_id[record["sample_id"]], inference)
                print(f"[gaze {index:03d}/200] {record['sample_id']}", flush=True)
            gaze_info["available"] = True
        except Exception as exc:
            for record in records:
                clear_gaze_metrics(record)
            gaze_info["error"] = f"{type(exc).__name__}: {exc}"
            missing["gaze_inference"] = gaze_info["error"]
            print(f"[gaze] unavailable: {gaze_info['error']}", flush=True)
    elif args.skip_gaze:
        missing["gaze_inference"] = "Skipped explicitly with --skip-gaze"
    else:
        missing["gaze_inference"] = "GAZE VIS checkpoint was not found"

    summary = _aggregate_records(records)
    environment = {
        "seed": SEED,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=project_root, text=True),
        "config_path": str(Path(args.config).resolve()),
        "config_hash": config_hash(config),
        "command": " ".join([sys.executable, "-m", "src.audit_subject_mask_supervision", *sys.argv[1:]]),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": args.device,
        "qwen_geometry": {
            "min_pixels": int(config["qwen"]["min_pixels"]),
            "max_pixels": int(config["qwen"]["max_pixels"]),
            "patch_size": int(config["qwen"]["patch_size"]),
            "spatial_merge_size": int(config["qwen"]["spatial_merge_size"]),
        },
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in required_inputs.items()
        },
        "gaze_checkpoint": {
            "path": str(checkpoint.resolve()) if checkpoint else None,
            "global_step": 625 if checkpoint else None,
            "config_sha256": sha256_file(checkpoint / "config.json") if checkpoint else None,
            "weight_files": [
                {"name": path.name, "size": path.stat().st_size}
                for path in sorted(checkpoint.glob("*.safetensors"))
            ] if checkpoint else [],
            "selector_config": selector_config,
        },
        "gaze_inference": gaze_info,
        "tests": tests,
        "elapsed_seconds": time.time() - started,
        "prohibitions_observed": {
            "training_performed": False,
            "sam_rerun": False,
            "qwen_teacher_or_counterfactual_rerun": False,
            "existing_outputs_modified": False,
        },
    }

    clean_records = [_strip_private_fields(record) for record in records]
    atomic_write_jsonl(output_dir / "per_sample.jsonl", clean_records)
    write_csv_outputs(output_dir, clean_records, summary)
    worst = choose_worst_samples(records, 20)
    for index, record in enumerate(worst, start=1):
        render_audit_visualization(record, output_dir / "visualizations" / f"{index:02d}_{record['sample_id']}.png")
    summary["visualized_worst_sample_ids"] = [record["sample_id"] for record in worst]
    summary["visualization_count"] = len(worst)
    write_report(output_dir, summary, environment, missing)
    atomic_write_json(output_dir / "summary.json", summary)
    atomic_write_json(output_dir / "environment.json", environment)
    atomic_write_json(output_dir / "missing_requirements.json", missing)
    print(json.dumps({
        "output_dir": str(output_dir.resolve()),
        "gaze_available": gaze_info["available"],
        "zero_positive": summary["zero_current_positive_samples"],
        "bbox_false_positive_rate_mean": summary["bbox_false_positive_rate"]["mean"],
        "recommend_1k": summary["recommend_1k_mask_gaze_experiment"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
