from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import cv2
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from src.data.build_manifest import _normalize_bboxes, _question_from_record, _resolve_image
from src.masks import bbox_mask, decode_rle, encode_rle
from src.sam_masks.subject_mask import load_sam_predictor, segment_subject
from src.utils import (
    append_jsonl,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json,
    config_hash,
    load_config,
    read_jsonl,
    seed_everything,
    sha256_file,
    stable_sample_id,
)


MANIFEST_NAME = "manifest_1k.jsonl"
MASKS_NAME = "subject_masks.jsonl"
TOKENS_NAME = "token_supervision_qwen25vl.jsonl"
FAILURES_NAME = "failures.jsonl"


def distribution(values: Iterable[float | int]) -> dict[str, float | int | None]:
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


def union_bbox_mask(shape: tuple[int, int], bboxes: list[list[float]]) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    for bbox in bboxes:
        result |= bbox_mask(shape, bbox)
    return result


def token_coverage_and_contribution(
    mask: np.ndarray,
    image_grid_thw: list[int] | tuple[int, int, int],
    *,
    patch_size: int,
    spatial_merge_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    temporal, patch_height, patch_width = (int(value) for value in image_grid_thw)
    if temporal <= 0 or patch_height <= 0 or patch_width <= 0:
        raise ValueError(f"Invalid image_grid_thw={image_grid_thw}")
    if patch_height % spatial_merge_size or patch_width % spatial_merge_size:
        raise ValueError(f"Grid is not divisible by merge size: {image_grid_thw}")
    resized_height = patch_height * patch_size
    resized_width = patch_width * patch_size
    resized = cv2.resize(
        np.asarray(mask, dtype=np.float32),
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    cell = patch_size * spatial_merge_size
    token_height = patch_height // spatial_merge_size
    token_width = patch_width // spatial_merge_size
    intersection_2d = resized.reshape(token_height, cell, token_width, cell).sum(axis=(1, 3))
    intersections = np.tile(intersection_2d.reshape(-1), temporal).astype(np.float64)
    coverage = intersections / float(cell * cell)
    total = float(intersections.sum())
    contribution = intersections / total if total > 0 else np.zeros_like(intersections)
    expected = temporal * token_height * token_width
    if coverage.size != expected or contribution.size != expected:
        raise AssertionError("Token geometry length mismatch")
    return np.clip(coverage, 0.0, 1.0).astype(np.float32), np.clip(contribution, 0.0, 1.0).astype(np.float32)


def adaptive_indices(contribution: np.ndarray, target: float) -> np.ndarray:
    contribution = np.asarray(contribution, dtype=np.float64)
    candidates = np.flatnonzero(contribution > 0.0)
    if not candidates.size:
        return np.empty((0,), dtype=np.int64)
    order = candidates[np.argsort(-contribution[candidates], kind="stable")]
    cumulative = np.cumsum(contribution[order])
    count = int(np.searchsorted(cumulative, target, side="left")) + 1
    count = max(1, min(count, order.size))
    return np.sort(order[:count]).astype(np.int64)


def make_labels(
    foreground: np.ndarray,
    reliable_background: np.ndarray,
    count: int,
    *,
    foreground_label: int = 1,
    background_label: int = 0,
    ignore_label: int = -1,
) -> np.ndarray:
    labels = np.full(count, ignore_label, dtype=np.int8)
    labels[np.asarray(reliable_background, dtype=np.int64)] = background_label
    labels[np.asarray(foreground, dtype=np.int64)] = foreground_label
    return labels


def _load_pilot_exclusions(config: dict[str, Any]) -> tuple[set[str], set[str]]:
    pilot_path = Path(config["paths"]["pilot_manifest"])
    if not pilot_path.exists():
        return set(), set()
    rows = list(read_jsonl(pilot_path))
    return ({str(row["sample_id"]) for row in rows}, {str(Path(row["image_path"]).resolve()) for row in rows})


def build_manifest(config: dict[str, Any]) -> dict[str, Any]:
    paths = config["paths"]
    data = config["data"]
    output_dir = Path(paths["output_dir"])
    destination = output_dir / MANIFEST_NAME
    report_path = output_dir / "manifest_report.json"
    if destination.exists():
        rows = list(read_jsonl(destination))
        if report_path.exists():
            return json.loads(report_path.read_text(encoding="utf-8"))
        return {"selected": len(rows), "selected_source_counts": dict(Counter(row["source"] for row in rows))}

    source_path = Path(paths["training_manifest"])
    image_root = Path(paths["image_root"])
    quotas = {str(key): int(value) for key, value in data["quotas"].items()}
    required_split = str(data["required_split"]).lower()
    pilot_ids, pilot_images = _load_pilot_exclusions(config)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    structural_failures: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    for line_number, raw in enumerate(read_jsonl(source_path), start=1):
        source = str(raw.get("dataset") or raw.get("source") or "").strip().lower()
        source_counts[source or "<missing>"] += 1
        if source not in quotas:
            continue
        split = str(raw.get("split") or "").strip().lower()
        if split != required_split:
            structural_failures["wrong_split"] += 1
            continue
        question = _question_from_record(raw, str(data.get("short_answer_suffix", "")))
        answer_value = raw.get("output", raw.get("answer"))
        answer = "" if answer_value is None else str(answer_value).strip()
        image_path = _resolve_image(raw, image_root, str(data.get("image_prefix_to_strip", "")))
        bbox_data = _normalize_bboxes(raw)
        if not question or not answer:
            structural_failures["empty_question_or_answer"] += 1
            continue
        if image_path is None:
            structural_failures["invalid_image_field"] += 1
            continue
        if bbox_data is None:
            structural_failures["invalid_bbox"] += 1
            continue
        bboxes, width, height = bbox_data
        resolved_image = str(image_path.resolve())
        sample_id = stable_sample_id(source, resolved_image, question, answer)
        groups[source].append(
            {
                "sample_id": sample_id,
                "source": source,
                "split": split,
                "image": resolved_image,
                "image_path": resolved_image,
                "question": question,
                "answer": answer,
                "bbox": bboxes[0],
                "bboxes": bboxes,
                "width": width,
                "height": height,
                "source_line": line_number,
                "source_manifest": str(source_path.resolve()),
            }
        )

    rng = random.Random(int(config["project"]["seed"]))
    selected: list[dict[str, Any]] = []
    selection_counts: Counter[str] = Counter()
    excluded_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    seen_images: set[str] = set()
    for source, quota in quotas.items():
        candidates = list(groups[source])
        rng.shuffle(candidates)
        accepted = 0
        for candidate in candidates:
            if accepted >= quota:
                break
            if candidate["sample_id"] in pilot_ids:
                excluded_counts["pilot_sample_id"] += 1
                continue
            if bool(data.get("exclude_pilot_images", True)) and candidate["image_path"] in pilot_images:
                excluded_counts["pilot_image"] += 1
                continue
            if candidate["sample_id"] in seen_ids:
                excluded_counts["duplicate_sample_id"] += 1
                continue
            if candidate["image_path"] in seen_images:
                excluded_counts["duplicate_selected_image"] += 1
                continue
            image_path = Path(candidate["image_path"])
            if not image_path.is_file():
                excluded_counts["image_missing"] += 1
                continue
            try:
                with Image.open(image_path) as image:
                    actual_size = image.size
                    image.verify()
            except Exception:
                excluded_counts["image_unreadable"] += 1
                continue
            if actual_size != (candidate["width"], candidate["height"]):
                excluded_counts["declared_image_size_mismatch"] += 1
                continue
            selected.append(candidate)
            seen_ids.add(candidate["sample_id"])
            seen_images.add(candidate["image_path"])
            accepted += 1
        selection_counts[source] = accepted

    rng.shuffle(selected)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(destination, selected)
    report = {
        "seed": int(config["project"]["seed"]),
        "requested": int(sum(quotas.values())),
        "selected": len(selected),
        "requested_quotas": quotas,
        "selected_source_counts": dict(sorted(selection_counts.items())),
        "input_source_counts": dict(sorted(source_counts.items())),
        "structurally_valid_counts": {source: len(groups[source]) for source in quotas},
        "structural_failure_counts": dict(sorted(structural_failures.items())),
        "selection_exclusion_counts": dict(sorted(excluded_counts.items())),
        "pilot_sample_id_overlap": len(seen_ids.intersection(pilot_ids)),
        "pilot_image_overlap": len(seen_images.intersection(pilot_images)),
        "unique_image_count": len(seen_images),
        "source_manifest": str(source_path.resolve()),
        "source_manifest_sha256": sha256_file(source_path),
    }
    atomic_write_json(report_path, report)
    print(f"[manifest] selected={len(selected)} sources={dict(selection_counts)}", flush=True)
    return report


def _fallback_reasons(instance: dict[str, Any], sam_config: dict[str, Any]) -> list[str]:
    if not instance.get("fallback"):
        return []
    metrics = instance["selected"]
    reasons: list[str] = []
    if not metrics["center_in_bbox"]:
        reasons.append("mask_center_outside_bbox")
    if float(metrics["bbox_coverage"]) < float(sam_config["subject_min_bbox_coverage"]):
        reasons.append("bbox_coverage_below_minimum")
    if float(metrics["image_area_ratio"]) > float(sam_config["subject_max_image_area"]):
        reasons.append("mask_area_above_maximum")
    if float(metrics["image_area_ratio"]) < float(sam_config["subject_min_image_area"]):
        reasons.append("mask_area_below_minimum")
    return reasons or ["sam_candidate_failed_subject_validity_rule"]


def _bbox_fallback_record(sample: dict[str, Any], reason: str, elapsed: float) -> dict[str, Any]:
    shape = (int(sample["height"]), int(sample["width"]))
    mask = union_bbox_mask(shape, sample["bboxes"])
    return {
        **sample,
        "image_height": shape[0],
        "image_width": shape[1],
        "subject_mask_rle": encode_rle(mask),
        "sam_pred_iou": None,
        "stability_score": None,
        "inside_bbox_ratio": None,
        "mask_area_ratio": float(mask.mean()),
        "mask_source": "bbox_fallback",
        "fallback_to_bbox": True,
        "fallback_reasons": [reason],
        "subject_instances": [],
        "timing": {"sam_subject_seconds": elapsed},
    }


def run_masks(config: dict[str, Any]) -> None:
    output_dir = Path(config["paths"]["output_dir"])
    manifest_path = output_dir / MANIFEST_NAME
    destination = output_dir / MASKS_NAME
    failure_path = output_dir / FAILURES_NAME
    manifest = list(read_jsonl(manifest_path))
    completed = {row["sample_id"] for row in read_jsonl(destination)} if destination.exists() else set()
    failure_keys = {
        (row.get("stage"), row.get("sample_id"), row.get("reason"))
        for row in read_jsonl(failure_path)
    } if failure_path.exists() else set()
    predictor = load_sam_predictor(config)
    sam_config = config["sam"]
    log_every = int(config["runtime"].get("log_every", 1))

    for index, sample in enumerate(manifest, start=1):
        if sample["sample_id"] in completed:
            continue
        started = time.perf_counter()
        processing_error: str | None = None
        try:
            image = np.asarray(Image.open(sample["image_path"]).convert("RGB"))
            result = segment_subject(image, sample["bboxes"], predictor, sam_config)
            mask = result.pop("subject_mask")
            result.pop("instance_masks")
            reasons = [reason for instance in result["subject_instances"] for reason in _fallback_reasons(instance, sam_config)]
            valid_metrics = [instance["selected"] for instance in result["subject_instances"] if not instance["fallback"]]
            any_fallback = bool(result["subject_mask_fallback"])
            if any_fallback and valid_metrics:
                mask_source = "mixed_sam_bbox_fallback"
            elif any_fallback:
                mask_source = "bbox_fallback"
            else:
                mask_source = "sam_bbox_prompt"
            record = {
                **sample,
                "image_height": int(image.shape[0]),
                "image_width": int(image.shape[1]),
                "subject_mask_rle": result["subject_mask_rle"],
                "sam_pred_iou": float(mean(metric["pred_iou"] for metric in valid_metrics)) if valid_metrics else None,
                "stability_score": float(mean(metric["stability"] for metric in valid_metrics)) if valid_metrics else None,
                "inside_bbox_ratio": float(mean(metric["inside_ratio"] for metric in valid_metrics)) if valid_metrics else None,
                "mask_area_ratio": float(mask.mean()),
                "mask_source": mask_source,
                "fallback_to_bbox": any_fallback,
                "fallback_reasons": reasons,
                "subject_instances": result["subject_instances"],
                "timing": {"sam_subject_seconds": time.perf_counter() - started},
            }
        except Exception as exc:
            processing_error = f"{type(exc).__name__}: {exc}"
            record = _bbox_fallback_record(sample, f"sam_exception: {processing_error}", time.perf_counter() - started)
        append_jsonl(destination, record)
        if record["fallback_to_bbox"]:
            reason = ";".join(record["fallback_reasons"])
            key = ("sam_subject", sample["sample_id"], reason)
            if key not in failure_keys:
                append_jsonl(
                    failure_path,
                    {
                        "stage": "sam_subject",
                        "sample_id": sample["sample_id"],
                        "source": sample["source"],
                        "reason": reason,
                        "recovered_with_bbox": True,
                        "exception": processing_error,
                    },
                )
                failure_keys.add(key)
        if index % log_every == 0:
            print(f"[mask {index:04d}/{len(manifest)}] {sample['sample_id']} fallback={record['fallback_to_bbox']}", flush=True)


def _processor(config: dict[str, Any]):
    from transformers import AutoProcessor

    checkpoint = Path(config["paths"]["qwen_checkpoint"])
    qwen = config["qwen"]
    processor = AutoProcessor.from_pretrained(
        checkpoint,
        local_files_only=True,
        use_fast=bool(qwen.get("use_fast", False)),
    )
    processor.image_processor.min_pixels = int(qwen["min_pixels"])
    processor.image_processor.max_pixels = int(qwen["max_pixels"])
    processor.image_processor.size = {
        "shortest_edge": int(qwen["min_pixels"]),
        "longest_edge": int(qwen["max_pixels"]),
    }
    return processor


def processor_configuration(config: dict[str, Any]) -> dict[str, Any]:
    checkpoint = Path(config["paths"]["qwen_checkpoint"])
    qwen = config["qwen"]
    payload = {
        "checkpoint": str(checkpoint.resolve()),
        "preprocessor_config_sha256": sha256_file(checkpoint / "preprocessor_config.json"),
        "min_pixels": int(qwen["min_pixels"]),
        "max_pixels": int(qwen["max_pixels"]),
        "patch_size": int(qwen["patch_size"]),
        "spatial_merge_size": int(qwen["spatial_merge_size"]),
        "use_fast": bool(qwen.get("use_fast", False)),
    }
    payload["hash"] = config_hash(payload)
    return payload


def run_tokens(config: dict[str, Any]) -> None:
    output_dir = Path(config["paths"]["output_dir"])
    source_path = output_dir / MASKS_NAME
    destination = output_dir / TOKENS_NAME
    samples = list(read_jsonl(source_path))
    completed = {row["sample_id"] for row in read_jsonl(destination)} if destination.exists() else set()
    processor = _processor(config)
    processor_info = processor_configuration(config)
    qwen = config["qwen"]
    supervision = config["supervision"]
    target = float(supervision["adaptive_mass_target"])
    reliable_max = float(supervision["reliable_background_max_coverage"])
    patch_size = int(qwen["patch_size"])
    merge = int(qwen["spatial_merge_size"])
    log_every = int(config["runtime"].get("log_every", 1))

    for index, sample in enumerate(samples, start=1):
        if sample["sample_id"] in completed:
            continue
        started = time.perf_counter()
        with Image.open(sample["image_path"]) as handle:
            image = handle.convert("RGB")
            processed = processor.image_processor(images=[image], return_tensors="pt")
        grid = [int(value) for value in processed["image_grid_thw"][0].tolist()]
        del processed
        visual_count = grid[0] * (grid[1] // merge) * (grid[2] // merge)
        subject = decode_rle(sample["subject_mask_rle"])
        bbox_pixels = union_bbox_mask(subject.shape, sample["bboxes"])
        mask_coverage, mask_contribution = token_coverage_and_contribution(
            subject, grid, patch_size=patch_size, spatial_merge_size=merge
        )
        bbox_coverage, bbox_contribution = token_coverage_and_contribution(
            bbox_pixels, grid, patch_size=patch_size, spatial_merge_size=merge
        )
        mask_foreground = adaptive_indices(mask_contribution, target)
        bbox_foreground = adaptive_indices(bbox_contribution, target)
        reliable_background_candidates = np.flatnonzero(
            (mask_coverage <= reliable_max) & (bbox_coverage <= reliable_max)
        ).astype(np.int64)
        # A very thin foreground can occupy <=1% of a cell. The explicit
        # foreground-priority rule removes such cells from the shared negative set.
        foreground_union = np.union1d(mask_foreground, bbox_foreground)
        reliable_background = np.setdiff1d(
            reliable_background_candidates, foreground_union, assume_unique=False
        ).astype(np.int64)
        mask_labels = make_labels(
            mask_foreground,
            reliable_background,
            visual_count,
            foreground_label=int(supervision["foreground_label"]),
            background_label=int(supervision["background_label"]),
            ignore_label=int(supervision["ignore_label"]),
        )
        bbox_labels = make_labels(
            bbox_foreground,
            reliable_background,
            visual_count,
            foreground_label=int(supervision["foreground_label"]),
            background_label=int(supervision["background_label"]),
            ignore_label=int(supervision["ignore_label"]),
        )
        background_confidence = 1.0 - np.maximum(mask_coverage, bbox_coverage)
        mask_confidence = np.zeros(visual_count, dtype=np.float32)
        bbox_confidence = np.zeros(visual_count, dtype=np.float32)
        mask_confidence[reliable_background] = background_confidence[reliable_background]
        bbox_confidence[reliable_background] = background_confidence[reliable_background]
        mask_confidence[mask_foreground] = mask_coverage[mask_foreground]
        bbox_confidence[bbox_foreground] = bbox_coverage[bbox_foreground]
        mask_set = set(mask_foreground.tolist())
        bbox_set = set(bbox_foreground.tolist())
        intersection = len(mask_set.intersection(bbox_set))
        union = len(mask_set.union(bbox_set))
        record = {
            "sample_id": sample["sample_id"],
            "source": sample["source"],
            "image": sample["image_path"],
            "image_grid_thw": grid,
            "processed_image_size": [grid[1] * patch_size, grid[2] * patch_size],
            "visual_token_count": int(visual_count),
            "mask_token_coverage": mask_coverage.astype(float).tolist(),
            "mask_token_contribution": mask_contribution.astype(float).tolist(),
            "bbox_token_coverage": bbox_coverage.astype(float).tolist(),
            "bbox_token_contribution": bbox_contribution.astype(float).tolist(),
            "mask_foreground_indices": mask_foreground.astype(int).tolist(),
            "bbox_foreground_indices": bbox_foreground.astype(int).tolist(),
            "reliable_background_indices": reliable_background.astype(int).tolist(),
            "mask_token_labels": mask_labels.astype(int).tolist(),
            "bbox_token_labels": bbox_labels.astype(int).tolist(),
            "token_confidence": mask_confidence.astype(float).tolist(),
            "mask_token_confidence": mask_confidence.astype(float).tolist(),
            "bbox_token_confidence": bbox_confidence.astype(float).tolist(),
            "adaptive_mass_target": target,
            "actual_mask_mass_coverage": float(np.clip(mask_contribution[mask_foreground].sum(), 0.0, 1.0)),
            "actual_bbox_mass_coverage": float(np.clip(bbox_contribution[bbox_foreground].sum(), 0.0, 1.0)),
            "foreground_overlap_iou": float(intersection / union) if union else 1.0,
            "mask_foreground_covered_by_bbox_rate": float(intersection / len(mask_set)) if mask_set else None,
            "processor_config_hash": processor_info["hash"],
            "fallback_to_bbox": bool(sample["fallback_to_bbox"]),
            "timing": {"qwen_preprocess_and_token_mapping_seconds": time.perf_counter() - started},
            "metadata": {
                "token_mapping": "qwen_processor_grid_then_area_pool_over_merged_patch_cells",
                "confidence_definition": "foreground uses group coverage; reliable background uses one minus max(mask,bbox) coverage; ignore is zero",
                "foreground_priority_over_background": True,
            },
        }
        append_jsonl(destination, record)
        if index % log_every == 0:
            print(
                f"[tokens {index:04d}/{len(samples)}] {sample['sample_id']} grid={grid} n={visual_count}",
                flush=True,
            )


def validate_outputs(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["paths"]["output_dir"])
    manifest = list(read_jsonl(output_dir / MANIFEST_NAME))
    masks = list(read_jsonl(output_dir / MASKS_NAME))
    tokens = list(read_jsonl(output_dir / TOKENS_NAME))
    failures = list(read_jsonl(output_dir / FAILURES_NAME)) if (output_dir / FAILURES_NAME).exists() else []
    errors: list[str] = []
    expected = int(config["project"]["sample_count"])
    manifest_ids = [row["sample_id"] for row in manifest]
    mask_ids = [row["sample_id"] for row in masks]
    token_ids = [row["sample_id"] for row in tokens]
    if len(manifest) != expected:
        errors.append(f"manifest_count={len(manifest)} expected={expected}")
    if manifest_ids != mask_ids or manifest_ids != token_ids:
        errors.append("sample IDs or ordering differ across the three JSONL files")
    if len(set(manifest_ids)) != len(manifest_ids):
        errors.append("duplicate sample IDs")
    requested_quotas = {str(key): int(value) for key, value in config["data"]["quotas"].items()}
    actual_source_counts = dict(Counter(row["source"] for row in manifest))
    if actual_source_counts != requested_quotas:
        errors.append(f"source quotas mismatch: actual={actual_source_counts}, requested={requested_quotas}")
    pilot_ids, pilot_images = _load_pilot_exclusions(config)
    if set(manifest_ids).intersection(pilot_ids):
        errors.append("pilot sample overlap is nonzero")
    if bool(config["data"].get("exclude_pilot_images", True)) and {row["image_path"] for row in manifest}.intersection(pilot_images):
        errors.append("pilot image overlap is nonzero")

    mask_by_id = {row["sample_id"]: row for row in masks}
    expected_processor_hash = processor_configuration(config)["hash"]
    for token in tokens:
        sample_id = token["sample_id"]
        mask_row = mask_by_id[sample_id]
        decoded = decode_rle(mask_row["subject_mask_rle"])
        if decoded.shape != (int(mask_row["image_height"]), int(mask_row["image_width"])):
            errors.append(f"{sample_id}: RLE/image shape mismatch")
        if decoded.shape != (int(mask_row["height"]), int(mask_row["width"])):
            errors.append(f"{sample_id}: RLE/manifest shape mismatch")
        if not decoded.any():
            errors.append(f"{sample_id}: empty subject mask")
        count = int(token["visual_token_count"])
        fields = (
            "mask_token_coverage", "mask_token_contribution", "bbox_token_coverage",
            "bbox_token_contribution", "mask_token_labels", "bbox_token_labels",
            "token_confidence", "mask_token_confidence", "bbox_token_confidence",
        )
        for field in fields:
            values = np.asarray(token[field])
            if values.size != count:
                errors.append(f"{sample_id}: {field} length={values.size} expected={count}")
            if not np.isfinite(values).all():
                errors.append(f"{sample_id}: {field} contains NaN/Inf")
        mask_fg = set(token["mask_foreground_indices"])
        bbox_fg = set(token["bbox_foreground_indices"])
        background = set(token["reliable_background_indices"])
        for field in ("mask_foreground_indices", "bbox_foreground_indices", "reliable_background_indices"):
            indices = [int(value) for value in token[field]]
            if len(indices) != len(set(indices)) or any(value < 0 or value >= count for value in indices):
                errors.append(f"{sample_id}: {field} contains duplicate or out-of-range index")
        if not mask_fg:
            errors.append(f"{sample_id}: zero mask foreground tokens")
        if not bbox_fg:
            errors.append(f"{sample_id}: zero bbox foreground tokens")
        if mask_fg.intersection(background) or bbox_fg.intersection(background):
            errors.append(f"{sample_id}: foreground/background conflict")
        if not math.isclose(sum(token["mask_token_contribution"]), 1.0, rel_tol=0.0, abs_tol=2e-5):
            errors.append(f"{sample_id}: mask contribution does not sum to one")
        if not math.isclose(sum(token["bbox_token_contribution"]), 1.0, rel_tol=0.0, abs_tol=2e-5):
            errors.append(f"{sample_id}: bbox contribution does not sum to one")
        recomputed_mask_mass = float(sum(token["mask_token_contribution"][index] for index in mask_fg))
        recomputed_bbox_mass = float(sum(token["bbox_token_contribution"][index] for index in bbox_fg))
        if not math.isclose(recomputed_mask_mass, float(token["actual_mask_mass_coverage"]), rel_tol=0.0, abs_tol=2e-5):
            errors.append(f"{sample_id}: stored mask mass differs from selected contributions")
        if not math.isclose(recomputed_bbox_mass, float(token["actual_bbox_mass_coverage"]), rel_tol=0.0, abs_tol=2e-5):
            errors.append(f"{sample_id}: stored bbox mass differs from selected contributions")
        if token["actual_mask_mass_coverage"] + 1e-6 < float(config["supervision"]["adaptive_mass_target"]):
            errors.append(f"{sample_id}: mask adaptive target not reached")
        if token["actual_bbox_mass_coverage"] + 1e-6 < float(config["supervision"]["adaptive_mass_target"]):
            errors.append(f"{sample_id}: bbox adaptive target not reached")
        merge = int(config["qwen"]["spatial_merge_size"])
        expected_count = int(token["image_grid_thw"][0]) * (int(token["image_grid_thw"][1]) // merge) * (int(token["image_grid_thw"][2]) // merge)
        if count != expected_count:
            errors.append(f"{sample_id}: grid-derived visual token count mismatch")
        valid_labels = {
            int(config["supervision"]["foreground_label"]),
            int(config["supervision"]["background_label"]),
            int(config["supervision"]["ignore_label"]),
        }
        if not set(token["mask_token_labels"]).issubset(valid_labels) or not set(token["bbox_token_labels"]).issubset(valid_labels):
            errors.append(f"{sample_id}: invalid token label value")
        expected_mask_labels = make_labels(
            np.asarray(sorted(mask_fg)), np.asarray(sorted(background)), count,
            foreground_label=int(config["supervision"]["foreground_label"]),
            background_label=int(config["supervision"]["background_label"]),
            ignore_label=int(config["supervision"]["ignore_label"]),
        )
        expected_bbox_labels = make_labels(
            np.asarray(sorted(bbox_fg)), np.asarray(sorted(background)), count,
            foreground_label=int(config["supervision"]["foreground_label"]),
            background_label=int(config["supervision"]["background_label"]),
            ignore_label=int(config["supervision"]["ignore_label"]),
        )
        if not np.array_equal(expected_mask_labels, np.asarray(token["mask_token_labels"], dtype=np.int8)):
            errors.append(f"{sample_id}: mask labels do not match foreground/background indices")
        if not np.array_equal(expected_bbox_labels, np.asarray(token["bbox_token_labels"], dtype=np.int8)):
            errors.append(f"{sample_id}: bbox labels do not match foreground/background indices")
        if token["processor_config_hash"] != expected_processor_hash:
            errors.append(f"{sample_id}: processor config hash mismatch")

    fallback_ids = {row["sample_id"] for row in masks if row["fallback_to_bbox"]}
    recovered_failure_ids = {
        row["sample_id"] for row in failures
        if row.get("stage") == "sam_subject" and row.get("recovered_with_bbox")
    }
    if fallback_ids != recovered_failure_ids:
        errors.append("fallback samples and recovered SAM failure records differ")

    checks = {
        "manifest_exactly_1000": len(manifest) == expected,
        "sample_ids_identical_and_ordered": manifest_ids == mask_ids == token_ids,
        "sample_ids_unique": len(set(manifest_ids)) == len(manifest_ids),
        "source_quotas_exact": actual_source_counts == requested_quotas,
        "pilot_sample_overlap_zero": not bool(set(manifest_ids).intersection(pilot_ids)),
        "pilot_image_overlap_zero": not bool({row["image_path"] for row in manifest}.intersection(pilot_images)),
        "all_rle_decodable_and_sized": not any("RLE/" in item for item in errors),
        "all_masks_nonempty": not any("empty subject mask" in item for item in errors),
        "all_nonempty_masks_have_foreground": not any("zero mask foreground" in item for item in errors),
        "all_bboxes_have_foreground": not any("zero bbox foreground" in item for item in errors),
        "all_contributions_sum_to_one": not any("contribution does not sum" in item for item in errors),
        "all_token_arrays_match_visual_count": not any(" length=" in item for item in errors),
        "all_grid_counts_match": not any("grid-derived" in item for item in errors),
        "no_foreground_background_conflicts": not any("foreground/background conflict" in item for item in errors),
        "no_nan_or_inf": not any("NaN/Inf" in item for item in errors),
        "all_adaptive_targets_reached": not any("adaptive target not reached" in item for item in errors),
        "all_indices_unique_and_in_range": not any("duplicate or out-of-range" in item for item in errors),
        "stored_mass_matches_selected_contributions": not any("stored " in item and "mass differs" in item for item in errors),
        "labels_match_indices": not any("labels do not match" in item for item in errors),
        "processor_hash_consistent": not any("processor config hash mismatch" in item for item in errors),
        "fallback_records_consistent": fallback_ids == recovered_failure_ids,
    }
    result = {"passed": not errors and all(checks.values()), "checks": checks, "error_count": len(errors), "errors": errors[:200]}
    atomic_write_json(output_dir / "validation.json", result)
    if errors:
        raise RuntimeError(f"Integrity validation failed with {len(errors)} errors; see validation.json")
    return result


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _indices_overlay(image: np.ndarray, indices: list[int], grid: list[int], color: tuple[float, float, float]) -> np.ndarray:
    _, patch_height, patch_width = (int(value) for value in grid)
    token_height, token_width = patch_height // 2, patch_width // 2
    cells = np.zeros((token_height * token_width, 4), dtype=np.float32)
    cells[np.asarray(indices, dtype=np.int64), :3] = color
    cells[np.asarray(indices, dtype=np.int64), 3] = 0.60
    cells = cells.reshape(token_height, token_width, 4)
    overlay = np.asarray(
        Image.fromarray((cells * 255).astype(np.uint8), mode="RGBA").resize(
            (image.shape[1], image.shape[0]), Image.Resampling.NEAREST
        )
    ).astype(np.float32) / 255.0
    base = image.astype(np.float32) / 255.0
    alpha = overlay[..., 3:4]
    return np.clip(base * (1.0 - alpha) + overlay[..., :3] * alpha, 0.0, 1.0)


def render_visualization(mask_row: dict[str, Any], token_row: dict[str, Any], path: Path, dpi: int) -> None:
    image = np.asarray(Image.open(mask_row["image_path"]).convert("RGB"))
    subject = decode_rle(mask_row["subject_mask_rle"])
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=dpi, constrained_layout=True)
    axes[0, 0].imshow(image)
    for bbox in mask_row["bboxes"]:
        x1, y1, x2, y2 = bbox
        axes[0, 0].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="yellow", linewidth=2))
    axes[0, 0].set_title("原图与BBox（黄色）")
    mask_view = image.astype(np.float32) / 255.0
    mask_view[subject] = 0.40 * mask_view[subject] + 0.60 * np.array([0.05, 0.90, 0.25])
    axes[0, 1].imshow(mask_view)
    axes[0, 1].set_title(f"主体mask（绿色）| area={mask_row['mask_area_ratio']:.4f}")
    axes[0, 2].imshow(_indices_overlay(image, token_row["mask_foreground_indices"], token_row["image_grid_thw"], (1.0, 0.25, 0.05)))
    axes[0, 2].set_title(f"Mask前景（橙色）| n={len(token_row['mask_foreground_indices'])} | mass={token_row['actual_mask_mass_coverage']:.3f}")
    axes[1, 0].imshow(_indices_overlay(image, token_row["bbox_foreground_indices"], token_row["image_grid_thw"], (0.90, 0.05, 0.75)))
    axes[1, 0].set_title(f"BBox前景（紫色）| n={len(token_row['bbox_foreground_indices'])} | mass={token_row['actual_bbox_mass_coverage']:.3f}")
    axes[1, 1].imshow(_indices_overlay(image, token_row["reliable_background_indices"], token_row["image_grid_thw"], (0.05, 0.65, 1.0)))
    axes[1, 1].set_title(f"共享可靠背景（蓝色）| n={len(token_row['reliable_background_indices'])}")
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.0,
        1.0,
        f"sample_id: {mask_row['sample_id']}\nsource: {mask_row['source']}\n"
        f"Q: {mask_row['question']}\nA: {mask_row['answer']}\n\n"
        f"fallback: {mask_row['fallback_to_bbox']}\n"
        f"mask/BBox foreground IoU: {token_row['foreground_overlap_iou']:.3f}\n"
        f"grid: {token_row['image_grid_thw']}\nvisual tokens: {token_row['visual_token_count']}",
        va="top",
        wrap=True,
        fontsize=10,
    )
    for axis in axes.flat[:5]:
        axis.axis("off")
    fig.suptitle("Mask-GAZE 1K离线监督样本")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def choose_visualizations(mask_rows: list[dict[str, Any]], token_rows: list[dict[str, Any]], count: int) -> list[str]:
    mask_by_id = {row["sample_id"]: row for row in mask_rows}
    token_by_id = {row["sample_id"]: row for row in token_rows}
    chosen: list[str] = []
    seen: set[str] = set()

    def add(sample_ids: Iterable[str]) -> None:
        for sample_id in sample_ids:
            if sample_id not in seen and len(chosen) < count:
                chosen.append(sample_id)
                seen.add(sample_id)

    for source in sorted({row["source"] for row in mask_rows}):
        source_rows = sorted((row for row in mask_rows if row["source"] == source), key=lambda row: row["sample_id"])
        add(row["sample_id"] for row in source_rows[:5])
    add(row["sample_id"] for row in mask_rows if row["fallback_to_bbox"])
    add(row["sample_id"] for row in sorted(mask_rows, key=lambda row: row["mask_area_ratio"])[:10])
    add(row["sample_id"] for row in sorted(token_rows, key=lambda row: row["foreground_overlap_iou"]))
    add(row["sample_id"] for row in mask_rows)
    assert all(sample_id in mask_by_id and sample_id in token_by_id for sample_id in chosen)
    return chosen


def finalize(config: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["paths"]["output_dir"])
    manifest = list(read_jsonl(output_dir / MANIFEST_NAME))
    masks = list(read_jsonl(output_dir / MASKS_NAME))
    tokens = list(read_jsonl(output_dir / TOKENS_NAME))
    failures = list(read_jsonl(output_dir / FAILURES_NAME)) if (output_dir / FAILURES_NAME).exists() else []
    mask_by_id = {row["sample_id"]: row for row in masks}
    token_by_id = {row["sample_id"]: row for row in tokens}
    source_counts = dict(sorted(Counter(row["source"] for row in manifest).items()))
    fallback_count = sum(bool(row["fallback_to_bbox"]) for row in masks)
    summary = {
        "requested_sample_count": int(config["project"]["sample_count"]),
        "success_sample_count": len(tokens),
        "failure_sample_count": len(manifest) - len(tokens),
        "sam_mask_count": len(masks) - fallback_count,
        "bbox_fallback_count": fallback_count,
        "bbox_fallback_rate": fallback_count / len(masks) if masks else None,
        "source_counts": source_counts,
        "mask_foreground_token_count": distribution(len(row["mask_foreground_indices"]) for row in tokens),
        "bbox_foreground_token_count": distribution(len(row["bbox_foreground_indices"]) for row in tokens),
        "reliable_background_token_count": distribution(len(row["reliable_background_indices"]) for row in tokens),
        "actual_mask_mass_coverage": distribution(row["actual_mask_mass_coverage"] for row in tokens),
        "actual_bbox_mass_coverage": distribution(row["actual_bbox_mass_coverage"] for row in tokens),
        "foreground_overlap_iou": distribution(row["foreground_overlap_iou"] for row in tokens),
        "mask_foreground_covered_by_bbox_rate": distribution(
            row["mask_foreground_covered_by_bbox_rate"] for row in tokens if row["mask_foreground_covered_by_bbox_rate"] is not None
        ),
        "zero_mask_foreground_count": sum(not row["mask_foreground_indices"] for row in tokens),
        "zero_bbox_foreground_count": sum(not row["bbox_foreground_indices"] for row in tokens),
        "sam_average_seconds": float(mean(row["timing"]["sam_subject_seconds"] for row in masks)),
        "sam_total_seconds": float(sum(row["timing"]["sam_subject_seconds"] for row in masks)),
        "qwen_preprocess_mapping_average_seconds": float(mean(row["timing"]["qwen_preprocess_and_token_mapping_seconds"] for row in tokens)),
        "qwen_preprocess_mapping_total_seconds": float(sum(row["timing"]["qwen_preprocess_and_token_mapping_seconds"] for row in tokens)),
        "failure_record_count": len(failures),
        "processor": processor_configuration(config),
        "validation": validation,
        "seed": int(config["project"]["seed"]),
        "config_hash": config_hash(config),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "prohibitions_observed": {
            "vis_training_performed": False,
            "sam_local_parts_generated": False,
            "qwen_counterfactual_scoring_performed": False,
            "language_model_loaded": False,
        },
    }
    per_source_rows = []
    for source in sorted(source_counts):
        ids = [row["sample_id"] for row in manifest if row["source"] == source]
        source_masks = [mask_by_id[sample_id] for sample_id in ids]
        source_tokens = [token_by_id[sample_id] for sample_id in ids]
        source_fallback = sum(bool(row["fallback_to_bbox"]) for row in source_masks)
        per_source_rows.append(
            {
                "source": source,
                "count": len(ids),
                "sam_mask_count": len(ids) - source_fallback,
                "bbox_fallback_count": source_fallback,
                "bbox_fallback_rate": source_fallback / len(ids),
                "mask_foreground_mean": mean(len(row["mask_foreground_indices"]) for row in source_tokens),
                "mask_foreground_median": float(np.median([len(row["mask_foreground_indices"]) for row in source_tokens])),
                "bbox_foreground_mean": mean(len(row["bbox_foreground_indices"]) for row in source_tokens),
                "reliable_background_mean": mean(len(row["reliable_background_indices"]) for row in source_tokens),
                "actual_mask_mass_mean": mean(row["actual_mask_mass_coverage"] for row in source_tokens),
                "actual_bbox_mass_mean": mean(row["actual_bbox_mass_coverage"] for row in source_tokens),
                "foreground_overlap_iou_mean": mean(row["foreground_overlap_iou"] for row in source_tokens),
                "sam_seconds_mean": mean(row["timing"]["sam_subject_seconds"] for row in source_masks),
            }
        )
    _write_csv(output_dir / "summary_by_source.csv", list(per_source_rows[0]), per_source_rows)

    visual_ids = choose_visualizations(masks, tokens, int(config["visualization"]["count"]))
    for index, sample_id in enumerate(visual_ids, start=1):
        render_visualization(
            mask_by_id[sample_id],
            token_by_id[sample_id],
            output_dir / "visualizations" / f"{index:02d}_{sample_id}.png",
            int(config["visualization"]["dpi"]),
        )
    summary["visualization_count"] = len(visual_ids)
    summary["visualized_sample_ids"] = visual_ids
    atomic_write_json(output_dir / "summary.json", summary)

    report = [
        "# Mask-GAZE 1K离线预实验数据生成报告",
        "",
        "## 结论",
        "",
        f"- 完整数据：**{'已生成' if validation['passed'] and len(tokens) == 1000 else '未完整生成'}**，成功 `{len(tokens)}/1000`。",
        f"- 来源：GQA `{source_counts.get('gqa', 0)}`、Visual7W `{source_counts.get('v7w', 0)}`、Flickr30k `{source_counts.get('flickr30k', 0)}`。",
        f"- SAM主体mask `{len(masks) - fallback_count}`，BBox fallback `{fallback_count}`（`{100 * summary['bbox_fallback_rate']:.2f}%`）。",
        f"- Mask前景Token数：`{canonical_json(summary['mask_foreground_token_count'])}`。",
        f"- BBox前景Token数：`{canonical_json(summary['bbox_foreground_token_count'])}`。",
        f"- 共享可靠背景Token数：`{canonical_json(summary['reliable_background_token_count'])}`。",
        f"- BBox/Mask前景Token平均IoU：`{summary['foreground_overlap_iou']['mean']:.4f}`。",
        f"- 零Mask前景Token：`{summary['zero_mask_foreground_count']}`；零BBox前景Token：`{summary['zero_bbox_foreground_count']}`。",
        f"- 完整性检查：**{'全部通过' if validation['passed'] else '失败'}**。",
        "",
        "## 生成方法",
        "",
        "- 从原GAZE 20K train清单按固定种子42抽取600/200/200，并排除pilot 200中的样本ID和图片。",
        "- SAM仅使用BBox prompt和每框3个候选；未运行多点提示或局部部件生成。",
        "- Qwen只加载processor进行真实动态分辨率预处理与网格计算；未加载语言模型。",
        "- Mask和BBox均按Token contribution降序选择最少Token达到90%累计质量。",
        "- 两组共享mask coverage与bbox coverage均不超过0.01的可靠背景；其余Token为ignore，前景优先。",
        "- `token_confidence`是mask组置信度别名；同时额外保存`mask_token_confidence`和`bbox_token_confidence`供配对训练使用。",
        "",
        "## 时间与复现",
        "",
        f"- SAM平均/总耗时：`{summary['sam_average_seconds']:.4f}s` / `{summary['sam_total_seconds']:.2f}s`。",
        f"- Qwen预处理与Token映射平均/总耗时：`{summary['qwen_preprocess_mapping_average_seconds']:.4f}s` / `{summary['qwen_preprocess_mapping_total_seconds']:.2f}s`。",
        f"- 配置哈希：`{summary['config_hash']}`。",
        f"- Processor配置哈希：`{summary['processor']['hash']}`。",
        f"- Git commit：`{summary['git_commit']}`。",
        "- 本轮未训练VIS、未生成SAM局部部件、未运行Qwen反事实评分。",
        "",
        "## 文件",
        "",
        f"- `{(output_dir / MANIFEST_NAME).resolve()}`",
        f"- `{(output_dir / MASKS_NAME).resolve()}`",
        f"- `{(output_dir / TOKENS_NAME).resolve()}`",
        f"- `{(output_dir / FAILURES_NAME).resolve()}`",
    ]
    (output_dir / "generation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def ensure_empty_failures_file(config: dict[str, Any]) -> None:
    path = Path(config["paths"]["output_dir"]) / FAILURES_NAME
    if not path.exists():
        atomic_write_jsonl(path, [])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 1K offline Mask-GAZE paired supervision without training")
    parser.add_argument("--config", default="configs/mask_gaze_1k.yaml")
    parser.add_argument("--stage", choices=("manifest", "masks", "tokens", "validate", "finalize", "all"), default="all")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(int(config["project"]["seed"]), bool(config["runtime"]["deterministic_algorithms"]))
    if args.stage in ("manifest", "all"):
        build_manifest(config)
        ensure_empty_failures_file(config)
    if args.stage in ("masks", "all"):
        run_masks(config)
    if args.stage in ("tokens", "all"):
        run_tokens(config)
    if args.stage in ("validate", "all"):
        validation = validate_outputs(config)
        print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
    else:
        validation_path = Path(config["paths"]["output_dir"]) / "validation.json"
        validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else {"passed": False}
    if args.stage in ("finalize", "all"):
        if not validation.get("passed"):
            validation = validate_outputs(config)
        summary = finalize(config, validation)
        print(json.dumps({
            "success": summary["success_sample_count"],
            "fallback_rate": summary["bbox_fallback_rate"],
            "overlap_iou_mean": summary["foreground_overlap_iou"]["mean"],
            "validation_passed": summary["validation"]["passed"],
        }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
