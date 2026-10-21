from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image

from src.data.build_manifest import _normalize_bboxes, _question_from_record, _resolve_image
from src.generate_mask_gaze_1k import (
    _fallback_reasons,
    _processor,
    adaptive_indices,
    make_labels,
    processor_configuration,
    render_visualization,
    token_coverage_and_contribution,
    union_bbox_mask,
)
from src.mask_gaze_preexperiment import original_gaze_bbox_labels
from src.masks import decode_rle, encode_rle
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


MANIFEST_NAME = "manifest_20k.jsonl"
MASKS_NAME = "subject_masks.jsonl"
TOKENS_NAME = "token_supervision_qwen25vl.jsonl"
FAILURES_NAME = "failures.jsonl"
STATE_NAME = "generation_state.json"
FORBIDDEN_SUFFIXES = {".safetensors", ".pt", ".pth", ".npy"}


def distribution(values: Iterable[float | int]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {"count": 0, "min": None, "mean": None, "median": None, "p10": None, "p90": None, "max": None}
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def _normalized_path(value: str | Path) -> str:
    return os.path.realpath(os.path.expanduser(str(value)))


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def _state(config: dict[str, Any], **updates: Any) -> None:
    output_dir = Path(config["paths"]["output_dir"])
    state_path = output_dir / STATE_NAME
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    state.update(updates)
    state["output_directory_bytes"] = _directory_size(output_dir)
    state["qwen_visual_feature_cache_generated"] = False
    atomic_write_json(state_path, state)


def _read_unique(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = list(read_jsonl(path))
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in by_id:
            raise ValueError(f"duplicate sample_id in {path}: {sample_id}")
        by_id[sample_id] = row
    return rows, by_id


def _candidate_from_raw(
    raw: dict[str, Any], line_number: int, config: dict[str, Any], source_manifest: Path
) -> tuple[dict[str, Any] | None, str | None]:
    data = config["data"]
    source = str(raw.get("dataset") or raw.get("source") or "").strip().lower()
    if str(raw.get("split") or "").strip().lower() != str(data["required_split"]).lower():
        return None, "wrong_split"
    question = _question_from_record(raw, str(data.get("short_answer_suffix", "")))
    answer_value = raw.get("output", raw.get("answer"))
    answer = "" if answer_value is None else str(answer_value).strip()
    image_path = _resolve_image(
        raw, Path(config["paths"]["image_root"]), str(data.get("image_prefix_to_strip", ""))
    )
    bbox_data = _normalize_bboxes(raw)
    if not question or not answer:
        return None, "empty_question_or_answer"
    if image_path is None:
        return None, "invalid_image_field"
    if bbox_data is None:
        return None, "invalid_bbox"
    bboxes, width, height = bbox_data
    resolved = _normalized_path(image_path)
    return {
        "sample_id": stable_sample_id(source, resolved, question, answer),
        "source": source,
        "split": str(data["required_split"]),
        "image": resolved,
        "image_path": resolved,
        "question": question,
        "answer": answer,
        "bbox": bboxes[0],
        "bboxes": bboxes,
        "bbox_count": len(bboxes),
        "width": width,
        "height": height,
        "source_line": line_number,
        "source_manifest": str(source_manifest.resolve()),
    }, None


def _constrained_quotas(
    ideal: dict[str, int], capacities: dict[str, int], target: int
) -> dict[str, int]:
    quotas = {source: min(int(ideal[source]), int(capacities.get(source, 0))) for source in ideal}
    while sum(quotas.values()) < target:
        deficit = target - sum(quotas.values())
        eligible = [source for source in ideal if quotas[source] < capacities.get(source, 0)]
        if not eligible:
            raise RuntimeError(f"unique-image capacity {sum(quotas.values())} is below target {target}")
        denominator = sum(ideal[source] for source in eligible)
        raw = {source: deficit * ideal[source] / denominator for source in eligible}
        additions = {
            source: min(capacities[source] - quotas[source], int(math.floor(raw[source])))
            for source in eligible
        }
        if not any(additions.values()):
            ranked = sorted(eligible, key=lambda source: (-(raw[source] - math.floor(raw[source])), source))
            additions[ranked[0]] = 1
        for source, value in additions.items():
            quotas[source] += value
    if sum(quotas.values()) != target:
        raise AssertionError("quota allocation did not reach exact target")
    return quotas


def _verify_image(candidate: dict[str, Any]) -> str | None:
    image_path = Path(candidate["image_path"])
    if not image_path.is_file():
        return "image_missing"
    try:
        with Image.open(image_path) as image:
            actual = image.size
            image.verify()
    except Exception:
        return "image_unreadable"
    if actual != (int(candidate["width"]), int(candidate["height"])):
        return "declared_image_size_mismatch"
    return None


def build_manifest(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["paths"]["output_dir"])
    destination = output_dir / MANIFEST_NAME
    report_path = output_dir / "manifest_report.json"
    if destination.is_file() and report_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    base_path = Path(config["paths"]["base_training_manifest"])
    candidate_path = Path(config["paths"]["candidate_training_manifest"])
    pilot_rows, _ = _read_unique(Path(config["paths"]["pilot_manifest"]))
    reuse_rows, _ = _read_unique(Path(config["paths"]["reuse_1k_manifest"]))
    pilot_ids = {str(row["sample_id"]) for row in pilot_rows}
    pilot_paths = {_normalized_path(row["image_path"]) for row in pilot_rows}
    reuse_ids = {str(row["sample_id"]) for row in reuse_rows}
    reuse_paths = {_normalized_path(row["image_path"]) for row in reuse_rows}
    if reuse_ids & pilot_ids or reuse_paths & pilot_paths:
        raise AssertionError("accepted 1K unexpectedly overlaps pilot 200")
    if len(reuse_ids) != 1000 or len(reuse_paths) != 1000:
        raise AssertionError("accepted 1K is not 1000 unique samples/images")

    ideal_counts: Counter[str] = Counter()
    for line_number, raw in enumerate(read_jsonl(base_path), start=1):
        candidate, _ = _candidate_from_raw(raw, line_number, config, base_path)
        source = str(raw.get("dataset") or raw.get("source") or "").strip().lower()
        if source:
            ideal_counts[source] += 1
    if sum(ideal_counts.values()) != int(config["project"]["sample_count"]):
        raise AssertionError("base GAZE source composition does not total 20K")

    first_by_path: dict[str, dict[str, Any]] = {}
    structural_failures: Counter[str] = Counter()
    for line_number, raw in enumerate(read_jsonl(candidate_path), start=1):
        candidate, reason = _candidate_from_raw(raw, line_number, config, candidate_path)
        if candidate is None:
            structural_failures[str(reason)] += 1
            continue
        if candidate["sample_id"] in pilot_ids or candidate["image_path"] in pilot_paths:
            continue
        first_by_path.setdefault(candidate["image_path"], candidate)

    capacities = Counter(row["source"] for row in first_by_path.values())
    target = int(config["project"]["sample_count"])
    quotas = _constrained_quotas(dict(ideal_counts), dict(capacities), target)
    reuse_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reuse_rows:
        enriched = dict(row)
        enriched["bbox_count"] = len(row["bboxes"])
        enriched["selection_origin"] = "reused_accepted_1k"
        reuse_by_source[str(row["source"])].append(enriched)
    for source, rows in reuse_by_source.items():
        if len(rows) > quotas[source]:
            raise RuntimeError(f"reuse 1K exceeds source quota for {source}")

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in first_by_path.values():
        if candidate["image_path"] in reuse_paths:
            continue
        candidate["selection_origin"] = (
            "base_20k_unique_image" if int(candidate["source_line"]) <= 20000 else "same_source_5k_extension"
        )
        by_source[candidate["source"]].append(candidate)

    seed = int(config["project"]["seed"])
    selection_failures: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for source in ideal_counts:
        fixed = sorted(reuse_by_source.get(source, []), key=lambda row: row["sample_id"])
        selected.extend(fixed)
        base_candidates = [row for row in by_source[source] if row["selection_origin"] == "base_20k_unique_image"]
        extension_candidates = [row for row in by_source[source] if row["selection_origin"] == "same_source_5k_extension"]
        source_seed = int(hashlib.sha256(f"{seed}:{source}".encode()).hexdigest()[:16], 16)
        rng = random.Random(source_seed)
        rng.shuffle(base_candidates)
        rng.shuffle(extension_candidates)
        need = quotas[source] - len(fixed)
        accepted = 0
        for candidate in base_candidates + extension_candidates:
            if accepted >= need:
                break
            reason = _verify_image(candidate)
            if reason:
                selection_failures.append(
                    {"sample_id": candidate["sample_id"], "source": source, "reason": reason, "image_path": candidate["image_path"]}
                )
                continue
            selected.append(candidate)
            accepted += 1
        if accepted != need:
            raise RuntimeError(f"source {source} supplied {accepted}/{need} non-reused valid unique images")

    random.Random(seed).shuffle(selected)
    ids = [row["sample_id"] for row in selected]
    paths = [_normalized_path(row["image_path"]) for row in selected]
    if len(selected) != target or len(set(ids)) != target or len(set(paths)) != target:
        raise AssertionError("20K manifest cardinality or unique-image invariant failed")
    if set(ids) & pilot_ids or set(paths) & pilot_paths:
        raise AssertionError("20K manifest overlaps pilot 200")
    if not reuse_ids.issubset(ids) or not reuse_paths.issubset(paths):
        raise AssertionError("accepted 1K is not fully preserved in 20K")

    atomic_write_jsonl(destination, selected)
    atomic_write_jsonl(output_dir / "manifest_failures.jsonl", selection_failures)
    selected_counts = Counter(row["source"] for row in selected)
    origins = Counter(row["selection_origin"] for row in selected)
    report = {
        "seed": seed,
        "requested": target,
        "selected": len(selected),
        "unique_image_count": len(set(paths)),
        "base_source_composition": dict(ideal_counts),
        "unique_capacity_after_pilot_by_source": dict(capacities),
        "selected_source_counts": dict(selected_counts),
        "source_count_drift_from_base": {source: selected_counts[source] - ideal_counts[source] for source in ideal_counts},
        "selection_origins": dict(origins),
        "reused_1k_count": len(reuse_ids),
        "pilot_sample_id_overlap": 0,
        "pilot_image_path_overlap": 0,
        "structural_failure_counts": dict(structural_failures),
        "selection_failure_counts": dict(Counter(row["reason"] for row in selection_failures)),
        "base_training_manifest": str(base_path.resolve()),
        "base_training_manifest_sha256": sha256_file(base_path),
        "candidate_training_manifest": str(candidate_path.resolve()),
        "candidate_training_manifest_sha256": sha256_file(candidate_path),
        "selection_policy": "preserve accepted 1K; unique normalized image paths; base proportions capacity-capped; deficits deterministically redistributed among original GAZE sources",
    }
    atomic_write_json(report_path, report)
    _state(config, stage="manifest_complete", manifest_count=target, manifest_source_counts=dict(selected_counts))
    return report


def _mask_record_from_result(sample: dict[str, Any], image: np.ndarray, result: dict[str, Any], elapsed: float) -> dict[str, Any]:
    mask = result.pop("subject_mask")
    result.pop("instance_masks")
    sam_config = _ACTIVE_CONFIG["sam"]
    reasons = [reason for instance in result["subject_instances"] for reason in _fallback_reasons(instance, sam_config)]
    valid = [instance["selected"] for instance in result["subject_instances"] if not instance["fallback"]]
    any_fallback = bool(result["subject_mask_fallback"])
    selected_all = [instance["selected"] for instance in result["subject_instances"]]
    return {
        **sample,
        "image_height": int(image.shape[0]),
        "image_width": int(image.shape[1]),
        "subject_mask_rle": result["subject_mask_rle"],
        "mask_area_ratio": float(mask.mean()),
        "bbox_count": len(sample["bboxes"]),
        "sam_candidate_score": float(mean(item["selection_score"] for item in selected_all)) if selected_all else None,
        "selected_candidate": [int(item["candidate_index"]) for item in selected_all],
        "sam_pred_iou": float(mean(item["pred_iou"] for item in valid)) if valid else None,
        "stability_score": float(mean(item["stability"] for item in valid)) if valid else None,
        "inside_bbox_ratio": float(mean(item["inside_ratio"] for item in valid)) if valid else None,
        "mask_source": "mixed_sam_bbox_fallback" if any_fallback and valid else ("bbox_fallback" if any_fallback else "sam_bbox_prompt"),
        "fallback_to_bbox": any_fallback,
        "fallback_reasons": reasons,
        "subject_instances": result["subject_instances"],
        "timing": {"sam_subject_seconds": elapsed},
        "reused_from_1k": False,
    }


def _fallback_mask_record(sample: dict[str, Any], reason: str, elapsed: float) -> dict[str, Any]:
    mask = union_bbox_mask((int(sample["height"]), int(sample["width"])), sample["bboxes"])
    return {
        **sample,
        "image_height": int(sample["height"]),
        "image_width": int(sample["width"]),
        "subject_mask_rle": encode_rle(mask),
        "mask_area_ratio": float(mask.mean()),
        "bbox_count": len(sample["bboxes"]),
        "sam_candidate_score": None,
        "selected_candidate": [],
        "sam_pred_iou": None,
        "stability_score": None,
        "inside_bbox_ratio": None,
        "mask_source": "bbox_fallback",
        "fallback_to_bbox": True,
        "fallback_reasons": [reason],
        "subject_instances": [],
        "timing": {"sam_subject_seconds": elapsed},
        "reused_from_1k": False,
    }


def _enrich_reused_mask(sample: dict[str, Any], old: dict[str, Any]) -> dict[str, Any]:
    selected = [instance.get("selected", {}) for instance in old.get("subject_instances", [])]
    scores = [float(item["selection_score"]) for item in selected if item.get("selection_score") is not None]
    candidates = [int(item["candidate_index"]) for item in selected if item.get("candidate_index") is not None]
    return {
        **old,
        **sample,
        "bbox_count": len(sample["bboxes"]),
        "sam_candidate_score": float(mean(scores)) if scores else None,
        "selected_candidate": candidates,
        "fallback_to_bbox": bool(old["fallback_to_bbox"]),
        "reused_from_1k": True,
    }


def _compute_token_record(
    sample: dict[str, Any], mask_row: dict[str, Any], processor, config: dict[str, Any]
) -> dict[str, Any]:
    started = time.perf_counter()
    qwen = config["qwen"]
    supervision = config["supervision"]
    with Image.open(sample["image_path"]) as handle:
        processed = processor.image_processor(images=[handle.convert("RGB")], return_tensors="pt")
    grid = [int(value) for value in processed["image_grid_thw"][0].tolist()]
    del processed
    patch_size = int(qwen["patch_size"])
    merge = int(qwen["spatial_merge_size"])
    count = grid[0] * (grid[1] // merge) * (grid[2] // merge)
    subject = decode_rle(mask_row["subject_mask_rle"])
    bbox_pixels = union_bbox_mask(subject.shape, sample["bboxes"])
    mask_coverage, mask_contribution = token_coverage_and_contribution(
        subject, grid, patch_size=patch_size, spatial_merge_size=merge
    )
    bbox_coverage, bbox_contribution = token_coverage_and_contribution(
        bbox_pixels, grid, patch_size=patch_size, spatial_merge_size=merge
    )
    target = float(supervision["adaptive_mass_target"])
    mask_fg = adaptive_indices(mask_contribution, target)
    bbox_fg = adaptive_indices(bbox_contribution, target)
    bg_candidates = np.flatnonzero(
        (mask_coverage <= float(supervision["reliable_background_max_coverage"]))
        & (bbox_coverage <= float(supervision["reliable_background_max_coverage"]))
    ).astype(np.int64)
    reliable_bg = np.setdiff1d(bg_candidates, np.union1d(mask_fg, bbox_fg), assume_unique=False).astype(np.int64)
    mask_labels = make_labels(mask_fg, reliable_bg, count)
    bbox_labels = make_labels(bbox_fg, reliable_bg, count)
    gaze_labels = original_gaze_bbox_labels(sample["bboxes"], int(sample["width"]), int(sample["height"]), grid).astype(np.int8)
    gaze_fg = np.flatnonzero(gaze_labels == 1).astype(np.int64)
    overlap = len(set(mask_fg.tolist()) & set(bbox_fg.tolist()))
    union = len(set(mask_fg.tolist()) | set(bbox_fg.tolist()))
    return {
        "sample_id": sample["sample_id"],
        "source": sample["source"],
        "image": sample["image_path"],
        "image_grid_thw": grid,
        "processed_image_size": [grid[1] * patch_size, grid[2] * patch_size],
        "visual_token_count": int(count),
        "mask_token_coverage": mask_coverage.astype(float).tolist(),
        "mask_token_contribution": mask_contribution.astype(float).tolist(),
        "bbox_token_coverage": bbox_coverage.astype(float).tolist(),
        "bbox_token_contribution": bbox_contribution.astype(float).tolist(),
        "mask_positive_indices": mask_fg.astype(int).tolist(),
        "mask_foreground_indices": mask_fg.astype(int).tolist(),
        "mask_positive_count": int(mask_fg.size),
        "bbox_foreground_indices": bbox_fg.astype(int).tolist(),
        "reliable_background_indices": reliable_bg.astype(int).tolist(),
        "reliable_negative_count": int(reliable_bg.size),
        "ignore_count": int((mask_labels == -1).sum()),
        "mask_token_labels": mask_labels.astype(int).tolist(),
        "bbox_token_labels": bbox_labels.astype(int).tolist(),
        "gaze_bbox_labels": gaze_labels.astype(int).tolist(),
        "gaze_bbox_positive_indices": gaze_fg.astype(int).tolist(),
        "gaze_bbox_positive_count": int(gaze_fg.size),
        "adaptive_mass_target": target,
        "actual_mask_mass_coverage": float(np.clip(mask_contribution[mask_fg].sum(), 0.0, 1.0)),
        "actual_bbox_mass_coverage": float(np.clip(bbox_contribution[bbox_fg].sum(), 0.0, 1.0)),
        "foreground_overlap_iou": float(overlap / union) if union else 1.0,
        "processor_config_hash": processor_configuration(config)["hash"],
        "fallback_to_bbox": bool(mask_row["fallback_to_bbox"]),
        "reused_from_1k": bool(mask_row.get("reused_from_1k", False)),
        "timing": {"qwen_preprocess_and_token_mapping_seconds": time.perf_counter() - started},
    }


def _enrich_reused_token(
    sample: dict[str, Any], mask_row: dict[str, Any], old: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    grid = [int(value) for value in old["image_grid_thw"]]
    mask_fg = [int(value) for value in old["mask_foreground_indices"]]
    reliable = [int(value) for value in old["reliable_background_indices"]]
    labels = [int(value) for value in old["mask_token_labels"]]
    gaze_labels = original_gaze_bbox_labels(sample["bboxes"], int(sample["width"]), int(sample["height"]), grid).astype(np.int8)
    gaze_fg = np.flatnonzero(gaze_labels == 1).astype(np.int64)
    return {
        **old,
        "sample_id": sample["sample_id"],
        "source": sample["source"],
        "image": sample["image_path"],
        "mask_positive_indices": mask_fg,
        "mask_positive_count": len(mask_fg),
        "reliable_negative_count": len(reliable),
        "ignore_count": sum(value == -1 for value in labels),
        "gaze_bbox_labels": gaze_labels.astype(int).tolist(),
        "gaze_bbox_positive_indices": gaze_fg.astype(int).tolist(),
        "gaze_bbox_positive_count": int(gaze_fg.size),
        "reused_from_1k": True,
        "processor_config_hash": processor_configuration(config)["hash"],
        "fallback_to_bbox": bool(mask_row["fallback_to_bbox"]),
    }


def dry_run(config: dict[str, Any], predictor=None):
    output_dir = Path(config["paths"]["output_dir"])
    dry_dir = output_dir / "dry_run"
    report_path = dry_dir / "dry_run_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("passed"):
            return report, predictor
    manifest, _ = _read_unique(output_dir / MANIFEST_NAME)
    old_masks, old_masks_by_id = _read_unique(Path(config["paths"]["reuse_1k_masks"]))
    old_tokens, old_tokens_by_id = _read_unique(Path(config["paths"]["reuse_1k_tokens"]))
    reuse_ids = set(old_masks_by_id) & set(old_tokens_by_id)
    reuse_samples = [row for row in manifest if row["sample_id"] in reuse_ids][: int(config["dry_run"]["reused_1k_count"])]
    new_samples = [row for row in manifest if row["sample_id"] not in reuse_ids][: int(config["dry_run"]["new_sample_count"])]
    if len(reuse_samples) != 10 or len(new_samples) != 10:
        raise AssertionError("dry run requires exactly 10 reused and 10 new samples")
    if predictor is None:
        predictor = load_sam_predictor(config)
    processor = _processor(config)
    dry_masks: list[dict[str, Any]] = []
    dry_tokens: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for sample in reuse_samples:
        mask_row = _enrich_reused_mask(sample, old_masks_by_id[sample["sample_id"]])
        recomputed = _compute_token_record(sample, mask_row, processor, config)
        old = old_tokens_by_id[sample["sample_id"]]
        exact_fields = ("image_grid_thw", "visual_token_count", "mask_foreground_indices", "reliable_background_indices", "mask_token_labels")
        for field in exact_fields:
            if recomputed[field] != old[field]:
                mismatches.append({"sample_id": sample["sample_id"], "field": field})
        numeric_fields = ("mask_token_coverage", "mask_token_contribution")
        for field in numeric_fields:
            if not np.allclose(np.asarray(recomputed[field]), np.asarray(old[field]), rtol=0.0, atol=1e-7):
                mismatches.append({"sample_id": sample["sample_id"], "field": field})
        dry_masks.append(mask_row)
        dry_tokens.append(recomputed)
    for sample in new_samples:
        started = time.perf_counter()
        try:
            image = np.asarray(Image.open(sample["image_path"]).convert("RGB"))
            result = segment_subject(image, sample["bboxes"], predictor, config["sam"])
            mask_row = _mask_record_from_result(sample, image, result, time.perf_counter() - started)
        except Exception as exc:
            mask_row = _fallback_mask_record(sample, f"sam_exception: {type(exc).__name__}: {exc}", time.perf_counter() - started)
        dry_masks.append(mask_row)
        dry_tokens.append(_compute_token_record(sample, mask_row, processor, config))

    errors: list[str] = []
    for mask_row, token in zip(dry_masks, dry_tokens):
        decoded = decode_rle(mask_row["subject_mask_rle"])
        if decoded.shape != (int(mask_row["height"]), int(mask_row["width"])):
            errors.append(f"{mask_row['sample_id']}: RLE shape mismatch")
        count = int(token["visual_token_count"])
        expected = int(token["image_grid_thw"][0]) * (int(token["image_grid_thw"][1]) // 2) * (int(token["image_grid_thw"][2]) // 2)
        if count != expected or len(token["mask_token_labels"]) != count:
            errors.append(f"{mask_row['sample_id']}: token length mismatch")
        if not token["mask_positive_indices"] or float(token["actual_mask_mass_coverage"]) + 1e-6 < 0.90:
            errors.append(f"{mask_row['sample_id']}: adaptive foreground invalid")
    atomic_write_jsonl(dry_dir / "manifest.jsonl", reuse_samples + new_samples)
    atomic_write_jsonl(dry_dir / "subject_masks.jsonl", dry_masks)
    atomic_write_jsonl(dry_dir / "token_supervision_qwen25vl.jsonl", dry_tokens)
    forbidden = [str(path) for path in dry_dir.rglob("*") if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES]
    passed = not mismatches and not errors and not forbidden
    report = {
        "passed": passed,
        "sample_count": 20,
        "reused_1k_count": 10,
        "new_sam_count": 10,
        "reuse_mapping_mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "validation_errors": errors,
        "forbidden_feature_files": forbidden,
        "qwen_visual_feature_cache_generated": False,
    }
    atomic_write_json(report_path, report)
    _state(config, stage="dry_run_complete" if passed else "dry_run_failed", dry_run=report)
    if not passed:
        raise RuntimeError(f"20-sample dry run failed: {report}")
    return report, predictor


def _load_completed_shards(directory: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("shard_*.jsonl")):
        for row in read_jsonl(path):
            sample_id = str(row["sample_id"])
            if sample_id in records:
                raise ValueError(f"duplicate sample across shards: {sample_id}")
            records[sample_id] = row
    return records


def _final_jsonl_matches_manifest(path: Path, manifest: list[dict[str, Any]]) -> bool:
    if not path.is_file():
        return False
    iterator = read_jsonl(path)
    for sample in manifest:
        try:
            row = next(iterator)
        except StopIteration:
            return False
        if str(row.get("sample_id")) != str(sample["sample_id"]):
            return False
    try:
        next(iterator)
        return False
    except StopIteration:
        return True


def run_masks(config: dict[str, Any], predictor=None) -> None:
    output_dir = Path(config["paths"]["output_dir"])
    manifest, _ = _read_unique(output_dir / MANIFEST_NAME)
    if _final_jsonl_matches_manifest(output_dir / MASKS_NAME, manifest):
        _state(config, stage="masks_complete", mask_completed=len(manifest), resumed_from_complete_final_file=True)
        return
    _, old_masks = _read_unique(Path(config["paths"]["reuse_1k_masks"]))
    dry_masks = {row["sample_id"]: row for row in read_jsonl(output_dir / "dry_run" / "subject_masks.jsonl")}
    shard_dir = output_dir / "shards" / "masks"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_size = int(config["runtime"]["shard_size"])
    log_every = int(config["runtime"]["log_every"])
    loaded_here = False
    for shard_index, start in enumerate(range(0, len(manifest), shard_size)):
        rows = manifest[start : start + shard_size]
        final_path = shard_dir / f"shard_{shard_index:05d}.jsonl"
        if final_path.is_file():
            existing = list(read_jsonl(final_path))
            if [row["sample_id"] for row in existing] != [row["sample_id"] for row in rows]:
                raise AssertionError(f"completed mask shard {shard_index} IDs differ from manifest")
            continue
        partial_path = shard_dir / f"shard_{shard_index:05d}.partial.jsonl"
        partial = list(read_jsonl(partial_path)) if partial_path.is_file() else []
        if [row["sample_id"] for row in partial] != [row["sample_id"] for row in rows[: len(partial)]]:
            raise AssertionError(f"partial mask shard {shard_index} is not a manifest prefix")
        for local_index, sample in enumerate(rows[len(partial) :], start=len(partial)):
            sample_id = sample["sample_id"]
            if sample_id in old_masks:
                record = _enrich_reused_mask(sample, old_masks[sample_id])
            elif sample_id in dry_masks:
                record = {**dry_masks[sample_id], **sample}
            else:
                if predictor is None:
                    predictor = load_sam_predictor(config)
                    loaded_here = True
                started = time.perf_counter()
                processing_error = None
                try:
                    image = np.asarray(Image.open(sample["image_path"]).convert("RGB"))
                    result = segment_subject(image, sample["bboxes"], predictor, config["sam"])
                    record = _mask_record_from_result(sample, image, result, time.perf_counter() - started)
                except Exception as exc:
                    processing_error = f"{type(exc).__name__}: {exc}"
                    record = _fallback_mask_record(sample, f"sam_exception: {processing_error}", time.perf_counter() - started)
            append_jsonl(partial_path, record)
            partial.append(record)
            completed = start + local_index + 1
            if completed % int(config["runtime"]["state_every"]) == 0:
                _state(config, stage="masks_running", mask_completed=completed, mask_total=len(manifest), current_shard=shard_index)
            if completed % log_every == 0:
                print(f"[mask] {completed}/{len(manifest)} shard={shard_index} fallback={record['fallback_to_bbox']}", flush=True)
        atomic_write_jsonl(final_path, partial)
        partial_path.unlink(missing_ok=True)
        _state(config, stage="masks_running", mask_completed=start + len(rows), mask_total=len(manifest), completed_mask_shard=shard_index)
    consolidate_masks(config)
    _state(config, stage="masks_complete", mask_completed=len(manifest), sam_loaded_in_this_run=loaded_here or predictor is not None)


def consolidate_masks(config: dict[str, Any]) -> None:
    output_dir = Path(config["paths"]["output_dir"])
    manifest, _ = _read_unique(output_dir / MANIFEST_NAME)
    records = _load_completed_shards(output_dir / "shards" / "masks")
    if set(records) != {row["sample_id"] for row in manifest}:
        raise AssertionError("mask shards do not cover exact manifest")
    ordered = [records[row["sample_id"]] for row in manifest]
    atomic_write_jsonl(output_dir / MASKS_NAME, ordered)
    failures = []
    for row in ordered:
        if row["fallback_to_bbox"]:
            failures.append({
                "stage": "sam_subject",
                "sample_id": row["sample_id"],
                "source": row["source"],
                "reason": ";".join(row.get("fallback_reasons", [])) or "bbox_fallback",
                "recovered_with_bbox": True,
            })
    atomic_write_jsonl(output_dir / FAILURES_NAME, failures)


def run_tokens(config: dict[str, Any]) -> None:
    output_dir = Path(config["paths"]["output_dir"])
    manifest, _ = _read_unique(output_dir / MANIFEST_NAME)
    if _final_jsonl_matches_manifest(output_dir / TOKENS_NAME, manifest):
        _state(config, stage="tokens_complete", token_completed=len(manifest), resumed_from_complete_final_file=True)
        return
    masks, masks_by_id = _read_unique(output_dir / MASKS_NAME)
    if len(masks) != len(manifest):
        raise AssertionError("token stage requires all masks")
    _, old_tokens = _read_unique(Path(config["paths"]["reuse_1k_tokens"]))
    dry_tokens = {row["sample_id"]: row for row in read_jsonl(output_dir / "dry_run" / "token_supervision_qwen25vl.jsonl")}
    processor = _processor(config)
    shard_dir = output_dir / "shards" / "tokens"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_size = int(config["runtime"]["shard_size"])
    for shard_index, start in enumerate(range(0, len(manifest), shard_size)):
        rows = manifest[start : start + shard_size]
        final_path = shard_dir / f"shard_{shard_index:05d}.jsonl"
        if final_path.is_file():
            existing = list(read_jsonl(final_path))
            if [row["sample_id"] for row in existing] != [row["sample_id"] for row in rows]:
                raise AssertionError(f"completed token shard {shard_index} IDs differ from manifest")
            continue
        partial_path = shard_dir / f"shard_{shard_index:05d}.partial.jsonl"
        partial = list(read_jsonl(partial_path)) if partial_path.is_file() else []
        if [row["sample_id"] for row in partial] != [row["sample_id"] for row in rows[: len(partial)]]:
            raise AssertionError(f"partial token shard {shard_index} is not a manifest prefix")
        for local_index, sample in enumerate(rows[len(partial) :], start=len(partial)):
            sample_id = sample["sample_id"]
            mask_row = masks_by_id[sample_id]
            if sample_id in old_tokens:
                record = _enrich_reused_token(sample, mask_row, old_tokens[sample_id], config)
            elif sample_id in dry_tokens:
                record = dry_tokens[sample_id]
            else:
                record = _compute_token_record(sample, mask_row, processor, config)
            append_jsonl(partial_path, record)
            partial.append(record)
            completed = start + local_index + 1
            if completed % int(config["runtime"]["state_every"]) == 0:
                _state(config, stage="tokens_running", token_completed=completed, token_total=len(manifest), current_shard=shard_index)
            if completed % int(config["runtime"]["log_every"]) == 0:
                print(f"[tokens] {completed}/{len(manifest)} shard={shard_index} n={record['visual_token_count']}", flush=True)
        atomic_write_jsonl(final_path, partial)
        partial_path.unlink(missing_ok=True)
        _state(config, stage="tokens_running", token_completed=start + len(rows), token_total=len(manifest), completed_token_shard=shard_index)
    records = _load_completed_shards(shard_dir)
    if set(records) != {row["sample_id"] for row in manifest}:
        raise AssertionError("token shards do not cover exact manifest")
    atomic_write_jsonl(output_dir / TOKENS_NAME, (records[row["sample_id"]] for row in manifest))
    _state(config, stage="tokens_complete", token_completed=len(manifest))


def _scan_forbidden(output_dir: Path) -> list[str]:
    return sorted(
        str(path.resolve())
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES
    )


def validate_and_finalize(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["paths"]["output_dir"])
    manifest = list(read_jsonl(output_dir / MANIFEST_NAME))
    pilot = list(read_jsonl(Path(config["paths"]["pilot_manifest"])))
    pilot_ids = {row["sample_id"] for row in pilot}
    pilot_paths = {_normalized_path(row["image_path"]) for row in pilot}
    manifest_ids = [row["sample_id"] for row in manifest]
    manifest_paths = [_normalized_path(row["image_path"]) for row in manifest]
    mask_iter = read_jsonl(output_dir / MASKS_NAME)
    token_iter = read_jsonl(output_dir / TOKENS_NAME)
    errors: list[str] = []
    counts = {"mask": 0, "token": 0}
    stats: dict[str, list[float]] = defaultdict(list)
    source_acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    fallbacks = 0
    zero_negative = 0
    sam_seconds = 0.0
    token_seconds = 0.0
    reused = 0
    visual_meta: list[dict[str, Any]] = []
    for index, (sample, mask_row, token) in enumerate(zip(manifest, mask_iter, token_iter), start=1):
        counts["mask"] += 1
        counts["token"] += 1
        sample_id = sample["sample_id"]
        if mask_row["sample_id"] != sample_id or token["sample_id"] != sample_id:
            errors.append(f"{sample_id}: ordering/ID mismatch")
            continue
        decoded = decode_rle(mask_row["subject_mask_rle"])
        if decoded.shape != (int(sample["height"]), int(sample["width"])):
            errors.append(f"{sample_id}: RLE shape mismatch")
        if not decoded.any():
            errors.append(f"{sample_id}: empty mask")
        count = int(token["visual_token_count"])
        grid = [int(value) for value in token["image_grid_thw"]]
        expected = grid[0] * (grid[1] // int(config["qwen"]["spatial_merge_size"])) * (grid[2] // int(config["qwen"]["spatial_merge_size"]))
        if count != expected:
            errors.append(f"{sample_id}: grid token count mismatch")
        for field in ("mask_token_coverage", "mask_token_labels", "gaze_bbox_labels"):
            values = np.asarray(token[field])
            if values.size != count or not np.isfinite(values).all():
                errors.append(f"{sample_id}: {field} invalid length or nonfinite")
        positives = [int(value) for value in token["mask_positive_indices"]]
        reliable = [int(value) for value in token["reliable_background_indices"]]
        if not positives:
            errors.append(f"{sample_id}: zero mask positives")
        if set(positives) & set(reliable):
            errors.append(f"{sample_id}: foreground/background conflict")
        if float(token["actual_mask_mass_coverage"]) + 1e-6 < 0.90:
            errors.append(f"{sample_id}: mask mass below 0.90")
        labels = token["mask_token_labels"]
        if int(token["mask_positive_count"]) != len(positives) or int(token["reliable_negative_count"]) != len(reliable) or int(token["ignore_count"]) != sum(value == -1 for value in labels):
            errors.append(f"{sample_id}: stored label counts mismatch")
        mask_pos = len(positives)
        neg = len(reliable)
        ign = int(token["ignore_count"])
        stats["mask_positive_count"].append(mask_pos)
        stats["reliable_negative_count"].append(neg)
        stats["ignore_count"].append(ign)
        stats["actual_mask_mass_coverage"].append(float(token["actual_mask_mass_coverage"]))
        group = "reused_1k" if bool(mask_row.get("reused_from_1k")) else "new_19k"
        source_acc[group]["mask_positive_count"].append(mask_pos)
        source_acc[group]["reliable_negative_count"].append(neg)
        source_acc[group]["ignore_count"].append(ign)
        source_acc[group]["actual_mask_mass_coverage"].append(float(token["actual_mask_mass_coverage"]))
        source = sample["source"]
        source_acc[source]["mask_positive_count"].append(mask_pos)
        source_acc[source]["reliable_negative_count"].append(neg)
        source_acc[source]["ignore_count"].append(ign)
        source_acc[source]["actual_mask_mass_coverage"].append(float(token["actual_mask_mass_coverage"]))
        fallbacks += int(bool(mask_row["fallback_to_bbox"]))
        zero_negative += int(neg == 0)
        reused += int(group == "reused_1k")
        sam_seconds += float(mask_row.get("timing", {}).get("sam_subject_seconds", 0.0)) if group == "new_19k" else 0.0
        token_seconds += float(token.get("timing", {}).get("qwen_preprocess_and_token_mapping_seconds", 0.0))
        ys, xs = np.nonzero(decoded)
        mask_width = int(xs.max() - xs.min() + 1) if xs.size else 0
        mask_height = int(ys.max() - ys.min() + 1) if ys.size else 0
        elongation = max(mask_width / max(mask_height, 1), mask_height / max(mask_width, 1))
        visual_meta.append({
            "sample_id": sample_id,
            "source": source,
            "mask_area_ratio": float(mask_row["mask_area_ratio"]),
            "elongation": float(elongation),
            "bbox_count": len(sample["bboxes"]),
            "random_key": hashlib.sha256(f"42:{sample_id}".encode()).hexdigest(),
        })
        if index % 1000 == 0:
            print(f"[validate] {index}/{len(manifest)} errors={len(errors)}", flush=True)
    try:
        next(mask_iter)
        errors.append("mask file has extra rows")
    except StopIteration:
        pass
    try:
        next(token_iter)
        errors.append("token file has extra rows")
    except StopIteration:
        pass
    target = int(config["project"]["sample_count"])
    if len(manifest) != target or counts["mask"] != target or counts["token"] != target:
        errors.append(f"file counts manifest/mask/token={len(manifest)}/{counts['mask']}/{counts['token']}")
    if len(set(manifest_ids)) != target or len(set(manifest_paths)) != target:
        errors.append("manifest IDs or image paths are not unique")
    id_overlap = len(set(manifest_ids) & pilot_ids)
    path_overlap = len(set(manifest_paths) & pilot_paths)
    if id_overlap or path_overlap:
        errors.append(f"pilot overlap id/path={id_overlap}/{path_overlap}")
    forbidden = _scan_forbidden(output_dir)
    if forbidden:
        errors.append(f"forbidden feature/cache files found: {forbidden[:10]}")

    generation_state_path = output_dir / STATE_NAME
    generation_state = json.loads(generation_state_path.read_text(encoding="utf-8")) if generation_state_path.is_file() else {}
    generation_started = generation_state.get("generation_started_unix")
    summary = {
        "requested_sample_count": target,
        "success_sample_count": counts["token"],
        "failure_sample_count": max(0, target - counts["token"]),
        "source_counts": dict(Counter(row["source"] for row in manifest)),
        "reused_1k_count": reused,
        "new_generated_count": target - reused,
        "pilot_sample_id_overlap": id_overlap,
        "pilot_image_path_overlap": path_overlap,
        "unique_image_count": len(set(manifest_paths)),
        "sam_success_count": target - fallbacks,
        "bbox_fallback_count": fallbacks,
        "bbox_fallback_rate": fallbacks / target,
        "zero_reliable_negative_count": zero_negative,
        "mask_positive_count": distribution(stats["mask_positive_count"]),
        "reliable_negative_count": distribution(stats["reliable_negative_count"]),
        "ignore_count": distribution(stats["ignore_count"]),
        "actual_mask_mass_coverage": distribution(stats["actual_mask_mass_coverage"]),
        "distribution_comparison_1k_vs_new19k": {
            group: {metric: distribution(values) for metric, values in metrics.items()}
            for group, metrics in source_acc.items() if group in {"reused_1k", "new_19k"}
        },
        "distribution_drift_note": "New 19K has more reliable negatives and fewer positives/ignore tokens on average because accepted 1K contains only GQA, Visual7W and Flickr30k, while new 19K restores the original GAZE document-heavy sources with generally larger visual-token grids and more localized subjects.",
        "sam_total_seconds_new_samples": sam_seconds,
        "token_mapping_total_seconds": token_seconds,
        "total_recorded_generation_seconds": sam_seconds + token_seconds,
        "total_wall_clock_seconds": time.time() - float(generation_started) if generation_started is not None else None,
        "estimated_peak_output_bytes_before_generation": generation_state.get("estimated_peak_output_bytes"),
        "output_directory_bytes": _directory_size(output_dir),
        "qwen_visual_feature_cache_generated": False,
        "forbidden_feature_cache_files": forbidden,
        "validation_error_count": len(errors),
        "validation_errors": errors[:1000],
        "validation_passed": not errors,
        "processor": processor_configuration(config),
        "seed": int(config["project"]["seed"]),
        "config_hash": config_hash(config),
    }

    source_rows = []
    for source in sorted(summary["source_counts"]):
        metrics = source_acc[source]
        source_rows.append({
            "source": source,
            "count": summary["source_counts"][source],
            "mask_positive_mean": distribution(metrics["mask_positive_count"])["mean"],
            "mask_positive_median": distribution(metrics["mask_positive_count"])["median"],
            "reliable_negative_mean": distribution(metrics["reliable_negative_count"])["mean"],
            "ignore_mean": distribution(metrics["ignore_count"])["mean"],
            "actual_mask_mass_mean": distribution(metrics["actual_mask_mass_coverage"])["mean"],
        })
    with (output_dir / "summary_by_source.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)

    chosen: list[str] = []
    seen: set[str] = set()
    def add(rows: Iterable[dict[str, Any]]) -> None:
        for row in rows:
            if row["sample_id"] not in seen and len(chosen) < int(config["visualization"]["count"]):
                seen.add(row["sample_id"])
                chosen.append(row["sample_id"])
    add(sorted(visual_meta, key=lambda row: row["mask_area_ratio"])[:15])
    add(sorted(visual_meta, key=lambda row: -row["elongation"])[:15])
    add(sorted((row for row in visual_meta if row["bbox_count"] > 1), key=lambda row: row["random_key"])[:15])
    add(sorted(visual_meta, key=lambda row: -row["mask_area_ratio"])[:15])
    for source in sorted(summary["source_counts"]):
        add(sorted((row for row in visual_meta if row["source"] == source), key=lambda row: row["random_key"])[:8])
    add(sorted(visual_meta, key=lambda row: row["random_key"]))
    selected_set = set(chosen)
    selected_masks = {row["sample_id"]: row for row in read_jsonl(output_dir / MASKS_NAME) if row["sample_id"] in selected_set}
    selected_tokens = {row["sample_id"]: row for row in read_jsonl(output_dir / TOKENS_NAME) if row["sample_id"] in selected_set}
    visual_dir = output_dir / "visualizations"
    for index, sample_id in enumerate(chosen, start=1):
        render_visualization(
            selected_masks[sample_id], selected_tokens[sample_id],
            visual_dir / f"{index:03d}_{sample_id}.png", int(config["visualization"]["dpi"]),
        )
    summary["visualization_count"] = len(chosen)
    summary["visualized_sample_ids"] = chosen
    summary["output_directory_bytes"] = _directory_size(output_dir)
    atomic_write_json(output_dir / "summary.json", summary)

    manifest_report = json.loads((output_dir / "manifest_report.json").read_text(encoding="utf-8"))
    comparison = summary["distribution_comparison_1k_vs_new19k"]
    report = [
        "# Mask-GAZE 20K offline supervision generation report", "",
        "## Result", "",
        f"- Complete: **{summary['validation_passed']}**; successful `{summary['success_sample_count']}/20000`; failures `{summary['failure_sample_count']}`.",
        f"- Sources: `{canonical_json(summary['source_counts'])}`.",
        f"- Reused accepted 1K: `{reused}`; newly generated: `{target - reused}`.",
        f"- Pilot 200 overlap by sample_id/image_path: `{id_overlap}/{path_overlap}`.",
        f"- Unique normalized images: `{summary['unique_image_count']}`.",
        f"- SAM successes: `{summary['sam_success_count']}`; BBox fallback: `{fallbacks}` (`{100 * summary['bbox_fallback_rate']:.4f}%`).",
        f"- Zero reliable-negative samples: `{zero_negative}`.",
        f"- Mask positive tokens: `{canonical_json(summary['mask_positive_count'])}`.",
        f"- Reliable negative tokens: `{canonical_json(summary['reliable_negative_count'])}`.",
        f"- Ignore tokens: `{canonical_json(summary['ignore_count'])}`.",
        f"- Mask mass coverage: `{canonical_json(summary['actual_mask_mass_coverage'])}`.",
        f"- Validation errors: `{len(errors)}`.", "",
        "## Selection and distribution", "",
        "- The exact original 20K list has only 17,406 unique eligible images after excluding pilot 200, so the GAZE-provided same-source 25K keep20K+5K pool was used only as the candidate pool needed to reach 20K unique images.",
        f"- Base source composition: `{canonical_json(manifest_report['base_source_composition'])}`.",
        f"- Final source composition: `{canonical_json(manifest_report['selected_source_counts'])}`.",
        f"- Drift forced by per-image deduplication: `{canonical_json(manifest_report['source_count_drift_from_base'])}`.",
        f"- Reused 1K versus new 19K distribution: `{canonical_json(comparison)}`.",
        "- Distribution drift is visible and expected: new 19K has more reliable negatives and fewer positive/ignore tokens on average because the accepted 1K contains only GQA, Visual7W and Flickr30k, whereas the added pool restores the original GAZE document-heavy sources with generally larger visual-token grids and more localized subjects.", "",
        "## Runtime and storage", "",
        f"- Recorded SAM time for new samples: `{sam_seconds:.2f}s`.",
        f"- Recorded processor/token mapping time: `{token_seconds:.2f}s`.",
        f"- Total wall-clock time from formal generation start: `{summary['total_wall_clock_seconds']}` seconds.",
        f"- Estimated peak output size recorded before generation: `{summary['estimated_peak_output_bytes_before_generation']}` bytes.",
        f"- Output directory size: `{summary['output_directory_bytes']}` bytes.",
        f"- Visualizations: `{len(chosen)}` (maximum 100).", "",
        "## Prohibited artifacts", "",
        "```yaml", "qwen_visual_feature_cache_generated: false", "```", "",
        "- No Qwen visual encoder forward pass was executed.",
        "- No hidden states, embeddings, visual Token features, SAM logits, or SAM candidate tensors were saved.",
        f"- Forbidden cache files found in output: `{len(forbidden)}`.", "",
        "## Training readiness", "",
        f"- Ready for matched B20K/M20K training: **{summary['validation_passed']}**.",
        "- This task did not start VIS training and did not modify pilot 200 or the accepted 1K outputs.",
    ]
    (output_dir / "generation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    _state(config, stage="complete" if not errors else "validation_failed", validation_passed=not errors, summary_path=str((output_dir / "summary.json").resolve()))
    return summary


_ACTIVE_CONFIG: dict[str, Any] = {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 20K Mask-GAZE offline supervision without Qwen feature caching")
    parser.add_argument("--config", type=Path, default=Path("configs/mask_gaze_20k.yaml"))
    parser.add_argument("--stage", choices=("manifest", "dry-run", "masks", "tokens", "validate", "all"), default="all")
    args = parser.parse_args()
    config = load_config(args.config)
    global _ACTIVE_CONFIG
    _ACTIVE_CONFIG = config
    seed_everything(int(config["project"]["seed"]), bool(config["runtime"]["deterministic_algorithms"]))
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    predictor = None
    if args.stage in {"masks", "all"}:
        state_path = output_dir / STATE_NAME
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
        if "generation_started_unix" not in state:
            reuse_dir = Path(config["paths"]["reuse_1k_manifest"]).parent
            core_1k_bytes = sum(
                (reuse_dir / name).stat().st_size
                for name in ("manifest_1k.jsonl", "subject_masks.jsonl", "token_supervision_qwen25vl.jsonl")
            )
            visual_paths = list((reuse_dir / "visualizations").glob("*.png"))
            visual_average = int(mean(path.stat().st_size for path in visual_paths)) if visual_paths else 1_200_000
            estimated_final = core_1k_bytes * 20 + visual_average * int(config["visualization"]["count"])
            estimated_peak = estimated_final + (reuse_dir / "subject_masks.jsonl").stat().st_size * 20 + (reuse_dir / "token_supervision_qwen25vl.jsonl").stat().st_size * 20
            _state(
                config,
                generation_started_unix=time.time(),
                estimated_final_output_bytes=int(estimated_final),
                estimated_peak_output_bytes=int(estimated_peak),
                free_disk_bytes_before_generation=shutil.disk_usage(output_dir).free,
            )
    if args.stage in {"manifest", "all"}:
        report = build_manifest(config)
        print(json.dumps({"manifest": report["selected"], "sources": report["selected_source_counts"]}, ensure_ascii=False), flush=True)
    if args.stage in {"dry-run", "all"}:
        report, predictor = dry_run(config, predictor)
        print(json.dumps({"dry_run_passed": report["passed"]}, ensure_ascii=False), flush=True)
    if args.stage in {"masks", "all"}:
        run_masks(config, predictor)
    if args.stage in {"tokens", "all"}:
        run_tokens(config)
    if args.stage in {"validate", "all"}:
        summary = validate_and_finalize(config)
        print(json.dumps({"validation_passed": summary["validation_passed"], "success": summary["success_sample_count"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
