from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image

from src.data.build_manifest import _question_from_record, _resolve_image
from src.generate_mask_gaze_1k import (
    adaptive_indices,
    processor_configuration,
    token_coverage_and_contribution,
    union_bbox_mask,
)
from src.mask_gaze_preexperiment import original_gaze_bbox_labels
from src.masks import decode_rle
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


MANIFEST = "manifest_20k_exact.jsonl"
MASKS = "subject_masks_20k_exact.jsonl"
TOKENS = "token_supervision_20k_exact.jsonl"
AUDIT = "reuse_and_generation_audit.json"
FAILURES = "failures.jsonl"
SUMMARY = "summary.json"
REPORT = "data_alignment_report.md"
MASK_PARTIAL = ".subject_masks_20k_exact.partial.jsonl"
TOKEN_PARTIAL = ".token_supervision_20k_exact.partial.jsonl"
FORBIDDEN_FEATURE_SUFFIXES = {".safetensors", ".pt", ".pth", ".npy", ".npz"}


def _normalized_path(value: str | Path) -> str:
    return os.path.realpath(os.path.expanduser(str(value)))


def _distribution(values: Iterable[int | float]) -> dict[str, int | float | None]:
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


def _bbox_signature(bboxes: Any, decimals: int) -> tuple[tuple[float, ...], ...] | None:
    if not isinstance(bboxes, list) or not bboxes:
        return None
    result: list[tuple[float, ...]] = []
    try:
        for bbox in bboxes:
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                return None
            values = tuple(round(float(value), decimals) for value in bbox)
            if not all(math.isfinite(value) for value in values):
                return None
            result.append(values)
    except (TypeError, ValueError, OverflowError):
        return None
    return tuple(result)


def _authority_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    paths = config["paths"]
    data = config["data"]
    source_path = Path(paths["authority_manifest"])
    image_root = Path(paths["image_root"])
    rows: list[dict[str, Any]] = []
    for source_line, raw in enumerate(read_jsonl(source_path), start=1):
        dataset = str(raw.get("dataset") or raw.get("source") or "").strip()
        question = _question_from_record(raw, str(data["short_answer_suffix"]))
        answer_value = raw.get("output", raw.get("answer"))
        answer = "" if answer_value is None else str(answer_value).strip()
        image_path = _resolve_image(raw, image_root, str(data["image_prefix_to_strip"]))
        if image_path is None:
            raise RuntimeError(f"authority line {source_line}: invalid image field")
        resolved = _normalized_path(image_path)
        bboxes = raw.get("bboxs", raw.get("bboxes"))
        sample_id = stable_sample_id(dataset.lower(), resolved, question, answer)
        rows.append(
            {
                "sample_id": sample_id,
                "dataset": dataset,
                "source": dataset.lower(),
                "image_path": resolved,
                "question": question,
                "answer": answer,
                "bboxs": bboxes,
                "width": raw.get("width"),
                "height": raw.get("height"),
                "split": raw.get("split"),
                "source_line": source_line,
                "source_manifest": str(source_path.resolve()),
                "instruction": raw.get("instruction"),
                "input": raw.get("input"),
                "output": raw.get("output"),
                "images": raw.get("images"),
            }
        )
    expected = int(config["project"]["sample_count"])
    if len(rows) != expected:
        raise RuntimeError(f"authority row count {len(rows)} != {expected}")
    ids = [row["sample_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError("stable sample IDs are not unique across the authority manifest")
    return rows


ALIGNMENT_FIELDS = (
    "sample_id",
    "dataset",
    "image_path",
    "question",
    "answer",
    "bboxs",
    "width",
    "height",
    "split",
    "source_line",
)


def _alignment_hash(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        payload = {field: row.get(field) for field in ALIGNMENT_FIELDS}
        digest.update(canonical_json(payload).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _pilot_audit(config: dict[str, Any], authority: list[dict[str, Any]]) -> dict[str, Any]:
    paths = config["paths"]
    decimals = int(config["data"]["bbox_signature_decimals"])
    pilot_manifest = list(read_jsonl(paths["pilot_manifest"]))
    pilot_masks = list(read_jsonl(paths["pilot_masks"]))
    pilot_dataset = list(read_jsonl(paths["pilot_dataset"]))
    by_id = {row["sample_id"]: row for row in authority}
    geometry = {
        (row["source"], row["image_path"], _bbox_signature(row["bboxs"], decimals))
        for row in authority
    }
    decoded = 0
    missing_or_damaged: list[str] = []
    for row in pilot_masks:
        try:
            mask = decode_rle(row["subject_mask_rle"])
            shape = (int(row["height"]), int(row["width"]))
            if mask.shape != shape or not mask.any():
                raise ValueError(f"invalid mask shape/nonempty: {mask.shape} vs {shape}")
            decoded += 1
        except Exception as exc:
            missing_or_damaged.append(f"{row.get('sample_id')}: {type(exc).__name__}: {exc}")
    id_matches = sum(str(row["sample_id"]) in by_id for row in pilot_manifest)
    geometry_matches = sum(
        (
            str(row.get("source") or row.get("dataset")).lower(),
            _normalized_path(row["image_path"]),
            _bbox_signature(row["bboxes"], decimals),
        )
        in geometry
        for row in pilot_manifest
    )
    npz_files = list(Path(paths["pilot_mask_dir"]).glob("*.subject.npz"))
    coverage_rows = sum(isinstance(row.get("subject_token_coverage"), list) for row in pilot_dataset)
    old_label_rows = sum(isinstance(row.get("token_labels"), list) for row in pilot_dataset)
    return {
        "pilot_manifest_rows": len(pilot_manifest),
        "pilot_pixel_mask_rle_rows": len(pilot_masks),
        "pilot_npz_mask_files": len(npz_files),
        "pilot_rle_decodable_nonempty": decoded,
        "pilot_sample_id_matches_authority": id_matches,
        "pilot_dataset_image_bbox_matches_authority": geometry_matches,
        "pilot_token_coverage_rows": coverage_rows,
        "pilot_old_token_label_rows_not_reused": old_label_rows,
        "pilot_missing_or_damaged_count": len(missing_or_damaged),
        "pilot_missing_or_damaged": missing_or_damaged,
        "note": "The one ID-only match is an authority bbox extending beyond the image; the pilot prompt stored the clipped edge box. Pixel RLE is reused by exact sample_id, while old token labels are never reused.",
    }


def build_manifest(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    authority = _authority_rows(config)
    atomic_write_jsonl(output_dir / MANIFEST, authority)
    image_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in authority:
        image_groups[row["image_path"]].append(row)
    decimals = int(config["data"]["bbox_signature_decimals"])
    report = {
        "authority_manifest": str(Path(config["paths"]["authority_manifest"]).resolve()),
        "authority_manifest_sha256": sha256_file(config["paths"]["authority_manifest"]),
        "authority_row_count": len(authority),
        "authority_alignment_hash": _alignment_hash(authority),
        "source_counts": dict(Counter(row["source"] for row in authority)),
        "unique_image_count": len(image_groups),
        "duplicate_image_extra_rows": sum(len(rows) - 1 for rows in image_groups.values()),
        "same_image_multiple_question_group_count": sum(
            len({row["question"] for row in rows}) > 1 for rows in image_groups.values()
        ),
        "same_image_different_bbox_group_count": sum(
            len({_bbox_signature(row["bboxs"], decimals) for row in rows}) > 1
            for rows in image_groups.values()
        ),
        "pilot": _pilot_audit(config, authority),
        "deduplicated": False,
        "qwen_visual_feature_cache_generated": False,
    }
    atomic_write_json(output_dir / AUDIT, report)
    return report


def _valid_reusable_mask(row: dict[str, Any]) -> tuple[bool, str | None]:
    if bool(row.get("fallback_to_bbox") or row.get("subject_mask_fallback")):
        return False, "stored_mask_is_bbox_fallback"
    rle = row.get("subject_mask_rle") or row.get("mask_rle")
    if not isinstance(rle, dict):
        return False, "stored_rle_missing"
    try:
        mask = decode_rle(rle)
        height = int(row.get("height", row.get("image_height")))
        width = int(row.get("width", row.get("image_width")))
        if mask.shape != (height, width):
            return False, "stored_rle_shape_mismatch"
        if not mask.any():
            return False, "stored_rle_empty"
    except Exception as exc:
        return False, f"stored_rle_decode_error:{type(exc).__name__}"
    return True, None


def _reuse_indexes(config: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[Any, list[dict[str, Any]]]]:
    paths = config["paths"]
    decimals = int(config["data"]["bbox_signature_decimals"])
    sources = (
        ("current_20k", paths["current_20k_masks"]),
        ("pilot_200", paths["pilot_masks"]),
        ("mask_gaze_1k", paths["reuse_1k_masks"]),
    )
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_geometry: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for declared_source, path in sources:
        for row in read_jsonl(path):
            origin = declared_source
            if declared_source == "current_20k" and bool(row.get("reused_from_1k")):
                origin = "mask_gaze_1k"
            wrapped = {"record": row, "origin": origin, "declared_source": declared_source}
            by_id[str(row["sample_id"])].append(wrapped)
            geometry = (
                str(row.get("source") or row.get("dataset") or "").lower(),
                _normalized_path(row.get("image_path") or row.get("image")),
                _bbox_signature(row.get("bboxes"), decimals),
            )
            by_geometry[geometry].append(wrapped)
    return by_id, by_geometry


def _candidate_priority(candidate: dict[str, Any], target_id: str) -> tuple[int, int]:
    record = candidate["record"]
    exact = int(str(record.get("sample_id")) == target_id)
    origin_rank = {"pilot_200": 0, "mask_gaze_1k": 1, "current_20k": 2}.get(candidate["origin"], 9)
    return (-exact, origin_rank)


def _find_reuse(
    sample: dict[str, Any],
    by_id: dict[str, list[dict[str, Any]]],
    by_geometry: dict[Any, list[dict[str, Any]]],
    decimals: int,
) -> tuple[dict[str, Any] | None, str | None, list[str]]:
    rejected: list[str] = []
    exact_candidates = sorted(by_id.get(sample["sample_id"], []), key=lambda item: _candidate_priority(item, sample["sample_id"]))
    for candidate in exact_candidates:
        valid, reason = _valid_reusable_mask(candidate["record"])
        if valid:
            return candidate, "sample_id", rejected
        rejected.append(f"{candidate['declared_source']}:{reason}")
    geometry = (sample["source"], sample["image_path"], _bbox_signature(sample["bboxs"], decimals))
    for candidate in sorted(by_geometry.get(geometry, []), key=lambda item: _candidate_priority(item, sample["sample_id"])):
        valid, reason = _valid_reusable_mask(candidate["record"])
        if valid:
            return candidate, "dataset_image_bbox", rejected
        rejected.append(f"{candidate['declared_source']}:{reason}")
    return None, None, rejected


def _clip_sam_bboxes(sample: dict[str, Any], image_shape: tuple[int, int]) -> tuple[list[list[float]] | None, str | None]:
    height, width = image_shape
    clipped: list[list[float]] = []
    signature = _bbox_signature(sample.get("bboxs"), 8)
    if signature is None:
        return None, "invalid_bbox_structure"
    for bbox in signature:
        x1, y1, x2, y2 = bbox
        left, right = sorted((max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))))
        top, bottom = sorted((max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))))
        if right <= left or bottom <= top:
            return None, "bbox_empty_after_image_clip"
        clipped.append([left, top, right, bottom])
    return clipped, None


def _base_mask_record(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": sample["sample_id"],
        "dataset": sample["dataset"],
        "source": sample["source"],
        "image_path": sample["image_path"],
        "question": sample["question"],
        "answer": sample["answer"],
        "bboxs": sample["bboxs"],
        "width": sample["width"],
        "height": sample["height"],
        "split": sample["split"],
        "source_line": sample["source_line"],
    }


def _reused_mask_record(sample: dict[str, Any], candidate: dict[str, Any], method: str) -> dict[str, Any]:
    old = candidate["record"]
    rle = old.get("subject_mask_rle") or old.get("mask_rle")
    mask = decode_rle(rle)
    return {
        **_base_mask_record(sample),
        "mask_rle": rle,
        "subject_mask_rle": rle,
        "mask_area_ratio": float(mask.mean()),
        "mask_source": f"reused_{candidate['origin']}",
        "reuse_source": candidate["origin"],
        "reuse_declared_source": candidate["declared_source"],
        "reuse_method": method,
        "reuse_sample_id": old.get("sample_id"),
        "sam_attempts": 0,
        "sam_candidate_score": old.get("sam_candidate_score"),
        "sam_pred_iou": old.get("sam_pred_iou"),
        "stability_score": old.get("stability_score"),
        "selected_candidate": old.get("selected_candidate"),
        "mask_pixel_valid": True,
        "mask_failure_reasons": [],
        "fallback_to_bbox": False,
        "timing": {"sam_subject_seconds": 0.0},
    }


def _shared_mask_record(sample: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    return {
        **previous,
        **_base_mask_record(sample),
        "reuse_method": "same_image_same_bbox",
        "reuse_sample_id": previous["sample_id"],
        "sam_attempts": 0,
        "timing": {"sam_subject_seconds": 0.0},
    }


def _invalid_mask_record(sample: dict[str, Any], reasons: list[str], attempts: int, elapsed: float) -> dict[str, Any]:
    return {
        **_base_mask_record(sample),
        "mask_rle": None,
        "subject_mask_rle": None,
        "mask_area_ratio": None,
        "mask_source": "unavailable",
        "reuse_source": None,
        "reuse_declared_source": None,
        "reuse_method": None,
        "reuse_sample_id": None,
        "sam_attempts": attempts,
        "sam_candidate_score": None,
        "sam_pred_iou": None,
        "stability_score": None,
        "selected_candidate": None,
        "mask_pixel_valid": False,
        "mask_failure_reasons": reasons,
        "fallback_to_bbox": False,
        "timing": {"sam_subject_seconds": elapsed},
    }


def _new_sam_mask(sample: dict[str, Any], predictor, config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    reasons: list[str] = []
    attempts = int(config["sam"]["attempts"])
    try:
        with Image.open(sample["image_path"]) as handle:
            image = np.asarray(handle.convert("RGB"))
    except Exception as exc:
        return _invalid_mask_record(sample, [f"image_read_error:{type(exc).__name__}:{exc}"], 0, time.perf_counter() - started)
    if image.shape[:2] != (int(sample["height"]), int(sample["width"])):
        return _invalid_mask_record(
            sample,
            [f"image_size_mismatch:actual={image.shape[1]}x{image.shape[0]}:declared={sample['width']}x{sample['height']}"],
            0,
            time.perf_counter() - started,
        )
    prompt_bboxes, bbox_error = _clip_sam_bboxes(sample, image.shape[:2])
    if bbox_error:
        return _invalid_mask_record(sample, [bbox_error], 0, time.perf_counter() - started)
    for attempt in range(1, attempts + 1):
        try:
            result = segment_subject(image, prompt_bboxes, predictor, config["sam"])
            subject = result["subject_mask"]
            if result["subject_mask_fallback"]:
                reasons.append(f"attempt_{attempt}:sam_quality_rule_fallback")
                continue
            if not subject.any():
                reasons.append(f"attempt_{attempt}:sam_empty_mask")
                continue
            selected = [instance.get("selected", {}) for instance in result["subject_instances"]]
            valid_selected = [item for item in selected if item]
            return {
                **_base_mask_record(sample),
                "mask_rle": result["subject_mask_rle"],
                "subject_mask_rle": result["subject_mask_rle"],
                "mask_area_ratio": float(subject.mean()),
                "mask_source": "sam_bbox_prompt_new",
                "reuse_source": None,
                "reuse_declared_source": None,
                "reuse_method": None,
                "reuse_sample_id": None,
                "sam_attempts": attempt,
                "sam_candidate_score": float(np.mean([item["selection_score"] for item in valid_selected])) if valid_selected else None,
                "sam_pred_iou": float(np.mean([item["pred_iou"] for item in valid_selected])) if valid_selected else None,
                "stability_score": float(np.mean([item["stability"] for item in valid_selected])) if valid_selected else None,
                "selected_candidate": [int(item["candidate_index"]) for item in valid_selected],
                "mask_pixel_valid": True,
                "mask_failure_reasons": [],
                "sam_attempt_warnings": reasons,
                "fallback_to_bbox": False,
                "timing": {"sam_subject_seconds": time.perf_counter() - started},
            }
        except Exception as exc:
            reasons.append(f"attempt_{attempt}:{type(exc).__name__}:{exc}")
    return _invalid_mask_record(sample, reasons or ["sam_failed_without_reason"], attempts, time.perf_counter() - started)


def generate_masks(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["paths"]["output_dir"])
    manifest = list(read_jsonl(output_dir / MANIFEST))
    partial_path = output_dir / MASK_PARTIAL
    partial = list(read_jsonl(partial_path)) if partial_path.is_file() else []
    if [row["sample_id"] for row in partial] != [row["sample_id"] for row in manifest[: len(partial)]]:
        raise RuntimeError("mask partial file is not an exact manifest prefix")
    by_id, by_geometry = _reuse_indexes(config)
    decimals = int(config["data"]["bbox_signature_decimals"])
    completed_geometry: dict[Any, dict[str, Any]] = {}
    for sample, row in zip(manifest[: len(partial)], partial):
        key = (sample["source"], sample["image_path"], _bbox_signature(sample["bboxs"], decimals))
        completed_geometry[key] = row
    predictor = None
    for index, sample in enumerate(manifest[len(partial) :], start=len(partial) + 1):
        geometry = (sample["source"], sample["image_path"], _bbox_signature(sample["bboxs"], decimals))
        if geometry in completed_geometry:
            record = _shared_mask_record(sample, completed_geometry[geometry])
        else:
            candidate, method, rejected = _find_reuse(sample, by_id, by_geometry, decimals)
            if candidate is not None and method is not None:
                record = _reused_mask_record(sample, candidate, method)
                record["rejected_reuse_candidates"] = rejected
            else:
                if predictor is None:
                    predictor = load_sam_predictor(config)
                record = _new_sam_mask(sample, predictor, config)
                record["rejected_reuse_candidates"] = rejected
            completed_geometry[geometry] = record
        append_jsonl(partial_path, record)
        partial.append(record)
        if index % int(config["runtime"]["log_every"]) == 0:
            print(
                f"[mask {index}/{len(manifest)}] source={record['mask_source']} valid={record['mask_pixel_valid']}",
                flush=True,
            )
    atomic_write_jsonl(output_dir / MASKS, partial)
    partial_path.unlink(missing_ok=True)
    failures = [
        {
            "sample_id": row["sample_id"],
            "source_line": row["source_line"],
            "stage": "subject_mask",
            "reasons": row["mask_failure_reasons"],
        }
        for row in partial
        if not row["mask_pixel_valid"]
    ]
    atomic_write_jsonl(output_dir / FAILURES, failures)
    source_counts = Counter(row["reuse_source"] for row in partial if row.get("reuse_source"))
    audit = json.loads((output_dir / AUDIT).read_text(encoding="utf-8"))
    audit.update(
        {
            "mask_rows": len(partial),
            "reused_row_count_by_origin": dict(source_counts),
            "reused_current_20k_rows": source_counts["current_20k"],
            "reused_pilot_200_rows": source_counts["pilot_200"],
            "reused_1k_rows": source_counts["mask_gaze_1k"],
            "same_geometry_reuse_rows": sum(row.get("reuse_method") == "same_image_same_bbox" for row in partial),
            "new_sam_row_count": sum(row.get("mask_source") == "sam_bbox_prompt_new" for row in partial),
            "new_sam_required_geometry_count": sum(
                row.get("mask_source") in {"sam_bbox_prompt_new", "unavailable"}
                and row.get("reuse_method") is None
                for row in partial
            ),
            "new_sam_invocation_count": sum(
                int(row.get("sam_attempts", 0)) > 0 and row.get("reuse_method") is None
                for row in partial
            ),
            "final_sam_failure_geometry_count": sum(
                row.get("mask_source") == "unavailable"
                and int(row.get("sam_attempts", 0)) > 0
                and row.get("reuse_method") is None
                for row in partial
            ),
            "final_mask_unavailable_count": len(failures),
            "sam_total_seconds": float(sum(row.get("timing", {}).get("sam_subject_seconds", 0.0) for row in partial)),
        }
    )
    atomic_write_json(output_dir / AUDIT, audit)
    return audit


def _processor(config: dict[str, Any]):
    from transformers import AutoProcessor

    checkpoint = Path(config["paths"]["qwen_checkpoint"])
    qwen = config["qwen"]
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True, use_fast=bool(qwen["use_fast"]))
    processor.image_processor.min_pixels = int(qwen["min_pixels"])
    processor.image_processor.max_pixels = int(qwen["max_pixels"])
    processor.image_processor.size = {
        "shortest_edge": int(qwen["min_pixels"]),
        "longest_edge": int(qwen["max_pixels"]),
    }
    return processor


def _token_record(
    sample: dict[str, Any], mask_row: dict[str, Any], processor, config: dict[str, Any], processor_hash: str
) -> dict[str, Any]:
    started = time.perf_counter()
    qwen = config["qwen"]
    supervision = config["supervision"]
    with Image.open(sample["image_path"]) as handle:
        processed = processor.image_processor(images=[handle.convert("RGB")], return_tensors="pt")
    grid = [int(value) for value in processed["image_grid_thw"][0].tolist()]
    del processed
    patch = int(qwen["patch_size"])
    merge = int(qwen["spatial_merge_size"])
    count = grid[0] * (grid[1] // merge) * (grid[2] // merge)
    shape = (int(sample["height"]), int(sample["width"]))
    bbox_pixels = union_bbox_mask(shape, sample["bboxs"])
    bbox_coverage, _ = token_coverage_and_contribution(
        bbox_pixels, grid, patch_size=patch, spatial_merge_size=merge
    )
    bbox_budget = original_gaze_bbox_labels(
        sample["bboxs"], int(sample["width"]), int(sample["height"]), grid
    ).astype(np.int8)
    if bbox_budget.size != count:
        raise RuntimeError(f"{sample['sample_id']}: bbox budget length mismatch")

    valid_pixel_mask = bool(mask_row["mask_pixel_valid"])
    if valid_pixel_mask:
        subject = decode_rle(mask_row["subject_mask_rle"])
        mask_coverage, mask_contribution = token_coverage_and_contribution(
            subject, grid, patch_size=patch, spatial_merge_size=merge
        )
        positives = adaptive_indices(mask_contribution, float(supervision["adaptive_mass_target"]))
    else:
        mask_coverage = np.zeros(count, dtype=np.float32)
        mask_contribution = np.zeros(count, dtype=np.float32)
        positives = np.empty((0,), dtype=np.int64)
    reliable = np.flatnonzero(
        (mask_coverage <= float(supervision["reliable_background_max_coverage"]))
        & (bbox_coverage <= float(supervision["reliable_background_max_coverage"]))
    ).astype(np.int64)
    reliable = np.setdiff1d(reliable, positives, assume_unique=False).astype(np.int64)
    labels = np.full(count, int(supervision["ignore_label"]), dtype=np.int8)
    labels[reliable] = int(supervision["background_label"])
    labels[positives] = int(supervision["foreground_label"])
    ignore = np.flatnonzero(labels == int(supervision["ignore_label"])).astype(np.int64)
    actual_mass = float(np.clip(mask_contribution[positives].sum(), 0.0, 1.0)) if positives.size else 0.0
    invalid_reasons = list(mask_row.get("mask_failure_reasons", []))
    if valid_pixel_mask and not positives.size:
        invalid_reasons.append("zero_mask_positive_tokens")
    if not reliable.size:
        invalid_reasons.append("zero_reliable_negative_tokens")
    if valid_pixel_mask and actual_mass + 1e-6 < float(supervision["adaptive_mass_target"]):
        invalid_reasons.append("adaptive_mask_mass_below_target")
    mask_loss_valid = valid_pixel_mask and bool(positives.size) and bool(reliable.size) and not invalid_reasons
    return {
        "sample_id": sample["sample_id"],
        "dataset": sample["dataset"],
        "source": sample["source"],
        "image_path": sample["image_path"],
        "question": sample["question"],
        "answer": sample["answer"],
        "bboxs": sample["bboxs"],
        "image_grid_thw": grid,
        "num_visual_tokens": int(count),
        "visual_token_count": int(count),
        "mask_rle_ref": {"file": MASKS, "sample_id": sample["sample_id"]},
        "mask_token_coverage": mask_coverage.astype(float).tolist(),
        "mask_token_contribution": mask_contribution.astype(float).tolist(),
        "bbox_token_coverage": bbox_coverage.astype(float).tolist(),
        "mask_token_labels": labels.astype(int).tolist(),
        "mask_positive_indices": positives.astype(int).tolist(),
        "reliable_negative_indices": reliable.astype(int).tolist(),
        "reliable_background_indices": reliable.astype(int).tolist(),
        "ignore_indices": ignore.astype(int).tolist(),
        "mask_positive_count": int(positives.size),
        "reliable_negative_count": int(reliable.size),
        "ignore_count": int(ignore.size),
        "actual_mask_mass_coverage": actual_mass,
        "adaptive_mass_target": float(supervision["adaptive_mass_target"]),
        "bbox_budget_mask": bbox_budget.astype(int).tolist(),
        "bbox_foreground_ratio": float(bbox_budget.mean()),
        "bbox_budget_mask_valid": bool(bbox_budget.any()),
        "mask_source": mask_row["mask_source"],
        "mask_pixel_valid": valid_pixel_mask,
        "mask_loss_valid": bool(mask_loss_valid),
        "mask_loss_invalid_reasons": invalid_reasons,
        "processor_config_hash": processor_hash,
        "timing": {"qwen_preprocess_and_token_mapping_seconds": time.perf_counter() - started},
    }


def generate_tokens(config: dict[str, Any]) -> None:
    output_dir = Path(config["paths"]["output_dir"])
    manifest = list(read_jsonl(output_dir / MANIFEST))
    masks = list(read_jsonl(output_dir / MASKS))
    if [row["sample_id"] for row in manifest] != [row["sample_id"] for row in masks]:
        raise RuntimeError("manifest/mask ordering mismatch before token generation")
    partial_path = output_dir / TOKEN_PARTIAL
    partial = list(read_jsonl(partial_path)) if partial_path.is_file() else []
    if [row["sample_id"] for row in partial] != [row["sample_id"] for row in manifest[: len(partial)]]:
        raise RuntimeError("token partial file is not an exact manifest prefix")
    processor = _processor(config)
    processor_hash = processor_configuration(config)["hash"]
    for index, (sample, mask_row) in enumerate(zip(manifest[len(partial) :], masks[len(partial) :]), start=len(partial) + 1):
        record = _token_record(sample, mask_row, processor, config, processor_hash)
        append_jsonl(partial_path, record)
        partial.append(record)
        if index % int(config["runtime"]["log_every"]) == 0:
            print(
                f"[token {index}/{len(manifest)}] n={record['num_visual_tokens']} mask_loss_valid={record['mask_loss_valid']}",
                flush=True,
            )
    atomic_write_jsonl(output_dir / TOKENS, partial)
    partial_path.unlink(missing_ok=True)


def _scan_forbidden(output_dir: Path) -> list[str]:
    return [
        str(path.resolve())
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_FEATURE_SUFFIXES
    ]


def validate(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["paths"]["output_dir"])
    authority = _authority_rows(config)
    manifest = list(read_jsonl(output_dir / MANIFEST))
    masks = list(read_jsonl(output_dir / MASKS))
    tokens = list(read_jsonl(output_dir / TOKENS))
    errors: list[str] = []
    expected_count = int(config["project"]["sample_count"])
    if not (len(authority) == len(manifest) == len(masks) == len(tokens) == expected_count):
        errors.append(f"row counts authority/manifest/masks/tokens={len(authority)}/{len(manifest)}/{len(masks)}/{len(tokens)}")
    authority_hash = _alignment_hash(authority)
    output_hash = _alignment_hash(manifest)
    if authority_hash != output_hash:
        errors.append("authority/output alignment hash mismatch")
    source_counts_authority = Counter(row["source"] for row in authority)
    source_counts_output = Counter(row["source"] for row in manifest)
    if source_counts_authority != source_counts_output:
        errors.append("source counts differ from authority")
    exact_fields_match = True
    rle_valid = 0
    valid_mask_positive_failures = 0
    token_length_failures = 0
    mass_failures = 0
    zero_negative = 0
    duplicate_sample_id_count = len(manifest) - len({row["sample_id"] for row in manifest})
    nonfinite_array_failures = 0
    label_index_consistency_failures = 0
    mask_contribution_sum_failures = 0
    bbox_budget_mask_failures = 0
    if duplicate_sample_id_count:
        errors.append(f"duplicate sample IDs: {duplicate_sample_id_count}")
    invalid_reason_counts: Counter[str] = Counter()
    for index, quadruple in enumerate(zip(authority, manifest, masks, tokens), start=1):
        expected, sample, mask_row, token = quadruple
        for field in ALIGNMENT_FIELDS:
            if sample.get(field) != expected.get(field):
                exact_fields_match = False
                errors.append(f"line {index}: authority mismatch in {field}")
                break
        ids = {sample.get("sample_id"), mask_row.get("sample_id"), token.get("sample_id")}
        if len(ids) != 1:
            errors.append(f"line {index}: sample_id ordering mismatch")
            continue
        if mask_row["mask_pixel_valid"]:
            try:
                decoded = decode_rle(mask_row["subject_mask_rle"])
                if decoded.shape != (int(sample["height"]), int(sample["width"])) or not decoded.any():
                    raise ValueError("shape/nonempty mismatch")
                rle_valid += 1
            except Exception as exc:
                errors.append(f"{sample['sample_id']}: RLE validation failed: {exc}")
        count = int(token["num_visual_tokens"])
        grid = [int(value) for value in token["image_grid_thw"]]
        merge = int(config["qwen"]["spatial_merge_size"])
        expected_tokens = grid[0] * (grid[1] // merge) * (grid[2] // merge)
        array_fields = (
            "mask_token_coverage",
            "mask_token_contribution",
            "bbox_token_coverage",
            "mask_token_labels",
            "bbox_budget_mask",
        )
        if count != expected_tokens or any(len(token[field]) != count for field in array_fields):
            token_length_failures += 1
            errors.append(f"{sample['sample_id']}: token length mismatch")
            continue
        mask_coverage = np.asarray(token["mask_token_coverage"], dtype=np.float64)
        mask_contribution = np.asarray(token["mask_token_contribution"], dtype=np.float64)
        bbox_coverage = np.asarray(token["bbox_token_coverage"], dtype=np.float64)
        labels = np.asarray(token["mask_token_labels"], dtype=np.int8)
        bbox_budget = np.asarray(token["bbox_budget_mask"], dtype=np.int8)
        if not (
            np.isfinite(mask_coverage).all()
            and np.isfinite(mask_contribution).all()
            and np.isfinite(bbox_coverage).all()
        ):
            nonfinite_array_failures += 1
            errors.append(f"{sample['sample_id']}: non-finite token supervision value")
        positives = np.asarray(token["mask_positive_indices"], dtype=np.int64)
        negatives = np.asarray(token["reliable_negative_indices"], dtype=np.int64)
        ignores = np.asarray(token["ignore_indices"], dtype=np.int64)
        expected_positive = np.flatnonzero(labels == int(config["supervision"]["foreground_label"]))
        expected_negative = np.flatnonzero(labels == int(config["supervision"]["background_label"]))
        expected_ignore = np.flatnonzero(labels == int(config["supervision"]["ignore_label"]))
        if not (
            np.array_equal(positives, expected_positive)
            and np.array_equal(negatives, expected_negative)
            and np.array_equal(ignores, expected_ignore)
            and positives.size + negatives.size + ignores.size == count
        ):
            label_index_consistency_failures += 1
            errors.append(f"{sample['sample_id']}: label/index partition mismatch")
        if not np.isin(bbox_budget, (0, 1)).all() or not bbox_budget.any():
            bbox_budget_mask_failures += 1
            errors.append(f"{sample['sample_id']}: invalid or empty GAZE bbox budget mask")
        if bool(mask_row["mask_pixel_valid"]):
            if not math.isclose(float(mask_contribution.sum()), 1.0, rel_tol=1e-5, abs_tol=1e-5):
                mask_contribution_sum_failures += 1
                errors.append(f"{sample['sample_id']}: mask contribution does not sum to one")
            if not token["mask_positive_indices"]:
                valid_mask_positive_failures += 1
                errors.append(f"{sample['sample_id']}: valid mask has no positive token")
            if float(token["actual_mask_mass_coverage"]) + 1e-6 < float(config["supervision"]["adaptive_mass_target"]):
                mass_failures += 1
                errors.append(f"{sample['sample_id']}: mask mass target not reached")
        zero_negative += int(not token["reliable_negative_indices"])
        if not token["mask_loss_valid"]:
            invalid_reason_counts.update(token["mask_loss_invalid_reasons"] or ["unspecified"])
    pilot_manifest = list(read_jsonl(config["paths"]["pilot_manifest"]))
    output_by_id = {row["sample_id"]: row for row in manifest}
    mask_by_id = {row["sample_id"]: row for row in masks}
    pilot_restored = sum(
        row["sample_id"] in output_by_id
        and int(output_by_id[row["sample_id"]]["source_line"]) == int(row["source_line"])
        for row in pilot_manifest
    )
    forbidden = _scan_forbidden(output_dir)
    if forbidden:
        errors.append(f"forbidden feature-like files found: {forbidden[:10]}")
    audit = json.loads((output_dir / AUDIT).read_text(encoding="utf-8"))
    mask_loss_valid_count = sum(bool(row["mask_loss_valid"]) for row in tokens)
    pilot_mask_rows = [mask_by_id[row["sample_id"]] for row in pilot_manifest if row["sample_id"] in mask_by_id]
    final_failures = [
        {
            "sample_id": token["sample_id"],
            "source_line": manifest[index]["source_line"],
            "stage": "mask_supervision",
            "mask_source": token["mask_source"],
            "reasons": token["mask_loss_invalid_reasons"],
        }
        for index, token in enumerate(tokens)
        if not token["mask_loss_valid"]
    ]
    atomic_write_jsonl(output_dir / FAILURES, final_failures)
    summary = {
        "validation_passed": not errors,
        "validation_error_count": len(errors),
        "validation_errors": errors[:1000],
        "row_count": len(manifest),
        "authority_row_count": len(authority),
        "sample_id_image_question_answer_bbox_order_exact": exact_fields_match and authority_hash == output_hash,
        "authority_alignment_hash": authority_hash,
        "output_alignment_hash": output_hash,
        "authority_manifest_sha256": sha256_file(config["paths"]["authority_manifest"]),
        "source_counts": dict(source_counts_output),
        "source_counts_match_authority": source_counts_output == source_counts_authority,
        "unique_image_count": len({row["image_path"] for row in manifest}),
        "image_deduplication_performed": False,
        "pilot_rows_restored_at_original_source_line": pilot_restored,
        "pilot_total": len(pilot_manifest),
        "pilot_pixel_masks_reused_from_pilot": sum(row.get("reuse_source") == "pilot_200" for row in pilot_mask_rows),
        "pilot_pixel_masks_sam_retried": sum(int(row.get("sam_attempts", 0)) > 0 for row in pilot_mask_rows),
        "pilot_pixel_masks_regenerated_by_sam": sum(row.get("mask_source") == "sam_bbox_prompt_new" for row in pilot_mask_rows),
        "pilot_pixel_masks_unavailable": sum(not row.get("mask_pixel_valid") for row in pilot_mask_rows),
        "valid_pixel_mask_rle_count": rle_valid,
        "invalid_pixel_mask_count": len(masks) - rle_valid,
        "mask_loss_valid_count": mask_loss_valid_count,
        "mask_loss_invalid_count": len(tokens) - mask_loss_valid_count,
        "mask_loss_invalid_reason_counts": dict(invalid_reason_counts),
        "zero_reliable_negative_count": zero_negative,
        "valid_mask_zero_positive_count": valid_mask_positive_failures,
        "mask_mass_below_target_count": mass_failures,
        "token_length_failure_count": token_length_failures,
        "duplicate_sample_id_count": duplicate_sample_id_count,
        "nonfinite_array_failure_count": nonfinite_array_failures,
        "label_index_consistency_failure_count": label_index_consistency_failures,
        "mask_contribution_sum_failure_count": mask_contribution_sum_failures,
        "bbox_budget_mask_failure_count": bbox_budget_mask_failures,
        "mask_positive_count": _distribution(row["mask_positive_count"] for row in tokens),
        "reliable_negative_count": _distribution(row["reliable_negative_count"] for row in tokens),
        "ignore_count": _distribution(row["ignore_count"] for row in tokens),
        "bbox_foreground_ratio": _distribution(row["bbox_foreground_ratio"] for row in tokens),
        "reused_current_20k_rows": audit.get("reused_current_20k_rows", 0),
        "reused_pilot_200_rows": audit.get("reused_pilot_200_rows", 0),
        "reused_1k_rows": audit.get("reused_1k_rows", 0),
        "new_sam_invocation_count": audit.get("new_sam_invocation_count", 0),
        "new_sam_required_geometry_count": audit.get("new_sam_required_geometry_count", 0),
        "new_sam_success_geometry_count": audit.get("new_sam_invocation_count", 0)
        - audit.get("final_sam_failure_geometry_count", 0),
        "new_sam_success_row_count": audit.get("new_sam_row_count", 0),
        "final_sam_failure_geometry_count": audit.get("final_sam_failure_geometry_count", 0),
        "final_sam_or_mask_failure_count": audit.get("final_mask_unavailable_count", 0),
        "same_image_multiple_question_group_count": audit["same_image_multiple_question_group_count"],
        "same_image_different_bbox_group_count": audit["same_image_different_bbox_group_count"],
        "duplicate_image_extra_rows": audit["duplicate_image_extra_rows"],
        "pool_only_sample_count": 0,
        "qwen_visual_feature_cache_generated": False,
        "forbidden_feature_cache_files": forbidden,
        "processor": processor_configuration(config),
        "config_hash": config_hash(config),
        "output_directory_bytes": sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file()),
    }
    atomic_write_json(output_dir / SUMMARY, summary)
    report = [
        "# Exact original-20K Mask supervision alignment report",
        "",
        f"- Validation passed: **{summary['validation_passed']}**; errors: `{summary['validation_error_count']}`.",
        f"- Authority/manifest/mask/token rows: `{len(authority)}/{len(manifest)}/{len(masks)}/{len(tokens)}`.",
        f"- Ordered alignment hash: `{authority_hash}` (output: `{output_hash}`).",
        f"- Exact line alignment: `{summary['sample_id_image_question_answer_bbox_order_exact']}`.",
        f"- Sources: `{canonical_json(summary['source_counts'])}`; exact match: `{summary['source_counts_match_authority']}`.",
        f"- Unique images: `{summary['unique_image_count']}`; duplicate-image extra rows retained: `{summary['duplicate_image_extra_rows']}`; deduplicated: `false`.",
        f"- Pilot restored at original positions: `{pilot_restored}/{len(pilot_manifest)}`.",
        f"- Pilot pixel masks reused/SAM-retried/successfully-regenerated/unavailable: `{summary['pilot_pixel_masks_reused_from_pilot']}/{summary['pilot_pixel_masks_sam_retried']}/{summary['pilot_pixel_masks_regenerated_by_sam']}/{summary['pilot_pixel_masks_unavailable']}`.",
        f"- Pilot discovery audit: `{canonical_json(audit['pilot'])}`.",
        f"- Reused rows — current 20K: `{summary['reused_current_20k_rows']}`, pilot 200: `{summary['reused_pilot_200_rows']}`, 1K: `{summary['reused_1k_rows']}`.",
        f"- Mask geometries requiring generation: `{summary['new_sam_required_geometry_count']}`; actual SAM invocations: `{summary['new_sam_invocation_count']}`; successful new geometries/rows: `{summary['new_sam_success_geometry_count']}/{summary['new_sam_success_row_count']}`; final SAM-failed geometries: `{summary['final_sam_failure_geometry_count']}`; unavailable rows: `{summary['final_sam_or_mask_failure_count']}`.",
        f"- Valid pixel masks: `{summary['valid_pixel_mask_rle_count']}`; mask-loss valid: `{summary['mask_loss_valid_count']}`; invalid: `{summary['mask_loss_invalid_count']}`.",
        f"- Zero reliable-negative rows retained: `{summary['zero_reliable_negative_count']}`.",
        f"- Valid-mask zero-positive rows: `{summary['valid_mask_zero_positive_count']}`; mass-below-90%: `{summary['mask_mass_below_target_count']}`; token-length failures: `{summary['token_length_failure_count']}`.",
        f"- Duplicate sample IDs: `{summary['duplicate_sample_id_count']}`; non-finite arrays: `{summary['nonfinite_array_failure_count']}`; label/index partition failures: `{summary['label_index_consistency_failure_count']}`; contribution-sum failures: `{summary['mask_contribution_sum_failure_count']}`; invalid BBox budget masks: `{summary['bbox_budget_mask_failure_count']}`.",
        f"- Same-image multi-question groups: `{summary['same_image_multiple_question_group_count']}`; same-image different-BBox groups: `{summary['same_image_different_bbox_group_count']}`.",
        f"- Pool-only samples admitted: `{summary['pool_only_sample_count']}`.",
        "",
        "```yaml",
        "qwen_visual_feature_cache_generated: false",
        "```",
        "",
        "No Qwen visual encoder forward, hidden state, embedding, or visual feature cache was generated. No VIS training was started.",
    ]
    (output_dir / REPORT).write_text("\n".join(report) + "\n", encoding="utf-8")
    if errors:
        raise RuntimeError(f"validation failed with {len(errors)} errors; inspect {output_dir / SUMMARY}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild Mask supervision in the exact original GAZE 20K row order")
    parser.add_argument("--config", type=Path, default=Path("configs/mask_gaze_20k_exact.yaml"))
    parser.add_argument("--stage", choices=("manifest", "masks", "tokens", "validate", "all"), default="all")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    seed_everything(int(config["project"]["seed"]), bool(config["runtime"]["deterministic_algorithms"]))
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    if args.stage in {"manifest", "all"}:
        result = build_manifest(config)
        print(json.dumps({"manifest_rows": result["authority_row_count"], "pilot": result["pilot"]}, ensure_ascii=False), flush=True)
    if args.stage in {"masks", "all"}:
        result = generate_masks(config)
        print(json.dumps({"mask_rows": result["mask_rows"], "new_sam": result["new_sam_invocation_count"]}, ensure_ascii=False), flush=True)
    if args.stage in {"tokens", "all"}:
        generate_tokens(config)
        print(json.dumps({"tokens": "complete"}, ensure_ascii=False), flush=True)
    if args.stage in {"validate", "all"}:
        result = validate(config)
        print(json.dumps({"validation_passed": result["validation_passed"], "rows": result["row_count"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
