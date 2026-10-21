from __future__ import annotations

import argparse
import html
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from src.data.build_manifest import _normalize_bboxes, _question_from_record, _resolve_image
from src.masks import decode_rle
from src.utils import atomic_write_json, atomic_write_jsonl, read_jsonl, stable_sample_id


SEED = 42
SPATIAL_MERGE_SIZE = 2


def distribution(values: Iterable[float | int]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
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


def original_gaze_bbox_labels(
    bboxes: list[list[float]],
    width: int,
    height: int,
    image_grid_thw: list[int],
) -> np.ndarray:
    """Reproduce data_qwen.pixel_bboxs_to_ref_token_mask exactly.

    GAZE projects pixel-space boxes directly onto the post-spatial-merge grid,
    truncates both endpoints, and fills the inclusive rectangle. This is not a
    pixel-coverage threshold and is intentionally separate from adaptive mass
    mapping used by the Mask-GAZE target.
    """
    temporal, grid_h, grid_w = (int(value) for value in image_grid_thw)
    token_h = grid_h // SPATIAL_MERGE_SIZE
    token_w = grid_w // SPATIAL_MERGE_SIZE
    if temporal <= 0 or token_h <= 0 or token_w <= 0 or width <= 0 or height <= 0:
        raise ValueError("invalid image or token geometry")
    mask = np.zeros((token_h, token_w), dtype=bool)
    for bbox in bboxes:
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = (float(value) for value in bbox)
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        x1 = min(1.0, max(0.0, x1 / float(width)))
        x2 = min(1.0, max(0.0, x2 / float(width)))
        y1 = min(1.0, max(0.0, y1 / float(height)))
        y2 = min(1.0, max(0.0, y2 / float(height)))
        x1_grid = max(0, min(token_w - 1, int(x1 * token_w)))
        y1_grid = max(0, min(token_h - 1, int(y1 * token_h)))
        x2_grid = min(int(x2 * token_w), token_w - 1)
        y2_grid = min(int(y2 * token_h), token_h - 1)
        x2_grid = max(x1_grid, x2_grid)
        y2_grid = max(y1_grid, y2_grid)
        mask[y1_grid : y2_grid + 1, x1_grid : x2_grid + 1] = True
    if temporal > 1:
        mask = np.tile(mask.reshape(-1), temporal).reshape(temporal, token_h, token_w)
    else:
        mask = mask.reshape(1, token_h, token_w)
    return mask.reshape(-1)


def three_value_pairwise_margin_loss(scores, labels, margin: float = 0.2):
    """Original GAZE full-pair ranking loss with label=-1 excluded."""
    import torch.nn.functional as functional

    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if positives.numel() == 0 or negatives.numel() == 0:
        return None
    differences = positives.unsqueeze(1) - negatives.unsqueeze(0) - float(margin)
    return functional.softplus(-differences).mean()


def _load_by_id(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    rows = list(read_jsonl(path))
    ids = [str(row.get("sample_id")) for row in rows]
    duplicates = sorted(sample_id for sample_id, count in Counter(ids).items() if count > 1)
    return rows, {str(row["sample_id"]): row for row in rows}, duplicates


def _verify_original_rows(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[Path, dict[int, dict[str, Any]]] = {}
    for row in manifest:
        grouped.setdefault(Path(row["source_manifest"]), {})[int(row["source_line"])] = row
    failures: list[dict[str, Any]] = []
    for source_path, expected in grouped.items():
        if not source_path.is_file():
            failures.append({"sample_id": None, "reason": "source_manifest_missing", "path": str(source_path)})
            continue
        remaining = set(expected)
        with source_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line_number not in expected:
                    continue
                raw = json.loads(line)
                manifest_row = expected[line_number]
                bbox_data = _normalize_bboxes(raw)
                question = _question_from_record(raw, "\nAnswer the question using a single word or phrase.")
                answer_value = raw.get("output", raw.get("answer"))
                answer = "" if answer_value is None else str(answer_value).strip()
                image_path = _resolve_image(raw, Path("/data/DataSet/vlm_dataset/cot_image_data"), "cot/")
                source = str(raw.get("dataset") or raw.get("source") or "").strip().lower()
                reasons: list[str] = []
                if bbox_data is None:
                    reasons.append("original_bbox_invalid")
                else:
                    boxes, width, height = bbox_data
                    if boxes != manifest_row["bboxes"] or int(width) != int(manifest_row["width"]) or int(height) != int(manifest_row["height"]):
                        reasons.append("original_geometry_mismatch")
                if image_path is None or str(image_path.resolve()) != str(Path(manifest_row["image_path"]).resolve()):
                    reasons.append("original_image_mismatch")
                reconstructed = stable_sample_id(source, str(Path(manifest_row["image_path"]).resolve()), question, answer)
                if reconstructed != manifest_row["sample_id"]:
                    reasons.append("original_content_or_id_mismatch")
                if reasons:
                    failures.append({"sample_id": manifest_row["sample_id"], "reason": ",".join(reasons)})
                remaining.remove(line_number)
                if not remaining:
                    break
        failures.extend(
            {"sample_id": expected[line_number]["sample_id"], "reason": "source_line_missing"}
            for line_number in sorted(remaining)
        )
    return failures


def _mask_shape_features(mask_row: dict[str, Any]) -> dict[str, float | bool]:
    mask = decode_rle(mask_row["subject_mask_rle"])
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return {"mask_nonempty": False, "elongation": 0.0, "fill_ratio": 0.0}
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    elongation = max(width / max(height, 1), height / max(width, 1))
    fill_ratio = float(mask.sum()) / float(width * height)
    return {"mask_nonempty": True, "elongation": float(elongation), "fill_ratio": fill_ratio}


def _render_token_overlay(image: np.ndarray, labels: np.ndarray, grid: list[int], palette: dict[int, tuple[int, int, int]]) -> np.ndarray:
    temporal, patch_h, patch_w = (int(value) for value in grid)
    token_h, token_w = patch_h // SPATIAL_MERGE_SIZE, patch_w // SPATIAL_MERGE_SIZE
    label_grid = labels.reshape(temporal, token_h, token_w)[0]
    rgb = np.zeros((token_h, token_w, 3), dtype=np.uint8)
    for value, color in palette.items():
        rgb[label_grid == value] = color
    resized = np.asarray(Image.fromarray(rgb).resize((image.shape[1], image.shape[0]), Image.Resampling.NEAREST))
    return np.clip(0.48 * image.astype(np.float32) + 0.52 * resized.astype(np.float32), 0, 255).astype(np.uint8)


def _select_visualizations(rows: list[dict[str, Any]], count: int = 60) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    selected: dict[str, dict[str, Any]] = {}
    for source in ("gqa", "v7w", "flickr30k"):
        candidates = [row for row in rows if row["source"] == source]
        for row in rng.sample(candidates, min(10, len(candidates))):
            row["visual_categories"].append(f"random_{source}")
            selected[row["sample_id"]] = row
    for category, key, reverse in (
        ("small_target", "mask_area_ratio", False),
        ("thin_region", "thin_score", True),
        ("multi_subject", "bbox_count", True),
    ):
        for row in sorted(rows, key=lambda item: item[key], reverse=reverse)[:12]:
            row["visual_categories"].append(category)
            selected[row["sample_id"]] = row
    remaining = [row for row in rows if row["sample_id"] not in selected]
    rng.shuffle(remaining)
    for row in remaining:
        if len(selected) >= count:
            break
        row["visual_categories"].append("random_fill")
        selected[row["sample_id"]] = row
    return list(selected.values())[: max(50, count)]


def _write_visualizations(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    palette = {-1: (245, 174, 35), 0: (38, 103, 230), 1: (38, 204, 84)}
    bbox_palette = {0: (38, 103, 230), 1: (230, 55, 45)}
    cards: list[str] = []
    category_counts: Counter[str] = Counter()
    for index, row in enumerate(rows, start=1):
        image = np.asarray(Image.open(row["image_path"]).convert("RGB"))
        subject = decode_rle(row["subject_mask_rle"])
        mask_labels = np.asarray(row["mask_token_labels"], dtype=np.int8)
        bbox_labels = np.asarray(row["gaze_bbox_labels"], dtype=np.int8)
        fig, axes = plt.subplots(1, 5, figsize=(23, 5.2), dpi=105)
        axes[0].imshow(image)
        for bbox in row["bboxes"]:
            x1, y1, x2, y2 = bbox
            axes[0].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="yellow", linewidth=2))
        axes[0].set_title("Original + BBox")
        subject_view = image.copy().astype(np.float32)
        subject_view[subject] = 0.48 * subject_view[subject] + 0.52 * np.asarray((35, 215, 95), dtype=np.float32)
        axes[1].imshow(subject_view.astype(np.uint8))
        axes[1].set_title(f"SAM subject mask\narea={row['mask_area_ratio']:.4f}")
        axes[2].imshow(_render_token_overlay(image, mask_labels, row["image_grid_thw"], palette))
        axes[2].set_title(f"Mask labels\n+{row['mask_positive_count']} / -{row['reliable_negative_count']} / ign {row['ignore_count']}")
        axes[3].imshow(_render_token_overlay(image, bbox_labels, row["image_grid_thw"], bbox_palette))
        axes[3].set_title(f"Original GAZE BBox labels\n+{row['gaze_bbox_positive_count']} / -{row['gaze_bbox_negative_count']}")
        difference = np.full_like(mask_labels, -1)
        difference[(bbox_labels == 1) & (mask_labels != 1)] = 0
        difference[(bbox_labels == 1) & (mask_labels == 1)] = 1
        difference[(bbox_labels == 0) & (mask_labels == 1)] = 2
        diff_palette = {-1: (80, 80, 80), 0: (230, 55, 45), 1: (38, 204, 84), 2: (150, 70, 210)}
        axes[4].imshow(_render_token_overlay(image, difference, row["image_grid_thw"], diff_palette))
        axes[4].set_title("Positive comparison\nred=BBox only, green=both, purple=mask only")
        for axis in axes:
            axis.axis("off")
        categories = sorted(set(row["visual_categories"]))
        category_counts.update(categories)
        fig.suptitle(f"{row['sample_id']} | {row['source']} | {', '.join(categories)}\nQ: {row['question']}\nA: {row['answer']}", fontsize=10)
        fig.legend(
            handles=[Patch(color=np.asarray(palette[value]) / 255.0, label=label) for value, label in ((1, "mask foreground"), (0, "reliable background"), (-1, "ignore"))],
            loc="lower center",
            ncol=3,
        )
        fig.tight_layout(rect=(0, 0.08, 1, 0.88))
        filename = f"{index:03d}_{row['sample_id']}.png"
        fig.savefig(output_dir / filename, bbox_inches="tight")
        plt.close(fig)
        cards.append(
            f'<article><a href="{html.escape(filename)}"><img src="{html.escape(filename)}" loading="lazy"></a>'
            f'<p><b>{html.escape(row["sample_id"])}</b> — {html.escape(", ".join(categories))}</p></article>'
        )
    document = """<!doctype html><html><head><meta charset="utf-8"><title>Mask-GAZE 1K acceptance</title>
<style>body{font-family:sans-serif;margin:20px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:14px}article{border:1px solid #ddd;padding:8px}img{width:100%}p{font-size:13px}</style></head><body>
<h1>Mask-GAZE 1K acceptance visualizations</h1><p>Current 1K sources are GQA, Visual7W, and Flickr30k. DocVQA/TextVQA are unavailable in this fixed sample.</p><main>""" + "".join(cards) + "</main></body></html>"
    (output_dir / "overview.html").write_text(document, encoding="utf-8")
    return {"count": len(rows), "category_counts": dict(sorted(category_counts.items())), "overview": str((output_dir / "overview.html").resolve())}


def run_acceptance(input_dir: Path, output_dir: Path, visualization_count: int = 60) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, manifest_by_id, manifest_duplicates = _load_by_id(input_dir / "manifest_1k.jsonl")
    masks, masks_by_id, mask_duplicates = _load_by_id(input_dir / "subject_masks.jsonl")
    tokens, tokens_by_id, token_duplicates = _load_by_id(input_dir / "token_supervision_qwen25vl.jsonl")
    id_sets_equal = set(manifest_by_id) == set(masks_by_id) == set(tokens_by_id)
    source_failures = _verify_original_rows(manifest)
    failures: list[dict[str, Any]] = list(source_failures)
    accepted_rows: list[dict[str, Any]] = []
    coverage_values: list[float] = []
    zero_mask_positive = 0
    zero_mask_negative = 0
    length_mismatches = 0
    nonempty_without_positive = 0
    abnormal_coverage = 0
    rle_failures = 0
    for manifest_row in manifest:
        sample_id = manifest_row["sample_id"]
        if sample_id not in masks_by_id or sample_id not in tokens_by_id:
            failures.append({"sample_id": sample_id, "reason": "missing_cross_file_record"})
            continue
        mask_row = masks_by_id[sample_id]
        token_row = tokens_by_id[sample_id]
        temporal, grid_h, grid_w = (int(value) for value in token_row["image_grid_thw"])
        expected_count = temporal * (grid_h // SPATIAL_MERGE_SIZE) * (grid_w // SPATIAL_MERGE_SIZE)
        array_fields = (
            "mask_token_coverage", "mask_token_contribution", "bbox_token_coverage", "bbox_token_contribution",
            "mask_token_labels", "bbox_token_labels", "token_confidence",
        )
        bad_fields = [field for field in array_fields if len(token_row[field]) != expected_count]
        if int(token_row["visual_token_count"]) != expected_count or bad_fields:
            length_mismatches += 1
            failures.append({"sample_id": sample_id, "reason": "token_length_mismatch", "fields": bad_fields})
        labels = np.asarray(token_row["mask_token_labels"], dtype=np.int8)
        positive_count = int((labels == 1).sum())
        negative_count = int((labels == 0).sum())
        ignore_count = int((labels == -1).sum())
        zero_mask_positive += int(positive_count == 0)
        zero_mask_negative += int(negative_count == 0)
        try:
            subject = decode_rle(mask_row["subject_mask_rle"])
            if list(subject.shape) != [int(mask_row["image_height"]), int(mask_row["image_width"])]:
                raise ValueError("RLE shape differs from recorded image size")
            nonempty = bool(subject.any())
        except Exception as error:
            rle_failures += 1
            nonempty = False
            failures.append({"sample_id": sample_id, "reason": "rle_decode_failure", "detail": str(error)})
        if nonempty and positive_count == 0:
            nonempty_without_positive += 1
            failures.append({"sample_id": sample_id, "reason": "nonempty_mask_without_positive"})
        coverage = float(token_row["actual_mask_mass_coverage"])
        coverage_values.append(coverage)
        if not math.isfinite(coverage) or coverage < float(token_row["adaptive_mass_target"]) - 1e-6:
            abnormal_coverage += 1
            failures.append({"sample_id": sample_id, "reason": "abnormal_adaptive_mass_coverage", "value": coverage})
        gaze_bbox = original_gaze_bbox_labels(
            manifest_row["bboxes"], int(manifest_row["width"]), int(manifest_row["height"]), token_row["image_grid_thw"]
        )
        shape_features = _mask_shape_features(mask_row)
        accepted_rows.append(
            {
                "sample_id": sample_id,
                "source": manifest_row["source"],
                "image_path": manifest_row["image_path"],
                "question": manifest_row["question"],
                "answer": manifest_row["answer"],
                "bboxes": manifest_row["bboxes"],
                "width": int(manifest_row["width"]),
                "height": int(manifest_row["height"]),
                "subject_mask_rle": mask_row["subject_mask_rle"],
                "image_grid_thw": token_row["image_grid_thw"],
                "visual_token_count": expected_count,
                "mask_token_labels": labels.astype(int).tolist(),
                "mask_token_coverage": token_row["mask_token_coverage"],
                "mask_positive_indices": np.flatnonzero(labels == 1).astype(int).tolist(),
                "reliable_background_indices": np.flatnonzero(labels == 0).astype(int).tolist(),
                "mask_positive_count": positive_count,
                "reliable_negative_count": negative_count,
                "ignore_count": ignore_count,
                "actual_mask_mass_coverage": coverage,
                "gaze_bbox_labels": gaze_bbox.astype(np.int8).tolist(),
                "gaze_bbox_positive_indices": np.flatnonzero(gaze_bbox).astype(int).tolist(),
                "gaze_bbox_positive_count": int(gaze_bbox.sum()),
                "gaze_bbox_negative_count": int((~gaze_bbox).sum()),
                "fallback_to_bbox": bool(token_row.get("fallback_to_bbox", False)),
                "mask_area_ratio": float(mask_row["mask_area_ratio"]),
                "bbox_count": len(manifest_row["bboxes"]),
                "elongation": shape_features["elongation"],
                "fill_ratio": shape_features["fill_ratio"],
                "thin_score": float(shape_features["elongation"]) / max(float(shape_features["fill_ratio"]), 1e-6),
                "visual_categories": [],
            }
        )
    duplicate_summary = {
        "manifest": manifest_duplicates,
        "subject_masks": mask_duplicates,
        "token_supervision": token_duplicates,
    }
    gate_checks = {
        "exactly_1000_rows_each": len(manifest) == len(masks) == len(tokens) == 1000,
        "no_duplicate_ids": not any(duplicate_summary.values()),
        "cross_file_id_sets_equal": id_sets_equal,
        "original_rows_verified": not source_failures,
        "all_rles_decode": rle_failures == 0,
        "token_lengths_match_grid": length_mismatches == 0,
        "nonempty_masks_have_positive": nonempty_without_positive == 0,
        "adaptive_mass_coverage_valid": abnormal_coverage == 0,
    }
    training_gate_passed = all(gate_checks.values())
    atomic_write_jsonl(output_dir / "accepted_supervision.jsonl", accepted_rows)
    atomic_write_jsonl(output_dir / "acceptance_failures.jsonl", failures)
    chosen = _select_visualizations(accepted_rows, visualization_count)
    visualization_summary = _write_visualizations(chosen, output_dir / "visualizations")
    summary = {
        "training_gate_passed": training_gate_passed,
        "gate_checks": gate_checks,
        "row_counts": {"manifest": len(manifest), "subject_masks": len(masks), "token_supervision": len(tokens)},
        "source_counts": dict(sorted(Counter(row["source"] for row in accepted_rows).items())),
        "duplicate_ids": duplicate_summary,
        "failure_count": len(failures),
        "mask_positive_token_count": distribution(row["mask_positive_count"] for row in accepted_rows),
        "reliable_negative_token_count": distribution(row["reliable_negative_count"] for row in accepted_rows),
        "ignore_token_count": distribution(row["ignore_count"] for row in accepted_rows),
        "original_gaze_bbox_positive_token_count": distribution(row["gaze_bbox_positive_count"] for row in accepted_rows),
        "adaptive_mask_mass_coverage": distribution(coverage_values),
        "zero_mask_positive_samples": zero_mask_positive,
        "zero_reliable_negative_samples": zero_mask_negative,
        "nonempty_mask_without_positive_samples": nonempty_without_positive,
        "token_length_mismatch_samples": length_mismatches,
        "abnormal_coverage_samples": abnormal_coverage,
        "rle_failure_samples": rle_failures,
        "mask_training_valid_samples": sum(row["mask_positive_count"] > 0 and row["reliable_negative_count"] > 0 for row in accepted_rows),
        "bbox_training_valid_samples": sum(row["gaze_bbox_positive_count"] > 0 and row["gaze_bbox_negative_count"] > 0 for row in accepted_rows),
        "visualizations": visualization_summary,
        "requested_but_unavailable_types": {
            "docvqa": "not present in fixed 1K manifest",
            "textvqa": "not present in fixed 1K manifest",
        },
        "bbox_baseline_definition": "Original GAZE direct inclusive rectangular projection on post-merge token grid; no pixel coverage threshold.",
    }
    atomic_write_json(output_dir / "acceptance_summary.json", summary)
    report = f"""# Mask-GAZE 1K data acceptance

- Training gate: **{'PASS' if training_gate_passed else 'STOP'}**
- Rows: manifest={len(manifest)}, masks={len(masks)}, token supervision={len(tokens)}
- Sources: {summary['source_counts']}
- Positive / reliable negative / ignore token distributions: see `acceptance_summary.json`
- Adaptive mask mass coverage: {summary['adaptive_mask_mass_coverage']}
- Zero positive samples: {zero_mask_positive}
- Zero reliable-negative samples (must be skipped by Mask training): {zero_mask_negative}
- Token/grid length mismatches: {length_mismatches}
- Nonempty masks without a positive token: {nonempty_without_positive}
- Original-row correspondence failures: {len(source_failures)}
- Visualizations: {visualization_summary['count']} (`visualizations/overview.html`)

## Dataset-type coverage

The fixed 1K manifest contains only GQA, Visual7W, and Flickr30k. Therefore DocVQA and TextVQA cannot be represented without changing the fixed experiment set. The visualization audit covers random examples from every available source plus small-target, thin-region, and multi-subject cases.

## BBox baseline

The saved `gaze_bbox_labels` reproduce original GAZE: direct pixel-BBox projection to the spatial-merge token grid with an inclusive binary rectangle. The adaptive 90% BBox labels in the offline-generation file are retained only as diagnostics and are not used for group B.
"""
    (output_dir / "acceptance_report.md").write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Mask-GAZE 1K acceptance gate")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/mask_gaze_1k"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mask_gaze_1k/preexperiment/acceptance"))
    parser.add_argument("--visualization-count", type=int, default=60)
    args = parser.parse_args()
    summary = run_acceptance(args.input_dir, args.output_dir, args.visualization_count)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["training_gate_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
