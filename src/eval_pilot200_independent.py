from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.audit_subject_mask_supervision import GazeVisionAuditor
from src.generate_mask_gaze_1k import adaptive_indices
from src.train_mask_gaze_preexperiment import (
    BUDGETS,
    _auditor_config,
    _load_scorer,
    select_gaze_indices,
)
from src.utils import atomic_write_json, atomic_write_jsonl, read_jsonl


def _image_path(row: dict[str, Any]) -> str:
    value = row.get("image_path", row.get("image"))
    if not value:
        raise ValueError(f"{row.get('sample_id')}: missing image path")
    return os.path.realpath(os.path.expanduser(str(value)))


def _load_unique(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = list(read_jsonl(path))
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in by_id:
            raise ValueError(f"duplicate sample_id in {path}: {sample_id}")
        by_id[sample_id] = row
    return rows, by_id


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty metric list")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if str(args.device) == "cpu":
        torch.set_num_threads(int(args.cpu_threads))
    pilot_manifest, pilot_by_id = _load_unique(args.pilot_manifest)
    audit_rows, audit_by_id = _load_unique(args.pilot_token_rows)
    one_k_manifest, _ = _load_unique(args.one_k_manifest)

    if set(pilot_by_id) != set(audit_by_id):
        raise AssertionError("pilot manifest and token-audit sample IDs do not match")
    one_k_ids = {str(row["sample_id"]) for row in one_k_manifest}
    one_k_paths = {_image_path(row) for row in one_k_manifest}
    id_overlap = sorted(set(pilot_by_id).intersection(one_k_ids))
    path_overlap = sorted({_image_path(row) for row in pilot_manifest}.intersection(one_k_paths))
    effective_ids = [
        str(row["sample_id"])
        for row in pilot_manifest
        if str(row["sample_id"]) not in one_k_ids and _image_path(row) not in one_k_paths
    ]
    if not effective_ids:
        raise RuntimeError("no non-overlapping pilot samples remain")

    mapping_mismatches: list[str] = []
    empty_foreground: list[str] = []
    grid_length_mismatches: list[str] = []
    effective_rows: list[dict[str, Any]] = []
    for sample_id in effective_ids:
        row = audit_by_id[sample_id]
        contribution = np.asarray(row["mask_contribution"], dtype=np.float64)
        expected_count = int(row["visual_token_count"])
        if contribution.size != expected_count:
            grid_length_mismatches.append(sample_id)
            continue
        foreground = adaptive_indices(contribution, 0.90).tolist()
        if foreground != row["adaptive_mask_tokens"]:
            mapping_mismatches.append(sample_id)
            continue
        if not foreground:
            empty_foreground.append(sample_id)
            continue
        row = dict(row)
        row["adaptive_mask_tokens"] = foreground
        effective_rows.append(row)
    if mapping_mismatches or empty_foreground or grid_length_mismatches:
        raise AssertionError(
            "pilot token validation failed: "
            f"mapping={len(mapping_mismatches)}, empty={len(empty_foreground)}, "
            f"length={len(grid_length_mismatches)}"
        )

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    overlap = {
        "pilot_sample_count": len(pilot_manifest),
        "one_k_sample_count": len(one_k_manifest),
        "sample_id_overlap_count": len(id_overlap),
        "sample_id_overlap": id_overlap,
        "image_path_overlap_count": len(path_overlap),
        "image_path_overlap": path_overlap,
        "effective_nonoverlap_sample_count": len(effective_rows),
    }
    atomic_write_json(output_dir / "overlap_check.json", overlap)

    condition_names = ("B_bbox_epoch_3", "C_mask_epoch_3")
    recalls = {name: {str(int(budget * 100)): [] for budget in BUDGETS} for name in condition_names}
    retained = {name: {str(int(budget * 100)): [] for budget in BUDGETS} for name in condition_names}
    progress_path = output_dir / "per_sample_metrics.jsonl"
    per_sample = list(read_jsonl(progress_path)) if progress_path.is_file() else []
    valid_effective_ids = {str(row["sample_id"]) for row in effective_rows}
    if any(str(row.get("sample_id")) not in valid_effective_ids for row in per_sample):
        raise AssertionError("resume checkpoint contains a sample outside the effective pilot set")
    completed_ids = {str(row["sample_id"]) for row in per_sample}
    if len(completed_ids) != len(per_sample):
        raise AssertionError("resume checkpoint contains duplicate sample IDs")
    for sample_payload in per_sample:
        for name in condition_names:
            for budget in BUDGETS:
                key = str(int(budget * 100))
                value = sample_payload["budgets"][name][key]
                recalls[name][key].append(float(value["subject_recall"]))
                retained[name][key].append(float(value["retains_at_least_one_subject_token"]))

    device = torch.device(args.device)
    scorers = {
        "B_bbox_epoch_3": _load_scorer(args.b_checkpoint, device).eval(),
        "C_mask_epoch_3": _load_scorer(args.c_checkpoint, device).eval(),
    }
    remaining_rows = [row for row in effective_rows if str(row["sample_id"]) not in completed_ids]
    auditor = None
    if remaining_rows:
        auditor = GazeVisionAuditor(args.gaze_checkpoint, _auditor_config(), args.device)
        auditor.model.budgets = 1.0
    started = time.perf_counter()

    for index, row in enumerate(remaining_rows, start=1):
        with Image.open(row["image_path"]) as opened:
            image = opened.convert("RGB")
        assert auditor is not None
        inputs = auditor.processor.image_processor(images=[image], return_tensors="pt")
        grid = [int(value) for value in inputs["image_grid_thw"][0].tolist()]
        expected_grid = [int(value) for value in row["image_grid_thw"]]
        if grid != expected_grid:
            raise AssertionError(f"{row['sample_id']}: processor grid {grid} != {expected_grid}")
        pixel_values = inputs["pixel_values"].to(device=auditor.device, dtype=auditor.model.dtype)
        grid_tensor = inputs["image_grid_thw"].to(auditor.device)
        _ = auditor.model(pixel_values, grid_thw=grid_tensor)
        features = auditor.model.last_token_features.detach().to(device=device, dtype=torch.float32)
        if features.shape != (int(row["visual_token_count"]), 3584):
            raise AssertionError(f"{row['sample_id']}: feature shape {tuple(features.shape)}")
        positives = torch.tensor(row["adaptive_mask_tokens"], dtype=torch.long, device=device)
        sample_payload: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "source": row["source"],
            "subject_token_count": int(positives.numel()),
            "budgets": {},
        }
        for name, scorer in scorers.items():
            scores = scorer(features)
            condition_payload: dict[str, Any] = {}
            for budget in BUDGETS:
                key = str(int(budget * 100))
                selected = select_gaze_indices(scores, features, grid, budget)
                selected_mask = torch.zeros(scores.numel(), dtype=torch.bool, device=device)
                selected_mask[selected] = True
                hit_count = int(selected_mask[positives].sum().item())
                recall = hit_count / int(positives.numel())
                has_subject = float(hit_count > 0)
                recalls[name][key].append(recall)
                retained[name][key].append(has_subject)
                condition_payload[key] = {
                    "subject_recall": recall,
                    "retains_at_least_one_subject_token": bool(has_subject),
                }
            sample_payload["budgets"][name] = condition_payload
        per_sample.append(sample_payload)
        total_completed = len(per_sample)
        if total_completed % 5 == 0 or index == len(remaining_rows):
            atomic_write_jsonl(progress_path, per_sample)
            atomic_write_json(
                output_dir / "run_state.json",
                {
                    "completed_sample_count": total_completed,
                    "target_sample_count": len(effective_rows),
                    "complete": total_completed == len(effective_rows),
                },
            )
        if total_completed % 25 == 0 or index == len(remaining_rows):
            elapsed = time.perf_counter() - started
            print(
                f"[pilot-independent] {total_completed}/{len(effective_rows)} "
                f"this_run_elapsed={elapsed:.1f}s",
                flush=True,
            )

    metrics: dict[str, Any] = {}
    for budget in BUDGETS:
        key = str(int(budget * 100))
        b_recall = _mean(recalls["B_bbox_epoch_3"][key])
        c_recall = _mean(recalls["C_mask_epoch_3"][key])
        b_retained = _mean(retained["B_bbox_epoch_3"][key])
        c_retained = _mean(retained["C_mask_epoch_3"][key])
        metrics[key] = {
            "B_subject_recall": b_recall,
            "C_subject_recall": c_recall,
            "C_minus_B_subject_recall": c_recall - b_recall,
            "B_retains_at_least_one_subject_token_ratio": b_retained,
            "C_retains_at_least_one_subject_token_ratio": c_retained,
            "C_minus_B_retains_at_least_one_subject_token_ratio": c_retained - b_retained,
        }

    passed = bool(
        metrics["10"]["C_minus_B_subject_recall"] > 0.0
        and metrics["20"]["C_minus_B_subject_recall"] > 0.0
    )
    summary = {
        "evaluation": "pilot_200_independent_B_bbox_epoch_3_vs_C_mask_epoch_3",
        "overlap": overlap,
        "adaptive_mapping": {
            "target_mass": 0.90,
            "algorithm": "same stable descending-contribution minimal prefix used by mask_gaze_1k",
            "mapping_mismatch_count": len(mapping_mismatches),
            "empty_foreground_count": len(empty_foreground),
            "samples_without_reliable_background_included": sum(
                int(row.get("reliable_background_token_count", 0)) == 0 for row in effective_rows
            ),
        },
        "checkpoints": {
            "B_bbox_epoch_3": str(args.b_checkpoint.resolve()),
            "C_mask_epoch_3": str(args.c_checkpoint.resolve()),
        },
        "selection_configuration": {
            "budgets": [0.10, 0.20, 0.30],
            "algorithm": "unchanged GAZE original Focal-Spatial",
            "spatial_ratio": 0.1,
            "cosine_penalty_weight": 1.0,
            "cosine_penalty_threshold": 0.7,
            "selector_focal_redundancy_limit": 0,
        },
        "metrics": metrics,
        "decision_rule": "pass iff C-B subject recall is positive at both 10% and 20% budgets",
        "initial_unseen_generalization_supported": passed,
        "elapsed_seconds": time.perf_counter() - started,
        "resumed_sample_count": len(completed_ids),
        "feature_cache_written": False,
        "training_performed": False,
        "device": str(args.device),
        "cpu_threads": int(args.cpu_threads) if str(args.device) == "cpu" else None,
    }
    atomic_write_jsonl(progress_path, per_sample)
    atomic_write_json(output_dir / "summary.json", summary)

    lines = [
        "# Pilot 200 independent Token evaluation",
        "",
        f"- Pilot / 1K sample-ID overlap: **{len(id_overlap)}**",
        f"- Pilot / 1K normalized image-path overlap: **{len(path_overlap)}**",
        f"- Effective unseen samples: **{len(effective_rows)}**",
        "- Training performed: **False**",
        "- Feature cache written: **False**",
        "- Adaptive mapping mismatch: **0**",
        f"- Samples without reliable background (still included): **{sum(int(row.get('reliable_background_token_count', 0)) == 0 for row in effective_rows)}**",
        "",
        "| Budget | B subject Recall | C subject Recall | C-B Recall | B retains >=1 | C retains >=1 | C-B retains >=1 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("10", "20", "30"):
        row = metrics[key]
        lines.append(
            f"| {key}% | {row['B_subject_recall']:.6f} | {row['C_subject_recall']:.6f} | "
            f"{row['C_minus_B_subject_recall']:+.6f} | "
            f"{row['B_retains_at_least_one_subject_token_ratio']:.6f} | "
            f"{row['C_retains_at_least_one_subject_token_ratio']:.6f} | "
            f"{row['C_minus_B_retains_at_least_one_subject_token_ratio']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            (
                "C is better than B on subject Recall at both 10% and 20%; "
                "the requested initial unseen-sample generalization gate **passes**."
                if passed
                else "C is not better than B at both 10% and 20%; the requested initial unseen-sample generalization gate **does not pass**."
            ),
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Independent no-training B/C token evaluation on pilot 200")
    parser.add_argument("--pilot-manifest", type=Path, default=root / "outputs/pilot_200/manifest.jsonl")
    parser.add_argument(
        "--pilot-token-rows", type=Path, default=root / "outputs/pilot_200/zero_step_audit/per_sample.jsonl"
    )
    parser.add_argument("--one-k-manifest", type=Path, default=root / "outputs/mask_gaze_1k/manifest_1k.jsonl")
    parser.add_argument(
        "--gaze-checkpoint", type=Path, default=Path("/home/ruimeng/gaze/output_ckpt666/VisionSelector-Qwen2.5-VL-7B")
    )
    parser.add_argument(
        "--b-checkpoint",
        type=Path,
        default=root / "outputs/mask_gaze_1k/preexperiment/training/bbox/epoch_3.safetensors",
    )
    parser.add_argument(
        "--c-checkpoint",
        type=Path,
        default=root / "outputs/mask_gaze_1k/preexperiment/training/mask/epoch_3.safetensors",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs/mask_gaze_1k/preexperiment/pilot200_independent",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=48)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
