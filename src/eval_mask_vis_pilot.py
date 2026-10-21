from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.mask_vis_online import OnlineVisionEncoder, load_config, load_full_checkpoint_scorer, load_scorer
from src.train_mask_gaze_preexperiment import select_gaze_indices
from src.utils import atomic_write_json, atomic_write_jsonl, read_jsonl


BUDGETS = (0.1, 0.2, 0.3)


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _load_pilot(config: dict[str, Any]) -> list[dict[str, Any]]:
    manifests = {row["sample_id"]: row for row in read_jsonl(config["paths"]["pilot_manifest"])}
    rows: list[dict[str, Any]] = []
    for audit in read_jsonl(config["paths"]["pilot_token_audit"]):
        sample_id = audit["sample_id"]
        if sample_id not in manifests:
            raise RuntimeError(f"pilot audit sample is absent from manifest: {sample_id}")
        manifest = manifests[sample_id]
        token_count = int(audit["visual_token_count"])
        positives = sorted({int(value) for value in audit["adaptive_mask_tokens"]})
        mask_coverage = np.asarray(audit["mask_token_coverage"], dtype=np.float32)
        bbox_coverage = np.asarray(audit["bbox_token_coverage"], dtype=np.float32)
        if len(mask_coverage) != token_count or len(bbox_coverage) != token_count:
            raise RuntimeError(f"{sample_id}: pilot coverage length mismatch")
        negatives = np.flatnonzero((mask_coverage <= 0.01) & (bbox_coverage <= 0.01)).astype(int).tolist()
        rows.append(
            {
                "sample_id": sample_id,
                "image_path": manifest.get("image_path") or manifest.get("image"),
                "image_grid_thw": audit["image_grid_thw"],
                "visual_token_count": token_count,
                "positive_indices": positives,
                "negative_indices": negatives,
            }
        )
    return rows


def _load_any_scorer(path: Path, device: torch.device) -> torch.nn.Module:
    if path.is_dir():
        return load_full_checkpoint_scorer(path, device)
    return load_scorer(path, device).eval()


@torch.inference_mode()
def evaluate(
    config: dict[str, Any],
    original_scorer_path: Path,
    mask_scorer_path: Path,
    output_dir: Path,
    device_name: str,
    max_samples: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_pilot(config)
    if max_samples is not None:
        rows = rows[:max_samples]
    device = torch.device(device_name)
    checkpoint = Path(config["paths"]["gaze_checkpoint"])
    original = _load_any_scorer(original_scorer_path, device)
    mask = _load_any_scorer(mask_scorer_path, device)
    encoder = OnlineVisionEncoder(checkpoint, device_name, config)
    scorers = {"gaze": original, "mask": mask}
    accumulators: dict[str, dict[str, list[float]]] = {}
    for name in scorers:
        metrics = {"score_gap": []}
        for budget in BUDGETS:
            key = str(int(budget * 100))
            metrics[f"recall_{key}"] = []
            metrics[f"any_subject_{key}"] = []
            metrics[f"reliable_background_retention_{key}"] = []
        accumulators[name] = metrics

    per_sample: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        with Image.open(row["image_path"]) as opened:
            image = opened.convert("RGB")
        features = encoder.extract_one(image, row["image_grid_thw"], row["sample_id"])
        positives = set(row["positive_indices"])
        negatives = set(row["negative_indices"])
        if not positives:
            raise RuntimeError(f"{row['sample_id']}: pilot sample has no subject token")
        record: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "positive_count": len(positives),
            "reliable_negative_count": len(negatives),
            "scorers": {},
        }
        for name, scorer in scorers.items():
            scores = scorer(features.float().unsqueeze(0)).squeeze(0)
            if scores.numel() != row["visual_token_count"]:
                raise RuntimeError(f"{row['sample_id']}: scorer length mismatch")
            score_gap = None
            if negatives:
                pos_tensor = torch.tensor(sorted(positives), device=device)
                neg_tensor = torch.tensor(sorted(negatives), device=device)
                score_gap = float((scores[pos_tensor].mean() - scores[neg_tensor].mean()).cpu())
                accumulators[name]["score_gap"].append(score_gap)
            scorer_record: dict[str, Any] = {"subject_background_score_gap": score_gap, "budgets": {}}
            for budget in BUDGETS:
                key = str(int(budget * 100))
                selected_tensor = select_gaze_indices(scores, features.float(), row["image_grid_thw"], budget)
                selected = set(selected_tensor.cpu().tolist())
                recall = len(positives & selected) / len(positives)
                any_subject = float(bool(positives & selected))
                background_retention = len(negatives & selected) / len(negatives) if negatives else None
                accumulators[name][f"recall_{key}"].append(recall)
                accumulators[name][f"any_subject_{key}"].append(any_subject)
                if background_retention is not None:
                    accumulators[name][f"reliable_background_retention_{key}"].append(background_retention)
                scorer_record["budgets"][key] = {
                    "selected_indices": sorted(selected),
                    "subject_recall": recall,
                    "retained_any_subject": bool(any_subject),
                    "reliable_background_retention": background_retention,
                }
            record["scorers"][name] = scorer_record
        per_sample.append(record)
        if index % 20 == 0 or index == len(rows):
            print(f"[pilot] {index}/{len(rows)} elapsed={time.perf_counter() - started:.1f}s", flush=True)
        del features

    summaries: dict[str, Any] = {}
    for name, metrics in accumulators.items():
        summaries[name] = {key: _mean(values) for key, values in metrics.items()}
    delta: dict[str, float | None] = {}
    for key in summaries["gaze"]:
        left, right = summaries["gaze"][key], summaries["mask"][key]
        delta[key] = None if left is None or right is None else right - left
    summary = {
        "sample_count": len(rows),
        "budgets": list(BUDGETS),
        "original_scorer": str(original_scorer_path.resolve()),
        "mask_scorer": str(mask_scorer_path.resolve()),
        "gaze": summaries["gaze"],
        "mask": summaries["mask"],
        "mask_minus_gaze": delta,
        "selection": {
            "spatial_ratio": 0.1,
            "cosine_penalty_weight": 1.0,
            "cosine_penalty_threshold": 0.7,
            "algorithm": "original GAZE Focal-Spatial",
        },
        "elapsed_seconds": time.perf_counter() - started,
        "qwen_visual_feature_cache_generated": False,
    }
    atomic_write_jsonl(output_dir / "per_sample.jsonl", per_sample)
    atomic_write_json(output_dir / "summary.json", summary)
    with (output_dir / "mask_minus_gaze.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "gaze", "mask", "mask_minus_gaze"])
        writer.writeheader()
        for key in summaries["gaze"]:
            writer.writerow(
                {"metric": key, "gaze": summaries["gaze"][key], "mask": summaries["mask"][key], "mask_minus_gaze": delta[key]}
            )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate GAZE and Mask-VIS on independent pilot 200")
    parser.add_argument("--config", required=True)
    parser.add_argument("--original-scorer", required=True)
    parser.add_argument("--mask-scorer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = evaluate(
        load_config(Path(args.config)),
        Path(args.original_scorer),
        Path(args.mask_scorer),
        Path(args.output_dir),
        args.device,
        args.max_samples,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

