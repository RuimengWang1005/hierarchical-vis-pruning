from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from torch import nn

from src.audit_subject_mask_supervision import GazeVisionAuditor
from src.mask_gaze_preexperiment import three_value_pairwise_margin_loss
from src.utils import atomic_write_json, atomic_write_jsonl, read_jsonl, seed_everything


SEED = 42
MARGIN = 0.2
LEARNING_RATE = 5e-5
EPOCHS = 3
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 32
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0
BUDGETS = (0.1, 0.2, 0.3)


class EfficientTransformerScorer(nn.Module):
    """Algebraically equivalent implementation of GAZE's TransformerScorer.

    mean_j(q_i @ k_j) equals q_i @ mean_j(k_j), avoiding materializing the
    quadratic attention matrix. Parameter names and initialization are exactly
    compatible with the original scorer checkpoint.
    """

    def __init__(self, in_features: int = 3584, hidden_dim: int = 1792) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.hidden_dim = int(hidden_dim)
        self.k_proj = nn.Linear(in_features, hidden_dim)
        self.q_proj = nn.Linear(in_features, hidden_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        keys = self.k_proj(features)
        queries = self.q_proj(features)
        mean_keys = keys.mean(dim=-2, keepdim=True)
        return (queries * mean_keys).sum(dim=-1) / math.sqrt(self.hidden_dim)


def scorer_state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode("utf-8"))
        raw = state[key].detach().cpu().contiguous().view(torch.uint8)
        digest.update(raw.numpy().tobytes())
    return digest.hexdigest()


def _auditor_config() -> dict[str, Any]:
    return {
        "paths": {"gaze_root": "/home/ruimeng/gaze"},
        "qwen": {
            # The local flash-attn rotary kernel casts q/k to fp32 while keeping
            # bf16 cos/sin and asserts. SDPA is the already-audited pilot path.
            "attention_implementation": "sdpa",
            "min_pixels": 3136,
            "max_pixels": 401408,
        },
    }


def precompute_features(
    acceptance_path: Path,
    checkpoint: Path,
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    feature_dir = output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    rows = list(read_jsonl(acceptance_path))
    auditor = GazeVisionAuditor(checkpoint, _auditor_config(), device)
    auditor.model.budgets = 1.0
    initial_state = {
        key: value.detach().cpu().contiguous()
        for key, value in auditor.model.importance_scorer.state_dict().items()
    }
    initial_path = output_dir / "initial_scorer.safetensors"
    save_file(initial_state, str(initial_path))
    initial_hash = scorer_state_hash(initial_state)
    index_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        destination = feature_dir / f"{row['sample_id']}.safetensors"
        if destination.is_file():
            cached = load_file(str(destination), device="cpu")["features"]
            if list(cached.shape) == [int(row["visual_token_count"]), 3584]:
                index_rows.append(
                    {"sample_id": row["sample_id"], "path": str(destination.resolve()), "shape": list(cached.shape)}
                )
                continue
        with Image.open(row["image_path"]) as opened:
            image = opened.convert("RGB")
        inputs = auditor.processor.image_processor(images=[image], return_tensors="pt")
        grid = [int(value) for value in inputs["image_grid_thw"][0].tolist()]
        if grid != [int(value) for value in row["image_grid_thw"]]:
            raise AssertionError(f"{row['sample_id']}: processor grid mismatch {grid} != {row['image_grid_thw']}")
        pixel_values = inputs["pixel_values"].to(device=auditor.device, dtype=auditor.model.dtype)
        grid_tensor = inputs["image_grid_thw"].to(auditor.device)
        with torch.inference_mode():
            _ = auditor.model(pixel_values, grid_thw=grid_tensor)
            features = auditor.model.last_token_features.detach().cpu().to(torch.bfloat16).contiguous()
        if list(features.shape) != [int(row["visual_token_count"]), 3584]:
            raise AssertionError(f"{row['sample_id']}: feature shape {list(features.shape)}")
        save_file({"features": features}, str(destination))
        index_rows.append({"sample_id": row["sample_id"], "path": str(destination.resolve()), "shape": list(features.shape)})
        if index % 25 == 0 or index == len(rows):
            print(f"[features] {index}/{len(rows)} elapsed={time.perf_counter() - started:.1f}s", flush=True)
    atomic_write_jsonl(output_dir / "feature_index.jsonl", index_rows)
    summary = {
        "sample_count": len(index_rows),
        "feature_dtype": "bfloat16",
        "feature_dimension": 3584,
        "vision_encoder_frozen": True,
        "language_model_loaded": False,
        "initial_scorer_path": str(initial_path.resolve()),
        "initial_scorer_sha256": initial_hash,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_write_json(output_dir / "feature_summary.json", summary)
    return summary


def _load_scorer(path: Path, device: torch.device) -> EfficientTransformerScorer:
    state = load_file(str(path), device="cpu")
    scorer = EfficientTransformerScorer().to(device=device, dtype=torch.float32)
    scorer.load_state_dict({key: value.float() for key, value in state.items()}, strict=True)
    return scorer


def _save_scorer(scorer: EfficientTransformerScorer, destination: Path) -> str:
    state = {key: value.detach().cpu().contiguous() for key, value in scorer.state_dict().items()}
    save_file(state, str(destination))
    return scorer_state_hash(state)


def _cosine_schedule(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps + 1) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))


def train_group(
    group: str,
    rows: list[dict[str, Any]],
    feature_dir: Path,
    initial_scorer: Path,
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    if group not in {"bbox", "mask"}:
        raise ValueError(group)
    torch_device = torch.device(device)
    seed_everything(SEED, deterministic_algorithms=False)
    scorer = _load_scorer(initial_scorer, torch_device).train()
    initial_hash = scorer_state_hash({key: value.detach().cpu() for key, value in scorer.state_dict().items()})
    optimizer = torch.optim.AdamW(
        scorer.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8, weight_decay=WEIGHT_DECAY
    )
    updates_per_epoch = math.ceil(len(rows) / GRADIENT_ACCUMULATION_STEPS)
    total_updates = updates_per_epoch * EPOCHS
    warmup_updates = math.ceil(total_updates * WARMUP_RATIO)
    global_update = 0
    history: list[dict[str, Any]] = []
    group_dir = output_dir / group
    group_dir.mkdir(parents=True, exist_ok=True)
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        order = list(range(len(rows)))
        random.Random(SEED + epoch - 1).shuffle(order)
        epoch_losses: list[float] = []
        skipped = 0
        accumulated = 0
        for position, row_index in enumerate(order, start=1):
            row = rows[row_index]
            labels_key = "gaze_bbox_labels" if group == "bbox" else "mask_token_labels"
            labels = torch.tensor(row[labels_key], dtype=torch.int8, device=torch_device)
            feature_path = feature_dir / f"{row['sample_id']}.safetensors"
            features = load_file(str(feature_path), device="cpu")["features"].to(torch_device, dtype=torch.float32)
            scores = scorer(features)
            loss = three_value_pairwise_margin_loss(scores, labels, margin=MARGIN)
            if loss is None:
                skipped += 1
                continue
            (loss / GRADIENT_ACCUMULATION_STEPS).backward()
            epoch_losses.append(float(loss.detach().cpu()))
            accumulated += 1
            if accumulated == GRADIENT_ACCUMULATION_STEPS or position == len(order):
                torch.nn.utils.clip_grad_norm_(scorer.parameters(), MAX_GRAD_NORM)
                global_update += 1
                scale = _cosine_schedule(global_update - 1, total_updates, warmup_updates)
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = LEARNING_RATE * scale
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
        checkpoint_path = group_dir / f"epoch_{epoch}.safetensors"
        checkpoint_hash = _save_scorer(scorer, checkpoint_path)
        epoch_record = {
            "epoch": epoch,
            "mean_pairwise_loss": float(np.mean(epoch_losses)),
            "median_pairwise_loss": float(np.median(epoch_losses)),
            "processed_samples": len(epoch_losses),
            "skipped_samples": skipped,
            "global_update": global_update,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(epoch_record)
        print(f"[{group}] epoch={epoch} loss={epoch_record['mean_pairwise_loss']:.6f} updates={global_update}", flush=True)
    config = {
        "group": group.upper(),
        "initial_scorer": str(initial_scorer.resolve()),
        "initial_scorer_sha256": initial_hash,
        "training_samples": len(rows),
        "attempted_samples_from_manifest": 1000,
        "excluded_common_invalid_samples": 1000 - len(rows),
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "optimizer": "AdamW",
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "cosine",
        "warmup_ratio": WARMUP_RATIO,
        "warmup_updates": warmup_updates,
        "total_optimizer_updates": total_updates,
        "margin": MARGIN,
        "loss": "full pairwise softplus(-(positive-negative-margin))",
        "labels": "B: original GAZE rectangle 1/outside 0; C: adaptive SAM mask 1/reliable background 0/uncertain -1 ignored",
        "seed": SEED,
        "trainable_module": "VIS importance_scorer only",
        "vision_encoder_frozen": True,
        "language_model_loaded": False,
        "other_modules_trainable": False,
        "feature_cache_dtype": "bfloat16",
        "training_compute_dtype": "float32",
        "history": history,
    }
    atomic_write_json(group_dir / "training_config_and_history.json", config)
    return config


def _load_selection_helpers():
    evaluation_root = Path("/home/ruimeng/gaze/qwen-evaluation1")
    sys.path.insert(0, str(evaluation_root))
    from token_compression.selector_model import _select_focal_indices, _select_indices_with_cosine_penalty

    return _select_focal_indices, _select_indices_with_cosine_penalty


def select_gaze_indices(
    scores: torch.Tensor,
    features: torch.Tensor,
    image_grid_thw: list[int],
    budget: float,
    spatial_ratio: float = 0.1,
    cosine_penalty_weight: float = 1.0,
    cosine_penalty_threshold: float = 0.7,
) -> torch.Tensor:
    """Exact original-strategy Focal-Spatial selection for one image."""
    select_focal, select_cosine = _load_selection_helpers()
    total = int(scores.numel())
    keep = max(1, int(total * float(budget)))
    if keep >= total or total <= 1:
        return torch.arange(total, device=scores.device)
    spatial_count = max(0, min(int(round(keep * spatial_ratio)), keep))
    focal_count = keep - spatial_count
    if focal_count <= 0:
        focal_count = 1
        spatial_count = keep - 1
    diagnostics = {"fallback_count": 0}
    focal = select_focal(
        scores,
        features,
        focal_count,
        cosine_penalty_weight,
        cosine_penalty_threshold,
        False,
        None,
        None,
        0.1,
        0.0,
        diagnostics,
    )
    if focal.numel() != focal_count:
        focal = torch.topk(scores, focal_count).indices
    focal = torch.sort(focal).values
    if spatial_count == 0:
        return focal
    temporal, patch_h, patch_w = (int(value) for value in image_grid_thw)
    height = max(temporal * (patch_h // 2), 1)
    width = max(patch_w // 2, 1)
    is_focal = torch.zeros(total, dtype=torch.bool, device=scores.device)
    is_focal[focal] = True
    non_focal = torch.where(~is_focal)[0]
    non_scores = scores[non_focal]
    non_features = features[non_focal]
    focal_features = features[focal]
    grid_cols = max(1, int(math.ceil(math.sqrt(spatial_count * (float(width) / float(height))))))
    grid_rows = max(1, int(math.ceil(float(spatial_count) / float(grid_cols))))
    row_step = max(1, int(math.ceil(float(height) / float(grid_rows))))
    col_step = max(1, int(math.ceil(float(width) / float(grid_cols))))
    row = (non_focal // width).clamp(min=0, max=height - 1)
    col = (non_focal % width).clamp(min=0, max=width - 1)
    cell_id = (row // row_step).clamp(max=grid_rows - 1) * grid_cols + (col // col_step).clamp(max=grid_cols - 1)
    candidates: list[torch.Tensor] = []
    for cell in range(grid_rows * grid_cols):
        relative = torch.where(cell_id == cell)[0]
        if relative.numel() > 0:
            candidates.append(relative[non_scores[relative].argmax()])
    if candidates:
        candidate_tensor = torch.stack(candidates)
        if candidate_tensor.numel() > spatial_count:
            chosen = select_cosine(
                non_scores[candidate_tensor], non_features[candidate_tensor], spatial_count,
                cosine_penalty_weight, cosine_penalty_threshold, reference_features=focal_features,
            )
            selected_relative = candidate_tensor[chosen]
        else:
            selected_relative = candidate_tensor
    else:
        selected_relative = torch.empty(0, dtype=torch.long, device=scores.device)
    selected_mask = torch.zeros(non_focal.numel(), dtype=torch.bool, device=scores.device)
    if selected_relative.numel() > 0:
        selected_mask[selected_relative] = True
    if selected_relative.numel() < spatial_count:
        remaining = torch.where(~selected_mask)[0]
        need = min(spatial_count - int(selected_relative.numel()), int(remaining.numel()))
        references = focal_features
        if selected_relative.numel() > 0:
            references = torch.cat([references, non_features[selected_relative]], dim=0)
        extra = select_cosine(
            non_scores[remaining], non_features[remaining], need,
            cosine_penalty_weight, cosine_penalty_threshold, reference_features=references,
        )
        selected_relative = torch.cat([selected_relative, remaining[extra]])
    if selected_relative.numel() > spatial_count:
        chosen = select_cosine(
            non_scores[selected_relative], non_features[selected_relative], spatial_count,
            cosine_penalty_weight, cosine_penalty_threshold, reference_features=focal_features,
        )
        selected_relative = selected_relative[chosen]
    spatial = torch.sort(non_focal[selected_relative[:spatial_count]]).values
    return torch.sort(torch.cat([focal, spatial])).values


def _metric_distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values), "mean": float(array.mean()), "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)), "p90": float(np.percentile(array, 90)),
    }


@torch.inference_mode()
def evaluate_checkpoint(
    name: str,
    scorer_path: Path,
    rows: list[dict[str, Any]],
    feature_dir: Path,
    device: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    torch_device = torch.device(device)
    scorer = _load_scorer(scorer_path, torch_device).eval()
    score_margins: list[float] = []
    pairwise_accuracies: list[float] = []
    margin_accuracies: list[float] = []
    recalls = {str(int(b * 100)): [] for b in BUDGETS}
    any_subject = {str(int(b * 100)): [] for b in BUDGETS}
    background_retention = {str(int(b * 100)): [] for b in BUDGETS}
    selected_background_fraction = {str(int(b * 100)): [] for b in BUDGETS}
    per_sample: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        features = load_file(str(feature_dir / f"{row['sample_id']}.safetensors"), device="cpu")["features"].to(torch_device, torch.float32)
        labels = torch.tensor(row["mask_token_labels"], dtype=torch.int8, device=torch_device)
        positives = torch.where(labels == 1)[0]
        negatives = torch.where(labels == 0)[0]
        scores = scorer(features)
        score_margin = None
        pairwise_accuracy = None
        margin_accuracy = None
        if positives.numel() > 0 and negatives.numel() > 0:
            differences = scores[positives].unsqueeze(1) - scores[negatives].unsqueeze(0)
            score_margin = float(scores[positives].mean() - scores[negatives].mean())
            pairwise_accuracy = float((differences > 0).float().mean())
            margin_accuracy = float((differences > MARGIN).float().mean())
            score_margins.append(score_margin)
            pairwise_accuracies.append(pairwise_accuracy)
            margin_accuracies.append(margin_accuracy)
        budget_metrics: dict[str, Any] = {}
        for budget in BUDGETS:
            key = str(int(budget * 100))
            selected = select_gaze_indices(scores, features, row["image_grid_thw"], budget)
            selected_mask = torch.zeros(labels.numel(), dtype=torch.bool, device=torch_device)
            selected_mask[selected] = True
            retained_positive = int(selected_mask[positives].sum())
            retained_negative = int(selected_mask[negatives].sum())
            recall = retained_positive / int(positives.numel())
            has_subject = float(retained_positive > 0)
            bg_retention = retained_negative / int(negatives.numel()) if negatives.numel() else None
            bg_fraction = retained_negative / int(selected.numel())
            recalls[key].append(recall)
            any_subject[key].append(has_subject)
            if bg_retention is not None:
                background_retention[key].append(bg_retention)
            selected_background_fraction[key].append(bg_fraction)
            budget_metrics[key] = {
                "selected_count": int(selected.numel()),
                "subject_recall": recall,
                "retains_subject": bool(has_subject),
                "background_token_retention_rate": bg_retention,
                "selected_background_fraction": bg_fraction,
                "selected_indices": selected.detach().cpu().tolist(),
            }
        per_sample.append(
            {
                "sample_id": row["sample_id"], "source": row["source"],
                "score_margin": score_margin, "pairwise_accuracy": pairwise_accuracy,
                "margin_pairwise_accuracy": margin_accuracy, "budgets": budget_metrics,
            }
        )
        if index % 100 == 0 or index == len(rows):
            print(f"[eval {name}] {index}/{len(rows)} elapsed={time.perf_counter() - started:.1f}s", flush=True)
    result = {
        "name": name,
        "scorer_checkpoint": str(scorer_path.resolve()),
        "sample_count": len(rows),
        "foreground_background_score_gap": _metric_distribution(score_margins),
        "pairwise_ranking_accuracy": _metric_distribution(pairwise_accuracies),
        "margin_satisfied_pairwise_accuracy": _metric_distribution(margin_accuracies),
        "budgets": {
            key: {
                "subject_token_recall": _metric_distribution(recalls[key]),
                "samples_retaining_at_least_one_subject_token": _metric_distribution(any_subject[key]),
                "background_token_retention_rate": _metric_distribution(background_retention[key]),
                "selected_background_fraction": _metric_distribution(selected_background_fraction[key]),
            }
            for key in recalls
        },
        "selection_configuration": {
            "algorithm": "unchanged GAZE original Focal-Spatial",
            "spatial_ratio": 0.1,
            "cosine_penalty_weight": 1.0,
            "cosine_penalty_threshold": 0.7,
            "selector_focal_redundancy_limit": 0,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    return result, per_sample


def run_training_and_evaluation(acceptance_dir: Path, work_dir: Path, device: str) -> dict[str, Any]:
    acceptance_summary = json.loads((acceptance_dir / "acceptance_summary.json").read_text(encoding="utf-8"))
    if not acceptance_summary["training_gate_passed"]:
        raise RuntimeError("Acceptance gate did not pass; training is prohibited")
    all_rows = list(read_jsonl(acceptance_dir / "accepted_supervision.jsonl"))
    common_rows = [
        row for row in all_rows
        if row["mask_positive_count"] > 0 and row["reliable_negative_count"] > 0
        and row["gaze_bbox_positive_count"] > 0 and row["gaze_bbox_negative_count"] > 0
    ]
    atomic_write_jsonl(work_dir / "common_training_manifest.jsonl", common_rows)
    initial_scorer = work_dir / "initial_scorer.safetensors"
    feature_dir = work_dir / "features"
    bbox_config = train_group("bbox", common_rows, feature_dir, initial_scorer, work_dir / "training", device)
    mask_config = train_group("mask", common_rows, feature_dir, initial_scorer, work_dir / "training", device)
    evaluations: list[dict[str, Any]] = []
    eval_dir = work_dir / "token_evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = [("A_original", initial_scorer)]
    for epoch in range(1, EPOCHS + 1):
        checkpoints.extend(
            [
                (f"B_bbox_epoch_{epoch}", work_dir / "training" / "bbox" / f"epoch_{epoch}.safetensors"),
                (f"C_mask_epoch_{epoch}", work_dir / "training" / "mask" / f"epoch_{epoch}.safetensors"),
            ]
        )
    for name, path in checkpoints:
        result, samples = evaluate_checkpoint(name, path, all_rows, feature_dir, device)
        evaluations.append(result)
        atomic_write_json(eval_dir / f"{name}_summary.json", result)
        atomic_write_jsonl(eval_dir / f"{name}_per_sample.jsonl", samples)
    comparison = {
        "acceptance_gate": "PASS",
        "common_training_samples": len(common_rows),
        "excluded_from_both_training_groups": len(all_rows) - len(common_rows),
        "A": {"training": "none", "checkpoint": str(initial_scorer.resolve())},
        "B": bbox_config,
        "C": mask_config,
        "token_evaluations": evaluations,
    }
    atomic_write_json(work_dir / "preexperiment_results.json", comparison)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate Mask-GAZE binary ranking pre-experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--acceptance-dir", type=Path, default=Path("outputs/mask_gaze_1k/preexperiment/acceptance"))
    common.add_argument("--work-dir", type=Path, default=Path("outputs/mask_gaze_1k/preexperiment"))
    common.add_argument("--device", default="cuda:0")
    feature_parser = subparsers.add_parser("features", parents=[common])
    feature_parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("/home/ruimeng/gaze/output_ckpt666/VisionSelector-Qwen2.5-VL-7B"),
    )
    subparsers.add_parser("train-eval", parents=[common])
    args = parser.parse_args()
    if args.command == "features":
        summary = precompute_features(
            args.acceptance_dir / "accepted_supervision.jsonl", args.checkpoint, args.work_dir, args.device
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        results = run_training_and_evaluation(args.acceptance_dir, args.work_dir, args.device)
        print(json.dumps({"common_training_samples": results["common_training_samples"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
