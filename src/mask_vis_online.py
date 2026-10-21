from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from src.train_mask_gaze_preexperiment import select_gaze_indices
from src.utils import append_jsonl, atomic_write_json, atomic_write_jsonl, read_jsonl, seed_everything, sha256_file


FEATURE_DIM = 3584
HIDDEN_DIM = 1792
FORBIDDEN_FEATURE_SUFFIXES = {".npy", ".npz"}


def _load_original_scorer_class():
    root = Path("/home/ruimeng/gaze/qwen-vl-finetune")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from compression_method.selector_scorer import TransformerScorer

    return TransformerScorer


def make_scorer(seed: int) -> nn.Module:
    """Construct the unchanged GAZE VIS scorer with its original initialization."""
    seed_everything(seed, deterministic_algorithms=False)
    scorer_class = _load_original_scorer_class()
    return scorer_class(in_features=FEATURE_DIM, hidden_dim=HIDDEN_DIM, init_scale=0.0001)


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def save_scorer(module: nn.Module, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu().contiguous() for name, value in module.state_dict().items()}
    save_file(state, str(path))
    return sha256_file(path)


def load_scorer(path: Path, device: torch.device) -> nn.Module:
    scorer = make_scorer(0)
    scorer.load_state_dict(load_file(str(path), device="cpu"), strict=True)
    return scorer.to(device=device, dtype=torch.float32)


def normalize_image_path(value: str) -> str:
    return os.path.realpath(os.path.abspath(os.path.expanduser(value)))


def expected_token_count(grid: Iterable[int], merge_size: int = 2) -> int:
    t, h, w = (int(value) for value in grid)
    if h % merge_size or w % merge_size:
        raise ValueError(f"grid {t, h, w} is not divisible by merge_size={merge_size}")
    return t * (h // merge_size) * (w // merge_size)


def prepare_valid_manifest(config: dict[str, Any], output: Path | None = None) -> dict[str, Any]:
    paths = config["paths"]
    manifest_path = Path(paths["manifest_20k"])
    masks_path = Path(paths["subject_masks_20k"])
    tokens_path = Path(paths["token_supervision_20k"])
    failures_path = Path(paths["failures_20k"])
    pilot_path = Path(paths["pilot_manifest"])
    output = output or Path(paths["valid_training_manifest"])

    pilot_ids: set[str] = set()
    pilot_images: set[str] = set()
    for row in read_jsonl(pilot_path):
        pilot_ids.add(str(row["sample_id"]))
        image = row.get("image_path") or row.get("image")
        if image:
            pilot_images.add(normalize_image_path(str(image)))

    valid_rows: list[dict[str, Any]] = []
    counts = {
        "input": 0,
        "fallback_to_bbox": 0,
        "zero_positive": 0,
        "zero_reliable_negative": 0,
        "excluded_union": 0,
        "valid": 0,
        "pilot_id_overlap": 0,
        "pilot_image_overlap": 0,
    }
    seen: set[str] = set()
    iterators = (read_jsonl(manifest_path), read_jsonl(masks_path), read_jsonl(tokens_path))
    for line_number, triple in enumerate(zip_longest(*iterators), start=1):
        if any(row is None for row in triple):
            raise RuntimeError(f"20K JSONL length mismatch at logical line {line_number}")
        manifest, mask, token = triple
        sample_ids = {str(manifest["sample_id"]), str(mask["sample_id"]), str(token["sample_id"])}
        if len(sample_ids) != 1:
            raise RuntimeError(f"sample_id mismatch at line {line_number}: {sorted(sample_ids)}")
        sample_id = sample_ids.pop()
        if sample_id in seen:
            raise RuntimeError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        counts["input"] += 1
        labels = token.get("mask_token_labels")
        if not isinstance(labels, list):
            raise RuntimeError(f"{sample_id}: mask_token_labels is missing")
        unexpected_labels = sorted(set(labels) - {-1, 0, 1})
        if unexpected_labels:
            raise RuntimeError(f"{sample_id}: unsupported token labels {unexpected_labels}")
        grid = [int(value) for value in token["image_grid_thw"]]
        visual_count = int(token["visual_token_count"])
        if expected_token_count(grid) != visual_count or len(labels) != visual_count:
            raise RuntimeError(
                f"{sample_id}: grid/count/label mismatch: grid={grid}, count={visual_count}, labels={len(labels)}"
            )
        positive_count = sum(value == 1 for value in labels)
        negative_count = sum(value == 0 for value in labels)
        if positive_count != int(token.get("mask_positive_count", positive_count)):
            raise RuntimeError(f"{sample_id}: stored positive count is inconsistent")
        if negative_count != int(token.get("reliable_negative_count", negative_count)):
            raise RuntimeError(f"{sample_id}: stored negative count is inconsistent")
        fallback = bool(mask.get("fallback_to_bbox") or token.get("fallback_to_bbox"))
        zero_positive = positive_count == 0
        zero_negative = negative_count == 0
        counts["fallback_to_bbox"] += int(fallback)
        counts["zero_positive"] += int(zero_positive)
        counts["zero_reliable_negative"] += int(zero_negative)
        image_path = str(manifest.get("image_path") or manifest.get("image"))
        id_overlap = sample_id in pilot_ids
        image_overlap = normalize_image_path(image_path) in pilot_images
        counts["pilot_id_overlap"] += int(id_overlap)
        counts["pilot_image_overlap"] += int(image_overlap)
        if id_overlap or image_overlap:
            raise RuntimeError(f"pilot 200 leakage detected: {sample_id} ({image_path})")
        if fallback or zero_positive or zero_negative:
            counts["excluded_union"] += 1
            continue
        valid_rows.append(
            {
                "sample_id": sample_id,
                "source": manifest.get("source", token.get("source")),
                "image_path": image_path,
                "image_grid_thw": grid,
                "visual_token_count": visual_count,
                "mask_token_labels": [int(value) for value in labels],
                "mask_positive_count": positive_count,
                "reliable_negative_count": negative_count,
                "ignore_count": sum(value == -1 for value in labels),
            }
        )
    counts["valid"] = len(valid_rows)
    if counts["input"] != 20_000:
        raise RuntimeError(f"expected 20000 source records, got {counts['input']}")
    failure_count = sum(1 for _ in read_jsonl(failures_path)) if failures_path.is_file() else 0
    atomic_write_jsonl(output, valid_rows)
    summary = {
        **counts,
        "failures_file_rows": failure_count,
        "verified_pre_bbox_scorer": None,
        "initialization_fallback": "original GAZE TransformerScorer constructor, seed 42",
        "output": str(output.resolve()),
        "source_files": {
            "manifest": str(manifest_path.resolve()),
            "masks": str(masks_path.resolve()),
            "tokens": str(tokens_path.resolve()),
            "failures": str(failures_path.resolve()),
        },
    }
    atomic_write_json(output.with_suffix(".summary.json"), summary)
    return summary


class JsonlImageDataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.offsets: list[int] = []
        with path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            row = json.loads(handle.readline())
        with Image.open(row["image_path"]) as opened:
            row["pil_image"] = opened.convert("RGB")
        return row


def collate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return records


class _ZeroScorer(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.zeros(features.shape[:-1], device=features.device, dtype=features.dtype)


class OnlineVisionEncoder:
    """Frozen, vision-only GAZE/Qwen encoder. It never writes token features."""

    def __init__(self, checkpoint: Path, device: str, config: dict[str, Any]) -> None:
        eval_root = Path(config["paths"]["gaze_eval_root"])
        if str(eval_root) not in sys.path:
            sys.path.insert(0, str(eval_root))
        os.environ["SELECTOR_STORE_VIS_ATTRS"] = "1"
        from token_compression.selector_model import Qwen2_5_VisionTransformerPretrainedModel_Selector
        from transformers import AutoConfig, AutoProcessor

        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        full_config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
        vision_config = full_config.vision_config
        vision_config._attn_implementation = str(config["qwen"].get("attention_implementation", "sdpa"))
        self.model = Qwen2_5_VisionTransformerPretrainedModel_Selector(vision_config)
        index = json.loads((checkpoint / "model.safetensors.index.json").read_text(encoding="utf-8"))
        files = sorted({name for key, name in index["weight_map"].items() if key.startswith("visual.")})
        state: dict[str, torch.Tensor] = {}
        for filename in files:
            with safe_open(str(checkpoint / filename), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key.startswith("visual.") and not key.startswith("visual.importance_scorer."):
                        state[key.removeprefix("visual.")] = handle.get_tensor(key)
        incompatible = self.model.load_state_dict(state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = [key for key in incompatible.missing_keys if not key.startswith("importance_scorer.")]
        if missing or unexpected:
            raise RuntimeError(f"vision state mismatch: missing={missing}, unexpected={unexpected}")
        del state
        self.model.importance_scorer = _ZeroScorer()
        self.model.budgets = 1.0
        dtype_name = str(config["qwen"].get("vision_dtype", "bfloat16"))
        self.dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
        self.model = self.model.to(device=self.device, dtype=self.dtype).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True, use_fast=False)
        processor = self.processor.image_processor
        processor.min_pixels = int(config["qwen"]["min_pixels"])
        processor.max_pixels = int(config["qwen"]["max_pixels"])
        processor.size = {"shortest_edge": processor.min_pixels, "longest_edge": processor.max_pixels}
        self.checkpoint = checkpoint

    @torch.inference_mode()
    def extract_one(self, image: Image.Image, expected_grid: list[int], sample_id: str) -> torch.Tensor:
        inputs = self.processor.image_processor(images=[image], return_tensors="pt")
        grid = [int(value) for value in inputs["image_grid_thw"][0].tolist()]
        if grid != [int(value) for value in expected_grid]:
            raise RuntimeError(f"{sample_id}: online processor grid {grid} != labels grid {expected_grid}")
        pixels = inputs["pixel_values"].to(device=self.device, dtype=self.dtype)
        grid_tensor = inputs["image_grid_thw"].to(self.device)
        _ = self.model(pixels, grid_thw=grid_tensor)
        features = self.model.last_token_features
        if features is None:
            raise RuntimeError("GAZE vision model did not expose last_token_features")
        expected = expected_token_count(grid)
        if tuple(features.shape) != (expected, FEATURE_DIM):
            raise RuntimeError(f"{sample_id}: online features {tuple(features.shape)} != {(expected, FEATURE_DIM)}")
        return features


def pairwise_margin_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
    chunk_size: int = 256,
) -> torch.Tensor | None:
    if scores.ndim != 1 or labels.ndim != 1 or scores.numel() != labels.numel():
        raise ValueError(f"score/label mismatch: scores={tuple(scores.shape)}, labels={tuple(labels.shape)}")
    positives = scores[labels == 1].float()
    negatives = scores[labels == 0].float()
    if positives.numel() == 0 or negatives.numel() == 0:
        return None
    total = scores.new_zeros((), dtype=torch.float32)
    pair_count = positives.numel() * negatives.numel()
    for start in range(0, positives.numel(), chunk_size):
        positive_chunk = positives[start : start + chunk_size]
        total = total + torch.nn.functional.softplus(
            -(positive_chunk[:, None] - negatives[None, :] - float(margin))
        ).sum()
    return total / pair_count


def _cosine_lambda(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return float(step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps + 1) / max(1, total_steps - warmup_steps)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def log(self, event: str, **payload: Any) -> None:
        record = {"event": event, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **payload}
        print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
        append_jsonl(self.path, record)


def _git_snapshot(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(args, cwd=project_root, text=True, capture_output=True, check=False)
        return completed.stdout.strip()

    return {"commit": run("git", "rev-parse", "HEAD"), "status": run("git", "status", "--short")}


def _save_training_checkpoint(
    path: Path,
    scorer: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    state: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(
        {
            "scorer": {key: value.detach().cpu() for key, value in scorer.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "training_state": state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        temporary,
    )
    os.replace(temporary, path)


def scan_for_feature_cache(output_dir: Path) -> list[str]:
    violations: list[str] = []
    if not output_dir.exists():
        return violations
    allowed_safetensors = {"initial_scorer.safetensors"}
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(output_dir)
        if any(part.lower() in {"feature", "features", "feature_cache"} for part in relative.parts):
            violations.append(str(relative))
        elif path.suffix.lower() in FORBIDDEN_FEATURE_SUFFIXES:
            violations.append(str(relative))
        elif path.suffix == ".safetensors" and not (
            path.name in allowed_safetensors or path.name.startswith("scorer_epoch_")
        ):
            violations.append(str(relative))
        elif path.suffix in {".pt", ".pth"} and not path.name.startswith("checkpoint_"):
            violations.append(str(relative))
    return violations


def train(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir or config["paths"]["training_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(output_dir / "train_log.jsonl")
    valid_path = Path(config["paths"]["valid_training_manifest"])
    if not valid_path.is_file():
        prepare_valid_manifest(config, valid_path)
    dataset = JsonlImageDataset(valid_path)
    sample_limit = min(len(dataset), args.max_samples or len(dataset))
    if sample_limit <= 0:
        raise RuntimeError("no training samples")
    seed_everything(args.seed, deterministic_algorithms=False)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    resumed = bool(args.resume)
    if resumed:
        scorer = make_scorer(args.seed).to(device=device, dtype=torch.float32)
        init_description = {"mode": "resume", "path": str(Path(args.resume).resolve())}
    elif args.scorer_init:
        init_path = Path(args.scorer_init)
        scorer = load_scorer(init_path, device)
        init_description = {
            "mode": "explicit_pre_bbox",
            "path": str(init_path.resolve()),
            "sha256": sha256_file(init_path),
        }
    elif args.init_mode == "original_random":
        scorer = make_scorer(args.seed).to(device=device, dtype=torch.float32)
        init_path = output_dir / "initial_scorer.safetensors"
        init_description = {
            "mode": "original_random",
            "seed": args.seed,
            "path": str(init_path.resolve()),
            "sha256": save_scorer(scorer, init_path),
        }
    else:
        raise RuntimeError("No verified pre-BBox scorer exists; use --init-mode original_random or --scorer-init PATH")
    init_description["parameter_count"] = parameter_count(scorer)
    logger.log("initialization", **init_description)

    optimizer = torch.optim.AdamW(
        scorer.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=float(config["training"]["weight_decay"]),
    )
    updates_per_epoch = math.ceil(math.ceil(sample_limit / args.batch_size) / args.gradient_accumulation_steps)
    total_updates = max(1, updates_per_epoch * args.epochs)
    warmup_steps = math.ceil(total_updates * float(config["training"]["warmup_ratio"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _cosine_lambda(step, total_updates, warmup_steps)
    )
    state = {"epoch": 0, "sample_offset": 0, "global_update": 0}
    if resumed:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        scorer.load_state_dict(payload["scorer"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        state.update(payload["training_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        if torch.cuda.is_available() and payload.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
        logger.log("resume", **state)

    snapshot = {
        "config": config,
        "cli": vars(args),
        "initialization": init_description,
        "git": _git_snapshot(Path(__file__).resolve().parents[1]),
        "seed": args.seed,
        "language_model_loaded": False,
        "vision_encoder_frozen": True,
        "qwen_visual_feature_cache_generated": False,
    }
    (output_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8")
    encoder = OnlineVisionEncoder(Path(config["paths"]["gaze_checkpoint"]), args.device, config)
    if any(parameter.requires_grad for parameter in encoder.model.parameters()):
        raise RuntimeError("vision encoder unexpectedly has trainable parameters")

    scorer.train()
    peak_memory = 0
    run_started = time.perf_counter()
    stop = False
    for epoch_index in range(int(state["epoch"]), args.epochs):
        order = list(range(sample_limit))
        random.Random(args.seed + epoch_index).shuffle(order)
        start_offset = int(state["sample_offset"]) if epoch_index == int(state["epoch"]) else 0
        subset = Subset(dataset, order[start_offset:])
        loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_records,
            pin_memory=device.type == "cuda",
        )
        losses: list[float] = []
        effective_samples = 0
        skipped = 0
        pending = 0
        processed = start_offset
        optimizer.zero_grad(set_to_none=True)
        epoch_started = time.perf_counter()
        for batch in loader:
            image_losses: list[torch.Tensor] = []
            for row in batch:
                features = encoder.extract_one(row.pop("pil_image"), row["image_grid_thw"], row["sample_id"])
                labels = torch.tensor(row["mask_token_labels"], dtype=torch.int8, device=device)
                if features.shape[0] != labels.numel() or features.shape[0] != int(row["visual_token_count"]):
                    logger.log(
                        "length_mismatch",
                        sample_id=row["sample_id"],
                        features=int(features.shape[0]),
                        labels=int(labels.numel()),
                        visual_token_count=int(row["visual_token_count"]),
                    )
                    raise RuntimeError(f"{row['sample_id']}: score/label/grid length mismatch")
                scores = scorer(features.float().unsqueeze(0)).squeeze(0)
                loss = pairwise_margin_loss(scores, labels, args.margin, int(config["training"]["pair_chunk_size"]))
                if loss is None:
                    skipped += 1
                else:
                    image_losses.append(loss)
                    losses.append(float(loss.detach().cpu()))
                    effective_samples += 1
                del features, labels, scores
            processed += len(batch)
            if not image_losses:
                continue
            batch_loss = torch.stack(image_losses).mean()
            (batch_loss / args.gradient_accumulation_steps).backward()
            pending += 1
            if pending == args.gradient_accumulation_steps or processed >= sample_limit:
                if pending < args.gradient_accumulation_steps:
                    correction = args.gradient_accumulation_steps / pending
                    for parameter in scorer.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                torch.nn.utils.clip_grad_norm_(scorer.parameters(), float(config["training"]["max_grad_norm"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
                state.update(epoch=epoch_index, sample_offset=processed, global_update=int(state["global_update"]) + 1)
                _save_training_checkpoint(output_dir / "checkpoint_last.pt", scorer, optimizer, scheduler, state)
                logger.log("optimizer_step", **state, loss=float(batch_loss.detach().cpu()))
                if device.type == "cuda":
                    peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device))
                if args.max_steps and int(state["global_update"]) >= args.max_steps:
                    stop = True
                    break
            if device.type == "cuda":
                peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device))
        if stop and processed < sample_limit:
            break
        state.update(epoch=epoch_index + 1, sample_offset=0)
        scorer_path = output_dir / f"scorer_epoch_{epoch_index + 1}.safetensors"
        scorer_hash = save_scorer(scorer, scorer_path)
        _save_training_checkpoint(output_dir / "checkpoint_last.pt", scorer, optimizer, scheduler, state)
        logger.log(
            "epoch_end",
            epoch=epoch_index + 1,
            mean_loss=float(np.mean(losses)) if losses else None,
            median_loss=float(np.median(losses)) if losses else None,
            effective_samples=effective_samples,
            skipped_samples=skipped,
            optimizer_updates=int(state["global_update"]),
            elapsed_seconds=time.perf_counter() - epoch_started,
            scorer_path=str(scorer_path.resolve()),
            scorer_sha256=scorer_hash,
        )
        if stop:
            break
    result = {
        **state,
        "wall_seconds": time.perf_counter() - run_started,
        "peak_cuda_memory_bytes": peak_memory,
        "feature_cache_violations": scan_for_feature_cache(output_dir),
        "formal_20k_completed": int(state["epoch"]) >= args.epochs and sample_limit == len(dataset),
    }
    atomic_write_json(output_dir / "run_summary.json", result)
    if result["feature_cache_violations"]:
        raise RuntimeError(f"feature cache artifacts detected: {result['feature_cache_violations']}")
    return result


def check_data(config: dict[str, Any], count: int = 20) -> dict[str, Any]:
    dataset = JsonlImageDataset(Path(config["paths"]["valid_training_manifest"]))
    checked: list[dict[str, Any]] = []
    for index in range(min(count, len(dataset))):
        row = dataset[index]
        labels = row["mask_token_labels"]
        checked.append(
            {
                "sample_id": row["sample_id"],
                "image_size": list(row.pop("pil_image").size),
                "token_count": len(labels),
                "positive": sum(value == 1 for value in labels),
                "negative": sum(value == 0 for value in labels),
                "ignore": sum(value == -1 for value in labels),
            }
        )
    result = {"checked": len(checked), "samples": checked}
    return result


def load_full_checkpoint_scorer(checkpoint: Path, device: torch.device) -> nn.Module:
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text(encoding="utf-8"))
    prefix = "visual.importance_scorer."
    keys = {key: filename for key, filename in index["weight_map"].items() if key.startswith(prefix)}
    state: dict[str, torch.Tensor] = {}
    for filename in sorted(set(keys.values())):
        with safe_open(str(checkpoint / filename), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(prefix):
                    state[key.removeprefix(prefix)] = handle.get_tensor(key)
    scorer = make_scorer(0)
    scorer.load_state_dict(state, strict=True)
    return scorer.to(device=device, dtype=torch.float32).eval()


def online_consistency_check(config: dict[str, Any], device: str, output: Path) -> dict[str, Any]:
    from src.audit_subject_mask_supervision import GazeVisionAuditor

    row = next(read_jsonl(config["paths"]["valid_training_manifest"]))
    with Image.open(row["image_path"]) as opened:
        image = opened.convert("RGB")
    checkpoint = Path(config["paths"]["gaze_checkpoint"])
    auditor_config = {
        "paths": {"gaze_root": str(Path(config["paths"]["gaze_eval_root"]).parent)},
        "qwen": {
            "attention_implementation": config["qwen"]["attention_implementation"],
            "min_pixels": config["qwen"]["min_pixels"],
            "max_pixels": config["qwen"]["max_pixels"],
        },
    }
    reference = GazeVisionAuditor(checkpoint, auditor_config, device)
    inference = reference.infer(image, row["image_grid_thw"])
    reference.model.budgets = 1.0
    inputs = reference.processor.image_processor(images=[image], return_tensors="pt")
    with torch.inference_mode():
        _ = reference.model(
            inputs["pixel_values"].to(reference.device, dtype=reference.model.dtype),
            grid_thw=inputs["image_grid_thw"].to(reference.device),
        )
        reference_features = reference.model.last_token_features.detach().clone()
    del reference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    online = OnlineVisionEncoder(checkpoint, device, config)
    online_features = online.extract_one(image, row["image_grid_thw"], row["sample_id"])
    delta = (reference_features.float() - online_features.float()).abs()
    scorer = load_full_checkpoint_scorer(checkpoint, online.device)
    with torch.inference_mode():
        reference_scores_same_scorer = scorer(reference_features.float().unsqueeze(0)).squeeze(0)
        scores = scorer(online_features.float().unsqueeze(0)).squeeze(0)
    score_delta = (reference_scores_same_scorer - scores).abs()
    integrated_bf16_score_delta = np.abs(scores.cpu().numpy() - np.asarray(inference["raw_scores"]))
    budgets: dict[str, Any] = {}
    for budget in (0.1, 0.2, 0.3):
        selected = select_gaze_indices(scores, online_features.float(), row["image_grid_thw"], budget)
        expected = np.asarray(inference["budgets"][str(int(budget * 100))]["selected"])
        actual = selected.detach().cpu().numpy()
        budgets[str(int(budget * 100))] = {
            "identical": bool(np.array_equal(actual, expected)),
            "selected_count": int(actual.size),
        }
    result = {
        "sample_id": row["sample_id"],
        "reference_shape": list(reference_features.shape),
        "online_shape": list(online_features.shape),
        "reference_dtype": str(reference_features.dtype),
        "online_dtype": str(online_features.dtype),
        "feature_max_abs_error": float(delta.max()),
        "feature_mean_abs_error": float(delta.mean()),
        "same_scorer_score_max_abs_error": float(score_delta.max()),
        "same_scorer_score_mean_abs_error": float(score_delta.mean()),
        "integrated_bf16_vs_float32_score_max_abs_error": float(integrated_bf16_score_delta.max()),
        "integrated_bf16_vs_float32_score_mean_abs_error": float(integrated_bf16_score_delta.mean()),
        "token_order_exact": bool(torch.equal(reference_features, online_features)),
        "budgets": budgets,
        "passed": bool(torch.equal(reference_features, online_features))
        and float(score_delta.max()) == 0.0
        and all(v["identical"] for v in budgets.values()),
    }
    atomic_write_json(output, result)
    if not result["passed"]:
        raise RuntimeError(f"online feature consistency failed: {result}")
    return result


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Online frozen-Qwen training for Mask-GAZE VIS")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("prepare", "check-data", "train", "dry-run", "benchmark", "consistency"), default="train")
    parser.add_argument("--scorer-init")
    parser.add_argument("--init-mode", choices=("original_random", "explicit"), default="original_random")
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main_train(argv: list[str] | None = None) -> int:
    args = build_train_parser().parse_args(argv)
    config = load_config(Path(args.config))
    if args.mode == "prepare":
        print(json.dumps(prepare_valid_manifest(config), ensure_ascii=False, indent=2))
    elif args.mode == "check-data":
        print(json.dumps(check_data(config, args.max_samples or 20), ensure_ascii=False, indent=2))
    elif args.mode == "consistency":
        output_dir = Path(args.output_dir or config["paths"]["dry_run_output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        print(json.dumps(online_consistency_check(config, args.device, output_dir / "online_consistency.json"), indent=2))
    elif args.mode in {"train", "dry-run", "benchmark"}:
        if args.mode == "dry-run":
            args.max_samples = min(args.max_samples or 20, 20)
            args.max_steps = min(args.max_steps or 2, 2)
        elif args.mode == "benchmark":
            args.max_samples = min(args.max_samples or 200, 200)
            args.max_steps = None
            args.epochs = 1
        result = train(config, args)
        if args.mode == "benchmark":
            seconds = float(result["wall_seconds"])
            sample_count = int(args.max_samples)
            result["measured_samples"] = sample_count
            result["estimated_full_epoch_seconds"] = seconds * 19668 / sample_count
            benchmark_dir = Path(args.output_dir or config["paths"]["dry_run_output_dir"])
            atomic_write_json(benchmark_dir / "benchmark_200.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0

