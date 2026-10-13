from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from PIL import Image

from src.masks import decode_rle, mask_iou
from src.qwen_teacher.counterfactual_score import counterfactual_scores
from src.qwen_teacher.feature_extractor import FrozenQwenTeacher
from src.qwen_teacher.semantic_score import combine_semantic_score, semantic_prescreen
from src.utils import append_jsonl, config_hash, load_config, read_jsonl, robust_unit_scale, seed_everything


def _score_and_select(candidates: list[dict[str, Any]], teacher_config: dict[str, Any]) -> list[str]:
    scored_indices = [index for index, candidate in enumerate(candidates) if candidate.get("counterfactual_score") is not None]
    if not scored_indices:
        return []
    semantic = robust_unit_scale([float(candidates[index]["semantic_score"]) for index in scored_indices])
    counterfactual = robust_unit_scale([float(candidates[index]["counterfactual_score"]) for index in scored_indices])
    sam_quality = robust_unit_scale([float(candidates[index]["sam_quality"]) for index in scored_indices])
    area_penalty = robust_unit_scale([float(candidates[index]["area_ratio"]) for index in scored_indices])
    for position, index in enumerate(scored_indices):
        candidate = candidates[index]
        candidate["normalized_scores"] = {
            "semantic": semantic[position],
            "counterfactual": counterfactual[position],
            "sam_quality": sam_quality[position],
            "area_penalty": area_penalty[position],
        }
        candidate["utility"] = float(
            float(teacher_config["semantic_weight"]) * semantic[position]
            + float(teacher_config["counterfactual_weight"]) * counterfactual[position]
            + float(teacher_config["sam_quality_weight"]) * sam_quality[position]
            - float(teacher_config["area_penalty_weight"]) * area_penalty[position]
        )

    ranking = sorted(scored_indices, key=lambda index: (-float(candidates[index]["utility"]), candidates[index]["candidate_id"]))
    selected: list[int] = []
    decoded: dict[int, np.ndarray] = {}
    for index in ranking:
        if float(candidates[index]["utility"]) < float(teacher_config["detail_min_utility"]):
            continue
        candidate_mask = decoded.setdefault(index, decode_rle(candidates[index]["mask_rle"]))
        accept = True
        for selected_index in selected:
            selected_mask = decoded.setdefault(selected_index, decode_rle(candidates[selected_index]["mask_rle"]))
            overlap = mask_iou(candidate_mask, selected_mask)
            area_ratio = min(candidate_mask.sum(), selected_mask.sum()) / max(candidate_mask.sum(), selected_mask.sum())
            nested_allowed = area_ratio <= float(teacher_config["nested_area_ratio_threshold"])
            if overlap > float(teacher_config["detail_diversity_iou"]) and not nested_allowed:
                accept = False
                break
        if accept:
            selected.append(index)
            if len(selected) >= int(teacher_config["max_detail_masks"]):
                break
    for index, candidate in enumerate(candidates):
        candidate["is_detail"] = index in selected
    return [candidates[index]["candidate_id"] for index in selected]


def score_sample(sample: dict[str, Any], teacher: FrozenQwenTeacher, config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    image = Image.open(sample["image_path"]).convert("RGB")
    visual, prompt_inputs, prompt_text = teacher.extract_visual(image, sample["question"])
    candidates = [dict(candidate) for candidate in sample["candidates"]]
    candidate_masks = [decode_rle(candidate["mask_rle"]) for candidate in candidates]
    similarities = teacher.semantic_similarities(
        visual, candidate_masks, sample["question"], sample["answer"]
    )
    for candidate, scores in zip(candidates, similarities):
        candidate.update(scores)
        candidate["semantic_score"] = combine_semantic_score(scores, config["teacher_score"])
        candidate["selected_for_counterfactual"] = False
        candidate["counterfactual_score"] = None
        candidate["utility"] = None
        candidate["is_detail"] = False

    prescreened = semantic_prescreen(candidates, int(config["teacher_score"]["semantic_top_k"]))
    for index in prescreened:
        candidates[index]["selected_for_counterfactual"] = True
    subject_mask = decode_rle(sample["subject_mask_rle"])
    full_nll, drops, counterfactual_metadata = counterfactual_scores(
        teacher,
        image,
        sample["question"],
        sample["answer"],
        visual,
        prompt_inputs,
        prompt_text,
        subject_mask,
        [candidate_masks[index] for index in prescreened],
    )
    for index, drop in zip(prescreened, drops):
        candidates[index]["counterfactual_score"] = float(drop)
    selected_ids = _score_and_select(candidates, config["teacher_score"])

    qwen_config = config["qwen"]
    result = dict(sample)
    result.update(
        {
            "image_grid_thw": visual.image_grid_thw,
            "visual_token_count": visual.visual_token_count,
            "qwen_resized_size": [visual.resized_width, visual.resized_height],
            "full_answer_nll": full_nll,
            "selected_detail_candidate_ids": selected_ids,
            "candidates": candidates,
            "timing": {**sample.get("timing", {}), "qwen_seconds": time.perf_counter() - started},
            "metadata": {
                "sam_checkpoint": config["paths"]["sam_checkpoint"],
                "qwen_checkpoint": config["paths"]["qwen_checkpoint"],
                "transformers_version": transformers.__version__,
                "qwen_dtype": str(qwen_config["dtype"]),
                "qwen_attention_implementation": qwen_config["attention_implementation"],
                "qwen_min_pixels": int(qwen_config["min_pixels"]),
                "qwen_max_pixels": int(qwen_config["max_pixels"]),
                "counterfactual_method": counterfactual_metadata["counterfactual_method"],
                "counterfactual_neutral_feature": counterfactual_metadata["neutral_feature"],
                "answer_token_count": counterfactual_metadata["answer_token_count"],
                "removed_answer_nll": counterfactual_metadata["removed_answer_nll"],
                "config_hash": config_hash(config),
                "gpu_peak_memory_mib": teacher.peak_memory_mib(),
            },
        }
    )
    return result


def run_teacher_stage(config_path: str | Path, *, overwrite: bool = False) -> None:
    config = load_config(config_path)
    seed_everything(int(config["project"]["seed"]), bool(config["runtime"]["deterministic_algorithms"]))
    output_dir = Path(config["paths"]["output_dir"])
    stage_path = output_dir / "teacher_scores.jsonl"
    failure_path = output_dir / "failures.jsonl"
    if overwrite:
        stage_path.unlink(missing_ok=True)
        failure_path.unlink(missing_ok=True)
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.touch(exist_ok=True)
    completed = {row["sample_id"] for row in read_jsonl(stage_path)} if stage_path.exists() else set()
    teacher = FrozenQwenTeacher(config)
    if teacher.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(teacher.device)
    for index, sample in enumerate(read_jsonl(output_dir / "filtered_candidates.jsonl"), start=1):
        if sample["sample_id"] in completed:
            continue
        try:
            result = score_sample(sample, teacher, config)
        except Exception as exc:
            append_jsonl(
                failure_path,
                {
                    "sample_id": sample["sample_id"],
                    "stage": "qwen_teacher",
                    "reason": type(exc).__name__,
                    "error": str(exc),
                },
            )
            print(f"[teacher {index:03d}] FAILED {sample['sample_id']}: {exc}", flush=True)
            if config["runtime"]["fail_fast"]:
                raise
            continue
        append_jsonl(stage_path, result)
        print(
            f"[teacher {index:03d}] {sample['sample_id']} candidates={len(result['candidates'])} details={len(result['selected_detail_candidate_ids'])}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score candidates with the frozen Qwen teacher")
    parser.add_argument("--config", default="configs/pilot_200.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_teacher_stage(args.config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

