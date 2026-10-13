from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.qwen_teacher.feature_extractor import FrozenQwenTeacher, VisualFeatures
from src.token_labels.mask_to_tokens import mask_to_token_coverage


def _longest_common_prefix(first: torch.Tensor, second: torch.Tensor) -> int:
    limit = min(first.numel(), second.numel())
    if limit == 0:
        return 0
    unequal = torch.nonzero(first[:limit] != second[:limit], as_tuple=False)
    return int(unequal[0].item()) if unequal.numel() else limit


def _per_row_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    losses = F.cross_entropy(
        shift_logits.transpose(1, 2),
        shift_labels,
        ignore_index=-100,
        reduction="none",
    )
    valid = shift_labels.ne(-100)
    return (losses * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


@torch.inference_mode()
def counterfactual_scores(
    teacher: FrozenQwenTeacher,
    image: Image.Image,
    question: str,
    answer: str,
    visual: VisualFeatures,
    prompt_inputs: dict[str, torch.Tensor],
    prompt_text: str,
    subject_mask: np.ndarray,
    candidate_masks: list[np.ndarray],
) -> tuple[float, list[float], dict[str, Any]]:
    tokenizer = teacher.processor.tokenizer
    full_text = prompt_text + answer
    full_inputs = teacher._to_device(
        teacher.processor(text=[full_text], images=[image], padding=True, return_tensors="pt")
    )
    input_ids = full_inputs["input_ids"]
    attention_mask = full_inputs["attention_mask"]
    prefix_ids = prompt_inputs["input_ids"][0]
    prefix_length = _longest_common_prefix(prefix_ids, input_ids[0])
    labels = torch.full_like(input_ids, -100)
    labels[:, prefix_length:] = input_ids[:, prefix_length:]
    if not bool(teacher.config["qwen"].get("answer_include_im_end", False)):
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        labels[labels == im_end_id] = -100
    answer_token_count = int(labels.ne(-100).sum().item())
    if answer_token_count <= 0:
        raise RuntimeError("Could not identify any answer tokens for counterfactual NLL")

    base_embeds = teacher.model.get_input_embeddings()(input_ids)
    image_positions = torch.nonzero(input_ids[0] == teacher.model.config.image_token_id, as_tuple=False).flatten()
    if len(image_positions) != visual.visual_token_count:
        raise AssertionError(f"Image placeholders {len(image_positions)} != visual features {visual.visual_token_count}")
    base_embeds[:, image_positions, :] = visual.embeddings.to(base_embeds.dtype).unsqueeze(0)

    subject_coverage = mask_to_token_coverage(
        subject_mask,
        visual.image_grid_thw,
        patch_size=int(teacher.config["qwen"]["patch_size"]),
        spatial_merge_size=int(teacher.config["qwen"]["spatial_merge_size"]),
    )
    background = subject_coverage <= float(teacher.config["qwen"]["background_token_max_subject_coverage"])
    if background.any():
        neutral = visual.embeddings[torch.as_tensor(background, device=teacher.device)].mean(dim=0)
        neutral_source = "same_image_background_mean"
    else:
        neutral = visual.embeddings.mean(dim=0)
        neutral_source = "same_image_global_mean_fallback"

    full_outputs = teacher.model(
        input_ids=input_ids,
        inputs_embeds=base_embeds,
        attention_mask=attention_mask,
        image_grid_thw=full_inputs["image_grid_thw"],
        use_cache=False,
        return_dict=True,
    )
    full_nll = float(_per_row_nll(full_outputs.logits, labels)[0].item())
    del full_outputs

    coverages = [
        mask_to_token_coverage(
            mask,
            visual.image_grid_thw,
            patch_size=int(teacher.config["qwen"]["patch_size"]),
            spatial_merge_size=int(teacher.config["qwen"]["spatial_merge_size"]),
        )
        for mask in candidate_masks
    ]
    batch_size = int(teacher.config["qwen"]["counterfactual_batch_size"])
    minimum = float(teacher.config["qwen"]["counterfactual_mask_threshold"])
    removed_nlls: list[float] = []
    for start in range(0, len(coverages), batch_size):
        chunk = coverages[start : start + batch_size]
        count = len(chunk)
        embeds = base_embeds.repeat(count, 1, 1)
        for row, coverage in enumerate(chunk):
            weights = torch.as_tensor(coverage, dtype=embeds.dtype, device=teacher.device)
            weights = torch.where(weights >= minimum, weights, torch.zeros_like(weights)).unsqueeze(-1)
            original = embeds[row, image_positions, :]
            embeds[row, image_positions, :] = original * (1.0 - weights) + neutral.to(embeds.dtype) * weights
        repeated_ids = input_ids.repeat(count, 1)
        repeated_attention = attention_mask.repeat(count, 1)
        repeated_labels = labels.repeat(count, 1)
        repeated_grid = full_inputs["image_grid_thw"].repeat(count, 1)
        outputs = teacher.model(
            input_ids=repeated_ids,
            inputs_embeds=embeds,
            attention_mask=repeated_attention,
            image_grid_thw=repeated_grid,
            use_cache=False,
            return_dict=True,
        )
        removed_nlls.extend(float(value) for value in _per_row_nll(outputs.logits, repeated_labels).detach().cpu().tolist())
        del outputs, embeds
    drops = [float(value - full_nll) for value in removed_nlls]
    metadata = {
        "counterfactual_method": "visual_token_feature_replacement",
        "neutral_feature": neutral_source,
        "answer_token_count": answer_token_count,
        "full_answer_nll": full_nll,
        "removed_answer_nll": removed_nlls,
    }
    return full_nll, drops, metadata

