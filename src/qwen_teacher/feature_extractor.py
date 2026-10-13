from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.token_labels.mask_to_tokens import mask_to_token_coverage


@dataclass
class VisualFeatures:
    embeddings: torch.Tensor
    image_grid_thw: list[int]
    visual_token_count: int
    resized_height: int
    resized_width: int


class FrozenQwenTeacher:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        qwen_config = config["qwen"]
        checkpoint = config["paths"]["qwen_checkpoint"]
        self.device = torch.device(qwen_config["device"])
        dtype_name = str(qwen_config["dtype"])
        self.dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_name]
        # Pin the slow processor explicitly. It is the implementation used by
        # the audited Transformers 4.50.0 checkpoint and avoids a future
        # default change silently altering mask/token alignment.
        self.processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True, use_fast=False)
        image_processor = self.processor.image_processor
        image_processor.min_pixels = int(qwen_config["min_pixels"])
        image_processor.max_pixels = int(qwen_config["max_pixels"])
        image_processor.size = {
            "shortest_edge": int(qwen_config["min_pixels"]),
            "longest_edge": int(qwen_config["max_pixels"]),
        }
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint,
            torch_dtype=self.dtype,
            attn_implementation=qwen_config["attention_implementation"],
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _to_device(self, inputs) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}

    def user_prompt(self, question: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @torch.inference_mode()
    def extract_visual(self, image: Image.Image, question: str) -> tuple[VisualFeatures, dict[str, torch.Tensor], str]:
        prompt = self.user_prompt(question)
        inputs = self._to_device(self.processor(text=[prompt], images=[image], padding=True, return_tensors="pt"))
        pixel_values = inputs["pixel_values"].to(dtype=self.model.visual.dtype)
        embeddings = self.model.visual(pixel_values, grid_thw=inputs["image_grid_thw"])
        grid = [int(value) for value in inputs["image_grid_thw"][0].tolist()]
        merge = int(self.config["qwen"]["spatial_merge_size"])
        expected = grid[0] * grid[1] * grid[2] // (merge * merge)
        if embeddings.shape[0] != expected:
            raise AssertionError(f"Qwen visual feature count {embeddings.shape[0]} != grid-derived {expected}")
        patch = int(self.config["qwen"]["patch_size"])
        features = VisualFeatures(
            embeddings=embeddings,
            image_grid_thw=grid,
            visual_token_count=expected,
            resized_height=grid[1] * patch,
            resized_width=grid[2] * patch,
        )
        return features, inputs, prompt

    @torch.inference_mode()
    def text_embedding(self, text: str) -> torch.Tensor:
        token_ids = self.processor.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        if token_ids.numel() == 0:
            return torch.zeros(self.model.config.hidden_size, dtype=self.dtype, device=self.device)
        embeddings = self.model.get_input_embeddings()(token_ids)
        return embeddings.float().mean(dim=1).squeeze(0)

    @torch.inference_mode()
    def semantic_similarities(
        self,
        visual: VisualFeatures,
        masks: list[np.ndarray],
        question: str,
        answer: str,
    ) -> list[dict[str, float]]:
        question_embedding = self.text_embedding(question)
        answer_embedding = self.text_embedding(answer)
        combined_embedding = self.text_embedding(f"{question}\n{answer}")
        text = torch.stack((question_embedding, answer_embedding, combined_embedding), dim=0)
        text = F.normalize(text, dim=-1)
        results: list[dict[str, float]] = []
        for mask in masks:
            coverage = mask_to_token_coverage(
                mask,
                visual.image_grid_thw,
                patch_size=int(self.config["qwen"]["patch_size"]),
                spatial_merge_size=int(self.config["qwen"]["spatial_merge_size"]),
            )
            weights = torch.as_tensor(coverage, dtype=torch.float32, device=self.device)
            denominator = weights.sum().clamp_min(1e-6)
            pooled = (visual.embeddings.float() * weights[:, None]).sum(dim=0) / denominator
            pooled = F.normalize(pooled, dim=0)
            similarities = torch.mv(text, pooled).detach().cpu().tolist()
            results.append(
                {
                    "question_similarity": float(similarities[0]),
                    "answer_similarity": float(similarities[1]),
                    "question_answer_similarity": float(similarities[2]),
                    "token_coverage_sum": float(weights.sum().item()),
                }
            )
        return results

    def peak_memory_mib(self) -> float:
        if self.device.type != "cuda":
            return 0.0
        return float(torch.cuda.max_memory_allocated(self.device) / (1024 * 1024))

