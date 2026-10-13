from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from src.masks import decode_rle, encode_rle
from src.sam_masks.subject_mask import load_sam_predictor
from src.utils import append_jsonl, load_config, read_jsonl, seed_everything


def _expanded_crop(bbox: list[float], width: int, height: int, padding: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    left = max(0, int(np.floor(x1 - pad_x)))
    top = max(0, int(np.floor(y1 - pad_y)))
    right = min(width, int(np.ceil(x2 + pad_x)))
    bottom = min(height, int(np.ceil(y2 + pad_y)))
    return left, top, max(left + 1, right), max(top + 1, bottom)


def _allocate_points(instance_masks: list[np.ndarray], total: int) -> list[int]:
    active = [index for index, mask in enumerate(instance_masks) if mask.any()]
    if not active or total <= 0:
        return [0] * len(instance_masks)
    if len(active) > total:
        # The per-image raw-candidate cap is strict. When there are more bbox
        # instances than prompts, cover the largest instances deterministically.
        active = sorted(active, key=lambda index: (-int(instance_masks[index].sum()), index))[:total]
    allocation = [0] * len(instance_masks)
    for index in active:
        allocation[index] = 1
    remaining = max(0, total - len(active))
    areas = np.asarray([float(instance_masks[index].sum()) for index in active], dtype=np.float64)
    raw = remaining * areas / areas.sum()
    base = np.floor(raw).astype(int)
    for index, value in zip(active, base.tolist()):
        allocation[index] += value
    residual = remaining - int(base.sum())
    order = np.argsort(-(raw - base), kind="mergesort")
    for offset in order[:residual]:
        allocation[active[int(offset)]] += 1
    return allocation


def _farthest_points(mask: np.ndarray, count: int, seed: int) -> np.ndarray:
    coordinates_yx = np.argwhere(mask)
    if count <= 0 or len(coordinates_yx) == 0:
        return np.empty((0, 2), dtype=np.float32)
    rng = np.random.default_rng(seed)
    if len(coordinates_yx) > 5000:
        coordinates_yx = coordinates_yx[rng.choice(len(coordinates_yx), size=5000, replace=False)]
    coords = coordinates_yx[:, ::-1].astype(np.float32)
    centroid = coords.mean(axis=0)
    first = int(np.argmin(((coords - centroid) ** 2).sum(axis=1)))
    selected = [first]
    minimum_distance = ((coords - coords[first]) ** 2).sum(axis=1)
    while len(selected) < min(count, len(coords)):
        next_index = int(np.argmax(minimum_distance))
        selected.append(next_index)
        distance = ((coords - coords[next_index]) ** 2).sum(axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
    return coords[selected]


def _component_count(mask: np.ndarray, minimum_fraction: float) -> int:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return 0
    threshold = max(1, int(mask.sum() * minimum_fraction))
    return int(sum(int(stats[index, cv2.CC_STAT_AREA]) >= threshold for index in range(1, count)))


@torch.inference_mode()
def generate_candidates(
    image: np.ndarray,
    subject_mask: np.ndarray,
    instance_masks: list[np.ndarray],
    bboxes: list[list[float]],
    predictor,
    sam_config: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    from segment_anything.utils.amg import calculate_stability_score

    height, width = image.shape[:2]
    allocations = _allocate_points(instance_masks, int(sam_config["point_prompt_count"]))
    candidates: list[dict[str, Any]] = []
    candidate_index = 0
    for instance_index, (bbox, instance_mask, point_count) in enumerate(zip(bboxes, instance_masks, allocations)):
        if point_count <= 0:
            continue
        left, top, right, bottom = _expanded_crop(
            bbox, width, height, float(sam_config["subject_crop_padding"])
        )
        crop_image = image[top:bottom, left:right]
        crop_subject = instance_mask[top:bottom, left:right]
        local_seed = seed ^ int(hashlib.sha1(f"{instance_index}:{bbox}".encode()).hexdigest()[:8], 16)
        points = _farthest_points(crop_subject, point_count, local_seed)
        if not len(points):
            continue
        predictor.set_image(crop_image)
        point_tensor = torch.as_tensor(points[:, None, :], dtype=torch.float32, device=predictor.device)
        point_tensor = predictor.transform.apply_coords_torch(point_tensor, crop_image.shape[:2])
        labels = torch.ones((len(points), 1), dtype=torch.int64, device=predictor.device)
        masks, predicted_iou, low_res_logits = predictor.predict_torch(
            point_coords=point_tensor,
            point_labels=labels,
            boxes=None,
            multimask_output=True,
            return_logits=False,
        )
        stability = calculate_stability_score(
            low_res_logits,
            predictor.model.mask_threshold,
            float(sam_config["stability_offset"]),
        )
        for point_index in range(masks.shape[0]):
            for multimask_index in range(masks.shape[1]):
                local_mask = masks[point_index, multimask_index].detach().cpu().numpy().astype(bool)
                full_mask = np.zeros((height, width), dtype=bool)
                full_mask[top:bottom, left:right] = local_mask
                area = int(full_mask.sum())
                intersection = int(np.logical_and(full_mask, subject_mask).sum())
                inside_ratio = float(intersection / area) if area else 0.0
                subject_area = int(subject_mask.sum())
                area_ratio = float(area / subject_area) if subject_area else 0.0
                border = bool(
                    local_mask[0:2, :].any()
                    or local_mask[-2:, :].any()
                    or local_mask[:, 0:2].any()
                    or local_mask[:, -2:].any()
                )
                components = _component_count(full_mask, float(sam_config["min_component_fraction"]))
                sam_quality = (
                    0.55 * float(predicted_iou[point_index, multimask_index].item())
                    + 0.35 * float(stability[point_index, multimask_index].item())
                    + 0.10 * inside_ratio
                    - (float(sam_config["border_touch_penalty"]) if border else 0.0)
                )
                candidates.append(
                    {
                        "candidate_id": f"c{candidate_index:03d}",
                        "mask_rle": encode_rle(full_mask),
                        "sam_pred_iou": float(predicted_iou[point_index, multimask_index].item()),
                        "stability": float(stability[point_index, multimask_index].item()),
                        "area": area,
                        "inside_ratio": inside_ratio,
                        "area_ratio": area_ratio,
                        "prompt_point": [float(points[point_index, 0] + left), float(points[point_index, 1] + top)],
                        "prompt_instance": instance_index,
                        "crop_box": [left, top, right, bottom],
                        "touches_crop_boundary": border,
                        "connected_components": components,
                        "sam_quality": float(sam_quality),
                    }
                )
                candidate_index += 1
    return candidates


def run_candidate_stage(config_path: str | Path, *, overwrite: bool = False) -> None:
    config = load_config(config_path)
    seed = int(config["project"]["seed"])
    seed_everything(seed, bool(config["runtime"]["deterministic_algorithms"]))
    output_dir = Path(config["paths"]["output_dir"])
    stage_path = output_dir / "raw_candidates.jsonl"
    if overwrite:
        stage_path.unlink(missing_ok=True)
    completed = {row["sample_id"] for row in read_jsonl(stage_path)} if stage_path.exists() else set()
    predictor = load_sam_predictor(config)
    for index, sample in enumerate(read_jsonl(output_dir / "subject_masks.jsonl"), start=1):
        if sample["sample_id"] in completed:
            continue
        started = time.perf_counter()
        image = np.asarray(Image.open(sample["image_path"]).convert("RGB"))
        subject = decode_rle(sample["subject_mask_rle"])
        instances = [decode_rle(item["mask_rle"]) for item in sample["subject_instances"]]
        candidates = generate_candidates(
            image,
            subject,
            instances,
            sample["bboxes"],
            predictor,
            config["sam"],
            seed ^ int(hashlib.sha1(sample["sample_id"].encode()).hexdigest()[:8], 16),
        )
        record = {
            **sample,
            "raw_candidate_count": len(candidates),
            "raw_candidates": candidates,
            "timing": {**sample.get("timing", {}), "sam_candidates_seconds": time.perf_counter() - started},
        }
        append_jsonl(stage_path, record)
        print(f"[candidates {index:03d}] {sample['sample_id']} raw={len(candidates)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multi-granularity local SAM candidates")
    parser.add_argument("--config", default="configs/pilot_200.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_candidate_stage(args.config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

