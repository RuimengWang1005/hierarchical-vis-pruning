from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.masks import bbox_mask, encode_rle
from src.utils import append_jsonl, load_config, read_jsonl, seed_everything


def load_sam_predictor(config: dict[str, Any]):
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as exc:
        raise RuntimeError("segment-anything is required; run scripts/bootstrap_env.sh") from exc
    sam_config = config["sam"]
    model = sam_model_registry[sam_config["model_type"]](checkpoint=config["paths"]["sam_checkpoint"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device=sam_config["device"])
    return SamPredictor(model)


def _subject_candidate_metrics(mask: np.ndarray, bbox: list[float], pred_iou: float, stability: float) -> dict[str, Any]:
    reference = bbox_mask(mask.shape, bbox)
    area = int(mask.sum())
    intersection = int(np.logical_and(mask, reference).sum())
    bbox_area = int(reference.sum())
    inside_ratio = float(intersection / area) if area else 0.0
    bbox_coverage = float(intersection / bbox_area) if bbox_area else 0.0
    if area:
        ys, xs = np.nonzero(mask)
        center = [float(xs.mean()), float(ys.mean())]
        center_in_bbox = bool(bbox[0] <= center[0] <= bbox[2] and bbox[1] <= center[1] <= bbox[3])
    else:
        center = [0.0, 0.0]
        center_in_bbox = False
    image_area_ratio = float(area / mask.size)
    score = 0.45 * pred_iou + 0.25 * stability + 0.20 * inside_ratio + 0.10 * min(1.0, bbox_coverage)
    if not center_in_bbox:
        score -= 0.25
    if image_area_ratio > 0.95:
        score -= 0.50
    return {
        "pred_iou": float(pred_iou),
        "stability": float(stability),
        "area": area,
        "inside_ratio": inside_ratio,
        "bbox_coverage": bbox_coverage,
        "image_area_ratio": image_area_ratio,
        "center": center,
        "center_in_bbox": center_in_bbox,
        "selection_score": float(score),
    }


@torch.inference_mode()
def segment_subject(image: np.ndarray, bboxes: list[list[float]], predictor, sam_config: dict[str, Any]) -> dict[str, Any]:
    from segment_anything.utils.amg import calculate_stability_score

    predictor.set_image(image)
    device = predictor.device
    boxes = torch.as_tensor(bboxes, dtype=torch.float32, device=device)
    transformed = predictor.transform.apply_boxes_torch(boxes, image.shape[:2])
    masks, iou_predictions, low_res_logits = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed,
        multimask_output=True,
        return_logits=False,
    )
    stability = calculate_stability_score(
        low_res_logits,
        predictor.model.mask_threshold,
        float(sam_config["stability_offset"]),
    )

    instances: list[np.ndarray] = []
    instance_metadata: list[dict[str, Any]] = []
    any_fallback = False
    for box_index, bbox in enumerate(bboxes):
        scored: list[tuple[dict[str, Any], np.ndarray]] = []
        for candidate_index in range(masks.shape[1]):
            candidate = masks[box_index, candidate_index].detach().cpu().numpy().astype(bool)
            metrics = _subject_candidate_metrics(
                candidate,
                bbox,
                float(iou_predictions[box_index, candidate_index].item()),
                float(stability[box_index, candidate_index].item()),
            )
            metrics["candidate_index"] = candidate_index
            scored.append((metrics, candidate))
        scored.sort(key=lambda item: item[0]["selection_score"], reverse=True)
        chosen_metrics, chosen_mask = scored[0]
        valid = (
            chosen_metrics["center_in_bbox"]
            and chosen_metrics["bbox_coverage"] >= float(sam_config["subject_min_bbox_coverage"])
            and chosen_metrics["image_area_ratio"] <= float(sam_config["subject_max_image_area"])
            and chosen_metrics["image_area_ratio"] >= float(sam_config["subject_min_image_area"])
        )
        fallback = not valid
        if fallback:
            chosen_mask = bbox_mask(image.shape[:2], bbox)
            any_fallback = True
        instances.append(chosen_mask)
        instance_metadata.append(
            {
                "bbox": [float(value) for value in bbox],
                "mask_rle": encode_rle(chosen_mask),
                "fallback": fallback,
                "selected": chosen_metrics,
                "alternatives": [item[0] for item in scored],
            }
        )

    union = np.logical_or.reduce(instances) if instances else np.zeros(image.shape[:2], dtype=bool)
    return {
        "subject_mask": union,
        "instance_masks": instances,
        "subject_mask_rle": encode_rle(union),
        "subject_mask_fallback": any_fallback,
        "subject_instances": instance_metadata,
    }


def run_subject_stage(config_path: str | Path, *, overwrite: bool = False) -> None:
    config = load_config(config_path)
    seed_everything(int(config["project"]["seed"]), bool(config["runtime"]["deterministic_algorithms"]))
    output_dir = Path(config["paths"]["output_dir"])
    stage_path = output_dir / "subject_masks.jsonl"
    if overwrite:
        stage_path.unlink(missing_ok=True)
    completed = {row["sample_id"] for row in read_jsonl(stage_path)} if stage_path.exists() else set()
    predictor = load_sam_predictor(config)
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    for index, sample in enumerate(read_jsonl(output_dir / "manifest.jsonl"), start=1):
        if sample["sample_id"] in completed:
            continue
        started = time.perf_counter()
        image = np.asarray(Image.open(sample["image_path"]).convert("RGB"))
        result = segment_subject(image, sample["bboxes"], predictor, config["sam"])
        instance_stack = np.stack(result.pop("instance_masks"), axis=0).astype(np.uint8)
        subject_array = result.pop("subject_mask").astype(np.uint8)
        np.savez_compressed(mask_dir / f"{sample['sample_id']}.subject.npz", subject=subject_array, instances=instance_stack)
        record = {
            **sample,
            **result,
            "timing": {"sam_subject_seconds": time.perf_counter() - started},
        }
        append_jsonl(stage_path, record)
        print(f"[subject {index:03d}] {sample['sample_id']} fallback={record['subject_mask_fallback']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SAM subject masks")
    parser.add_argument("--config", default="configs/pilot_200.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_subject_stage(args.config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

