import numpy as np

from src.masks import encode_rle
from src.sam_masks.mask_filter import filter_candidates


def _candidate(identifier, mask, quality):
    return {
        "candidate_id": identifier,
        "mask_rle": encode_rle(mask),
        "sam_pred_iou": 0.95,
        "stability": 0.96,
        "area": int(mask.sum()),
        "inside_ratio": 1.0,
        "area_ratio": float(mask.mean()),
        "prompt_point": [0.0, 0.0],
        "touches_crop_boundary": False,
        "connected_components": 1,
        "sam_quality": quality,
    }


def test_filter_deduplicates_similar_masks_but_preserves_nested_scale():
    outer = np.zeros((20, 20), dtype=bool)
    outer[2:18, 2:18] = True
    duplicate = outer.copy()
    inner = np.zeros_like(outer)
    inner[7:13, 7:13] = True
    config = {
        "min_inside_ratio": 0.85,
        "min_area_ratio": 0.005,
        "max_area_ratio": 0.8,
        "min_pred_iou": 0.85,
        "min_stability": 0.9,
        "max_connected_components": 8,
        "duplicate_iou": 0.85,
        "duplicate_area_similarity": 0.8,
        "max_filtered_candidates": 12,
    }
    retained, reasons = filter_candidates(
        [_candidate("outer", outer, 0.99), _candidate("duplicate", duplicate, 0.90), _candidate("inner", inner, 0.95)],
        config,
    )
    assert {item["candidate_id"] for item in retained} == {"outer", "inner"}
    assert reasons["duplicate"] == 1

