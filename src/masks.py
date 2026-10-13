from __future__ import annotations

from typing import Any

import numpy as np
from pycocotools import mask as mask_utils


def encode_rle(mask: np.ndarray) -> dict[str, Any]:
    binary = np.asfortranarray(mask.astype(np.uint8))
    encoded = mask_utils.encode(binary)
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(v) for v in encoded["size"]], "counts": counts}


def decode_rle(rle: dict[str, Any]) -> np.ndarray:
    payload = {"size": rle["size"], "counts": rle["counts"]}
    if isinstance(payload["counts"], str):
        payload["counts"] = payload["counts"].encode("ascii")
    return mask_utils.decode(payload).astype(bool)


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.logical_and(first, second).sum(dtype=np.int64)
    union = np.logical_or(first, second).sum(dtype=np.int64)
    return float(intersection / union) if union else 0.0


def bbox_mask(shape: tuple[int, int], bbox: list[float]) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = bbox
    left = max(0, min(width, int(np.floor(min(x1, x2)))))
    right = max(0, min(width, int(np.ceil(max(x1, x2)))))
    top = max(0, min(height, int(np.floor(min(y1, y2)))))
    bottom = max(0, min(height, int(np.ceil(max(y1, y2)))))
    result = np.zeros((height, width), dtype=bool)
    if right > left and bottom > top:
        result[top:bottom, left:right] = True
    return result

