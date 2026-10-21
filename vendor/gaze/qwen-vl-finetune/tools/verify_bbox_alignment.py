#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify bbox-to-resize alignment and export visualizations.

This script samples bbox-annotated items from a VISCOT-style jsonl and:
1) Computes the resize size using qwen-vl-utils smart_resize (min/max pixels).
2) Optionally gets the processor resize size via Qwen2VLImageProcessor.
3) Draws bboxes on original and resized images for visual inspection.

Usage (example):
  python tools/verify_bbox_alignment.py \
    --dataset-name viscot_mkz_bbox_4k \
    --num-samples 8 \
    --min-pixels 3136 --max-pixels 401408 \
    --use-processor \
    --model-path /data/ModelZoo/LVLM/Qwen-Qwen2.5-VL-3B-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw


def _add_repo_paths() -> Path:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]  # .../gaicolgemii
    qwen_utils = repo_root / "qwen-vl-utils" / "src"
    if qwen_utils.exists():
        sys.path.insert(0, str(qwen_utils))
    if repo_root.exists():
        sys.path.insert(0, str(repo_root))
    return repo_root


_REPO_ROOT = _add_repo_paths()


def _load_smart_resize():
    try:
        from qwen_vl_utils.vision_process import smart_resize, MIN_PIXELS, MAX_PIXELS  # type: ignore
        return smart_resize, MIN_PIXELS, MAX_PIXELS
    except Exception:
        # Local fallback (copied behavior from qwen_vl_utils.smart_resize)
        def round_by_factor(number: int, factor: int) -> int:
            return round(number / factor) * factor

        def ceil_by_factor(number: int, factor: int) -> int:
            return int((number + factor - 1) // factor) * factor

        def floor_by_factor(number: int, factor: int) -> int:
            return (number // factor) * factor

        def smart_resize(height: int, width: int, factor: int = 28, min_pixels: int = 4 * 28 * 28, max_pixels: int = 16384 * 28 * 28):
            if max(height, width) / min(height, width) > 200:
                raise ValueError("absolute aspect ratio too large")
            h_bar = max(factor, round_by_factor(height, factor))
            w_bar = max(factor, round_by_factor(width, factor))
            if h_bar * w_bar > max_pixels:
                beta = ((height * width) / max_pixels) ** 0.5
                h_bar = max(factor, floor_by_factor(int(height / beta), factor))
                w_bar = max(factor, floor_by_factor(int(width / beta), factor))
            elif h_bar * w_bar < min_pixels:
                beta = (min_pixels / (height * width)) ** 0.5
                h_bar = ceil_by_factor(int(height * beta), factor)
                w_bar = ceil_by_factor(int(width * beta), factor)
            return int(h_bar), int(w_bar)

        return smart_resize, 4 * 28 * 28, 16384 * 28 * 28


def _load_processor(model_path: str):
    try:
        from transformers import AutoProcessor  # type: ignore
    except Exception as exc:
        print(f"[WARN] transformers not available: {exc}")
        return None
    try:
        processor = AutoProcessor.from_pretrained(model_path).image_processor
        return processor
    except Exception as exc:
        print(f"[WARN] failed to load processor from {model_path}: {exc}")
        return None


def _load_preprocessor_config(model_path: str) -> Dict[str, Any]:
    cfg_path = Path(model_path) / "preprocessor_config.json"
    if not cfg_path.exists():
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _get_default_paths(dataset_name: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        from qwenvl.data import data_dict  # type: ignore
    except Exception:
        return None, None
    if dataset_name in data_dict:
        return data_dict[dataset_name]["annotation_path"], data_dict[dataset_name]["data_path"]
    return None, None


def _iter_jsonl(path: str, max_lines: int) -> Iterable[Dict[str, Any]]:
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if max_lines > 0 and count >= max_lines:
                break
            count += 1
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _resolve_image_path(sample: Dict[str, Any], image_root: Optional[str]) -> Optional[str]:
    image_rel = None
    if "image" in sample:
        image_rel = sample["image"]
    elif "images" in sample:
        images = sample.get("images")
        if isinstance(images, list) and images:
            image_rel = images[0]
        elif isinstance(images, str):
            image_rel = images
    if image_rel is None:
        return None
    image_rel = str(image_rel)
    if image_rel.startswith("cot/"):
        image_rel = image_rel.replace("cot/", "", 1)
    if os.path.isabs(image_rel):
        return image_rel
    if image_root:
        return os.path.join(image_root, image_rel)
    return image_rel


def _filter_bboxs(bboxs: Any) -> List[List[float]]:
    out: List[List[float]] = []
    if not isinstance(bboxs, list):
        return out
    for b in bboxs:
        if not isinstance(b, (list, tuple)) or len(b) != 4:
            continue
        try:
            x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        except Exception:
            continue
        out.append([x1, y1, x2, y2])
    return out


def _draw_bboxes(img: Image.Image, bboxs: List[List[float]], color: str = "red") -> Image.Image:
    draw = ImageDraw.Draw(img)
    for b in bboxs:
        x1, y1, x2, y2 = b
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    return img


def _scale_bboxs(bboxs: List[List[float]], sx: float, sy: float) -> List[List[float]]:
    scaled = []
    for x1, y1, x2, y2 in bboxs:
        scaled.append([x1 * sx, y1 * sy, x2 * sx, y2 * sy])
    return scaled


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify bbox alignment with image resize.")
    parser.add_argument("--dataset-name", default="viscot_mkz_bbox_4k")
    parser.add_argument("--jsonl", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-lines", type=int, default=200000)
    parser.add_argument("--out-dir", default="debug_bbox_alignment")
    parser.add_argument("--use-processor", action="store_true")
    parser.add_argument("--model-path", default="/data/ModelZoo/LVLM/Qwen-Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    args = parser.parse_args()

    jsonl_path = args.jsonl
    image_root = args.image_root
    if jsonl_path is None or image_root is None:
        d_jsonl, d_root = _get_default_paths(args.dataset_name)
        jsonl_path = jsonl_path or d_jsonl
        image_root = image_root or d_root

    if not jsonl_path or not os.path.exists(jsonl_path):
        print(f"[ERR] jsonl not found: {jsonl_path}")
        return 1

    smart_resize, default_min, default_max = _load_smart_resize()

    processor = _load_processor(args.model_path) if args.use_processor else None
    pre_cfg = _load_preprocessor_config(args.model_path)

    patch_size = args.patch_size or pre_cfg.get("patch_size") or getattr(processor, "patch_size", None) or 14
    min_pixels = args.min_pixels if args.min_pixels is not None else pre_cfg.get("min_pixels", default_min)
    max_pixels = args.max_pixels if args.max_pixels is not None else pre_cfg.get("max_pixels", default_max)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.jsonl"

    rng = random.Random(args.seed)

    selected: List[Dict[str, Any]] = []
    skipped_no_bbox = 0
    skipped_no_img = 0
    for sample in _iter_jsonl(jsonl_path, args.max_lines):
        bboxs = _filter_bboxs(sample.get("bboxs"))
        if not bboxs:
            skipped_no_bbox += 1
            continue
        image_path = _resolve_image_path(sample, image_root)
        if not image_path or not os.path.exists(image_path):
            skipped_no_img += 1
            continue
        selected.append(sample)
        if len(selected) >= args.num_samples:
            break

    if not selected:
        print("[ERR] no valid samples found (bboxs + existing images).")
        return 1

    with open(summary_path, "w", encoding="utf-8") as sf:
        for idx, sample in enumerate(selected):
            image_path = _resolve_image_path(sample, image_root)
            if not image_path:
                continue
            image = Image.open(image_path).convert("RGB")
            orig_w, orig_h = image.size

            bboxs = _filter_bboxs(sample.get("bboxs"))
            if not bboxs:
                continue

            smart_h, smart_w = smart_resize(
                orig_h, orig_w, factor=28, min_pixels=min_pixels, max_pixels=max_pixels
            )

            # Draw original
            orig_vis = _draw_bboxes(image.copy(), bboxs, color="red")
            base = _safe_name(os.path.splitext(os.path.basename(image_path))[0])
            orig_out = out_dir / f"{idx:03d}_{base}_orig.png"
            orig_vis.save(orig_out)

            # Draw smart-resized
            smart_img = image.resize((smart_w, smart_h))
            sx, sy = smart_w / max(orig_w, 1), smart_h / max(orig_h, 1)
            smart_bboxs = _scale_bboxs(bboxs, sx, sy)
            smart_vis = _draw_bboxes(smart_img, smart_bboxs, color="yellow")
            smart_out = out_dir / f"{idx:03d}_{base}_smart.png"
            smart_vis.save(smart_out)

            # Processor resize size (if available)
            proc_h = proc_w = None
            grid_thw = None
            if processor is not None:
                try:
                    visual = processor.preprocess(image, return_tensors="pt")
                    grid = visual.get("image_grid_thw")
                    if grid is not None:
                        if hasattr(grid, "tolist"):
                            grid = grid.tolist()
                        if isinstance(grid, list) and len(grid) > 0:
                            grid_thw = grid[0]
                        elif isinstance(grid, (tuple, list)) and len(grid) == 3:
                            grid_thw = list(grid)
                    if grid_thw and len(grid_thw) == 3:
                        proc_h = int(grid_thw[1]) * int(patch_size)
                        proc_w = int(grid_thw[2]) * int(patch_size)
                except Exception as exc:
                    print(f"[WARN] processor.preprocess failed for {image_path}: {exc}")

            if proc_h and proc_w:
                proc_img = image.resize((proc_w, proc_h))
                sx2, sy2 = proc_w / max(orig_w, 1), proc_h / max(orig_h, 1)
                proc_bboxs = _scale_bboxs(bboxs, sx2, sy2)
                proc_vis = _draw_bboxes(proc_img, proc_bboxs, color="cyan")
                proc_out = out_dir / f"{idx:03d}_{base}_proc.png"
                proc_vis.save(proc_out)

            summary = {
                "image": image_path,
                "orig_size": [orig_w, orig_h],
                "smart_resize": [smart_w, smart_h],
                "processor_resize": [proc_w, proc_h] if proc_h and proc_w else None,
                "grid_thw": grid_thw,
                "patch_size": patch_size,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "bbox_count": len(bboxs),
                "match_smart_proc": (proc_w == smart_w and proc_h == smart_h) if proc_h and proc_w else None,
            }
            sf.write(json.dumps(summary, ensure_ascii=True) + "\n")

    print(f"[OK] wrote {len(selected)} samples to {out_dir}")
    print(f"[INFO] skipped no-bbox: {skipped_no_bbox}, no-image: {skipped_no_img}")
    print(f"[INFO] summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
