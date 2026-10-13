from __future__ import annotations

import argparse
import html
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from src.masks import decode_rle
from src.token_labels.mask_to_tokens import token_grid_shape
from src.utils import load_config, read_jsonl


COLORS = {
    -1: np.array([1.0, 0.75, 0.0]),
    0: np.array([0.1, 0.35, 0.95]),
    1: np.array([0.1, 0.8, 0.25]),
    2: np.array([0.95, 0.1, 0.15]),
}


def _overlay(image: np.ndarray, mask: np.ndarray, color: tuple[float, float, float], alpha: float = 0.45) -> np.ndarray:
    result = image.astype(np.float32) / 255.0
    color_array = np.asarray(color, dtype=np.float32)
    result[mask] = (1.0 - alpha) * result[mask] + alpha * color_array
    return np.clip(result, 0.0, 1.0)


def render_sample(sample: dict[str, Any], output_path: Path, config: dict[str, Any]) -> None:
    image = np.asarray(Image.open(sample["image_path"]).convert("RGB"))
    subject = decode_rle(sample["subject_mask_rle"])
    candidates = sample["candidates"]
    details = [candidate for candidate in candidates if candidate.get("is_detail")]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=int(config["visualization"]["dpi"]))

    axes[0, 0].imshow(image)
    for bbox in sample["bboxes"]:
        x1, y1, x2, y2 = bbox
        axes[0, 0].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="yellow", linewidth=2))
    axes[0, 0].set_title(f"Original + bbox | {sample['source']} | {sample['sample_id']}")

    axes[0, 1].imshow(_overlay(image, subject, (0.1, 0.8, 0.25)))
    fallback = sample.get("subject_mask_fallback", False)
    axes[0, 1].set_title(f"Subject mask | fallback={fallback}")

    candidate_view = image.astype(np.float32) / 255.0
    palette = plt.get_cmap("tab20")
    candidate_lines = []
    for index, candidate in enumerate(candidates):
        mask = decode_rle(candidate["mask_rle"])
        candidate_view = _overlay((candidate_view * 255).astype(np.uint8), mask, palette(index % 20)[:3], alpha=0.20)
        ys, xs = np.nonzero(mask)
        if len(xs):
            axes[0, 2].text(xs.mean(), ys.mean(), str(index), color="white", fontsize=9, ha="center", va="center", bbox={"facecolor": "black", "alpha": 0.65, "pad": 1})
        semantic = candidate.get("semantic_score")
        drop = candidate.get("counterfactual_score")
        utility = candidate.get("utility")
        candidate_lines.append(
            f"{index}: sem={semantic:.3f} drop={drop:.3f} u={utility:.3f}"
            if drop is not None and utility is not None
            else f"{index}: sem={semantic:.3f} drop=NA u=NA"
        )
    axes[0, 2].imshow(candidate_view)
    axes[0, 2].set_title(f"Filtered candidates: {len(candidates)}")

    detail_view = image.copy()
    for index, candidate in enumerate(details):
        detail_view = (_overlay(detail_view, decode_rle(candidate["mask_rle"]), (0.95, 0.1 + 0.2 * index, 0.15), 0.48) * 255).astype(np.uint8)
    axes[1, 0].imshow(detail_view)
    axes[1, 0].set_title(f"Selected detail masks: {len(details)}")

    temporal, grid_h, grid_w = token_grid_shape(sample["image_grid_thw"], int(config["qwen"]["spatial_merge_size"]))
    labels = np.asarray(sample["token_labels"], dtype=np.int8).reshape(temporal, grid_h, grid_w)[0]
    label_rgb = np.zeros((grid_h, grid_w, 3), dtype=np.float32)
    for label, color in COLORS.items():
        label_rgb[labels == label] = color
    label_rgb = np.asarray(Image.fromarray((label_rgb * 255).astype(np.uint8)).resize((image.shape[1], image.shape[0]), Image.Resampling.NEAREST)) / 255.0
    token_view = 0.45 * (image.astype(np.float32) / 255.0) + 0.55 * label_rgb
    axes[1, 1].imshow(np.clip(token_view, 0, 1))
    axes[1, 1].set_title(f"Token grid {grid_h}x{grid_w}: red=detail green=foreground blue=background orange=ignore")

    cf_mask = np.zeros(image.shape[:2], dtype=bool)
    for candidate in candidates:
        if candidate.get("selected_for_counterfactual"):
            cf_mask |= decode_rle(candidate["mask_rle"])
    axes[1, 2].imshow(_overlay(image, cf_mask, (0.65, 0.15, 0.85), 0.50))
    method = sample.get("metadata", {}).get("counterfactual_method", "unknown")
    axes[1, 2].set_title(f"Counterfactual token regions | {method}")

    for axis in axes.flat:
        axis.axis("off")
    score_text = "\n".join(candidate_lines[:12]) if candidate_lines else "No valid local candidates"
    fig.text(0.01, 0.01, score_text, family="monospace", fontsize=8, va="bottom")
    fig.suptitle(f"Q: {sample['question']}\nA: {sample['answer']}", fontsize=12)
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _write_overview(samples: list[dict[str, Any]], output_dir: Path, count: int, seed: int) -> None:
    rng = random.Random(seed)
    chosen = rng.sample(samples, min(count, len(samples)))
    cards = []
    for sample in chosen:
        filename = f"{sample['sample_id']}.png"
        cards.append(
            f'<article><a href="{html.escape(filename)}"><img src="{html.escape(filename)}" loading="lazy"></a>'
            f'<p><b>{html.escape(sample["source"])}</b> — {html.escape(sample["question"])}</p></article>'
        )
    document = """<!doctype html><html><head><meta charset=\"utf-8\"><title>Pilot review</title>
<style>body{font-family:sans-serif;margin:20px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}article{border:1px solid #ddd;padding:8px}img{width:100%;height:auto}p{font-size:13px}</style></head><body>
<h1>Hierarchical VIS pilot — deterministic 50-sample review</h1><main>""" + "".join(cards) + "</main></body></html>"
    (output_dir / "overview.html").write_text(document, encoding="utf-8")


def run_visualization_stage(config_path: str | Path, *, overwrite: bool = False) -> None:
    config = load_config(config_path)
    output_dir = Path(config["paths"]["output_dir"])
    visualization_dir = output_dir / "visualizations"
    samples = list(read_jsonl(output_dir / "dataset.jsonl"))
    for index, sample in enumerate(samples, start=1):
        destination = visualization_dir / f"{sample['sample_id']}.png"
        if destination.exists() and not overwrite:
            continue
        render_sample(sample, destination, config)
        print(f"[visualize {index:03d}] {sample['sample_id']}", flush=True)
    _write_overview(samples, visualization_dir, int(config["visualization"]["overview_count"]), int(config["project"]["seed"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render all pilot quality-control visualizations")
    parser.add_argument("--config", default="configs/pilot_200.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_visualization_stage(args.config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

