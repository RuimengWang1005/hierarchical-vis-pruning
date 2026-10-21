#!/usr/bin/env python3
# cd /home/bbhan/projects/trytrytrywb/qwen-vl-finetune
# python tools/sample_viscot.py --target_samples 160000 --seed 42

"""
Sample entries from mkz Visual-CoT jsonl shards with a fixed ratio.

Usage:
    python tools/sample_viscot.py --target_samples 160000 --seed 42

By default this script reads jsonl files under:
    /home/bbhan/projects/trytrytrywb/viscot_dataset

and samples with ratio:
    gqa:openimages:v7w:docvqa:textvqa:infographicsvqa:dude:vsr:sroie = 38:15:10:12:10:5:5:3:2

Output format is Alpaca-style jsonl lines (instruction/input/output/images) and
keeps bbox fields (bboxs/width/height) for downstream token-mask supervision.

If some shards do not have enough rows for the requested quota, this script can
either:
  - repeat samples (with replacement) to hit the exact `target_samples`, or
  - keep the old behavior and just output fewer rows.
"""

import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_DATA_DIR = "/home/bbhan/gaze/gaicolgemii211/viscot_dataset"
DATASET_SOURCES = {
    "gqa": "gqa_cot_train.jsonl",
    "v7w": "visual7w_cot_train.jsonl",
    #"docvqa": "docvqa_cot_train.jsonl",
    #"textvqa": "textvqa_cot_train.jsonl",
    #"infographicsvqa": "infographicsvqa_cot_train.jsonl",
    "dude": "dude_cot_train.jsonl",
    "vsr": "vsr_cot_train.jsonl",
    #"sroie": "sroie_cot_train.jsonl",
    "flickr30k": "flickr30k_cot_train.jsonl",
    
    
}

DEFAULT_RATIOS = {
    "gqa": 30,
    #"docvqa": 15,
    #"textvqa": 15,
    "vsr": 10,
    "dude": 20,
    #"infographicsvqa": 5,
    #"sroie": 5,
    "v7w": 10,
    "flickr30k": 10
}

def _compute_target_counts(target_samples: int, ratios: Dict[str, int]) -> Dict[str, int]:
    total = sum(ratios.values())
    raw: Dict[str, float] = {k: target_samples * (v / total) for k, v in ratios.items()}
    base: Dict[str, int] = {k: int(raw[k]) for k in ratios}
    remainder = target_samples - sum(base.values())
    if remainder > 0:
        frac = sorted(((raw[k] - base[k], k) for k in ratios), reverse=True)
        for _, k in frac[:remainder]:
            base[k] += 1
    return base


def _reservoir_sample_jsonl(path: Path, k: int, rng: random.Random) -> Tuple[List[Dict], int]:
    if k <= 0:
        return [], 0
    reservoir: List[Dict] = []
    num_seen = 0
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            num_seen += 1
            obj = json.loads(line)
            if len(reservoir) < k:
                reservoir.append(obj)
                continue
            j = rng.randint(1, num_seen)
            if j <= k:
                reservoir[j - 1] = obj
    return reservoir, num_seen


def _build_image_path(dataset: str, image: str) -> str:
    image = image.strip()
    if image.startswith("cot/"):
        return image
    if "/" in image:
        # If already contains a subdir, just ensure it has cot/ prefix.
        return image if image.startswith("cot/") else f"cot/{image.lstrip('/')}"
    return f"cot/{dataset}/{image}"


def _to_alpaca_record(
    raw: Dict,
    short_answer_prompt: bool = True,
) -> Dict:
    dataset = raw.get("dataset") or raw.get("sub_dataset") or "unknown"
    split = raw.get("split") or "train"
    question = (raw.get("question") or "").strip()
    answer = raw.get("answer")
    if answer is None:
        answer = raw.get("output")
    answer = "" if answer is None else str(answer).strip()

    image = raw.get("image")
    if image is None and "images" in raw and isinstance(raw["images"], list) and raw["images"]:
        image = raw["images"][0]
    image = "" if image is None else str(image)
    image_path = _build_image_path(dataset, image)

    instruction = f"<image>\n{question}"
    if short_answer_prompt:
        instruction += "\nAnswer the question using a single word or phrase."

    out: Dict = {
        "instruction": instruction,
        "input": "",
        "output": answer,
        "images": [image_path],
        "dataset": dataset,
        "split": split,
    }

    # Keep bbox fields for downstream token-mask supervision.
    for k in ("bboxs", "width", "height"):
        if k in raw:
            out[k] = raw[k]

    return out


def _default_output_path(data_dir: str, target_samples: int, ratios: Dict[str, int], seed: int) -> str:
    ordered_keys = [k for k in ratios.keys() if k in DATASET_SOURCES]
    ratio_tag = "-".join(str(ratios[k]) for k in ordered_keys) if ordered_keys else "custom"
    sample_tag = f"{target_samples // 1000}k" if target_samples % 1000 == 0 else str(target_samples)
    return str(Path(data_dir) / f"viscot_mix_{sample_tag}_{ratio_tag}_seedww{seed}.jsonl")


def sample_viscot_dataset(
    data_dir: str,
    output_path: str,
    target_samples: int = 160000,
    seed: int = 42,
    short_answer_prompt: bool = True,
    ratios: Dict[str, int] = None,
    fill_shortfall: str = "repeat",
):
    """
    Sample entries from Visual-CoT jsonl shards with a fixed dataset ratio.
    """
    ratios = DEFAULT_RATIOS if ratios is None else ratios
    target_counts = _compute_target_counts(target_samples, ratios)

    data_root = Path(data_dir)
    rng = random.Random(seed)

    sampled_records: List[Dict] = []
    stats: Dict[str, Dict[str, int]] = {}

    print("Sampling plan:", target_counts)
    for dataset_name, filename in DATASET_SOURCES.items():
        k = target_counts.get(dataset_name, 0)
        path = data_root / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing source file: {path}")
        print(f"Sampling {k} from {path} ...")
        sampled_unique, num_available = _reservoir_sample_jsonl(path, k, rng)
        repeated = 0
        sampled_total = sampled_unique
        if fill_shortfall == "repeat" and len(sampled_unique) < k:
            if not sampled_unique:
                raise ValueError(f"Dataset {dataset_name} is empty but requested k={k}: {path}")
            repeated = k - len(sampled_unique)
            sampled_total = sampled_unique + rng.choices(sampled_unique, k=repeated)

        stats[dataset_name] = {
            "requested": k,
            "available": num_available,
            "sampled_unique": len(sampled_unique),
            "sampled_total": len(sampled_total),
            "repeated": repeated,
        }
        sampled_records.extend(_to_alpaca_record(s, short_answer_prompt=short_answer_prompt) for s in sampled_total)

    rng.shuffle(sampled_records)

    if fill_shortfall == "repeat":
        assert len(sampled_records) == target_samples, f"{len(sampled_records)} != {target_samples}"

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for rec in sampled_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats_path = out_path.with_suffix(".stats.json")
    with stats_path.open("w") as f:
        json.dump(
            {
                "target_samples": target_samples,
                "seed": seed,
                "ratios": ratios,
                "counts": target_counts,
                "actual": stats,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved: {out_path}")
    print(f"Saved stats: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Sample Visual-CoT shards with fixed ratios")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Directory containing *.jsonl shards")
    parser.add_argument("--output", type=str, default=None, help="Output jsonl path (default: auto-named under data_dir)")
    parser.add_argument("--target_samples", type=int, default=160000, help="Total number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--no_short_answer_prompt",
        action="store_true",
        help="Do not append short-answer prompt suffix",
    )
    parser.add_argument(
        "--fill_shortfall",
        type=str,
        default="repeat",
        choices=("repeat", "drop"),
        help="When a dataset shard is smaller than its quota: repeat samples to hit target, or drop (output fewer rows).",
    )

    args = parser.parse_args()
    output_path = args.output or _default_output_path(args.data_dir, args.target_samples, DEFAULT_RATIOS, args.seed)

    sample_viscot_dataset(
        data_dir=args.data_dir,
        output_path=output_path,
        target_samples=args.target_samples,
        seed=args.seed,
        short_answer_prompt=(not args.no_short_answer_prompt),
        fill_shortfall=args.fill_shortfall,
    )


if __name__ == "__main__":
    main()