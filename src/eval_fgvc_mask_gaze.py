from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file
from tqdm import tqdm


EVALUATION_ROOT = Path("/home/ruimeng/gaze/qwen-evaluation1")
sys.path.insert(0, str(EVALUATION_ROOT))
import eval_fgvc_selector as fgvc  # noqa: E402


DATASET_ROOTS = {
    "bird-200": "/data/DataSet/vlm_dataset/CUB_200_2011",
    "fgvc_aircraft": "/data/DataSet/cls_dataset/fgvc_aircraft",
    "stanford_cars": "/data/DataSet/cls_dataset/standford_cars",
    "flowers102": "/data/DataSet/cls_dataset/Flowers102/rawdata/flowers-102",
    "oxford_pet": "/data/DataSet/cls_dataset/oxford_pets",
}


def _base_args(device: str, output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        dataset="bird-200",
        image_folder=DATASET_ROOTS["bird-200"],
        model_path="/home/ruimeng/gaze/output_ckpt666/VisionSelector-Qwen2.5-VL-7B",
        options_root="/home/ruimeng/FinePruner_TIP2026/data",
        results_output_folder=str(output_dir),
        prompt_query=None,
        seed=42,
        limit=None,
        max_examples_per_class=999999,
        choice_enumeration="ABCD",
        method="selector",
        budgets=0.1,
        spatial_ratio=0.1,
        contextual_ratio=0.1,
        cosine_penalty_weight=1.0,
        cosine_penalty_threshold=0.7,
        distance_aware_redundancy=False,
        distance_sigma=0.1,
        distance_floor=0.0,
        distance_apply_to="focal",
        focal_selection_strategy="original",
        candidate_multiplier=3.0,
        candidate_max_ratio=0.6,
        cluster_cosine_threshold=0.85,
        cluster_max_weight=0.5,
        cluster_top_mean_weight=0.5,
        cluster_top_n=3,
        count_penalty_weight=0.5,
        local_penalty_weight=0.5,
        local_cosine_threshold=0.7,
        temperature=0.0,
        top_p=None,
        num_beams=1,
        max_new_tokens=32,
        attn_implementation="flash_attention_2",
        torch_dtype="bfloat16",
        device=device,
        device_map=device,
        min_pixels=200704,
        max_pixels=1605632,
    )


def _load_external_scorer(model, path: Path) -> str:
    state = load_file(str(path), device="cpu")
    target = model.visual.importance_scorer
    dtype = next(target.parameters()).dtype
    device = next(target.parameters()).device
    target.load_state_dict({key: value.to(device=device, dtype=dtype) for key, value in state.items()}, strict=True)
    return str(path.resolve())


def evaluate_condition(model, processor, tokenizer, args, dataset: str, condition: str, scorer_path: Path, budget: float, output_dir: Path):
    args.dataset = dataset
    args.image_folder = DATASET_ROOTS[dataset]
    args.budgets = float(budget)
    model.visual.budgets = float(budget)
    model.visual.config.budgets = float(budget)
    loaded_scorer = _load_external_scorer(model, scorer_path)
    fgvc.set_seed(args.seed)
    examples, options_path = fgvc.load_examples(args)
    prompt_query = fgvc.DATASET_CONFIGS[dataset]["prompt_query"]
    results = []
    correct = 0
    generation_failures = 0
    started = time.time()
    description = f"{dataset}:{condition}:b{budget:.1f}"
    for example in tqdm(examples, desc=description):
        image_path = fgvc.resolve_image_path(dataset, args.image_folder, example["image"])
        prompt, gold_label, label_to_option = fgvc.prepare_prompt_mc(
            example["options"], prompt_query, choice_enumeration=args.choice_enumeration
        )
        answer = fgvc.generate_answer(model, processor, tokenizer, image_path, prompt, args)
        parsed = fgvc.parse_prediction(answer, choice_enumeration=args.choice_enumeration)
        generation_failures += int(parsed is None)
        correct += int(parsed == gold_label)
        results.append(
            {
                "image": example["image"], "prediction_raw": answer, "prediction_parsed": parsed,
                "gold_label": gold_label, "gold_text": label_to_option[gold_label], "options": label_to_option,
            }
        )
    elapsed = time.time() - started
    payload = {
        "dataset": dataset,
        "condition": condition,
        "scorer_checkpoint": loaded_scorer,
        "budget": budget,
        "metrics": {
            "num_examples": len(results),
            "accuracy": correct / len(results),
            "generation_failures": generation_failures / len(results),
            "elapsed_seconds": elapsed,
        },
        "selection_configuration": {
            "spatial_ratio": 0.1, "cosine_penalty_weight": 1.0,
            "cosine_penalty_threshold": 0.7, "focal_selection_strategy": "original",
            "min_pixels": args.min_pixels, "max_pixels": args.max_pixels,
        },
        "options_path": options_path,
        "predictions": results,
    }
    destination_dir = output_dir / dataset / condition / f"budget_{int(budget * 100)}"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_results.json"
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"dataset": dataset, "condition": condition, "budget": budget, "metrics": payload["metrics"]}), flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Full FGVC A/B/C evaluation with one model load")
    parser.add_argument("--datasets", nargs="+", required=True, choices=sorted(DATASET_ROOTS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mask_gaze_1k/preexperiment/fgvc_evaluation"))
    parser.add_argument("--initial-scorer", type=Path, default=Path("outputs/mask_gaze_1k/preexperiment/initial_scorer.safetensors"))
    parser.add_argument("--bbox-scorer", type=Path, default=Path("outputs/mask_gaze_1k/preexperiment/training/bbox/epoch_3.safetensors"))
    parser.add_argument("--mask-scorer", type=Path, default=Path("outputs/mask_gaze_1k/preexperiment/training/mask/epoch_3.safetensors"))
    args = parser.parse_args()
    model_args = _base_args(args.device, args.output_dir)
    model, processor, tokenizer = fgvc.load_model(model_args)
    fgvc.replace_layers(model_args, model)
    conditions = (
        ("A_original", args.initial_scorer),
        ("B_bbox_epoch_3", args.bbox_scorer),
        ("C_mask_epoch_3", args.mask_scorer),
    )
    for dataset in args.datasets:
        for condition, scorer_path in conditions:
            for budget in (0.1, 0.2, 0.3):
                # Exact A/10% and A/20% full-set runs already exist from the
                # same checkpoint and selector configuration. Keep them as
                # immutable references and only compute the missing A/30% run.
                if condition == "A_original" and budget in (0.1, 0.2):
                    print(f"[reuse legacy full-set result] {dataset} {condition} {budget}", flush=True)
                    continue
                destination_dir = args.output_dir / dataset / condition / f"budget_{int(budget * 100)}"
                if any(destination_dir.glob("*_results.json")):
                    print(f"[skip existing] {dataset} {condition} {budget}", flush=True)
                    continue
                evaluate_condition(model, processor, tokenizer, model_args, dataset, condition, scorer_path, budget, args.output_dir)


if __name__ == "__main__":
    main()
