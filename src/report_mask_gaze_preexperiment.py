from __future__ import annotations

import argparse
import csv
import glob
import html
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.masks import decode_rle
from src.utils import atomic_write_json, read_jsonl


def paired_bootstrap(values: np.ndarray, seed: int = 42, draws: int = 10000) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 500):
        count = min(500, draws - start)
        indices = rng.integers(0, values.size, size=(count, values.size))
        means[start : start + count] = values[indices].mean(axis=1)
    return {
        "mean_difference": float(values.mean()),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
        "probability_positive": float((means > 0).mean()),
    }


def _load_token_eval(base: Path, name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = json.loads((base / f"{name}_summary.json").read_text(encoding="utf-8"))
    samples = list(read_jsonl(base / f"{name}_per_sample.jsonl"))
    return summary, samples


def _fgvc_results(work_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    legacy_root = Path("/home/ruimeng/gaze/qwen-evaluation1/resultrw/fgvc_eval")
    for dataset in ("bird-200", "fgvc_aircraft", "stanford_cars", "flowers102", "oxford_pet"):
        for budget in (10, 20):
            pattern = legacy_root / dataset / f"0.{budget // 10}" / f"VisionSelector-Qwen2.5-VL-7B_selector_0.{budget // 10}" / "*_results.json"
            matches = sorted(glob.glob(str(pattern)))
            if matches:
                payload = json.loads(Path(matches[-1]).read_text(encoding="utf-8"))
                rows.append({"dataset": dataset, "condition": "A_original", "budget": budget, **payload["metrics"], "path": matches[-1]})
    root = work_dir / "fgvc_evaluation"
    for path in sorted(root.glob("*/*/budget_*/*_results.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "dataset": payload["dataset"], "condition": payload["condition"],
                "budget": int(round(float(payload["budget"]) * 100)), **payload["metrics"], "path": str(path.resolve()),
            }
        )
    return rows


def _downstream_results(work_dir: Path) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted((work_dir / "downstream_evaluation").rglob("*_results.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        config = payload.get("config", {})
        results = payload.get("results", {})
        task = "docvqa_val"
        metric = results.get(task, {}).get("anls,none")
        if metric is not None:
            relative = path.relative_to(work_dir / "downstream_evaluation")
            if len(relative.parts) < 3 or not relative.parts[1].startswith("budget_"):
                continue
            condition = relative.parts[0]
            budget = int(relative.parts[1].replace("budget_", ""))
            row = {
                "condition": condition, "budget": budget, "task": task,
                "anls": float(metric), "path": str(path.resolve()), "duplicate_paths": [],
            }
            key = (condition, budget)
            if key in by_key:
                if not np.isclose(by_key[key]["anls"], row["anls"], rtol=0.0, atol=1e-12):
                    raise ValueError(f"conflicting duplicate downstream results for {key}: {by_key[key]['anls']} vs {row['anls']}")
                by_key[key]["duplicate_paths"].append(row["path"])
            else:
                by_key[key] = row
    return [by_key[key] for key in sorted(by_key)]


def _downstream_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = {(row["condition"], int(row["budget"])): row for row in rows}
    comparisons = []
    for budget in (10, 20, 30):
        a = lookup.get(("A_original", budget))
        b = lookup.get(("B_bbox_epoch_3", budget))
        c = lookup.get(("C_mask_epoch_3", budget))
        comparisons.append(
            {
                "budget": budget,
                "C_minus_B": None if b is None or c is None else float(c["anls"] - b["anls"]),
                "C_minus_A": None if a is None or c is None else float(c["anls"] - a["anls"]),
                "B_minus_A": None if a is None or b is None else float(b["anls"] - a["anls"]),
            }
        )
    return {"expected_rows": 9, "complete": len(rows) == 9, "comparisons": comparisons}


def _fgvc_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = {(row["dataset"], row["condition"], int(row["budget"])): row for row in rows}
    conditions = ("A_original", "B_bbox_epoch_3", "C_mask_epoch_3")
    datasets = ("bird-200", "fgvc_aircraft", "stanford_cars", "flowers102", "oxford_pet")
    macro: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    for budget in (10, 20, 30):
        for condition in conditions:
            values = [lookup[(dataset, condition, budget)]["accuracy"] for dataset in datasets if (dataset, condition, budget) in lookup]
            if values:
                macro.append(
                    {
                        "condition": condition, "budget": budget, "dataset_count": len(values),
                        "macro_accuracy": float(np.mean(values)),
                    }
                )
        for left, right in (("B_bbox_epoch_3", "C_mask_epoch_3"), ("A_original", "C_mask_epoch_3")):
            for dataset in datasets:
                left_row = lookup.get((dataset, left, budget))
                right_row = lookup.get((dataset, right, budget))
                if left_row is None or right_row is None:
                    continue
                left_payload = json.loads(Path(left_row["path"]).read_text(encoding="utf-8"))
                right_payload = json.loads(Path(right_row["path"]).read_text(encoding="utf-8"))
                left_correct = {
                    prediction["image"]: float(prediction.get("prediction_parsed") == prediction.get("gold_label"))
                    for prediction in left_payload["predictions"]
                }
                right_correct = {
                    prediction["image"]: float(prediction.get("prediction_parsed") == prediction.get("gold_label"))
                    for prediction in right_payload["predictions"]
                }
                common = sorted(set(left_correct) & set(right_correct))
                differences = np.asarray([right_correct[key] - left_correct[key] for key in common], dtype=np.float64)
                paired.append(
                    {
                        "dataset": dataset, "budget": budget, "comparison": f"{right}_minus_{left}",
                        "sample_count": len(common), **paired_bootstrap(differences, seed=42 + budget + len(paired)),
                    }
                )
    return {"macro_accuracy": macro, "paired_accuracy_differences": paired}


def _selection_overlay(image: np.ndarray, row: dict[str, Any], selected: list[int]) -> np.ndarray:
    labels = np.asarray(row["mask_token_labels"], dtype=np.int8)
    temporal, patch_h, patch_w = (int(value) for value in row["image_grid_thw"])
    token_h, token_w = patch_h // 2, patch_w // 2
    grid = np.full(labels.size, -9, dtype=np.int8)
    selected_array = np.asarray(selected, dtype=np.int64)
    grid[selected_array] = labels[selected_array]
    rgb = np.zeros((token_h, token_w, 3), dtype=np.uint8)
    reshaped = grid.reshape(temporal, token_h, token_w)[0]
    rgb[reshaped == 1] = (30, 220, 80)
    rgb[reshaped == 0] = (35, 105, 235)
    rgb[reshaped == -1] = (230, 145, 30)
    resized = np.asarray(Image.fromarray(rgb).resize((image.shape[1], image.shape[0]), Image.Resampling.NEAREST))
    active = np.any(resized != 0, axis=2)
    output = (image.astype(np.float32) * 0.62).astype(np.uint8)
    output[active] = (0.30 * image[active] + 0.70 * resized[active]).astype(np.uint8)
    return output


def render_cases(work_dir: Path, accepted: list[dict[str, Any]], per_group: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    accepted_by_id = {row["sample_id"]: row for row in accepted}
    by_group = {group: {row["sample_id"]: row for row in rows} for group, rows in per_group.items()}
    candidates = []
    for sample_id in accepted_by_id:
        delta = (
            by_group["C_mask_epoch_3"][sample_id]["budgets"]["10"]["subject_recall"]
            - by_group["B_bbox_epoch_3"][sample_id]["budgets"]["10"]["subject_recall"]
        )
        candidates.append((delta, sample_id))
    chosen_ids = [sample_id for _, sample_id in sorted(candidates, reverse=True)[:10]]
    chosen_ids += [sample_id for _, sample_id in sorted(candidates)[:5] if sample_id not in chosen_ids]
    for key in ("small_target", "thin_region", "multi_subject"):
        if key == "small_target":
            ordered = sorted(accepted, key=lambda row: row["mask_area_ratio"])
        elif key == "thin_region":
            ordered = sorted(accepted, key=lambda row: row["thin_score"], reverse=True)
        else:
            ordered = sorted(accepted, key=lambda row: row["bbox_count"], reverse=True)
        for row in ordered[:5]:
            if row["sample_id"] not in chosen_ids:
                chosen_ids.append(row["sample_id"])
    output_dir = work_dir / "result_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    for index, sample_id in enumerate(chosen_ids, start=1):
        row = accepted_by_id[sample_id]
        image = np.asarray(Image.open(row["image_path"]).convert("RGB"))
        subject = decode_rle(row["subject_mask_rle"])
        fig, axes = plt.subplots(1, 5, figsize=(23, 5.2), dpi=105)
        axes[0].imshow(image)
        for bbox in row["bboxes"]:
            x1, y1, x2, y2 = bbox
            axes[0].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="yellow", linewidth=2))
        axes[0].set_title("Original + BBox")
        mask_view = image.copy().astype(np.float32)
        mask_view[subject] = 0.45 * mask_view[subject] + 0.55 * np.asarray((35, 220, 85))
        axes[1].imshow(mask_view.astype(np.uint8))
        axes[1].set_title("SAM subject mask")
        for axis, group, title in zip(
            axes[2:], ("A_original", "B_bbox_epoch_3", "C_mask_epoch_3"), ("A original", "B BBox", "C Mask")
        ):
            budget = by_group[group][sample_id]["budgets"]["10"]
            axis.imshow(_selection_overlay(image, row, budget["selected_indices"]))
            axis.set_title(f"{title} @10%\nrecall={budget['subject_recall']:.3f}, bg={budget['selected_background_fraction']:.3f}")
        for axis in axes:
            axis.axis("off")
        delta = (
            by_group["C_mask_epoch_3"][sample_id]["budgets"]["10"]["subject_recall"]
            - by_group["B_bbox_epoch_3"][sample_id]["budgets"]["10"]["subject_recall"]
        )
        fig.suptitle(f"{sample_id} | {row['source']} | C-B recall delta={delta:+.3f}\nQ: {row['question']}", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        filename = f"{index:03d}_{sample_id}.png"
        fig.savefig(output_dir / filename, bbox_inches="tight")
        plt.close(fig)
        cards.append(f'<article><a href="{filename}"><img src="{filename}" loading="lazy"></a><p>{html.escape(sample_id)} | Δ={delta:+.3f}</p></article>')
    overview = """<!doctype html><html><head><meta charset="utf-8"><title>Mask-GAZE A/B/C cases</title><style>body{font-family:sans-serif;margin:20px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:14px}article{border:1px solid #ddd;padding:8px}img{width:100%}</style></head><body><h1>Mask-GAZE A/B/C @10%</h1><p>green=selected subject, blue=selected reliable background, orange=selected ignore</p><main>""" + "".join(cards) + "</main></body></html>"
    (output_dir / "overview.html").write_text(overview, encoding="utf-8")
    return {"count": len(chosen_ids), "overview": str((output_dir / "overview.html").resolve())}


def generate_report(work_dir: Path) -> dict[str, Any]:
    token_dir = work_dir / "token_evaluation"
    names = ["A_original"] + [f"{group}_{kind}_epoch_{epoch}" for epoch in range(1, 4) for group, kind in (("B", "bbox"), ("C", "mask"))]
    summaries: dict[str, Any] = {}
    per_group: dict[str, list[dict[str, Any]]] = {}
    for name in names:
        summaries[name], per_group[name] = _load_token_eval(token_dir, name)
    final_table = []
    for name in ("A_original", "B_bbox_epoch_3", "C_mask_epoch_3"):
        summary = summaries[name]
        row: dict[str, Any] = {
            "condition": name,
            "score_gap": summary["foreground_background_score_gap"]["mean"],
            "pairwise_accuracy": summary["pairwise_ranking_accuracy"]["mean"],
        }
        for budget in (10, 20, 30):
            metrics = summary["budgets"][str(budget)]
            row[f"recall_{budget}"] = metrics["subject_token_recall"]["mean"]
            row[f"any_subject_{budget}"] = metrics["samples_retaining_at_least_one_subject_token"]["mean"]
            row[f"background_retention_{budget}"] = metrics["background_token_retention_rate"]["mean"]
            row[f"selected_background_fraction_{budget}"] = metrics["selected_background_fraction"]["mean"]
        final_table.append(row)
    epoch_deltas: dict[str, Any] = {}
    for epoch in range(1, 4):
        b = {row["sample_id"]: row for row in per_group[f"B_bbox_epoch_{epoch}"]}
        c = {row["sample_id"]: row for row in per_group[f"C_mask_epoch_{epoch}"]}
        epoch_result: dict[str, Any] = {}
        for budget in (10, 20, 30):
            recall_diffs = np.asarray([c[sid]["budgets"][str(budget)]["subject_recall"] - b[sid]["budgets"][str(budget)]["subject_recall"] for sid in b])
            any_diffs = np.asarray([float(c[sid]["budgets"][str(budget)]["retains_subject"]) - float(b[sid]["budgets"][str(budget)]["retains_subject"]) for sid in b])
            bg_diffs = np.asarray([
                c[sid]["budgets"][str(budget)]["background_token_retention_rate"] - b[sid]["budgets"][str(budget)]["background_token_retention_rate"]
                for sid in b if b[sid]["budgets"][str(budget)]["background_token_retention_rate"] is not None
            ])
            epoch_result[str(budget)] = {
                "subject_recall_C_minus_B": paired_bootstrap(recall_diffs, seed=42 + epoch + budget),
                "retains_any_subject_C_minus_B": paired_bootstrap(any_diffs, seed=142 + epoch + budget),
                "background_retention_C_minus_B": paired_bootstrap(bg_diffs, seed=242 + epoch + budget),
            }
        epoch_deltas[str(epoch)] = epoch_result
    accepted = list(read_jsonl(work_dir / "acceptance" / "accepted_supervision.jsonl"))
    visualization_summary = render_cases(work_dir, accepted, per_group)
    fgvc_rows = _fgvc_results(work_dir)
    fgvc_analysis = _fgvc_analysis(fgvc_rows)
    downstream_rows = _downstream_results(work_dir)
    downstream_analysis = _downstream_analysis(downstream_rows)
    downstream_duplicate_retries = sum(len(row["duplicate_paths"]) for row in downstream_rows)
    low_budget_recall_stable = all(
        epoch_deltas[str(epoch)][str(budget)]["subject_recall_C_minus_B"]["ci95_low"] > 0
        for epoch in (1, 2, 3) for budget in (10, 20)
    )
    fgvc_macro = {
        (row["condition"], int(row["budget"])): row["macro_accuracy"]
        for row in fgvc_analysis["macro_accuracy"]
    }
    fgvc_macro_deltas = {
        str(budget): float(fgvc_macro[("C_mask_epoch_3", budget)] - fgvc_macro[("B_bbox_epoch_3", budget)])
        for budget in (10, 20, 30)
        if ("C_mask_epoch_3", budget) in fgvc_macro and ("B_bbox_epoch_3", budget) in fgvc_macro
    }
    tolerance = 0.005
    docvqa_no_material_regression = downstream_analysis["complete"] and all(
        row["C_minus_A"] is not None and row["C_minus_A"] >= -tolerance
        for row in downstream_analysis["comparisons"]
    )
    fgvc_no_material_macro_regression = len(fgvc_macro_deltas) == 3 and all(
        value >= -tolerance for value in fgvc_macro_deltas.values()
    )
    evidence_complete = downstream_analysis["complete"] and len(fgvc_rows) == 45
    recommend_expand = bool(
        evidence_complete and low_budget_recall_stable
        and docvqa_no_material_regression and fgvc_no_material_macro_regression
    )
    evidence_gate = {
        "complete": evidence_complete,
        "low_budget_recall_stable": low_budget_recall_stable,
        "material_regression_tolerance_absolute": tolerance,
        "docvqa_no_material_regression_vs_A": docvqa_no_material_regression,
        "fgvc_no_material_macro_regression_vs_B": fgvc_no_material_macro_regression,
        "fgvc_macro_C_minus_B": fgvc_macro_deltas,
        "recommend_expand_to_20k": recommend_expand,
    }
    result = {
        "token_final_table": final_table,
        "paired_C_minus_B_by_epoch": epoch_deltas,
        "fgvc_results": fgvc_rows,
        "fgvc_analysis": fgvc_analysis,
        "downstream_results": downstream_rows,
        "downstream_analysis": downstream_analysis,
        "downstream_duplicate_retries": downstream_duplicate_retries,
        "evidence_gate": evidence_gate,
        "result_visualizations": visualization_summary,
    }
    atomic_write_json(work_dir / "analysis_summary.json", result)
    with (work_dir / "abc_token_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final_table[0]))
        writer.writeheader()
        writer.writerows(final_table)
    acceptance = json.loads((work_dir / "acceptance" / "acceptance_summary.json").read_text(encoding="utf-8"))
    bbox_training = json.loads((work_dir / "training" / "bbox" / "training_config_and_history.json").read_text(encoding="utf-8"))
    mask_training = json.loads((work_dir / "training" / "mask" / "training_config_and_history.json").read_text(encoding="utf-8"))
    lines = [
        "# Mask-GAZE 1K binary-ranking pre-experiment", "",
        "## Data acceptance", "",
        f"- Gate: **{'PASS' if acceptance['training_gate_passed'] else 'STOP'}**",
        f"- Rows: {acceptance['row_counts']}",
        f"- Sources: {acceptance['source_counts']}",
        f"- Mask positive tokens: {acceptance['mask_positive_token_count']}",
        f"- Reliable negative tokens: {acceptance['reliable_negative_token_count']}",
        f"- Ignore tokens: {acceptance['ignore_token_count']}",
        f"- Adaptive mass coverage: {acceptance['adaptive_mask_mass_coverage']}",
        f"- Zero positive / length mismatch / abnormal coverage: {acceptance['zero_mask_positive_samples']} / {acceptance['token_length_mismatch_samples']} / {acceptance['abnormal_coverage_samples']}",
        f"- Samples skipped before paired training for missing reliable negatives: {acceptance['zero_reliable_negative_samples']}",
        "- DocVQA/TextVQA visualization cases are unavailable because the fixed 1K manifest contains only GQA, Visual7W, and Flickr30k; no out-of-manifest examples were mixed into acceptance.",
        "", "## Training configuration", "",
        "| Item | B: original BBox | C: SAM mask |", "|---|---:|---:|",
        f"| Initial scorer hash | `{bbox_training['initial_scorer_sha256']}` | `{mask_training['initial_scorer_sha256']}` |",
        f"| Samples | {bbox_training['training_samples']} | {mask_training['training_samples']} |",
        f"| Epochs / optimizer updates | {bbox_training['epochs']} / {bbox_training['total_optimizer_updates']} | {mask_training['epochs']} / {mask_training['total_optimizer_updates']} |",
        f"| Batch / grad accumulation | {bbox_training['batch_size']} / {bbox_training['gradient_accumulation_steps']} | {mask_training['batch_size']} / {mask_training['gradient_accumulation_steps']} |",
        f"| LR / scheduler / warmup | {bbox_training['learning_rate']} / cosine / {bbox_training['warmup_ratio']} | {mask_training['learning_rate']} / cosine / {mask_training['warmup_ratio']} |",
        f"| Pairwise margin | {bbox_training['margin']} | {mask_training['margin']} |",
        "| Trainable module | VIS scorer only | VIS scorer only |",
        "| Frozen/unloaded | vision encoder frozen; language model unloaded | vision encoder frozen; language model unloaded |",
        "", f"- A uses the unchanged initial GAZE scorer `{bbox_training['initial_scorer']}` and receives zero training updates.",
        f"- B and C jointly exclude the same {bbox_training['excluded_common_invalid_samples']} samples without reliable negatives; neither group silently substitutes samples.",
        "", "### Per-epoch pairwise loss (diagnostic only)", "",
        "| Epoch | B loss | C loss | B/C optimizer update |", "|---:|---:|---:|---:|",
        f"| 1 | {bbox_training['history'][0]['mean_pairwise_loss']:.6f} | {mask_training['history'][0]['mean_pairwise_loss']:.6f} | {bbox_training['history'][0]['global_update']} |",
        f"| 2 | {bbox_training['history'][1]['mean_pairwise_loss']:.6f} | {mask_training['history'][1]['mean_pairwise_loss']:.6f} | {bbox_training['history'][1]['global_update']} |",
        f"| 3 | {bbox_training['history'][2]['mean_pairwise_loss']:.6f} | {mask_training['history'][2]['mean_pairwise_loss']:.6f} | {bbox_training['history'][2]['global_update']} |",
        "", "Training loss is reported only as a diagnostic and is not used for the expansion decision.",
        "", "## Token-level A/B/C (epoch 3)", "",
        "| Condition | Score gap | Pairwise acc. | Recall@10 | Recall@20 | Recall@30 | Any@10 | Any@20 | Any@30 | BG retain@10 | BG retain@20 | BG retain@30 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in final_table:
        lines.append(
            f"| {row['condition']} | {row['score_gap']:.4f} | {row['pairwise_accuracy']:.4f} | "
            f"{row['recall_10']:.4f} | {row['recall_20']:.4f} | {row['recall_30']:.4f} | "
            f"{row['any_subject_10']:.4f} | {row['any_subject_20']:.4f} | {row['any_subject_30']:.4f} | "
            f"{row['background_retention_10']:.4f} | {row['background_retention_20']:.4f} | {row['background_retention_30']:.4f} |"
        )
    lines.extend(["", "### Paired C minus B subject-recall deltas (epoch 3)", "", "| Budget | Mean delta | Bootstrap 95% CI | P(delta > 0) |", "|---:|---:|---:|---:|"])
    for budget in (10, 20, 30):
        delta = epoch_deltas["3"][str(budget)]["subject_recall_C_minus_B"]
        lines.append(
            f"| {budget}% | {delta['mean_difference']:+.4f} | [{delta['ci95_low']:+.4f}, {delta['ci95_high']:+.4f}] | {delta['probability_positive']:.4f} |"
        )
    lines.extend(["", "## FGVC zero-shot accuracy", "", "| Dataset | Condition | Budget | Accuracy | N |", "|---|---|---:|---:|---:|"])
    for row in sorted(fgvc_rows, key=lambda item: (item["dataset"], int(item["budget"]), item["condition"])):
        lines.append(f"| {row['dataset']} | {row['condition']} | {row['budget']}% | {row['accuracy']:.4f} | {row['num_examples']} |")
    lines.extend(["", "### FGVC five-dataset macro accuracy", "", "| Budget | A | B | C | C - B |", "|---:|---:|---:|---:|---:|"])
    for budget in (10, 20, 30):
        a = fgvc_macro.get(("A_original", budget))
        b = fgvc_macro.get(("B_bbox_epoch_3", budget))
        c = fgvc_macro.get(("C_mask_epoch_3", budget))
        if a is not None and b is not None and c is not None:
            lines.append(f"| {budget}% | {a:.4f} | {b:.4f} | {c:.4f} | {c - b:+.4f} |")
    lines.extend(["", "## Original GAZE downstream task", "", "| Task | Condition | Budget | ANLS |", "|---|---|---:|---:|"])
    for row in sorted(downstream_rows, key=lambda item: (item["budget"], item["condition"])):
        lines.append(f"| {row['task']} | {row['condition']} | {row['budget']}% | {row['anls']:.4f} |")
    lines.extend(["", "### DocVQA deltas", "", "| Budget | C - B | C - A | B - A |", "|---:|---:|---:|---:|"])
    for row in downstream_analysis["comparisons"]:
        def show(value: float | None) -> str:
            return "pending" if value is None else f"{value:+.4f}"
        lines.append(f"| {row['budget']}% | {show(row['C_minus_B'])} | {show(row['C_minus_A'])} | {show(row['B_minus_A'])} |")
    pilot_summary_path = work_dir / "pilot200_independent" / "summary.json"
    pilot_summary = json.loads(pilot_summary_path.read_text(encoding="utf-8")) if pilot_summary_path.is_file() else None
    if pilot_summary is None:
        expansion_lines = [
            "", "## Expansion decision (pending independent validation)", "",
            f"- Evidence complete: **{evidence_gate['complete']}** (requires 45 FGVC rows and 9 DocVQA rows).",
            f"- Stable C>B low-budget subject recall across all three epochs: **{low_budget_recall_stable}**.",
            f"- No material DocVQA regression versus A at an absolute {tolerance:.3f} tolerance: **{docvqa_no_material_regression}**.",
            f"- No material five-dataset FGVC macro regression versus B at the same tolerance: **{fgvc_no_material_macro_regression}**.",
            "- Current conclusion: **训练集内结果支持Mask监督有效，等待pilot 200独立验证**。",
            f"- The earlier in-training evidence gate result is retained for audit only: **{recommend_expand}**; it is not used as the expansion decision before the independent check.",
            "- This gate treats changes inside 0.5 percentage point as non-material; it does not claim downstream statistical significance because lmms-eval did not retain paired per-sample ANLS values.",
        ]
    else:
        pilot_metrics = pilot_summary["metrics"]
        pilot_passed = bool(pilot_summary["initial_unseen_generalization_supported"])
        expansion_lines = [
            "", "## Expansion decision (after independent pilot 200 validation)", "",
            "- Temporary pre-validation conclusion retained for audit: **训练集内结果支持Mask监督有效，等待pilot 200独立验证**。",
            f"- Independent effective sample count: **{pilot_summary['overlap']['effective_nonoverlap_sample_count']}**; sample-ID overlap: **{pilot_summary['overlap']['sample_id_overlap_count']}**; image-path overlap: **{pilot_summary['overlap']['image_path_overlap_count']}**.",
            f"- Independent C-B subject Recall: **{pilot_metrics['10']['C_minus_B_subject_recall']:+.6f}** at 10%, **{pilot_metrics['20']['C_minus_B_subject_recall']:+.6f}** at 20%, and **{pilot_metrics['30']['C_minus_B_subject_recall']:+.6f}** at 30%.",
            f"- Requested initial unseen-sample generalization gate passed: **{pilot_passed}**.",
            f"- Recommendation to continue offline mask-data expansion under the requested gate: **{pilot_passed}**.",
            "- The independent gains at 10% and 20% are positive but small; the at-least-one-subject-token rate does not improve, so this is preliminary support rather than a claim of a large effect.",
            "- This gate treats changes inside 0.5 percentage point as non-material; it does not claim downstream statistical significance because lmms-eval did not retain paired per-sample ANLS values.",
        ]
    lines.extend(
        [
            "", "## Implementation notes", "",
            "- Group B uses original GAZE's direct inclusive rectangular projection on the post-merge grid, not a 0.5 coverage threshold and not adaptive BBox mass mapping.",
            "- Group C forms pairs only from label 1 versus label 0. Label -1 is absent from the loss.",
            "- The loss remains full-pair `softplus(-(s_pos-s_neg-margin))` with margin 0.2.",
            "- Inference keeps spatial ratio 0.1, cosine redundancy weight 1.0, threshold 0.7, the same budgets, and the same Focal-Spatial selection logic.",
            "- Cached-feature selection was checked against the original GAZE full forward on one sample at all three budgets; selected indices matched exactly.",
            f"- Downstream aggregation found {downstream_duplicate_retries} identical retry result and deduplicated it by condition/budget; conflicting retry metrics would raise an error.",
            "", "## Code changes", "",
            "- Added strict 1K acceptance, original-GAZE BBox projection reconstruction, and three-value label validation.",
            "- Added a three-value pairwise margin-ranking loss: label 1 is positive, label 0 is negative, and label -1 is excluded.",
            "- Added frozen-feature VIS-only B/C training with matched initialization/order/hyperparameters and per-epoch checkpoints.",
            "- Added full token, DocVQA, FGVC, paired-bootstrap, and qualitative A/B/C evaluation/reporting.",
            "", "## Visualizations", "",
            f"- Acceptance overview: `{acceptance['visualizations']['overview']}`",
            f"- A/B/C case overview: `{visualization_summary['overview']}`",
        ]
    )
    lines.extend(expansion_lines)
    (work_dir / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/mask_gaze_1k/preexperiment"))
    args = parser.parse_args()
    result = generate_report(args.work_dir)
    print(json.dumps({"token_rows": len(result["token_final_table"]), "fgvc_rows": len(result["fgvc_results"]), "downstream_rows": len(result["downstream_results"])}, indent=2))


if __name__ == "__main__":
    main()
