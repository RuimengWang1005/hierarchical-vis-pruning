# Hierarchical VIS Pruning — 200-sample pseudo-label pilot

This repository generates deterministic four-way Qwen visual-token labels:

- `2`: task-discriminative detail
- `1`: ordinary subject/foreground
- `0`: background
- `-1`: uncertain boundary or label conflict

It is an independent Git project. `/home/ruimeng/gaze` and `/home/bbhan/gaze` are read-only inputs; no file in either GAZE tree is modified or imported at runtime.

## Recovered GAZE inputs

The actual training registration in `/home/ruimeng/gaze/qwen-vl-finetune/qwenvl/data/__init__.py` identifies:

- 20K training list: `/home/bbhan/gaze/viscot_mix_20k_30-15-15-10-5-5-5-5-5_seed42.jsonl`
- image root: `/data/DataSet/vlm_dataset/cot_image_data/`

The 20K file has exactly 20,000 `split=train` rows and SHA-256
`0d43a5c31b89f4a38b4db2af2274abd6e550bbf526f686c073a435d27ef4e157`.
It covers GQA, DocVQA, TextVQA, VSR, Visual7W (`v7w`), DUDE, SROIE,
Flickr30k, and InfographicsVQA. The similarly named five-source file under
`/home/ruimeng/gaze/viscot_dataset/` is deliberately not used.

The frozen teacher checkpoint is
`/data/ModelZoo/LVLM/Qwen-Qwen2.5-VL-7B-Instruct`. The SAM checkpoint is
`/data/ModelZoo/CV/SAM/sam_vit_h_4b8939.pth`.

No GAZE source file was copied. The following files were consulted read-only to
match behavior and are named here for provenance:

- `qwen-vl-finetune/qwenvl/data/__init__.py`: annotation and image roots
- `qwen-vl-finetune/scripts/sft_7b.sh`: model path, seed, and training-time
  `min_pixels=3136`, `max_pixels=401408`
- `qwen-vl-finetune/qwenvl/data/data_qwen.py`: `cot/` removal, Qwen processor
  use, `image_grid_thw`, and bbox coordinate convention

## Environment

Audited base environment: Python 3.10, PyTorch 2.5.1+cu121, Transformers
4.50.0, and four RTX 4090 GPUs. The project reuses the installed `vs_env`
packages while installing only Meta's small Segment Anything Python package
into a project-local `.venv`. It does not download any model weights.

```bash
cd /home/ruimeng/hierarchical-vis-pruning
bash scripts/bootstrap_env.sh
```

All input paths, thresholds, devices, scoring weights, preprocessing limits,
and seed 42 live in `configs/pilot_200.yaml`.

## Run

Run tests first:

```bash
PYTHONPATH=. .venv/bin/python -m pytest
```

Run the full resumable pipeline:

```bash
bash scripts/run_pilot_200.sh
```

Each stage can be rerun independently. Existing JSONL rows are skipped unless
`--overwrite` is supplied.

```bash
STAGE=manifest bash scripts/run_pilot_200.sh
STAGE=subject bash scripts/run_pilot_200.sh
STAGE=candidates bash scripts/run_pilot_200.sh
STAGE=filter bash scripts/run_pilot_200.sh
STAGE=teacher bash scripts/run_pilot_200.sh
STAGE=tokens bash scripts/run_pilot_200.sh
STAGE=visualize bash scripts/run_pilot_200.sh
STAGE=report bash scripts/run_pilot_200.sh
STAGE=validate bash scripts/run_pilot_200.sh
```

The major stages are:

1. Audit exact data/model/checkpoint availability and versions.
2. Proportionally stratify exactly 200 unique valid samples from GAZE's 20K
   training list with seed 42.
3. Generate one SAM subject mask per bbox and union instances, with an explicit
   bbox fallback.
4. Expand each subject bbox by 15%, distribute 24 farthest-point prompts across
   instances, batch three SAM masks per point, then filter and deduplicate to at
   most 12 masks.
5. Extract post-merger Qwen visual features, pool them with continuous mask
   coverage, and retain at most five candidates for counterfactual scoring.
6. Compute answer-length-normalized NLL before and after replacing covered
   visual features with the same image's background-token mean. This preserves
   sequence length and position; it never uses a black image patch.
7. Select zero to three details, map masks through the real per-image
   `image_grid_thw`, and assert label length equals the exact number of visual
   tokens sent to the LLM.
8. Render every successful sample, a deterministic 50-sample HTML overview,
   and aggregate JSON/CSV reports.

## Outputs

Generated data is ignored by Git under `outputs/pilot_200/`:

- `environment.json`: hardware, package, model, and input audit
- `manifest.jsonl`: the exact deterministic 200-row sample list
- `manifest_report.json`, `manifest_failures.jsonl`: sampling provenance
- `subject_masks.jsonl`, `raw_candidates.jsonl`, `filtered_candidates.jsonl`:
  resumable stage records and filtering reasons
- `teacher_scores.jsonl`: separate question, answer, question+answer,
  counterfactual, SAM, normalized, and utility scores
- `dataset.jsonl`: final four-way token labels and continuous confidences
- `masks/*.subject.npz`: compact subject/instance arrays for diagnostics; JSONL
  mask fields use compressed COCO RLE
- `visualizations/*.png`: full per-sample review panels
- `visualizations/overview.html`: deterministic 50-sample review page
- `failures.jsonl`: processing failures with stage and exception
- `summary.json`, `summary.csv`: requested distributions, timings, token ratios,
  fallback rate, alignment checks, and peak GPU memory
- `validation_report.json`: strict final cardinality, RLE-shape, class-domain,
  candidate-cap, visualization-count, and token-length assertions

## Token alignment

Qwen's image processor directly resizes each image to dimensions derived from
its true `image_grid_thw`; it does not add padding in the audited Transformers
4.50.0 implementation. A mask is resized to that exact pixel canvas with area
interpolation and averaged over each `patch_size * spatial_merge_size` cell.
The code does not assume a fixed grid. Unit tests cover exact cell coverage,
dynamic grid sizes, four labels, RLE round trips, deduplication, and preservation
of nested candidates.

## Known limits

- The pilot uses feature-level counterfactual replacement, which is less
  human-visible than image blurring. Visualizations therefore show the exact
  token regions whose features were replaced.
- Semantic scores compare pooled post-merger visual vectors with frozen Qwen
  input-token embeddings. Separate components are retained for later ablation.
- Initial utility weights and thresholds are pilot heuristics, not scientific
  conclusions.
- The optional 160/40 VIS training smoke test is intentionally deferred until
  the 200 labels pass human review; this repository makes no benchmark claim.

