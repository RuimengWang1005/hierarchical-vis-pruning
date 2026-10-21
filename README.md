# Hierarchical VIS Pruning — 200-sample pseudo-label pilot

This repository generates deterministic four-way Qwen visual-token labels:

- `2`: task-discriminative detail
- `1`: ordinary subject/foreground
- `0`: background
- `-1`: uncertain boundary or label conflict

The recommended v2 output also stores a three-class soft target for every
visual token. Its fixed class order is `[background, foreground, detail]`, and
the three entries sum to one. Hard labels remain useful for review and for
legacy consumers, but training should prefer `token_soft_targets`.

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

To regenerate the v2 token supervision from the frozen v1 SAM/Qwen scores
without rerunning either model:

```bash
CONFIG=configs/pilot_200_v2.yaml STAGE=tokens bash scripts/run_pilot_200.sh --overwrite
CONFIG=configs/pilot_200_v2.yaml STAGE=visualize bash scripts/run_pilot_200.sh --overwrite
CONFIG=configs/pilot_200_v2.yaml STAGE=report bash scripts/run_pilot_200.sh
CONFIG=configs/pilot_200_v2.yaml STAGE=validate bash scripts/run_pilot_200.sh
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

## V2 supervision

For subject coverage `s` and selected-detail coverage `d`, v2 clips detail to
the subject (`d = min(d, s)`) and writes this loss-ready target:

```text
background = 1 - s
foreground = s - d
detail = d
```

Partial text/object coverage therefore supplies a proportional gradient rather
than disappearing behind an ignore label. Hard-label thresholds are 0.01 for
background, 0.10 for foreground, and 0.20 for detail. A deterministic argmax
fallback guarantees at least one positive token for every non-empty subject
and at least one detail token for every selected detail mask. The aggregate
report includes class mass, raw inverse-frequency weights, more conservative
square-root-inverse suggested weights, per-source positive/detail coverage, and
fallback counts.

For a three-logit VIS head, the intended loss is weighted soft cross-entropy:

```text
loss = mean(-sum_c class_weight[c] * token_soft_targets[c] * log_softmax(logits)[c])
```

Use `suggested_soft_loss_class_weights` from `summary.json` as the initial
weights. Do not apply the hard-label `-1` ignore mask to this soft loss; the
hard labels and orange cells are only the compatibility/QC view in v2.

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

## Mask-GAZE 1K binary-ranking pre-experiment

The Mask-GAZE experiment is independent of the three-level pilot. It trains
only the existing VIS importance scorer; the Qwen vision encoder is run once
under inference mode to build a frozen feature cache, and the language model is
not loaded during training.

```bash
# Strict correspondence/RLE/grid/coverage gate plus 60 review images.
PYTHONPATH=. .venv/bin/python -m src.mask_gaze_preexperiment

# Frozen Qwen post-merger feature cache (requires a GPU).
PYTHONPATH=. .venv/bin/python -m src.train_mask_gaze_preexperiment features --device cuda:0

# Three epochs for paired BBox/Mask scorers and unchanged GAZE selection eval.
PYTHONPATH=. .venv/bin/python -m src.train_mask_gaze_preexperiment train-eval --device cuda:0

# Paired bootstrap analysis and A/B/C case visualizations.
PYTHONPATH=. .venv/bin/python -m src.report_mask_gaze_preexperiment
```

Group B deliberately reconstructs the original GAZE target: each pixel BBox
is directly projected to the post-spatial-merge grid and filled as an inclusive
binary rectangle. It does not consume the adaptive 90%-mass BBox diagnostic in
the offline JSONL. Group C uses label `1` for adaptive SAM subject tokens,
label `0` for reliable background, and excludes label `-1` from every ranking
pair. Both groups use the original full-pair softplus margin-ranking form and
margin 0.2.

The experiment artifacts live under `outputs/mask_gaze_1k/preexperiment/`:

- `acceptance/acceptance_report.md` and `acceptance_summary.json`
- `acceptance/accepted_supervision.jsonl`, including original-GAZE BBox labels
- `training/{bbox,mask}/epoch_{1,2,3}.safetensors`
- `token_evaluation/*_summary.json` and `*_per_sample.jsonl`
- `analysis_summary.json`, `abc_token_results.csv`, and visualization overviews
- `fgvc_evaluation/` and `downstream_evaluation/` for full downstream runs

