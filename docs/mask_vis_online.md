# Mask-GAZE 20K online VIS training

The training path reads original images, runs the frozen GAZE/Qwen2.5-VL vision
tower online, and immediately feeds the current image's merged visual tokens to
the unchanged GAZE `TransformerScorer`. It does not load a language model and it
does not write visual token features.

No verified scorer from before GAZE BBox supervision exists on this server. The
default and documented initialization is therefore the original GAZE scorer
constructor with seed 42. A genuinely pre-BBox scorer can instead be supplied
explicitly with `--scorer-init`.

Run all commands from `/home/ruimeng/hierarchical-vis-pruning`.

Prepare the fixed 19,668-sample training list:

```bash
.venv/bin/python scripts/train_mask_vis_online.py --config configs/train_mask_vis_20k.yaml --mode prepare
```

Time 200 images without touching the formal training directory:

```bash
.venv/bin/python scripts/train_mask_vis_online.py --config configs/train_mask_vis_20k.yaml --mode benchmark --device cuda:0 --output-dir outputs/mask_vis_20k_benchmark_200 --epochs 1 --batch-size 1 --gradient-accumulation-steps 32 --learning-rate 5e-5 --margin 0.2 --seed 42 --max-samples 200 --num-workers 2 --init-mode original_random
```

Train one formal epoch:

```bash
.venv/bin/python scripts/train_mask_vis_online.py --config configs/train_mask_vis_20k.yaml --mode train --device cuda:0 --output-dir outputs/mask_vis_20k_online --epochs 1 --batch-size 1 --gradient-accumulation-steps 32 --learning-rate 5e-5 --margin 0.2 --seed 42 --num-workers 2 --init-mode original_random
```

Resume an interrupted one-epoch run from its exact sample/update position:

```bash
.venv/bin/python scripts/train_mask_vis_online.py --config configs/train_mask_vis_20k.yaml --mode train --device cuda:0 --output-dir outputs/mask_vis_20k_online --epochs 1 --batch-size 1 --gradient-accumulation-steps 32 --learning-rate 5e-5 --margin 0.2 --seed 42 --num-workers 2 --resume outputs/mask_vis_20k_online/checkpoint_last.pt
```

Evaluate a trained epoch on pilot 200 against the original GAZE scorer:

```bash
.venv/bin/python scripts/eval_mask_vis_pilot.py --config configs/train_mask_vis_20k.yaml --original-scorer /home/ruimeng/gaze/output_ckpt666/VisionSelector-Qwen2.5-VL-7B --mask-scorer outputs/mask_vis_20k_online/scorer_epoch_1.safetensors --output-dir outputs/mask_vis_20k_online/pilot_200_eval --device cuda:0
```

## General-task evaluation

The complete 7B GAZE general-task suite present on this server contains
`ai2d_no_mask`, `chartqa`, `docvqa_val`, `mme`, `mmmu_val`, `ocrbench`, `pope`,
and `scienceqa_img`, each at 10%, 20%, and 30% token budgets. The runner loads
the same full GAZE checkpoint and changes only the VIS scorer for the Mask-VIS
condition. It never writes visual-feature caches.

Run the embedded GAZE scorer and Mask-VIS scorer separately so both sides use
the currently available `output_ckpt666` checkpoint:

```bash
HF_HOME=/home/ruimeng/.cache/huggingface /home/ruimeng/miniconda3/envs/vs_env/bin/hf auth login
HF_OFFLINE=0 scripts/run_general_eval.sh gaze 0
HF_OFFLINE=0 scripts/run_general_eval.sh mask 0 outputs/mask_vis_20k_online/scorer_epoch_1.safetensors
```

`HF_OFFLINE=0` is needed if the Hugging Face dataset cache was cleared. Dataset
downloads, when needed, are confined to `/home/ruimeng/.cache/huggingface`; the
script refuses cache and output roots outside `/home/ruimeng`. After a complete
download, use `HF_OFFLINE=1` for subsequent runs. Seven of the original task
configs set `dataset_kwargs.token=True`, so the one-time Hugging Face login is
required even when the underlying dataset repository is public.

For a lower-cost first pass, run only 10% and 20% budgets:

```bash
BUDGETS="0.1 0.2" HF_OFFLINE=0 scripts/run_general_eval.sh gaze 0
BUDGETS="0.1 0.2" HF_OFFLINE=0 scripts/run_general_eval.sh mask 0 outputs/mask_vis_20k_online/scorer_epoch_1.safetensors
```

Run one task at a time by overriding `TASKS`, for example:

```bash
TASKS=pope BUDGETS="0.1 0.2" OUTPUT_ROOT=/home/ruimeng/hierarchical-vis-pruning/outputs/mask_vis_20k_online/general_eval_pope HF_OFFLINE=0 scripts/run_general_eval.sh gaze 0
TASKS=pope BUDGETS="0.1 0.2" OUTPUT_ROOT=/home/ruimeng/hierarchical-vis-pruning/outputs/mask_vis_20k_online/general_eval_pope HF_OFFLINE=0 scripts/run_general_eval.sh mask 0 outputs/mask_vis_20k_online/scorer_epoch_1.safetensors
```

Print the metric-by-metric comparison after both conditions finish:

```bash
.venv/bin/python scripts/summarize_general_eval.py --root outputs/mask_vis_20k_online/general_eval
```

If the Hugging Face metadata request for DocVQA is unstable after all six
validation Parquet shards have been downloaded, evaluate the local snapshot
without copying it:

```bash
TASKS=docvqa_val_local \
INCLUDE_PATH=/home/ruimeng/hierarchical-vis-pruning/configs/lmms_tasks/local_docvqa \
OUTPUT_ROOT=/home/ruimeng/hierarchical-vis-pruning/outputs/mask_vis_20k_online/general_eval_docvqa \
HF_OFFLINE=1 \
scripts/run_general_eval.sh mask 0 outputs/mask_vis_20k_online/scorer_epoch_1.safetensors
```
