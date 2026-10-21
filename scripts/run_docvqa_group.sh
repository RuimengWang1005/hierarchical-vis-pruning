#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 CONDITION SCORER_PATH CUDA_INDEX" >&2
  exit 2
fi

CONDITION="$1"
SCORER_PATH="$2"
CUDA_INDEX="$3"
PROJECT_ROOT="/home/ruimeng/hierarchical-vis-pruning"
EVAL_ROOT="/home/ruimeng/gaze/qwen-evaluation1"
LMMS_ROOT="/home/ruimeng/VisionSelector/lmms-eval"
MODEL_PATH="/home/ruimeng/gaze/output_ckpt666/VisionSelector-Qwen2.5-VL-7B"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/mask_gaze_1k/preexperiment/downstream_evaluation"

export CUDA_VISIBLE_DEVICES="${CUDA_INDEX}"
export EXTERNAL_VIS_SCORER="${SCORER_PATH}"
export PYTHONPATH="${PROJECT_ROOT}/tools/scorer_injection:${EVAL_ROOT}:${LMMS_ROOT}:${PROJECT_ROOT}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

for BUDGET in 0.1 0.2 0.3; do
  BUDGET_PERCENT="$(awk -v value="${BUDGET}" 'BEGIN { printf "%d", value * 100 }')"
  OUTPUT_DIR="${OUTPUT_ROOT}/${CONDITION}/budget_${BUDGET_PERCENT}"
  if find "${OUTPUT_DIR}" -type f -name '*_results.json' -print -quit 2>/dev/null | grep -q .; then
    echo "[skip existing] ${CONDITION} budget=${BUDGET}"
    continue
  fi
  mkdir -p "${OUTPUT_DIR}"
  "${PROJECT_ROOT}/.venv/bin/python" -m lmms_eval \
    --model qwen2_5_vl_with_token_compression \
    --model_args "pretrained=${MODEL_PATH},method=selector,budgets=${BUDGET},spatial_ratio=0.1,cosine_penalty_weight=1,cosine_penalty_threshold=0.7,selector_focal_redundancy_limit=0,max_pixels=1605632,min_pixels=200704,attn_implementation=flash_attention_2" \
    --tasks docvqa_val \
    --batch_size 1 \
    --device cuda:0 \
    --seed 42 \
    --output_path "${OUTPUT_DIR}"
done
