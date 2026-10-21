#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_general_eval.sh gaze CUDA_INDEX
  scripts/run_general_eval.sh mask CUDA_INDEX SCORER_PATH
  scripts/run_general_eval.sh mask_gaze CUDA_INDEX

Environment overrides:
  TASKS            Comma-separated lmms-eval tasks. Each task runs separately.
  BUDGETS          Space-separated budgets (default: "0.1 0.2 0.3").
  MODEL_PATH       Full checkpoint override for gaze or mask_gaze.
  OUTPUT_ROOT      Result root under /home/ruimeng.
  HF_OFFLINE       1 to require an existing dataset cache; 0 to allow downloads.
  HF_HOME          Hugging Face cache root (must remain under /home/ruimeng).
  MAIN_PROCESS_PORT  Accelerate port (default: 28180).
  INCLUDE_PATH     Optional directory containing additional lmms-eval tasks.
  DRY_RUN          1 to print commands without launching evaluation.
EOF
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage >&2
  exit 2
fi

CONDITION="$1"
CUDA_INDEX="$2"
SCORER_PATH="${3:-}"

PROJECT_ROOT="/home/ruimeng/hierarchical-vis-pruning"
EVAL_ROOT="/home/ruimeng/gaze/qwen-evaluation1"
LMMS_ROOT="/home/ruimeng/VisionSelector/lmms-eval"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
GAZE_MODEL_PATH="/home/ruimeng/gaze/output_ckpt666/VisionSelector-Qwen2.5-VL-7B"
MASK_GAZE_MODEL_PATH="${PROJECT_ROOT}/outputs/mask_gaze_20k_full/VisionSelector-Qwen2.5-VL-7B"
MODEL_PATH_OVERRIDE="${MODEL_PATH:-}"
INJECTION_ROOT="${PROJECT_ROOT}/tools/scorer_injection"

# This is the complete 7B GAZE general-task suite found in
# gaze/qwen-evaluation1/result/eval_time. TextVQA is intentionally excluded:
# only unrelated 5%-budget baselines and one 3B visualization run are present.
DEFAULT_TASKS="ai2d_no_mask,chartqa,docvqa_val,mme,mmmu_val,ocrbench,pope,scienceqa_img"
TASKS="${TASKS:-${DEFAULT_TASKS}}"
BUDGETS="${BUDGETS:-0.1 0.2 0.3}"
HF_HOME="${HF_HOME:-/home/ruimeng/.cache/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
HF_OFFLINE="${HF_OFFLINE:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-28180}"
INCLUDE_PATH="${INCLUDE_PATH:-}"
DRY_RUN="${DRY_RUN:-0}"

case "${CONDITION}" in
  gaze)
    if [[ -n "${SCORER_PATH}" ]]; then
      echo "The gaze condition uses the scorer embedded in the full GAZE checkpoint; do not pass SCORER_PATH." >&2
      exit 2
    fi
    MODEL_PATH="${MODEL_PATH_OVERRIDE:-${GAZE_MODEL_PATH}}"
    unset EXTERNAL_VIS_SCORER || true
    ;;
  mask)
    if [[ -z "${SCORER_PATH}" || ! -f "${SCORER_PATH}" ]]; then
      echo "The mask condition requires an existing scorer .safetensors file." >&2
      exit 2
    fi
    MODEL_PATH="${MODEL_PATH_OVERRIDE:-${GAZE_MODEL_PATH}}"
    export EXTERNAL_VIS_SCORER="$(realpath "${SCORER_PATH}")"
    ;;
  mask_gaze)
    if [[ -n "${SCORER_PATH}" ]]; then
      echo "The mask_gaze condition loads its scorer from the full checkpoint; do not pass SCORER_PATH." >&2
      exit 2
    fi
    MODEL_PATH="${MODEL_PATH_OVERRIDE:-${MASK_GAZE_MODEL_PATH}}"
    unset EXTERNAL_VIS_SCORER || true
    ;;
  *)
    echo "CONDITION must be 'gaze', 'mask', or 'mask_gaze'." >&2
    usage >&2
    exit 2
    ;;
esac

if [[ "${CONDITION}" == "mask_gaze" ]]; then
  DEFAULT_OUTPUT_ROOT="${PROJECT_ROOT}/outputs/mask_gaze_20k_full/general_eval"
else
  DEFAULT_OUTPUT_ROOT="${PROJECT_ROOT}/outputs/mask_vis_20k_online/general_eval"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}}"

for required in "${PYTHON_BIN}" "${MODEL_PATH}/config.json" "${INJECTION_ROOT}/sitecustomize.py"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required path does not exist: ${required}" >&2
    exit 1
  fi
done

case "${OUTPUT_ROOT}" in
  /home/ruimeng/*) ;;
  *)
    echo "OUTPUT_ROOT must be under /home/ruimeng: ${OUTPUT_ROOT}" >&2
    exit 2
    ;;
esac
case "${HF_HOME}" in
  /home/ruimeng/*) ;;
  *)
    echo "HF_HOME must be under /home/ruimeng: ${HF_HOME}" >&2
    exit 2
    ;;
esac
if [[ "${TASKS}" != "${DEFAULT_TASKS}" && "${OUTPUT_ROOT}" == "${DEFAULT_OUTPUT_ROOT}" ]]; then
  echo "A TASKS subset requires a separate OUTPUT_ROOT so it cannot make the full suite look complete." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_INDEX}"
export PYTHONPATH="${INJECTION_ROOT}:${EVAL_ROOT}:${LMMS_ROOT}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME
export HF_DATASETS_CACHE
export HF_HUB_DISABLE_TELEMETRY=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

if [[ "${HF_OFFLINE}" == "1" ]]; then
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
else
  unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE || true
fi

# Seven tasks in the original GAZE suite set dataset_kwargs.token=True. The
# Hugging Face client treats this as a hard requirement even for public repos.
TOKEN_REQUIRED=0
for TOKEN_TASK in ai2d_no_mask chartqa docvqa_val mme ocrbench pope scienceqa_img; do
  if [[ ",${TASKS}," == *",${TOKEN_TASK},"* ]]; then
    TOKEN_REQUIRED=1
    break
  fi
done
TOKEN_FILE="${HF_TOKEN_PATH:-${HF_HOME}/token}"
if [[ "${TOKEN_REQUIRED}" == "1" \
      && -z "${HF_TOKEN:-}" \
      && -z "${HUGGING_FACE_HUB_TOKEN:-}" \
      && ! -s "${TOKEN_FILE}" \
      && ! -s "/home/ruimeng/.huggingface/token" ]]; then
  echo "A Hugging Face token is required by the selected lmms-eval task configs." >&2
  echo "Log in once with:" >&2
  echo "  HF_HOME=${HF_HOME} /home/ruimeng/miniconda3/envs/vs_env/bin/hf auth login" >&2
  exit 1
fi

MODEL_ARGS="pretrained=${MODEL_PATH},method=selector,budgets=PLACEHOLDER,spatial_ratio=0.1,cosine_penalty_weight=1,cosine_penalty_threshold=0.7,selector_focal_redundancy_limit=0,max_pixels=1605632,min_pixels=200704,video_reader_backend=decord,video_max_pixels=151200,video_min_pixels=100352,attn_implementation=flash_attention_2"

cd "${EVAL_ROOT}"

IFS=',' read -r -a TASKS_ARRAY <<< "${TASKS}"

if [[ "${#TASKS_ARRAY[@]}" -eq 0 ]]; then
  echo "TASKS must contain at least one task." >&2
  exit 2
fi

for TASK in "${TASKS_ARRAY[@]}"; do
  TASK="${TASK//[[:space:]]/}"
  if [[ -z "${TASK}" ]]; then
    echo "TASKS contains an empty task name: ${TASKS}" >&2
    exit 2
  fi

  for BUDGET in ${BUDGETS}; do
    OUTPUT_PATH="${OUTPUT_ROOT}/${CONDITION}/${TASK}/budget_${BUDGET}"
    LOG_FILE="${OUTPUT_PATH}/log_eval.log"

    if find "${OUTPUT_PATH}" -type f -name '*_results.json' -print -quit 2>/dev/null | grep -q .; then
      echo "[skip existing] condition=${CONDITION} task=${TASK} budget=${BUDGET} output=${OUTPUT_PATH}"
      continue
    fi

    CURRENT_MODEL_ARGS="${MODEL_ARGS/PLACEHOLDER/${BUDGET}}"
    CMD=(
      "${PYTHON_BIN}" -m accelerate.commands.launch
      --main_process_port "${MAIN_PROCESS_PORT}"
      --mixed_precision bf16
      --num_processes 1
      -m lmms_eval
      --model qwen2_5_vl_with_token_compression
      --model_args "${CURRENT_MODEL_ARGS}"
      --tasks "${TASK}"
      --batch_size 1
      --log_samples
      --output_path "${OUTPUT_PATH}"
      --log_samples_suffix "general_${CONDITION}_${TASK}_b${BUDGET}"
    )
    if [[ -n "${INCLUDE_PATH}" ]]; then
      if [[ ! -d "${INCLUDE_PATH}" ]]; then
        echo "INCLUDE_PATH does not exist: ${INCLUDE_PATH}" >&2
        exit 2
      fi
      CMD+=(--include_path "${INCLUDE_PATH}")
    fi

    echo "[general eval] condition=${CONDITION} task=${TASK} budget=${BUDGET}"
    echo "[general eval] full_checkpoint=${MODEL_PATH}"
    if [[ "${CONDITION}" == "mask" ]]; then
      echo "[general eval] external_scorer=${EXTERNAL_VIS_SCORER}"
    elif [[ "${CONDITION}" == "mask_gaze" ]]; then
      echo "[general eval] external_scorer=none (embedded Mask-GAZE scorer)"
    else
      echo "[general eval] external_scorer=none (embedded GAZE scorer)"
    fi
    echo "[general eval] output=${OUTPUT_PATH}"

    if [[ "${DRY_RUN}" == "1" ]]; then
      printf '[dry run]'
      printf ' %q' "${CMD[@]}"
      printf '\n'
      continue
    fi

    mkdir -p "${OUTPUT_PATH}"
    EVAL_TIME=true "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
    if ! find "${OUTPUT_PATH}" -type f -name '*_results.json' -print -quit | grep -q .; then
      echo "lmms-eval returned without a result JSON; stop before advancing." >&2
      echo "Failed combination: condition=${CONDITION} task=${TASK} budget=${BUDGET}" >&2
      echo "Inspect: ${LOG_FILE}" >&2
      exit 1
    fi
  done
done

