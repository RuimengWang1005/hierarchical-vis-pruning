#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/pilot_200.yaml}"
PYTHON="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
STAGE="${STAGE:-all}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

exec "${PYTHON}" -m src.pipeline --config "${CONFIG}" --stage "${STAGE}" "$@"

