#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_PYTHON="${BASE_PYTHON:-/home/ruimeng/miniconda3/envs/vs_env/bin/python}"

cd "${PROJECT_ROOT}"
if [[ ! -x .venv/bin/python ]]; then
  "${BASE_PYTHON}" -m venv --system-site-packages .venv
fi
.venv/bin/python -m pip install --no-deps \
  'git+https://github.com/facebookresearch/segment-anything.git@6fdee8f2727f4506cfbbe553e23b895e27956588'
.venv/bin/python -c 'import cv2, pycocotools, qwen_vl_utils, segment_anything, torch, transformers, yaml; print("runtime imports OK")'

