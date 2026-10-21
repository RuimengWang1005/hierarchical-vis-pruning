#!/bin/bash

# 网络与底层通信配置 (解决4090各种怪异报错)
export NCCL_SOCKET_IFNAME=eno2
export GLOO_SOCKET_IFNAME=eno2
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export BBOX_DEBUG=${BBOX_DEBUG:-1}

export TORCH_DISTRIBUTED_RUN_TIMEOUT=3600
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=1

deepspeed=./scripts/zero2.json
llm=/data/ModelZoo/LVLM/Qwen-Qwen2.5-VL-7B-Instruct

lr=5e-5
# 【修改点1】降低单卡批次，防止显存OOM
batch_size=${BATCH_SIZE:-1}
# 【修改点2】提高梯度累加，弥补批次降低的影响 (1*32 = 原来的 4*8)
grad_accum_steps=${GRAD_ACCUM_STEPS:-32}

spatial_ratio=${SPATIAL_RATIO:-${CONTEXTUAL_RATIO:-0.1}}
cosine_penalty_weight=${COSINE_PENALTY_WEIGHT:-1}
cosine_penalty_threshold=${COSINE_PENALTY_THRESHOLD:-0.7}

entry_file=qwenvl/train/train_qwen_selector.py
datasets=${DATASETS:-viscot_mask_20k_exact}
run_name="qwen25vl-mask-20k-exact"
output_dir=${OUTPUT_DIR:-/home/ruimeng/hierarchical-vis-pruning/outputs/mask_gaze_20k_full/VisionSelector-Qwen2.5-VL-7B}

args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm False \
    --tune_compressor True \
    --budget 0.2 \
    --spatial_ratio ${spatial_ratio} \
    --cosine_penalty_weight ${cosine_penalty_weight} \
    --cosine_penalty_threshold ${cosine_penalty_threshold} \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --reg_weight_start 0.1 \
    --reg_weight_end 2.0 \
    --bbox_loss_weight 1 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --min_pixels 3136 \
    --max_pixels 401408 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --remove_unused_columns False \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --run_name ${run_name} \
    --seed 42 \
    --data_seed 42 \
    --report_to none"

# 【修改点3】强制指定 1 号卡！终于解救你的显存冲突了！
CUDA_VISIBLE_DEVICES="3" torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
