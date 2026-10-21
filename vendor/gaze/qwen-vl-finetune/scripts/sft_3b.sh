export NCCL_SOCKET_IFNAME=eno2
export GLOO_SOCKET_IFNAME=eno2

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

set -euo pipefail

# Ensure paths are resolved relative to repo root even when launched from elsewhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_DEVICE_MAX_CONNECTIONS=1
# Enable bbox budget debug logs (set to 0 to disable)
export BBOX_DEBUG=${BBOX_DEBUG:-1}

# Distributed training configuration
export TORCH_DISTRIBUTED_RUN_TIMEOUT=3600
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
# NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)  # Automatically detects available GPUs
NPROC_PER_NODE=1
# Zero Stage 3 策略。这是一种显存优化技术，把模型参数切分到不同显卡上。
deepspeed=./scripts/zero3.json
# Model configuration
llm=/data/ModelZoo/LVLM/Qwen-Qwen2.5-VL-3B-Instruct

# Training hyperparameters
lr=5e-5
batch_size=${BATCH_SIZE:-8}
grad_accum_steps=${GRAD_ACCUM_STEPS:-4}
spatial_ratio=${SPATIAL_RATIO:-${CONTEXTUAL_RATIO:-0.1}}
cosine_penalty_weight=${COSINE_PENALTY_WEIGHT:-1}
cosine_penalty_threshold=${COSINE_PENALTY_THRESHOLD:-0.7}

# Training entry point
entry_file=qwenvl/train/train_qwen_selector.py

# Dataset configuration (replace with public dataset names)
# Default to mkz 4k mix dataset; override with env var DATASETS=viscot (or others).
datasets=${DATASETS:-viscot_mkz_bbox_4k}
# Output configuration
run_name="qwen25vl-baseline"
output_dir=../output_ckpt/VisionSelector-Qwen2.5-VL-3B

# Training arguments
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
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --seed 42 \
    --data_seed 42 \
    --report_to none"

# Launch training
CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
