#!/bin/bash

# 1. 核心环境变量补丁
export CUDA_HOME=/xxx/fake_cuda
export PATH=$CUDA_HOME/bin:$PATH
export DS_SKIP_CUDA_CHECK=1
export DS_BUILD_OPS=0
export HF_HUB_OFFLINE=1
export NCCL_P2P_DISABLE=1

# 2. Wandb 配置
export WANDB_API_KEY="xxx"
export WANDB_PROJECT="Qwen3-Tool-SFT"

# 定义输出目录（和训练参数中的 output_dir 保持一致）
OUTPUT_DIR="compass/train/qwen3_8b"
# 自动检测最新的检查点（Swift 会按步数生成 checkpoint-xxx 目录）
LATEST_CHECKPOINT=$(ls -dt ${OUTPUT_DIR}/checkpoint-* 2>/dev/null | head -n 1)

echo "🔥 正在启动 Qwen3-8B ..."

# 3. 执行训练（支持断点续训）
CUDA_VISIBLE_DEVICES=6,7 \
NPROC_PER_NODE=2 \
swift sft \
    --model compass/train/model/Qwen3-8B \
    --train_type lora \
    --dataset compass/train/datasets/profile_based_part1_100users.jsonl compass/train/datasets/profile_based_part2_100users.jsonl compass/train/datasets/history_based_200users.jsonl\
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --max_length 32768 \
    --gradient_checkpointing true \
    --output_dir ${OUTPUT_DIR} \
    --run_name "Qwen3-8B" \
    --logging_steps 5 \
    --save_steps 100 \
    --save_total_limit 3 \
    --dataloader_num_workers 2 \
    --report_to wandb \
    # 关键：如果检测到断点则续训，否则从头开始
    $( [ -n "${LATEST_CHECKPOINT}" ] && echo "--resume_from_checkpoint ${LATEST_CHECKPOINT}" )