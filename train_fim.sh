#!/bin/bash

set -x

MODEL_PATH=/root/autodl-tmp/deepseek-coder-1.3b-base

CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
  --model_name_or_path ${MODEL_PATH} \
  --trust_remote_code \
  --stage sft \
  --do_train \
  --finetuning_type lora \
  --lora_rank 8 \
  --lora_target q_proj,v_proj \
  --dataset mixed_finetune_dataset \
  --dataset_dir ../data \
  --template deepseek \
  --cutoff_len 2048 \
  --max_samples 100000 \
  --overwrite_cache \
  --preprocessing_num_workers 16 \
  --dataloader_num_workers 4 \
  --output_dir output/lora_saves \
  --logging_steps 10 \
  --save_steps 500 \
  --plot_loss \
  --overwrite_output_dir \
  --save_only_model false \
  --report_to none \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --num_train_epochs 3.0 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16 \
  --ddp_timeout 180000000
