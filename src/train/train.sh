#!/bin/bash

export PYTHONPATH=$(pwd)

script_dir=$(dirname "$0")

deepspeed --include localhost:0,1,2,3 $script_dir/train/train.py \
    --deepspeed $script_dir/train/ds_zero2_no_offload.json \
    --llm_pretrained_model_name_or_path lmsys/vicuna-7b-v1.5 \
    --train_type train_both \
    --use_lora True \
    --lora_r 128 \
    --lora_alpha 256 \
    --vision_hidden_size 1027 \
    --source joint_all \
    --bf16 true \
    --fp16 false \
    --dataloader_pin_memory True \
    --dataloader_num_workers 8 \
    --dataloader_persistent_workers True \
    --output_dir $script_dir/output/llemr_vicuna \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy steps \
    --eval_steps 0.05 \
    --save_strategy steps \
    --save_steps 0.05 \
    --save_total_limit 4 \
    --load_best_model_at_end True \
    --run_name llemr_vicuna \
    --learning_rate 1e-4 \
    --logging_steps 1
