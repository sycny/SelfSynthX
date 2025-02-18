#!/bin/bash

set -e

dataset=$1  
cuda_id=$2  

model_path='./llava-1.5-7b-hf'
retriever_path='./e5-large-v2'
data_path='./datasets/finer'
output_dir='./checkpoint'
project=$dataset

export WANDB_PROJECT=$project

today=$(date +%Y-%m-%d)

export CUDA_VISIBLE_DEVICES=$cuda_id

get_available_port() {
    while true; do
        port=$(shuf -i 20000-30000 -n 1)
        if ! ss -tuln | grep -q ":$port "; then
            echo $port
            return 0
        fi
    done
}

available_port=$(get_available_port)
echo "Using available port: $available_port"

accelerate launch --config_file "configs/deepspeed_config.yaml"  --main_process_port $available_port src/step2_llava_sft.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 \
    --model_name_or_path $model_path \
    --data_path $data_path \
    --dataset_name $dataset/round-1-QA_gen_$dataset.json \
    --remove_unused_columns false \
    --freeze_vision True \
    --freeze_llm False \
    --output_dir $output_dir/$dataset-$today \
    --bf16 true \
    --dataloader_pin_memory True \
    --dataloader_num_workers 64 \
    --dataloader_persistent_workers True \
    --num_train_epochs 3 \
    --gradient_checkpointing True \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --report_to none \
    --learning_rate 1e-5 \
    --logging_steps 1 \
    --resume_from_checkpoint False \
    --use_4bit_quantization False \
    --warmup_ratio 0.05

echo "All steps completed successfully."