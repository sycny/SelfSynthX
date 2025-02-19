#!/bin/bash

dataset=$1
version=$2
cuda_id=$3



output_dir='./checkpoint'
sft_path=./checkpoint-3/$dataset-$version/merged_model

model_path='./llava-1.5-7b-hf'

export CUDA_VISIBLE_DEVICES=$cuda_id
echo "Using CUDA device: $CUDA_VISIBLE_DEVICES"

export HF_DATASETS_CACHE='./HF_HOME/datasets'

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


accelerate launch --main_process_port $available_port \
        --num_processes=1  \
        -m lmms_eval \
        --model llava_hf  \
        --model_args pretrained=$sft_path,device_map="cuda" \
        --tasks mmstar,seedbench_2_plus,mmmu_val,mme,mmbench_en_dev     \
        --batch_size 1     \
        --log_samples    \
        --output_path ./logs/