#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

# Check for the required arguments.
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <dataset> <cuda_id>"
    exit 1
fi

dataset=$1
cuda_id=$2

# Set the date and CUDA device.
today=$(date +%Y-%m-%d)
export CUDA_VISIBLE_DEVICES=$cuda_id
#disable long logging from vllm
export VLLM_LOGGING_LEVEL=ERROR

base_path="/home/"
output_name="merged_model"
model_path="$base_path/Huggingface/llava-1.5-7b-hf"
retriever_path="$base_path/Huggingface/e5-large-v2"
data_path="$base_path/datasets/finer"
export WANDB_PROJECT=$dataset

# Function to find an available port.
get_available_port() {
    while true; do
        port=$(shuf -i 20000-30000 -n 1)
        if ! ss -tuln | grep -q ":$port "; then
            echo $port
            return 0
        fi
    done
}

# Round 1: Initial data generation and training.
round=1
output_dir="$base_path/checkpoint-$round"

# Step 1: Data generation scripts.
python src/step1.1_desc_gen_vllm.py --dataset $dataset --data_path $data_path --model_path $model_path
python src/step1.2_concepts_gen.py --dataset $dataset --data_path $data_path --retriever_path $retriever_path
python src/step1.3_QA_gen_vllm.py --dataset $dataset --data_path $data_path --model_path $model_path

# Get an available port for training.
available_port=$(get_available_port)
echo "Using available port: $available_port"

# Step 2: Training with Accelerate.
#if you have troble installing deepspeed, you can try to use gpu_all_config.yaml
accelerate launch --config_file "configs/deepspeed_config.yaml" --main_process_port $available_port src/step2_llava_sft.py \
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
    --num_train_epochs 2 \
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

echo "First round training completed successfully."

# Function to execute steps for rounds 2 and 3.
prev_sft_model=$model_path
run_round() {
    local round=$1
    local prev_round=$((round - 1))
    local lora_path="$base_path/checkpoint-$prev_round/$dataset-$today"
    local sft_model="$lora_path/$output_name"
    local output_dir="$base_path/checkpoint-$round"

    #local lr=$(python -c "print(3e-5 / (2 ** ($round - 1)))")

    # Step 1: Model saving and evaluation scripts.
    python src/stepx_llava_lora_save.py --dataset $dataset --model_path $prev_sft_model --output_path $lora_path --output_file $output_name
    python src/step2.1_batch_eval.py --dataset $dataset --model_path $sft_model --data_path $data_path --version $today --round $prev_round
    python src/step2.5_ans_gen.py --dataset $dataset --model_path $sft_model --data_path $data_path --round $round
    python src/step2.6_ans_select.py --dataset $dataset --data_path $data_path --retriever_path $retriever_path --round $round

    # Get an available port for training.
    available_port=$(get_available_port)
    echo "Using available port: $available_port"

    # Step 2: Training with Accelerate.
    accelerate launch --config_file "configs/deepspeed_config.yaml" --main_process_port $available_port src/step2_llava_sft.py \
        --lora_enable True --lora_r 128 --lora_alpha 256 \
        --model_name_or_path $sft_model \
        --data_path $data_path \
        --dataset_name $dataset/round-$round-QA_gen_$dataset.json \
        --remove_unused_columns false \
        --freeze_vision True \
        --freeze_llm False \
        --output_dir $output_dir/$dataset-$today \
        --bf16 true \
        --dataloader_pin_memory True \
        --dataloader_num_workers 64 \
        --dataloader_persistent_workers True \
        --num_train_epochs 2 \
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

    prev_sft_model=$sft_model
    echo "Round $round training completed successfully."
}

# Execute rounds 2 and 3.
for round in 2 3 4; do
    run_round $round
done

round=4
lora_path="$base_path/checkpoint-$round/$dataset-$today"
sft_model="$lora_path/$output_name"
output_dir="$base_path/checkpoint-$round"

python src/stepx_llava_lora_save.py --dataset $dataset --model_path $prev_sft_model --output_path $lora_path --output_file $output_name
python src/step2.1_batch_eval.py --dataset $dataset --model_path $sft_model --data_path $data_path --version $today --round $round
echo "All running completed successfully."
