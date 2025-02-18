#!/bin/bash

set -e

dataset=$1  
cuda_id=$2      

model_path='./llava-1.5-7b-hf'
retriever_path='./e5-large-v2'
data_path='./datasets/finer'

export CUDA_VISIBLE_DEVICES=$cuda_id

python src/step1.1_desc_gen_vllm.py --dataset $dataset --data_path $data_path --model_path $model_path
python src/step1.2_concepts_gen.py --dataset $dataset --data_path $data_path --retriever_path $retriever_path
python src/step1.3_QA_gen_vllm.py --dataset $dataset --data_path $data_path --model_path $model_path