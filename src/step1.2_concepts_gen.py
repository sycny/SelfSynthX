import json
import time
import random
import numpy as np
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str, default='llava')
        parser.add_argument("--dataset", type=str, default="cub-200")
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--retriever_path", type=str, default='./e5-large-v2')
        parser.add_argument("--data_path", type=str, default="./datasets/finer")
        parser.add_argument("--output_file", type=str, default="step1_desc_30.json")
        parser.add_argument("--device", type=str, default="cuda")
        self.args = parser.parse_args()
        
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f) 
    return data

def prepare_concpets(data_src):
    
    concpets = {}
    raw_concpets = load_json(data_src)
    for item in raw_concpets.values():
        concpets[item["name"]]=item["visual_features"]
        
    return concpets

def get_embeddings(texts, tokenizer, model):
    
    def average_pool(last_hidden_states
                    ,attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        
    embeddings = average_pool(outputs[0], inputs['attention_mask'])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    return embeddings

#tau should be set to 0.01
def compute_info_nce_scores(A_embeddings, B_embeddings, C_embeddings, tau=0.1):
   
    S_AB = torch.mm(A_embeddings, B_embeddings.T)  # [m, n]
    S_BC = torch.mm(B_embeddings, C_embeddings.T)  # [n, l]
    
    exp_pos = torch.exp(S_AB / tau)  # [m, n]
    exp_neg = torch.exp(S_BC / tau)  # [n, l]
    exp_neg_sum = torch.sum(exp_neg, dim=1, keepdim=True)  # [n, 1]
    
    exp_neg_sum_broadcasted = exp_neg_sum.T  # [1, n] ->  [m, n]

    mutual_info = torch.sum(torch.log(exp_pos / (exp_pos + exp_neg_sum_broadcasted)), dim=0)  # [n]
    
    return mutual_info

def select_top_k_concepts(info_nce_scores, B_texts, k=5):
    _, top_k_indices = torch.topk(info_nce_scores, k, largest=True)
    top_k_texts = [B_texts[i] for i in top_k_indices.tolist()]
    return top_k_texts

def save_results(results, output_file):
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results have been saved to {output_file}")

def precompute_embeddings(texts, tokenizer, model):
    all_embeddings = {}
    #for key, text_list in tqdm(texts.items(), desc="Computing embeddings"):
    for key, text_list in texts.items():
        all_embeddings[key] = get_embeddings(text_list, tokenizer, model)
    return all_embeddings


def sample_negative_embeddings(all_embeddings, current_key,  desc, n_samples=1000):
    other_keys = [k for k in desc.keys() if k != current_key]
    #sampled_keys = random.sample(other_keys, min(n_samples, len(other_keys)))
    #sampled_embeddings = torch.cat([all_embeddings[k] for k in sampled_keys], dim=0)
    sampled_embeddings = torch.cat([all_embeddings[k] for k in other_keys], dim=0) #use all the other embeddings as the negative examples
    return sampled_embeddings


def main(config, k=5, n_negative=3000):
    
    desc_path = f'{config.args.data_path}/{config.args.dataset}/step1.1_desc_gen_{config.args.dataset}.json'
    raw_train_path = f'{config.args.data_path}/{config.args.dataset}/{config.args.dataset}-train.json'
    concepts_file = f'{config.args.data_path}/{config.args.dataset}/{config.args.dataset}_visual_features.json'
    output_file = f'{config.args.data_path}/{config.args.dataset}/step1.2_concepts_{config.args.dataset}.json'
    
    start_time = time.time()
    # Load data
    desc = load_json(desc_path)
    raw_train = load_json(raw_train_path)
    concepts_list = prepare_concpets(concepts_file)
    
    # Load Contriever model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.args.retriever_path)
    model = AutoModel.from_pretrained(config.args.retriever_path, device_map=config.args.device)
    
    model_load_time = time.time()
    print(f"Model loading took {model_load_time - start_time:.2f} seconds")

    # Precompute all embeddings
    all_texts = {**desc, **concepts_list}
    all_embeddings = precompute_embeddings(all_texts, tokenizer, model)

    new_train = []
    
    # Process each data point as Set A
    #for key in tqdm(desc.keys(), desc="Processing data points"):
    for key in desc.keys():
        iteration_start = time.time()
        
        fine_grained_label = raw_train[int(key)]["label"]
        # Retrieve precomputed embeddings and sample negative examples
        A_embeddings = all_embeddings[key]
        B_embeddings = all_embeddings[fine_grained_label] # directly index it 
        B_texts = concepts_list[fine_grained_label]
        # B_embeddings = get_embeddings(B_texts, tokenizer, model)  # B_texts might change, so compute on-the-fly
        C_embeddings = sample_negative_embeddings(all_embeddings, key, desc, n_negative)
        
        # Compute InfoNCE scores


        info_nce_scores = compute_info_nce_scores(A_embeddings=A_embeddings, B_embeddings=B_embeddings, C_embeddings=C_embeddings)
        
        mean_MI_values = torch.mean(info_nce_scores).item()
        std_MI_values = torch.std(info_nce_scores).item()
        threshold = mean_MI_values + 0.4*std_MI_values

        top_texts = []
        for concept, nce_loss in zip(B_texts, info_nce_scores.tolist()):
            if nce_loss > threshold:
                top_texts.append((concept, nce_loss))

        if len(top_texts) < 3:
            sorted_scores = sorted(zip(B_texts, info_nce_scores.tolist()), key=lambda x: x[1], reverse=True)
            for i in range(3):
                if sorted_scores[i] not in top_texts:
                    top_texts.append(sorted_scores[i])

        # 按 nce_loss 从高到低排序，并提取 concept
        sorted_list = [concept[0] for concept in sorted(top_texts, key=lambda x: x[1], reverse=True)]

        raw_train[int(key)]['concepts'] = sorted_list
        new_train.append(raw_train[int(key)])

    
    # Save all results
    save_results(new_train, output_file)

if __name__ == "__main__":
    config = Config()
    print(config.args)
    k = None
    n_negative = None  # Number of negative examples to sample
    main(config, k, n_negative)