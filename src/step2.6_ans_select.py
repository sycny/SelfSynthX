import re
import json
import time
import random
import numpy as np
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from intstructions import explain_QA, simple_QA, CoT_QA

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str, default='llava')
        parser.add_argument("--dataset", type=str, default="cub-200")
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--retriever_path", type=str, default='./e5-large-v2')
        parser.add_argument("--data_path", type=str, default="./data/datasets/finer")
        parser.add_argument("--output_file", type=str, default="step2.6_ans.json")
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument("--round", type=str, default="2")
        parser.add_argument("--ins_numbers", type=int, default=8)
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

def simplify_text(text):
    return re.sub(r"[^a-zA-Z\- ]+", '', text).replace('-', ' ').lower()


def build_qa(info_nce_scores, B_texts, B_query_texts, answer, config, simple_queries, k=1):
    
    repeat_time = int(len(B_texts) / config.args.ins_numbers) # how many times the question repeated.
    
    new_QA = []
    
    for i in range(config.args.ins_numbers):
        
        score = info_nce_scores[i*repeat_time: i*repeat_time+repeat_time]
        output = B_texts[i*repeat_time: i*repeat_time+repeat_time]
        query = B_query_texts[i*repeat_time: i*repeat_time+repeat_time]
        
        if query[0] in simple_queries:
            
            for selected_answer in output:
                if simplify_text(answer) in simplify_text(selected_answer):
                    selected_answer = selected_answer.replace('<image>', '')
                    new_QA.append({
                        'Q': query[0],
                        'A': selected_answer.strip(),
                    })
                    break
        else:
            _, top_k_indices = torch.topk(score, 1, largest=True)
            top_1_ans = output[top_k_indices.item()]
            topk_1_query = query[top_k_indices.item()]
            
            if simplify_text(answer) in simplify_text(top_1_ans):
                    selected_answer = top_1_ans
                    selected_query = topk_1_query
            else:
                selected_answer = f'This is an image of {answer}.'
                selected_query = random.choice(simple_queries)
                
            selected_answer = selected_answer.replace('<image>', '')
            new_QA.append({
                'Q': selected_query,
                'A': selected_answer.strip(),
            })
              
    return new_QA
                 


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


def sample_negative_embeddings(all_embeddings, current_concept_list,  total_concept_list):
    other_concepts = [concept for concept in total_concept_list if k not in current_concept_list]
    sampled_embeddings = torch.cat([all_embeddings[concept] for concept in other_concepts], dim=0) #use all the other embeddings as the negative examples
    return sampled_embeddings


def main(config, k, simple_queries):
    
    ans_path = f'{config.args.data_path}/{config.args.dataset}/round-{config.args.round}-step2.5_ans_{config.args.dataset}.json' # all the train-synthized answer
    query_path = f'{config.args.data_path}/{config.args.dataset}/round-{config.args.round}-step2.5_query_{config.args.dataset}.json' # all the train-synthized answer
    selected_concepts_path = f'{config.args.data_path}/{config.args.dataset}/step1.2_concepts_{config.args.dataset}.json' # all the 
    concepts_path = f'{config.args.data_path}/{config.args.dataset}/{config.args.dataset}_visual_features.json' # all the labels and their concepts
    output_path = f'{config.args.data_path}/{config.args.dataset}/round-{config.args.round}-QA_gen_{config.args.dataset}.json' 
    
    
    start_time = time.time()
    # Load data
    ans_list = load_json(ans_path)
    query_list = load_json(query_path)
    selected_concepts = load_json(selected_concepts_path)
    concepts_list = prepare_concpets(concepts_path)
    
    all_concepts = []
    for item in concepts_list.values():
        all_concepts.extend(item) # all the cnocepts in a list
    
    all_concepts_dict = {}
    for concept in all_concepts:
        all_concepts_dict[concept] = concept
    
    # Load Contriever model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.args.retriever_path)
    model = AutoModel.from_pretrained(config.args.retriever_path, device_map=config.args.device)
    
    model_load_time = time.time()
    print(f"Model loading took {model_load_time - start_time:.2f} seconds")

    # Precompute all embeddings
    all_texts = {**ans_list, **concepts_list, **all_concepts_dict}
    all_embeddings = precompute_embeddings(all_texts, tokenizer, model)

    new_train = []
    
    # Process each data point as Set A
    #for key in tqdm(ans_list.keys(), desc="Processing data points"):
    for key in ans_list.keys():
        
        fine_grained_label = selected_concepts[int(key)]["label"]
        # Retrieve precomputed embeddings and sample negative examples
        A_texts = selected_concepts[int(key)]["concepts"] # this a list of accuare concepts for this image 
        A_embeddings = get_embeddings(A_texts, tokenizer, model)
        
        B_embeddings = all_embeddings[key]   # this is the ans embeddings.
        B_texts = ans_list[key] # this is a list of all the answers
        Query_B_texts = query_list[key]
        
        C_embeddings = sample_negative_embeddings(all_embeddings, current_concept_list=A_texts, total_concept_list=all_concepts)
        
        # Compute InfoNCE scores
        info_nce_scores = compute_info_nce_scores(A_embeddings=A_embeddings, B_embeddings=B_embeddings, C_embeddings=C_embeddings)
        #mean_MI_values = torch.mean(info_nce_scores).item()
        #print("info_nce_scores", info_nce_scores)
    
        new_QA = build_qa(info_nce_scores, B_texts, Query_B_texts, fine_grained_label, config, simple_queries, k)
        selected_concepts[int(key)]['new_QA'] = new_QA
        new_train.append(selected_concepts[int(key)])
    
    # Save all results
    save_results(new_train, output_path)

if __name__ == "__main__":
    config = Config()
    print(config.args)
    k = 1
    #n_negative = None  # Number of negative examples to sample
    explain_queries, simple_queries, CoT_queries = explain_QA.get(config.args.dataset), simple_QA.get(config.args.dataset), CoT_QA.get(config.args.dataset)
    main(config, config.args.ins_numbers, simple_queries)