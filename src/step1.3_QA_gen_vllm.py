import re
import json
import os
import random
import argparse
from PIL import Image
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from intstructions import explain_QA, simple_QA, CoT_QA

import difflib


def fuzzy_match(label, answer, threshold=0.7):
    ratio = difflib.SequenceMatcher(None, label, answer).ratio()
    return ratio >= threshold

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, default="cub-200")
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--model_path", type=str, default='./llava-1.5-7b-hf')
        parser.add_argument("--data_path", type=str, default="./datasets/finer")
        parser.add_argument("--output_file", type=str, default="round-1-QA_gen")
        parser.add_argument("--tensor_parallel_size", type=int, default=1)
        parser.add_argument("--repeat_times", type=int, default=8)
        parser.add_argument("--gpu_use", type=float, default=0.8)
        parser.add_argument("--temperature", type=float, default=0.3)
        parser.add_argument("--max_tokens", type=int, default=128)
        self.args = parser.parse_args()

class LLaVAModel:
    def __init__(self, config):
        self.config = config
        self.llm = LLM(
            model=config.args.model_path,
            tensor_parallel_size=self.config.args.tensor_parallel_size,
            gpu_memory_utilization=self.config.args.gpu_use)
        self.processor = AutoProcessor.from_pretrained(config.args.model_path)
        print("Model loaded successfully")
    
    def apply_template(self, instruction):
        conversation = [
                            {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": instruction},
                                {"type": "image"},
                                ],
                            },
                        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        return prompt

    def batch_run(self, image_paths, instructions):
        inputs = []
        for image_path, instruction in zip(image_paths, instructions):
            instruction_templated = self.apply_template(instruction=instruction)
            img = Image.open(image_path)
            inputs.append({
                "prompt": instruction_templated,           
                "multi_modal_data": {
                "image": img
            },
            })
        
        sampling_params = SamplingParams(
            temperature=self.config.args.temperature,
            max_tokens=self.config.args.max_tokens,
            )
        outputs = self.llm.generate(inputs, sampling_params=sampling_params, use_tqdm=False)
        
        results = []
        for query, output in zip(instructions, outputs):
            output = output.outputs[0].text
            input = query.split('\n')[-1].split(': ')[-1].replace("'", '')
            results.append({'Q':input, 'A':output})
            
        return results

class DataLoader:
    @staticmethod
    def load_data(data_src):
        with open(data_src, "r") as f:
                data =  json.load(f)
        return data


class DescriptionGenerator:
    def __init__(self, model, config, explain_queries, simple_queries, CoT_queries):
        self.model = model
        self.config = config
        self.data_path = config.args.data_path
        self.dataset = config.args.dataset
        self.explain_queries = explain_queries
        self.simple_queries = simple_queries
        self.CoT_queries = CoT_queries
        
        #self.instruction = instruction
        
    def build_query(self, item, repeat_times):
        
        concepts = item['concepts']
        new_queries =[]
        
        for _ in range(repeat_times):
            if random.random()<0.4:
                query = random.choice(self.simple_queries)
            elif random.random() > 0.6 and random.random()< 0.8:
                query = random.choice(self.explain_queries)
            else:
                query = random.choice(self.CoT_queries)
            label = item['label']
            concepts_str = ('').join(['- ' + concept + '\n' for concept in concepts]) #('\n').join(concepts)
            if  self.dataset in ['cub-200', 'stanford_dogs']:
                new_query = f"This is a picture of a {label} with the following visual features:\n\n{concepts_str}\n" +\
                "Based on the information provided, please answer the following question.\n" +\
                f"Question: '{query}'"
            elif self.dataset in ['HAM10000']:
                new_query = f"This is a dermatoscopic image of {label} disease with the following visual features:\n\n{concepts_str}\n" +\
                "Based on the information provided, please answer the following question.\n" +\
                f"Question: '{query}'"
            elif self.dataset in ['PLD', 'fgvc']:
                new_query = f"This is a picture of {label} with the following visual features:\n\n{concepts_str}\n" +\
                "Based on the information provided, please answer the following question.\n" +\
                f"Question: '{query}'"
            elif self.dataset in ['chest-xray']:
                new_query = f"This is a chest-xray of {label} with the following visual features:\n\n{concepts_str}\n" +\
                "Based on the information provided, please answer the following question.\n" +\
                f"Question: '{query}'"
            new_queries.append(new_query)
                            
        return new_queries
        

    def generate(self, batch_size, output_file):
        data_src = f"{self.data_path}/{self.dataset}/step1.2_concepts_{self.dataset}.json"
        raw_data = DataLoader.load_data(data_src)
        
        total = len(raw_data)
        all_results = []

        for i in tqdm(range(0, total, batch_size)):
            instructions =[]
            batch = raw_data[i:i+batch_size]
            image_paths = [os.path.join(self.data_path, item['img_path']) for item in batch]
            for item in batch:
                instructions.extend(self.build_query(item, self.config.args.repeat_times))
            batch_results = self.model.batch_run(image_paths * self.config.args.repeat_times, instructions)

            for j, item in enumerate(batch):
                item['new_QA'] = []  # Initialize new_QA list for each image
                
                # Process repeat_times number of QA pairs for the current image
                for k in range(self.config.args.repeat_times):  # For each instruction
                    # Calculate the correct index in batch_results
                    qa_pair = batch_results[j * self.config.args.repeat_times + k]  # Adjust indexing
                    
                    answer_included = False 
                    if item['label'].lower() in qa_pair['A'].lower():
                        item['new_QA'].append({
                            'Q': qa_pair['Q'],
                            'A': qa_pair['A'].strip()
                        })
                        answer_included = True
                    elif fuzzy_match(item['label'].lower(), qa_pair['A'].lower(), threshold=0.7):
                        item['new_QA'].append({
                            'Q': qa_pair['Q'],
                            'A': qa_pair['A'].strip() + f". More specifically, this image shows {item['label']}."
                        })
                        answer_included = True

                        if not answer_included:
                            item['new_QA'].append({
                                'Q': qa_pair['Q'],
                                'A': f"This is an image showing {item['label']}."
                            })
                    

            all_results.extend(batch)  
                
        output_file = f"{self.data_path}/{self.dataset}/{output_file}_{self.dataset}.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"Generated descriptions saved to {output_file}")

def main():
    config = Config()
    print(config.args)
    model = LLaVAModel(config)
    #model = None
    explain_queries, simple_queries, CoT_queries = explain_QA.get(config.args.dataset), simple_QA.get(config.args.dataset), CoT_QA.get(config.args.dataset)
    generator = DescriptionGenerator(model, config, explain_queries, simple_queries, CoT_queries)
    generator.generate(config.args.batch_size, config.args.output_file)

if __name__ == "__main__":
    main()