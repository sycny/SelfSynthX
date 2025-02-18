import re
import json
import random
import os
import argparse
from PIL import Image
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from intstructions import explain_QA, simple_QA, CoT_QA

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, default="cub-200")
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--model_path", type=str, default='./llava-1.5-7b-hf')
        parser.add_argument("--data_path", type=str, default="./data/datasets/finer")
        parser.add_argument("--round", type=str, default="1")
        parser.add_argument("--tensor_parallel_size", type=int, default=1)
        parser.add_argument("--repeat_times", type=int, default=3)
        parser.add_argument("--ins_numbers", type=int, default=8)
        parser.add_argument("--gpu_use", type=float, default=0.8)
        parser.add_argument("--temperature", type=float, default=0.7)
        parser.add_argument("--max_tokens", type=int, default=196)
        parser.add_argument("--top_p", type=float, default=0.6)
        
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

    def batch_run(self, image_paths, instructions, num_samples):
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
            top_p = self.config.args.top_p,
            n=num_samples,
            )
        
        outputs = self.llm.generate(inputs, sampling_params=sampling_params, use_tqdm=False)
        #print("outputs[0].outputs", outputs[0].outputs)
        #print("len(outputs)", len(outputs))
        
        final_re = []
        for item in outputs:
            output_texts = [output.text.strip() for output in item.outputs]
            final_re.append(output_texts)
        return final_re
        

class DataLoader:
    @staticmethod
    def load_data(data_src):
        with open(data_src, "r") as f:
            return json.load(f)

class DescriptionGenerator:
    def __init__(self, model, config, explain_queries, simple_queries, CoT_queries):
        self.model = model
        self.config = config
        self.data_path = config.args.data_path
        self.dataset = config.args.dataset
        self.explain_queries = explain_queries
        self.simple_queries = simple_queries
        self.CoT_queries = CoT_queries
        

        
    def build_query(self, item, ins_number):
        new_queries =[]
        
        for _ in range(ins_number):
            if random.random()<0.5:
                query = random.choice(self.explain_queries)
            else:
                query = random.choice(self.CoT_queries)
            new_queries.append(query)
                            
        return new_queries
    
    def generate(self, batch_size, round):
        data_src = f"{self.data_path}/{self.dataset}/{self.dataset}-train.json"
        test_data = DataLoader.load_data(data_src)
        
        total = len(test_data)
        results = {}
        queries = {}

        for i in tqdm(range(0, total, batch_size)):
            batch = test_data[i:i+batch_size]
            image_paths = [os.path.join(self.data_path, item['img_path']) for item in batch]
            
            # Build queries for each item and keep track of them
            batch_instructions = []
            batch_image_paths = []
            for image_path, item in zip(image_paths, batch):
                image_id = item['id']
                # Generate ins_numbers queries for the image
                item_queries = self.build_query(item, self.config.args.ins_numbers)
                # Repeat the image_path for each query
                batch_image_paths.extend([image_path] * len(item_queries))
                # Collect all queries
                batch_instructions.extend(item_queries)
                # Initialize results and queries dicts
                results.setdefault(image_id, [])
                queries.setdefault(image_id, [])
            
            # Now, batch_image_paths and batch_instructions are aligned
            # For each pair, generate repeat_times answers
            all_descriptions = self.model.batch_run(batch_image_paths, batch_instructions, self.config.args.repeat_times)
            # all_descriptions is a list where each element corresponds to an input pair and contains a list of repeat_times outputs

            # Now, we need to map the outputs back to the images and queries
            idx = 0
            for image_path, item in zip(image_paths, batch):
                image_id = item['id']
                for _ in range(self.config.args.ins_numbers):
                    # Get the outputs and query for this image and query
                    outputs = all_descriptions[idx]  # List of repeat_times answers
                    query = batch_instructions[idx]
                    # Append each answer and corresponding query to the results
                    for output in outputs:
                        results[image_id].append(output)
                        queries[image_id].append(query)
                    idx += 1

        # Save results and queries to files
        ans_output_file = f"{self.data_path}/{self.dataset}/round-{round}-step2.5_ans_{self.dataset}.json"
        with open(ans_output_file, 'w') as f:
            json.dump(results, f, indent=2)
        queries_output_file = f"{self.data_path}/{self.dataset}/round-{round}-step2.5_query_{self.dataset}.json"
        with open(queries_output_file, 'w') as f:
            json.dump(queries, f, indent=2)

        print(f"Generated descriptions saved to {ans_output_file} and {queries_output_file}")
    

def main():
    config = Config()
    print(config.args)
    model = LLaVAModel(config)
    explain_queries, simple_queries, CoT_queries = explain_QA.get(config.args.dataset), simple_QA.get(config.args.dataset), CoT_QA.get(config.args.dataset)
    generator = DescriptionGenerator(model, config, explain_queries, simple_queries, CoT_queries)
    generator.generate(config.args.batch_size, config.args.round)

if __name__ == "__main__":
    main()