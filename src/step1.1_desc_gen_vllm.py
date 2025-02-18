import re
import json
import os
import argparse
from PIL import Image
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from intstructions import desc_instructions

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, default="cub-200")
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--model_path", type=str, default='./llava-1.5-7b-hf')
        parser.add_argument("--data_path", type=str, default="./datasets/finer")
        parser.add_argument("--output_file", type=str, default="step1.1_desc_gen")
        parser.add_argument("--tensor_parallel_size", type=int, default=1)
        parser.add_argument("--repeat_times", type=int, default=10)
        parser.add_argument("--gpu_use", type=float, default=0.8)
        parser.add_argument("--temperature", type=float, default=1)
        parser.add_argument("--max_tokens", type=int, default=256)
        
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
            n=num_samples,
            )
        
        outputs = self.llm.generate(inputs, sampling_params=sampling_params, use_tqdm=False)
        
        return [[output.text for output in item.outputs] for item in outputs]
        #return [[output.text + path for output in item.outputs] for item, path in zip(outputs,image_paths)]

class DataLoader:
    @staticmethod
    def load_data(data_src):
        with open(data_src, "r") as f:
            return json.load(f)

class DescriptionGenerator:
    def __init__(self, model, config, instructions):
        self.model = model
        self.config = config
        self.data_path = config.args.data_path
        self.dataset = config.args.dataset
        self.instructions = instructions
        
    def generate(self, batch_size, output_file):
        data_src = f"{self.data_path}/{self.dataset}/{self.dataset}-train.json"
        test_data = DataLoader.load_data(data_src)
        
        total = len(test_data)
        results = {}

        for i in tqdm(range(0, total, batch_size)):
            batch = test_data[i:i+batch_size]
            image_paths = [os.path.join(self.data_path, item['img_path']) for item in batch]
            
            all_descriptions = self.model.batch_run(image_paths * len(self.instructions), self.instructions * len(image_paths),  self.config.args.repeat_times)
            #print("all_descriptions", all_descriptions)
            for j, item in enumerate(batch):
                image_id = item['id']
                results[image_id] = []
                for k in range(len(self.instructions)):  # For each instruction
                    results[image_id].extend(all_descriptions[j + k * len(image_paths)])

        output_file = f"{self.data_path}/{self.dataset}/{output_file}_{self.dataset}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Generated descriptions saved to {output_file}")

def main():
    config = Config()
    print(config.args)
    model = LLaVAModel(config)
    selected_instructions = desc_instructions.get(config.args.dataset)
    generator = DescriptionGenerator(model, config, selected_instructions)
    generator.generate(config.args.batch_size, config.args.output_file)

if __name__ == "__main__":
    main()