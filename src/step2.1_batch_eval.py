import re
import json
import torch
import os
import argparse
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor
from intstructions import explain_QA, simple_QA, CoT_QA


class Config:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, default="stanford_dogs")
        parser.add_argument("--batch_size", type=int, default=48)
        parser.add_argument("--model_path", type=str, default='./llava-1.5-7b-hf')
        parser.add_argument("--version", type=str, default='2024-09-11')
        parser.add_argument("--round", type=str, default='1')
        parser.add_argument("--data_path", type=str, default="./data/datasets/finer")
        self.args = parser.parse_args()

class LLaVAModel:
    def __init__(self, config):
        self.model = AutoModelForVision2Seq.from_pretrained(
            config.args.model_path, 
            torch_dtype=torch.float16, 
            device_map='cuda'
        )
        #self.model.load_adapter(config.args.lora_path)
        self.processor = AutoProcessor.from_pretrained(config.args.model_path)
        self.processor.tokenizer.padding_side = "left"  
        print(f"Model loaded successfully. Model_path:{config.args.model_path}.")# Lora_path:{config.args.lora_path}")
        
        
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

    def batch_run(self, image_paths, questions):
        images = [Image.open(path) for path in image_paths]
        questions_templated = [self.apply_template(instruction=question) for question in questions]
        inputs = self.processor(questions_templated, images, return_tensors='pt', padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=256, do_sample= False)# do_sample=True, temperature=1)
        
        outputs = self.processor.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        results = []
        for query, output in zip(questions, outputs):
            results.append({'Q':query, 'A':output})
            
        return results

class DataLoader:
    @staticmethod
    def load_data(data_src):
        with open(data_src, "r") as f:
            return json.load(f)

    @staticmethod
    def get_label_stats(data):
        fine_labels = set()
        for item in data:
            fine_labels.update(item["label"])
        return len(fine_labels)

class Evaluator:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.data_path = config.args.data_path
        self.dataset = config.args.dataset

        self.explain_queries = explain_QA.get(config.args.dataset)
        self.simple_queries = simple_QA.get(config.args.dataset) 
        self.CoT_queries = CoT_QA.get(config.args.dataset)
        
        self.fine_question = self.explain_queries[0]#'What is the name of the dog? Give me your reason.'

    def evaluate(self, batch_size):
        data_src = f"{self.data_path}/{self.dataset}/{self.dataset}-test.json"
        test_data = DataLoader.load_data(data_src)
        
        fine_labels_count = DataLoader.get_label_stats(test_data)
        print(f"Total fine labels: {fine_labels_count}")

        coarse_cor, fine_cor = 0, 0
        total = len(test_data)
        all_pred = []

        for i in tqdm(range(0, total, batch_size)):
            batch = test_data[i:i+batch_size]
            image_paths = [os.path.join(self.data_path, item['img_path']) for item in batch]
            
            fine_preds = self.model.batch_run(image_paths, [self.fine_question] * len(batch))
            for j, (item, fine_pred) in enumerate(zip(batch, fine_preds)):
                count = i + j + 1
                cor = self.check_prediction(count, fine_pred['A'], item['label'], "fine")
                fine_cor += cor
                fine_pred['id'] = item['id']
                fine_pred['cor'] = cor
                all_pred.append(fine_pred)

            if (i + batch_size) % 1 == 0 or (i + batch_size) >= total:
                self.print_accuracy(coarse_cor, fine_cor, count)

        self.print_accuracy(coarse_cor, fine_cor, total, final=True)
        
        output_file = f"{self.data_path}/{self.dataset}/step2.1_batch_eval_{self.dataset}_{self.config.args.version}_{self.config.args.round}.json"
        with open(output_file, 'w') as f:
            json.dump(all_pred, f, indent=2)

        print(f"Generated descriptions saved to {output_file}")

    @staticmethod
    def check_prediction(count, pred, label, level):
        def simplify_text(text):
            return re.sub(r"[^a-zA-Z\- ]+", '', text).replace('-', ' ').lower()
        
        simple_pred = simplify_text(pred)
        simple_label = simplify_text(label)
        if simple_label in simple_pred:
            #print(f"case:({count}) is correct, {level}_pred:{pred}, {level}_label:{label}")
            return 1
        return 0

    @staticmethod
    def print_accuracy(coarse_cor, fine_cor, count, final=False):
        prefix = "Final " if final else ""
        print(f"{prefix} Fine_accuracy: {fine_cor}/{count}={fine_cor/count:.4f}")

def main():
    config = Config()
    print(config.args)
    model = LLaVAModel(config)
    evaluator = Evaluator(model, config)
    evaluator.evaluate(config.args.batch_size)

if __name__ == "__main__":
    main()