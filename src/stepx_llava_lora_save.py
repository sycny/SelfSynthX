import os
import torch
import argparse
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel, PeftConfig

class Config:
    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser(description="LLaVA model with LoRA support")
        parser.add_argument("--dataset", type=str, default="cub-200", help="Dataset name")
        parser.add_argument("--model_path", type=str, default='./llava-1.5-7b-hf', help="Base model path")
        parser.add_argument("--output_path", type=str, default='./checkpoint/cub-200-2024-09-10', help="LoRA checkpoint path")
        parser.add_argument("--output_file", type=str, default="merged_model", help="Output model name")
        return parser.parse_args()
    
    
def get_latest_checkpoint(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    
    if not checkpoints:
        return None
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))
    latest_checkpoint = checkpoints[-1]
    
    return os.path.join(output_dir, latest_checkpoint)

class LLaVAModel:
    def __init__(self, model_path, lora_path, output_path, dataset, output_file):
        self.model_path = model_path
        self.lora_path = lora_path
        self.output_path = output_path
        self.dataset = dataset
        self.output_file = output_file

    def load_and_merge_model(self):
        if self.lora_path:
            # load base model
            base_model = AutoModelForVision2Seq.from_pretrained(
                self.model_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16
            )
            Processor = AutoProcessor.from_pretrained(self.model_path)
            peft_config = PeftConfig.from_pretrained(self.lora_path)
            model = PeftModel.from_pretrained(base_model, self.lora_path)

            merged_model = model.merge_and_unload()

            temp_model_path = os.path.join(self.output_path, self.output_file)
            merged_model.save_pretrained(temp_model_path)
            Processor.save_pretrained(temp_model_path)

            print(f"Model merged successfully and saved to {temp_model_path}")
        else:
            print("No LoRA path provided, skipping model merging.")

def main():
    args = Config.get_args()
    
    lora_path = get_latest_checkpoint(args.output_path)
    print(f"Find lora_path: {lora_path}")
    llava_model = LLaVAModel(args.model_path, lora_path, args.output_path, args.dataset, args.output_file)
    llava_model.load_and_merge_model()

if __name__ == "__main__":
    main()