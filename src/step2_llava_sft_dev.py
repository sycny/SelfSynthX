# modified based on https://github.com/yuanzhoulvpi2017/zero_nlp/tree/main/train_llava
# used for development(especially for LLaVA-Next Series)

# import debugpy

# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import copy
import logging
import os
import json
import random
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from PIL import Image, ImageDraw
from typing import Dict, List, Tuple, Optional, Sequence

import torch
import transformers
from torchvision import transforms
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
# from trl import (
#     ModelConfig,
#     SFTConfig,
#     SFTTrainer,
#     get_peft_config,
#     get_quantization_config,
#     get_kbit_device_map,
# )
# from args import SFTScriptArguments
from preprocess import load_model_processor

logger = logging.getLogger(__name__)

class DataLoader:
    @staticmethod
    def load_data(data_src):
        with open(data_src, "r") as f:
            return json.load(f)

@dataclass
class QaImageOutput:
    q_input_ids: torch.Tensor
    pixel_values: torch.Tensor
    a_input_ids: torch.Tensor

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="/home/local/PARTNERS/ys670/camca/ys670/Huggingface/llama3-llava-next-8b-hf",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_4bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit."},
    )
    freeze_vision: bool = field(default=True)
    freeze_llm: bool = field(default=True)
    lora_enable: Optional[bool] = field(default=False, metadata={"help": "whether using lora fine-tuning model."})
    lora_r: int = field(default=32, metadata={"help": "Lora attention dimension"})
    lora_alpha: int = field(default=16, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "Lora dropout"})


@dataclass
class DataArguments:
    data_path: str = field(default='/home/local/PARTNERS/ys670/camca/ys670/datasets/finer', metadata={"help": "Path to the training folders."})
    dataset_name: str = field(default='cub-200/step1.3_QA_gen.json_cub-200.json', metadata={"help": "Path to the training ocr json files."}) 
    image_path: str = field(default=None, metadata={"help": "Path to the training ocr image folders."})
    image_folder: Optional[str] = field(default="")
    first_dataset_num_proc: int = 32

     
def add_backdoor_trigger(image: Image.Image, trigger_size: int = 15, position: tuple = (0, 0), color: tuple = (255, 0, 0)):
    draw = ImageDraw.Draw(image)
    x, y = position
    draw.rectangle([x, y, x + trigger_size, y + trigger_size], fill=color)
    return image

def resize_image(image: Image.Image, size: tuple = (336, 336)) -> Image.Image:
    return image.resize(size, Image.Resampling.LANCZOS)

def auging(raw_image):

    augmentations = [
        (transforms.RandomHorizontalFlip(p=1.0), 0.5),   
        (transforms.RandomVerticalFlip(p=1.0), 0.3),     
        (transforms.RandomRotation(degrees=30), 0.7),    
        (transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), 0.7),  
        (transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2)), 0.5),  
        (transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)), 0.4),  
        (transforms.RandomPerspective(distortion_scale=0.5, p=1.0), 0.6)  
    ]
    
    num_augs_to_apply = random.randint(1, 4)
    chosen_augs = random.sample(augmentations, num_augs_to_apply)
    
    aug_image = raw_image
    for aug, prob in chosen_augs:
        if random.random() < prob: 
            aug_image = aug(aug_image)
    
    return aug_image
   
    
class LlavaDataset(Dataset):
    def __init__(self, data_path: str, dataset_path: str) -> None:
        super().__init__()

        self.chat_data, self.image_dir = self.build_dataset(data_dir=data_path, dataset_name=dataset_path)

    def build_dataset(self, data_dir: str, dataset_name: str) -> Tuple[List[Dict], Path]:
        
        data_dir = Path(data_dir)
        file_path = data_dir.joinpath(dataset_name)
        image_dir = data_dir
        raw_data =  DataLoader.load_data(file_path)
        
        new_data = []
        
        for line in raw_data:
            for qa in line['new_QA']:
                new_line = dict()
                new_line['id'] = line['id']
                new_line["image"] = line['img_path']
                new_line['conversations'] = []
                new_line['conversations'].append({'from':'human', 'value':qa["Q"]})
                new_line['conversations'].append({'from':'gpt','value':qa["A"]})
                
                new_data.append(new_line)
        
        return new_data, image_dir

    def __len__(self):
        return len(self.chat_data)

    def __getitem__(self, index) -> Tuple[str, str, Path]:
        
        cur_data = self.chat_data[index]
        
        conversations = cur_data.get("conversations")

        human_input = conversations[0].get("value")
        chatbot_output = conversations[1].get("value")

        image_path = self.image_dir.joinpath(cur_data.get("image"))
        return human_input, chatbot_output, image_path


class TrainLLavaModelCollator:
    def __init__(self, processor: AutoProcessor, IGNORE_INDEX: int) -> None:
        self.processor = processor
        self.ignore_index = IGNORE_INDEX

    def __call__(self, examples: List) -> Dict[str, torch.Tensor]:
        # Get the texts and images, and apply the chat template
        texts = [] # [self.processor.apply_chat_template(example[0], tokenize=False) for example in examples]
        images = [] #[example["images"][0] for example in examples]
        for example in examples:
            
            conversation = [
                    {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": example[1]},
                        {"type": "image"},
                        ],
                    },
                    {
                    "role": "assistant" ,
                    "content": [
                    {"type": "text", "text": example[0]},
                    ],
                    }
            ]
            texts.append(self.processor.apply_chat_template(conversation))
            
            raw_image = Image.open(example[2])
            aug_image = auging(raw_image)
            
            images.append(aug_image)
            
        # Tokenize the texts and process the images
        batch = self.processor(texts, images, return_tensors="pt", padding=True)

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = self.ignore_index
        batch["labels"] = labels

        return batch

def load_dataset_collator(processor, dataargs: DataArguments):

    llava_dataset = LlavaDataset(
        data_path=dataargs.data_path, 
        dataset_path = dataargs.dataset_name
    )
    data_collator = TrainLLavaModelCollator(processor, -100)

    return llava_dataset, data_collator

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    model, processor, peft_config = load_model_processor(model_args=model_args, script_args= training_args)
    #model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        
    train_dataset, data_collator = load_dataset_collator(processor, data_args)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )

    trainer.train()

    # if training_args.resume_from_checkpoint is not None:
    #     checkpoint = training_args.resume_from_checkpoint
    # trainer.train(resume_from_checkpoint=checkpoint) #not working throw Segmentation fault (core dumped) fault.
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    train()