# modified based on https://github.com/yuanzhoulvpi2017/zero_nlp/tree/main/train_llava


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
        default="./llama3-llava-next-8b-hf",
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
    data_path: str = field(default='./datasets/finer', metadata={"help": "Path to the training folders."})
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

def build_qaimage(processor: AutoProcessor, q_text: str, a_text: str, image_path: Path, ):
    
    conversation = [
                        {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": q_text},
                            {"type": "image"},
                            ],
                        },
                    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    image_file = image_path
    raw_image = Image.open(image_file).convert('RGB') 
    
    aug_image = auging(raw_image) # this line applies the augmentation
    
    #raw_image = resize_image(raw_image, size=(336, 336))
    
    inputs = processor(prompt, aug_image, return_tensors="pt")  # .to(0, torch.float16)

    a_text = a_text.strip('\n')   # in some cases, the beginning of the answer with \n
    
    a_input_ids = processor.tokenizer(
        a_text,
        return_tensors="pt",
        padding="longest",
        truncation=True,
    )["input_ids"]

    res = QaImageOutput(
        q_input_ids=inputs.get("input_ids"),
        pixel_values=inputs.get("pixel_values"),
        a_input_ids=a_input_ids,
    )
    return res    
    
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

    
    
        # to stablize the training, we can add some general data to the training set, 
        # the following data is from LLaVA-Instruct-150K/complex_reasoning_77k.json, which is the original finetune data for LLaVA  
        # You can also discard this part, if you wish to be faster and do not care about the general capability of the model.
        geneal_data_path = './LLaVA-Instruct-150K/complex_reasoning_77k.json'
        general_data =  DataLoader.load_data(geneal_data_path)
        total_case_num = len(new_data)
        
        # Filter general data to only include existing images
        filtered_general_data = []
        for item in general_data:
            # LLaVA-Instruct-150K use the COCO dataset, so please download the COCO dataset
            image_path = os.path.join(image_dir, 'train2014/COCO_train2014_'+item['image'])
            #print("image_path", image_path)
            if os.path.exists(image_path):
                item['image'] = 'train2014/COCO_train2014_' + item['image']  # Update image path
                item['conversations'][0]['value'] = item['conversations'][0]['value'].replace('<image>', '').replace('\n', '')
                filtered_general_data.append(item)
        print("filtered_general_data", filtered_general_data[0])
        # Sample from filtered data
        sampled_general_data = random.sample(filtered_general_data, min(total_case_num, len(filtered_general_data)))
        print(f"sampled {len(sampled_general_data)} valid images out of {len(general_data)} total images")
        print("domain data", len(new_data))
        
        new_data.extend(sampled_general_data)
        
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
        self.ingnore_index = IGNORE_INDEX

    def convert_one_piece(
        self,
        q_input_ids: torch.Tensor,
        a_input_ids: torch.Tensor,
    ):
        input_ids = torch.concat(
            [
                q_input_ids,
                a_input_ids,
                torch.tensor(self.processor.tokenizer.eos_token_id).reshape(1, -1),
            ],
            axis=1,
        )
        labels = torch.concat(
            [
                torch.full(q_input_ids.shape, self.ingnore_index),
                a_input_ids,
                torch.tensor(self.processor.tokenizer.eos_token_id).reshape(1, -1),
            ],
            axis=1,
        )

        return input_ids, labels

    def __call__(self, features: List) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        pixel_values = []
        max_input_len_list = []

        for feature in features:
            qaimage_output = build_qaimage(
                self.processor, feature[0], feature[1], feature[2]
            )
            temp_input_ids, temp_labels = self.convert_one_piece(
                qaimage_output.q_input_ids, qaimage_output.a_input_ids
            )
            max_input_len_list.append(temp_input_ids.shape[1])
            input_ids_list.append(temp_input_ids)
            labels_list.append(temp_labels)
            pixel_values.append(qaimage_output.pixel_values)

        max_input_len = max(max_input_len_list)

        final_input_ids = torch.concat(
            [
                torch.concat(
                    [
                        torch.full(
                            (1, max_input_len - max_input_len_list[index]),
                            self.processor.tokenizer.pad_token_id,
                        ),
                        value,
                    ],
                    axis=1,
                )
                for index, value in enumerate(input_ids_list)
            ]
        )
        final_labels = torch.concat(
            [
                torch.concat(
                    [
                        torch.full(
                            (1, max_input_len - max_input_len_list[index]),
                            self.ingnore_index,
                        ),
                        value,
                    ],
                    axis=1,
                )
                for index, value in enumerate(labels_list)
            ]
        )
        final_pixel_values = torch.concat(pixel_values, axis=0)
        attention_mask = torch.ones_like(final_input_ids)
        attention_mask[final_input_ids == self.processor.tokenizer.pad_token_id] = 0
        return {
            "input_ids": final_input_ids,
            "labels": final_labels,
            "pixel_values": final_pixel_values,
            "attention_mask": attention_mask,
        }

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
        
    train_dataset, data_collator = load_dataset_collator(processor, data_args)
    print("We are using this file for tuning: ",data_args.dataset_name)
    # print(train_dataset[0])
    # print("Here", data_collator([train_dataset[100]]))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )

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