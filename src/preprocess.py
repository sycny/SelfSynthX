import os
import sys
import torch
import json
import logging
import datasets
from accelerate import PartialState

from accelerate import Accelerator
from PIL import Image

from pathlib import Path
from torch.utils.data import Dataset
from transformers import HfArgumentParser, AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig

from args import DPOScriptArguments, ModelArguments, DataArguments



def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def load_model_processor(model_args, script_args
                        ):
    
    bnb_config = None
    if model_args.use_4bit_quantization:
        config = BitsAndBytesConfig(
            load_in_4bit=True, # quantize the model to 4-bits when you load it
            bnb_4bit_quant_type="nf4", # use a special 4-bit data type for weights initialized from a normal distribution
            bnb_4bit_use_double_quant=True, # nested quantization scheme to quantize the already quantized weights
            bnb_4bit_compute_dtype='int8', # use bfloat16 for faster computation,
            llm_int8_skip_modules=["vision_tower", "multi_modal_projector"]
        )
      
    model = AutoModelForVision2Seq.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        #device_map={"": Accelerator().local_process_index},
        low_cpu_mem_usage=True,
    )
    print("The model to be fine tuned is: ", model_args.model_name_or_path)
    model.enable_input_require_grads()
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, model_max_length=2048)
    processor.tokenizer.padding_side = "right"   # during training, one always uses padding on the right, not sure why
    config = None
    if model_args.lora_enable and 'False' in script_args.resume_from_checkpoint:
        logging.warning("Loading model to Lora")

        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules= find_all_linear_names(model), # model_args.lora_target_modules,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=["multi_modal_projector"],
        )
        model = get_peft_model(model, config)
    elif model_args.lora_enable and script_args.resume_from_checkpoint:
        
        from peft import PeftModel
        
        model = PeftModel.from_pretrained(model,
                                  script_args.resume_from_checkpoint,
                                  is_trainable=True,) # 👈 here,
    elif not model_args:
        logging.warning("Full Parameters Training without lora")
        pass
    
    if model_args.freeze_vision:
        logging.warning("Freeze vision_tower!")

        for param in model.vision_tower.parameters():
            param.requires_grad = False
            
    if model_args.freeze_llm:
        logging.warning("Freeze LLM!")

        for param in model.language_model.parameters():
            param.requires_grad = False
            
    for name, param in model.named_parameters():
        if param.requires_grad:
            pass
            #print(f"{name} is trainable")

    return model, processor, config


def load_ref_model(model_args):
    
    model = AutoModelForVision2Seq.from_pretrained(
    pretrained_model_name_or_path=model_args.model_name_or_path,
    torch_dtype=torch.bfloat16,
    #device_map={"": Accelerator().local_process_index},
    #low_cpu_mem_usage=True,
    )
    
    for param in model.parameters():
        param.requires_grad = False
    
    return model
    

if __name__ == "__main__":
    
    parser = HfArgumentParser(
        (DPOScriptArguments, ModelArguments, DataArguments))
    script_args, model_args, data_args = parser.parse_args_into_dataclasses()
    
    
    # Step 1: load dataset from folder, in this step, we only store strings. No image file, no template.
    torch_dataset = LlavaDataset(path='./data/cub-200-train.json', data_args=data_args)
    
    #we must have a generator for the following datasets building
    def gen():
        for idx in torch_dataset:
            yield idx

   
    hug_dataset = datasets.Dataset.from_generator(generator=gen)
    
    
    model, processor = load_model_processor(model_args=model_args, script_args= script_args)
    
    def process(row):
        
        row["prompt"] = 'USER: <image>\n' + row["prompt"][0]["content"]  + '\nASSISTANT:'
        row["chosen"] = row["chosen"][0]["content"]
        row["rejected"] = row["rejected"][0]["content"]
        row["image"] = Image.open(fp=row["image"])
        
        return row
    
    hug_dataset = hug_dataset.map(process, num_proc=data_args.dataset_num_proc)

    
    


        
        
    
    
    
    

        
        
        

        
        
        
        
