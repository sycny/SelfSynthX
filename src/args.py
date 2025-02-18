from dataclasses import dataclass, field
from typing import Optional, List, Literal, Dict, Any
from enum import Enum
from transformers import TrainingArguments



class FDivergenceType(Enum):
    REVERSE_KL = "reverse_kl"
    JS_DIVERGENCE = "js_divergence"
    ALPHA_DIVERGENCE = "alpha_divergence"

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="/data/Huggingface/llava-1.5-7b-hf",
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
    #lora_target_modules: List[str] = field(default='all-linear', metadata={"help": "List of module names to apply Lora to"})
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            'gate_proj',
            'up_proj',
            'down_proj'
        ],
        metadata={"help": "List of module names to apply Lora to"}
    )
    
@dataclass
class DataArguments:
    data_path: str = field(default='./datasets/finer', metadata={"help": "Path to the training ocr json files."})
    dataset_name: str = field(default='cub-200', metadata={"help": "Path to the training ocr json files."}) 
    image_path: str = field(default=None, metadata={"help": "Path to the training ocr image folders."})
    image_folder: Optional[str] = field(default="./datasets/finer")
    first_dataset_num_proc: int = 32

@dataclass
class DPOScriptArguments(TrainingArguments):
    
    beta: float = 0.01
    label_smoothing: float = 0
    loss_type: Literal[
        "sigmoid", "hinge", "ipo", "bco_pair", "sppo_hard", "nca_pair", "robust", "aot", "aot_pair", "exo_pair"
    ] = "sigmoid"
    label_pad_token_id: int = -100
    padding_value: Optional[int] = None
    truncation_mode: str = "keep_end"
    max_length: Optional[int] = 1024
    max_prompt_length: Optional[int] = 1024
    max_target_length: Optional[int] = 1024
    is_encoder_decoder: Optional[bool] = False
    disable_dropout: bool = True
    generate_during_eval: bool = False
    precompute_ref_log_probs: bool = False
    dataset_num_proc: Optional[int] = 1
    model_init_kwargs: Optional[Dict] = None
    ref_model_init_kwargs: Optional[Dict] = None
    model_adapter_name: Optional[str] = None
    ref_adapter_name: Optional[str] = None
    reference_free: bool = False
    force_use_ref_model: bool = False
    f_divergence_type: Optional[FDivergenceType] = FDivergenceType.REVERSE_KL
    f_alpha_divergence_coef: Optional[float] = 1.0
    sync_ref_model: bool = False
    ref_model_mixup_alpha: float = 0.9
    ref_model_sync_steps: int = 64
    rpo_alpha: Optional[float] = None
    remove_unused_columns: bool = field(default=False)
    max_completion_length: Optional[int] = None
     
    
    
    dataset_cache_path: str = './dataset_cache'
    per_device_train_batch_size: int = field(default=16, metadata={"help": "Batch size per GPU for training"})
    per_device_eval_batch_size: int = field(default=1, metadata={"help": "Batch size per GPU for evaluation"})
    num_train_epochs: int = field(default=2, metadata={"help": "Total number of training epochs"})
    #max_steps: int = field(default=2000, metadata={"help": "Total number of training steps"})
    logging_steps: int = field(default=1, metadata={"help": "Number of steps between logging"})
    #save_steps: int = field(default=500, metadata={"help": "Number of steps between model saves"})
    save_strategy: str = field(default="epoch", metadata={"help": "Save models"})
    gradient_accumulation_steps: int = field(default=4, metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass"})
    gradient_checkpointing: bool = field(default=False, metadata={"help": "Enable gradient checkpointing"})
    learning_rate: float = field(default=0.5, metadata={"help": "Initial learning rate (after warmup period)"})
    eval_steps: int = field(default=50, metadata={"help": "Number of steps between evaluations"})
    output_dir: str = field(default="/data/ys07245/checkpoints/llava_loraft_dpo_cub-200_projector/", metadata={"help": "Output directory for checkpoints and logs"})
    report_to: str = field(default="none", metadata={"help": "Report results to this platform"})
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "The scheduler type to use"})
    warmup_steps: int = field(default=0, metadata={"help": "Number of steps for the warmup phase"})
    optimizer_type: str = field(default="adamw_torch", metadata={"help": "The optimizer to use"})
    gradient_checkpointing_use_reentrant: bool = field(default=True, metadata={"help": "Use reentrant version of gradient checkpointing"})
    seed: int = field(default=42, metadata={"help": "Random seed for initialization"})
    bf16: bool = field(default=True, metadata={"help": "Use bfloat16 precision"})
    run_name: str = field(default="dpo_llava", metadata={"help": "Name of the run"})
    batch_eval_metrics: bool = field(default=False)
    full_determinism: bool = field(default=False)
    #$accelerator_config = None

@dataclass
class SFTScriptArguments(TrainingArguments):
    # dataset_name: str = field(
    #     default="timdettmers/openassistant-guanaco",
    #     metadata={"help": "the dataset name"},
    # )
    dataset_train_split: str = field(default="train", metadata={"help": "The dataset split to train on"})
    dataset_test_split: str = field(default="test", metadata={"help": "The dataset split to evaluate on"})
    config: str = field(default=None, metadata={"help": "Path to the optional config file"})
    gradient_checkpointing_use_reentrant: bool = field(
        default=False,
        metadata={"help": "Whether to apply `use_reentrant` for gradient_checkpointing"},
    )
    dataset_text_field: Optional[str] = ""
    packing: bool = False
    max_seq_length: Optional[int] = None
    dataset_num_proc: Optional[int] = None
    dataset_batch_size: int = 1000
    neftune_noise_alpha: Optional[float] = None
    model_init_kwargs: Optional[Dict[str, Any]] = None
    dataset_kwargs: Optional[Dict[str, Any]] = None
    eval_packing: Optional[bool] = None
    num_of_sequences: int = 1024
    chars_per_token: float = 3.6
    use_liger: bool = False

