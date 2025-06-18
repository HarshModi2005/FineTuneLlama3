"""Configuration management for Dan Brown fine-tuning project."""

import yaml
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class ModelConfig:
    name: str
    max_seq_length: int
    load_in_4bit: bool
    dtype: str = None


@dataclass 
class LoRAConfig:
    r: int
    alpha: int
    dropout: float
    bias: str
    target_modules: list
    use_rslora: bool = False


@dataclass
class TrainingConfig:
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    warmup_steps: int
    max_steps: int
    learning_rate: float
    weight_decay: float
    lr_scheduler_type: str
    optim: str
    logging_steps: int
    seed: int
    use_gradient_checkpointing: str


@dataclass
class DataConfig:
    train_file: str
    validation_split: float


@dataclass
class GenerationConfig:
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    do_sample: bool


@dataclass
class OutputConfig:
    model_save_path: str
    checkpoint_dir: str
    log_dir: str
    examples_dir: str


class Config:
    """Main configuration class."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        self.model = ModelConfig(**config_dict['model'])
        self.lora = LoRAConfig(**config_dict['lora'])
        self.training = TrainingConfig(**config_dict['training'])
        self.data = DataConfig(**config_dict['data'])
        self.generation = GenerationConfig(**config_dict['generation'])
        self.output = OutputConfig(**config_dict['output'])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'model': self.model.__dict__,
            'lora': self.lora.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'generation': self.generation.__dict__,
            'output': self.output.__dict__
        } 