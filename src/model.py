"""Model management for Dan Brown fine-tuning project."""

import torch
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from peft import PeftModel
from pathlib import Path
import logging
from typing import Optional, Dict, Any, List
import json

logger = logging.getLogger(__name__)


class DanBrownModelManager:
    """Manages model loading, training, and inference."""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def load_model(self):
        """Load the base model and tokenizer."""
        logger.info(f"Loading model: {self.config.model.name}")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model.name,
            max_seq_length=self.config.model.max_seq_length,
            dtype=self.config.model.dtype,
            load_in_4bit=self.config.model.load_in_4bit,
        )
        
        logger.info("Model loaded successfully")
        return self.model, self.tokenizer
    
    def setup_peft_model(self):
        """Setup PEFT (LoRA) configuration."""
        logger.info("Setting up PEFT model...")
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora.r,
            target_modules=self.config.lora.target_modules,
            lora_alpha=self.config.lora.alpha,
            lora_dropout=self.config.lora.dropout,
            bias=self.config.lora.bias,
            use_gradient_checkpointing=self.config.training.use_gradient_checkpointing,
            random_state=self.config.training.seed,
            use_rslora=self.config.lora.use_rslora,
        )
        
        logger.info("PEFT model setup complete")
    
    def setup_trainer(self, train_dataset, eval_dataset=None):
        """Setup the SFT trainer."""
        logger.info("Setting up trainer...")
        
        # Create output directories
        Path(self.config.output.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.output.log_dir).mkdir(parents=True, exist_ok=True)
        
        training_args = TrainingArguments(
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            warmup_steps=self.config.training.warmup_steps,
            max_steps=self.config.training.max_steps,
            learning_rate=self.config.training.learning_rate,
            fp16=not self._is_bfloat16_supported(),
            bf16=self._is_bfloat16_supported(),
            logging_steps=self.config.training.logging_steps,
            optim=self.config.training.optim,
            weight_decay=self.config.training.weight_decay,
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            seed=self.config.training.seed,
            output_dir=self.config.output.checkpoint_dir,
            logging_dir=self.config.output.log_dir,
            save_strategy="steps",
            save_steps=50,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=50 if eval_dataset else None,
            load_best_model_at_end=True if eval_dataset else False,
        )
        
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=self.config.model.max_seq_length,
            dataset_num_proc=2,
            packing=False,
            args=training_args,
        )
        
        logger.info("Trainer setup complete")
    
    def train(self):
        """Train the model."""
        if not self.trainer:
            raise ValueError("Trainer not setup. Call setup_trainer() first.")
        
        logger.info("Starting training...")
        
        # Print memory stats
        self._log_gpu_memory("Before training")
        
        # Train
        trainer_stats = self.trainer.train()
        
        # Print final stats
        self._log_gpu_memory("After training")
        self._log_training_stats(trainer_stats)
        
        return trainer_stats
    
    def save_model(self, save_path: Optional[str] = None):
        """Save the trained model."""
        if save_path is None:
            save_path = self.config.output.model_save_path
        
        logger.info(f"Saving model to {save_path}")
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save config
        config_save_path = Path(save_path) / "training_config.json"
        with open(config_save_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        logger.info("Model saved successfully")
    
    def load_trained_model(self, model_path: str):
        """Load a previously trained model."""
        logger.info(f"Loading trained model from {model_path}")
        
        # Load base model first
        self.load_model()
        
        # Load PEFT weights
        self.model = PeftModel.from_pretrained(self.model, model_path)
        
        logger.info("Trained model loaded successfully")
    
    def generate_story(self, prompt: str, **generation_kwargs) -> str:
        """Generate a story given a prompt."""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Setup generation parameters
        gen_config = {
            'max_new_tokens': self.config.generation.max_new_tokens,
            'temperature': self.config.generation.temperature,
            'top_p': self.config.generation.top_p,
            'top_k': self.config.generation.top_k,
            'repetition_penalty': self.config.generation.repetition_penalty,
            'do_sample': self.config.generation.do_sample,
            'use_cache': True,
        }
        gen_config.update(generation_kwargs)
        
        # Format prompt
        alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Input:
{}

### Response:
{}"""
        
        formatted_prompt = alpaca_prompt.format(prompt, "")
        
        # Enable inference mode
        FastLanguageModel.for_inference(self.model)
        
        # Tokenize
        inputs = self.tokenizer([formatted_prompt], return_tensors="pt").to("cuda")
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_config)
        
        # Decode
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Extract only the generated part
        response = response.split("### Response:")[-1].strip()
        
        return response
    
    def batch_generate(self, prompts: List[str], **generation_kwargs) -> List[str]:
        """Generate stories for multiple prompts."""
        stories = []
        for prompt in prompts:
            try:
                story = self.generate_story(prompt, **generation_kwargs)
                stories.append(story)
                logger.info(f"Generated story for prompt: {prompt[:50]}...")
            except Exception as e:
                logger.error(f"Error generating story for prompt '{prompt[:50]}...': {e}")
                stories.append(f"Error: {str(e)}")
        
        return stories
    
    def _is_bfloat16_supported(self) -> bool:
        """Check if bfloat16 is supported."""
        try:
            from unsloth import is_bfloat16_supported
            return is_bfloat16_supported()
        except:
            return torch.cuda.is_bf16_supported()
    
    def _log_gpu_memory(self, stage: str):
        """Log GPU memory usage."""
        if torch.cuda.is_available():
            gpu_stats = torch.cuda.get_device_properties(0)
            memory_used = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
            percentage = round(memory_used / max_memory * 100, 3)
            
            logger.info(f"{stage} - GPU: {gpu_stats.name}")
            logger.info(f"{stage} - Memory: {memory_used}/{max_memory} GB ({percentage}%)")
    
    def _log_training_stats(self, trainer_stats):
        """Log training statistics."""
        runtime = trainer_stats.metrics['train_runtime']
        logger.info(f"Training completed in {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
        logger.info(f"Training loss: {trainer_stats.metrics.get('train_loss', 'N/A')}")
        
        if 'eval_loss' in trainer_stats.metrics:
            logger.info(f"Validation loss: {trainer_stats.metrics['eval_loss']}") 