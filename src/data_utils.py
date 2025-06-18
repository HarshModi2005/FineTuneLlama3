"""Data utilities for Dan Brown fine-tuning project."""

import pickle
import pandas as pd
from datasets import Dataset
from typing import List, Dict, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data loading and preprocessing."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Input:
{}

### Response:
{}"""
        self.EOS_TOKEN = tokenizer.eos_token
    
    def load_pickle_data(self, file_path: str) -> Tuple[List[str], List[str]]:
        """Load inputs and outputs from pickle file."""
        logger.info(f"Loading data from {file_path}")
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            inputs = pickle.load(f)
            outputs = pickle.load(f)
        
        logger.info(f"Loaded {len(inputs)} training examples")
        return inputs, outputs
    
    def format_prompts(self, examples: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Format examples using Alpaca prompt template."""
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        
        for input_text, output_text in zip(inputs, outputs):
            # Add EOS token to prevent infinite generation
            text = self.alpaca_prompt.format(input_text, output_text) + self.EOS_TOKEN
            texts.append(text)
        
        return {"text": texts}
    
    def create_dataset(self, inputs: List[str], outputs: List[str], 
                      validation_split: float = 0.1) -> Tuple[Dataset, Dataset]:
        """Create train and validation datasets."""
        logger.info("Creating datasets...")
        
        # Create main dataset
        dataset = Dataset.from_dict({
            "input": inputs,
            "output": outputs
        })
        
        # Apply formatting
        dataset = dataset.map(self.format_prompts, batched=True)
        
        # Split into train/validation
        if validation_split > 0:
            split_dataset = dataset.train_test_split(test_size=validation_split, seed=42)
            train_dataset = split_dataset['train']
            val_dataset = split_dataset['test']
            logger.info(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")
        else:
            train_dataset = dataset
            val_dataset = None
            logger.info(f"Train size: {len(train_dataset)}")
        
        return train_dataset, val_dataset
    
    def save_examples(self, inputs: List[str], outputs: List[str], 
                     output_path: str, num_examples: int = 5):
        """Save example data for inspection."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        examples_df = pd.DataFrame({
            'input': inputs[:num_examples],
            'output': outputs[:num_examples]
        })
        
        examples_df.to_csv(output_path, index=False)
        logger.info(f"Saved {num_examples} examples to {output_path}")


def create_story_prompts() -> List[str]:
    """Create diverse story prompts for Dan Brown style generation."""
    prompts = [
        "Write a story about Robert Langdon discovering ancient symbols in Paris",
        "Create a thriller involving a professor solving a mystery in the Vatican",
        "Write about a symbologist uncovering secrets in ancient temples",
        "Describe Robert Langdon's adventure in a mysterious library",
        "Create a story about decoding cryptic messages in Renaissance art",
        "Write about a professor investigating conspiracy theories in Rome",
        "Develop a plot involving secret societies and hidden knowledge",
        "Create a thriller about archaeological discoveries with modern implications",
        "Write about a symbologist racing against time to prevent disaster",
        "Describe an adventure involving religious artifacts and ancient mysteries"
    ]
    return prompts


def analyze_dataset_statistics(inputs: List[str], outputs: List[str]) -> Dict[str, float]:
    """Analyze dataset statistics."""
    import numpy as np
    
    input_lengths = [len(text.split()) for text in inputs]
    output_lengths = [len(text.split()) for text in outputs]
    
    stats = {
        'num_examples': len(inputs),
        'avg_input_length': np.mean(input_lengths),
        'avg_output_length': np.mean(output_lengths),
        'max_input_length': np.max(input_lengths),
        'max_output_length': np.max(output_lengths),
        'min_input_length': np.min(input_lengths),
        'min_output_length': np.min(output_lengths)
    }
    
    return stats 