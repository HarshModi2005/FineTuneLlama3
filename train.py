#!/usr/bin/env python3
"""
Dan Brown Style Story Generator - Training Script

This script trains a LLaMA-3 model to generate Dan Brown style stories.
"""

import logging
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config import Config
from model import DanBrownModelManager
from data_utils import DataProcessor, analyze_dataset_statistics
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

def setup_logging(level: str = "INFO"):
    """Setup logging with rich handler."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )

def main():
    parser = argparse.ArgumentParser(description="Train Dan Brown Style Story Generator")
    parser.add_argument("--config", type=str, default="config.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--data", type=str, help="Path to training data pickle file")
    parser.add_argument("--output", type=str, help="Output directory for saved model")
    parser.add_argument("--resume", type=str, help="Resume training from checkpoint")
    parser.add_argument("--log-level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Load data and setup model without training")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    console.print("[bold blue]Dan Brown Style Story Generator - Training[/bold blue]")
    console.print("=" * 60)
    
    try:
        # Load configuration
        console.print("Loading configuration...")
        config = Config(args.config)
        
        # Override config with command line arguments
        if args.data:
            config.data.train_file = args.data
        if args.output:
            config.output.model_save_path = args.output
        
        logger.info(f"Configuration loaded from {args.config}")
        
        # Initialize model manager
        console.print("Initializing model manager...")
        model_manager = DanBrownModelManager(config)
        
        # Load model
        console.print("Loading base model...")
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("Loading model...", total=None)
            model, tokenizer = model_manager.load_model()
            progress.update(task, description="Model loaded successfully")
        
        # Setup PEFT
        console.print("Setting up PEFT (LoRA)...")
        model_manager.setup_peft_model()
        
        # Load and process data
        console.print("Loading training data...")
        data_processor = DataProcessor(tokenizer)
        
        if not Path(config.data.train_file).exists():
            console.print(f"[red]Error: Data file not found: {config.data.train_file}[/red]")
            console.print("[yellow]Please provide a valid data file path using --data argument[/yellow]")
            return
        
        inputs, outputs = data_processor.load_pickle_data(config.data.train_file)
        
        # Analyze dataset
        stats = analyze_dataset_statistics(inputs, outputs)
        console.print("\n📈 [bold]Dataset Statistics:[/bold]")
        console.print(f"  • Examples: {stats['num_examples']}")
        console.print(f"  • Avg input length: {stats['avg_input_length']:.1f} words")
        console.print(f"  • Avg output length: {stats['avg_output_length']:.1f} words")
        console.print(f"  • Max input length: {stats['max_input_length']} words")
        console.print(f"  • Max output length: {stats['max_output_length']} words")
        
        # Create datasets
        train_dataset, val_dataset = data_processor.create_dataset(
            inputs, outputs, config.data.validation_split
        )
        
        # Save examples
        Path(config.output.examples_dir).mkdir(parents=True, exist_ok=True)
        data_processor.save_examples(
            inputs, outputs, 
            f"{config.output.examples_dir}/training_examples.csv"
        )
        
        if args.dry_run:
            console.print("[yellow]Dry run completed. Exiting without training.[/yellow]")
            return
        
        # Setup trainer
        console.print("Setting up trainer...")
        model_manager.setup_trainer(train_dataset, val_dataset)
        
        # Train model
        console.print("Starting training...")
        console.print(f"  • Max steps: {config.training.max_steps}")
        console.print(f"  • Learning rate: {config.training.learning_rate}")
        console.print(f"  • Batch size: {config.training.per_device_train_batch_size}")
        console.print(f"  • Gradient accumulation: {config.training.gradient_accumulation_steps}")
        
        trainer_stats = model_manager.train()
        
        # Save model
        console.print("Saving trained model...")
        model_manager.save_model()
        
        console.print("[bold green]Training completed successfully![/bold green]")
        console.print(f"Model saved to: {config.output.model_save_path}")
        
        # Generate sample story
        console.print("\n[bold]Generating sample story...[/bold]")
        sample_prompt = "Write a story about Robert Langdon's adventure in a mysterious ancient library"
        
        try:
            sample_story = model_manager.generate_story(
                sample_prompt, 
                max_new_tokens=500,
                temperature=0.8
            )
            
            console.print(f"\n[bold]Sample Story:[/bold]")
            console.print(f"[italic]Prompt: {sample_prompt}[/italic]")
            console.print("-" * 60)
            console.print(sample_story[:500] + "..." if len(sample_story) > 500 else sample_story)
            
            # Save sample
            sample_path = Path(config.output.examples_dir) / "sample_generation.txt"
            with open(sample_path, 'w') as f:
                f.write(f"Prompt: {sample_prompt}\n\n")
                f.write("Generated Story:\n")
                f.write(sample_story)
            
            console.print(f"\nSample saved to: {sample_path}")
            
        except Exception as e:
            logger.error(f"Error generating sample story: {e}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        console.print(f"[red]Training failed: {e}[/red]")
        raise

if __name__ == "__main__":
    main() 