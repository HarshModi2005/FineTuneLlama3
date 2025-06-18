#!/usr/bin/env python3
"""
Dan Brown Style Story Generator - Inference Script

This script generates Dan Brown style stories using a trained model.
"""

import logging
import argparse
from pathlib import Path
import sys
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config import Config
from model import DanBrownModelManager
from data_utils import create_story_prompts
from rich.console import Console
from rich.logging import RichHandler
from rich.prompt import Prompt
from rich.panel import Panel
from rich.table import Table

console = Console()

def setup_logging(level: str = "INFO"):
    """Setup logging with rich handler."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )

def interactive_mode(model_manager):
    """Run interactive story generation."""
    console.print("\n[bold blue]Interactive Story Generation Mode[/bold blue]")
    console.print("Type 'quit' to exit, 'help' for commands")
    console.print("-" * 50)
    
    while True:
        try:
            user_input = Prompt.ask("\n[bold]Enter your story prompt")
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                console.print("Goodbye!")
                break
            elif user_input.lower() == 'help':
                show_help()
                continue
            elif user_input.lower() == 'examples':
                show_example_prompts()
                continue
            elif user_input.strip() == '':
                console.print("Please enter a valid prompt")
                continue
            
            # Generate story
            console.print("\nGenerating story...")
            story = model_manager.generate_story(user_input)
            
            # Display result
            console.print(Panel(
                f"[bold]Prompt:[/bold] {user_input}\n\n[bold]Generated Story:[/bold]\n{story}",
                title="Dan Brown Style Story",
                border_style="blue"
            ))
            
            # Ask to save
            save = Prompt.ask("Save this story? (y/n)", default="n")
            if save.lower() in ['y', 'yes']:
                save_story(user_input, story)
                
        except KeyboardInterrupt:
            console.print("\nGoodbye!")
            break
        except Exception as e:
            console.print(f"Error: {e}")

def show_help():
    """Display help information."""
    help_table = Table(title="Available Commands")
    help_table.add_column("Command", style="cyan")
    help_table.add_column("Description", style="white")
    
    help_table.add_row("quit/exit/q", "Exit the program")
    help_table.add_row("help", "Show this help message")
    help_table.add_row("examples", "Show example prompts")
    
    console.print(help_table)

def show_example_prompts():
    """Display example prompts."""
    prompts = create_story_prompts()
    
    console.print("\n[bold]Example Prompts:[/bold]")
    for i, prompt in enumerate(prompts[:5], 1):
        console.print(f"{i}. {prompt}")

def save_story(prompt, story):
    """Save generated story to file."""
    output_dir = Path("generated_stories")
    output_dir.mkdir(exist_ok=True)
    
    # Create filename from prompt
    filename = "story_" + prompt[:30].replace(" ", "_").replace("/", "_") + ".txt"
    filename = "".join(c for c in filename if c.isalnum() or c in "._-")
    
    output_path = output_dir / filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Prompt: {prompt}\n\n")
        f.write("Generated Story:\n")
        f.write(story)
    
    console.print(f"Story saved to: {output_path}")

def batch_generate(model_manager, prompts, output_dir, temperature=0.7):
    """Generate stories for multiple prompts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"Generating {len(prompts)} stories...")
    
    results = []
    
    for i, prompt in enumerate(prompts, 1):
        console.print(f"[{i}/{len(prompts)}] Generating: {prompt[:50]}...")
        
        try:
            story = model_manager.generate_story(prompt, temperature=temperature)
            
            # Save individual story
            filename = f"story_{i:03d}.txt"
            story_path = output_dir / filename
            
            with open(story_path, 'w', encoding='utf-8') as f:
                f.write(f"Prompt: {prompt}\n\n")
                f.write("Generated Story:\n")
                f.write(story)
            
            results.append({
                'prompt': prompt,
                'story': story,
                'filename': filename
            })
            
            console.print(f"Saved to: {story_path}")
            
        except Exception as e:
            console.print(f"Error generating story {i}: {e}")
            results.append({
                'prompt': prompt,
                'story': f"Error: {e}",
                'filename': None
            })
    
    # Save summary
    summary_path = output_dir / "generation_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    console.print(f"\nBatch generation completed!")
    console.print(f"Results saved to: {output_dir}")
    console.print(f"Summary saved to: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate Dan Brown Style Stories")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model directory")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--prompt", type=str,
                       help="Single prompt to generate story for")
    parser.add_argument("--prompts-file", type=str,
                       help="File containing prompts (one per line)")
    parser.add_argument("--output", type=str, default="generated_stories",
                       help="Output directory for generated stories")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Generation temperature (0.1-1.0)")
    parser.add_argument("--max-tokens", type=int, default=1000,
                       help="Maximum tokens to generate")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--examples", action="store_true",
                       help="Generate stories for example prompts")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    console.print("[bold blue]Dan Brown Style Story Generator - Inference[/bold blue]")
    console.print("=" * 60)
    
    try:
        # Load configuration
        config = Config(args.config)
        
        # Override generation settings
        config.generation.temperature = args.temperature
        config.generation.max_new_tokens = args.max_tokens
        
        # Initialize model manager
        console.print("Initializing model manager...")
        model_manager = DanBrownModelManager(config)
        
        # Load trained model
        console.print(f"Loading trained model from {args.model}...")
        model_manager.load_trained_model(args.model)
        
        console.print("Model loaded successfully!")
        
        # Different modes of operation
        if args.interactive:
            interactive_mode(model_manager)
            
        elif args.prompt:
            # Single prompt generation
            console.print(f"Generating story for: {args.prompt}")
            story = model_manager.generate_story(args.prompt)
            
            console.print(Panel(
                f"[bold]Prompt:[/bold] {args.prompt}\n\n[bold]Generated Story:[/bold]\n{story}",
                title="Generated Story",
                border_style="green"
            ))
            
            # Save story
            save_story(args.prompt, story)
            
        elif args.prompts_file:
            # Batch generation from file
            console.print(f"Loading prompts from {args.prompts_file}...")
            
            if not Path(args.prompts_file).exists():
                console.print(f"Error: Prompts file not found: {args.prompts_file}")
                return
            
            with open(args.prompts_file, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
            
            console.print(f"Loaded {len(prompts)} prompts")
            batch_generate(model_manager, prompts, args.output, args.temperature)
            
        elif args.examples:
            # Generate for example prompts
            console.print("Generating stories for example prompts...")
            prompts = create_story_prompts()
            batch_generate(model_manager, prompts, args.output, args.temperature)
            
        else:
            # Default to interactive mode
            console.print("No specific mode selected. Starting interactive mode...")
            interactive_mode(model_manager)
    
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        console.print(f"[red]Generation failed: {e}[/red]")
        raise

if __name__ == "__main__":
    main() 