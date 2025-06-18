#!/usr/bin/env python3
"""
Dan Brown Style Story Generator - Evaluation Script

This script evaluates the quality of generated stories.
"""

import logging
import argparse
from pathlib import Path
import sys
import json
import pandas as pd
from typing import List, Dict, Any
import re

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config import Config
from model import DanBrownModelManager
from data_utils import create_story_prompts
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.progress import track

console = Console()

def setup_logging(level: str = "INFO"):
    """Setup logging with rich handler."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )

def calculate_story_metrics(story: str) -> Dict[str, Any]:
    """Calculate various metrics for a story."""
    words = story.split()
    sentences = re.split(r'[.!?]+', story)
    paragraphs = story.split('\n\n')
    
    # Dan Brown specific elements
    character_names = ['Langdon', 'Robert', 'Professor']
    location_keywords = ['Rome', 'Vatican', 'Paris', 'temple', 'church', 'museum', 'library']
    mystery_keywords = ['symbol', 'ancient', 'secret', 'mystery', 'conspiracy', 'code', 'cipher']
    
    char_mentions = sum(story.count(name) for name in character_names)
    location_mentions = sum(story.lower().count(word.lower()) for word in location_keywords)
    mystery_mentions = sum(story.lower().count(word.lower()) for word in mystery_keywords)
    
    metrics = {
        'word_count': len(words),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'paragraph_count': len([p for p in paragraphs if p.strip()]),
        'avg_words_per_sentence': len(words) / max(len(sentences), 1),
        'avg_sentences_per_paragraph': len(sentences) / max(len(paragraphs), 1),
        'character_mentions': char_mentions,
        'location_mentions': location_mentions,
        'mystery_mentions': mystery_mentions,
        'dan_brown_score': (char_mentions + location_mentions + mystery_mentions) / max(len(words), 1) * 100
    }
    
    return metrics

def evaluate_coherence(story: str) -> float:
    """Simple coherence evaluation based on sentence transitions."""
    sentences = re.split(r'[.!?]+', story)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) < 2:
        return 0.0
    
    # Simple coherence check - presence of transition words
    transition_words = ['however', 'therefore', 'meanwhile', 'suddenly', 'then', 'next', 'finally']
    coherence_score = 0
    
    for sentence in sentences:
        if any(word in sentence.lower() for word in transition_words):
            coherence_score += 1
    
    return coherence_score / len(sentences)

def evaluate_creativity(story: str) -> float:
    """Evaluate creativity based on unique word usage."""
    words = re.findall(r'\b\w+\b', story.lower())
    unique_words = set(words)
    
    if len(words) == 0:
        return 0.0
    
    # Creativity score based on vocabulary diversity
    creativity_score = len(unique_words) / len(words)
    return creativity_score

def evaluate_repetition(story: str) -> float:
    """Evaluate repetitiveness of the story."""
    sentences = re.split(r'[.!?]+', story)
    sentences = [s.strip().lower() for s in sentences if s.strip()]
    
    if len(sentences) < 2:
        return 0.0
    
    # Check for repeated sentences or phrases
    unique_sentences = set(sentences)
    repetition_score = 1.0 - (len(unique_sentences) / len(sentences))
    
    return repetition_score

def comprehensive_evaluation(model_manager, prompts: List[str], output_dir: str) -> Dict[str, Any]:
    """Perform comprehensive evaluation of the model."""
    console.print("Starting comprehensive evaluation...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for prompt in track(prompts, description="Evaluating stories..."):
        try:
            # Generate story
            story = model_manager.generate_story(prompt, max_new_tokens=500)
            
            # Calculate metrics
            basic_metrics = calculate_story_metrics(story)
            coherence = evaluate_coherence(story)
            creativity = evaluate_creativity(story)
            repetition = evaluate_repetition(story)
            
            result = {
                'prompt': prompt,
                'story': story,
                'word_count': basic_metrics['word_count'],
                'sentence_count': basic_metrics['sentence_count'],
                'paragraph_count': basic_metrics['paragraph_count'],
                'avg_words_per_sentence': basic_metrics['avg_words_per_sentence'],
                'character_mentions': basic_metrics['character_mentions'],
                'location_mentions': basic_metrics['location_mentions'],
                'mystery_mentions': basic_metrics['mystery_mentions'],
                'dan_brown_score': basic_metrics['dan_brown_score'],
                'coherence_score': coherence,
                'creativity_score': creativity,
                'repetition_score': repetition,
                'overall_quality': (coherence + creativity + (1 - repetition)) / 3
            }
            
            results.append(result)
            
        except Exception as e:
            console.print(f"Error evaluating prompt '{prompt[:50]}...': {e}")
            continue
    
    # Calculate aggregate statistics
    if results:
        df = pd.DataFrame(results)
        
        aggregate_stats = {
            'total_stories': len(results),
            'avg_word_count': df['word_count'].mean(),
            'avg_dan_brown_score': df['dan_brown_score'].mean(),
            'avg_coherence': df['coherence_score'].mean(),
            'avg_creativity': df['creativity_score'].mean(),
            'avg_repetition': df['repetition_score'].mean(),
            'avg_overall_quality': df['overall_quality'].mean(),
            'best_story_idx': df['overall_quality'].idxmax(),
            'worst_story_idx': df['overall_quality'].idxmin()
        }
        
        # Save detailed results
        results_path = output_dir / "evaluation_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save aggregate statistics
        stats_path = output_dir / "evaluation_summary.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(aggregate_stats, f, indent=2)
        
        # Save CSV for analysis
        csv_path = output_dir / "evaluation_metrics.csv"
        df.to_csv(csv_path, index=False)
        
        console.print(f"\nEvaluation completed!")
        console.print(f"Results saved to: {output_dir}")
        
        return aggregate_stats
    
    return {}

def display_evaluation_results(stats: Dict[str, Any]):
    """Display evaluation results in a nice table."""
    if not stats:
        console.print("No evaluation results to display")
        return
    
    # Create summary table
    table = Table(title="Model Evaluation Summary", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="white")
    table.add_column("Scale", style="dim")
    
    table.add_row("Stories Evaluated", str(stats['total_stories']), "-")
    table.add_row("Avg Word Count", f"{stats['avg_word_count']:.1f}", "words")
    table.add_row("Dan Brown Style Score", f"{stats['avg_dan_brown_score']:.2f}", "0-100")
    table.add_row("Coherence Score", f"{stats['avg_coherence']:.2f}", "0-1")
    table.add_row("Creativity Score", f"{stats['avg_creativity']:.2f}", "0-1")
    table.add_row("Repetition Score", f"{stats['avg_repetition']:.2f}", "0-1 (lower better)")
    table.add_row("Overall Quality", f"{stats['avg_overall_quality']:.2f}", "0-1")
    
    console.print(table)

def compare_models(model_paths: List[str], prompts: List[str], output_dir: str):
    """Compare multiple models on the same prompts."""
    console.print(f"Comparing {len(model_paths)} models...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_results = {}
    
    for model_path in model_paths:
        model_name = Path(model_path).name
        console.print(f"\nEvaluating model: {model_name}")
        
        try:
            # Load configuration and model
            config = Config()
            model_manager = DanBrownModelManager(config)
            model_manager.load_trained_model(model_path)
            
            # Evaluate
            stats = comprehensive_evaluation(model_manager, prompts, output_dir / model_name)
            comparison_results[model_name] = stats
            
        except Exception as e:
            console.print(f"Error evaluating model {model_name}: {e}")
            continue
    
    # Save comparison results
    comparison_path = output_dir / "model_comparison.json"
    with open(comparison_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    # Display comparison table
    if comparison_results:
        table = Table(title="Model Comparison", show_header=True)
        table.add_column("Model", style="cyan")
        table.add_column("Overall Quality", style="white")
        table.add_column("Dan Brown Score", style="white")
        table.add_column("Coherence", style="white")
        table.add_column("Creativity", style="white")
        
        for model_name, stats in comparison_results.items():
            if stats:
                table.add_row(
                    model_name,
                    f"{stats['avg_overall_quality']:.3f}",
                    f"{stats['avg_dan_brown_score']:.2f}",
                    f"{stats['avg_coherence']:.3f}",
                    f"{stats['avg_creativity']:.3f}"
                )
        
        console.print(table)
        console.print(f"\nDetailed comparison saved to: {comparison_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Dan Brown Style Story Generator")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model directory")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--prompts-file", type=str,
                       help="File containing evaluation prompts (one per line)")
    parser.add_argument("--output", type=str, default="evaluation_results",
                       help="Output directory for evaluation results")
    parser.add_argument("--num-prompts", type=int, default=10,
                       help="Number of example prompts to use if no prompts file provided")
    parser.add_argument("--compare", nargs="+",
                       help="Compare multiple models (provide multiple model paths)")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    console.print("[bold blue]Dan Brown Style Story Generator - Evaluation[/bold blue]")
    console.print("=" * 60)
    
    try:
        # Load prompts
        if args.prompts_file and Path(args.prompts_file).exists():
            with open(args.prompts_file, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
        else:
            prompts = create_story_prompts()[:args.num_prompts]
        
        console.print(f"Using {len(prompts)} evaluation prompts")
        
        if args.compare:
            # Compare multiple models
            compare_models(args.compare, prompts, args.output)
        else:
            # Evaluate single model
            config = Config(args.config)
            model_manager = DanBrownModelManager(config)
            
            console.print(f"Loading model from {args.model}...")
            model_manager.load_trained_model(args.model)
            
            stats = comprehensive_evaluation(model_manager, prompts, args.output)
            display_evaluation_results(stats)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        console.print(f"[red]Evaluation failed: {e}[/red]")
        raise

if __name__ == "__main__":
    main() 