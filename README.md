# Dan Brown Style Story Generator

A sophisticated fine-tuned LLaMA-3-8B model that generates thrilling stories in the distinctive style of Dan Brown, featuring his signature protagonists like Robert Langdon, mysterious symbols, ancient conspiracies, and edge-of-your-seat narrative tension.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Features

- **Fine-tuned LLaMA-3-8B**: Trained specifically on Dan Brown's literary style using advanced parameter-efficient fine-tuning
- **LoRA Integration**: Low-Rank Adaptation for memory-efficient training with minimal computational overhead
- **Unsloth Optimization**: Leverages Unsloth framework for 2x faster training and 50% memory reduction
- **4-bit Quantization**: BitsAndBytes integration for optimized inference on consumer hardware
- **Interactive Generation**: Professional CLI interface with rich formatting and real-time story creation
- **Comprehensive Evaluation**: Advanced metrics for story quality assessment and model comparison
- **Batch Processing**: Efficient multi-story generation with automated result organization
- **Model Comparison**: Systematic evaluation framework for comparing different model versions

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/dan-brown-story-generator.git
cd dan-brown-story-generator

# Run the setup script
./setup.sh

# Or install manually
pip install -r requirements.txt
pip install -e .
```

### Training a Model

```bash
# Train with default configuration
python train.py --data path/to/your/data.pkl

# Custom training with specific parameters
python train.py \
    --data path/to/data.pkl \
    --config config.yaml \
    --output models/my_dan_brown_model \
    --log-level INFO
```

### Generating Stories

```bash
# Interactive mode
python generate.py --model models/my_dan_brown_model --interactive

# Single story generation
python generate.py \
    --model models/my_dan_brown_model \
    --prompt "Write a story about Robert Langdon discovering ancient symbols in Paris"

# Batch generation with example prompts
python generate.py \
    --model models/my_dan_brown_model \
    --examples \
    --output generated_stories/
```

### Evaluating Model Performance

```bash
# Evaluate a single model
python evaluate.py --model models/my_dan_brown_model

# Compare multiple models
python evaluate.py --compare models/model_v1 models/model_v2 models/model_v3
```

## Dataset Creation Methodology

The training dataset was created using a sophisticated data generation pipeline that ensures high-quality prompt-response pairs:

### Chapter-to-Prompt Generation Process

1. **Source Material Extraction**: Dan Brown novels were systematically divided into individual chapters, preserving narrative structure and stylistic elements.

2. **Automated Prompt Generation**: Each chapter was processed through a pre-trained LLaMA model to generate contextually appropriate prompts that would logically result in the chapter's content. This reverse-engineering approach ensures that prompts are:
   - Narratively consistent with Dan Brown's style
   - Contextually appropriate for the chapter content
   - Varied in structure and complexity

3. **Input-Output Pair Creation**: 
   - **Input**: Generated prompts (e.g., "Write a story about Robert Langdon investigating mysterious symbols in the Vatican")
   - **Output**: Original Dan Brown chapter text
   - **Quality Assurance**: Each pair was validated for prompt-chapter alignment

4. **Dataset Characteristics**:
   - High semantic coherence between prompts and responses
   - Authentic Dan Brown narrative style preservation
   - Diverse prompt structures covering various story elements
   - Comprehensive coverage of recurring themes (symbols, conspiracies, historical mysteries)

This methodology ensures that the model learns not just to mimic Dan Brown's writing style, but to understand the relationship between story prompts and their narrative development, resulting in more coherent and contextually appropriate story generation.

## Technical Architecture

This project implements a sophisticated fine-tuning pipeline leveraging cutting-edge parameter-efficient training techniques:

### Core Technologies

- **Unsloth Framework**: Optimized training infrastructure providing 2x speedup and 50% memory reduction
- **LoRA (Low-Rank Adaptation)**: Parameter-efficient fine-tuning with rank-16 decomposition
- **4-bit Quantization**: BitsAndBytes integration for memory-optimized inference
- **SFT (Supervised Fine-Tuning)**: TRL-based training with custom dataset processing
- **Mixed Precision Training**: FP16/BF16 support for optimal hardware utilization

### Key Technical Innovations

- **Gradient Checkpointing**: Unsloth's optimized checkpointing for memory efficiency
- **Adaptive Batch Sizing**: Dynamic batch accumulation based on sequence length
- **Custom Tokenization**: Alpaca-style prompt formatting with EOS token handling
- **Advanced Evaluation Metrics**: Multi-dimensional story quality assessment

## Project Structure

```
dan-brown-story-generator/
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── config.py                 # Configuration management
│   ├── model.py                  # Model management and training
│   └── data_utils.py             # Data processing utilities
├── examples/                     # Example files and demonstrations
│   └── example_prompts.txt       # Sample story prompts
├── train.py                      # Training script
├── generate.py                   # Story generation script
├── evaluate.py                   # Model evaluation script
├── config.yaml                   # Default configuration
├── requirements.txt              # Python dependencies
├── setup.py                      # Package installation
├── README.md                     # This file
└── notebooks/                    # Jupyter notebooks (optional)
    └── DanBrown_Training_Clean.ipynb
```

## Configuration

The project uses a YAML configuration file for easy parameter management:

```yaml
# Model Configuration
model:
  name: "unsloth/llama-3-8b-bnb-4bit"
  max_seq_length: 2048
  load_in_4bit: true

# LoRA Configuration
lora:
  r: 16
  alpha: 16
  dropout: 0.0
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training Configuration
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  max_steps: 100
  learning_rate: 2e-4
  weight_decay: 0.01

# Generation Configuration
generation:
  max_new_tokens: 4096
  temperature: 0.7
  top_p: 0.9
  repetition_penalty: 1.1
```

## Usage Examples

### 1. Training Your Own Model

```python
from src.config import Config
from src.model import DanBrownModelManager
from src.data_utils import DataProcessor

# Load configuration
config = Config("config.yaml")

# Initialize model manager
model_manager = DanBrownModelManager(config)

# Load and setup model
model, tokenizer = model_manager.load_model()
model_manager.setup_peft_model()

# Prepare data
data_processor = DataProcessor(tokenizer)
inputs, outputs = data_processor.load_pickle_data("data/training_data.pkl")
train_dataset, val_dataset = data_processor.create_dataset(inputs, outputs)

# Train
model_manager.setup_trainer(train_dataset, val_dataset)
trainer_stats = model_manager.train()

# Save
model_manager.save_model("models/my_model")
```

### 2. Generating Stories Programmatically

```python
from src.config import Config
from src.model import DanBrownModelManager

# Load trained model
config = Config()
model_manager = DanBrownModelManager(config)
model_manager.load_trained_model("models/my_model")

# Generate story
prompt = "Write about Robert Langdon's adventure in the Vatican"
story = model_manager.generate_story(prompt, temperature=0.8)
print(story)
```

### 3. Batch Story Generation

```python
prompts = [
    "Create a thriller in the Louvre Museum",
    "Write about ancient symbols in Rome",
    "Describe a conspiracy in Barcelona"
]

stories = model_manager.batch_generate(prompts, temperature=0.7)
for prompt, story in zip(prompts, stories):
    print(f"Prompt: {prompt}")
    print(f"Story: {story[:200]}...")
    print("-" * 50)
```

## Evaluation Framework

The evaluation system provides comprehensive metrics for assessing story quality and model performance:

### Core Metrics

- **Basic Metrics**: Word count, sentence count, paragraph structure analysis
- **Dan Brown Style Score**: Quantitative assessment of character names, locations, and mystery elements
- **Coherence Score**: Narrative flow and transition quality evaluation
- **Creativity Score**: Vocabulary diversity and uniqueness measurement
- **Repetition Score**: Detection and quantification of repetitive content
- **Overall Quality**: Composite score combining all metrics with weighted importance

### Advanced Evaluation Features

- **Comparative Analysis**: Side-by-side model performance comparison
- **Statistical Reporting**: Detailed analytics with confidence intervals
- **Semantic Similarity**: Vector-based content similarity analysis
- **Style Consistency**: Dan Brown authorship likelihood scoring

## Advanced Features

### Model Comparison
```bash
python evaluate.py --compare \
    models/baseline_model \
    models/improved_model_v1 \
    models/improved_model_v2 \
    --output comparison_results/
```

### Custom Prompts File
```bash
# Create a file with your prompts (one per line)
echo "Write about Robert Langdon in Egypt" > my_prompts.txt
echo "Create a Vatican conspiracy story" >> my_prompts.txt

python generate.py \
    --model models/my_model \
    --prompts-file my_prompts.txt \
    --output my_stories/
```

### Interactive Story Generation
The interactive mode provides a rich CLI experience:
- Real-time story generation
- Built-in example prompts
- Story saving functionality
- Help system

## Training Optimization

### Best Practices for Fine-Tuning

1. **Data Quality**: Ensure your training data captures Dan Brown's distinctive style with proper chapter-prompt alignment
2. **Sequence Length**: Adjust `max_seq_length` based on your average story length and available GPU memory
3. **Learning Rate**: Start with 2e-4 and use learning rate scheduling for optimal convergence
4. **Batch Size**: Optimize based on GPU memory constraints and gradient accumulation steps
5. **Training Steps**: Monitor validation loss and implement early stopping to prevent overfitting
6. **LoRA Parameters**: Experiment with rank values (8, 16, 32) to balance performance and efficiency

### Memory Optimization Techniques

- **Gradient Checkpointing**: Reduces memory usage at the cost of computation time
- **Mixed Precision Training**: Utilizes FP16/BF16 for memory efficiency
- **Dynamic Batching**: Adapts batch size based on sequence length distribution
- **4-bit Quantization**: Enables training on consumer GPUs with limited VRAM

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch size or sequence length
python train.py --config config.yaml  # Edit batch size in config
```

**Model Loading Errors**
```bash
# Ensure correct model path
python generate.py --model path/to/correct/model/directory
```

**Dependencies Issues**
```bash
# Reinstall with force
pip install --force-reinstall -r requirements.txt
```

**Unsloth Installation Issues**
```bash
# Install from source if needed
pip install git+https://github.com/unslothai/unsloth.git
```

## Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Unsloth**: For the efficient fine-tuning framework and optimization techniques
- **Meta**: For the LLaMA-3 base model architecture
- **Hugging Face**: For the transformers library and ecosystem
- **Microsoft**: For the LoRA technique and parameter-efficient training
- **Tim Dettmers**: For BitsAndBytes quantization library
- **Dan Brown**: For the inspiring literary style and narrative structure

## Technical Support

For technical questions, issues, or collaborations:
- Open an issue on GitHub with detailed error logs
- Provide system specifications and environment details
- Include configuration files and command-line arguments used

## Future Enhancements

### Planned Features

- Support for other thriller authors' styles through multi-task learning
- Multi-language story generation with cross-lingual transfer
- Web interface with real-time generation and editing capabilities
- Integration with popular writing tools and platforms
- Advanced story structure analysis and plot consistency checking
- Character consistency tracking across generated narratives
- Automated hyperparameter optimization using Optuna
- Distributed training support for larger models

### Research Directions

- Exploration of newer PEFT techniques (AdaLoRA, QLoRA variants)
- Integration of retrieval-augmented generation for factual accuracy
- Style transfer capabilities for adapting to different authors
- Reinforcement learning from human feedback (RLHF) integration

---

**If you find this repository helpful for your research or projects, please consider citing it and starring the repository.**
