## Project Overview

This is a Korean language fine-tuning project for GPT-OSS 20B model using Unsloth framework. The project focuses on educational content generation in Korean using LoRA (Low-Rank Adaptation) techniques for efficient training.

## Development Setup

### Required Dependencies
The project requires the following Python packages (install via uv):
```bash
uv add transformers torch unsloth trl datasets huggingface-hub peft
```

### Core Libraries Used
- **transformers**: Hugging Face transformers library for model handling
- **unsloth**: Efficient fine-tuning framework with memory optimizations
- **trl**: Transformer Reinforcement Learning for SFT training
- **datasets**: Dataset loading and processing (using maywell/korean_textbooks)
- **peft**: Parameter Efficient Fine-Tuning for LoRA implementation
- **huggingface-hub**: Model uploading and repository management

## Model Architecture

### Base Model Configuration
- **Base Model**: `unsloth/gpt-oss-20b-unsloth-bnb-4bit` (20B parameter model)
- **Quantization**: 4-bit quantization for memory efficiency
- **Context Length**: 1024 tokens maximum sequence length
- **Fine-tuning Method**: LoRA with the following parameters:
  - Rank (r): 8
  - Alpha: 16
  - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
  - Dropout: 0 (optimized)
  - Bias: none (optimized)

### Training Configuration
- **Dataset**: maywell/korean_textbooks (Claude evolution variant)
- **Batch Size**: 1 per device with 4 gradient accumulation steps
- **Learning Rate**: 2e-4 with linear scheduler
- **Training Steps**: 30 (configurable via max_steps parameter)
- **Optimizer**: AdamW 8-bit for memory efficiency

## Key Functions

### Model Loading and Setup
The script includes comprehensive model loading with memory optimization:
- Automatic dtype detection
- 4-bit quantization loading
- LoRA adapter configuration
- Memory-efficient gradient checkpointing

### Training Process
- SFT (Supervised Fine-Tuning) using TRL framework
- Memory monitoring throughout training process
- Performance metrics collection (runtime, memory usage)

### Model Upload System
Two main upload functions are available:
- `save_and_upload_model()`: Save fine-tuned model and upload to Hugging Face
- `upload_existing_model()`: Upload previously saved model

### Helper Functions
- `fix_adapter_config()`: Fixes adapter configuration for compatibility
- `create_model_card()`: Generates comprehensive Korean model documentation
- `formatting_prompts_func()`: Handles dataset text formatting

## Memory Management

The script includes comprehensive GPU memory monitoring:
- Pre-training memory baseline measurement
- Peak memory usage tracking during training
- Memory percentage calculations
- LoRA-specific memory overhead analysis

## Usage Patterns

### Training Execution
1. Load base model with quantization
2. Configure LoRA parameters
3. Load and format Korean textbook dataset
4. Execute supervised fine-tuning
5. Monitor memory usage throughout process

### Model Testing
The script includes inference examples using chat templates with Korean educational prompts. Temperature and top-p parameters are configurable for response variety control.

### Model Deployment
The upload system creates production-ready model repositories with:
- Comprehensive Korean documentation
- Proper licensing and attribution
- Usage examples and system requirements
- Educational domain specialization tags

## Project Structure

The project has been refactored into a modular architecture for better maintainability:

```
core/
├── __init__.py
├── config/
│   ├── __init__.py
│   └── model_config.py      # Centralized configuration management
├── model/
│   ├── __init__.py
│   ├── loader.py           # Model loading and PEFT setup
│   └── inference.py        # Inference and testing
├── data/
│   ├── __init__.py
│   └── processor.py        # Dataset loading and processing
├── training/
│   ├── __init__.py
│   ├── trainer.py          # SFT training orchestration
│   └── monitor.py          # GPU memory monitoring
└── deployment/
    ├── __init__.py
    ├── uploader.py         # Model upload to Hugging Face
    └── model_card.py       # Model documentation generation

main.py                     # Main execution script
requirements.txt           # Dependencies
```

### Key Modules

- **ModelConfig**: Centralized configuration with all original parameters preserved
- **ModelLoader**: Handles model loading and LoRA setup
- **DataProcessor**: Dataset loading and formatting functions
- **Trainer**: SFT training with memory monitoring
- **ModelUploader**: Hugging Face Hub integration
- **ModelInference**: Testing and inference functionality

### Running the Code

```bash
# Install dependencies
uv sync

# Run full pipeline
python main.py

# Choose execution mode:
# 1. Full fine-tuning pipeline
# 2. Inference only
# 3. Upload only
```

### Original Logic Preservation

All original functionality has been preserved:
- Identical model parameters and training configuration
- Same memory monitoring and statistics output
- Original inference test examples
- Complete upload workflow with model card generation

The output directory structure includes:
- `outputs/`: Training checkpoints and logs
- `korean_textbook_model/`: Final fine-tuned model artifacts


###Issue fix
https://github.com/unslothai/unsloth-zoo/pull/234#issuecomment-3182335661