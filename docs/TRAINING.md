# Training Guide

This guide explains how to train the Hirnu language model.

## Prerequisites

Before training:
1. Complete the [Setup Guide](SETUP.md)
2. Prepare your data following [Data Preparation Guide](DATA_PREPARATION.md)
3. Verify datasets exist in `data/processed/`

## Quick Start

Basic training with default configuration:

```bash
python scripts/train.py
```

This will:
- Load the base model specified in config
- Apply LoRA adapters (if enabled)
- Train on `data/processed/train.jsonl`
- Validate on `data/processed/valid.jsonl`
- Save checkpoints to `models/checkpoints/`
- Save final model to `models/hirnu-finetuned/`

## Training Configuration

### Edit Configuration File

Customize training by editing `configs/training_config.yaml`:

```yaml
model:
  name: "mlx-community/Llama-3.2-3B-Instruct-4bit"
  output_dir: "models/hirnu-finetuned"

training:
  num_epochs: 3
  batch_size: 4
  learning_rate: 1.0e-5
  warmup_steps: 100
  max_seq_length: 2048
```

### Key Parameters

#### Model Settings

- `model.name`: Base model to fine-tune
- `model.output_dir`: Where to save trained model

#### Training Hyperparameters

- `num_epochs`: Number of training epochs (default: 3)
- `batch_size`: Training batch size (default: 4)
- `learning_rate`: Learning rate (default: 1e-5)
- `max_seq_length`: Maximum sequence length (default: 2048)

Adjust based on:
- **Smaller batch_size** if you have memory issues
- **Higher learning_rate** for faster training (but less stable)
- **More epochs** for better performance (but watch for overfitting)

#### LoRA Configuration

LoRA (Low-Rank Adaptation) enables efficient fine-tuning:

```yaml
training:
  use_lora: true
  lora_rank: 8
  lora_alpha: 16
  lora_dropout: 0.05
```

Benefits of LoRA:
- Reduces memory usage
- Faster training
- Smaller model checkpoints
- Easy to swap different adapters

Parameters:
- `lora_rank`: Rank of adaptation matrices (4-64, higher = more capacity)
- `lora_alpha`: Scaling factor (typically 2x rank)
- `lora_dropout`: Dropout rate for regularization

### Command-Line Options

Override configuration with command-line arguments:

```bash
# Use custom config file
python scripts/train.py --config my_training_config.yaml

# Use custom data files
python scripts/train.py \
  --train-data my_data/train.jsonl \
  --valid-data my_data/valid.jsonl

# Resume from checkpoint
python scripts/train.py --resume-from models/checkpoints/checkpoint-1000

# Dry run (test config without training)
python scripts/train.py --dry-run
```

## Monitoring Training

### Training Logs

Logs are saved to `outputs/logs/`:
- Console output shows progress
- Detailed logs saved to `hirnu_TIMESTAMP.log`

### Metrics

During training, monitor:
- **Loss**: Should decrease over time
- **Perplexity**: Should decrease (exp(loss))
- **Validation metrics**: Check for overfitting

Example output:
```
Epoch 1/3
  Batch 100/1000 - Loss: 2.345 - LR: 1.0e-5
  Batch 200/1000 - Loss: 2.123 - LR: 1.0e-5
  ...
  Validation - Loss: 2.234 - Perplexity: 9.34
```

### Checkpoints

Checkpoints are automatically saved:
- Location: `models/checkpoints/`
- Frequency: Every N steps (configured in `checkpointing.save_steps`)
- Retention: Keep last N checkpoints (configured in `checkpointing.keep_last_n`)

## Memory Management

### If you encounter OOM (Out of Memory) errors:

1. **Reduce batch size:**
   ```yaml
   training:
     batch_size: 2  # or even 1
   ```

2. **Use gradient accumulation:**
   ```yaml
   training:
     batch_size: 1
     gradient_accumulation_steps: 4  # Effective batch size = 4
   ```

3. **Reduce sequence length:**
   ```yaml
   training:
     max_seq_length: 1024  # Instead of 2048
   ```

4. **Enable LoRA:**
   ```yaml
   training:
     use_lora: true
   ```

## Training Tips

### For Best Results

1. **Start with defaults**: Use default configuration for first training run
2. **Monitor validation**: Watch for overfitting (validation loss increasing)
3. **Use LoRA**: More efficient and flexible than full fine-tuning
4. **Smaller learning rate**: Safer for small datasets (try 5e-6 or 1e-6)
5. **More data**: Better results with more diverse training examples

### For Faster Experimentation

1. **Reduce epochs**: Use 1 epoch for quick tests
2. **Smaller dataset**: Test with subset of data first
3. **Skip validation**: Comment out validation steps temporarily
4. **Lower save frequency**: Reduce checkpoint saving overhead

### For Production Models

1. **More epochs**: 5-10 epochs for better performance
2. **Learning rate scheduling**: Implement warmup and decay
3. **Multiple runs**: Train with different random seeds
4. **Hyperparameter tuning**: Experiment with different configurations

## Evaluation

After training, evaluate your model:

```bash
python scripts/evaluate.py --model-path models/hirnu-finetuned
```

For LoRA models:

```bash
python scripts/evaluate.py \
  --model-path mlx-community/Llama-3.2-3B-Instruct-4bit \
  --adapter-path models/checkpoints/adapters.npz
```

Evaluation metrics:
- Perplexity
- Translation accuracy (if applicable)
- Character error rate
- Word accuracy

## Inference

Test your trained model interactively:

```bash
python scripts/inference.py --model-path models/hirnu-finetuned
```

For LoRA models:

```bash
python scripts/inference.py \
  --model-path mlx-community/Llama-3.2-3B-Instruct-4bit \
  --adapter-path models/checkpoints/adapters.npz
```

### Interactive Commands

In interactive mode:
```
> /translate Hello, how are you?
Translating to Hirnu...
Hirnu: [translation]

> /generate Tell me about Hirnu language
Generating...
Generated: [response]

> /quit
```

### Single Translation

```bash
python scripts/inference.py \
  --model-path models/hirnu-finetuned \
  --translate "Hello world"
```

## Advanced Topics

### Custom Training Loop

For advanced customization, edit `src/training/trainer.py`:
- Custom loss functions
- Additional metrics
- Custom callbacks

### Distributed Training

MLX supports multi-GPU training. Configure in training script.

### Hyperparameter Search

Use tools like Optuna or Ray Tune for automated hyperparameter optimization.

## Troubleshooting

### Training is very slow

- Reduce sequence length
- Increase batch size (if memory allows)
- Check data loading efficiency
- Verify MLX is using GPU acceleration

### Loss not decreasing

- Check learning rate (may be too low or too high)
- Verify data quality
- Try different optimizer settings
- Increase model capacity (higher LoRA rank)

### Validation loss increasing

- Overfitting - reduce epochs
- Add dropout/regularization
- Get more training data
- Try data augmentation

### Model not learning

- Check data format is correct
- Verify labels are present
- Increase learning rate
- Train for more epochs
- Check for data preprocessing issues

## Next Steps

After successful training:
1. Evaluate model performance
2. Test with real examples
3. Iterate on data and configuration
4. Deploy model for inference
5. Share results and checkpoints

## Resources

- MLX Documentation: https://ml-explore.github.io/mlx/
- MLX-LM Repository: https://github.com/ml-explore/mlx-examples/tree/main/llms
- Project configs: `configs/`
- Source code: `src/`
