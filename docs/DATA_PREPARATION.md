# Data Preparation Guide

This guide explains how to prepare your Hirnu language data for model training.

## Overview

The data preparation pipeline consists of:

1. **Data Collection** - Gather grammar, vocabulary, and text data
2. **Preprocessing** - Clean and normalize raw data
3. **Format Conversion** - Convert to MLX-compatible JSONL format
4. **Dataset Splitting** - Create train/test/valid splits
5. **Validation** - Verify dataset quality

## Data Requirements

### MLX Format Requirements

MLX fine-tuning requires data in JSONL (JSON Lines) format with three splits:
- `train.jsonl` - Training data
- `test.jsonl` - Test data (for evaluation during training)
- `valid.jsonl` - Validation data (for final evaluation)

### Supported Formats

The pipeline supports three MLX data formats:

#### 1. Chat Format (Recommended for Q&A and conversation)

```json
{
  "messages": [
    {"role": "system", "content": "System prompt here"},
    {"role": "user", "content": "User message"},
    {"role": "assistant", "content": "Assistant response"}
  ]
}
```

#### 2. Completion Format (For prompt-completion pairs)

```json
{
  "prompt": "Translate to Hirnu: Hello",
  "completion": "Hirnu translation here"
}
```

#### 3. Text Format (For general text generation)

```json
{
  "text": "Full text content here"
}
```

## Organizing Raw Data

### Directory Structure

Place your raw data in these directories:

```
data/raw/
├── grammar/       # Grammar rules and examples
├── vocabulary/    # Word definitions and translations
└── texts/         # Hirnu texts and stories
```

### Grammar Data

Place grammar files in `data/raw/grammar/`:
- Grammar rules
- Sentence patterns
- Language structure examples

Example file structure:
```
data/raw/grammar/
├── basic_rules.txt
├── verb_conjugation.txt
└── sentence_structure.txt
```

### Vocabulary Data

Place vocabulary files in `data/raw/vocabulary/`:
- Word lists
- English-Hirnu translations
- Definitions

Example file structure:
```
data/raw/vocabulary/
├── common_words.txt
├── verbs.txt
└── nouns.txt
```

### Text Data

Place Hirnu texts in `data/raw/texts/`:
- Stories
- Dialogues
- Example sentences

Example file structure:
```
data/raw/texts/
├── story_01.txt
├── dialogue_01.txt
└── examples.txt
```

## Data Format Guidelines

### For Translation Tasks

Structure your data as English-Hirnu pairs. Example format in a text file:

```
EN: Hello, how are you?
HI: [Hirnu translation]

EN: What is your name?
HI: [Hirnu translation]
```

### For Text Generation

Simply include Hirnu text content. The system will use it for language modeling.

### For Q&A Tasks

Structure as question-answer pairs:

```
Q: What is Hirnu?
A: Hirnu is an ancient Scandinavian language...

Q: How do you say "hello" in Hirnu?
A: [Hirnu translation]
```

## Customizing Data Processing

### Modify the Preprocessor

Edit `src/data/preprocessor.py` to customize how your data is processed:

```python
def preprocess_vocabulary_data(self, vocab_dir: Path) -> List[Dict[str, str]]:
    """Process vocabulary data into training examples."""
    examples = []

    # Your custom processing logic here
    # For example, parse your specific file format

    return examples
```

### Modify the Converter

Edit `src/data/converter.py` to customize format conversion:

```python
def to_chat_format(self, example: Dict[str, Any]) -> Dict[str, Any]:
    """Convert example to chat format."""
    # Customize based on your data structure

    messages = [
        {"role": "system", "content": self.chat_template["system"]},
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]}
    ]

    return {"messages": messages}
```

## Running Data Preparation

### Basic Usage

```bash
python scripts/prepare_data.py
```

This will:
1. Process all raw data
2. Convert to MLX format (configured in `configs/data_config.yaml`)
3. Create train/test/valid splits (80/10/10 by default)
4. Validate output datasets

### Custom Configuration

Use a custom config file:

```bash
python scripts/prepare_data.py --config my_config.yaml
```

### Validation Only

Validate existing datasets without reprocessing:

```bash
python scripts/prepare_data.py --validate-only
```

### Skip Validation

Skip validation step (faster, for development):

```bash
python scripts/prepare_data.py --skip-validation
```

## Configuration Options

Edit `configs/data_config.yaml` to customize:

### Dataset Split Ratios

```yaml
splits:
  train: 0.8
  test: 0.1
  valid: 0.1
  random_seed: 42
```

### MLX Format

```yaml
format:
  type: "chat"  # Options: "chat", "completion", "text"
  max_length: 2048
```

### Preprocessing Options

```yaml
preprocessing:
  lowercase: false
  remove_special_chars: false
  normalize_whitespace: true
  min_text_length: 10
  max_text_length: 4096
```

## Validating Datasets

The validation step checks:
- File existence
- JSON format validity
- Required fields presence
- Data structure compliance

If validation fails, review the error messages and fix the issues in your raw data or preprocessing logic.

## Example Workflow

1. **Add raw data:**
   ```bash
   # Add your files to data/raw directories
   cp my_grammar_files/* data/raw/grammar/
   cp my_vocab_files/* data/raw/vocabulary/
   cp my_texts/* data/raw/texts/
   ```

2. **Configure format:**
   ```bash
   # Edit configs/data_config.yaml
   # Set format.type to "chat", "completion", or "text"
   ```

3. **Customize processing (if needed):**
   ```bash
   # Edit src/data/preprocessor.py
   # Implement your custom data parsing logic
   ```

4. **Run preparation:**
   ```bash
   python scripts/prepare_data.py
   ```

5. **Verify output:**
   ```bash
   # Check the generated files
   head -n 5 data/processed/train.jsonl
   ```

## Sample Data

For testing without real data, create sample files:

```bash
# Create sample text
echo "Sample Hirnu text for testing" > data/raw/texts/sample.txt

# Run preparation
python scripts/prepare_data.py
```

## Next Steps

After data preparation:
1. Review generated datasets in `data/processed/`
2. Proceed to training - see [TRAINING.md](TRAINING.md)
3. Adjust configuration if needed and re-run preparation

## Troubleshooting

### No examples found

- Verify files exist in `data/raw/` directories
- Check file formats are readable
- Review custom preprocessing logic

### Validation errors

- Check JSONL format (one JSON object per line)
- Verify required fields are present
- Review error messages for specific issues

### Dataset too small

- Add more raw data
- Adjust split ratios in configuration
- Consider data augmentation techniques
