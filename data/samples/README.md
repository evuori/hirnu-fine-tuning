# Sample Data

This directory contains sample data files demonstrating different MLX formats.

## Files

- `example_chat.jsonl` - Example of chat format (recommended for Q&A)
- `example_completion.jsonl` - Example of completion format (for prompt-completion pairs)

## Using Sample Data

These are minimal examples for testing the pipeline. For actual training, you'll need:
- Much more data (hundreds to thousands of examples)
- Diverse examples covering different aspects of Hirnu language
- Real Hirnu language content in `data/raw/`

## Format Examples

### Chat Format
```json
{
  "messages": [
    {"role": "system", "content": "System prompt"},
    {"role": "user", "content": "User message"},
    {"role": "assistant", "content": "Assistant response"}
  ]
}
```

### Completion Format
```json
{
  "prompt": "Input text",
  "completion": "Expected output"
}
```

### Text Format
```json
{
  "text": "Full text content for language modeling"
}
```

## Next Steps

1. Add your real Hirnu data to `data/raw/`
2. Customize preprocessing in `src/data/preprocessor.py`
3. Run data preparation: `python scripts/prepare_data.py`
