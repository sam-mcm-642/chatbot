import json
from pathlib import Path
from typing import List, Dict

def validate_example(example: Dict) -> bool:
    """Check if training example is valid"""
    if 'text' not in example:
        return False
    if len(example['text'].strip()) < 10:
        return False
    return True

def format_conversation(user_message: str, assistant_message: str) -> str:
    """Format a conversation pair for training"""
    # Llama chat format
    formatted = f"<|user|>\n{user_message.strip()}<|end|>\n<|assistant|>\n{assistant_message.strip()}<|end|>\n"
    return formatted

def prepare_manual_dataset(
    input_file: str,
    output_file: str = "data/processed/training_data.jsonl"
):
    """
    Convert manual training data to MLX format
    
    Input file should be JSON with format:
    [
        {"user": "question", "assistant": "answer"},
        {"user": "question", "assistant": "answer"}
    ]
    """
    
    # Read input
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Convert to training format
    training_examples = []
    for item in data:
        formatted_text = format_conversation(item['user'], item['assistant'])
        training_examples.append({"text": formatted_text})
    
    # Validate
    valid_examples = [ex for ex in training_examples if validate_example(ex)]
    
    print(f"Total examples: {len(data)}")
    print(f"Valid examples: {len(valid_examples)}")
    print(f"Filtered out: {len(data) - len(valid_examples)}")
    
    # Save as JSONL
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for example in valid_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"Saved to {output_file}")
    
    # Show sample
    if valid_examples:
        print("\nSample training example:")
        print(valid_examples[0]['text'][:200] + "...")
    
    return output_file

def split_train_val(
    input_file: str,
    train_ratio: float = 0.9
):
    """Split data into train and validation sets"""
    
    with open(input_file, 'r') as f:
        examples = [json.loads(line) for line in f]
    
    import random
    random.shuffle(examples)
    
    split_idx = int(len(examples) * train_ratio)
    train_data = examples[:split_idx]
    val_data = examples[split_idx:]
    
    # Save splits
    train_path = input_file.replace('.jsonl', '_train.jsonl')
    val_path = input_file.replace('.jsonl', '_val.jsonl')
    
    with open(train_path, 'w') as f:
        for ex in train_data:
            f.write(json.dumps(ex) + '\n')
    
    with open(val_path, 'w') as f:
        for ex in val_data:
            f.write(json.dumps(ex) + '\n')
    
    print(f"Train: {len(train_data)} examples -> {train_path}")
    print(f"Val: {len(val_data)} examples -> {val_path}")
    
    return train_path, val_path

if __name__ == "__main__":
    # Example: prepare your manual dataset
    # First create data/raw/manual_training.json with your Q&A pairs
    prepare_manual_dataset(
        "data/raw/manual_training.json",
        "data/processed/training_data.jsonl"
    )
    
    # Split into train/val
    split_train_val("data/processed/training_data.jsonl")