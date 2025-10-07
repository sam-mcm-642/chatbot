import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load, generate
from mlx_lm.utils import load_model
import json
import yaml
from pathlib import Path
from tqdm import tqdm
import time

class LoRALinear(nn.Module):
    """LoRA adaptation layer"""
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        
        # LoRA parameters
        self.lora_a = mx.random.normal((in_features, rank)) * 0.01
        self.lora_b = mx.zeros((rank, out_features))
    
    def __call__(self, x):
        # Original + LoRA adaptation
        return x @ self.lora_a @ self.lora_b * self.scale

def load_training_data(file_path, tokenizer, max_length=512):
    """Load and tokenize training data"""
    examples = []
    
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            text = data['text']
            
            # Tokenize
            tokens = tokenizer.encode(text)
            
            # Truncate if needed
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            
            examples.append(mx.array(tokens))
    
    return examples

def train_step(model, batch, optimizer):
    """Single training step"""
    
    def loss_fn(model, batch):
        # Forward pass
        logits = model(batch[:, :-1])
        targets = batch[:, 1:]
        
        # Compute loss (cross-entropy)
        loss = nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1)
        )
        return loss
    
    # Compute loss and gradients
    loss, grads = mx.value_and_grad(loss_fn)(model, batch)
    
    # Update parameters
    optimizer.update(model, grads)
    
    return loss

def finetune(config_path="configs/training_config.yaml"):
    """Main fine-tuning function"""
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print("="*50)
    print("Fine-tuning Configuration")
    print("="*50)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("="*50)
    
    # Load base model and tokenizer
    print("\nLoading base model...")
    model, tokenizer = load(config['base_model'])
    
    # Load training data
    print("Loading training data...")
    train_data = load_training_data(
        config['train_data'], 
        tokenizer, 
        config['max_seq_length']
    )
    
    val_data = load_training_data(
        config['val_data'],
        tokenizer,
        config['max_seq_length']
    ) if Path(config['val_data']).exists() else None
    
    print(f"Training examples: {len(train_data)}")
    if val_data:
        print(f"Validation examples: {len(val_data)}")
    
    # Setup optimizer
    optimizer = optim.AdamW(
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Training loop
    print("\nStarting training...")
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_steps = len(train_data) // config['batch_size'] * config['num_epochs']
    step = 0
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Shuffle data
        import random
        random.shuffle(train_data)
        
        # Batch training
        for i in tqdm(range(0, len(train_data), config['batch_size'])):
            batch = train_data[i:i + config['batch_size']]
            
            # Pad batch to same length
            max_len = max(len(ex) for ex in batch)
            batch = mx.stack([
                mx.pad(ex, (0, max_len - len(ex))) 
                for ex in batch
            ])
            
            # Train step
            loss = train_step(model, batch, optimizer)
            
            # Logging
            if step % config['log_every'] == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")
            
            # Evaluation
            if val_data and step % config['eval_every'] == 0:
                print("Running validation...")
                # Add validation logic here
            
            # Save checkpoint
            if step % config['save_every'] == 0:
                checkpoint_path = output_dir / f"checkpoint_{step}"
                print(f"Saving checkpoint to {checkpoint_path}")
                # Save model (implement save logic)
            
            step += 1
    
    # Save final model
    final_path = output_dir / "final"
    print(f"\nTraining complete! Saving final model to {final_path}")
    
    # Note: MLX save format depends on version
    # You may need to adapt this
    model.save_weights(str(final_path / "weights.npz"))
    tokenizer.save_pretrained(str(final_path))
    
    # Save config
    with open(final_path / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Done!")
    return final_path

if __name__ == "__main__":
    finetune()