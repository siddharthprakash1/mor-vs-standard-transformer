# train_standard_400m.py
import os
import json
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2TokenizerFast
from standard_transformer import StandardTransformerLM, StandardTransformerConfig

# Simple text dataset
class SimpleTextDataset(Dataset):
    def __init__(self, text, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Tokenize the entire text
        tokens = tokenizer.encode(text)
        # Create overlapping chunks
        self.examples = []
        stride = max_length // 2  # 50% overlap
        for i in range(0, len(tokens) - max_length + 1, stride):
            chunk = tokens[i:i + max_length]
            self.examples.append(torch.tensor(chunk, dtype=torch.long))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def load_sample_text():
    """Load sample text for training"""
    sample_text = """
    The quick brown fox jumps over the lazy dog. This is a sample text for training a language model.
    Machine learning is a fascinating field that combines statistics, computer science, and domain expertise.
    Natural language processing enables computers to understand and generate human language.
    Deep learning models like transformers have revolutionized the field of artificial intelligence.
    """ * 1000  # Repeat to have enough data
    return sample_text

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model configuration optimized for ~400M parameters and 8GB VRAM
    config = StandardTransformerConfig(
        vocab_size=50257,
        d_model=1024,      # Reduced from 1280
        n_heads=16,        # Reduced from 20
        d_ff=4096,         # Reduced from 5120
        n_layers=24,       # Reduced from 36 (still 2x MoR's 12 layers)
        dropout=0.1,
        max_seq_len=512
    )
    
    # Calculate approximate parameter count
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    model = StandardTransformerLM(config).to(device)
    param_count = count_params(model)
    print(f"Model config: {config.d_model}d, {config.n_heads}h, {config.d_ff}ff, {config.n_layers}L")
    print(f"Approximate parameter count: {param_count/1e6:.1f}M")
    
    # Training setup
    max_iters = 2500  # Updated to 2500 steps
    eval_interval = 100
    early_stopping_patience = 15  # Stop if no improvement for 15 evals
    learning_rate = 1e-4
    batch_size = 1  # Small batch size for memory constraints
    
    # Load data
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    text = load_sample_text()
    dataset = SimpleTextDataset(text, tokenizer, max_length=config.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Training Standard Transformer for {max_iters} steps on {len(dataset)} batches...")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')
    
    # Training tracking
    train_losses = []
    train_steps = []
    eval_losses = []
    eval_steps = []
    best_eval_loss = float('inf')
    patience_counter = 0
    
    model.train()
    step = 0
    start_time = time.time()
    
    while step < max_iters:
        for batch in dataloader:
            if step >= max_iters:
                break
                
            batch = batch.to(device)
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda'):
                logits, loss, aux = model(batch, targets=batch)
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Logging
            if step % 50 == 0:
                elapsed = time.time() - start_time
                memory_used = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                lm_loss = aux.get('loss_lm', loss).item()
                
                print(f"step {step}/{max_iters} ({100*step/max_iters:.1f}%) "
                      f"loss {loss.item():.3f} lm {lm_loss:.3f} "
                      f"mem {memory_used:.1f}GB")
            
            # Record training loss
            train_losses.append(loss.item())
            train_steps.append(step)
            
            # Evaluation
            if step % eval_interval == 0 and step > 0:
                model.eval()
                eval_loss = 0
                eval_batches = 0
                
                with torch.no_grad():
                    for eval_batch in dataloader:
                        if eval_batches >= 10:  # Limit eval batches
                            break
                        eval_batch = eval_batch.to(device)
                        with torch.amp.autocast('cuda'):
                            _, loss_eval, _ = model(eval_batch, targets=eval_batch)
                        eval_loss += loss_eval.item()
                        eval_batches += 1
                
                eval_loss /= eval_batches
                eval_losses.append(eval_loss)
                eval_steps.append(step)
                
                print(f"Eval at step {step}: loss {eval_loss:.4f}")
                
                # Early stopping check
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    patience_counter = 0
                    # Save best model
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'config': config,
                        'step': step,
                        'loss': eval_loss
                    }, 'best_standard_model.pt')
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at step {step} (no improvement for {early_stopping_patience} evaluations)")
                        break
                
                model.train()
            
            step += 1
        
        # Break outer loop if early stopping triggered
        if patience_counter >= early_stopping_patience:
            break
    
    # Save final model and training data
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'step': step,
        'train_losses': train_losses,
        'train_steps': train_steps,
        'eval_losses': eval_losses,
        'eval_steps': eval_steps
    }, 'final_standard_model.pt')
    
    # Save training data for visualization
    training_data = {
        'train_losses': train_losses,
        'train_steps': train_steps,
        'eval_losses': eval_losses,
        'eval_steps': eval_steps,
        'config': {
            'd_model': config.d_model,
            'n_heads': config.n_heads,
            'd_ff': config.d_ff,
            'n_layers': config.n_layers,
            'max_iters': max_iters,
            'param_count': param_count
        }
    }
    
    with open('standard_training_data.json', 'w') as f:
        json.dump(training_data, f)
    
    print(f"Training completed! Final step: {step}")
    print(f"Best eval loss: {best_eval_loss:.4f}")
    print("Saved: final_standard_model.pt, best_standard_model.pt, standard_training_data.json")

if __name__ == "__main__":
    main()