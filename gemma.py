import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import time
import json
import datetime

# Set your HF token
os.environ["HF_TOKEN"] = "Insert your Hugging face token"

def load_wikipedia_dataset():
    """Load Wikipedia dataset for more realistic training"""
    try:
        from datasets import load_dataset
        print("Loading Wikipedia dataset...")

        # Use the current format - wikimedia/wikipedia with 2023 snapshot
        dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        
        # Process articles and extract text
        articles = []
        total_chars = 0
        max_articles = 10000  # Limit for memory management
        
        print("Processing Wikipedia articles...")
        for i, article in enumerate(dataset):
            if i >= max_articles:
                break
                
            # Extract text content from the article
            text = article.get('text', '')
            if len(text) > 100:  # Only include substantial articles
                articles.append(text)
                total_chars += len(text)
                
                if (i + 1) % 1000 == 0:
                    print(f"Loaded {i + 1} Wikipedia articles...")
                    
                # Stop if we have enough content (50MB+ of text)
                if total_chars > 50_000_000:
                    break
        
        print(f"Loaded {len(articles)} Wikipedia articles, total chars: {total_chars:,}")
        
        # Combine all articles into one large text
        combined_text = '\n\n'.join(articles)
        return combined_text
        
    except Exception as e:
        print(f"Failed to load Wikipedia dataset: {e}")
        print("Falling back to OpenWebText...")
        
        # Fallback to OpenWebText
        try:
            dataset = load_dataset("openwebtext", split="train", streaming=True)
            
            articles = []
            total_chars = 0
            max_articles = 5000
            
            print("Processing OpenWebText articles...")
            for i, article in enumerate(dataset):
                if i >= max_articles:
                    break
                    
                text = article.get('text', '')
                if len(text) > 100:
                    articles.append(text)
                    total_chars += len(text)
                    
                    if (i + 1) % 1000 == 0:
                        print(f"Loaded {i + 1} OpenWebText articles...")
                        
                    if total_chars > 30_000_000:
                        break
            
            print(f"Loaded {len(articles)} OpenWebText articles, total chars: {total_chars:,}")
            combined_text = '\n\n'.join(articles)
            return combined_text
            
        except Exception as e2:
            print(f"Failed to load OpenWebText: {e2}")
            print("Using fallback sample text...")
            
            # Final fallback to sample text
            return """
            The quick brown fox jumps over the lazy dog. This is a sample text for training.
            Machine learning is a subset of artificial intelligence that focuses on algorithms.
            Natural language processing enables computers to understand human language.
            Deep learning uses neural networks with multiple layers to learn complex patterns.
            """ * 1000  # Repeat to make it longer

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Tokenize the entire text
        tokens = tokenizer.encode(text)
        
        # Create overlapping sequences
        self.sequences = []
        stride = max_length // 2  # 50% overlap
        
        for i in range(0, len(tokens) - max_length + 1, stride):
            sequence = tokens[i:i + max_length]
            self.sequences.append(sequence)
        
        print(f"Created {len(self.sequences)} training sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Input is sequence[:-1], target is sequence[1:]
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        targets = torch.tensor(sequence[1:], dtype=torch.long)
        
        return input_ids, targets

def train_standard_gemma():
    """Train standard Gemma 3 270M model"""
    print("=== Training Standard Gemma 3 270M ===")
    
    # Initialize training statistics tracker
    training_stats = {
        'experiment_info': {
            'model_name': 'Standard Gemma 3 270M',
            'timestamp': datetime.datetime.now().isoformat(),
            'device': str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
            'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else None
        },
        'hyperparameters': {
            'learning_rate': 1e-4,
            'batch_size': 1,
            'gradient_accumulation_steps': 8,
            'max_sequence_length': 128,
            'num_epochs': 1,
            'max_batches': 400,
            'scheduler': 'CosineAnnealingLR'
        },
        'training_progress': [],
        'final_results': {}
    }
    
    # Load tokenizer and model - using correct 270M model name
    model_name = "google/gemma-3-270m"
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ["HF_TOKEN"])
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        token=os.environ["HF_TOKEN"],
        torch_dtype=torch.float32,  # Use float32 for consistency with MoR
        device_map="auto",
        attn_implementation="eager"  # Use eager attention as recommended
    )
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Add model configuration to stats
    training_stats['model_config'] = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params
    }
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Load dataset
    text_data = load_wikipedia_dataset()
    
    # Create dataset and dataloader with aggressive memory optimization
    dataset = TextDataset(text_data, tokenizer, max_length=128)  # Reduced from 256
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=False)
    
    # Add dataset info to stats
    training_stats['dataset_info'] = {
        'num_sequences': len(dataset),
        'text_length_chars': len(text_data),
        'max_sequence_length': 128
    }
    
    # Setup optimizer with higher learning rate
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)  # Increased learning rate for longer training
    
    # Add learning rate scheduler for longer training
    scheduler = CosineAnnealingLR(optimizer, T_max=100)
    
    # Training parameters - optimized for 8GB GPU
    gradient_accumulation_steps = 8  # Increased to maintain effective batch size
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    model.train()
    total_loss = 0
    step = 0
    
    print(f"Starting training for 400 batches...")
    start_time = time.time()
    
    num_batches = 0
    
    for batch_idx, (input_ids, targets) in enumerate(dataloader):
        try:
            input_ids = input_ids.to(model.device)
            targets = targets.to(model.device)
            
            # Clear cache before forward pass
            torch.cuda.empty_cache()
            
            # Forward pass
            outputs = model(input_ids=input_ids, labels=targets)
            loss = outputs.loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            total_loss += loss.item()
            
            # Clear intermediate variables
            del outputs
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"CUDA OOM at batch {batch_idx}, clearing cache and continuing...")
                torch.cuda.empty_cache()
                continue
            else:
                raise e
        
        # Update weights every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            
            # Clear cache after optimizer step
            torch.cuda.empty_cache()
            
            if step % 5 == 0:  # More frequent logging
                avg_loss = total_loss / step if step > 0 else 0
                current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
                gpu_memory_used = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                gpu_memory_max = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                
                # Calculate perplexity
                perplexity = torch.exp(loss * gradient_accumulation_steps)
                
                # Log training progress
                step_stats = {
                    'step': step,
                    'batch': batch_idx + 1,
                    'loss': loss.item() * gradient_accumulation_steps,  # Actual loss before scaling
                    'avg_loss': avg_loss,
                    'perplexity': float(perplexity),
                    'learning_rate': current_lr,
                    'gpu_memory_used_gb': gpu_memory_used,
                    'gpu_memory_max_gb': gpu_memory_max,
                    'timestamp': datetime.datetime.now().isoformat()
                }
                training_stats['training_progress'].append(step_stats)
                
                print(f"Step {step}, Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}, LR: {current_lr:.2e}")
                print(f"GPU Memory: {gpu_memory_used:.2f} GB / {gpu_memory_max:.2f} GB")
                
                # Update learning rate
                scheduler.step()
        
        num_batches += 1
        
        # Extended training for better results (matching MoR implementation)
        if num_batches >= 400:
            break
    
    end_time = time.time()
    training_time = end_time - start_time
    final_avg_loss = total_loss / step if step > 0 else 0
    
    # Calculate perplexity
    perplexity = torch.exp(torch.tensor(final_avg_loss)).item()
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Final average loss: {final_avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    
    # Test generation
    print("\n=== Testing Generation ===")
    model.eval()
    
    test_prompt = "The future of artificial intelligence"
    inputs = tokenizer.encode(test_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs, 
            max_length=inputs.shape[1] + 50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text}")
    
    # Compile final results
    training_stats['final_results'] = {
        'training_time_seconds': training_time,
        'total_steps': step,
        'total_batches': num_batches,
        'final_average_loss': final_avg_loss,
        'perplexity': perplexity,
        'generated_text_sample': generated_text,
        'training_completed': True,
        'completion_timestamp': datetime.datetime.now().isoformat()
    }
    
    # Save training statistics to JSON file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"standard_gemma_training_stats_{timestamp}.json"
    
    try:
        with open(json_filename, 'w') as f:
            json.dump(training_stats, f, indent=2, default=str)
        print(f"\n✅ Training statistics saved to: {json_filename}")
    except Exception as e:
        print(f"❌ Failed to save training statistics: {e}")
    
    return {
        'model': 'Standard Gemma 3 270M',
        'training_time': training_time,
        'final_loss': final_avg_loss,
        'perplexity': perplexity,
        'parameters': total_params,
        'generated_text': generated_text,
        'stats_file': json_filename
    }

if __name__ == "__main__":
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Train standard model
    results = train_standard_gemma()
    
    print("\n=== Standard Gemma Training Results ===")
    for key, value in results.items():
        if key != 'generated_text':
            print(f"{key}: {value}")