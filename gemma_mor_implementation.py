import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import time
import math
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

class MoRConfig:
    def __init__(self):
        # MoR Configuration matching Gemma dimensions - further optimized for 8GB GPU
        self.hidden_size = 2304  # Gemma 2B hidden size
        self.num_shared_layers = 2  # Further reduced for memory efficiency
        self.num_routing_tokens = 4  # Reduced from 8
        self.routing_temperature = 1.0
        self.kv_cache_size = 512  # Reduced from 1024

class MoRRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.routing_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, config.num_routing_tokens),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, hidden_states):
        # hidden_states: [batch_size, seq_len, hidden_size]
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Compute routing weights for each token
        routing_weights = self.routing_mlp(hidden_states)  # [batch_size, seq_len, num_routing_tokens]
        
        # Apply temperature scaling
        routing_weights = routing_weights / self.config.routing_temperature
        routing_weights = torch.softmax(routing_weights, dim=-1)
        
        return routing_weights

class MoRKVCache(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cache_size = config.kv_cache_size
        
        # Initialize cache storage
        self.register_buffer('key_cache', torch.zeros(config.num_routing_tokens, config.kv_cache_size, config.hidden_size))
        self.register_buffer('value_cache', torch.zeros(config.num_routing_tokens, config.kv_cache_size, config.hidden_size))
        self.register_buffer('cache_positions', torch.zeros(config.num_routing_tokens, dtype=torch.long))
    
    def update_cache(self, keys, values, routing_weights):
        # keys, values: [batch_size, seq_len, hidden_size]
        # routing_weights: [batch_size, seq_len, num_routing_tokens]
        
        batch_size, seq_len, hidden_size = keys.shape
        
        # Detach inputs to prevent gradient flow through cache
        keys = keys.detach()
        values = values.detach()
        routing_weights = routing_weights.detach()
        
        for token_idx in range(self.config.num_routing_tokens):
            # Get the weighted keys and values for this routing token
            token_weights = routing_weights[:, :, token_idx].unsqueeze(-1)  # [batch_size, seq_len, 1]
            
            weighted_keys = keys * token_weights  # [batch_size, seq_len, hidden_size]
            weighted_values = values * token_weights
            
            # Average across batch and sequence dimensions
            avg_key = weighted_keys.mean(dim=(0, 1))  # [hidden_size]
            avg_value = weighted_values.mean(dim=(0, 1))  # [hidden_size]
            
            # Update cache (in-place operations on buffers are safe after detach)
            pos = self.cache_positions[token_idx] % self.cache_size
            self.key_cache[token_idx, pos] = avg_key
            self.value_cache[token_idx, pos] = avg_value
            self.cache_positions[token_idx] += 1
    
    def retrieve_from_cache(self, routing_weights):
        # routing_weights: [batch_size, seq_len, num_routing_tokens]
        batch_size, seq_len, num_tokens = routing_weights.shape
        
        retrieved_keys = torch.zeros(batch_size, seq_len, self.config.hidden_size, device=routing_weights.device)
        retrieved_values = torch.zeros(batch_size, seq_len, self.config.hidden_size, device=routing_weights.device)
        
        for token_idx in range(num_tokens):
            # Get cached keys and values for this token
            cached_keys = self.key_cache[token_idx]  # [cache_size, hidden_size]
            cached_values = self.value_cache[token_idx]  # [cache_size, hidden_size]
            
            # Weight by routing probability
            token_weights = routing_weights[:, :, token_idx].unsqueeze(-1)  # [batch_size, seq_len, 1]
            
            # Simple retrieval: use the most recent cached entry
            recent_key = cached_keys[self.cache_positions[token_idx] % self.cache_size]  # [hidden_size]
            recent_value = cached_values[self.cache_positions[token_idx] % self.cache_size]  # [hidden_size]
            
            # Add to retrieved tensors
            retrieved_keys += token_weights * recent_key.unsqueeze(0).unsqueeze(0)
            retrieved_values += token_weights * recent_value.unsqueeze(0).unsqueeze(0)
        
        return retrieved_keys, retrieved_values
    
    def reset_cache(self):
        """Reset cache between batches to prevent gradient accumulation"""
        self.key_cache.zero_()
        self.value_cache.zero_()
        self.cache_positions.zero_()

class MoRLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.router = MoRRouter(config)
        self.kv_cache = MoRKVCache(config)
        
        # Shared computation layers
        self.shared_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=8,
                dim_feedforward=config.hidden_size * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(config.num_shared_layers)
        ])
    
    def forward(self, hidden_states):
        # Get routing weights
        routing_weights = self.router(hidden_states)
        
        # Apply shared layers
        for layer in self.shared_layers:
            hidden_states = layer(hidden_states)
        
        # Update KV cache
        keys = hidden_states
        values = hidden_states
        self.kv_cache.update_cache(keys, values, routing_weights)
        
        # Retrieve from cache and combine
        retrieved_keys, retrieved_values = self.kv_cache.retrieve_from_cache(routing_weights)
        
        # Combine current hidden states with retrieved information
        enhanced_states = hidden_states + 0.1 * (retrieved_keys + retrieved_values)
        
        return enhanced_states

class GemmaMoRModel(nn.Module):
    def __init__(self, base_model, config):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.mor_layer = MoRLayer(config)
        
        # Get the language modeling head from base model
        self.lm_head = base_model.lm_head
        
        # Move MoR layer to the same device as base model
        self._device = next(base_model.parameters()).device
        self.mor_layer = self.mor_layer.to(self._device)
        
        # Freeze base model parameters to prevent gradient computation
        for param in self.base_model.parameters():
            param.requires_grad = False
        
    def forward(self, input_ids, labels=None):
        # Get embeddings from base model for MoR processing
        # Detach to prevent gradient flow through base model
        with torch.no_grad():
            embeddings = self.base_model.model.embed_tokens(input_ids).detach()
        
        # Apply MoR processing to embeddings (this will have gradients)
        enhanced_embeddings = self.mor_layer(embeddings)
        
        # For training, we'll use a simplified approach:
        # Just pass the MoR-enhanced embeddings through the language head
        # This avoids the complex layer-by-layer processing that was causing issues
        
        # Apply layer norm (detach to prevent gradient flow)
        with torch.no_grad():
            normalized_embeddings = self.base_model.model.norm(enhanced_embeddings.detach()).detach()
        
        # Re-attach gradients for the final computation
        normalized_embeddings = normalized_embeddings + enhanced_embeddings - enhanced_embeddings.detach()
        
        # Get logits using the MoR-enhanced embeddings
        logits = self.lm_head(normalized_embeddings)
        
        loss = None
        if labels is not None:
            # Shift labels for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return type('ModelOutput', (), {'loss': loss, 'logits': logits})()

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

def train_gemma_mor():
    """Train Gemma with MoR implementation"""
    print("=== Training Gemma 3 270M with MoR ===")
    
    # Initialize training statistics tracker
    training_stats = {
        'experiment_info': {
            'model_name': 'Gemma 3 270M with MoR',
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
    
    # Load base model
    model_name = "google/gemma-2-2b-it"
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ["HF_TOKEN"])
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        token=os.environ["HF_TOKEN"],
        torch_dtype=torch.float32,
        device_map="auto"
    )
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create MoR configuration and model
    mor_config = MoRConfig()
    model = GemmaMoRModel(base_model, mor_config)
    
    # Add model configuration to stats
    training_stats['model_config'] = {
        'hidden_size': mor_config.hidden_size,
        'num_shared_layers': mor_config.num_shared_layers,
        'num_routing_tokens': mor_config.num_routing_tokens,
        'routing_temperature': mor_config.routing_temperature,
        'kv_cache_size': mor_config.kv_cache_size,
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.mor_layer.parameters() if p.requires_grad)
    }
    
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
    
    # Setup optimizer - only train MoR parameters
    mor_parameters = [p for p in model.mor_layer.parameters() if p.requires_grad]
    print(f"Training {len(mor_parameters)} MoR parameters")
    optimizer = optim.AdamW(mor_parameters, lr=1e-4)  # Slightly increased LR for longer training
    
    # Add learning rate scheduler for longer training
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=100)
    
    # Training parameters - optimized for 8GB GPU
    num_epochs = 1
    gradient_accumulation_steps = 8  # Increased to maintain effective batch size
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model.mor_layer, 'gradient_checkpointing_enable'):
        model.mor_layer.gradient_checkpointing_enable()
    
    model.train()
    total_loss = 0
    step = 0
    
    print(f"Starting MoR training for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (input_ids, targets) in enumerate(dataloader):
            try:
                input_ids = input_ids.to(next(model.parameters()).device)
                targets = targets.to(next(model.parameters()).device)
                
                # Reset MoR cache between batches to prevent gradient accumulation
                model.mor_layer.kv_cache.reset_cache()
                
                # Clear cache before forward pass
                torch.cuda.empty_cache()
                
                # Forward pass
                outputs = model(input_ids=input_ids, labels=targets)
                loss = outputs.loss / gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                total_loss += loss.item()
                epoch_loss += loss.item()
                
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
                    
                    # Log training progress
                    step_stats = {
                        'step': step,
                        'epoch': epoch + 1,
                        'batch': batch_idx + 1,
                        'loss': loss.item() * gradient_accumulation_steps,  # Actual loss before scaling
                        'avg_loss': avg_loss,
                        'learning_rate': current_lr,
                        'gpu_memory_used_gb': gpu_memory_used,
                        'gpu_memory_max_gb': gpu_memory_max,
                        'timestamp': datetime.datetime.now().isoformat()
                    }
                    training_stats['training_progress'].append(step_stats)
                    
                    print(f"Step {step}, Average Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")
                    print(f"GPU Memory: {gpu_memory_used:.2f} GB / {gpu_memory_max:.2f} GB")
                    
                    # Update learning rate
                    scheduler.step()
            
            num_batches += 1
            
            # Extended training for better results (based on research showing longer training improves performance)
            if num_batches >= 400:  # Increased from 50 based on research recommendations
                break
        
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_epoch_loss:.4f}")
    
    end_time = time.time()
    training_time = end_time - start_time
    final_avg_loss = total_loss / step if step > 0 else 0
    
    # Calculate perplexity
    perplexity = torch.exp(torch.tensor(final_avg_loss)).item()
    
    print(f"MoR training completed in {training_time:.2f} seconds")
    print(f"Final average loss: {final_avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    
    # Test generation
    print("\n=== Testing MoR Generation ===")
    model.eval()
    
    test_prompt = "The future of artificial intelligence"
    inputs = tokenizer.encode(test_prompt, return_tensors="pt").to(next(model.parameters()).device)
    
    with torch.no_grad():
        # Simple generation (since we have custom model)
        generated_ids = inputs.clone()
        max_new_tokens = 50
        
        for _ in range(max_new_tokens):
            outputs = model(generated_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), 1)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
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
    json_filename = f"gemma_mor_training_stats_{timestamp}.json"
    
    try:
        with open(json_filename, 'w') as f:
            json.dump(training_stats, f, indent=2, default=str)
        print(f"\n✅ Training statistics saved to: {json_filename}")
    except Exception as e:
        print(f"❌ Failed to save training statistics: {e}")
    
    return {
        'model': 'Gemma 3 with MoR',
        'training_time': training_time,
        'final_loss': final_avg_loss,
        'perplexity': perplexity,
        'parameters': sum(p.numel() for p in model.parameters()),
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
    
    # Train MoR model
    results = train_gemma_mor()
    
    print("\n=== MoR Training Results ===")
    for key, value in results.items():
        if key != 'generated_text':
            print(f"{key}: {value}")