# train_mor.py (DM-aligned 400M preset + 8GB VRAM guardrails)
import os
import json
import time
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2TokenizerFast
from mor_minimal import MoRLM, MoRConfig

# ----------------------------
# Simple text dataset
# ----------------------------
class SimpleTextDataset(Dataset):
    def __init__(self, text, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        tokens = tokenizer.encode(text)
        self.examples = []
        stride = max_length // 2
        for i in range(0, len(tokens) - max_length + 1, stride):
            chunk = tokens[i:i + max_length]
            self.examples.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


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


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----------------------------
    # 400M-class config (shared across recursions; R adds compute, not params)
    # With tied embeddings to keep total ~400M (otherwise ~450M).
    # Short seq_len to help 8GB VRAM with checkpointing.
    # ----------------------------
    config = MoRConfig(
        vocab_size=50257,
        d_model=1024,
        n_heads=16,
        d_ff=4096,
        n_layers_shared=28,      # 28 shared layers ~ 400M with tie-emb
        max_recursions=3,        # shared weights, so no extra params
        dropout=0.0,
        max_seq_len=256,         # keep small for memory
        kv_share_from_first=True,
        ponder_cost=0.01,
        use_checkpoint=False,  # Disable checkpointing to avoid tensor mismatch
        tie_embeddings=True,     # IMPORTANT to keep ~400M
    )

    model = MoRLM(config).to(device)
    params_m = count_params(model) / 1e6
    print(f"Model: {config.d_model}d, {config.n_heads}h, {config.d_ff}ff, {config.n_layers_shared}L, {config.max_recursions}R  ~{params_m:.1f}M params (tie_emb={config.tie_embeddings})")

    # ----------------------------
    # Training setup
    # ----------------------------
    max_iters = 2500
    eval_interval = 100
    early_stopping_patience = 15
    base_lr = 2e-4
    weight_decay = 0.01

    batch_size = 1
    grad_accum_steps = 16  # simulate larger batch; keep per-step memory tiny
    total_tokens_per_step = batch_size * config.max_seq_len * grad_accum_steps

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    text = load_wikipedia_dataset()
    dataset = SimpleTextDataset(text, tokenizer, max_length=config.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay, betas=(0.9, 0.95))

    # Mixed precision (prefer bf16 on Ada; fallback fp16)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(not use_bf16))

    # LR schedule: cosine decay with warmup
    warmup_steps = max(50, eval_interval)
    def get_lr(step):
        if step < warmup_steps:
            return base_lr * (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, (max_iters - warmup_steps)))
        return 0.1 * base_lr + 0.9 * base_lr * 0.5 * (1 + math.cos(math.pi * progress))

    train_losses, train_steps = [], []
    eval_losses, eval_steps = [], []
    best_eval_loss = float("inf")
    patience = 0

    step = 0
    model.train()
    start_time = time.time()

    while step < max_iters:
        for batch in dataloader:
            if step >= max_iters:
                break

            lr = get_lr(step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            batch = batch.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=True):
                logits, loss, aux = model(batch, targets=batch)
                loss = loss / grad_accum_steps

            if use_bf16:
                loss.backward()
            else:
                scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if use_bf16:
                    optimizer.step()
                else:
                    scaler.step(optimizer)
                    scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if step % 50 == 0:
                elapsed = time.time() - start_time
                mem_gb = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
                lm_loss = aux.get("loss_lm", loss)
                if isinstance(lm_loss, torch.Tensor):
                    lm_loss = lm_loss.item()
                ponder = aux.get("ponder", aux.get("expected_updates", 0.0))
                if isinstance(ponder, torch.Tensor):
                    ponder = ponder.item()
                print(f"step {step}/{max_iters}  lr {lr:.2e}  loss {loss.item()*grad_accum_steps:.3f}  lm {lm_loss:.3f}  ponder {ponder:.3f}  mem {mem_gb:.1f}GB  tok/step~{total_tokens_per_step}")

            train_losses.append(float(loss.item()*grad_accum_steps))
            train_steps.append(step)

            if step % eval_interval == 0 and step > 0:
                model.eval()
                eval_loss, nb = 0.0, 0
                with torch.no_grad():
                    for eval_batch in dataloader:
                        if nb >= 10:
                            break
                        eval_batch = eval_batch.to(device, non_blocking=True)
                        with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=True):
                            _, e_loss, _ = model(eval_batch, targets=eval_batch)
                        eval_loss += float(e_loss.item())
                        nb += 1
                eval_loss /= max(1, nb)
                eval_losses.append(eval_loss)
                eval_steps.append(step)
                print(f"Eval step {step}: loss {eval_loss:.4f}")

                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    patience = 0
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'config': config,
                        'step': step,
                        'loss': eval_loss,
                    }, 'best_mor_model.pt')
                else:
                    patience += 1
                    if patience >= early_stopping_patience:
                        print(f"Early stopping at step {step} (no improvement for {early_stopping_patience} evals)")
                        break
                model.train()

            step += 1

        if patience >= early_stopping_patience:
            break

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'step': step,
        'train_losses': train_losses,
        'train_steps': train_steps,
        'eval_losses': eval_losses,
        'eval_steps': eval_steps,
    }, 'final_mor_model.pt')

    training_data = {
        'train_losses': train_losses,
        'train_steps': train_steps,
        'eval_losses': eval_losses,
        'eval_steps': eval_steps,
        'config': {
            'd_model': config.d_model,
            'n_heads': config.n_heads,
            'd_ff': config.d_ff,
            'n_layers_shared': config.n_layers_shared,
            'max_recursions': config.max_recursions,
            'max_iters': max_iters,
            'param_count': count_params(model),
            'tie_embeddings': config.tie_embeddings,
            'max_seq_len': config.max_seq_len,
        }
    }

    with open('mor_training_data.json', 'w') as f:
        json.dump(training_data, f)

    print(f"Training completed! Final step: {step}")
    print(f"Best eval loss: {best_eval_loss:.4f}")
    print("Saved: final_mor_model.pt, best_mor_model.pt, mor_training_data.json")


if __name__ == "__main__":
    main()