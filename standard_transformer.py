# standard_transformer.py
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ----------------------------
# Utilities (same as MoR)
# ----------------------------
def rms_norm(x, weight, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight

class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return rms_norm(x, self.weight)

# ----------------------------
# Standard Multi-Head Self-Attention
# ----------------------------
class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, flash=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.flash = flash and hasattr(F, "scaled_dot_product_attention")

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        qkv = self.qkv(x)  # [B, T, 3D]
        q, k, v = qkv.split(D, dim=-1)
        q = rearrange(q, "b t (h d) -> b h t d", h=self.n_heads)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.n_heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.n_heads)

        if self.flash:
            attn = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            scale = 1.0 / math.sqrt(self.d_head)
            scores = torch.einsum("bhtd,bhTd->bhtT", q, k) * scale
            # Apply causal mask
            causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
            scores = scores.masked_fill(causal_mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            attn = torch.einsum("bhtT,bhTd->bhtd", attn, v)

        y = rearrange(attn, "b h t d -> b t (h d)")
        y = self.out(y)
        return y

# ----------------------------
# Standard Transformer Block
# ----------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiheadSelfAttention(d_model, n_heads, dropout=dropout)
        self.norm2 = RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pre-norm architecture (same as MoR)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# ----------------------------
# Standard Transformer Language Model
# ----------------------------
@dataclass
class StandardTransformerConfig:
    vocab_size: int
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    n_layers: int = 12  # Standard depth (equivalent to MoR's shared_layers * max_recursions)
    dropout: float = 0.0
    max_seq_len: int = 512

class StandardTransformerLM(nn.Module):
    def __init__(self, cfg: StandardTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        
        # Standard transformer layers (no sharing)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        # idx: [B, T]
        B, T = idx.shape
        device = idx.device
        
        # Token + positional embeddings
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=device))[None, :, :]

        # Pass through all transformer blocks sequentially
        for block in self.blocks:
            x = block(x)

        # Final norm and projection
        x = self.norm_f(x)
        logits = self.lm_head(x)  # [B, T, V]

        loss = None
        aux = {}
        if targets is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                targets[:, 1:].contiguous().view(-1)
            )
            aux = {"loss_lm": loss.detach()}
        
        return logits, loss, aux

    def count_parameters(self):
        """Count total parameters for comparison"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ----------------------------
# Helper function to create comparable configs
# ----------------------------
def create_comparable_configs(vocab_size=50257):
    """
    Create MoR and Standard configs with similar parameter counts
    """
    # MoR config (from your minimal version)
    mor_config = {
        'vocab_size': vocab_size,
        'd_model': 512,
        'n_heads': 8,
        'd_ff': 2048,
        'n_layers_shared': 6,
        'max_recursions': 2,
        'dropout': 0.0,
        'max_seq_len': 512,
    }
    
    # Standard transformer with equivalent depth
    # MoR effective depth: 6 layers * 2 recursions = 12 layers
    standard_config = StandardTransformerConfig(
        vocab_size=vocab_size,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        n_layers=12,  # Match MoR's effective depth
        dropout=0.0,
        max_seq_len=512,
    )
    
    return mor_config, standard_config

def create_400m_comparable_configs(vocab_size=50257):
    """
    Create configs comparable to your 400M MoR model
    """
    # 400M MoR config
    mor_400m_config = {
        'vocab_size': vocab_size,
        'd_model': 1024,
        'n_heads': 16,
        'd_ff': 4096,
        'n_layers_shared': 12,
        'max_recursions': 3,
        'dropout': 0.1,
        'max_seq_len': 1024,
    }
    
    # Standard transformer with equivalent depth
    # MoR effective depth: 12 layers * 3 recursions = 36 layers
    standard_400m_config = StandardTransformerConfig(
        vocab_size=vocab_size,
        d_model=1024,
        n_heads=16,
        d_ff=4096,
        n_layers=36,  # Match MoR's effective depth
        dropout=0.1,
        max_seq_len=1024,
    )
    
    return mor_400m_config, standard_400m_config

# ----------------------------
# Example usage and comparison
# ----------------------------
if __name__ == "__main__":
    # Create comparable models
    mor_cfg, std_cfg = create_comparable_configs()
    
    # Standard transformer
    std_model = StandardTransformerLM(std_cfg)
    print(f"Standard Transformer parameters: {std_model.count_parameters():,}")
    
    # Test forward pass
    batch_size, seq_len = 2, 128
    dummy_input = torch.randint(0, std_cfg.vocab_size, (batch_size, seq_len))
    dummy_targets = torch.randint(0, std_cfg.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        logits, loss, aux = std_model(dummy_input, dummy_targets)
        print(f"Output shape: {logits.shape}")
        print(f"Loss: {loss.item():.4f}")