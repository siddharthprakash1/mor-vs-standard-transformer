# mor_minimal.py
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ----------------------------
# Utilities
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
# Attention with active-token masking and optional KV-sharing
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

    def forward(
        self, x, attn_mask=None, kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, reuse_kv=False
    ):
        # x: [B, T, D]
        B, T, D = x.shape
        qkv = self.qkv(x)  # [B, T, 3D]
        q, k, v = qkv.split(D, dim=-1)
        q = rearrange(q, "b t (h d) -> b h t d", h=self.n_heads)
        k_new = rearrange(k, "b t (h d) -> b h t d", h=self.n_heads)
        v_new = rearrange(v, "b t (h d) -> b h t d", h=self.n_heads)

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            if reuse_kv:
                # KV sharing: reuse previous recursion's K,V rather than new ones for memory/prefill speed
                k_cat, v_cat = k_prev, v_prev
            else:
                # Standard cache extend/replace: for simplicity, we replace (same-length blocks across recursions)
                k_cat, v_cat = k_new, v_new
        else:
            k_cat, v_cat = k_new, v_new

        if self.flash:
            # attn_mask: [B, 1, T, T] with True where we should mask
            # For active-token attention, we build mask outside
            attn = F.scaled_dot_product_attention(
                q, k_cat, v_cat,
                attn_mask=None if attn_mask is None else (~attn_mask).to(q.dtype).masked_fill(attn_mask, float("-inf")),
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            scale = 1.0 / math.sqrt(self.d_head)
            scores = torch.einsum("bhtd,bhTd->bhtT", q, k_cat) * scale
            if attn_mask is not None:
                scores = scores.masked_fill(attn_mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            attn = torch.einsum("bhtT,bhTd->bhtd", attn, v_cat)

        y = rearrange(attn, "b h t d -> b t (h d)")
        y = self.out(y)
        return y, (k_cat, v_cat)

# ----------------------------
# Transformer block (shared)
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

    def forward(self, x, attn_mask=None, kv_cache=None, reuse_kv=False):
        a, kv_out = self.attn(self.norm1(x), attn_mask=attn_mask, kv_cache=kv_cache, reuse_kv=reuse_kv)
        x = x + a
        m = self.mlp(self.norm2(x))
        x = x + m
        return x, kv_out

# ----------------------------
# Lightweight token router (ACT-style halting)
# ----------------------------
class HaltingRouter(nn.Module):
    def __init__(self, d_model, epsilon=0.01):
        super().__init__()
        self.epsilon = epsilon
        self.proj = nn.Linear(d_model, 1)

    def forward(self, h, halting_prob, remainders, n_updates, still_running_mask):
        # h: [B, T, D]; returns p_t (prob to halt this step), updated accumulators
        p_t = torch.sigmoid(self.proj(h)).squeeze(-1)  # [B, T]
        # Apply still_running mask
        p_t = p_t * still_running_mask
        # New accumulators as in ACT
        new_halts = (halting_prob + p_t > 1 - self.epsilon).float()
        p_t_adjusted = torch.where(
            new_halts.bool(),
            (1 - self.epsilon - halting_prob).clamp(min=0.0),
            p_t
        )
        halting_prob = halting_prob + p_t_adjusted
        remainders = remainders + new_halts * (1 - halting_prob)
        n_updates = n_updates + (still_running_mask > 0).float()
        still_running_mask = still_running_mask * (1.0 - new_halts)
        return p_t_adjusted, halting_prob, remainders, n_updates, still_running_mask

# ----------------------------
# MoR: shared stack used across recursion steps with token-level halting
# ----------------------------
@dataclass
class MoRConfig:
    vocab_size: int
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    n_layers_shared: int = 6
    max_recursions: int = 2
    dropout: float = 0.0
    max_seq_len: int = 512
    kv_share_from_first: bool = False
    ponder_cost: float = 0.01  # encourages early halting

class MoRLM(nn.Module):
    def __init__(self, cfg: MoRConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        # Shared stack
        self.blocks = nn.ModuleList([TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
                                     for _ in range(cfg.n_layers_shared)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.router = HaltingRouter(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.register_buffer("causal_mask", torch.triu(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool), 1))

    def forward(self, idx, targets=None):
        # idx: [B, T]
        B, T = idx.shape
        device = idx.device
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=device))[None, :, :]

        # ACT accumulators
        halting_prob = torch.zeros(B, T, device=device)
        remainders = torch.zeros(B, T, device=device)
        n_updates = torch.zeros(B, T, device=device)
        still_running = torch.ones(B, T, device=device)

        # Attention mask base (causal)
        base_attn_mask = self.causal_mask[:T, :T][None, None, :, :]  # [1,1,T,T]

        # Optional KV sharing: capture first recursion K,V per block
        first_recursion_kv = [None] * len(self.blocks)

        # Hidden state across recursions
        h = x

        for r in range(self.cfg.max_recursions):
            # Compute routing probabilities for tokens still running
            p_t, halting_prob, remainders, n_updates, still_running = self.router(
                h, halting_prob, remainders, n_updates, still_running
            )

            # Build active-token attention mask: tokens that have halted should not attend or be attended
            active = (still_running > 0).float()  # [B, T]
            # Create a mask that masks any position where either query or key is inactive
            # attn_mask True => mask out
            # We combine causal mask with activeness
            q_active = active[:, None, :, None]  # [B,1,T,1]
            k_active = active[:, None, None, :]  # [B,1,1,T]
            active_pair = (q_active * k_active) > 0.5
            attn_mask = (~active_pair) | base_attn_mask  # combine with causal

            # Pass through shared stack; maintain per-layer KV cache; optionally reuse first recursion KV
            kv_caches = [None] * len(self.blocks)
            for i, block in enumerate(self.blocks):
                reuse_kv = (self.cfg.kv_share_from_first and r > 0 and first_recursion_kv[i] is not None)
                kv_in = first_recursion_kv[i] if (self.cfg.kv_share_from_first and r > 0) else None
                h, kv_out = block(h, attn_mask=attn_mask, kv_cache=kv_in, reuse_kv=reuse_kv)
                kv_caches[i] = kv_out
                if r == 0 and first_recursion_kv[i] is None:
                    first_recursion_kv[i] = kv_out

            # Mix the new h with previous h based on continuing tokens vs halted tokens.
            # Tokens that halted at this step should keep their state; continuing tokens update.
            continue_mask = (still_running > 0).float().unsqueeze(-1)
            h = continue_mask * h + (1 - continue_mask) * h.detach()  # keep halted tokens stable

            # Early exit if all tokens halted
            if still_running.sum() == 0:
                break

        h = self.norm_f(h)
        logits = self.lm_head(h)  # [B, T, V]

        loss = None
        aux = {}
        if targets is not None:
            loss_lm = F.cross_entropy(logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                                      targets[:, 1:].contiguous().view(-1))
            # Ponder loss: encourage fewer recursions
            expected_updates = (n_updates + remainders).mean()
            loss = loss_lm + self.cfg.ponder_cost * expected_updates
            aux = {"loss_lm": loss_lm.detach(), "expected_updates": expected_updates.detach()}
        else:
            # For inference, still provide expected_updates in aux
            expected_updates = (n_updates + remainders).mean()
            aux = {"expected_updates": expected_updates.detach()}
            
        return logits, loss, aux