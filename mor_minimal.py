import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ----------------------------
# Norms
# ----------------------------

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight


class RMSNorm(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight)


# ----------------------------
# Multi-Head Self Attention with optional KV sharing across recursions
# ----------------------------
class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, flash: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        # Separate projections so we can reuse K/V across recursions (compute Q only)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.flash = flash and hasattr(F, "scaled_dot_product_attention")

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, D] -> [B, H, T, Dh]
        return rearrange(x, "b t (h d) -> b h t d", h=self.n_heads)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,  # boolean mask broadcastable to [B, 1, T, S]
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        reuse_kv: bool = False,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x: [B, T, D]
        B, T, D = x.shape
        q = self._shape(self.q_proj(x))  # [B, H, T, Dh]

        if kv_cache is not None and reuse_kv:
            k, v = kv_cache
        else:
            kv = self.kv_proj(x)
            k_lin, v_lin = kv.split(D, dim=-1)
            k, v = self._shape(k_lin), self._shape(v_lin)

        if self.flash:
            # Flash attention: use causal if no explicit mask, otherwise use explicit mask
            if attn_mask is None:
                attn = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=True,
                )
            else:
                # Combine causal mask with explicit mask
                causal_mask = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
                combined_mask = attn_mask | ~causal_mask[None, None, :, :]
                attn = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=combined_mask,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=False,
                )
        else:
            scale = 1.0 / math.sqrt(self.d_head)
            scores = torch.einsum("bhtd,bhTd->bhtT", q, k) * scale
            if attn_mask is not None:
                # attn_mask True => mask out
                scores = scores.masked_fill(attn_mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            attn = torch.einsum("bhtT,bhTd->bhtd", attn, v)

        y = rearrange(attn, "b h t d -> b t (h d)")
        y = self.out(y)
        return y, (k, v)


# ----------------------------
# Transformer block (shared across recursions)
# ----------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiheadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        reuse_kv: bool = False,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        a, kv_out = self.attn(self.norm1(x), attn_mask=attn_mask, kv_cache=kv_cache, reuse_kv=reuse_kv)
        x = x + a
        m = self.mlp(self.norm2(x))
        x = x + m
        return x, kv_out


# ----------------------------
# Lightweight token router (ACT-style halting)
# ----------------------------
class HaltingRouter(nn.Module):
    def __init__(self, d_model: int, epsilon: float = 0.01):
        super().__init__()
        self.epsilon = epsilon
        self.proj = nn.Linear(d_model, 1)

    def forward(
        self,
        h: torch.Tensor,
        halting_prob: torch.Tensor,
        still_running_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # h: [B, T, D]; returns p_t_adjusted and updated accumulators
        p = torch.sigmoid(self.proj(h)).squeeze(-1)  # [B, T]
        p = p * still_running_mask  # zero out halted tokens

        # Decide which tokens will halt this step (ACT)
        new_halts = (halting_prob + p > 1.0 - self.epsilon).float()
        p_t = torch.where(new_halts.bool(), (1.0 - self.epsilon - halting_prob).clamp(min=0.0), p)

        # Update accumulators
        halting_prob = halting_prob + p_t
        n_updates_inc = still_running_mask  # tokens that were running this step
        still_running_mask = still_running_mask * (1.0 - new_halts)
        return p_t, halting_prob, still_running_mask, n_updates_inc


# ----------------------------
# MoR Language Model
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
    kv_share_from_first: bool = True
    ponder_cost: float = 0.01  # encourages early halting
    use_checkpoint: bool = True
    tie_embeddings: bool = True  # tie lm_head to tok_emb to control params


class MoRLM(nn.Module):
    def __init__(self, cfg: MoRConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers_shared)
        ])
        self.norm_f = RMSNorm(cfg.d_model)
        self.router = HaltingRouter(cfg.d_model)
        # If not tying, create a separate head. If tying, we'll use F.linear with tok_emb.weight.
        if not cfg.tie_embeddings:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        else:
            self.lm_head = None

    def _build_query_mask(self, still_running_q: torch.Tensor, T: int) -> torch.Tensor:
        # still_running_q: [B, T] float in {0,1}
        # Build a boolean mask where True means mask out the entire row for halted queries.
        # Shape -> [B, 1, T, T], broadcast over heads.
        q_mask_rows = (still_running_q == 0)  # [B, T] True where halted
        attn_mask = q_mask_rows[:, None, :, None].expand(-1, 1, -1, T)  # [B,1,T,T]
        return attn_mask

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.shape
        device = idx.device
        assert T <= self.cfg.max_seq_len, "Sequence length exceeds configured max_seq_len"

        # Embeddings
        pos = torch.arange(T, device=device)
        h = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]

        # ACT accumulators
        halting_prob = torch.zeros(B, T, device=device)
        still_running = torch.ones(B, T, device=device)
        expected_updates = torch.zeros((), device=device)
        n_updates = torch.zeros(B, T, device=device)

        # Optional KV share: capture K/V from first recursion per layer
        first_kv: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * len(self.blocks)

        # Accumulate outputs per token
        y_accum = torch.zeros(B, T, self.cfg.d_model, device=device, dtype=h.dtype)

        # Training path through recursions
        for r in range(self.cfg.max_recursions):
            # Mask queries of tokens that already halted BEFORE this recursion
            attn_mask = self._build_query_mask(still_running, T)

            # Shared stack forward; reuse KV from first recursion if enabled
            kv_outs: List[Tuple[torch.Tensor, torch.Tensor]] = []
            h_rec = h
            for i, block in enumerate(self.blocks):
                reuse_kv = (self.cfg.kv_share_from_first and r > 0 and first_kv[i] is not None)
                kv_in = first_kv[i] if (self.cfg.kv_share_from_first and r > 0) else None

                if self.cfg.use_checkpoint and self.training:
                    def run_block(x):
                        out, kv = block(x, attn_mask=attn_mask, kv_cache=kv_in, reuse_kv=reuse_kv)
                        # Checkpoint can't return non-Tensor; stash kv via closure list
                        kv_outs.append(kv)
                        return out
                    h_rec = torch.utils.checkpoint.checkpoint(run_block, h_rec, use_reentrant=False)
                else:
                    h_rec, kv = block(h_rec, attn_mask=attn_mask, kv_cache=kv_in, reuse_kv=reuse_kv)
                    kv_outs.append(kv)

                if r == 0 and first_kv[i] is None:
                    first_kv[i] = kv_outs[-1]

            h = h_rec  # post-recursion hidden state

            # Route and accumulate outputs (ACT). Route on post-step state.
            p_t, halting_prob, still_running, n_updates_inc = self.router(h, halting_prob, still_running)
            y_accum = y_accum + p_t.unsqueeze(-1) * h
            n_updates = n_updates + n_updates_inc
            expected_updates = expected_updates + n_updates_inc.mean()

            # Early exit if all tokens halted
            if still_running.sum() == 0:
                break

        # If some tokens didn't reach threshold, allocate their remaining mass to the last state
        remainder = (1.0 - halting_prob).clamp(min=0.0)
        if (remainder > 0).any():
            y_accum = y_accum + remainder.unsqueeze(-1) * h

        # Final norm, head, and loss
        h_final = self.norm_f(y_accum)
        if self.cfg.tie_embeddings:
            logits = F.linear(h_final, self.tok_emb.weight)  # tied weights
        else:
            logits = self.lm_head(h_final)

        loss = None
        aux = {"expected_updates": (expected_updates / max(1, self.cfg.max_recursions)).detach()}
        if targets is not None:
            # Shifted LM loss
            B, T, V = logits.shape
            loss_lm = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, V),
                targets[:, 1:].contiguous().view(-1),
            )
            ponder = (n_updates + remainder).mean()
            loss = loss_lm + self.cfg.ponder_cost * ponder
            aux.update({"loss_lm": loss_lm.detach(), "ponder": ponder.detach()})

        return logits, loss, aux