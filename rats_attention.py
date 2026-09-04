"""Reusable register-attention primitive used by recurrent vision models."""

from contextlib import nullcontext

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


class RATSAttention(nn.Module):
    """Shared QKV with identity register projections in refine and broadcast."""

    STAGES = ("compress", "refine", "broadcast")

    def __init__(self, dim, heads=6, sdpa_backend="flash"):
        super().__init__()
        self.dim = int(dim)
        self.heads = int(heads)
        self.sdpa_backend = str(sdpa_backend).lower()
        if self.dim < 1 or self.heads < 1 or self.dim % self.heads:
            raise ValueError(
                f"dim={self.dim} must be positive and divisible by heads={self.heads}"
            )
        if self.sdpa_backend not in {"flash", "auto"}:
            raise ValueError("sdpa_backend must be flash or auto")
        self.head_dim = self.dim // self.heads
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.out_proj = nn.ModuleDict(
            {stage: nn.Linear(self.dim, self.dim) for stage in self.STAGES}
        )

    def _split_heads(self, value):
        if value.ndim != 3 or value.size(-1) != self.dim:
            raise ValueError(
                f"Expected a (batch, length, {self.dim}) tensor, got {tuple(value.shape)}"
            )
        batch, length, _ = value.shape
        return value.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, stage, query, key, value, return_weights=False):
        if stage == "compress":
            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)
        elif stage == "refine":
            pass
        elif stage == "broadcast":
            query = self.q_proj(query)
        else:
            raise ValueError(f"Unsupported RATS attention stage: {stage}")

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)
        if return_weights:
            scores = query @ key.transpose(-2, -1) / self.head_dim**0.5
            head_weights = scores.softmax(dim=-1)
            context = head_weights @ value
            weights = head_weights.mean(dim=1)
        else:
            context_manager = (
                sdpa_kernel(SDPBackend.FLASH_ATTENTION)
                if query.device.type == "cuda" and self.sdpa_backend == "flash"
                else nullcontext()
            )
            with context_manager:
                context = F.scaled_dot_product_attention(query, key, value)
            weights = None
        context = context.transpose(1, 2).contiguous().reshape(
            query.size(0), query.size(2), self.dim
        )
        output = self.out_proj[stage](context)
        return (output, weights) if return_weights else output
