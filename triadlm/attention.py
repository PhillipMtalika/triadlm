"""Attention math isolated for unit testing (spec §3).

The model imports from here so mask semantics are tested in exactly one place.
"""
import math
import torch
from torch import Tensor


def causal_mask(t: int, device: torch.device | str = "cpu") -> Tensor:
    """Lower-triangular boolean mask (True = allowed to attend)."""
    return torch.tril(torch.ones(t, t, dtype=torch.bool, device=device))


def causal_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Manual scaled dot-product attention with an explicit causal mask.

    q/k/v: (B, nh, T, hs) -> (B, nh, T, hs). No dropout here; the Block
    handles regularization so this function is pure math for tests.
    """
    _, _, t, hs = q.shape
    scores: Tensor = q @ k.transpose(-2, -1) / math.sqrt(hs)
    allowed: Tensor = causal_mask(t, q.device)
    scores = scores.masked_fill(~allowed, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v
