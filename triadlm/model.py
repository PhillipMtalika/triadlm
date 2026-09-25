"""Shared decoder-only Transformer LM (spec §3)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch import Tensor


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    block_size: int = 512
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1
    bias: bool = False
    tie_weights: bool = True
    pad_id: int = 2


class CausalSelfAttention(nn.Module):
    """(B, T, C) -> (B, T, C) with an explicit causal mask."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x).view(B, T, 3, self.n_head, C // self.n_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.attn_drop.p if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Pre-norm: x = x + attn(ln1(x)); x = x + mlp(ln2(x))."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_weights:
            self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self) -> int:
        """Total parameter count (tied embeddings counted once by PyTorch)."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        """idx: (B, T) int64 -> (logits (B, T, vocab), loss or None)."""
        B, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(f"T={T} > block_size={self.config.block_size}")
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=self.config.pad_id)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int | None = None) -> Tensor:
        """Sample autoregressively; temperature<=0 means greedy (deterministic)."""
        self.eval()
        greedy = temperature <= 0
        for _ in range(max_new_tokens):
            idx_in = idx[:, -self.config.block_size:]
            logits, _ = self(idx_in)
            logits = logits[:, -1, :]
            if greedy:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-6)
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

    def configure_optimizers(self, weight_decay: float, lr: float,
                             betas: tuple[float, float]) -> torch.optim.Optimizer:
        """AdamW with decay on 2D weights only (no decay on biases/LayerNorm/embeddings)."""
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() >= 2 and not n.endswith("bias") and "ln" not in n and "wte" not in n and "wpe" not in n:
                decay.append(p)
            else:
                no_decay.append(p)
        return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                                  {"params": no_decay, "weight_decay": 0.0}],
                                 lr=lr, betas=betas)


def config_from_dict(d: dict) -> GPTConfig:
    """Build GPTConfig from a YAML `model:` mapping (config-driven, no hardcoded hparams)."""
    return GPTConfig(
        vocab_size=int(d.get("vocab_size", 8192)),
        block_size=int(d.get("block_size", 512)),
        n_layer=int(d.get("n_layer", 8)),
        n_head=int(d.get("n_head", 8)),
        n_embd=int(d.get("n_embd", 512)),
        dropout=float(d.get("dropout", 0.1)),
        bias=bool(d.get("bias", False)),
        tie_weights=bool(d.get("tie_weights", True)),
    )


def estimate_params(n_layer: int, n_head: int, n_embd: int,
                    vocab_size: int, block_size: int) -> int:
    """params ≈ 12 * n_layer * n_embd^2 + embeddings (spec §3 guide)."""
    _ = (n_head, block_size)
    return 12 * n_layer * n_embd * n_embd + vocab_size * n_embd
