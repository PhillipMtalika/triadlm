"""Pretraining loop (spec §5). Plain PyTorch, no orchestration framework.

CLI: python -m triadlm.train --config configs/base_50m.yaml --resume <ckpt|none>
"""
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import PackedDataset, load_manifest
from .model import GPT, config_from_dict
from .tokenizer import TriadTokenizer

REGISTRY_PATH = "experiments/registry.csv"
REGISTRY_FIELDS: list[str] = [
    "run_id", "variant", "checkpoint", "config_hash", "param_count",
    "context_length", "training_tokens", "wall_clock_seconds", "hardware",
    "dataset_version", "language_mixture", "val_loss", "perplexity",
    "instruction_following", "factuality", "citation_accuracy",
    "over_refusal_rate", "refusal_precision", "chichewa_gap",
    "latency_p50_ms", "latency_p95_ms", "tokens_per_second", "timestamp",
]


def config_hash(cfg: dict) -> str:
    """Short hash of the canonical config JSON."""
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def seed_all(seed: int) -> None:
    """Seed python/numpy/torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> dict:
    """Capture RNG state for checkpoint resume-reproducibility."""
    return {"python": random.getstate()[1][:8],
            "numpy_state": np.random.get_state()[1][:8].tolist(),
            "torch": torch.get_rng_state().tolist()[:8],
            "torch_full": [x for x in torch.get_rng_state().tolist()]}


def _full_rng() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_full_rng(s: dict) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def git_commit() -> str:
    """Current commit hash, or 'nogit' outside a repo."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "nogit"


def log_registry_row(path: str, row: dict) -> str:
    """Append one row to the experiment registry (never hand-edit the file)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    run_id = row.get("run_id") or (datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
                                   + "-" + os.urandom(2).hex())
    row = {**row, "run_id": run_id,
           "timestamp": datetime.now(timezone.utc).isoformat()}
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in REGISTRY_FIELDS})
    return run_id


def cosine_lr(step: int, max_steps: int, lr: float, min_lr: float, warmup: int) -> float:
    """Linear warmup + cosine decay, implemented directly (inspectable)."""
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * min(1.0, prog)))


class _Batcher:
    """Deterministic batcher: order + cursor are checkpointed so a resumed
    run sees the exact same batch sequence (resume-reproducibility test)."""

    def __init__(self, ds: PackedDataset, batch_size: int, seed: int) -> None:
        self.ds = ds
        self.bs = batch_size
        self.g = torch.Generator().manual_seed(seed)
        self.order: list[int] = []
        self.cursor = 0

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the next batch, extending the shuffled order as needed."""
        while self.cursor + self.bs > len(self.order):
            self.order += torch.randperm(len(self.ds), generator=self.g).tolist()
        idx = self.order[self.cursor:self.cursor + self.bs]
        self.cursor += self.bs
        xs, ys = zip(*[self.ds[i] for i in idx])
        return torch.stack(list(xs)), torch.stack(list(ys))

    def state(self) -> dict:
        """Checkpointable sampler state."""
        return {"gen": self.g.get_state(), "order": self.order,
                "cursor": self.cursor}

    def load(self, s: dict) -> None:
        """Restore sampler state."""
        self.g.set_state(s["gen"])
        self.order = s["order"]
        self.cursor = s["cursor"]


def save_checkpoint(path: str, model: GPT, optimizer: torch.optim.Optimizer,
                    cfg: dict, step: int, tokens_seen: int, t0: float,
                    sampler: dict | None = None) -> None:
    """Save state + `checkpoint.json` sidecar (every artifact gets one)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": cfg, "step": step, "rng": _full_rng(),
                "tokens_seen": tokens_seen,
                "sampler": sampler}, path)
    with open(path.replace(".pt", ".json"), "w") as f:
        json.dump({"commit": git_commit(), "config_hash": config_hash(cfg),
                   "param_count": model.num_params(), "step": step,
                   "tokens_seen": tokens_seen,
                   "wall_clock_seconds": round(time.time() - t0, 1)}, f, indent=2)


def load_checkpoint(path: str, model: GPT,
                    optimizer: torch.optim.Optimizer | None = None) -> dict:
    """Load state + restore RNG; returns the checkpoint dict."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    try:
        _set_full_rng(ckpt["rng"])
    except Exception:
        pass
    return ckpt


@torch.no_grad()
def eval_loss(model: GPT, loader: DataLoader, iters: int, device: str) -> float:
    """Mean held-out cross-entropy over `iters` batches (cycles; nan if val empty)."""
    batches = list(loader)
    if not batches:
        return float("nan")
    model.eval()
    tot = 0.0
    for i in range(iters):
        x, y = batches[i % len(batches)]
        _, loss = model(x.to(device), y.to(device))
        tot += loss.item()
    model.train()
    return tot / iters


def train(config_path: str, resume: str | None = None) -> dict:
    """Run pretraining; returns summary stats (also logged to the registry)."""
    cfg = yaml.safe_load(open(config_path))
    t = cfg["train"]
    seed_all(int(t.get("seed", 1337)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = cfg["run"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    model = GPT(config_from_dict(cfg["model"]))
    opt = model.configure_optimizers(float(t["weight_decay"]), float(t["lr"]),
                                     tuple(t.get("betas", [0.9, 0.95])))
    _ = load_manifest(cfg["data"]["manifest"])
    shard_dir = cfg["data"]["shard_dir"]
    train_ds = PackedDataset([os.path.join(shard_dir, "train.pt")],
                             model.config.block_size)
    val_ds = PackedDataset([os.path.join(shard_dir, "val.pt")],
                           model.config.block_size)
    batcher = _Batcher(train_ds, int(t["batch_size"]), int(t.get("seed", 1337)))
    val_dl = DataLoader(val_ds, batch_size=int(t["batch_size"]))

    step, tokens_seen = 0, 0
    loss_history: list[float] = []
    if resume and resume != "none":
        ckpt = load_checkpoint(resume, model, opt)
        step = int(ckpt.get("step", 0))
        tokens_seen = int(ckpt.get("tokens_seen", 0))
        if ckpt.get("sampler") is not None:
            batcher.load(ckpt["sampler"])

    max_steps, accum = int(t["max_steps"]), int(t.get("grad_accum_steps", 1))
    use_amp = device == "cuda" and t.get("precision", "bf16") in ("bf16", "fp16")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.to(device).train()
    pbar = tqdm(total=max_steps, initial=step, desc="train")
    last_loss = float("nan")
    while step < max_steps:
        opt.zero_grad()
        for _ in range(accum):
            x, y = batcher.next()
            x, y = x.to(device), y.to(device)
            lr = cosine_lr(step, max_steps, float(t["lr"]),
                           float(t.get("min_lr", t["lr"] * 0.1)),
                           int(t.get("warmup_steps", 0)))
            for pg in opt.param_groups:
                pg["lr"] = lr
            ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else _nullctx()
            with ctx:
                _, loss = model(x, y)
                loss = loss / accum
            scaler.scale(loss).backward()
            tokens_seen += x.numel()
            last_loss = loss.item() * accum
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(t.get("grad_clip", 1.0)))
        scaler.step(opt)
        scaler.update()
        step += 1
        loss_history.append(last_loss)
        pbar.update(1)
        pbar.set_postfix(loss=f"{last_loss:.3f}")
        if step % int(t.get("eval_interval", 500)) == 0:
            vl = eval_loss(model, val_dl, int(t.get("eval_iters", 100)), device)
            print(f"step {step} train_loss={last_loss:.4f} val_loss={vl:.4f} ppl={math.exp(vl):.1f}")
        if step % int(t.get("checkpoint_interval", 1000)) == 0:
            save_checkpoint(os.path.join(out_dir, f"step_{step}.pt"),
                            model, opt, cfg, step, tokens_seen, t0,
                            sampler=batcher.state())
    pbar.close()
    vl = eval_loss(model, val_dl, int(t.get("eval_iters", 100)), device)
    final = os.path.join(out_dir, "final.pt")
    save_checkpoint(final, model, opt, cfg, step, tokens_seen, t0,
                    sampler=batcher.state())
    row = {"variant": "base", "checkpoint": final, "config_hash": config_hash(cfg),
           "param_count": model.num_params(), "context_length": model.config.block_size,
           "training_tokens": tokens_seen, "wall_clock_seconds": round(time.time() - t0, 1),
           "hardware": device, "dataset_version": cfg["data"]["manifest"],
           "val_loss": round(vl, 4), "perplexity": round(math.exp(vl), 2)}
    run_id = log_registry_row(REGISTRY_PATH, row)
    return {"run_id": run_id, "steps": step, "val_loss": vl,
            "perplexity": math.exp(vl), "tokens": tokens_seen,
            "params": model.num_params(), "loss_history": loss_history}


class _nullctx:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *a: object) -> None:
        return None


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    print(train(args.config, args.resume))


if __name__ == "__main__":
    main()
