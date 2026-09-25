"""Supervised fine-tuning on demonstrations (spec §7.1).

Loss is masked so only response tokens contribute (prompt tokens use
ignore_index). CLI: python -m triadlm.finetune_sft --config ... --checkpoint ...
"""
import json
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .generate import load_for_inference
from .model import config_from_dict
from .tokenizer import TriadTokenizer
from .train import config_hash, load_checkpoint, log_registry_row, save_checkpoint, seed_all

PROMPT_TMPL = "### Instruction:\n{prompt}\n\n### Response:\n{response}"


class SFTDataset(Dataset):
    """(prompt, response) pairs with prompt tokens masked to ignore_index."""

    def __init__(self, path: str, tok: TriadTokenizer, block_size: int, pad_id: int = 2) -> None:
        self.rows: list[dict] = [json.loads(l) for l in open(path) if l.strip()]
        self.tok = tok
        self.block = block_size
        self.pad_id = pad_id

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        r = self.rows[i]
        prompt_ids = self.tok.encode(r["prompt"], add_bos=True, add_eos=False)
        full = self.tok.encode(PROMPT_TMPL.format(prompt=r["prompt"],
                                                  response=r["response"]))[:self.block + 1]
        x = torch.tensor(full[:-1], dtype=torch.long)
        y = torch.tensor(full[1:], dtype=torch.long)
        # mask prompt positions: everything up to len(prompt_ids)-1 in x-space
        cut = max(0, min(len(prompt_ids) - 1, len(y)))
        y[:cut] = -100
        return x, y


def _pad_collate(batch: list[tuple[torch.Tensor, torch.Tensor]],
                 pad_id: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    L = max(x.size(0) for x, _ in batch)
    xs = torch.stack([F.pad(x, (0, L - x.size(0)), value=pad_id) for x, _ in batch])
    ys = torch.stack([F.pad(y, (0, L - y.size(0)), value=-100) for _, y in batch])
    return xs, ys


def train_sft(model: torch.nn.Module, tok: TriadTokenizer, data_path: str,
              steps: int = 500, batch_size: int = 4, lr: float = 1e-4,
              device: str = "cpu") -> dict:
    """Run masked SFT; returns stats."""
    ds = SFTDataset(data_path, tok, model.config.block_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    collate_fn=_pad_collate)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.to(device).train()
    it = iter(dl)
    last = float("nan")
    for _ in tqdm(range(steps), desc="sft"):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(dl)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               y.reshape(-1), ignore_index=-100)
        loss.backward()
        opt.step()
        last = loss.item()
    return {"loss": last, "steps": steps, "examples": len(ds)}


def main() -> None:
    import argparse
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--demos", default="data/demonstrations.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=500)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    seed_all(int(cfg["train"].get("seed", 1337)))
    model, tok = load_for_inference(args.checkpoint)
    stats = train_sft(model, tok, args.demos, steps=args.steps)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_checkpoint(args.out, model, opt, cfg, 0, 0, __import__("time").time())
    log_registry_row("experiments/registry.csv",
                     {"variant": "sft", "checkpoint": args.out,
                      "config_hash": config_hash(cfg),
                      "param_count": model.num_params(),
                      "notes": json.dumps(stats)})
    print(stats)


if __name__ == "__main__":
    main()
