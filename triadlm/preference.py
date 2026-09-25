"""DPO-style preference optimization (spec §7.3). No reward model in v1."""
import copy
import json
import os
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def dpo_loss(policy_logp_chosen: Tensor, policy_logp_rejected: Tensor,
             ref_logp_chosen: Tensor, ref_logp_rejected: Tensor,
             beta: float = 0.1) -> Tensor:
    """L = -log sigmoid(beta*(logp_c - logp_r) - beta*(ref_c - ref_r))."""
    return -F.logsigmoid(beta * ((policy_logp_chosen - policy_logp_rejected)
                                 - (ref_logp_chosen - ref_logp_rejected))).mean()


class PreferenceDataset(Dataset):
    """Pairwise rows: {prompt, chosen, rejected, principles?}."""

    def __init__(self, path: str, tok: object, block_size: int) -> None:
        self.rows: list[dict] = [json.loads(l) for l in open(path) if l.strip()]
        self.tok = tok
        self.block = block_size

    def __len__(self) -> int:
        return len(self.rows)

    def _enc(self, prompt: str, resp: str) -> tuple[Tensor, Tensor]:
        ids = self.tok.encode(f"### Instruction:\n{prompt}\n\n### Response:\n{resp}")[:self.block + 1]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y

    def __getitem__(self, i: int) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        r = self.rows[i]
        return self._enc(r["prompt"], r["chosen"]), self._enc(r["prompt"], r["rejected"])


def _pad_pair(batch: list[tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]]
              ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
    xc = torch.nn.utils.rnn.pad_sequence([p[0][0] for p in batch], batch_first=True)
    xr = torch.nn.utils.rnn.pad_sequence([p[1][0] for p in batch], batch_first=True)
    L = max(xc.size(1), xr.size(1))
    xc = F.pad(xc, (0, L - xc.size(1)))
    xr = F.pad(xr, (0, L - xr.size(1)))
    yc = torch.stack([F.pad(p[0][1], (0, L - p[0][1].size(0)), value=-100) for p in batch])
    yr = torch.stack([F.pad(p[1][1], (0, L - p[1][1].size(0)), value=-100) for p in batch])
    return (xc, yc), (xr, yr)


def seq_logp(model: torch.nn.Module, x: Tensor, y: Tensor) -> Tensor:
    """Mean log-prob per response token (prompt already masked to -100)."""
    logits, _ = model(x)
    lp = F.log_softmax(logits, dim=-1).gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    mask = (y != -100).float()
    return (lp * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def train_dpo(base_checkpoint: str, ref_checkpoint: str,
              preference_data_path: str, out_dir: str, beta: float = 0.1) -> None:
    """Train DPO from base against frozen ref; writes out_dir/dpo.pt + sidecar."""
    import time
    from .generate import load_for_inference
    from .train import save_checkpoint
    model, tok = load_for_inference(base_checkpoint)
    ref, _ = load_for_inference(ref_checkpoint or base_checkpoint)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    ds = PreferenceDataset(preference_data_path, tok, model.config.block_size)
    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=_pad_pair)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()
    it = iter(dl)
    for _ in tqdm(range(50), desc="dpo"):
        try:
            (xc, yc), (xr, yr) = next(it)
        except StopIteration:
            it = iter(dl)
            (xc, yc), (xr, yr) = next(it)
        opt.zero_grad()
        with torch.no_grad():
            rc, rr = seq_logp(ref, xc, yc), seq_logp(ref, xr, yr)
        pc, pr = seq_logp(model, xc, yc), seq_logp(model, xr, yr)
        loss = dpo_loss(pc, pr, rc, rr, beta)
        loss.backward()
        opt.step()
    os.makedirs(out_dir, exist_ok=True)
    import time
    import torch as _t
    base_cfg = _t.load(base_checkpoint, map_location="cpu",
                       weights_only=False)["config"]
    base_cfg = {**base_cfg, "dpo_beta": beta}
    save_checkpoint(os.path.join(out_dir, "dpo.pt"), model, opt,
                    base_cfg, 0, 0, time.time())
