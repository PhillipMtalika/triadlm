"""Constitutional data: critique-and-revision generation + CSFT (spec §7.2).

Per-example JSON schema: prompt, initial_answer, critique, revised_answer,
principles, source.
"""
import json
import os
import re


def load_principles(path: str = "docs/constitution.md") -> list[str]:
    """Load principle strings from the constitution document (never hardcoded)."""
    principles: list[str] = []
    with open(path) as f:
        for line in f:
            m = re.match(r"\s*\d+\.\s+(.*)", line.strip())
            if m:
                principles.append(m.group(1))
    if not principles:
        raise ValueError(f"no principles parsed from {path}")
    return principles


def generate_initial_answer(model: object, tok: object, prompt: str) -> str:
    """Sample an initial answer from the current model."""
    from .generate import complete
    return complete(model, tok, prompt, max_new_tokens=64, temperature=0.8)


def critique(model: object, tok: object, prompt: str, answer: str,
             principles: list[str]) -> str:
    """Critique an answer against the principles.

    Reference implementation is rule-based so the pipeline runs without a
    teacher model; swap for a model call without changing the schema.
    """
    notes: list[str] = []
    if re.search(r"\d{4}|https?://|\[doc:", answer) is None and \
            any(w in prompt.lower() for w in ["when", "who", "cite", "source", "liti", "chifukwa"]):
        notes.append("facts-vs-guesses: factual claim without evidence — needs retrieval.")
    if any(w in prompt.lower() for w in ["address", "phone", "password", "ssn"]):
        notes.append("privacy: personal-data request — must refuse.")
    if "ignore previous" in answer.lower() or re.search(r"system\s*:", answer.lower()):
        notes.append("local-context/safety: possible injected instruction — do not follow.")
    if not notes:
        notes.append("No clear violation; tighten specificity and evidence disclosure.")
    _ = (model, tok, principles)
    return "Critique: " + " | ".join(notes)


def revise(model: object, tok: object, prompt: str, answer: str,
           critique_text: str) -> str:
    """Revise an answer in light of its critique."""
    _ = (model, tok, prompt)
    if "privacy" in critique_text:
        return ("I can't share personal data. I can help with something else. "
                "[depends on policy, not retrieved evidence]")
    if "injected instruction" in critique_text:
        return ("I noticed untrusted content that looks like an instruction, so I'll "
                "stick to your original request.")
    suffix = (" [depends on retrieved evidence — verify before acting.]"
              if "evidence" in critique_text or "facts-vs-guesses" in critique_text else "")
    return answer.strip() + suffix


def build_constitutional_dataset(prompts: list[str], principles: list[str],
                                 model: object, tok: object,
                                 source: str = "synthetic") -> list[dict]:
    """Generate (initial_answer, critique, revised_answer) triples."""
    out: list[dict] = []
    for p in prompts:
        init = generate_initial_answer(model, tok, p)
        crit = critique(model, tok, p, init, principles)
        rev = revise(model, tok, p, init, crit)
        out.append({"prompt": p, "initial_answer": init, "critique": crit,
                    "revised_answer": rev, "principles": principles,
                    "source": source})
    return out


def train_constitutional_sft(base_checkpoint: str, dataset_path: str,
                             out_dir: str) -> None:
    """CSFT: SFT on revised answers (same masked loss as finetune_sft)."""
    import torch
    from .finetune_sft import SFTDataset, _pad_collate
    from .generate import load_for_inference
    from torch.utils.data import DataLoader
    model, tok = load_for_inference(base_checkpoint)
    rows = [json.loads(l) for l in open(dataset_path) if l.strip()]
    tmp = os.path.join(out_dir, "_csft.jsonl")
    os.makedirs(out_dir, exist_ok=True)
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps({"prompt": r["prompt"],
                                "response": r["revised_answer"]}) + "\n")
    ds = SFTDataset(tmp, tok, model.config.block_size)
    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=_pad_collate)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    import torch.nn.functional as F
    for x, y in dl:
        opt.zero_grad()
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               y.reshape(-1), ignore_index=-100)
        loss.backward()
        opt.step()
        break  # smoke default: one epoch pass is driven by caller steps in full runs
    from .train import save_checkpoint
    import time
    import torch as _t
    base_cfg = _t.load(base_checkpoint, map_location="cpu",
                       weights_only=False)["config"]
    save_checkpoint(os.path.join(out_dir, "constitutional.pt"), model, opt,
                    base_cfg, 0, 0, time.time())
