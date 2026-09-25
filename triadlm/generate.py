"""Sampling / decoding entrypoint (spec §6).

CLI: python -m triadlm.generate --checkpoint <path> --prompt "..." \\
     --max_new_tokens 200 --temperature 0.8 --top_k 40
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_for_inference(checkpoint_path: str) -> tuple["GPT", "TriadTokenizer"]:
    """Load (model, tokenizer) from a checkpoint + its saved config."""
    from triadlm.model import GPT, config_from_dict
    from triadlm.tokenizer import TriadTokenizer
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = GPT(config_from_dict(cfg["model"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    tok_dir = cfg.get("tokenizer", {}).get("save_dir", "")
    tok = TriadTokenizer.load(tok_dir)
    return model, tok


def complete(model: "GPT", tok: "TriadTokenizer", prompt: str,
             max_new_tokens: int = 200, temperature: float = 0.8,
             top_k: int | None = 40) -> str:
    """Complete a prompt (programmatic entrypoint for the eval harness)."""
    ids = tok.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=max_new_tokens,
                             temperature=temperature, top_k=top_k)
    return tok.decode(out[0].tolist()[len(ids):])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    args = ap.parse_args()
    model, tok = load_for_inference(args.checkpoint)
    t0 = time.time()
    text = complete(model, tok, args.prompt, args.max_new_tokens,
                    args.temperature, None if args.top_k < 0 else args.top_k)
    print(text)
    print(f"[latency_ms={(time.time() - t0) * 1000:.0f}]")


if __name__ == "__main__":
    main()
