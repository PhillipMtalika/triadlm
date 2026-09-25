"""Demo / API layer (spec §11). One endpoint per branch + comparison.

Variant ids are base | sft | constitutional | grounded (never product names).
"""
import os
import sys
import time

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(title="TriadLM Demo API")
STATE: dict = {}


class CompleteRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128


class GroundedRequest(BaseModel):
    prompt: str


class CompareRequest(BaseModel):
    prompt: str


def _complete(variant: str, prompt: str, max_new_tokens: int) -> dict:
    from triadlm.generate import complete
    t0 = time.perf_counter()
    text = complete(STATE[variant]["model"], STATE[variant]["tok"], prompt,
                    max_new_tokens=max_new_tokens, temperature=0.0, top_k=None)
    return {"text": text, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


@app.get("/v1/health")
def health() -> dict:
    """Liveness + loaded variants."""
    return {"ok": True, "variants": sorted(STATE.keys())}


@app.post("/v1/complete/base")
def complete_base(r: CompleteRequest) -> dict:
    """Base pretraining-only branch."""
    return _complete("base", r.prompt, r.max_new_tokens)


@app.post("/v1/complete/sft")
def complete_sft(r: CompleteRequest) -> dict:
    """SFT branch."""
    return _complete("sft", r.prompt, r.max_new_tokens)


@app.post("/v1/complete/constitutional")
def complete_constitutional(r: CompleteRequest) -> dict:
    """Constitutional-alignment branch."""
    return _complete("constitutional", r.prompt, r.max_new_tokens)


@app.post("/v1/complete/grounded")
def complete_grounded(r: GroundedRequest) -> dict:
    """Grounded tool-use branch with citations + chosen action."""
    from triadlm.generate import complete
    from triadlm.retrieval import DocumentIndex
    from triadlm.tools import (CalculatorTool, DateTimeTool,
                               DocumentSearchTool, decide_action)
    import re
    t0 = time.perf_counter()
    action = decide_action(r.prompt)
    if action == "refuse":
        return {"text": "I can't help with that. I can help with something else instead.",
                "citations": [], "action": action,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
    st = STATE["grounded"]
    cites: list[str] = []
    ctx: list[str] = []
    if action == "retrieve_verify_answer":
        m = re.findall(r"(?=[\d\s()\-+*/^%.]*\d)[\d\s()\-+*/^%.]+", r.prompt)
        if m:
            res = CalculatorTool().call(expression=max(m, key=len).strip())
            ctx.append(f"Calculator: {res.output or res.error}")
            cites.append(res.citation.render())
    if any(w in r.prompt.lower() for w in ["today", "date", "time"]):
        res = DateTimeTool().call()
        ctx.append(f"Date/Time: {res.output}")
        cites.append(res.citation.render())
    if action in ("retrieve_then_answer", "retrieve_verify_answer") and st.get("index") is not None:
        res = DocumentSearchTool(st["index"]).call(query=r.prompt)
        ctx.append(f"Retrieved:\n{res.output}")
        cites.append(res.citation.render())
    aug = r.prompt + ("\n\n[Context]\n" + "\n".join(ctx) if ctx else "")
    text = complete(st["model"], st["tok"], aug, max_new_tokens=128,
                    temperature=0.0, top_k=None)
    if cites:
        text += "\nSources: " + " ".join(cites)
    return {"text": text, "citations": cites, "action": action,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


@app.post("/v1/compare")
def compare(r: CompareRequest) -> dict:
    """All branches for one prompt in a single response."""
    out: dict = {"prompt": r.prompt}
    for v in ("base", "sft", "constitutional"):
        out[v] = _complete(v, r.prompt, 128)["text"] if v in STATE else "not loaded"
    out["grounded"] = complete_grounded(GroundedRequest(prompt=r.prompt)) \
        if "grounded" in STATE else "not loaded"
    return out


@app.get("/")
def ui() -> FileResponse:
    """Static single-page comparison UI."""
    return FileResponse(os.path.join(os.path.dirname(__file__), "ui", "index.html"))


def load_variant(variant: str, checkpoint: str, docs_path: str | None = None) -> None:
    """Load one variant checkpoint into memory."""
    import torch
    from triadlm.model import GPT, config_from_dict
    from triadlm.tokenizer import TriadTokenizer
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = GPT(config_from_dict(ckpt["config"]["model"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    entry: dict = {"model": model,
                   "tok": TriadTokenizer.load(ckpt["config"]["tokenizer"]["save_dir"])}
    if docs_path and os.path.exists(docs_path):
        from triadlm.retrieval import DocumentIndex
        entry["index"] = DocumentIndex.from_jsonl(docs_path)
    STATE[variant] = entry


def main() -> None:
    import argparse
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None)
    ap.add_argument("--sft", default=None)
    ap.add_argument("--constitutional", default=None)
    ap.add_argument("--grounded", default=None)
    ap.add_argument("--docs", default="data/documents.jsonl")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    for v in ("base", "sft", "constitutional", "grounded"):
        ckpt = getattr(args, v)
        if ckpt:
            load_variant(v, ckpt, args.docs)
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
