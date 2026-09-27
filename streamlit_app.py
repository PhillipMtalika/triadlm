"""TriadLM Streamlit demo (free Community Cloud): one prompt, four branches.

Loads base/sft/constitutional/dpo checkpoints from the Hub once
(@st.cache_resource), generates sequentially to stay under 1GB RAM.
"""
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import torch

MODEL_REPO = "phillipmtalika/triadlm-m1-50m"
DATA_REPO = "phillipmtalika/triadlm-corpus-v2"

BRANCHES = {
    "base": "final.pt",
    "sft": "sft.pt",
    "constitutional": "constitutional_out/constitutional.pt",
    "grounded": "dpo_out/dpo.pt",
}


@st.cache_resource(show_spinner="Downloading checkpoints (first run only)...")
def load_all():
    """Download + load tokenizer, doc index (models load per-branch below)."""
    from huggingface_hub import snapshot_download
    from triadlm.retrieval import DocumentIndex
    from triadlm.tokenizer import TriadTokenizer
    mdir = snapshot_download(repo_id=MODEL_REPO)
    ddir = snapshot_download(repo_id=DATA_REPO, repo_type="dataset")
    tok = TriadTokenizer.load(os.path.join(ddir, "tokenizer"))
    index = DocumentIndex.from_jsonl("data/documents.jsonl")
    return mdir, tok, index


def branch_output(branch: str, prompt: str, mdir: str, tok, index) -> tuple[str, str]:
    """Load one branch, generate, unload (RAM-safe sequential)."""
    import re
    from triadlm.generate import complete
    from triadlm.model import GPT, config_from_dict
    from triadlm.tools import (CalculatorTool, DateTimeTool,
                               DocumentSearchTool, WebSearchTool, decide_action)
    ckpt = torch.load(os.path.join(mdir, BRANCHES[branch]),
                      map_location="cpu", weights_only=False)
    model = GPT(config_from_dict(ckpt["config"]["model"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    extra = ""
    if branch == "grounded":
        action = decide_action(prompt)
        if action == "refuse":
            return "I can't help with that. I can help with something else instead.", "refuse"
        cites, ctx = [], []
        if action == "retrieve_verify_answer":
            m = re.findall(r"(?=[\d\s()\-+*/^%.]*\d)[\d\s()\-+*/^%.]+", prompt)
            if m:
                r = CalculatorTool().call(expression=max(m, key=len).strip())
                ctx.append(f"Calculator: {r.output or r.error}")
                cites.append(r.citation.render())
        if any(w in prompt.lower() for w in ["today", "date", "time"]):
            r = DateTimeTool().call()
            ctx.append(f"Date/Time: {r.output}")
            cites.append(r.citation.render())
        if action in ("retrieve_then_answer", "retrieve_verify_answer"):
            r = DocumentSearchTool(index).call(query=prompt)
            ctx.append(f"Retrieved:\n{r.output}")
            cites.append(r.citation.render())
            try:  # live Wikipedia (free, logged); silent offline
                w = WebSearchTool().call(query=prompt)
                if w.output:
                    ctx.append(f"Wikipedia:\n{w.output}")
                    cites.append(w.citation.render())
            except Exception:
                pass
        aug = prompt + ("\n\n[Context]\n" + "\n".join(ctx) if ctx else "")
        text = complete(model, tok, aug, max_new_tokens=128,
                        temperature=0.7, top_k=40, repetition_penalty=1.15)
        if cites:
            text += "\nSources: " + " ".join(cites)
        extra = f"action={action}"
    else:
        text = complete(model, tok, prompt, max_new_tokens=128,
                        temperature=0.7, top_k=40, repetition_penalty=1.15)
    del model
    gc.collect()
    return text, extra


st.set_page_config(page_title="TriadLM compare", layout="wide")
st.title("TriadLM — one prompt, four branches")
st.caption("30M model: base | sft | constitutional | grounded (tools + citations). $0 demo.")

mdir, tok, index = load_all()
prompt = st.text_area("Prompt", "What is the capital of Malawi? Cite a source.")
if st.button("Compare"):
    cols = st.columns(4)
    for col, branch in zip(cols, ("base", "sft", "constitutional", "grounded")):
        with col:
            st.subheader(branch)
            with st.spinner(f"generating {branch}..."):
                text, extra = branch_output(branch, prompt, mdir, tok, index)
            st.write(text)
            if extra:
                st.caption(extra)
