"""TriadLM Spaces entrypoint: loads all four branches from the Hub once,
then serves the comparison API + static UI. Runs on free CPU (4x30M ~ 500MB)."""
import os
import shutil
import sys

sys.path.insert(0, "/app")

MODEL_REPO = os.environ.get("MODEL_REPO", "phillipmtalika/triadlm-m1-50m")

BRANCHES = {
    "base": "final.pt",
    "sft": "sft.pt",
    "constitutional": "constitutional_out/constitutional.pt",
    "grounded": "dpo_out/dpo.pt",
}


def main() -> object:
    from huggingface_hub import snapshot_download
    import demo.server as S
    dst = snapshot_download(repo_id=MODEL_REPO)
    print(f"model repo -> {dst}", flush=True)
    local = "/tmp/triadlm_ckpts"
    os.makedirs(local, exist_ok=True)
    for branch, rel in BRANCHES.items():
        tgt = os.path.join(local, branch + ".pt")
        shutil.copy(os.path.join(dst, rel), tgt)
        S.load_variant(branch, tgt, "docs.jsonl")
    print("variants:", sorted(S.STATE.keys()), flush=True)
    return S.app


app = main()
