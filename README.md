# TriadLM

Shared decoder-only LM with three tracks: **base** (pretraining) → **sft** →
**constitutional** (+**dpo**) → **grounded** (tool use). Spec:
`triadlm-technical-spec (1).md`. Variant ids are `base|sft|constitutional|dpo|grounded`
(never product names — spec §12 guardrail).

## Layout (spec §0)

`triadlm/` (tokenizer, attention, model, data, train, generate, finetune_sft,
constitutional, preference, retrieval, tools, safety) · `configs/` (tiny,
base_50m, base_125m) · `data/` (raw, preprocess, quality_report, manifests,
shards) · `evals/` (60 probes, run_eval, scorers, error taxonomy) ·
`experiments/` (registry.csv, run logs, notes) · `demo/` (FastAPI + static UI) ·
`docs/` (constitution, model/data cards, threat model).

## Milestone 0 (done, CPU)

```bash
pip install -r requirements.txt
python -m pytest tests/ -q                      # 6 tests: round-trip, causality,
                                                # loss-shift, ckpt restore, greedy, resume
python -m data.preprocess --config configs/tiny.yaml --corpus-name toy
python -m data.quality_report --manifest data/manifests/toy.json
python -m triadlm.train --config configs/tiny.yaml
python -m triadlm.generate --checkpoint data/checkpoints/tiny/final.pt \
  --prompt "..." --temperature 0
python -m triadlm.finetune_sft --config configs/tiny.yaml \
  --checkpoint data/checkpoints/tiny/final.pt --out data/checkpoints/tiny/sft.pt
python -m evals.run_eval --checkpoint <ckpt> --variant base|sft|constitutional|grounded
python demo/server.py --base ... --sft ... --constitutional ... --grounded ...  # :8000
```

## Results so far (toy, 3.3M — plumbing, not quality)

| variant | run_id | factuality | citation_acc | over_refusal | gap |
|---|---|---|---|---|---|
| base | 20260925221225-a142 | 0.200 | 0.0 | 0.0 | 0.077 |
| sft | 20260925221237-6c6a | 0.067 | — | 0.0 | -0.077 |
| grounded | 20260925221245-5aa7 | 0.133 | 1.0 | 0.0 | 0.077 |
| constitutional | 20260925221415-6c5f | 0.033 | — | 0.0 | 0.0 |

See `experiments/001_base_vs_sft.md`, `002_constitution_ablation.md`.
Every number traces to a `run_id`; registry is append-only.

## Next (spec order — do not skip)

1. **M1**: scale corpus (licensed EN/Chichewa), train `base_50m.yaml` on single GPU,
   n-gram baseline comparison. Resume test already gates M1→M2.
2. **M2**: regenerate constitutional triples + DPO with the M1 generator; freeze
   base+SFT (registry entry) before touching retrieval/tools.
3. **M3**: probes already 60/50–100 target; add failure modes to error taxonomy.
4. **M4**: docs + article. `base_125m.yaml` only after 50M is stable.
