# 003 — Kaggle free-T4 base_50m validation run

Falsifiable question: does the M1 pipeline (AMP fp16, ckpts, resume-ready
batcher, registry, eval) run end-to-end on a free Kaggle T4?

Runs:
- train: run_id `20260926005606-cd6d` (configs/kaggle_50m.yaml, 500 steps,
  13 min, loss 9.09 -> 0.04, 29.6M params, 8.19M training tokens)
- eval base: run_id `20260926010127-b47d` (factuality 0.10, citation 0.0,
  over_refusal 0.0, chichewa_gap 0.077, p50 latency 2198.6 ms CPU)

Artifacts (free HF Hub): `phillipmtalika/triadlm-kaggle-50m`
(final.pt, step_250.pt, step_500.pt + checkpoint.json sidecars + corpus_v1.json).

Result: yes — pipeline validated $0. val_loss=nan (356-token val shard can't
fill a 512 block) and final loss 0.04 = memorization of the 37k-token corpus,
as predicted. Decision: real M1 needs ~1000x tokens (Wikipedia dumps / OSCAR
via Kaggle Datasets) + more Chichewa (JW300/Masakhane) before training longer.
Next: SFT/constitutional/grounded variants of THIS checkpoint, then Spaces demo.
