# Model card — TriadLM

- Architecture: shared decoder-only Transformer (GPT-style, pre-norm, SDPA causal).
- Variants: `base` (pretrain) → `sft` → `constitutional` (+`dpo`) → `grounded` (tool use).
- Configs: `configs/tiny.yaml` (M0 toy), `configs/base_50m.yaml` (M1 target, ~50M),
  `configs/base_125m.yaml` (stretch, after 50M stable).
- Tokenizer: byte-level BPE trained on project corpus (`TriadTokenizer`); never an
  off-the-shelf English tokenizer (Chichewa-gap bias).
- Intended use: research on constitutional alignment + grounded tool use, EN/Chichewa.
- Limits: small models hallucinate; grounded answers need citation checks
  (`evals/score_citations.py`); see `docs/threat-model.md`.
