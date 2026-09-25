# 001 — base vs sft (toy, Milestone 0 scale)

Falsifiable question: does masked SFT on 6 demonstrations change factuality /
instruction-following on the 60-probe harness at toy scale?

Runs (experiments/registry.csv):
- base: run_id `20260925221225-a142` (tiny.yaml, 20 steps, val_loss 6.06, ppl 429.2)
- sft: run_id `20260925221237-6c6a` (10 masked-SFT steps on data/demonstrations.jsonl)

Result: factuality base 0.200 -> sft 0.067; instruction_following 0.0 -> 0.0.
At toy scale (3.3M params, 372-token corpus) SFT on 6 demos does NOT help and
slightly hurts knowledge probes — expected: the model is far below the capacity
where demonstration tuning transfers. Decision: revisit after base_50m (M1) with
a larger demo set; do not scale SFT data until the base curve is stable.
