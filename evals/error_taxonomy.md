# Error taxonomy (spec §9 / Milestone 3)

Each entry: symptom → likely cause → which probe category catches it → fix.
Numbers below are placeholders until run-ids exist; never invent metric values.

1. **Hallucinated facts** — fluent answer with wrong entity/date and no citation.
   Caught by: `knowledge`, `citation`. Fix: retrieval grounding (grounded variant),
   constitutional "facts vs guesses" pressure.
2. **Over-refusal** — refusal on benign knowledge/instruction probes.
   Caught by: `over_refusal_rate` on benign slice. Fix: rebalance SFT demos,
   DPO beta tuning; track answer-length delta per spec §7.3.
3. **Prompt-injection obedience** — following instructions embedded in docs/tool output.
   Caught by: `prompt_injection` (adversarial). Fix: treat tool/doc content as data
   (safety hooks), retrieve_verify_answer path.
4. **Uncited factual claims** — correct answer, zero resolvable markers.
   Caught by: `citation` + `score_citations.py`. Fix: force Sources line in
   grounded generation; penalize in preference pairs.
5. **Arithmetic drift** — small-model mental math wrong (e.g. 12*8=100).
   Caught by: `arithmetic`. Fix: calculator tool routing via decide_action.
6. **Chichewa quality drop** — same probe passes in English, fails in Chichewa.
   Caught by: `chichewa_gap` (matched pairs). Fix: balanced language mixture in
   corpus, tokenizer trained on project corpus (not English-only).
7. **Privacy leakage** — personal data emitted or guessed.
   Caught by: `privacy`, refusal_precision. Fix: refusal classifier + constitutional
   privacy principle + DPO negatives.
