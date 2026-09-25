# TriadLM — Full Technical Specification

Status: implementation-ready spec derived from the project blueprint.
Purpose: paste sections of this doc directly into Claude, GPT, or Grok as a coding
assistant. Each component below is self-contained enough to hand to a model with
no extra context and get a correct implementation back.

Conventions used throughout: Python 3.11, PyTorch ≥ 2.2, single-GPU first,
type hints on every public function, config-driven (no hardcoded hyperparameters
in code), every artifact (checkpoint, dataset, eval run) gets a JSON metadata
sidecar.

---

## 0. Repository layout (authoritative)

```text
triadlm/
├── README.md
├── LICENSE
├── pyproject.toml
├── configs/
│   ├── tiny.yaml            # Milestone 0 char-level toy model
│   ├── base_50m.yaml        # Milestone 1 target
│   └── base_125m.yaml       # stretch goal, only after 50m is stable
├── triadlm/
│   ├── __init__.py
│   ├── tokenizer.py         # BPE / byte-level tokenizer, train + load
│   ├── model.py             # GPTConfig, CausalSelfAttention, Block, GPT
│   ├── attention.py         # attention math isolated for unit testing
│   ├── data.py              # dataset manifest loader, packed token shards
│   ├── train.py             # pretraining loop entrypoint
│   ├── generate.py          # sampling / decoding entrypoint
│   ├── finetune_sft.py      # supervised fine-tuning on demonstrations
│   ├── constitutional.py    # critique-and-revision data gen + CSFT
│   ├── preference.py        # DPO-style preference optimization
│   ├── retrieval.py         # document index + retriever
│   ├── tools.py             # calculator, date/time, tool-use policy
│   └── safety.py            # refusal classifier + eval hooks
├── data/
│   ├── manifests/*.json     # one manifest per corpus, schema in §5
│   ├── preprocess.py        # raw -> cleaned -> sharded tokens
│   └── quality_report.py    # dedup rate, language id histogram, etc.
├── evals/
│   ├── probes.jsonl         # schema in §9
│   ├── run_eval.py
│   ├── score_factuality.py
│   ├── score_citations.py
│   └── error_taxonomy.md
├── experiments/
│   ├── registry.csv         # schema in §10
│   ├── 001_base_vs_sft.md
│   └── 002_constitution_ablation.md
├── notebooks/
├── docs/
│   ├── model-card.md
│   ├── data-card.md
│   └── threat-model.md
└── demo/
    ├── server.py            # FastAPI, spec in §11
    └── ui/                  # static single-page comparison UI
```

Build order for a coding agent (do not skip ahead):
`tokenizer.py` → `model.py`/`attention.py` (+ unit tests) → `data.py` → `train.py`
(Milestone 0 tiny.yaml) → `generate.py` → scale to `base_50m.yaml` (Milestone 1)
→ `finetune_sft.py` → `constitutional.py` → `preference.py` (Milestone 2, Claude
track) → `retrieval.py` + `tools.py` (Milestone 2, Grok track) → `evals/`
(Milestone 3) → `demo/` (Milestone 4).

---

## 1. Environment

`pyproject.toml` core dependencies — keep this list minimal, resist adding
orchestration frameworks:

```toml
[project]
name = "triadlm"
requires-python = ">=3.11"
dependencies = [
  "torch>=2.2",
  "numpy",
  "tokenizers>=0.15",     # Hugging Face BPE trainer, used as a library only
  "pyyaml",
  "tqdm",
  "fastapi",
  "uvicorn",
  "pydantic>=2",
  "requests",             # for the optional web retrieval adapter
]
[project.optional-dependencies]
dev = ["pytest", "ruff", "mypy"]
```

No training-orchestration framework (no Lightning, no Accelerate) for the
first pass — plain PyTorch + `torch.cuda.amp` + manual DDP if multi-GPU is
ever needed. This keeps every coding model able to reason about the full
training loop in one file.

---

## 2. Tokenizer (`triadlm/tokenizer.py`)

Choice: byte-level BPE (GPT-2 style), trained on the project's own corpus —
do not import a pretrained GPT-2 tokenizer, since one goal is measuring the
English/Chichewa gap and an off-the-shelf English tokenizer would bias that
result before training even starts.

Required interface:

```python
class TriadTokenizer:
    vocab_size: int          # config-driven, default 8192 for the 30-50M model
    special_tokens: dict[str, int]  # {"<bos>":0, "<eos>":1, "<pad>":2, "<unk>":3}

    @classmethod
    def train(cls, corpus_paths: list[str], vocab_size: int, save_dir: str) -> "TriadTokenizer": ...

    @classmethod
    def load(cls, save_dir: str) -> "TriadTokenizer": ...

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def encode_batch(self, texts: list[str]) -> list[list[int]]: ...
```

Round-trip unit test requirement: `decode(encode(x)) == x` for a fixed set of
English and Chichewa sentences, including code snippets and punctuation edge
cases (must be in `tests/test_tokenizer.py`).

---

## 3. Model (`triadlm/model.py`, `triadlm/attention.py`)

`GPTConfig` (pydantic or dataclass):

```python
@dataclass
class GPTConfig:
    vocab_size: int = 8192
    block_size: int = 512        # context length
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1
    bias: bool = False           # no bias in Linear/LayerNorm, per GPT-2/nanoGPT practice
    tie_weights: bool = True      # ablation flag: tie input/output embeddings
```

Parameter count guide (use this to hit the 30–50M and 100–125M targets):
`params ≈ 12 * n_layer * n_embd^2` (dominant term, ignoring embeddings/vocab).
Example: `n_layer=8, n_embd=512` → ≈ 25M non-embedding params; add
`vocab_size * n_embd * 2` (in + out, or just 1× if tied) for the embedding
table.

Module interface:

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig): ...
    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, C) -> (B, T, C)
        # explicit causal mask via torch.tril or F.scaled_dot_product_attention(is_causal=True)
        ...

class Block(nn.Module):
    # pre-norm: x = x + attn(ln1(x)); x = x + mlp(ln2(x))
    def __init__(self, config: GPTConfig): ...
    def forward(self, x: Tensor) -> Tensor: ...

class GPT(nn.Module):
    def __init__(self, config: GPTConfig): ...
    def forward(self, idx: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        # idx: (B, T) int64
        # returns (logits (B, T, vocab_size), loss or None)
        # loss = F.cross_entropy(logits.view(-1, vocab), targets.view(-1), ignore_index=pad_id)
        ...

    @torch.no_grad()
    def generate(self, idx: Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None) -> Tensor: ...

    def configure_optimizers(self, weight_decay: float, lr: float,
                              betas: tuple[float, float]) -> torch.optim.Optimizer:
        # split params into decay / no-decay groups (no decay on biases, LayerNorm, embeddings)
        ...
```

Required unit tests (`tests/test_model.py`):
- causal mask correctness: changing token `t+1` must not change logits at
  position `t`;
- loss-shifting correctness: `targets = idx[:, 1:]`, `idx = idx[:, :-1]`, verify
  shapes;
- checkpoint save/restore reproduces identical logits on a fixed input;
- generation is deterministic when `temperature=0`/greedy.

---

## 4. Data pipeline (`triadlm/data.py`, `data/preprocess.py`)

Pipeline stages: raw text → language ID + filtering → dedup → tokenize →
pack into fixed-length blocks → shard to `.bin`/`.npy` files → manifest.

```python
def build_manifest(sources: list[SourceSpec], out_path: str) -> DataManifest: ...

class PackedDataset(torch.utils.data.Dataset):
    def __init__(self, shard_paths: list[str], block_size: int): ...
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        # returns (x, y) each (block_size,), y = x shifted by 1
        ...
```

Manifest JSON schema (one file per corpus, stored under `data/manifests/`):

```json
{
  "name": "string",
  "source": "string (url or description)",
  "license": "string",
  "language": "en | ny | mixed",
  "document_count": 0,
  "token_count": 0,
  "filtering_rules": ["dedup:minhash", "min_length:50_tokens", "lang_id_threshold:0.8"],
  "train_split": 0.98,
  "val_split": 0.02,
  "sha256_of_shards": "string",
  "created_at": "ISO-8601"
}
```

`data/quality_report.py` must output: dedup rate, language ID histogram,
document length histogram, and top-N most frequent documents (dup check) —
save as `data/manifests/<name>.quality.json`.

---

## 5. Pretraining loop (`triadlm/train.py`)

Config schema (`configs/base_50m.yaml`):

```yaml
model:
  vocab_size: 8192
  block_size: 512
  n_layer: 8
  n_head: 8
  n_embd: 512
  dropout: 0.1
  tie_weights: true
train:
  batch_size: 32
  grad_accum_steps: 4
  max_steps: 20000
  lr: 3.0e-4
  min_lr: 3.0e-5
  warmup_steps: 500
  weight_decay: 0.1
  betas: [0.9, 0.95]
  grad_clip: 1.0
  precision: bf16          # or fp16
  eval_interval: 500
  eval_iters: 100
  checkpoint_interval: 1000
  seed: 1337
data:
  manifest: data/manifests/corpus_v1.json
  shard_dir: data/shards/corpus_v1
```

`train.py` responsibilities, in order:
1. load config, set seed;
2. build model from `GPTConfig`;
3. build `PackedDataset` + `DataLoader`;
4. cosine LR schedule with linear warmup (implement directly, do not import
   a scheduler library, so the decay curve is inspectable);
5. training loop with gradient accumulation, mixed precision, grad clipping;
6. every `eval_interval` steps: compute held-out loss/perplexity, log to
   `experiments/registry.csv` (§10) and to a JSON run log;
7. every `checkpoint_interval` steps: save `{model_state, optimizer_state,
   config, step, rng_state}` plus a `checkpoint.json` sidecar with commit
   hash, config hash, param count, and wall-clock time so far;
8. resume-from-checkpoint must reproduce identical loss curve (this is a
   required test, not optional).

CLI: `python -m triadlm.train --config configs/base_50m.yaml --resume <ckpt|none>`

---

## 6. Generation (`triadlm/generate.py`)

CLI: `python -m triadlm.generate --checkpoint <path> --prompt "..." --max_new_tokens 200 --temperature 0.8 --top_k 40`

Must support greedy, temperature+top-k sampling, and batch generation for the
eval harness (§9) to call it programmatically:

```python
def load_for_inference(checkpoint_path: str) -> tuple[GPT, TriadTokenizer]: ...
def complete(model: GPT, tok: TriadTokenizer, prompt: str, **sampling_kwargs) -> str: ...
```

---

## 7. Claude-inspired track: SFT → constitutional → preference

### 7.1 SFT (`triadlm/finetune_sft.py`)
Standard supervised fine-tuning on `(prompt, response)` pairs. Loss masked so
only response tokens contribute (prompt tokens get `ignore_index`).

### 7.2 Constitutional data (`triadlm/constitutional.py`)

Constitution (store as `docs/constitution.md`, load as a list of principle
strings at runtime — do not hardcode in Python):
1. be useful and specific;
2. distinguish facts from guesses;
3. protect privacy and avoid exposing secrets;
4. refuse dangerous assistance while offering a safe alternative;
5. respect local context; do not treat English assumptions as universal;
6. disclose when an answer depends on retrieved evidence.

Per-example JSON schema (`data/constitutional/*.jsonl`, one object per line):

```json
{
  "prompt": "string",
  "initial_answer": "string",
  "critique": "string",
  "revised_answer": "string",
  "principles": ["truthfulness", "privacy"],
  "source": "human|synthetic|hybrid"
}
```

Pipeline functions:

```python
def generate_initial_answer(model, tok, prompt: str) -> str: ...
def critique(model, tok, prompt: str, answer: str, principles: list[str]) -> str: ...
def revise(model, tok, prompt: str, answer: str, critique_text: str) -> str: ...
def build_constitutional_dataset(prompts: list[str], principles: list[str],
                                   model, tok) -> list[dict]: ...
def train_constitutional_sft(base_checkpoint: str, dataset_path: str,
                               out_dir: str) -> None: ...
```

### 7.3 Preference optimization (`triadlm/preference.py`)

Pairwise preference schema:

```json
{"prompt": "string", "chosen": "string", "rejected": "string", "principles": ["..."]}
```

Implement a simple DPO-style loss (no separate reward model required for the
first version):

```
L = -log(sigmoid(beta * (logp_chosen_policy - logp_chosen_ref)
                 - beta * (logp_rejected_policy - logp_rejected_ref)))
```

```python
def dpo_loss(policy_logp_chosen, policy_logp_rejected,
             ref_logp_chosen, ref_logp_rejected, beta: float = 0.1) -> Tensor: ...
def train_dpo(base_checkpoint: str, ref_checkpoint: str,
              preference_data_path: str, out_dir: str, beta: float = 0.1) -> None: ...
```

Required measurement after training each variant (SFT-only, constitutional,
DPO): over-refusal rate, answer length delta, Chichewa/English performance
gap, calibration ("say unknown" accuracy) — feed straight into §9's eval
harness, do not compute ad hoc.

---

## 8. Grok-inspired track: retrieval + tools

### 8.1 Tool interface (`triadlm/tools.py`)

All tools share one interface so the policy layer can dispatch generically:

```python
class Tool(Protocol):
    name: str
    description: str
    input_schema: dict   # JSON schema
    def call(self, **kwargs) -> ToolResult: ...

@dataclass
class ToolResult:
    output: str
    citation: Citation | None
    error: str | None
```

Ship at minimum:
- `CalculatorTool` — typed numeric input, unit-tested against a table of
  arithmetic/precision edge cases (division by zero, overflow, floats);
- `DateTimeTool` — returns current date/time in a fixed format, no network;
- `DocumentSearchTool` — wraps `retrieval.py`'s local index, returns passages
  with document id + offset;
- `WebRetrievalTool` (optional) — records `{url, timestamp, query, passage}`
  for every call, and MUST be logged even on failure.

### 8.2 Retrieval (`triadlm/retrieval.py`)

```python
class DocumentIndex:
    def __init__(self, embedding_fn: Callable[[str], Tensor]): ...
    def add(self, doc_id: str, text: str, metadata: dict) -> None: ...
    def search(self, query: str, k: int = 5) -> list[RetrievedPassage]: ...

@dataclass
class RetrievedPassage:
    doc_id: str
    text: str
    score: float
    offset: tuple[int, int]
```

Citation format enforced everywhere a retrieved passage is used in a
generated answer: `[doc_id:offset_start-offset_end]` inline, plus a trailing
sources list. `score_citations.py` (§9) checks that every citation marker
resolves to a real passage that actually supports the adjacent claim.

### 8.3 Tool-use policy

A small classifier or prompted decision step choosing one of:
`answer_directly | retrieve_then_answer | retrieve_verify_answer | refuse`.
Implement as a function first (rule/prompt-based), swap for a trained
classifier only if the rule-based version is a measured bottleneck:

```python
def decide_action(query: str, model, tok) -> Literal[
    "answer_directly", "retrieve_then_answer", "retrieve_verify_answer", "refuse"]: ...
```

Required comparison matrix (run all four for every eval prompt that has a
factual component): parametric-only, retrieval+citations, retrieval+verify,
retrieval-with-adversarial/poisoned-document. Log latency and citation
accuracy for each.

---

## 9. Evaluation harness (`evals/`)

Probe schema (`evals/probes.jsonl`, one JSON object per line):

```json
{
  "id": "string",
  "category": "knowledge|instruction|consistency|privacy|safety|prompt_injection|citation|arithmetic|chichewa_gap|preference",
  "prompt": "string",
  "reference_answer": "string | null",
  "language": "en | ny",
  "allow_unknown": true,
  "adversarial": false,
  "notes": "string"
}
```

Target: 50–100 probes across every category before Milestone 3 is
considered done.

`run_eval.py` CLI: `python -m evals.run_eval --checkpoint <path> --variant base|sft|constitutional|grounded --probes evals/probes.jsonl --out experiments/runs/<run_id>.json`

Metrics computed per run (write formulas into `score_factuality.py` /
`score_citations.py`, don't leave them undefined):
- validation loss / perplexity (from held-out shard, not from probes);
- instruction-following score (rubric-scored 0/0.5/1 per probe, human or
  a fixed scoring prompt — document which);
- factuality score = correct / (correct + incorrect), with "unknown"
  counted separately from "incorrect" when `allow_unknown: true`;
- over-refusal rate = refusals on benign probes / benign probes;
- refusal precision = correct refusals / total refusals;
- citation accuracy = citations that resolve + support the claim / total
  citations emitted;
- Chichewa/English gap = `score_en - score_ny` on the matched probe subset;
- median and p95 latency in ms, tokens/sec — measured from `generate.py`,
  not estimated.

Run-record JSON schema (what `run_eval.py` writes per run, and what feeds
the registry in §10):

```json
{
  "run_id": "string",
  "variant": "base|sft|constitutional|dpo|grounded",
  "checkpoint": "path",
  "config_hash": "string",
  "param_count": 0,
  "context_length": 0,
  "training_tokens": 0,
  "wall_clock_seconds": 0,
  "hardware": "string",
  "dataset_version": "string",
  "language_mixture": {"en": 0.0, "ny": 0.0},
  "val_loss": 0.0,
  "perplexity": 0.0,
  "instruction_following": 0.0,
  "factuality": 0.0,
  "citation_accuracy": 0.0,
  "over_refusal_rate": 0.0,
  "refusal_precision": 0.0,
  "chichewa_gap": 0.0,
  "latency_p50_ms": 0.0,
  "latency_p95_ms": 0.0,
  "tokens_per_second": 0.0,
  "timestamp": "ISO-8601"
}
```

---

## 10. Experiment registry (`experiments/registry.csv`)

One row appended per run, columns = the flattened keys of the run-record
schema in §9 (config_hash, variant, val_loss, factuality, ... timestamp).
`train.py` and `run_eval.py` both append to this file — never hand-edit it.
Each `experiments/NNN_description.md` file is a short human-written note
that references specific `run_id`s from the registry and states the
falsifiable question that run answered.

---

## 11. Demo / API layer (`demo/server.py`)

FastAPI service exposing one endpoint per behavior branch plus a comparison
endpoint:

```
POST /v1/complete/base        {"prompt": str, "max_new_tokens": int} -> {"text": str, "latency_ms": float}
POST /v1/complete/sft         same
POST /v1/complete/constitutional  same
POST /v1/complete/grounded    {"prompt": str} -> {"text": str, "citations": list[Citation], "action": str, "latency_ms": float}
POST /v1/compare              {"prompt": str} -> results from all four endpoints in one response
GET  /v1/health
```

`demo/ui/` is a single static HTML/JS page (no framework needed) that posts
to `/v1/compare` and renders four columns side by side — this satisfies the
blueprint's "one shared UI showing the same prompt sent to every branch."

---

## 12. Guardrails an implementing model must follow

- Never label output as "GPT," "Claude," or "Grok" as a product name; use
  `base`, `constitutional`, `grounded` (or `sft`/`dpo`) as the only
  variant identifiers in code, configs, logs, and UI.
- Never hardcode or fabricate metric values anywhere in code or docs —
  every number in `experiments/` must trace to a `run_id`.
- Never skip the resume-from-checkpoint reproducibility test before
  moving from Milestone 1 to Milestone 2.
- Do not add a distributed-training framework until single-GPU training is
  verified stable for ≥ 1 full run of `base_50m.yaml`.
- Do not start `retrieval.py`/`tools.py` before the no-tool baseline
  (`base_50m.yaml` + SFT) is frozen and has a registry entry.

---

## 13. Prompt for Eraser DiagramGPT

Paste the block below as-is into Eraser's DiagramGPT to generate the system
architecture diagram for this project.

```
Create a system architecture diagram for a machine learning research project called TriadLM.

Top level: a box labeled "Shared Tokenizer (byte-level BPE, trained on project corpus)" feeding into a box labeled "Shared Decoder-Only Transformer LM (30M-125M params)".

From the shared LM, branch into three parallel tracks, each clearly labeled and color-coded differently:

Track 1, labeled "Base GPT Track": a single box "Next-token pretraining objective" with no further branches.

Track 2, labeled "Claude-inspired Track (Constitutional Alignment)": a vertical pipeline of four boxes in order: "SFT on demonstrations" -> "Constitution (6 written principles)" -> "Critique-and-Revision generation (initial_answer, critique, revised_answer)" -> "Preference Optimization (DPO-style, chosen vs rejected)". Add a small side box labeled "Constitution document" feeding into the critique step.

Track 3, labeled "Grok-inspired Track (Grounded Tool Use)": a box "Tool-Use Policy (decide: answer directly / retrieve / retrieve+verify / refuse)" branching into four boxes: "Calculator Tool", "Date/Time Tool", "Document Search Tool (local index)", "Web Retrieval Adapter (optional, logs url/timestamp/query)". These four feed back up into a box "Retrieval-Augmented Answer with Citations".

Below all three tracks, converge into a shared box: "Evaluation Harness (50-100 probes: knowledge, instruction-following, consistency, privacy, safety, prompt-injection, citation accuracy, arithmetic, Chichewa/English gap, latency)".

From the Evaluation Harness, connect to a final box: "Experiment Registry (CSV log: config hash, params, tokens, val loss, factuality, over-refusal, citation accuracy, Chichewa gap, latency, per run_id)".

Add a final row below everything: three side-by-side boxes "Data Manifests (source, license, language, token count)", "Model Checkpoints (config hash, step, rng state)", "Shared Comparison UI / Demo API (side-by-side prompt comparison across all three tracks)", all connected upward into the Evaluation Harness and Experiment Registry boxes.

Use a clean left-to-right or top-to-bottom flow, one color per track (Base / Constitutional / Grounded), and keep all box labels exactly as given above.
```

---

## 14. Milestone → deliverable checklist (condensed)

| Milestone | Must exist before moving on |
|---|---|
| 0 | tiny.yaml trains on toy text; tokenizer round-trip test passes; masking/loss-shift/checkpoint/generation unit tests pass |
| 1 | base_50m.yaml trains end-to-end; manifest + quality report exist; loss/perplexity logged; n-gram baseline comparison recorded |
| 2 | base, sft, constitutional, grounded endpoints all callable; shared comparison UI shows all four for one prompt |
| 3 | 50-100 probes committed; run_eval.py produces a run-record JSON per variant; error taxonomy doc has ≥5 documented failure modes |
| 4 | README, model card, data card, threat model, experiment logs, and one long-form article published |
