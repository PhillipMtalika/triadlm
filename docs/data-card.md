# Data card — TriadLM corpus

- Raw: `data/raw/*.txt` (project-owned EN/Chichewa text).
- Pipeline: `data/preprocess.py` (lang-id heuristic, sha256-exact dedup, tokenize,
  pack, shard) → manifest `data/manifests/<name>.json` (schema: spec §4).
- Quality: `data/quality_report.py` → `<name>.quality.json` (dedup rate, language
  histogram, length histogram, top duplicates).
- Splits: deterministic hash split (default train 0.98 / val 0.02).
- Gaps: seed corpus is tiny (~1KB); scale with licensed sources before M1, and
  record license per manifest (one file per corpus).
