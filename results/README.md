# Reference results (tracked)

Curated, paper-ready artifacts produced **without a GPU** on real MASSIVE-tr.
Bulk run dumps live in `outputs/` (git-ignored).

- **`alignment_validation.txt`** — mandatory BIO re-alignment round-trip (plan §8)
  on dev (2,033 utterances, 11,033 slot-bearing words), all 3 encoders × 4 schemes.
  **All recover gold word labels exactly (100%, 0 mismatches);** only character
  segmentation truncates at `max_length=64`.
  Regenerate: `python scripts/validate_alignment.py --data data/raw/tr-TR.jsonl --models berturk mbert xlmr`

- **`tokenizer_metrics.md`** (paper Table 1) — per-encoder fertility + morpheme-
  boundary alignment (Morfessor reference) + native-vs-whitespace divergence.
  Key findings: mBERT fragments Turkish most (fertility 2.67) and aligns least
  (respect 0.16) vs BERTurk (2.05 / 0.41) → motivates H3; native ≡ whitespace for
  all three encoders (0% divergence).
  Regenerate: `python scripts/tokenizer_report.py --reference morphological --scope slot`

- **`corpus_stats.md`** (paper Data section) — 60 intents / 55 slot types / 18
  domains; 18% of slot-bearing words carry ≥1 affix.
  Regenerate: `python scripts/corpus_stats.py`

The main results table (`tab:main`) and the error-analysis tables are produced on
a GPU (Kaggle) by `run_matrix.py` + `aggregate.py` + `error_report.py`.
