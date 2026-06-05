# Reference results (tracked)

Curated, paper-ready artifacts. Bulk run dumps live in `outputs/` (git-ignored).

- `alignment_validation.txt` — the mandatory BIO re-alignment round-trip check
  (plan §8) on MASSIVE-tr **dev** (2,033 utterances, 11,033 slot-bearing words),
  for all 3 encoders × 4 segmentation schemes. **All schemes recover gold word
  labels exactly (100%, 0 mismatches).** Only character segmentation truncates
  words at `max_length=64` (114 for BERTurk/mBERT, 154 for XLM-R). Regenerate:

  ```bash
  python scripts/validate_alignment.py --data data/raw/tr-TR.jsonl \
      --models berturk mbert xlmr > results/alignment_validation.txt
  ```

This is the source for the paper's methodology-validation table (`paper/main.tex`,
`tab:align`). The main results table and error-analysis tables are produced on a
GPU (Kaggle) by `scripts/run_matrix.py` + `scripts/aggregate.py` +
`scripts/error_report.py`.
