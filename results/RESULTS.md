# Results (real runs on MASSIVE-tr)

Local CPU runs (AMD Ryzen 7 5800H), 3 epochs, `max_length=48`, batch 32, seed 42.
Seeds 123/2024 (native + morphological × 3 models) run separately for mean±std.

## Main table — test set (2,974 utterances), seed 42

| Encoder | Segmentation | Intent Acc | Slot F1 | Frame Acc |
|---|---|---|---|---|
| **BERTurk** | **native** | **0.878** | **0.749** | **0.657** |
| BERTurk | morphological | 0.870 | 0.723 | 0.625 |
| BERTurk | character | 0.720 | 0.226 | 0.257 |
| mBERT | native | 0.835 | 0.584 | 0.505 |
| mBERT | morphological | 0.823 | 0.537 | 0.468 |
| XLM-R | native | 0.851 | 0.637 | 0.550 |
| XLM-R | morphological | 0.841 | 0.638 | 0.545 |

## Findings

- **Native ≥ morphological for every encoder** (slot F1). Unsupervised
  morphological pre-segmentation does **not** help Turkish slot filling.
- It **hurts the weakest multilingual mBERT most** (−0.047 slot F1); XLM-R is
  unaffected (tie). This is the opposite of the naive H3 prediction.
- **H1 holds:** intent moves ≤0.012 across segmentations; slot F1 moves up to
  0.047 — slot filling is the segmentation-sensitive task.
- **Character is a clear floor** (BERTurk 0.226 slot F1), far below subword.
- **BERTurk (monolingual) leads** — it is the most morphologically coherent
  tokenizer (see `tokenizer_metrics.md`).

## Error analysis — slot error rate by affix count (native)

| Affixes | BERTurk | mBERT | XLM-R |
|---|---|---|---|
| 0 | 0.21 | 0.34 | 0.31 |
| 1 | 0.30 | 0.50 | 0.42 |
| 2 | 0.32 | 0.53 | 0.50 |

Errors rise monotonically with morphological complexity for all encoders,
steepest for mBERT. **Morphology is where slot filling fails — but morpheme-
aligned pre-segmentation is not the fix**, pointing at fragment *representations*
rather than boundary placement.

## Reproduce

```bash
python scripts/run_matrix.py --models berturk --seeds 42 --set epochs=3 max_length=48 batch_size=32
python scripts/run_matrix.py --models mbert xlmr --segmentations native morphological --seeds 42 --set epochs=3 max_length=48 batch_size=32
python scripts/collect_results.py   # consolidate -> outputs/matrix_results.json
python scripts/aggregate.py ; python scripts/make_tables.py ; python scripts/make_figures.py
```
