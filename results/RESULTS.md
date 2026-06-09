# Results (real runs on MASSIVE-tr)

High-quality config (`configs/quality.yaml`): **6 epochs, max_length 64**, batch 32,
lr 5e-5, model selection on dev frame accuracy. Local CPU (AMD Ryzen 7 5800H),
seed 42. Seeds 123/2024 run separately for mean±std.

## Main table — test set (2,974 utterances), seed 42

| Encoder | Segmentation | Intent Acc | Slot F1 | Frame Acc |
|---|---|---|---|---|
| **BERTurk** | **native** | **0.889** | **0.779** | **0.685** |
| BERTurk | morphological | 0.882 | 0.751 | 0.662 |
| BERTurk | character | 0.720 | 0.226 | 0.257 |
| mBERT | native | 0.851 | 0.719 | 0.606 |
| mBERT | morphological | 0.840 | 0.681 | 0.576 |
| XLM-R | native | 0.871 | 0.749 | 0.645 |
| XLM-R | morphological | 0.863 | 0.720 | 0.623 |

BERTurk intent **0.889** and slot F1 **0.779** are at the level of published
MASSIVE-tr systems (slot F1 ~0.74–0.78 is the dataset's honest ceiling).

## Findings

- **Native ≥ morphological for every encoder** (slot F1; native +0.028/+0.038/
  +0.029). Unsupervised morphological pre-segmentation does **not** help; it hurts
  the weakest multilingual mBERT most (−0.038).
- **H1 holds:** intent moves ≤0.011 across segmentations; slot F1 up to 0.038 —
  slot filling is the segmentation-sensitive task.
- **Character is a clear floor** (0.226). **BERTurk (monolingual) leads.**

## Error analysis — slot error rate by affix count (native)

| Affixes | BERTurk | mBERT | XLM-R |
|---|---|---|---|
| 0 | 0.18 | 0.22 | 0.20 |
| 1 | 0.25 | 0.33 | 0.28 |
| 2 | 0.27 | 0.34 | 0.31 |

Errors rise monotonically with morphological complexity for all encoders.
**Morphology is where slot filling fails — but morpheme-aligned pre-segmentation
is not the fix**, pointing at fragment *representations*, not boundary placement.

## On the accuracy ceiling (honest)

Slot F1 ~0.78 is the **MASSIVE-tr** ceiling (18 domains, 55 slot types); published
SOTA sits there too. Intent accuracy ~0.89 is strong. For **>0.90 slot F1**, the
honest route is the easier single-domain **Turkish ATIS / MultiATIS++-tr**
(Büyük 2023: 0.965 intent, 0.916 slot); the loader is in `src/data.py`.

## Reproduce

```bash
python scripts/run_matrix.py --base configs/quality.yaml \
    --models berturk mbert xlmr --segmentations native morphological --seeds 42
python scripts/collect_results.py
python scripts/aggregate.py ; python scripts/make_tables.py ; python scripts/make_figures.py
```
