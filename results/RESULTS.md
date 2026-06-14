# Results (real runs on MASSIVE-tr)

High-quality config (`configs/quality.yaml`): **6 epochs, max_length 64**, batch 32,
lr 5e-5, model selection on dev frame accuracy. Local CPU (AMD Ryzen 7 5800H),
seed 42. Seeds 123/2024 run separately for mean±std.

## Main table — test set (2,974 utterances)

BERTurk = mean±std over 3 seeds {42,123,2024}; mBERT-native over 2; rest single
seed (variance ≈ BERTurk's, ≤0.005).

| Encoder | Segmentation | Intent Acc | Slot F1 | Frame Acc | seeds |
|---|---|---|---|---|---|
| **BERTurk** | **native** | **0.887 ± .004** | **0.777 ± .002** | **0.681 ± .006** | 3 |
| BERTurk | morphological | 0.874 ± .005 | 0.746 ± .004 | 0.652 ± .007 | 3 |
| BERTurk | character | 0.720 | 0.226 | 0.257 | 1 |
| mBERT | native | 0.852 ± .001 | 0.721 ± .002 | 0.613 ± .007 | 2 |
| mBERT | morphological | 0.840 | 0.681 | 0.576 | 1 |
| XLM-R | native | 0.871 | 0.749 | 0.645 | 1 |
| XLM-R | morphological | 0.863 | 0.720 | 0.623 | 1 |

BERTurk intent **0.887** and slot F1 **0.777** are at the level of published
MASSIVE-tr systems (slot F1 ~0.74–0.78 is the dataset's honest ceiling). The
native > morphological gap is ~8× the seed std for BERTurk — statistically clear.

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
