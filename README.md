# Segmentation Strategies for Turkish Joint Intent Detection & Slot Filling

A morphology-aware evaluation of how **segmentation / tokenization strategies**
affect **joint intent detection and slot filling** in Turkish — an agglutinative
language where a single surface word (e.g. *fatura+lar+ı+mız+dan*) packs a root
plus several suffixes, and slot values live *inside* that structure.

> Research code for a planned **AIST 2026** submission (NLP & Computational
> Linguistics track). No data collection — we use the open, citable
> **MASSIVE-tr** benchmark. The contribution is **analysis and methodology**, not
> data: a controlled comparison of segmentation as an independent variable, a
> fair **BIO label re-alignment** procedure across schemes, and a fine-grained
> **morpheme-boundary error analysis** of slot filling.

## Research questions

- **RQ1** How much does segmentation (native subword / whitespace / morphological
  / char) change Turkish **slot F1**? Is **intent detection** comparatively
  robust? *(H1: intent is robust, slot filling is sensitive.)*
- **RQ2** Does morphological pre-segmentation (Morfessor) beat the model's native
  subwording, especially on **affix-heavy slots**? *(H2: small overall, but
  significant for words with ≥2 affixes.)*
- **RQ3** Do monolingual (BERTurk) vs multilingual (mBERT, XLM-R) encoders react
  differently to the segmentation choice? *(H3: multilingual models, which
  fragment Turkish more, gain more from pre-segmentation.)*
- **RQ4 (methodological)** How do you re-align BIO labels across segmentation
  schemes *fairly*, and how much does mis-alignment distort results?

## What's here

```
src/
  data.py            MASSIVE-tr download + annot_utt -> word-level BIO parser
  segmentation.py    native | whitespace | morphological (Morfessor) | char
  alignment.py       BIO re-projection + mandatory round-trip validation (§8)
  model.py           Joint intent+slot (shared encoder, 2 heads, optional CRF)
  dataset.py         pre-aligned torch dataset + padding collator
  evaluate.py        intent acc, span-level slot F1 (seqeval), frame accuracy
  error_analysis.py  morphology-aware slot error buckets (the headline section)
  train.py           one (model x segmentation x seed) run
scripts/
  download_data.py       fetch MASSIVE-tr
  validate_alignment.py  the §8 guard: 100% label recovery across schemes
  run_matrix.py          sweep models x segmentations x seeds
  aggregate.py           mean±std tables + paired bootstrap significance
  error_report.py        morphological error analysis for a run
configs/                 base.yaml, models.yaml, smoke.yaml
tests/                   sample_tr.jsonl + no-network core tests
```

## Install

```bash
python -m pip install -r requirements.txt
```

Core stack: PyTorch + HuggingFace Transformers + seqeval + Morfessor. The
morphological strategy uses **Morfessor** (unsupervised, pure-Python, fully
reproducible). Zemberek is supported as an optional JVM-based extension (see
*Morphology backends* below).

## Quickstart

```bash
# 0) Sanity: no-network core tests
python tests/test_core.py

# 1) Data
python scripts/download_data.py                      # -> data/raw/tr-TR.jsonl

# 2) Mandatory alignment validation (plan §8) — must be 100% before training
python scripts/validate_alignment.py --data data/raw/tr-TR.jsonl --models berturk

# 3) One run
python -m src.train --config configs/base.yaml \
    --set model_name=dbmdz/bert-base-turkish-cased segmentation=morphological seed=42

# 4) Full matrix (3 models x 3 segmentations x 3 seeds = 27 runs)
python scripts/run_matrix.py

# 5) Tables + significance
python scripts/aggregate.py

# 6) Morphological error analysis for a run
python scripts/error_report.py \
    --predictions outputs/berturk_native_seed42/test_predictions.json \
    --data data/raw/tr-TR.jsonl
```

### CPU smoke test (no GPU, no download)

```bash
python -m src.train --config configs/smoke.yaml
```

Runs the entire pipeline on the bundled synthetic sample with a tiny model, so
you can verify wiring before pushing to a GPU box / Kaggle. See
[`RUNNING.md`](RUNNING.md) for the Kaggle/Colab recipe (the real matrix runs in
minutes per cell on a single T4).

## Method in one paragraph

Every strategy turns text into **pre-tokens** that are then sub-tokenized by the
model's own tokenizer. `native` feeds the raw string (offset-based alignment;
captures SentencePiece cross-whitespace merging). `whitespace`, `morphological`,
and `char` feed pre-split units (`is_split_into_words=True`). Word-level BIO tags
are re-projected onto pre-tokens (first piece keeps `B-`, the rest become `I-`),
and **evaluation always folds predictions back to the surface word** via each
word's *head subword*, so F1 is comparable across schemes. A hard round-trip
guard (`validate_alignment.py`) requires 100% recovery of gold word labels before
any training (plan §8).

## Morphology backends

- **Morfessor** (default): unsupervised, trained on the train-split word types
  only — reproducible from this repo, no external resources.
- **Zemberek** (optional): rule-based Turkish morphology, higher fidelity but
  JVM-based. `src/segmentation.py::ZemberekSegmenter` is a documented extension
  point; wire in a JAR/JPype bridge and register `morphological-zemberek`.

The error analysis uses the Morfessor morpheme count as an affix-count *proxy*;
swap in Zemberek/Zeyrek for exact counts in the camera-ready.

## Reproducibility

Public data (MASSIVE, CC BY 4.0), fixed seed list, deterministic alignment, and
a validation guard. All configs, scripts, and result tables are in-repo. See the
plan (`AIST_2026_plan` / project notes) for the full experimental design.

## License

Code: MIT (see `LICENSE`). Data: MASSIVE is CC BY 4.0 (downloaded separately).
