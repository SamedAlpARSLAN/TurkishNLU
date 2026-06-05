# Segmentation Strategies for Turkish Joint Intent Detection & Slot Filling

[![ci](https://github.com/SamedAlpARSLAN/TurkishNLU/actions/workflows/ci.yml/badge.svg)](https://github.com/SamedAlpARSLAN/TurkishNLU/actions/workflows/ci.yml)

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
  data.py             MASSIVE-tr download + annot_utt -> word-level BIO parser
  segmentation.py     native | whitespace | morphological (Morfessor/Zeyrek) | char
  alignment.py        BIO re-projection + mandatory round-trip validation (§8)
  model.py            Joint intent+slot (shared encoder, 2 heads, optional CRF)
  dataset.py          pre-aligned torch dataset + padding collator
  evaluate.py         intent acc, span-level slot F1 (seqeval), frame accuracy
  tokenizer_metrics.py fertility + morpheme-boundary alignment (motivation)
  error_analysis.py   morphology-aware slot error buckets (the headline section)
  train.py            one (model x segmentation x seed) run
scripts/
  download_data.py       fetch MASSIVE-tr
  corpus_stats.py        dataset descriptive stats (Data section)
  tokenizer_report.py    fertility + boundary alignment per encoder (Table 1)
  validate_alignment.py  the §8 guard: 100% label recovery across schemes
  run_matrix.py          sweep models x segmentations x seeds
  aggregate.py           mean±std tables + bootstrap + McNemar significance
  error_report.py        7-axis morphological error analysis for a run
  make_tables.py         auto-generate LaTeX tables (paper/tables_auto.tex)
  make_figures.py        paper figures (fertility, affix-error, frame heatmap)
  predict.py             load a trained run and tag new Turkish utterances
configs/                 base.yaml, models.yaml, smoke.yaml
results/                 tracked, paper-ready artifacts (alignment, tokenizer, corpus)
paper/                   anonymized Springer LNCS skeleton (main.tex + refs)
tests/                   sample_tr.jsonl + no-network core tests
```

### Training-free analyses (run now, no GPU)

```bash
python scripts/corpus_stats.py        # 60 intents / 55 slots / 18% affixed slot words
python scripts/tokenizer_report.py    # per-encoder fertility + morpheme alignment
python scripts/validate_alignment.py --data data/raw/tr-TR.jsonl --models berturk mbert xlmr
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

- **Morfessor** (`morphological`, default): unsupervised, trained on the
  train-split word types only — reproducible, no external resources, surface-
  faithful (splits ~60% of Turkish word types, e.g. `faturalarımdan ->
  fatura+ları+mdan`).
- **Zeyrek** (`morphological-zeyrek`): pure-Python rule-based Turkish analyzer (a
  Zemberek port). Linguistic morpheme boundaries (`evlerimden -> ev+ler+im+den`);
  conservatively falls back to a single segment under phonological deviation.
  Needs `pip install zeyrek nltk`.
- **Zemberek (JVM)** (optional): the original framework, highest fidelity but
  Java-based — wire in a JAR/JPype bridge if desired.

The error analysis uses the Morfessor morpheme count as an affix-count *proxy*;
swap in Zemberek/Zeyrek for exact counts in the camera-ready.

## Ablations & extras

- **CRF** (`use_crf=true`), **focal slot loss** (`slot_loss=focal`),
  **subword pooling** (`subword_pool=first|mean|max` — how a fragmented word's
  sub-tokens form its representation), and **constrained BIO decoding**
  (`bio_repair=true`) are config flags on any run.
- **Second domain:** `src/data.py::load_atis_format` reads JointBERT-style
  `seq.in`/`seq.out`/`label` splits, so you can drop in **MultiATIS++-tr** (flight
  domain) to test generalizability beyond MASSIVE.
- **Morphology backends:** `morphological` (Morfessor) or `morphological-zeyrek`
  (rule-based) as the segmentation strategy.

## Reproducibility

Public data (MASSIVE, CC BY 4.0), fixed seed list, deterministic alignment, and
a validation guard. All configs, scripts, and result tables are in-repo. See the
plan (`AIST_2026_plan` / project notes) for the full experimental design.

## License

Code: MIT (see `LICENSE`). Data: MASSIVE is CC BY 4.0 (downloaded separately).
