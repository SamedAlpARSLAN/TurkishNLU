"""Tokenizer-level morphological diagnostics (motivation analysis).

Descriptive metrics that quantify *how* each encoder's subword tokenizer relates
to Turkish morphology, computed without any training. They motivate the study
and support H3 (multilingual encoders fragment Turkish more / align worse).

Metrics (following the tokenizer-evaluation literature for morphologically rich
languages):
  - fertility: mean number of subwords per word (compression / fragmentation).
  - boundary precision/recall/F1: do the tokenizer's subword boundaries match
    morpheme boundaries from a reference analyzer (Zeyrek/Morfessor)?
        P = correct_pred_boundaries / all_pred_boundaries
        R = correct_pred_boundaries / all_gold_boundaries
  - morpheme_respect: fraction of words whose subword cuts are a subset of the
    morpheme cuts (the tokenizer never splits *inside* a morpheme).

Boundary metrics measure agreement with the chosen reference segmenter; with a
rule-based reference (Zeyrek) they approximate morphological correctness, with an
unsupervised one (Morfessor) they measure agreement with data-driven morphology.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .error_analysis import morpheme_cutpoints, subword_cutpoints
from .segmentation import Segmenter


@dataclass
class TokenizerMetrics:
    n_words: int
    fertility: float
    ref_coverage: float  # fraction of words the reference actually segments
    boundary_precision: float
    boundary_recall: float
    boundary_f1: float
    morpheme_respect: float

    def row(self) -> dict:
        return {
            "n_words": self.n_words,
            "fertility": round(self.fertility, 3),
            "ref_coverage": round(self.ref_coverage, 3),
            "boundary_P": round(self.boundary_precision, 3),
            "boundary_R": round(self.boundary_recall, 3),
            "boundary_F1": round(self.boundary_f1, 3),
            "morpheme_respect": round(self.morpheme_respect, 3),
        }


def native_vs_whitespace_divergence(sentences: list[list[str]], tokenizer) -> dict:
    """Quantify how often raw-string (native) tokenization differs from per-word
    (whitespace) tokenization — i.e. cross-whitespace merging by SentencePiece.

    This justifies treating ``native`` and ``whitespace`` as distinct strategies:
    for WordPiece models they coincide (~0 divergence), for SentencePiece/BPE the
    tokenizer can merge across spaces, so the schemes genuinely differ.
    """
    n_diff = 0
    tok_saved = 0
    total = 0
    for words in sentences:
        if not words:
            continue
        total += 1
        native = tokenizer(" ".join(words), add_special_tokens=False)["input_ids"]
        per_word = sum(
            len(tokenizer(w, add_special_tokens=False)["input_ids"]) for w in words
        )
        if len(native) != per_word:
            n_diff += 1
        tok_saved += per_word - len(native)
    return {
        "sentences": total,
        "divergent_pct": round(100 * n_diff / total, 2) if total else 0.0,
        "avg_tokens_saved": round(tok_saved / total, 3) if total else 0.0,
    }


def compute_metrics(
    words: list[str], tokenizer, reference: Segmenter, dedup: bool = True
) -> TokenizerMetrics:
    """Compute fertility + boundary alignment against a reference segmenter.

    Fertility is over all words. Boundary P/R/F1 and morpheme-respect are computed
    only over words the reference actually segments (≥1 morpheme boundary), since
    boundary alignment is undefined where no gold morpheme boundary exists; the
    fraction of such words is reported as ``ref_coverage`` for transparency.

    ``dedup`` evaluates each word *type* once (type-level), the convention for
    intrinsic tokenizer metrics; set False for token-weighted figures.
    """
    items = Counter(words) if not dedup else {w: 1 for w in set(words)}

    n_words = 0
    n_subtokens = 0
    n_segmentable = 0
    correct_b = pred_b = gold_b = 0
    respect_n = respect_total = 0

    for word, weight in items.items():
        if not word:
            continue
        n_words += weight
        sub = subword_cutpoints(word, tokenizer)
        n_subtokens += (len(sub) + 1) * weight  # boundaries+1 = #subwords
        gold = morpheme_cutpoints(word, reference)
        if not gold:  # reference left the word whole -> no boundary to judge
            continue
        n_segmentable += weight
        correct_b += len(sub & gold) * weight
        pred_b += len(sub) * weight
        gold_b += len(gold) * weight
        if sub:
            respect_total += weight
            respect_n += int(sub.issubset(gold)) * weight

    p = correct_b / pred_b if pred_b else 0.0
    r = correct_b / gold_b if gold_b else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return TokenizerMetrics(
        n_words=n_words,
        fertility=n_subtokens / n_words if n_words else 0.0,
        ref_coverage=n_segmentable / n_words if n_words else 0.0,
        boundary_precision=p,
        boundary_recall=r,
        boundary_f1=f1,
        morpheme_respect=respect_n / respect_total if respect_total else 0.0,
    )
