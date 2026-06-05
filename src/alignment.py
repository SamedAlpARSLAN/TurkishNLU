"""BIO label re-projection across segmentation schemes (plan §7.2, AS4).

The independent variable changes the token units, so word-level BIO tags must be
re-projected onto the new units, and — crucially — evaluation must always fold
predictions back to a *common reference unit* (the surface word) so F1 stays
comparable across schemes. This module implements both directions plus the
mandatory round-trip validation from plan §8.

Two encoding paths
------------------
* pre-tokenized (whitespace / morphological / char): segments are fed with
  ``is_split_into_words=True`` and aligned via ``word_ids()``.
* native: the raw string is tokenized and subwords are mapped to source words by
  character-offset overlap (captures SentencePiece cross-whitespace merging).

In both paths each surface word has exactly one *head subword* whose tag is the
word's tag. Reading predictions at the head folds everything back to word level.
"""
from __future__ import annotations

from dataclasses import dataclass

from .data import Example
from .segmentation import Segmenter
from .utils import get_logger

logger = get_logger()

IGNORE = -100  # HF/torch ignore index for the slot loss


# ─────────────────────────────────────────────────────────────────────────────
# B/I-aware projection of a word tag onto its segments
# ─────────────────────────────────────────────────────────────────────────────
def project_tag(word_tag: str, n_segments: int) -> list[str]:
    """Split one word's BIO tag across ``n_segments`` pre-tokens.

    B-X -> [B-X, I-X, I-X, ...]; I-X -> [I-X, ...]; O -> [O, ...]. This keeps the
    span semantics intact (exactly one B per span, no spurious boundaries).
    """
    if word_tag == "O":
        return ["O"] * n_segments
    suffix = word_tag[2:]
    if word_tag.startswith("B-"):
        return ["B-" + suffix] + ["I-" + suffix] * (n_segments - 1)
    if word_tag.startswith("I-"):
        return ["I-" + suffix] * n_segments
    raise ValueError(f"malformed BIO tag: {word_tag!r}")


def segment_with_tags(
    words: list[str], word_tags: list[str], segmenter: Segmenter
) -> tuple[list[str], list[str], list[int]]:
    """Return (pretokens, pretoken_tags, pretoken_word_index)."""
    pretokens: list[str] = []
    ptags: list[str] = []
    pword_idx: list[int] = []
    for wi, (word, tag) in enumerate(zip(words, word_tags)):
        segs = segmenter.segment_word(word)
        for seg, ptag in zip(segs, project_tag(tag, len(segs))):
            pretokens.append(seg)
            ptags.append(ptag)
            pword_idx.append(wi)
    return pretokens, ptags, pword_idx


# ─────────────────────────────────────────────────────────────────────────────
# Aligned example
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Aligned:
    input_ids: list[int]
    attention_mask: list[int]
    slot_tags: list[str | None]  # per subword; None -> IGNORE in loss
    word_head_pos: list[int]  # per source word -> subword index of its head; -1 if lost
    n_words: int

    @property
    def n_truncated_words(self) -> int:
        return sum(1 for p in self.word_head_pos if p < 0)


def _assign_word_by_overlap(a: int, b: int, spans: list[tuple[int, int]]) -> int:
    """Map a subword char span [a,b) to the source word with the most overlap."""
    best, best_ov = -1, 0
    for wi, (ws, we) in enumerate(spans):
        ov = min(b, we) - max(a, ws)
        if ov > best_ov:
            best, best_ov = wi, ov
    return best


def align_example(
    words: list[str],
    word_tags: list[str],
    segmenter: Segmenter,
    tokenizer,
    max_length: int = 128,
) -> Aligned:
    """Encode one example under a segmentation strategy, with word-level heads."""
    n_words = len(words)

    if segmenter.requires_pretokenization:
        pretokens, ptags, pword_idx = segment_with_tags(words, word_tags, segmenter)
        enc = tokenizer(
            pretokens,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
        )
        word_ids = enc.word_ids()

        slot_tags: list[str | None] = []
        pretoken_head: dict[int, int] = {}
        seen: set[int] = set()
        for sub_pos, pid in enumerate(word_ids):
            if pid is None:
                slot_tags.append(None)
            elif pid not in seen:
                seen.add(pid)
                pretoken_head[pid] = sub_pos
                slot_tags.append(ptags[pid])
            else:
                slot_tags.append(None)

        # A word's head = the head subword of its FIRST pre-token.
        word_first_pretoken: dict[int, int] = {}
        for pid, wi in enumerate(pword_idx):
            word_first_pretoken.setdefault(wi, pid)
        word_head_pos = [
            pretoken_head.get(word_first_pretoken.get(wi, -1), -1)
            for wi in range(n_words)
        ]
    else:
        # native: raw string + offset overlap
        text = " ".join(words)
        spans: list[tuple[int, int]] = []
        cursor = 0
        for w in words:
            start = text.index(w, cursor)
            spans.append((start, start + len(w)))
            cursor = start + len(w)

        enc = tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length,
        )
        slot_tags = []
        word_head_pos = [-1] * n_words
        seen_word: set[int] = set()
        for sub_pos, (a, b) in enumerate(enc["offset_mapping"]):
            if a == b:  # special token
                slot_tags.append(None)
                continue
            wi = _assign_word_by_overlap(a, b, spans)
            if wi == -1:
                slot_tags.append(None)
            elif wi not in seen_word:
                seen_word.add(wi)
                word_head_pos[wi] = sub_pos
                slot_tags.append(word_tags[wi])
            else:
                slot_tags.append(None)

    return Aligned(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        slot_tags=slot_tags,
        word_head_pos=word_head_pos,
        n_words=n_words,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fold predictions back to word level (the common reference unit)
# ─────────────────────────────────────────────────────────────────────────────
def fold_to_words(aligned: Aligned, subword_tags: list[str]) -> list[str]:
    """Read predictions at each word's head subword -> word-level BIO sequence.

    Truncated words (head lost) default to ``O`` so lengths still match gold.
    """
    out = []
    for pos in aligned.word_head_pos:
        out.append(subword_tags[pos] if 0 <= pos < len(subword_tags) else "O")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Mandatory round-trip validation (plan §8)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ValidationReport:
    n_examples: int
    n_words: int
    n_mismatch: int
    n_truncated_words: int

    @property
    def passed(self) -> bool:
        return self.n_mismatch == 0

    def summary(self) -> str:
        rate = 100.0 * (self.n_words - self.n_mismatch) / max(self.n_words, 1)
        return (
            f"round-trip {rate:.4f}% exact | mismatches={self.n_mismatch} | "
            f"truncated_words={self.n_truncated_words} / {self.n_words} | "
            f"examples={self.n_examples} | {'PASS' if self.passed else 'FAIL'}"
        )


def validate_roundtrip(
    examples: list[Example],
    segmenter: Segmenter,
    tokenizer,
    max_length: int = 128,
) -> ValidationReport:
    """Project gold tags through the scheme, fold back, require 100% recovery.

    A scheme that does not pass must not be used for experiments (plan §8).
    Truncated words are reported separately (they reflect ``max_length``, not a
    projection bug) and excluded from the mismatch count.
    """
    n_words = n_mismatch = n_trunc = 0
    for ex in examples:
        aligned = align_example(ex.tokens, ex.slots, segmenter, tokenizer, max_length)
        # Gold "dense" subword tags = what we'd train on, with heads carrying tags.
        gold_dense = ["O"] * len(aligned.input_ids)
        for pos, tag in zip(aligned.word_head_pos, ex.slots):
            if 0 <= pos < len(gold_dense):
                gold_dense[pos] = tag
        recovered = fold_to_words(aligned, gold_dense)
        for wi, (gold, got) in enumerate(zip(ex.slots, recovered)):
            n_words += 1
            if aligned.word_head_pos[wi] < 0:
                n_trunc += 1
                continue
            if gold != got:
                n_mismatch += 1
    report = ValidationReport(len(examples), n_words, n_mismatch, n_trunc)
    level = logger.info if report.passed else logger.error
    level("[%s] %s", segmenter.name, report.summary())
    return report
