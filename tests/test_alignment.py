"""Alignment tests with a real fast tokenizer (downloads a tiny model once).

Skipped automatically if transformers or the model download is unavailable, so
the no-network `test_core.py` remains the offline guarantee.

Run:  python -m pytest tests/test_alignment.py -q
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.alignment import align_example, fold_to_words, validate_roundtrip
from src.data import load_examples
from src.segmentation import build_segmenter

TOK_ID = "prajjwal1/bert-tiny"


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(TOK_ID, use_fast=True)
    except Exception as exc:  # offline / hub down
        pytest.skip(f"tokenizer unavailable: {exc}")


@pytest.fixture(scope="module")
def examples():
    return load_examples(pathlib.Path(__file__).with_name("sample_tr.jsonl"), "train")


@pytest.mark.parametrize("strategy", ["native", "whitespace", "morphological", "char"])
def test_roundtrip_100pct(tokenizer, examples, strategy):
    words = [w for ex in examples for w in ex.tokens]
    seg = build_segmenter(strategy, train_words=words)
    report = validate_roundtrip(examples, seg, tokenizer, max_length=64)
    assert report.passed, report.summary()
    assert report.n_mismatch == 0


def test_head_carries_word_tag(tokenizer, examples):
    """Folding gold subword tags at the heads must reproduce gold word tags."""
    seg = build_segmenter("char", train_words=["x"])  # finest split, hardest case
    ex = examples[1]  # multi-token slot ("her gün")
    aligned = align_example(ex.tokens, ex.slots, seg, tokenizer, max_length=64)
    dense = ["O"] * len(aligned.input_ids)
    for pos, tag in zip(aligned.word_head_pos, ex.slots):
        if pos >= 0:
            dense[pos] = tag
    assert fold_to_words(aligned, dense) == ex.slots


def test_native_uses_offsets_not_pretokenization(tokenizer):
    seg = build_segmenter("native")
    assert seg.requires_pretokenization is False
