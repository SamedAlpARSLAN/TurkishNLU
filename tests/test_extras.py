"""Tests for evaluate/model extras (need torch+seqeval; run via pytest locally).

    python -m pytest tests/test_extras.py -q
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.evaluate import repair_bio


def test_repair_bio_fixes_dangling_i():
    # I-X with no preceding B-X/I-X must become B-X
    assert repair_bio(["O", "I-time", "I-time", "O", "I-date"]) == [
        "O", "B-time", "I-time", "O", "B-date"
    ]


def test_repair_bio_fixes_type_switch():
    # I-city right after loc-span is invalid -> B-city
    assert repair_bio(["B-loc", "I-loc", "I-city"]) == ["B-loc", "I-loc", "B-city"]


def test_repair_bio_leaves_valid_unchanged():
    valid = ["O", "B-time", "I-time", "O", "B-date", "I-date"]
    assert repair_bio(valid) == valid
