"""Morphology-aware slot error analysis for one run (plan §9).

    python scripts/error_report.py \
        --predictions outputs/berturk_native_seed42/test_predictions.json \
        --data data/raw/tr-TR.jsonl
"""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from src.data import load_examples
from src.error_analysis import analyze, format_report
from src.segmentation import build_segmenter
from src.utils import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--data", default="data/raw/tr-TR.jsonl")
    args = ap.parse_args()

    records = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    train = load_examples(args.data, "train")
    # Morfessor proxy for morpheme counts, fit on the train split (plan §9 note).
    segmenter = build_segmenter(
        "morphological", train_words=[w for ex in train for w in ex.tokens]
    )
    report = analyze(records, train, segmenter)
    print("\n" + format_report(report))

    out = Path(args.predictions).with_name("error_analysis.json")
    write_json(report, out)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
