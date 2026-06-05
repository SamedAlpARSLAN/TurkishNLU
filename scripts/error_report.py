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


def _infer_model(predictions_path: str) -> str | None:
    """Read the sibling results.json to recover which encoder produced these."""
    results = Path(predictions_path).with_name("results.json")
    if results.exists():
        return json.loads(results.read_text(encoding="utf-8"))["config"]["model_name"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--data", default="data/raw/tr-TR.jsonl")
    ap.add_argument("--model", default=None,
                    help="HF tokenizer id for the §9.4 segmentation-shift analysis "
                         "(default: inferred from the run's results.json)")
    args = ap.parse_args()

    records = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    train = load_examples(args.data, "train")
    # Morfessor proxy for morpheme counts, fit on the train split (plan §9 note).
    segmenter = build_segmenter(
        "morphological", train_words=[w for ex in train for w in ex.tokens]
    )

    tokenizer = None
    model_id = args.model or _infer_model(args.predictions)
    if model_id:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    report = analyze(records, train, segmenter, tokenizer=tokenizer)
    print("\n" + format_report(report))

    out = Path(args.predictions).with_name("error_analysis.json")
    write_json(report, out)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
