"""Mandatory BIO re-alignment validation across schemes (plan §8, AS4).

For each (tokenizer x segmentation) pair, project gold word tags through the
scheme and fold them back: the recovery must be 100%. Any failing scheme exits
non-zero and must not be used in experiments.

    # full check on real data with the Turkish tokenizer
    python scripts/validate_alignment.py --data data/raw/tr-TR.jsonl

    # offline smoke check on the bundled sample with a tiny tokenizer
    python scripts/validate_alignment.py --data tests/sample_tr.jsonl \
        --tokenizer prajjwal1/bert-tiny --strategies native whitespace char
"""
import argparse
import sys

import _bootstrap  # noqa: F401
from transformers import AutoTokenizer

from src.alignment import validate_roundtrip
from src.data import load_examples
from src.segmentation import ALL_STRATEGIES, build_segmenter

MODEL_SET = {
    "berturk": "dbmdz/bert-base-turkish-cased",
    "mbert": "bert-base-multilingual-cased",
    "xlmr": "xlm-roberta-base",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/tr-TR.jsonl")
    ap.add_argument("--partition", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--tokenizer", default=None, help="single HF tokenizer id")
    ap.add_argument("--models", nargs="*", default=None, help="named set: berturk mbert xlmr")
    ap.add_argument("--strategies", nargs="*", default=list(ALL_STRATEGIES))
    ap.add_argument("--max-length", type=int, default=64)
    args = ap.parse_args()

    examples = load_examples(args.data, args.partition)
    train_words = [w for ex in examples for w in ex.tokens]  # for morfessor fit

    if args.tokenizer:
        tokenizers = {args.tokenizer: args.tokenizer}
    else:
        names = args.models or ["berturk"]
        tokenizers = {n: MODEL_SET[n] for n in names}

    all_pass = True
    print(f"\nValidating {len(examples)} '{args.partition}' examples\n" + "=" * 70)
    for tname, tid in tokenizers.items():
        tok = AutoTokenizer.from_pretrained(tid, use_fast=True)
        for strat in args.strategies:
            seg = build_segmenter(strat, train_words=train_words)
            rep = validate_roundtrip(examples, seg, tok, args.max_length)
            status = "PASS" if rep.passed else "FAIL"
            print(f"[{tname:<7}] {strat:<14} {status}  ({rep.summary()})")
            all_pass &= rep.passed
    print("=" * 70)
    print("ALL PASS" if all_pass else "SOME FAILED")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
