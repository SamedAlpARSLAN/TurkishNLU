"""Tokenizer morphological diagnostics across encoders (motivation table).

Computes fertility + morpheme-boundary alignment for each encoder on MASSIVE-tr,
using a reference morphological segmenter (Zeyrek rule-based by default, or
Morfessor). No training required. Produces results/tokenizer_metrics.md.

    python scripts/tokenizer_report.py --data data/raw/tr-TR.jsonl --reference zeyrek
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from transformers import AutoTokenizer

from src.data import load_examples
from src.segmentation import build_segmenter
from src.tokenizer_metrics import compute_metrics
from src.utils import get_logger, write_json

logger = get_logger()
MODELS = {
    "berturk": "dbmdz/bert-base-turkish-cased",
    "mbert": "bert-base-multilingual-cased",
    "xlmr": "xlm-roberta-base",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/tr-TR.jsonl")
    ap.add_argument("--reference", default="zeyrek", choices=["zeyrek", "morphological"])
    ap.add_argument("--scope", default="slot", choices=["slot", "all"],
                    help="evaluate on slot-bearing words only, or all words")
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    args = ap.parse_args()

    train = load_examples(args.data, "train")
    if args.scope == "slot":
        words = [w for ex in train for w, t in zip(ex.tokens, ex.slots) if t != "O"]
    else:
        words = [w for ex in train for w in ex.tokens]
    logger.info("Reference=%s scope=%s words=%d", args.reference, args.scope, len(words))

    ref = build_segmenter(
        args.reference, train_words=[w for ex in train for w in ex.tokens]
    )

    rows = {}
    for name in args.models:
        tok = AutoTokenizer.from_pretrained(MODELS[name], use_fast=True)
        m = compute_metrics(words, tok, ref)
        rows[name] = m.row()
        logger.info("%-8s %s", name, m.row())

    # Markdown table
    header = ("| Encoder | Fertility | Ref.cov | Boundary P | Boundary R "
              "| Boundary F1 | Morpheme respect |")
    sep = "|---|---|---|---|---|---|---|"
    lines = [
        f"### Tokenizer morphological alignment (reference: {args.reference}, "
        f"{args.scope} words, type-level)\n", header, sep,
    ]
    for name in args.models:
        r = rows[name]
        lines.append(
            f"| {name} | {r['fertility']} | {r['ref_coverage']} | {r['boundary_P']} "
            f"| {r['boundary_R']} | {r['boundary_F1']} | {r['morpheme_respect']} |"
        )
    md = "\n".join(lines)
    print("\n" + md + "\n")
    Path("results").mkdir(exist_ok=True)
    Path("results/tokenizer_metrics.md").write_text(md + "\n", encoding="utf-8")
    write_json({"reference": args.reference, "scope": args.scope, "rows": rows},
               "results/tokenizer_metrics.json")
    logger.info("Wrote results/tokenizer_metrics.md")


if __name__ == "__main__":
    main()
