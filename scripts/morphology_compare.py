"""Compare the two morphological backends: unsupervised Morfessor vs rule-based
Zeyrek, on MASSIVE-tr slot words (training-free). Quantifies how far the
reproducible unsupervised proxy is from the linguistic analysis — a methodological
point for the paper (§7.1 ambiguity note, Limitations).

    python scripts/morphology_compare.py --data data/raw/tr-TR.jsonl
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from src.data import load_examples
from src.error_analysis import morpheme_cutpoints
from src.segmentation import build_segmenter
from src.utils import get_logger, write_json

logger = get_logger()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/tr-TR.jsonl")
    ap.add_argument("--scope", default="slot", choices=["slot", "all"])
    args = ap.parse_args()

    train = load_examples(args.data, "train")
    if args.scope == "slot":
        words = {w for ex in train for w, t in zip(ex.tokens, ex.slots) if t != "O"}
    else:
        words = {w for ex in train for w in ex.tokens}

    morf = build_segmenter("morphological", train_words=[w for ex in train for w in ex.tokens])
    zey = build_segmenter("morphological-zeyrek")

    n = exact = both_multi = 0
    morf_morphs = zey_morphs = 0
    correct_b = pred_b = gold_b = 0  # Morfessor boundaries vs Zeyrek (reference)
    zey_segmentable = 0

    for w in words:
        if not w:
            continue
        n += 1
        ms, zs = morf.segment_word(w), zey.segment_word(w)
        morf_morphs += len(ms)
        zey_morphs += len(zs)
        if ms == zs:
            exact += 1
        if len(ms) > 1 and len(zs) > 1:
            both_multi += 1
        mcut = morpheme_cutpoints(w, morf)
        zcut = morpheme_cutpoints(w, zey)
        if zcut:  # Zeyrek gives a linguistic boundary to compare against
            zey_segmentable += 1
            correct_b += len(mcut & zcut)
            pred_b += len(mcut)
            gold_b += len(zcut)

    p = correct_b / pred_b if pred_b else 0.0
    r = correct_b / gold_b if gold_b else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    report = {
        "scope": args.scope,
        "n_word_types": n,
        "exact_agreement": round(exact / n, 3),
        "both_segment": round(both_multi / n, 3),
        "zeyrek_coverage": round(zey_segmentable / n, 3),
        "avg_morphs_morfessor": round(morf_morphs / n, 3),
        "avg_morphs_zeyrek": round(zey_morphs / n, 3),
        "morfessor_vs_zeyrek_boundary_P": round(p, 3),
        "morfessor_vs_zeyrek_boundary_R": round(r, 3),
        "morfessor_vs_zeyrek_boundary_F1": round(f1, 3),
    }
    for k, v in report.items():
        print(f"  {k:34} {v}")
    Path("results").mkdir(exist_ok=True)
    write_json(report, "results/morphology_compare.json")
    logger.info("Wrote results/morphology_compare.json")


if __name__ == "__main__":
    main()
