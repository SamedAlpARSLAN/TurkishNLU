"""Descriptive statistics for MASSIVE-tr (paper's Data section).

Reports split sizes, label counts, utterance length, and — the morphologically
interesting part — the affix-count distribution of slot-bearing words (via the
Morfessor proxy). No training. Writes results/corpus_stats.md.

    python scripts/corpus_stats.py --data data/raw/tr-TR.jsonl
"""
import argparse
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401

from src.data import build_label_maps, load_examples
from src.error_analysis import affix_count
from src.segmentation import build_segmenter
from src.utils import get_logger, write_json

logger = get_logger()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/tr-TR.jsonl")
    args = ap.parse_args()

    splits = {p: load_examples(args.data, p) for p in ("train", "dev", "test")}
    train = splits["train"]
    lm = build_label_maps(train)

    # affix-count distribution over slot-bearing words (Morfessor proxy)
    seg = build_segmenter("morphological", train_words=[w for ex in train for w in ex.tokens])
    slot_words = [w for ex in train for w, t in zip(ex.tokens, ex.slots) if t != "O"]
    affix_dist = Counter(min(affix_count(w, seg), 3) for w in slot_words)
    n_slot = sum(affix_dist.values()) or 1

    avg_len = sum(len(ex.tokens) for ex in train) / max(len(train), 1)
    scenarios = Counter(ex.scenario for ex in train)

    lines = ["## MASSIVE-tr corpus statistics\n",
             "| Split | Utterances |", "|---|---|"]
    for p, exs in splits.items():
        lines.append(f"| {p} | {len(exs)} |")
    lines += [
        "",
        f"- Intents: **{lm.num_intents}**  |  Slot types: "
        f"**{len([s for s in lm.slot2id if s.startswith('B-')])}**  |  "
        f"BIO labels: **{lm.num_slots}**",
        f"- Domains/scenarios: **{len(scenarios)}**  |  Avg tokens/utterance: "
        f"**{avg_len:.2f}**",
        f"- Slot-bearing words (train): **{n_slot}**",
        "",
        "### Affix-count distribution of slot-bearing words (Morfessor proxy)\n",
        "| Affixes | Share | Count |", "|---|---|---|",
    ]
    for k in (0, 1, 2, 3):
        label = "3+" if k == 3 else str(k)
        lines.append(f"| {label} | {100*affix_dist.get(k,0)/n_slot:.1f}% | {affix_dist.get(k,0)} |")

    multi = 100 * sum(v for k, v in affix_dist.items() if k >= 1) / n_slot
    lines += ["", f"**{multi:.1f}%** of slot-bearing words carry ≥1 affix — the "
              "agglutinative pressure this study targets."]

    md = "\n".join(lines)
    print("\n" + md + "\n")
    Path("results").mkdir(exist_ok=True)
    Path("results/corpus_stats.md").write_text(md + "\n", encoding="utf-8")
    write_json(
        {"splits": {p: len(e) for p, e in splits.items()},
         "intents": lm.num_intents,
         "slot_types": len([s for s in lm.slot2id if s.startswith("B-")]),
         "avg_tokens": avg_len,
         "affix_distribution": {str(k): affix_dist.get(k, 0) for k in range(4)}},
        "results/corpus_stats.json",
    )
    logger.info("Wrote results/corpus_stats.md")


if __name__ == "__main__":
    main()
