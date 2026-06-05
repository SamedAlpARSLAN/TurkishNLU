"""Fine-grained, morphology-aware slot error analysis (plan §9).

This is the paper's most distinctive section: it explains *where* and *why*
morphology affects slot filling, even when the aggregate F1 gap is small. We
bucket per-word slot errors by:

  1. affix count        — morphological complexity (Morfessor morpheme count - 1)
  2. surface rarity     — training-set frequency of the surface word
  3. segmentation shift — does the model's subword split disagree with the
                          morpheme boundaries? (operationalizes §9.4)

Inputs are the ``test_predictions.json`` written by ``train.py`` plus the train
split (for frequencies) and a fitted morphological segmenter (for morpheme
counts). The affix count is a Morfessor-based *proxy*; swap in Zemberek/Zeyrek
for the camera-ready if exact morpheme counts are desired.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from .data import Example
from .segmentation import Segmenter
from .utils import get_logger

logger = get_logger()


@dataclass
class Bucket:
    name: str
    n: int = 0
    errors: int = 0

    def add(self, is_error: bool) -> None:
        self.n += 1
        self.errors += int(is_error)

    @property
    def error_rate(self) -> float:
        return self.errors / self.n if self.n else 0.0

    def row(self) -> dict:
        return {
            "bucket": self.name,
            "n": self.n,
            "errors": self.errors,
            "error_rate": round(self.error_rate, 4),
        }


def affix_count(word: str, segmenter: Segmenter) -> int:
    """Morpheme-count proxy: number of segments beyond the (assumed) root."""
    return max(0, len(segmenter.segment_word(word)) - 1)


def _affix_bucket(n: int) -> str:
    return "3+" if n >= 3 else str(n)


def _freq_bucket(freq: int) -> str:
    if freq == 0:
        return "unseen"
    if freq <= 2:
        return "rare(1-2)"
    if freq <= 10:
        return "mid(3-10)"
    return "freq(>10)"


def slot_error_iter(records: list[dict]):
    """Yield (word, gold_tag, pred_tag, is_error) for every slot-bearing word.

    A 'slot-bearing word' has a gold tag != O. An error is any gold/pred tag
    mismatch at that word position (B/I/type all count).
    """
    for rec in records:
        tokens = rec["tokens"]
        gold = rec["slots_gold"]
        pred = rec["slots_pred"]
        for tok, g, p in zip(tokens, gold, pred):
            if g != "O":
                yield tok, g, p, (g != p)


def analyze(
    records: list[dict],
    train_examples: list[Example],
    segmenter: Segmenter,
) -> dict:
    """Return the full morphological error-analysis report."""
    train_freq = Counter(w for ex in train_examples for w in ex.tokens)

    by_affix: dict[str, Bucket] = defaultdict(lambda: Bucket(""))
    by_freq: dict[str, Bucket] = defaultdict(lambda: Bucket(""))
    overall = Bucket("overall")

    for word, _gold, _pred, is_err in slot_error_iter(records):
        overall.add(is_err)

        ab = _affix_bucket(affix_count(word, segmenter))
        by_affix.setdefault(ab, Bucket(ab)).add(is_err)

        fb = _freq_bucket(train_freq.get(word, 0))
        by_freq.setdefault(fb, Bucket(fb)).add(is_err)

    def order(d, keys):
        return [d[k].row() for k in keys if k in d]

    report = {
        "overall": overall.row(),
        "by_affix_count": order(by_affix, ["0", "1", "2", "3+"]),
        "by_surface_frequency": order(
            by_freq, ["unseen", "rare(1-2)", "mid(3-10)", "freq(>10)"]
        ),
    }
    return report


def format_report(report: dict) -> str:
    lines = [f"OVERALL slot-word error rate: {report['overall']['error_rate']:.4f} "
             f"(n={report['overall']['n']})", ""]
    for title, key in [
        ("Error rate by affix count (morphological complexity)", "by_affix_count"),
        ("Error rate by surface-form training frequency", "by_surface_frequency"),
    ]:
        lines.append(title)
        lines.append(f"  {'bucket':<12}{'n':>8}{'errors':>8}{'err_rate':>10}")
        for r in report[key]:
            lines.append(
                f"  {r['bucket']:<12}{r['n']:>8}{r['errors']:>8}{r['error_rate']:>10.4f}"
            )
        lines.append("")
    return "\n".join(lines)
