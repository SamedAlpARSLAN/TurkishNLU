"""Aggregate matrix results into paper-ready tables + significance (plan §7.5).

Reads outputs/matrix_results.json (or scans outputs/*/results.json), reports
mean +/- std over seeds per (model, segmentation), and runs a paired bootstrap
on frame accuracy for the headline comparison (morphological vs native).

    python scripts/aggregate.py
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from src.utils import get_logger

logger = get_logger()
METRICS = ("intent_acc", "slot_f1", "frame_acc")


def load_cells(path: str) -> list[dict]:
    p = Path(path)
    if p.exists():
        return [c for c in json.loads(p.read_text(encoding="utf-8")) if "error" not in c]
    # Fallback: scan per-run result files.
    cells = []
    for rp in Path("outputs").glob("*/results.json"):
        r = json.loads(rp.read_text(encoding="utf-8"))
        cfg = r["config"]
        cells.append({"model": cfg["model_name"].split("/")[-1],
                      "segmentation": cfg["segmentation"], "seed": cfg["seed"],
                      **r["test"]})
    return cells


def summarize(cells: list[dict]) -> dict:
    groups: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for c in cells:
        key = (c["model"], c["segmentation"])
        for m in METRICS:
            if m in c:
                groups[key][m].append(c[m])
    table = {}
    for (model, seg), vals in groups.items():
        table[f"{model} / {seg}"] = {
            m: {"mean": round(float(np.mean(vals[m])), 4),
                "std": round(float(np.std(vals[m])), 4),
                "n": len(vals[m])}
            for m in METRICS if vals[m]
        }
    return table


def markdown_table(table: dict) -> str:
    lines = ["| Model / Segmentation | Intent Acc | Slot F1 | Frame Acc |",
             "|---|---|---|---|"]
    for key in sorted(table):
        row = table[key]
        def cell(m):
            if m not in row:
                return "—"
            return f"{row[m]['mean']:.3f} ± {row[m]['std']:.3f}"
        lines.append(f"| {key} | {cell('intent_acc')} | {cell('slot_f1')} | {cell('frame_acc')} |")
    return "\n".join(lines)


def paired_bootstrap_frame(pred_a: str, pred_b: str, n_boot: int = 10000, seed: int = 0):
    """Paired bootstrap p-value on frame accuracy between two prediction files.

    frame-correct(example) = intent correct AND all slot tags correct.
    Returns (mean_diff (a-b), p_two_sided).
    """
    a = {r["uid"]: r for r in json.loads(Path(pred_a).read_text(encoding="utf-8"))}
    b = {r["uid"]: r for r in json.loads(Path(pred_b).read_text(encoding="utf-8"))}
    uids = sorted(set(a) & set(b))

    def frame(r):
        return int(r["intent_gold"] == r["intent_pred"] and r["slots_gold"] == r["slots_pred"])

    da = np.array([frame(a[u]) for u in uids])
    db = np.array([frame(b[u]) for u in uids])
    diff = da - db
    obs = float(diff.mean())
    rng = np.random.default_rng(seed)
    n = len(diff)
    boots = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    # two-sided p-value against H0: mean diff == 0 (centered bootstrap)
    centered = boots - boots.mean()
    p = float((np.abs(centered) >= abs(obs)).mean())
    return obs, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="outputs/matrix_results.json")
    ap.add_argument("--compare", nargs=2, metavar=("PRED_A", "PRED_B"),
                    help="two test_predictions.json paths for a paired bootstrap")
    args = ap.parse_args()

    cells = load_cells(args.matrix)
    table = summarize(cells)
    md = markdown_table(table)
    print("\n" + md + "\n")
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/results_table.md").write_text(md, encoding="utf-8")
    logger.info("Wrote outputs/results_table.md")

    if args.compare:
        diff, p = paired_bootstrap_frame(*args.compare)
        print(f"\nPaired bootstrap (frame acc): A-B = {diff:+.4f}, p = {p:.4f}")


if __name__ == "__main__":
    main()
