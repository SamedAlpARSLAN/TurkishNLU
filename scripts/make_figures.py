"""Generate publication figures from the result JSONs (paper-ready PDF + PNG).

    python scripts/make_figures.py

Figures (each skipped gracefully if its input is missing):
  fertility.pdf      per-encoder fertility + morpheme-respect (results/tokenizer_metrics.json)
  affix_error.pdf    slot error rate vs affix count (outputs/*/error_analysis.json)
  frame_heatmap.pdf  segmentation x model frame accuracy (outputs/matrix_results.json)
"""
import json
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.utils import get_logger  # noqa: E402

logger = get_logger()
OUT = Path("results/figures")


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=160)
    plt.close(fig)
    logger.info("wrote %s", OUT / f"{name}.pdf")


def fig_fertility():
    p = Path("results/tokenizer_metrics.json")
    if not p.exists():
        return
    rows = json.loads(p.read_text(encoding="utf-8"))["rows"]
    names = list(rows)
    fert = [rows[n]["fertility"] for n in names]
    resp = [rows[n]["morpheme_respect"] for n in names]
    fig, ax1 = plt.subplots(figsize=(5, 3.2))
    x = range(len(names))
    ax1.bar([i - 0.2 for i in x], fert, width=0.4, label="fertility", color="#4C72B0")
    ax1.set_ylabel("fertility (subwords/word)")
    ax2 = ax1.twinx()
    ax2.bar([i + 0.2 for i in x], resp, width=0.4, label="morpheme respect", color="#DD8452")
    ax2.set_ylabel("morpheme respect")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(names)
    ax1.set_title("Native-tokenizer morphology (MASSIVE-tr slot words)")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.95), fontsize=8)
    _save(fig, "fertility")


def fig_affix_error():
    files = sorted(Path("outputs").glob("**/error_analysis.json"))
    if not files:
        return
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for f in files:
        rep = json.loads(f.read_text(encoding="utf-8"))
        buckets = rep.get("by_affix_count", [])
        if not buckets:
            continue
        xs = [b["bucket"] for b in buckets]
        ys = [b["error_rate"] for b in buckets]
        ax.plot(xs, ys, marker="o", label=f.parent.name)
    ax.set_xlabel("affix count")
    ax.set_ylabel("slot-word error rate")
    ax.set_title("Slot error rate vs morphological complexity")
    if len(files) > 1:
        ax.legend(fontsize=7)
    _save(fig, "affix_error")


def fig_frame_heatmap():
    p = Path("outputs/matrix_results.json")
    if not p.exists():
        return
    cells = [c for c in json.loads(p.read_text(encoding="utf-8")) if "frame_acc" in c]
    if not cells:
        return
    agg = defaultdict(list)
    for c in cells:
        agg[(c["model"], c["segmentation"])].append(c["frame_acc"])
    models = sorted({m for m, _ in agg})
    segs = sorted({s for _, s in agg})
    grid = [[sum(agg[(m, s)]) / len(agg[(m, s)]) if agg[(m, s)] else float("nan")
             for s in segs] for m in models]
    fig, ax = plt.subplots(figsize=(1.2 * len(segs) + 2, 0.8 * len(models) + 2))
    im = ax.imshow(grid, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(segs)), segs)
    ax.set_yticks(range(len(models)), models)
    for i in range(len(models)):
        for j in range(len(segs)):
            ax.text(j, i, f"{grid[i][j]:.3f}", ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(im, label="frame accuracy")
    ax.set_title("Frame accuracy: segmentation x model")
    _save(fig, "frame_heatmap")


def main():
    fig_fertility()
    fig_affix_error()
    fig_frame_heatmap()
    if not OUT.exists():
        logger.warning("no figures produced — run tokenizer_report / matrix first")


if __name__ == "__main__":
    main()
