"""Consolidate every per-run outputs/*/results.json into one clean
outputs/matrix_results.json (short model names), so aggregate/make_tables/
make_figures see the full matrix even across multiple run_matrix invocations.

    python scripts/collect_results.py
"""
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from src.utils import get_logger, write_json

logger = get_logger()


def main():
    cells = []
    for rp in sorted(Path("outputs").glob("*/results.json")):
        r = json.loads(rp.read_text(encoding="utf-8"))
        cfg = r["config"]
        run_name = cfg.get("run_name") or ""
        model = run_name.split("_")[0] if run_name else cfg["model_name"].split("/")[-1]
        cells.append({
            "model": model,
            "segmentation": cfg["segmentation"],
            "seed": cfg["seed"],
            **r["test"],
            "dev_best": r.get("dev_best_metric"),
        })
    write_json(cells, "outputs/matrix_results.json")
    logger.info("Collected %d run(s) -> outputs/matrix_results.json", len(cells))
    for c in cells:
        print(f"  {c['model']:9} {c['segmentation']:14} seed{c['seed']} "
              f"| intent {c.get('intent_acc')} slotF1 {c.get('slot_f1')} frame {c.get('frame_acc')}")


if __name__ == "__main__":
    main()
