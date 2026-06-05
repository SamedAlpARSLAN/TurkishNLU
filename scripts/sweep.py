"""Lightweight hyperparameter sweep for one (model, segmentation) cell.

Use this on Kaggle to tune lr / epochs / batch before launching the full matrix.

    python scripts/sweep.py --model dbmdz/bert-base-turkish-cased \
        --segmentation morphological --lr 2e-5 3e-5 5e-5 --epochs 3 5
"""
import argparse
import itertools
from pathlib import Path

import _bootstrap  # noqa: F401

from src.train import TrainConfig, run_training
from src.utils import get_logger, write_json

logger = get_logger()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dbmdz/bert-base-turkish-cased")
    ap.add_argument("--segmentation", default="morphological")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", nargs="*", type=float, default=[3e-5, 5e-5])
    ap.add_argument("--epochs", nargs="*", type=int, default=[5])
    ap.add_argument("--batch", nargs="*", type=int, default=[32])
    ap.add_argument("--data", default="data/raw/tr-TR.jsonl")
    args = ap.parse_args()

    grid = list(itertools.product(args.lr, args.epochs, args.batch))
    logger.info("Sweep: %d configurations", len(grid))
    rows = []
    for lr, epochs, batch in grid:
        cfg = TrainConfig.from_dict({
            "model_name": args.model, "segmentation": args.segmentation,
            "seed": args.seed, "lr": lr, "epochs": epochs, "batch_size": batch,
            "data_jsonl": args.data, "output_dir": "outputs/sweep",
            "run_name": f"lr{lr}_ep{epochs}_bs{batch}",
        })
        res = run_training(cfg)
        rows.append({"lr": lr, "epochs": epochs, "batch": batch,
                     "dev_best": res["dev_best_metric"], **res["test"]})
        logger.info("lr=%s ep=%s bs=%s -> dev %.4f | test %s",
                    lr, epochs, batch, res["dev_best_metric"], res["test"])

    rows.sort(key=lambda r: r["dev_best"], reverse=True)
    print("\nBest configs by dev frame accuracy:")
    for r in rows[:5]:
        print(f"  lr={r['lr']} ep={r['epochs']} bs={r['batch']} "
              f"| dev={r['dev_best']:.4f} test_frame={r.get('frame_acc')}")
    Path("outputs").mkdir(exist_ok=True)
    write_json(rows, "outputs/sweep/sweep_results.json")
    logger.info("best: %s", rows[0] if rows else None)


if __name__ == "__main__":
    main()
