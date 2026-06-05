"""Run the primary experiment matrix: models x segmentations x seeds (plan §8).

    python scripts/run_matrix.py                      # full 3x3x3 = 27 runs
    python scripts/run_matrix.py --models berturk --seeds 42   # a slice

Each cell is saved under outputs/<tag>/results.json by train.py; this script
also writes a combined outputs/matrix_results.json for aggregate.py.
"""
import argparse
import traceback

import _bootstrap  # noqa: F401
import yaml

from src.train import TrainConfig, run_training
from src.utils import get_logger, write_json

logger = get_logger()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/base.yaml")
    ap.add_argument("--models-config", default="configs/models.yaml")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--segmentations", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[], help="override base key=value")
    args = ap.parse_args()

    base = yaml.safe_load(open(args.base, encoding="utf-8")) or {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        base[k] = yaml.safe_load(v)
    mc = yaml.safe_load(open(args.models_config, encoding="utf-8"))
    available = {**mc.get("models", {}), **mc.get("extra_models", {})}

    models = {k: available[k] for k in (args.models or mc["models"])}
    segmentations = args.segmentations or mc["segmentations"]
    seeds = args.seeds or mc["seeds"]

    logger.info(
        "Matrix: %d models x %d segs x %d seeds = %d runs",
        len(models), len(segmentations), len(seeds),
        len(models) * len(segmentations) * len(seeds),
    )

    combined = []
    for mname, mid in models.items():
        for seg in segmentations:
            for seed in seeds:
                cfg = TrainConfig.from_dict(
                    {**base, "model_name": mid, "segmentation": seg, "seed": seed,
                     "run_name": f"{mname}_{seg}_seed{seed}"}
                )
                try:
                    res = run_training(cfg)
                    combined.append(
                        {"model": mname, "segmentation": seg, "seed": seed,
                         **res["test"], "dev_best": res["dev_best_metric"]}
                    )
                except Exception as exc:  # keep the sweep alive
                    logger.error("Run %s FAILED: %s", cfg.tag, exc)
                    logger.debug(traceback.format_exc())
                    combined.append(
                        {"model": mname, "segmentation": seg, "seed": seed,
                         "error": str(exc)}
                    )
                    write_json(combined, "outputs/matrix_results.json")  # checkpoint
    write_json(combined, "outputs/matrix_results.json")
    logger.info("Wrote outputs/matrix_results.json (%d cells)", len(combined))


if __name__ == "__main__":
    main()
