"""Training loop for one (model x segmentation x seed) cell of the matrix (§8).

Run a single experiment from a config dict/yaml. The alignment round-trip check
(plan §8) runs as a hard guard before training: a scheme that does not recover
100% of gold word labels aborts the run.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .data import build_label_maps, download_massive, load_examples
from .dataset import JointDataset, make_collate
from .evaluate import evaluate
from .model import JointIntentSlot
from .segmentation import build_segmenter
from .utils import get_device, get_logger, set_seed, write_json

logger = get_logger()


@dataclass
class TrainConfig:
    model_name: str = "dbmdz/bert-base-turkish-cased"
    segmentation: str = "native"
    data_jsonl: str = "data/raw/tr-TR.jsonl"
    auto_download: bool = True
    max_length: int = 64
    batch_size: int = 32
    eval_batch_size: int = 64
    lr: float = 5e-5
    epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    dropout: float = 0.1
    seed: int = 42
    use_crf: bool = False
    slot_loss_weight: float = 1.0
    eval_metric: str = "frame_acc"  # model selection on dev
    output_dir: str = "outputs"
    run_name: str = ""
    limit_train: int | None = None  # cap train size (smoke tests)
    save_model: bool = False
    validate_alignment: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def tag(self) -> str:
        base = self.model_name.split("/")[-1]
        return self.run_name or f"{base}_{self.segmentation}_seed{self.seed}"


def _all_train_words(examples) -> list[str]:
    return [w for ex in examples for w in ex.tokens]


def run_training(cfg: TrainConfig) -> dict:
    set_seed(cfg.seed)
    device = get_device()
    logger.info("Run '%s' on %s", cfg.tag, device)

    # ── Data ────────────────────────────────────────────────────────────────
    data_path = Path(cfg.data_jsonl)
    if cfg.auto_download and not data_path.exists():
        data_path = download_massive(data_path.parent or "data/raw")
    train = load_examples(data_path, "train")
    dev = load_examples(data_path, "dev")
    test = load_examples(data_path, "test")
    if cfg.limit_train:
        train = train[: cfg.limit_train]
    label_maps = build_label_maps(train)
    logger.info("intents=%d slots=%d", label_maps.num_intents, label_maps.num_slots)

    # ── Tokenizer + segmenter ───────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    segmenter = build_segmenter(
        cfg.segmentation, train_words=_all_train_words(train), seed=cfg.seed
    )

    # ── Mandatory alignment guard (plan §8) ─────────────────────────────────
    if cfg.validate_alignment:
        from .alignment import validate_roundtrip

        report = validate_roundtrip(dev, segmenter, tokenizer, cfg.max_length)
        if not report.passed:
            raise RuntimeError(
                f"alignment validation FAILED for '{cfg.segmentation}': "
                f"{report.summary()} — refusing to train (plan §8)."
            )

    # ── Datasets ────────────────────────────────────────────────────────────
    collate = make_collate(tokenizer.pad_token_id)
    ds_train = JointDataset(train, segmenter, tokenizer, label_maps, cfg.max_length)
    ds_dev = JointDataset(dev, segmenter, tokenizer, label_maps, cfg.max_length)
    ds_test = JointDataset(test, segmenter, tokenizer, label_maps, cfg.max_length)
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)
    dl_dev = DataLoader(ds_dev, batch_size=cfg.eval_batch_size, collate_fn=collate)
    dl_test = DataLoader(ds_test, batch_size=cfg.eval_batch_size, collate_fn=collate)

    # ── Model / optim ───────────────────────────────────────────────────────
    model = JointIntentSlot(
        cfg.model_name,
        num_intents=label_maps.num_intents,
        num_slots=label_maps.num_slots,
        dropout=cfg.dropout,
        use_crf=cfg.use_crf,
        slot_loss_weight=cfg.slot_loss_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    total_steps = len(dl_train) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(cfg.warmup_ratio * total_steps), total_steps
    )

    # ── Train ───────────────────────────────────────────────────────────────
    best_metric = -1.0
    best_state = None
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        t0 = time.time()
        for batch in dl_train:
            optimizer.zero_grad()
            out = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                intent_labels=batch["intent_labels"].to(device),
                slot_labels=batch["slot_labels"].to(device),
            )
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
        dev_res = evaluate(model, dl_dev, label_maps, device)
        metric = getattr(dev_res, cfg.eval_metric)
        history.append({"epoch": epoch, "train_loss": running / len(dl_train), "dev": dev_res.as_row()})
        logger.info(
            "epoch %d | loss %.4f | dev %s | %.1fs",
            epoch, running / len(dl_train), dev_res.as_row(), time.time() - t0,
        )
        if metric > best_metric:
            best_metric = metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # ── Test with best dev checkpoint ───────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
    test_res = evaluate(model, dl_test, label_maps, device)
    logger.info("TEST %s | %s", cfg.tag, test_res.as_row())

    # ── Persist ─────────────────────────────────────────────────────────────
    out_dir = Path(cfg.output_dir) / cfg.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "config": asdict(cfg),
        "labels": {"intents": label_maps.num_intents, "slots": label_maps.num_slots},
        "dev_best_metric": best_metric,
        "history": history,
        "test": test_res.as_row(),
    }
    write_json(results, out_dir / "results.json")
    # Stash per-example test predictions for error analysis (§9).
    write_json(
        [
            {
                "uid": r.uid, "tokens": r.tokens,
                "intent_gold": r.intent_gold, "intent_pred": r.intent_pred,
                "slots_gold": r.slots_gold, "slots_pred": r.slots_pred,
            }
            for r in test_res.records
        ],
        out_dir / "test_predictions.json",
    )
    if cfg.save_model:
        torch.save(model.state_dict(), out_dir / "model.pt")
    logger.info("Saved results -> %s", out_dir / "results.json")
    return results


def main():
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, help="YAML config path")
    ap.add_argument("--set", nargs="*", default=[], help="override key=value pairs")
    args = ap.parse_args()

    cfg_dict: dict = {}
    if args.config:
        cfg_dict = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg_dict[k] = yaml.safe_load(v)
    run_training(TrainConfig.from_dict(cfg_dict))


if __name__ == "__main__":
    main()
