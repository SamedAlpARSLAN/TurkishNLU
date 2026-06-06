"""Inference: load a trained run and predict intent + slots for Turkish text.

    python scripts/predict.py --run outputs/berturk_morphological_seed42 \
        --text "yarın sabah dokuzda alarm kur" --data data/raw/tr-TR.jsonl

The run directory must contain model.pt (train with save_model=true), plus the
results.json (config) and label_maps.json written by train.py. --data is only
needed when the run used a morphological segmentation (to refit Morfessor).
"""
import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import torch
from transformers import AutoTokenizer

from src.alignment import align_example
from src.data import load_examples
from src.evaluate import fold_pred_to_words, repair_bio
from src.model import JointIntentSlot
from src.segmentation import build_segmenter
from src.utils import get_device


def load_run(run_dir: Path):
    cfg = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))["config"]
    maps = json.loads((run_dir / "label_maps.json").read_text(encoding="utf-8"))
    id2slot = {v: k for k, v in maps["slot2id"].items()}
    id2intent = {v: k for k, v in maps["intent2id"].items()}
    return cfg, maps, id2slot, id2intent


def predict_text(text, model, tokenizer, segmenter, cfg, maps, id2slot, id2intent, device):
    words = text.split()
    aligned = align_example(words, ["O"] * len(words), segmenter, tokenizer, cfg["max_length"])
    head_mask = [0 if t is None else 1 for t in aligned.slot_tags]
    batch = {
        "input_ids": torch.tensor([aligned.input_ids]),
        "attention_mask": torch.tensor([aligned.attention_mask]),
        "head_mask": torch.tensor([head_mask], dtype=torch.bool),
        "group_head": torch.tensor([aligned.sub_to_head]),
    }
    intent_ids, slot_ids = model.predict(
        batch["input_ids"].to(device), batch["attention_mask"].to(device),
        batch["head_mask"].to(device), batch["group_head"].to(device),
    )
    tags = fold_pred_to_words(aligned.word_head_pos, slot_ids[0].cpu().tolist(), id2slot)
    if cfg.get("bio_repair"):
        tags = repair_bio(tags)
    return id2intent.get(int(intent_ids[0]), "<unk>"), list(zip(words, tags))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="trained run dir (needs model.pt)")
    ap.add_argument("--text", action="append", default=[], help="utterance (repeatable)")
    ap.add_argument("--stdin", action="store_true", help="read one utterance per line")
    ap.add_argument("--data", default="data/raw/tr-TR.jsonl", help="for morphological refit")
    args = ap.parse_args()

    run_dir = Path(args.run)
    cfg, maps, id2slot, id2intent = load_run(run_dir)
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    train_words = None
    if "morpholog" in cfg["segmentation"]:
        train_words = [w for ex in load_examples(args.data, "train") for w in ex.tokens]
    segmenter = build_segmenter(cfg["segmentation"], train_words=train_words)

    model = JointIntentSlot(
        cfg["model_name"], num_intents=len(maps["intent2id"]), num_slots=len(maps["slot2id"]),
        use_crf=cfg.get("use_crf", False), subword_pool=cfg.get("subword_pool", "first"),
    ).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    model.eval()

    texts = list(args.text)
    if args.stdin:
        texts += [ln.strip() for ln in sys.stdin if ln.strip()]
    if not texts:
        texts = ["yarın sabah dokuzda alarm kur"]

    for text in texts:
        intent, pairs = predict_text(
            text, model, tokenizer, segmenter, cfg, maps, id2slot, id2intent, device
        )
        print(f"\n> {text}")
        print(f"  intent: {intent}")
        print("  slots : " + "  ".join(f"{w}/{t}" for w, t in pairs))


if __name__ == "__main__":
    main()
