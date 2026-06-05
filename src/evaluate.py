"""Evaluation: intent accuracy, span-level slot F1, and frame accuracy (§7.5).

All metrics are computed at the *word* level: subword predictions are folded back
to surface words via the per-word head positions, so schemes are comparable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from seqeval.scheme import IOB2

from .data import LabelMaps


def fold_pred_to_words(word_head_pos, sub_pred_ids, id2slot) -> list[str]:
    out = []
    for p in word_head_pos:
        if 0 <= p < len(sub_pred_ids):
            out.append(id2slot.get(int(sub_pred_ids[p]), "O"))
        else:
            out.append("O")  # truncated word
    return out


@dataclass
class EvalRecord:
    uid: str
    tokens: list[str]
    intent_gold: str
    intent_pred: str
    slots_gold: list[str]
    slots_pred: list[str]


@dataclass
class EvalResult:
    intent_acc: float
    slot_precision: float
    slot_recall: float
    slot_f1: float
    frame_acc: float
    n: int
    records: list[EvalRecord] = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "intent_acc": round(self.intent_acc, 4),
            "slot_f1": round(self.slot_f1, 4),
            "slot_precision": round(self.slot_precision, 4),
            "slot_recall": round(self.slot_recall, 4),
            "frame_acc": round(self.frame_acc, 4),
            "n": self.n,
        }


@torch.no_grad()
def evaluate(model, dataloader, label_maps: LabelMaps, device: str) -> EvalResult:
    model.eval()
    id2slot = label_maps.id2slot
    id2intent = label_maps.id2intent

    gold_seqs: list[list[str]] = []
    pred_seqs: list[list[str]] = []
    records: list[EvalRecord] = []
    intent_correct = 0
    frame_correct = 0
    total = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention = batch["attention_mask"].to(device)
        head_mask = batch["head_mask"].to(device)
        intent_pred, slot_ids = model.predict(input_ids, attention, head_mask)
        intent_pred = intent_pred.cpu().tolist()
        slot_ids = slot_ids.cpu().tolist()

        for b in range(len(batch["uids"])):
            gold_tags = batch["gold_word_tags"][b]
            pred_tags = fold_pred_to_words(
                batch["word_head_pos"][b], slot_ids[b], id2slot
            )
            gold_seqs.append(gold_tags)
            pred_seqs.append(pred_tags)

            g_intent = id2intent[int(batch["intent_labels"][b])]
            p_intent = id2intent.get(int(intent_pred[b]), "<unk>")
            i_ok = g_intent == p_intent
            s_ok = pred_tags == gold_tags
            intent_correct += int(i_ok)
            frame_correct += int(i_ok and s_ok)
            total += 1
            records.append(
                EvalRecord(
                    uid=batch["uids"][b],
                    tokens=batch["tokens"][b],
                    intent_gold=g_intent,
                    intent_pred=p_intent,
                    slots_gold=gold_tags,
                    slots_pred=pred_tags,
                )
            )

    kw = dict(mode="strict", scheme=IOB2, zero_division=0)
    return EvalResult(
        intent_acc=intent_correct / max(total, 1),
        slot_precision=float(precision_score(gold_seqs, pred_seqs, **kw)),
        slot_recall=float(recall_score(gold_seqs, pred_seqs, **kw)),
        slot_f1=float(f1_score(gold_seqs, pred_seqs, **kw)),
        frame_acc=frame_correct / max(total, 1),
        n=total,
        records=records,
    )


def slot_classification_report(result: EvalResult) -> str:
    gold = [r.slots_gold for r in result.records]
    pred = [r.slots_pred for r in result.records]
    return classification_report(gold, pred, mode="strict", scheme=IOB2, digits=4)
