"""Torch dataset + collator: pre-align every example under a segmentation scheme.

Each example is aligned once (alignment.align_example). We keep, alongside the
model inputs, the per-word head positions and the gold word-level tags so that
evaluation can fold subword predictions back to the common word reference unit.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from .alignment import IGNORE, align_example
from .data import Example, LabelMaps
from .segmentation import Segmenter
from .utils import get_logger

logger = get_logger()


@dataclass
class Feature:
    input_ids: list[int]
    attention_mask: list[int]
    head_mask: list[int]  # 1 at supervised (pre-token head) positions
    slot_label_ids: list[int]  # IGNORE off heads
    sub_to_head: list[int]  # per subword -> its word's head position (for pooling)
    intent_id: int
    word_head_pos: list[int]
    gold_word_tags: list[str]
    tokens: list[str]
    uid: str


class JointDataset(Dataset):
    def __init__(
        self,
        examples: list[Example],
        segmenter: Segmenter,
        tokenizer,
        label_maps: LabelMaps,
        max_length: int = 128,
    ):
        self.features: list[Feature] = []
        self.label_maps = label_maps
        n_unk_intent = 0
        for ex in examples:
            aligned = align_example(
                ex.tokens, ex.slots, segmenter, tokenizer, max_length
            )
            slot_ids = [
                IGNORE if t is None else label_maps.slot2id.get(t, label_maps.slot2id["O"])
                for t in aligned.slot_tags
            ]
            head_mask = [0 if t is None else 1 for t in aligned.slot_tags]
            intent_id = label_maps.intent2id.get(ex.intent, -1)
            if intent_id < 0:
                n_unk_intent += 1
                continue
            self.features.append(
                Feature(
                    input_ids=aligned.input_ids,
                    attention_mask=aligned.attention_mask,
                    head_mask=head_mask,
                    slot_label_ids=slot_ids,
                    sub_to_head=aligned.sub_to_head,
                    intent_id=intent_id,
                    word_head_pos=aligned.word_head_pos,
                    gold_word_tags=ex.slots,
                    tokens=ex.tokens,
                    uid=ex.uid,
                )
            )
        if n_unk_intent:
            logger.warning("Dropped %d examples with unseen intent label", n_unk_intent)
        logger.info(
            "Built %d features for scheme '%s'", len(self.features), segmenter.name
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Feature:
        return self.features[idx]


def make_collate(pad_token_id: int):
    """Return a collate_fn that pads model inputs and carries eval metadata."""

    def collate(batch: list[Feature]) -> dict:
        max_len = max(len(f.input_ids) for f in batch)

        def pad(seq, value):
            return seq + [value] * (max_len - len(seq))

        input_ids = torch.tensor([pad(f.input_ids, pad_token_id) for f in batch])
        attention = torch.tensor([pad(f.attention_mask, 0) for f in batch])
        head_mask = torch.tensor([pad(f.head_mask, 0) for f in batch], dtype=torch.bool)
        slot_labels = torch.tensor([pad(f.slot_label_ids, IGNORE) for f in batch])
        group_head = torch.tensor([pad(f.sub_to_head, -1) for f in batch])
        intent_labels = torch.tensor([f.intent_id for f in batch])
        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "head_mask": head_mask,
            "slot_labels": slot_labels,
            "group_head": group_head,
            "intent_labels": intent_labels,
            # eval metadata (python lists; not moved to device)
            "word_head_pos": [f.word_head_pos for f in batch],
            "gold_word_tags": [f.gold_word_tags for f in batch],
            "uids": [f.uid for f in batch],
            "tokens": [f.tokens for f in batch],
        }

    return collate
