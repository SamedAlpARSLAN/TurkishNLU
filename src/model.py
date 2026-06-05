"""Joint intent detection + slot filling model (plan §7.4).

A shared encoder feeds two heads: a sentence-level intent classifier off the
[CLS] state and a token-level slot tagger off the sequence states. The joint
loss is ``intent_loss + slot_loss`` (Chen et al., 2019). A linear-chain CRF over
the supervised (head) positions is available as an ablation (plan §8a).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

IGNORE = -100


# ─────────────────────────────────────────────────────────────────────────────
# Minimal linear-chain CRF (self-contained; used only for the CRF ablation)
# ─────────────────────────────────────────────────────────────────────────────
class LinearChainCRF(nn.Module):
    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.start = nn.Parameter(torch.empty(num_tags))
        self.end = nn.Parameter(torch.empty(num_tags))
        self.trans = nn.Parameter(torch.empty(num_tags, num_tags))
        nn.init.uniform_(self.start, -0.1, 0.1)
        nn.init.uniform_(self.end, -0.1, 0.1)
        nn.init.uniform_(self.trans, -0.1, 0.1)

    def _numerator(self, emissions, tags, mask):
        # emissions: (T,B,C), tags: (T,B), mask: (T,B) bool
        T, B = tags.shape
        score = self.start[tags[0]] + emissions[0].gather(1, tags[0].unsqueeze(1)).squeeze(1)
        for t in range(1, T):
            score = score + (
                self.trans[tags[t - 1], tags[t]]
                + emissions[t].gather(1, tags[t].unsqueeze(1)).squeeze(1)
            ) * mask[t]
        last = mask.sum(0).long() - 1
        last_tags = tags.gather(0, last.unsqueeze(0)).squeeze(0)
        return score + self.end[last_tags]

    def _denominator(self, emissions, mask):
        T, B, C = emissions.shape
        alpha = self.start + emissions[0]
        for t in range(1, T):
            broadcast = alpha.unsqueeze(2) + self.trans + emissions[t].unsqueeze(1)
            new = torch.logsumexp(broadcast, dim=1)
            alpha = torch.where(mask[t].unsqueeze(1), new, alpha)
        return torch.logsumexp(alpha + self.end, dim=1)

    def neg_log_likelihood(self, emissions, tags, mask):
        num = self._numerator(emissions, tags, mask)
        den = self._denominator(emissions, mask)
        return (den - num).mean()

    @torch.no_grad()
    def decode(self, emissions, mask):
        T, B, C = emissions.shape
        history = []
        score = self.start + emissions[0]
        for t in range(1, T):
            broadcast = score.unsqueeze(2) + self.trans
            best, idx = broadcast.max(dim=1)
            score_t = best + emissions[t]
            score = torch.where(mask[t].unsqueeze(1), score_t, score)
            history.append(idx)
        score = score + self.end
        best_paths = []
        lengths = mask.sum(0).long()
        for b in range(B):
            length = int(lengths[b])
            best_last = int(score[b].argmax())
            path = [best_last]
            for hist in reversed(history[: length - 1]):
                best_last = int(hist[b][best_last])
                path.append(best_last)
            path.reverse()
            best_paths.append(path)
        return best_paths


# ─────────────────────────────────────────────────────────────────────────────
# Joint model
# ─────────────────────────────────────────────────────────────────────────────
class JointIntentSlot(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_intents: int,
        num_slots: int,
        dropout: float = 0.1,
        use_crf: bool = False,
        slot_loss_weight: float = 1.0,
        slot_loss: str = "ce",  # "ce" | "focal"
        focal_gamma: float = 2.0,
        subword_pool: str = "first",  # "first" | "mean" | "max"
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.intent_head = nn.Linear(hidden, num_intents)
        self.slot_head = nn.Linear(hidden, num_slots)
        self.num_intents = num_intents
        self.num_slots = num_slots
        self.slot_loss_weight = slot_loss_weight
        self.use_crf = use_crf
        self.crf = LinearChainCRF(num_slots) if use_crf else None
        self.ce = nn.CrossEntropyLoss(ignore_index=IGNORE)
        self.slot_loss = slot_loss
        self.focal_gamma = focal_gamma
        self.subword_pool = subword_pool

    def forward(self, input_ids, attention_mask, intent_labels=None, slot_labels=None,
                group_head=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq = out.last_hidden_state  # (B,T,H)
        cls = seq[:, 0]  # [CLS]
        intent_logits = self.intent_head(self.dropout(cls))
        slot_input = seq
        if self.subword_pool != "first" and group_head is not None:
            slot_input = self._pool_subwords(seq, group_head)
        slot_logits = self.slot_head(self.dropout(slot_input))  # (B,T,C)

        result = {"intent_logits": intent_logits, "slot_logits": slot_logits}
        if intent_labels is None or slot_labels is None:
            return result

        intent_loss = self.ce(intent_logits, intent_labels)
        if self.use_crf:
            slot_loss = self._crf_loss(slot_logits, slot_labels)
        else:
            slot_loss = self._token_slot_loss(slot_logits, slot_labels)
        result["loss"] = intent_loss + self.slot_loss_weight * slot_loss
        result["intent_loss"] = intent_loss.detach()
        result["slot_loss"] = slot_loss.detach()
        return result

    def _token_slot_loss(self, slot_logits, slot_labels):
        """Token-level slot loss: standard CE, or focal CE to down-weight the
        dominant 'O' class and rare-slot imbalance (ablation, plan §9)."""
        logits = slot_logits.reshape(-1, self.num_slots)
        labels = slot_labels.reshape(-1)
        if self.slot_loss != "focal":
            return self.ce(logits, labels)
        mask = labels != IGNORE
        if mask.sum() == 0:
            return logits.sum() * 0.0
        logp = F.log_softmax(logits[mask], dim=-1)
        lp = logp.gather(1, labels[mask].unsqueeze(1)).squeeze(1)
        return (-((1.0 - lp.exp()) ** self.focal_gamma) * lp).mean()

    def _pool_subwords(self, seq, group_head):
        """Replace each word's head-position representation with a pool (mean/max)
        of all sub-tokens of that word. An ablation on how a fragmented word is
        aggregated into a single representation for slot prediction (plan §7.1)."""
        pooled = seq.clone()
        for b in range(seq.size(0)):
            g = group_head[b]
            for h in torch.unique(g[g >= 0]).tolist():
                grp = seq[b][g == h]
                pooled[b, h] = grp.mean(0) if self.subword_pool == "mean" else grp.amax(0)
        return pooled

    # CRF operates over the supervised (head) positions only. We pack the
    # per-example heads left-aligned, run the chain there, and ignore the rest.
    def _pack_heads(self, slot_logits, slot_labels):
        B, T, C = slot_logits.shape
        valid = slot_labels != IGNORE  # (B,T)
        lengths = valid.sum(1)
        L = int(lengths.max().clamp(min=1))
        emis = slot_logits.new_zeros(B, L, C)
        tags = slot_labels.new_zeros(B, L)
        mask = torch.zeros(B, L, dtype=torch.bool, device=slot_logits.device)
        index = []
        for b in range(B):
            pos = valid[b].nonzero(as_tuple=False).squeeze(1)
            n = pos.numel()
            if n == 0:
                index.append(pos)
                continue
            emis[b, :n] = slot_logits[b, pos]
            tags[b, :n] = slot_labels[b, pos]
            mask[b, :n] = True
            index.append(pos)
        return emis, tags, mask, index

    def _crf_loss(self, slot_logits, slot_labels):
        emis, tags, mask, _ = self._pack_heads(slot_logits, slot_labels)
        # CRF wants (T,B,C); guarantee a valid first step for empty rows.
        mask[:, 0] = True
        return self.crf.neg_log_likelihood(
            emis.transpose(0, 1), tags.transpose(0, 1), mask.transpose(0, 1)
        )

    @torch.no_grad()
    def predict(self, input_ids, attention_mask, head_mask=None, group_head=None):
        """Return (intent_ids (B,), slot_ids (B,T)). Reads heads via ``head_mask``.

        For softmax we argmax everywhere (eval only reads head positions); for CRF
        we decode the packed head chain and scatter tags back to head positions.
        """
        out = self.forward(input_ids, attention_mask, group_head=group_head)
        intent_pred = out["intent_logits"].argmax(-1)
        slot_logits = out["slot_logits"]
        if not self.use_crf:
            return intent_pred, slot_logits.argmax(-1)

        B, T, C = slot_logits.shape
        valid = head_mask.bool() if head_mask is not None else attention_mask.bool()
        emis, _, mask, index = self._pack_heads(
            slot_logits, valid.long().masked_fill(~valid, IGNORE)
        )
        mask[:, 0] = True
        paths = self.crf.decode(emis.transpose(0, 1), mask.transpose(0, 1))
        slot_ids = torch.zeros(B, T, dtype=torch.long, device=slot_logits.device)
        for b in range(B):
            pos = index[b]
            for k, p in enumerate(pos.tolist()):
                if k < len(paths[b]):
                    slot_ids[b, p] = paths[b][k]
        return intent_pred, slot_ids
