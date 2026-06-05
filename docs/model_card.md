# Model card — Joint Intent + Slot model (Turkish)

## Model details
- **Architecture:** a pretrained transformer encoder (BERTurk, mBERT, or XLM-R)
  with two heads — a sentence-level intent classifier off `[CLS]` and a
  token-level BIO slot tagger off the sequence states; joint loss
  `intent_loss + λ·slot_loss` (Joint BERT, Chen et al. 2019).
- **Ablations:** linear-chain CRF, focal slot loss, subword pooling
  (first/mean/max), constrained BIO decoding.
- **Independent variable:** segmentation strategy applied before subwording
  (native / whitespace / morphological / character).
- **Inputs:** Turkish utterances. **Outputs:** one intent label + a BIO slot tag
  per surface word (predictions are folded from sub-tokens to words).

## Intended use
- **Primary:** research on how segmentation affects Turkish task-oriented NLU.
- **Also:** a baseline joint intent+slot model for Turkish (via `predict.py`).
- **Out of scope:** production dialogue systems without further validation;
  languages other than Turkish; safety-critical decisions.

## Training data
- **MASSIVE-tr** (Amazon MASSIVE, CC BY 4.0): 11,514 / 2,033 / 2,974
  train/dev/test utterances, 60 intents, 55 slot types, 18 domains. Official
  splits; no additional data collected.

## Evaluation
- Intent accuracy, span-level slot F1 (seqeval, strict IOB2), and frame accuracy
  (intent + all slots correct), at the surface-word level. Mean±std over 3 seeds;
  paired bootstrap and exact McNemar for significance.

## Limitations & ethical considerations
- The morphological strategy uses an **unsupervised Morfessor proxy** (most-
  probable analysis); it diverges from rule-based analysis (Zeyrek) and is an
  explicit error source. See `results/morphology_compare.json`.
- Single benchmark/domain; results may not transfer to spontaneous speech or
  other domains. MultiATIS++-tr is supported as an optional second domain.
- MASSIVE is a virtual-assistant dataset; intents/slots reflect its design and
  may encode its biases. The models are not evaluated for fairness across user
  groups and should not be deployed without such evaluation.
- No personal data beyond the public MASSIVE corpus is used.

## Reproducibility
Public data, fixed seeds (42/123/2024), deterministic alignment with a 100%
round-trip guard, pinned versions (`requirements-lock.txt`), and full configs.
Repository: <ANONYMIZED-REPO-URL> (de-anonymised in camera-ready).
