### Tokenizer morphological alignment (reference: morphological, slot words, type-level)

| Encoder | Fertility | Ref.cov | Boundary P | Boundary R | Boundary F1 | Morpheme respect |
|---|---|---|---|---|---|---|
| berturk | 2.053 | 0.518 | 0.46 | 0.541 | 0.497 | 0.417 |
| mbert | 2.671 | 0.518 | 0.306 | 0.541 | 0.391 | 0.163 |
| xlmr | 2.21 | 0.518 | 0.443 | 0.6 | 0.51 | 0.354 |

### native vs whitespace divergence (cross-whitespace merging)

| Encoder | Divergent sentences | Avg tokens saved |
|---|---|---|
| berturk | 0.0% | 0.0 |
| mbert | 0.0% | 0.0 |
| xlmr | 0.0% | 0.0 |

Finding: divergence is ~0% for ALL three encoders, including SentencePiece XLM-R — its `▁` space marker preserves word boundaries on whitespace-separated input, so native and whitespace coincide. The meaningful contrast is model-default subword (native≈whitespace) vs morphological vs character.
