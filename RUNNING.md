# Running the experiments (Kaggle / Colab / local GPU)

The full matrix is small: each cell fine-tunes a base encoder on ~12k MASSIVE-tr
examples in **minutes** on a single T4/P100. The whole 3×3×3 sweep fits
comfortably in one Kaggle session (weekly ~30 GPU-hours free).

> **You have no local GPU** (this repo was bootstrapped on a CPU-only box), so
> the recipe below targets **Kaggle**. Colab is identical apart from the
> file-upload step.

## Kaggle recipe

1. Create a new **Kaggle Notebook** → Settings → **Accelerator: GPU T4 x1**, and
   **Internet: On** (needed to download MASSIVE + HuggingFace models).
2. Get the code into the notebook. Easiest once it's on GitHub:
   ```python
   !git clone https://github.com/<you>/turkish-nlu-segmentation.git
   %cd turkish-nlu-segmentation
   !pip install -q -r requirements.txt
   ```
   (Or zip the project, add it as a Kaggle *Dataset*, and `%cd` into it.)
3. Data + the mandatory alignment guard:
   ```python
   !python scripts/download_data.py
   !python scripts/validate_alignment.py --data data/raw/tr-TR.jsonl \
       --models berturk mbert xlmr
   ```
   All schemes must report **PASS** (100% recovery) before training.
4. Run the matrix (use `nohup`/background for the full sweep):
   ```python
   !python scripts/run_matrix.py            # 27 runs -> outputs/
   ```
   Slice it while iterating:
   ```python
   !python scripts/run_matrix.py --models berturk --segmentations native morphological --seeds 42
   ```
5. Tables + significance + error analysis:
   ```python
   !python scripts/aggregate.py
   !python scripts/error_report.py \
       --predictions outputs/berturk_morphological_seed42/test_predictions.json \
       --data data/raw/tr-TR.jsonl
   # headline significance: morphological vs native, same model/seed
   !python scripts/aggregate.py --compare \
       outputs/xlmr_morphological_seed42/test_predictions.json \
       outputs/xlmr_native_seed42/test_predictions.json
   ```
6. **Save your outputs.** Kaggle wipes `/kaggle/working` between sessions unless
   you *Save Version*. Zip and download:
   ```python
   !zip -r outputs.zip outputs && echo "download outputs.zip from the Output tab"
   ```

## Local GPU

Same commands, no Kaggle steps:
```bash
python -m pip install -r requirements.txt
python scripts/download_data.py
python scripts/validate_alignment.py --data data/raw/tr-TR.jsonl --models berturk mbert xlmr
python scripts/run_matrix.py
python scripts/aggregate.py
```

## Tips

- **Time budget.** Order of magnitude per cell (T4): BERTurk/mBERT a few minutes,
  XLM-R a bit more, ×5 epochs. The 27-cell sweep is roughly 1–3 GPU-hours total.
- **Add `char` / MultiATIS++** only after the core 3×3×3 is in (plan §11, §14).
- **Determinism.** Seeds are fixed (42/123/2024); `set_seed` also pins cuDNN.
  Small run-to-run noise on GPU is expected — that's why we use ≥3 seeds and
  report mean±std (plan §7.5).
- **VRAM.** `batch_size=32`, `max_length=64` fits a T4 easily. If you raise
  `max_length`, watch the truncation count printed by the alignment validator.
