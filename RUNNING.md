# Running the experiments (Kaggle / Colab / local GPU)

The full matrix is small: each cell fine-tunes a base encoder on ~12k MASSIVE-tr
examples in **minutes** on a single T4/P100. The whole 3×3×3 sweep fits
comfortably in one Kaggle session (weekly ~30 GPU-hours free).

> **You have no local GPU** (this repo was bootstrapped on a CPU-only box), so
> the recipe below targets **Kaggle**. Colab is identical apart from the
> file-upload step.

## Kaggle recipe

### Step 0 — prerequisites (do these first, or the clone/pip will fail)

- **Verify your Kaggle account by phone.** Profile (top-right) → *Settings* →
  *Phone Verification*. **Internet access is locked until you verify.** This is
  the #1 cause of `Could not resolve host: github.com`.
- **Make the GitHub repo Public** (so Kaggle can clone it without a token):
  on GitHub → the repo → *Settings* → *General* → *Danger Zone* → *Change
  visibility* → **Public**.

### Step 1 — notebook settings (right-hand panel)

Open the notebook, click the **`⋮` / Settings** panel on the right and set:
- **Accelerator → GPU T4 x2** (or P100).
- **Internet → On**  ← this fixes "Could not resolve host".

If the **Internet** toggle is greyed out, you skipped the phone verification above.

### Step 2 — get the code (each line on its OWN line; do NOT use `&&`)

```python
!git clone https://github.com/SamedAlpARSLAN/TurkishNLU.git
```
```python
%cd TurkishNLU
```
```python
!pip install -q -r requirements.txt
```

> ⚠️ `%cd` is a Jupyter magic and does **not** accept `&&`. Writing
> `%cd TurkishNLU && pip install ...` fails with
> `No such file or directory: 'TurkishNLU && pip install ...'`. Keep `%cd` on its
> own line. (No GPU is needed yet for steps 1–3; turn it on before step 4.)
>
> Offline alternative (no Internet): download the repo ZIP from GitHub, add it as
> a Kaggle *Dataset*, then `%cd /kaggle/input/<dataset-name>`. You still need
> Internet for `pip install` (morfessor/seqeval/zeyrek aren't pre-installed).
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

## Troubleshooting

| Error you see | Cause | Fix |
|---|---|---|
| `Could not resolve host: github.com` | Internet is **off** | Settings → Internet → On (verify phone first) |
| `Internet` toggle greyed out | Account not phone-verified | Profile → Settings → Phone Verification |
| `No such file or directory: 'TurkishNLU && pip install ...'` | `%cd` used with `&&` | Put `%cd TurkishNLU` on its own line |
| `fatal: repository not found` / auth prompt | Repo is **private** | Make it Public, or clone with a token |
| `ModuleNotFoundError: morfessor/seqeval/zeyrek` | `pip install` skipped/failed | Re-run the `!pip install -q -r requirements.txt` cell (Internet on) |
| `CUDA ... not available` in training | GPU accelerator off | Settings → Accelerator → GPU T4 |

## Tips

- **Time budget.** Order of magnitude per cell (T4): BERTurk/mBERT a few minutes,
  XLM-R a bit more, ×5 epochs. The 27-cell sweep is roughly 1–3 GPU-hours total.
- **Add `char` / MultiATIS++** only after the core 3×3×3 is in (plan §11, §14).
- **Determinism.** Seeds are fixed (42/123/2024); `set_seed` also pins cuDNN.
  Small run-to-run noise on GPU is expected — that's why we use ≥3 seeds and
  report mean±std (plan §7.5).
- **VRAM.** `batch_size=32`, `max_length=64` fits a T4 easily. If you raise
  `max_length`, watch the truncation count printed by the alignment validator.
