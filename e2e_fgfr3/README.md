# FGFR3 mutation with E2E-ABMILX

This runs the FGFR3 mutation experiment with the end-to-end **ResNet-18 + ABMILX**
pipeline (Tang et al., NeurIPS 2025) instead of H-optimus-0 + Chowder. It uses
the **same data and the same split** as the Chowder run:

| | Chowder run | This run |
|---|---|---|
| Slides, labels, patients | `results/models_trained_125/train_manifest.json` | same manifest |
| Tiles | `results/tiles_out/<slide>/coords.npy` (224 px, ~1.0 µm/px, Otsu, ≤5000/slide, seed 0) | the same coordinates |
| CV folds | `make_folds(..., repeat=0)`, patient-grouped, 5 folds | built by that same function (`split_0`) |
| Test set of fold *k* | Chowder's held-out fold *k* | the same slides |

The E2E code is vendored in `third_party/E2E-WSI-ABMILX/` and is **not modified**.
It was copied from `github.com/Sabbir-Asif/wsi` at `6459565`, including the local
`CLAM/save_patches.py` fix (an unused `get_encoder` import removed).
Everything specific to this experiment lives in `e2e_fgfr3/`.

## Run it (GPU PC)

Prerequisites: the FGFR3 run's `image/*.svs`, `results/tiles_out/` and
`results/models_trained_125/train_manifest.json` at the repo root; Python
3.10–3.12; `libturbojpeg` (`sudo apt install libturbojpeg`).

```bash
git fetch && git checkout e2e-fgfr3
# CUDA build of torch (pick the index that matches your driver):
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 ./e2e_fgfr3/run_e2e_fgfr3.sh
```

That is the whole pipeline. Re-running skips finished steps (LMDBs, folds).

Useful variations:

```bash
# out of GPU memory: fewer patches per slide per step / smaller encoder chunks
./e2e_fgfr3/run_e2e_fgfr3.sh -- --same_psize=256 --p_batch_size=256

# training crashed during fold 3: resume there (earlier folds' predictions are kept)
SKIP_INSTALL=1 FOLD_START=3 ./e2e_fgfr3/run_e2e_fgfr3.sh

# prepare data only
SKIP_TRAIN=1 ./e2e_fgfr3/run_e2e_fgfr3.sh

# all options
./e2e_fgfr3/run_e2e_fgfr3.sh --help
```

If the FGFR3 models were trained in a virtualenv other than `.venv/` or
`dist/.venv/`, pass it as `FGFR3_PYTHON=/path/to/python`. The folds are built
with that interpreter because `StratifiedGroupKFold` shuffling can change between
scikit-learn versions. The version used is recorded in `folds_summary.json`.

## What the script does

1. **Environment**: creates `.venv-e2e/` (separate from the FGFR3 env) with
   `torch>=2.5.1`, `requirements-e2e.txt`, then `torchaug==0.6.1 --no-deps`.
2. **Inputs**: checks that every manifest slide has its `.svs` and its FGFR3 tiles.
   Slides without tiles are tiled with the FGFR3 tiler and its original settings.
3. **`tools/coords_to_clam_h5.py`**: writes CLAM `patches/<slide>.h5` files that
   E2E's `save_patches.py` reads:
   - `main/`: the FGFR3 tiles, read at the same level and size as Chowder's
     embedding step.
   - `sub/`: the 2×2 blocks containing those tiles (~2 µm/px). These are used only
     for E2E's multi-scale training input, never at test time.
4. **LMDBs**: E2E's own `CLAM/save_patches.py` writes `main_224` (val/test),
   `main_256` (train) and `sub_256` (train) into `results/e2e_fgfr3/lmdb/`.
5. **`tools/make_e2e_folds.py`**: `folds_rep0/fold_{k}.csv`.
   - test = FGFR3 held-out fold k.
   - val = 12.5% of the remaining patients (grouped by patient, stratified by label).
   - train = the rest.
6. **Training**: `tools/train_with_oof_export.py` runs E2E's `main.py` with
   `--datasets=bio_fgfr3 --csv_path=folds_rep0/ --cv_fold=5`. It also records
   each fold's final test predictions.

## Outputs (`results/e2e_fgfr3/`)

```
runs/fgfr3/r18_abmilx_fgfr3_rep0/
  fold_{k}_model_best.pt       best-on-val main model (+ its EMA weights)
  fold_{k}_ema_model_best.pt   best-on-val EMA model
  oof_predictions.csv          fold,slide,patient,label,prob_main,prob_ema
  train_<timestamp>.log        full log; per-fold test metrics at the end
folds_rep0/                    fold_{k}.csv + folds_summary.json
lmdb/                          main_224.lmdb, main_256.lmdb, sub_256.lmdb
clam_h5/                       main/ and sub/ patch coordinate files
```

`oof_predictions.csv` holds one row per slide, scored by the fold that held it
out (out-of-fold). `prob_*` = P(FGFR3 mutant), slide level. The Chowder metrics
average the slides of each patient first (`groupby(patient).mean()`).

## Design notes

These differ between the two methods by design. They come from the E2E recipe
and were not changed:

- **Validation set.** Chowder trains a fixed 30 epochs with no validation set.
  E2E picks the best epoch on validation AUC and early-stops after 10 epochs
  without improvement. The validation set is therefore carved out of the
  training part, and the test folds are untouched. With ~31 val slides (4–5
  mutant) per fold, epoch selection is noisy.
- **Loss.** Chowder weights the mutant class (`pos_weight`). E2E uses
  unweighted cross-entropy.
- **Repeats.** Only repeat 0 (5 models). Chowder has 25 repeats × 5 folds. The
  equivalent Chowder number is the `split_0` folds alone. Other repeats can be run
  with `REPEAT=<r>`.
- **Config.** `config/r18_fgfr3.yaml` is the authors'
  `r18_brca_subtyping.yaml` (binary TCGA task) with `cv_fold: 5` and without
  `val2test`. `val2test` would merge the val slides into the test fold.
  `val_ratio` must stay non-zero, otherwise E2E validates on the test fold.
- **Never pass `--wandb`.** E2E then runs `git add`/`git commit -a` in the
  repository it lives in, i.e. this one.

## Verified

- `make_e2e_folds.py` on the committed 310-slide manifest: 5 folds of 62 test
  slides (8–9 mutant each). Every slide is tested exactly once and no patient
  appears in two splits.
- Full script on macOS CPU with two E2E demo slides linked under eight names
  (4 WT / 4 MUT, 2 folds, 2 epochs, `--same_psize=16`): install, tile
  conversion, all three LMDBs, folds, training of both folds, checkpoints and
  `oof_predictions.csv`. Exit code 0.
- On macOS the FGFR3 tiler hangs with more than one worker (`--workers 1` works).
  This does not affect Linux or already-tiled slides.
- Not run here: GPU training on the real 310 slides.
