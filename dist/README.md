# FGFR3MUT — Standalone Reproduction Package

This folder contains a self-contained, one-command reproduction of the
FGFR3-mutation experiment from:

> Bannier, P., et al. **"AI allows pre-screening of FGFR3 mutational status
> using routine histology slides of muscle-invasive bladder cancer."**
> *Nature Communications* **15**, 10914 (2024).
> DOI: [10.1038/s41467-024-55331-6](https://doi.org/10.1038/s41467-024-55331-6)

It downloads the cBioPortal metadata, builds ground-truth labels, tiles the
whole-slide images, embeds the tiles with **H-optimus-0**, trains a **Chowder**
multiple-instance-learning model, and runs out-of-fold inference with the same
metrics the paper reports (AUC, AP, sensitivity, specificity, PPV, NPV with
bootstrap confidence intervals) plus an ROC curve.

Everything is orchestrated by a single script:

```bash
./run_all.sh
```

---

## 1. Requirements

| Requirement | Notes |
|---|---|
| Python 3.9–3.12 | `run_all.sh` creates `.venv/` and installs everything |
| Internet | cBioPortal metadata, H-optimus-0 weights (~1.1 GB from Hugging Face), and GDC if you download slides |
| Disk | a few GB per 5 slides (tiles + feature cache); more with `SAVE_PNGS=1` |
| Slides | `.svs` files in `image/` (see below) |
| GPU | optional; CPU works but embedding is much slower |

No `git` is needed: the paper's `fgfr3mut` code (Chowder, bootstrap metrics)
is vendored under `scripts/fgfr3mut_training/fgfr3mut/`.

---

## 2. Quick start

### Step 1 — put slides in `image/`

Copy your TCGA BLCA `.svs` files into `image/`:

```bash
cp /path/to/TCGA-*.svs image/
```

Or download the 5 demo slides this package was developed with
(3 mutant + 2 wild-type, ~5 GB total):

```bash
./scripts/utils/download_svs.sh --demo
```

The demo downloader needs the GDC Data Transfer Tool
(`gdc-client`): put it in `bin/gdc-client`, install it with
`conda install -c bioconda gdc-client`, or pass `-c /path/to/gdc-client`.
Slides are open-access; no GDC token is required.

> **Training needs at least 2 patients per class** (label `1` = FGFR3 mutant,
> `0` = wild-type), because folds are grouped by patient. A single slide is
> enough for inference with existing checkpoints, but not for training.

### Step 2 — run everything

```bash
./run_all.sh
```

First run creates `.venv/`, installs the requirements, and downloads the
H-optimus-0 weights. Later runs reuse the virtualenv, metadata, tiles,
embeddings and checkpoints.

At the end you get:

```
results/labels.csv          slide,label ground truth
results/predictions.csv     per-slide/per-patient predictions
results/metrics.json        fixed-threshold + paper-style bootstrap metrics
results/roc_curve.png       ROC curve (paper Fig. 2 style)
results/tiles_out/          per-slide coords/masks/metadata
results/features_cache/     H-optimus-0 embeddings (N, 1539)
results/models_trained/     Chowder checkpoints + train_manifest.json (one per fold/repeat)
data/                       cBioPortal metadata per slide
```

---

## 3. What `run_all.sh` does

| Step | Action |
|---|---|
| 1 | Create `.venv/` and install `requirements.txt` |
| 2 | Discover `image/*.svs` |
| 3 | Download cBioPortal metadata for every slide (study `blca_tcga_pan_can_atlas_2018`) |
| 4 | Build `results/labels.csv` (activating-only FGFR3 status = paper's positive class) |
| 5 | Tile each WSI: 224 × 224 px at ~1.0 µm/px, ≥60% tissue, max 5000 tiles/slide |
| 6 | Embed tiles with frozen H-optimus-0, then train Chowder with patient-grouped CV (one model per fold) |
| 7 | Score each slide with the checkpoint that held it out (out-of-fold), write predictions, metrics and ROC curve |

Each stage is idempotent: existing metadata, tiles and embeddings are reused.
Delete the corresponding folder, or use `--force-fetch` / `--force-tile`, to
redo a stage. Checkpoints are **not** reused: every training run starts from a
clean `results/models_trained/` so inference can never mix models from
different configurations.

---

## 4. Common options

```bash
# Quick smoke run on CPU (fewer tiles, fewer epochs)
MAX_TILES=500 EPOCHS=5 ./run_all.sh

# Reproduce the paper's CV recipe (25 repeats x 5 folds = 125 models, 30 epochs)
REPEATS=25 MAX_TILES=5000 EPOCHS=30 ./run_all.sh

# Score with the ensemble average (use when applying saved models to new slides)
./run_all.sh --no-oof

# Force a specific device
./run_all.sh --device cpu
./run_all.sh --device cuda:0

# Also save every kept tile as a PNG (large: several GB)
./run_all.sh --save-pngs

# Reuse everything except training (e.g. you already have checkpoints)
SKIP_TRAIN=1 ./run_all.sh

# Skip network/CPU stages when their outputs already exist
SKIP_FETCH=1 SKIP_TILE=1 SKIP_TRAIN=1 ./run_all.sh

# See all options
./run_all.sh --help
```

Options can be passed as flags or environment variables. The most useful are:

| Variable | Default | Meaning |
|---|---|---|
| `MAX_TILES` | `5000` | Tile cap per slide |
| `EPOCHS` | `30` | Chowder training epochs |
| `SPLITS` | `5` | CV folds (auto-reduced to the smallest class count) |
| `REPEATS` | `1` | CV repeats (paper recipe: `25` repeats × 5 folds = 125 models) |
| `SPLIT_UNIT` | `patient` | `patient` (paper, no leakage) or `slide` (old behaviour) |
| `OOF` | auto | `1` = out-of-fold scoring, `0` = ensemble all checkpoints |
| `DEVICE` | `auto` | `cpu`, `cuda:0`, … |
| `SAVE_PNGS` | `0` | Write tile PNGs too |
| `N_BOOTSTRAP` | `1000` | Bootstrap resamples for paper-style CIs |
| `TARGET_SENS` | `0.92` | Operating-point sensitivity |
| `SKIP_INSTALL` | `0` | Use the current Python instead of `.venv/` |

If you use `SKIP_INSTALL=1`, make sure the current Python has the packages in
`requirements.txt`; you can also point at another interpreter with
`PYTHON=/path/to/python`.

---

## 5. Manual / advanced usage

The individual scripts are ordinary command-line programs; `run_all.sh` just
calls them in order. Run them from this directory.

```bash
# 1. metadata for one slide
python scripts/analysis/fetch_cbioportal_data.py \
  --image image/TCGA-FJ-A871-01Z-00-DX5.8F79D0A8-5DE6-4159-AA77-61DACB21E867.svs \
  --outdir data --per-image-dir

# 2. labels (works before tiling; image/ is the slide list)
python scripts/analysis/make_labels_csv.py \
  --tiles_root results/tiles_out --slides_dir image \
  --data_dir data --out_csv results/labels.csv

# 3. tile one slide
python scripts/preprocessing/fgfr3mut_tile_wsi.py \
  --svs_path image/TCGA-FJ-A871-01Z-00-DX5.8F79D0A8-5DE6-4159-AA77-61DACB21E867.svs \
  --out_dir results/tiles_out --tissue_detector otsu --device auto \
  --target_mpp 1.0 --tile_size 224 --min_tissue_fraction 0.60 \
  --max_tiles 5000 --seed 0

# 4. embed + train
python scripts/fgfr3mut_training/train_fgfr3mut.py \
  --tiles_root results/tiles_out --slides_dir image \
  --labels_csv results/labels.csv --epochs 30

# 5. inference + metrics + ROC (--oof = scored slides were also training slides)
python scripts/fgfr3mut_training/infer_fgfr3mut.py \
  --tiles_root results/tiles_out --slides_dir image \
  --labels_csv results/labels.csv --oof
```

Scoring mode matters:

* `--oof` (default in `run_all.sh` when it trains in the same run) scores each
  slide **only** with the fold checkpoint that never saw it. This is the honest
  number and the one to report.
* Without `--oof`, predictions are the mean over all checkpoints. That is the
  right mode for applying saved models to **new** slides, but on training
  slides it includes a model that memorised each slide and inflates AUC.

`infer_fgfr3mut.py` rebuilds the exact training folds from
`results/models_trained/train_manifest.json`, which also records the
`n_extreme` value each checkpoint was trained with. You no longer need to pass
`--n_extreme` by hand: inference reads it from the manifest automatically.

There is also a synthetic smoke test that needs no SVS or GPU:

```bash
python scripts/fgfr3mut_training/train_fgfr3mut.py \
  --smoke_test --out_dir /tmp/fgfr3_smoke --epochs 1 --device cpu
```

---

## 6. Reproducibility notes (please read)

This package reproduces the **method and data flow** of the paper, but two
components cannot be byte-identical to the authors' internal pipeline:

1. **Tissue detection.** The paper uses Owkin's proprietary **BUNet** model,
   which is not public. This package uses the public **Otsu saturation**
   fallback (`--tissue_detector otsu`), as in the repository's documented
   stand-in. It is faster and dependency-free, but keeps a different (smaller)
   set of tiles.
2. **Tiling implementation.** The authors' internal tiling tool is not public.
   `fgfr3mut_tile_wsi.py` reproduces the published geometry: ~1.0 µm/px,
   224 × 224 tiles, ≥60% tissue, up to 5000 tiles per slide.

Everything downstream is the paper's own code or a direct reimplementation:

* **H-optimus-0** frozen tile encoder (`bioptimus/H-optimus-0`, 1536-d).
* **Chowder** MIL model, from the authors' `fgfr3mut` repository (vendored).
* **Patient-grouped cross-validation**: all slides of one `TCGA-XX-XXXX`
  patient stay in the same fold (the paper's unit), so a patient's genotype
  cannot leak across train/validation.
* **Metrics** use the authors' own `fgfr3mut/bootstrap.py`, including the
  sensitivity-targeted operating point and 1000-resample bootstrap CIs.

The labels use the paper's **activating-only** positive class (the 11
activating FGFR3 mutations), via `reference/per_slide_table.csv`. Slides not
present in that table are labelled from the freshly downloaded cBioPortal
records (any FGFR3 mutation). Every label's source is printed at step 4.

Consequently, expect **qualitatively similar but not numerically identical**
results to the paper, especially with a handful of slides. The paper's final
models were trained on hundreds of cases; with the 5 demo slides,
cross-validation drops to 2 folds (4 patients: 2 MUT / 2 WT) and confidence
intervals are very wide.

---

## 7. Troubleshooting

**`openslide` import error / `libopenslide` not found**

```bash
# Debian/Ubuntu
sudo apt install libopenslide0
# or let pip ship the library:
pip install openslide-bin
```

**`python3 -m venv` fails**

```bash
sudo apt install python3-venv       # Debian/Ubuntu
```

**cBioPortal request errors**

The script aborts rather than silently writing partial labels. Retry, or use
`SKIP_FETCH=1` if `data/` is already populated. You can test connectivity with:

```bash
curl -s https://www.cbioportal.org/api/studies/blca_tcga_pan_can_atlas_2018 | head
```

**H-optimus-0 download fails**

The weights are fetched from Hugging Face on first use and cached in
`~/.cache/huggingface/`. Check your connection/proxy. No login is required.

**Out of memory while tiling**

Lower the number of tile workers (`--workers` on `fgfr3mut_tile_wsi.py`) or
set `--preload off` (the default). `run_all.sh` does not preload the pyramid.

**No checkpoints / `size mismatch` at inference**

Checkpoints must be trained with the same `n_extreme` used at inference.
`train_fgfr3mut.py` records the real value per checkpoint in
`results/models_trained/train_manifest.json`, and `infer_fgfr3mut.py` reads it
back automatically, so simply re-run `./run_all.sh` if checkpoints and manifest
ever get out of sync (it retrains from scratch).

**`--oof` fails because a scored slide was not in training**

Out-of-fold scoring needs a checkpoint that held each slide out. If you score
new slides, use `--no-oof` (ensemble mode) or drop `--oof`.

**Fewer than 2 patients in a class**

Folds are grouped by patient (`TCGA-XX-XXXX`), because all slides of one patient
share the same genotype. Two slides from one patient therefore cannot form two
folds. Add slides from more patients, or run with `SKIP_TRAIN=1` if you only
want to apply existing checkpoints.

---

## 8. Directory layout

```
.
├── run_all.sh                    # the single entry point
├── requirements.txt
├── README.md
├── image/                        # PUT YOUR .svs FILES HERE
├── data/                         # cBioPortal metadata (generated)
├── results/                      # labels, tiles, features, models, metrics
├── reference/
│   └── per_slide_table.csv       # activating-only FGFR3 status (paper labels)
├── scripts/
│   ├── analysis/
│   │   ├── fetch_cbioportal_data.py
│   │   └── make_labels_csv.py
│   ├── preprocessing/
│   │   ├── fgfr3mut_tile_wsi.py
│   │   └── background_remover.py
│   ├── fgfr3mut_training/
│   │   ├── train_fgfr3mut.py
│   │   ├── infer_fgfr3mut.py
│   │   └── fgfr3mut/             # vendored paper code (MIT license)
│   └── utils/
│       ├── download_svs.sh
│       ├── demo_slides.txt
│       └── tcga_blca_slides.tsv
└── bin/                          # optional: put gdc-client here
```

---

## 9. Citation

```bibtex
@article{bannier2024fgfr3,
  title   = {AI allows pre-screening of FGFR3 mutational status using routine
             histology slides of muscle-invasive bladder cancer},
  author  = {Bannier, Pierre-Antoine and others},
  journal = {Nature Communications},
  volume  = {15},
  pages   = {10914},
  year    = {2024},
  doi     = {10.1038/s41467-024-55331-6}
}
```

The vendored `fgfr3mut` code is Copyright (c) Owkin, Inc. and affiliates,
distributed under the MIT license; see
`scripts/fgfr3mut_training/fgfr3mut/LICENSE`.
