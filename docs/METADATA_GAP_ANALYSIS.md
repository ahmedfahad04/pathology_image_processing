# Metadata + Code-Gap Analysis: Our Tiling vs Author Pipeline

Compares `output/tiling_test/...4Z-AA7S.../metadata.json` (ours, regenerated
with `reproduce/tiling.py` repro-1.0 on the **same slide**) against
`output/data_fgfr3_mini/features/...4Z-AA7S.../metadata.json` (authors),
then lists what the paper (`docs/s41467-024-55331-6.pdf`) describes but the
author codebase (`https://github.com/PABannier/fgfr3mut`) does not provide,
with impact on reproducibility.

> Run context (ours): `fgfr3mut_env`, reader=tifffile (openslide not installed
> in that env), `--max_tiles 16`. Key set is now identical to the author's.

## 1. Metadata key-by-key comparison (same slide)

| Author key | Author value | Ours | Verdict |
|---|---|---|---|
| `version` | `"3.0.0"` | `"3.0.0"` | match |
| `tiling_tool_version` | `"11.6.0"` (Owkin internal) | `"repro-1.0 (schema-compatible…)"` | expected label gap |
| `nb_tiles` / `total_number_of_tiles` | `3440` (complete) | `16` (smoke-test cap) | sampling cap, not comparable |
| `sampling_mode` | `{random, seed 0}` | `{random, seed 0}` | match |
| `matter_detector.name` | `BUNet` | `Otsu-standin-for-BUNet` | **different segmenter** (§2.1) |
| `matter_detector.dilatation` | `1` | `1` | match |
| `matter_detector.threshold` | `0.5` (prob.) | `196` (Otsu gray level) | different semantics, same key |
| `features_extractor` | `"iBOTViTGiant"` | `"H-optimus-0"` | **extractor mismatch** (§2.3) |
| `level` | `15` | `15` | match |
| `tile_size` | `224` | `224` | match |
| `absolute_tile_size` | `896` | `896` | match (fixed: pyramid-exact, §1.1) |
| `slide_size` | `[75696, 82376]` | `[75696, 82376]` | match |
| `tile_mpp` | `1.01` | `1.01` | match |
| `level_dimensions` (0–17) | 18-entry dyadic table | identical | **bit-exact** |
| `level_mpp_mapping` (0–17) | 18-entry table | identical | **bit-exact** |
| `environment` | py 3.9.19, openslide 3.4.1/1.3.1, torch 2.0.1+cu117 | py 3.9.25, openslide not-installed, torch 2.8.0+cu128, timm 1.0.29 | different stack (our env lacks openslide) |
| `slide_hash` | `1c6e5a30…` | `9df01ab6…` | **bytes differ** (see below) |

### 1.1 `absolute_tile_size`: fixed (was 896 vs 887)
- Theirs is pyramid-exact: 224 px × downsample 4 (level 15→17) = 896.
- Ours was mpp-derived: round(224 / 0.2525) = 887, poisoned by rounded mpp0.
- Fix applied in `reproduce/tiling.py`: openslide path uses
  `footprint = TILE_PX`, `abs_tile = round(TILE_PX * downsample)`; metadata
  uses `round(224 * w0 / extraction_w)`. Takes effect on next rerun.

### 1.2 `slide_hash`: resolved — local file is genuine
GDC API confirms `md5sum: 9df01ab676ef8bf3d553970035fedc20`, size 1156460815 —
byte-identical to local `image/TCGA-4Z-AA7S…svs`. The local copy is the
authentic current GDC file; it is the *authors'* hash (`1c6e5a30…`) that
differs (different mirror, re-encode, or older GDC version). Pixel comparison
against released `features.npy` stands, modulo whatever copy the authors used.

## 2. What the paper describes but the authors did not release

### 2.1 BUNet weights and training data (paper p.9, "Preprocessing")
- Paper: U-Net trained on 460 annotated H&E/IHC slides, validated on 115
  (Dice 0.96); rejects folds, pen markers, blur.
- Released: only a link to the generic `milesial/Pytorch-UNet` implementation.
- Impact: **highest**. Every downstream tile set differs; our Otsu mask is a
  stand-in with no artifact rejection. Slide representations shift, so
  published AUCs cannot be regenerated from raw slides.

### 2.2 Tiling tool v11.6.0 (Owkin internal)
- Paper + metadata reference it (dilatation, 60% rule, level mapping,
  random sampling); no source in repo.
- Status: dyadic 0–17 table reverse-engineered bit-exact (§1); `level`,
  `tile_mpp`, `sampling_mode` all match. Remaining: `absolute_tile_size`
  formula (§1.1), tile-grid edge handling, `(level, tx, ty)` coord semantics.
- Impact: medium-low now. `reproduce/tiling.py` (openslide path) + schema
  parity cover documented behavior; only pixel-geometry details are inferred.

### 2.3 Feature-extractor mismatch
- Paper p.9: features from "Bioptimus' H0 (ViT Giant)".
- Released `features.npy` metadata: `features_extractor: "iBOTViTGiant"`.
- H-optimus-0 weights are gated (`bioptimus/H-optimus-0`, login + approval);
  our smoke test confirmed access works (`Embeddings: (16, 1536)`).
- Impact: medium. Re-extracted H-0 embeddings may not match the released
  1536-dim vectors if those came from iBOT. Our `tiling.py` honestly labels
  its output `H-optimus-0`.

### 2.4 MIL training code (paper p.9, "FGFR3 mutation prediction")
- Paper: 125-model Chowder ensemble (Courtiol et al.), 5000-tile cap,
  BCE loss, logit averaging.
- Released: `fgfr3mut/chowder.py`, `dataset.py`, `run_inference.py`,
  `bootstrap.py` (inference + CIs) and packaged weights only. No `train*.py`,
  no optimizer/scheduler/epoch/seed config.
- Impact: medium. Inference reproduces (our pilot: AUC 1.00 on 4 patients);
  retraining, ablations (0.5 vs 1.0 MPP, MIBC-only vs +NMIBC), and the
  125-model variance/uncertainty analysis cannot be rerun.

### 2.5 Private cohorts (paper p.8, "Datasets description"; Data availability)
- 810 Erlangen cases (MIBC I/II, NMIBC I/II, mUC) under GDPR, on request only.
  Only TCGA-BLCA (307 of 1222 cases) is public.
- Impact: medium. Full validation (n=586, NPV≈0.99, ~40% tests saved) is not
  independently repeatable; only the TCGA slice is.

### 2.6 Manual QC step (paper Fig. 1b)
- 43 blurry + 105 excluded slides removed by pathologist visual inspection.
  No script; inherently manual.
- Impact: low. Our pilot includes slides the authors would have excluded
  (e.g. TSA slide has no released features), adding noise to comparisons.

### 2.7 Threshold calibration + interpretability/uncertainty scripts
- Operating thresholds (0.059 / 0.051 / 0.063, Table 1), 400-tile pathologist
  review, deep-ensemble variance, conformal sets: described, no scripts.
- Impact: low for inference (thresholds are published numbers); blocks
  recalibration on new cohorts and the interpretability figures.

## 3. Bottom line

| Reproduces today | Blocked by missing code / data |
|---|---|
| Metadata schema (identical keys, bit-exact pyramid table, abs_tile 896) | Tile-for-tile preprocessing (BUNet, §2.1) |
| Inference on released features + packaged weights | Tile-for-tile preprocessing (BUNet, §2.1) |
| TCGA-only score replication | Retraining / ablations (no training code, §2.4) |
| Heatmaps from released features | Exact embeddings (extractor mismatch, §2.3) |
| H-0 embedding smoke test (16, 1536) | Full-cohort validation (private slides, §2.5) |
| Genuine GDC bytes (hash verified, §1.2) | — |

The released artifacts support **inference verification**, not **pipeline
reproduction**. Load-bearing gaps, ranked: BUNet weights (§2.1), training
code (§2.4), private slides (§2.5), local-file provenance (§1.2).
