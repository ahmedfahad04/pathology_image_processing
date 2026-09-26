# `fgfr3mut_tile_wsi.py` — FGFR3MUT-style WSI Tiling (Explained)

**Script:** `scripts/preprocessing/fgfr3mut_tile_wsi.py`
**Spec source:** `docs/fgfr3mut_pipeline/training.md` Steps 1–2 ("Matter detection" + "Tiling"), which is itself verified against the paper Methods *and* the real downloaded feature metadata (`output/data_fgfr3_mini/*/metadata.json`).

---

## Does it remove the background?

**Yes — background removal is the first thing the script does, before any tiling.**

Per `training.md` Step 1–2, the FGFR3MUT preprocessing pipeline is three stages:

1. **Matter (tissue) detection** — a U-Net-family model segments tissue vs. background/artifacts (folded tissue, pen marks, blur), thresholded at 0.5 (`metadata.json` → `matter_detector: {name: "BUNet", threshold: 0.5}`, Dice 0.96 on the paper's held-out set).
2. **Tiling** — cut 224×224 px tiles at ~1.0 MPP; **keep a tile only if ≥60% of its pixels are tissue** per the matter-detector mask.
3. **Feature extraction** — frozen ViT-Giant encoder → 1536-dim embedding per kept tile (this script does *not* do this step).

This script implements steps 1–2. It builds a tissue mask first, then walks the tile grid and only emits cells whose tissue fraction ≥ `min_tissue_fraction` (default 0.60). Background-only regions never appear in the output `coords.npy`, so **all downstream steps only ever see tissue tiles** — no separate background-removal step is needed later.

Two caveats vs. the real pipeline (both documented in the script's docstring and `training.md`):

- The paper's exact detector (**BUNet**) is Owkin-internal (`classic_algos` package) and was never released. This script offers two public stand-ins via `--tissue_detector`:
  - **`grandqc` (default)** — GrandQC (Weng et al., *Nat Commun* 2024): UNet++ + timm-efficientnet-b0, 2-class tissue/background, Dice 0.957. Closest public match to BUNet.
  - **`otsu`** — Otsu threshold on HSV saturation. Zero dependencies but crude: ~9× lower tissue recall than BUNet on a test slide (471 vs 4125 kept tiles). Fallback only.
- Unlike BUNet's continuous `mask.npy`, the Otsu path is a hard threshold; GrandQC's output is argmax over 2-class logits. Either way the contract is the sam?.

---

## What MPP means — and what happens if you change it

**MPP (microns-per-pixel)** is the *physical* size of one pixel. It fixes the scale bar independently of the display: at 1.0 MPP, one pixel = 1 µm, so a 224×224 tile covers exactly **224×224 µm of tissue** (the geometry the paper specifies). Scanners record at ~0.25 MPP (level 0 of the pyramid; e.g. `native_mpp=0.2277` for the verified TCGA slide).

The script does not resample arbitrarily — `pick_tiling_level` snaps to the **nearest existing pyramid level**, so the achieved MPP is close to but not exactly the target (`training.md` Step 2 **[verified from data]**: observed `tile_mpp` values 0.9108 / 0.9824 / 1.01 across the mini-dataset; treat "1.0 MPP" as a target, not a guarantee). The real fgfr3mut metadata shows the same level-snapping variation.

| `target_mpp`                | Effect                                                                                                                                                                                               |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Increase** (e.g. 2.0) | Coarser / zoomed out. Each tile covers 448×448 µm — more tissue per tile, far fewer tiles for the same slide, but cellular detail lost (nuclei are ~7–10 µm, i.e. only ~3–5 px wide at 2 MPP). |
| **Decrease** (e.g. 0.5) | Finer detail; each tile covers only 112×112 µm. ~4× more tiles for the same physical coverage → more compute, and each tile sees less context.                                                   |
| **1.0 (default)**       | Paper's setting.`training.md` Step 2 cites **Suppl. Table 4: 1.0 MPP outperformed 0.5 MPP** for this task (more spatial context per tile) — so going finer is *not* automatically better. |

The grid spacing follows the same logic: `step_l0 = tile_size * target_mpp / native_mpp` (level-0 pixels), so the tile's **physical footprint stays 224 µm** regardless of which pyramid level is actually read.

---

## Parameters

| Parameter                      | Default                                                             | Meaning                                                                                                                              | If you change it                                                                                                                                                         |
| ------------------------------ | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--target_mpp`               | `1.0`                                                             | Target physical resolution (µm/px). Paper: 1.0.                                                                                     | See MPP table above.                                                                                                                                                     |
| `--tile_size`                | `224`                                                             | Tile side in**pixels** at the target resolution.                                                                               | Keep 224 for downstream compatibility (feature extractor / Chowder expect 224×224 tiles).                                                                               |
| `--min_tissue_fraction`      | `0.60`                                                            | Min fraction of a tile that must be tissue to keep it (paper: 0.60).                                                                 | Raise → fewer but purer-tissue tiles (risk dropping tumor-edge tiles). Lower → more tiles, background/glass bleeds in.                                                 |
| `--max_tiles`                | `5000`                                                            | Cap per slide; candidates randomly subsampled if exceeded (paper: "maximum of 5000 tiles uniformly sampled",`training.md` Step 4). | Raise → more instances/slide, slower training. Cap only binds on tissue-rich slides.                                                                                    |
| `--seed`                     | `0`                                                               | Seed for that random subsample.                                                                                                      | **[verified from data]** real metadata stamps `sampling_mode: {mode: "random", seed: 0}` — different seed → different tile subset → different feature matrix. |
| `--tissue_detector`          | `grandqc`                                                         | Matter-detector backend (BUNet stand-in).                                                                                            | `grandqc`: accurate, needs weights + `segmentation_models_pytorch`. `otsu`: no deps but ~9× lower tissue recall — would badly under-sample tissue.               |
| `--grandqc_weights`          | `scripts/preprocessing/models/grandqc/Tissue_Detection_MPP10.pth` | Path to GrandQC`Tissue_Detection_MPP10.pth` (Zenodo 14507273).                                                                     | Must exist when using`grandqc`; script fails fast with the download URL if missing.                                                                                    |
| `--device`                   | `auto`                                                            | `auto` → `cuda:0` if available else `cpu`; explicit `cuda*` falls back to CPU with a warning.                               | Only affects speed (GrandQC segmentation and optional PNG export).                                                                                                       |
| `--save_pngs`                | off                                                                 | Also write each kept tile as PNG.                                                                                                    | Slow + disk-heavy; default output is just`coords.npy` + `metadata.json` (coordinates, not pixels).                                                                   |
| `--svs_path` / `--out_dir` | required                                                            | Input WSI / output root.                                                                                                             | Output layout:`<out_dir>/<slide_filename>/coords.npy` + `metadata.json`.                                                                                             |

---

## Function-by-function

### Utilities

- **`log(msg)`** — timestamped, flushed `print` (works with `tail -f`).
- **`_tqdm(iterable)`** — tqdm progress bar if installed, else a plain iterator (no hard dependency).
- **`resolve_device(requested)`** — resolves `auto` to `cuda:0`/`cpu`; explicit `cuda*` without a GPU or torch → warn + fall back to CPU.

### Resolution handling (Step 2 geometry)

- **`get_native_mpp(slide)`** — reads `openslide.mpp-x` (level-0 MPP) from slide metadata; falls back to `0.25` if the file doesn't record it.
- **`pick_tiling_level(slide, native_mpp, target_mpp)`** — computes `downsample = target_mpp / native_mpp`, asks OpenSlide for the closest pyramid level (`get_best_level_for_downsample`), returns `(level, level_downsample, level_mpp)`. This is why achieved MPP snaps (0.9108 instead of exactly 1.0) — same behaviour as the real pipeline's `tile_mpp`.

### Matter detection (Step 1 — the background-removal stage)

- **`build_tissue_mask(slide)`** *(otsu backend)* — grabs a ≤2048 px thumbnail → RGB→HSV → **Otsu threshold on the saturation channel** (glass is pale/unsaturated, tissue is saturated) → morphological close 5×5 (fill holes) + open 3×3 (kill speckle). Returns `(mask 0/255, thumb_downsample)`.
- **`load_grandqc_model(weights_path, device)`** — builds `smp.UnetPlusPlus(timm-efficientnet-b0, classes=2)`, loads the Zenodo weights, returns `(model, preprocessing_fn)` in eval mode. Raises a download hint if weights are absent.
- **`build_tissue_mask_grandqc(slide, model, preprocessing_fn, ...)`** — mirrors GrandQC's own `wsi_tis_detect.py` inference exactly, because the model was trained/evaluated under these conditions:

  1. thumbnail at **MPP 10** (`model_mpp=10.0`, ~1/40 of native — tissue detection is a coarse, slide-level task);
  2. **re-encode as JPEG quality 80** (model behaves suboptimally on uncompressed inputs);
  3. split into 512×512 patches — last row/column crops are taken from the image tail, not zero-padded;
  4. ImageNet-normalize → forward pass → `argmax` over the 2-class logits;
  5. **invert** (GrandQC class 0 = tissue → we store 255 = tissue) so the output contract matches the Otsu path.

  Returns `(mask 0/255, thumb_downsample)`.

### Grid generation (Step 2 — applying the ≥60% rule)

- **`tissue_fraction_in_cell(tissue_mask, thumb_downsample, x0, y0, step_l0)`** — maps a level-0 cell `(x0, y0, step, step)` into thumbnail coordinates and returns the fraction of mask pixels > 0 inside it.
- **`generate_tile_grid(...)`** — iterates every `step_l0`-spaced cell over the whole slide; keeps `(x0, y0)` where `frac >= min_tissue_fraction`. **This is where background is discarded.** Logs `kept X/Y cells`.
- **`extract_tile(slide, x0, y0, level, level_downsample, step_l0, tile_size)`** — reads the region from the picked pyramid level (in level-0 coordinates), converts to RGB, resizes to exactly `tile_size × tile_size` with bilinear interpolation (needed because level MPP ≠ target MPP exactly).

### Orchestration

- **`tile_slide(...)`** — the full pipeline for one slide:

  1. resolve device, open slide;
  2. `get_native_mpp` → `pick_tiling_level`;
  3. build tissue mask (GrandQC or Otsu);
  4. compute `step_l0 = round(tile_size * target_mpp / native_mpp)`;
  5. `generate_tile_grid` → tissue-passing candidates;
  6. if `len(candidates) > max_tiles`: seeded (`seed`) random subsample without replacement, sorted to keep slide-scan order;
  7. save `coords.npy` (`(n_tiles, 2)` int32 level-0 top-left corners) and `metadata.json` (mpp, level, detector, `nb_tiles`, `sampling_mode`, `min_tissue_fraction`, slide size);
  8. optionally export tile PNGs (`--save_pngs`);
  9. returns the output directory path.

  Note: by default **coordinates, not pixels, are saved** — downstream steps re-read regions from the original WSI using `coords.npy` + `metadata.json`.
- **`main()`** — argparse CLI; every flag in the Parameters table maps 1:1 to a `tile_slide` keyword argument.

---

## Where this fits in the pipeline

```
slide.svs
   │
   ├─ Step 1: matter detection  (build_tissue_mask_grandqc / _otsu)   ← background removed here
   ├─ Step 2: tiling            (generate_tile_grid, ≥60% rule)       ← tissue-only tile coords
   │        → coords.npy + metadata.json
   │
   └─ Step 3: feature extraction (NOT in this script — H-optimus-0 / iBOTViTGiant per training.md)
            → features.npy (n_tiles, 1536)  →  Chowder MIL training/inference
```

If you are reproducing on TCGA: you **do not** need to run this script — the HuggingFace dataset (`PABannier/fgfr3mut`) already ships precomputed `features.npy` + `mask.npy` + `metadata.json` (`training.md` Step 3). This script is for tiling **your own raw `.svs` files** with the same geometry as the original pipeline.

*Related: `training.md` (full reproduction spec), `inference.md` (downstream stages), script docstring in `scripts/preprocessing/fgfr3mut_tile_wsi.py`.*
