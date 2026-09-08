# Pathology Image Processing — Pipeline Summary

**WSI:** `image/TCGA-FJ-A871-01Z-00-DX5.8F79D0A8-5DE6-4159-AA77-61DACB21E867.svs` (141432×89585 at level 0, ~1.1 GB)
**Date:** 2026-09-07
**Goal:** Detect nuclei and characterize nucleus features (count, size, shape, chromatin texture, N/C ratio) for downstream pathology analysis.

---

## 1. Overview of Two Pipelines

| Aspect                 | Pipeline 1: Full Preprocessing                           | Pipeline 2: Direct Macenko Normalization                                                                                                     |
| ---------------------- | -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Script                 | `scripts/preprocess_pipeline.py:36`                    | `scripts/patch_normalization_pipeline.py:43` + `scripts/macenko_normalizer.py:20` (refactored from `122_normalizing_HnE_images.py:47`) |
| Philosophy             | Filter aggressively, keep only high-quality tissue tiles | Extract unbiased, keep everything, normalize stain only                                                                                      |
| Filtering              | Yes (tissue%, focus, stain)                              | No (all tiles retained)                                                                                                                      |
| Output size (full WSI) | 3235 / 15800 tiles accepted (~20.5%)                     | 15800 / 15800 tiles (100%, paired)                                                                                                           |

---

## 2. Pipeline 1 — Full Preprocessing (Noise + Background + Tissue + Color)

### 2.1 Steps

1. **Tile Extraction** `scripts/tile_extractor.py:27` — `patch_size=1024`, `overlap=128`, `stride=896`. Tiled reading via OpenSlide, never loads full 38 GB image. Generates 15800 coordinates.
2. **Noise Removal** `scripts/noise_remover.py:26` — Gaussian (k=3) → Median (k=3) → Morphological opening (disk=2). Targets Gaussian sensor noise, salt-and-pepper, debris.
3. **Background Removal** `scripts/background_remover.py:27` — Grayscale Otsu + inversion + closing (disk=5) + remove small objects (<1% tile). Produces binary tissue mask.
4. **Tissue Detection** `scripts/tissue_detector.py:25` — `min_tissue_percent=5.0`, `focus_method=laplacian` threshold 50.0, HSV stain check. Decision: `is_valid = has_tissue AND is_focused AND is_stained` `scripts/tissue_detector.py:235`.
5. **Color Normalization** `scripts/color_normalizer.py:30` — `method=percentile` (low=1, high=99) in `config.yaml:39`. Per-channel percentile scaling.

### 2.2 Expected Output

```
output/preprocessed/
├── tiles/          # 3235 PNGs, e.g., tile_000172.png (31.2% tissue, focus 79.0)
├── masks/          # 3235 binary masks, tissue=white (255), background=black (0)
├── metadata/
│   ├── tiles.json              # [{tile_id, x, y, width, height, tissue_percentage, is_focused, focus_score, is_stained, filename, mask_filename}]
│   ├── processing_stats.json   # {total_tiles:15800, accepted:3235, rejected:12565, mean_tissue:65.3%, duration:2435s}
│   └── report.html
└── rejected/       # optional, if quality_control.save_rejected=true
```

Verified run `output/preprocessed/metadata/processing_stats.json:1` on 2026-09-03.

### 2.3 How to Utilize

- **QC filtering:** Use `tiles.json` to sample only high-tissue / in-focus tiles for inference, avoiding wasted compute on glass.
- **Mask-restricted inference:** Feed `masks/` to nucleus segmenter to skip background pixels, reducing false positives on glass.
- **Pre-cleaned input:** Denoised + background-masked tiles improve classical segmentation (watershed, thresholding) where noise is problematic.
- **Config reuse:** `config.yaml:14` centralizes `patch_extraction`, `noise_reduction`, `background_removal`, `tissue_detection` for reproducible sweeps.

---

## 3. Pipeline 2 — Direct Macenko Normalization (Raw + Normalized, Separate Folders)

### 3.1 Steps

1. **Tile Extraction** — Same `TileExtractor` (1024/128), no denoising. Reads region via `extract_tile():107`.
2. **Save Raw** — `output/patches/raw/patch_000000.png` (RGB→BGR via cv2). One file per coordinate.
3. **Macenko Normalize** `scripts/macenko_normalizer.py:47` — Workflow from `122_normalizing_HnE_images.py:62-154`:
   - OD = -log10((I+1)/Io), Io=240, beta=0.15 transparent filter
   - Covariance SVD (2 largest eigenvectors), angle phi, robust extremes alpha=1
   - Stain vectors HE, concentrations C via lstsq, normalization by `HERef=[[0.5626,0.2159],[0.7201,0.8012],[0.4062,0.5581]]` and `maxCRef=[1.9705,1.0308]` `scripts/macenko_normalizer.py:8`
   - Reconstruct `Inorm = Io * exp(-HERef·C2)`, clipped to 254. Edge case: if ODhat<10, return original with warning.
   - Optional `normalize_with_he():116` returns Inorm, H, E separately.
4. **Save Normalized** — `output/patches/normalized/patch_000000.png` (identical filename, different folder for pairing).
5. **Metadata** — `output/patches/metadata/tiles.json` links `patch_id, x, y, raw_filename, normalized_filename, raw_path, normalized_path`.

### 3.2 Configuration

`scripts/config.yaml:52`:

```yaml
macenko_normalization: {Io: 240, alpha: 1, beta: 0.15}
macenko_output: {base_dir: "../output/patches", raw_dir: "raw", normalized_dir: "normalized"}
```

### 3.3 Expected Output

```
output/patches/
├── raw/            # Raw extracted patches (15800 if --all, 10 in current test)
├── normalized/     # Macenko-normalized patches, 1:1 paired by filename
└── metadata/
    ├── tiles.json  # [{patch_id, x, y, width, height, raw_filename, normalized_filename}]
    └── stats.json  # {total_available:15800, total_requested, processed, failed, duration_seconds}
```

Current test: `output/patches/raw/` and `output/patches/normalized/` each 10 files (first 10 grid positions, mostly background). Tissue demo `output/patches_tissue_demo/` (ROI 12544,896,5000,5000) shows meaningful shift e.g., raw mean [226,216,228] → normalized [220,214,229] `scripts/patch_normalization_pipeline.py:1`.

### 3.4 Usage

```bash
cd scripts
python patch_normalization_pipeline.py --max-tiles 10                          # quick test (default)
python patch_normalization_pipeline.py --max-tiles 100 --output-dir ../output/pilot
python patch_normalization_pipeline.py --roi 12544 896 5000 5000 --max-tiles 10  # tissue-rich ROI
python patch_normalization_pipeline.py --all                                   # full WSI (~15-20 min CPU)
python patch_normalization_pipeline.py --svs ../image/other.svs --level 1 --all
```

Same patch index in both folders enables: side-by-side QC, stain-augmentation training, and direct comparison in viewers.

### 3.5 How to Utilize

- **Stain-invariant training:** Use `normalized/` as input to StarDist / HoVer-Net / CellViT to reduce batch effects across TCGA sites. Keep `raw/` for augmentation and stain-variation modeling.
- **Hematoxylin channel:** `normalize_with_he()` yields H-only image for nucleus-specific thresholding without eosin cytoplasm interference.
- **No information loss:** Raw retained; morphology (nuclear boundary, chromatin granules) not smoothed by Gaussian/median.
- **Pairwise learning:** Raw→normalized mapping can train lightweight stain-transfer models or serve as validation for Macenko parameters.
- **Seamless iteration:** Separate folders simplify PyTorch `ImageFolder` or `paired Dataset` without parsing metadata.

---

## 4. Comparison for Future Nucleus Work

| Criterion             | Pipeline 1                                        | Pipeline 2                                                |
| --------------------- | ------------------------------------------------- | --------------------------------------------------------- |
| Tissue purity         | High (filters glass/OOF)                          | Low (requires downstream filtering if needed)             |
| Morphology fidelity   | Risk of over-smoothing faint nuclei               | Preserved (no denoising)                                  |
| Stain constancy       | Percentile (fast, approximate)                    | Macenko (gold standard, stain-vector based)               |
| Raw recoverability    | No (masked/blurred tiles only)                    | Yes (raw + normalized)                                    |
| Metadata for sampling | Rich (tissue%, focus, stain)                      | Minimal (coords only)                                     |
| Best downstream use   | Classical pipelines sensitive to noise/background | Deep learning nucleus segmenters needing stain invariance |

**Recommended workflow:** Use Pipeline 2 as primary input for nucleus detection. Optionally join `output/preprocessed/masks/` or `tissue_detector` logic as a secondary filter (e.g., skip inference where tissue% <5% or Laplacian <50) rather than in-place denoising.

---

## 5. Will These Pipelines Help Nucleus Detection and Characterization?

Yes, as preprocessing but not as the detector. Both address the two dominant failure modes for H&E nucleus models: out-of-distribution stain color and background/OOF false positives.

Pipeline 1 helps by guaranteeing that a detector sees only in-focus tissue, which improves precision and reduces compute (80% tiles discarded) — useful when scaling to many WSIs or when using classical methods. Its limitation is that aggressive denoising/masking can attenuate small/pale nuclei and bias sampling toward dense tissue, potentially underrepresenting sparse or edge nuclei.

Pipeline 2 helps more directly for modern deep nucleus detection and characterization. Macenko normalization aligns hematoxylin/eosin vectors to a common reference, so a model trained on one site generalizes to others — critical for TCGA heterogeneity. Keeping `raw/` and `normalized/` separate supports systematic evaluation (e.g., measure detector performance delta with/without normalization) and enables extraction of accurate nuclear features: normalized H-channel yields stable thresholding for size/shape, while normalized RGB preserves N/C contrast for chromatin and cytoplasm assessment. The pipeline does not segment nuclei itself; a dedicated segmenter (StarDist, HoVer-Net) must follow. With that segmenter, the pipelines provide the stain-consistent, well-paired, QC-annotated tile set required for reliable nucleus detection and quantitative phenotyping.

*Generated for Pathology Image Processing Project — 2026-09-07*
