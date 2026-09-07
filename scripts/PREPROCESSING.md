# Pathology Image Preprocessing Pipeline

## Overview

This pipeline preprocesses Whole Slide Images (WSI) for downstream analysis (nucleus segmentation, cell detection, etc.). It handles the complete workflow from raw SVS files to clean, normalized tiles ready for machine learning inference.

**Target Image:** TCGA H&E stained pathology slide (141,432 × 89,585 pixels at Level 0)

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PREPROCESSING PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐  │
│  │   SVS File   │───▶│    Tile      │───▶│    Noise     │───▶│Background│  │
│  │   (1.1 GB)   │    │  Extractor   │    │   Remover    │    │ Remover  │  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────┘  │
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   Tissue     │───▶│    Color     │───▶│    Save      │                  │
│  │   Detector   │    │  Normalizer  │    │  Tiles/Meta  │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Descriptions

### 1. Tile Extractor (`tile_extractor.py`)

**Purpose:** Extract overlapping patches from large SVS files without loading the entire image into memory.

**Why Overlapping Patches?**

- **Boundary artifacts:** Nuclei at tile edges need context from neighboring tiles
- **Stitching:** Overlap enables Non-Maximum Suppression (NMS) for deduplication
- **No gaps:** Overlap ensures no tissue is missed between tiles

**Key Parameters:**

| Parameter      | Default | Description                           |
| -------------- | ------- | ------------------------------------- |
| `patch_size` | 1024    | Size of each square patch (pixels)    |
| `overlap`    | 128     | Overlap between adjacent patches      |
| `stride`     | 896     | Calculated as`patch_size - overlap` |

**How It Works:**

1. Opens SVS file using `tifffile` (tiled JPEG reading)
2. Generates grid coordinates with stride = 896
3. Extracts only the needed tiles (never loads full 38GB image)
4. Pads edge tiles to maintain consistent size

**Memory Efficiency:**

- Full image: ~38 GB uncompressed
- Single tile: ~3 MB (1024×1024×3 bytes)
- Only one tile in memory at a time

---

### 2. Noise Remover (`noise_remover.py`)

**Purpose:** Remove various types of noise from pathology image tiles.

**Why Noise Removal?**

- Scanner artifacts (salt-and-pepper noise)
- Electronic sensor noise (Gaussian noise)
- Dust and debris on slide

**Filter Pipeline:**

```
Input → Gaussian Blur → Median Filter → Morphological Opening → Output
```

| Filter                | Purpose                | Why This Order?                               |
| --------------------- | ---------------------- | --------------------------------------------- |
| Gaussian Blur         | Smooth random noise    | First pass: reduce high-frequency noise       |
| Median Filter         | Remove impulse noise   | Second pass: remove remaining salt-and-pepper |
| Morphological Opening | Remove small artifacts | Final pass: clean up debris                   |

**Key Parameters:**

| Parameter              | Default | Description                          |
| ---------------------- | ------- | ------------------------------------ |
| `gaussian_kernel`    | 3       | Kernel size for Gaussian blur        |
| `median_kernel`      | 3       | Kernel size for median filter        |
| `morphological_disk` | 2       | Radius for morphological disk kernel |

---

### 3. Background Remover (`background_remover.py`)

**Purpose:** Separate tissue from glass/background in pathology tiles.

**Why Background Removal?**

- Reduces processing time (skip glass regions)
- Improves segmentation accuracy (no false positives on glass)
- Enables tissue percentage calculation

**Method: Otsu's Thresholding**

1. Convert RGB to grayscale
2. Apply Otsu's automatic thresholding
3. Invert mask (tissue=white, background=black)
4. Post-process: morphological closing + remove small objects

**Post-Processing Steps:**

| Step                  | Purpose                         |
| --------------------- | ------------------------------- |
| Morphological Closing | Fill small holes in tissue mask |
| Remove Small Objects  | Remove debris and artifacts     |

**Key Parameters:**

| Parameter              | Default | Description                     |
| ---------------------- | ------- | ------------------------------- |
| `method`             | "otsu"  | Thresholding method             |
| `morphological_disk` | 5       | Radius for closing holes        |
| `min_area_percent`   | 1.0     | Minimum object size (% of tile) |

---

### 4. Tissue Detector (`tissue_detector.py`)

**Purpose:** Identify whether a tile contains valid tissue for processing.

**Why Tissue Detection?**

- Not all extracted tiles contain tissue
- Processing empty tiles wastes computation
- Focus check ensures image quality

**Assessment Criteria:**

| Criterion         | Method             | Threshold           |
| ----------------- | ------------------ | ------------------- |
| Tissue Percentage | Otsu thresholding  | >10% tissue         |
| Focus Quality     | Laplacian variance | >100 (variance)     |
| Staining Quality  | HSV analysis       | Moderate saturation |

**Decision Logic:**

```
is_valid = has_tissue AND is_focused AND is_stained
```

**Key Parameters:**

| Parameter              | Default     | Description                   |
| ---------------------- | ----------- | ----------------------------- |
| `min_tissue_percent` | 10.0        | Minimum tissue % to keep tile |
| `focus_method`       | "laplacian" | Method for focus assessment   |
| `focus_threshold`    | 100.0       | Minimum focus score           |

---

### 5. Color Normalizer (`color_normalizer.py`)

**Purpose:** Standardize stain colors across tiles for consistent analysis.

**Why Color Normalization?**

- Stain variation across slides/labs
- Machine learning models are sensitive to color
- Reduces batch effects

**Method: Percentile Normalization**

1. Convert to HED (Hematoxylin-Eosin-DAB) color space
2. Normalize hematoxylin channel (nuclei stain)
3. Normalize eosin channel (cytoplasm stain)
4. Convert back to RGB

**Key Parameters:**

| Parameter           | Default      | Description          |
| ------------------- | ------------ | -------------------- |
| `method`          | "percentile" | Normalization method |
| `percentile_low`  | 1            | Lower percentile     |
| `percentile_high` | 99           | Upper percentile     |

---

### 6. Main Pipeline (`preprocess_pipeline.py`)

**Purpose:** Orchestrate all modules into a single, configurable pipeline.

**Pipeline Flow:**

```python
for each tile in SVS:
    1. Extract tile from SVS
    2. Remove noise
    3. Remove background
    4. Detect tissue
    5. If valid:
        - Normalize color
        - Save tile
        - Save metadata
    6. Else:
        - Save to rejected (optional)
```

---

## Configuration (`config.yaml`)

All parameters are centralized in `config.yaml`:

```yaml
input:
  svs_path: "../image/TCGA-...svs"
  level: 0

patch_extraction:
  patch_size: 1024
  overlap: 128

noise_reduction:
  gaussian_kernel: 3
  median_kernel: 3
  morphological_disk: 2

background_removal:
  method: "otsu"
  morphological_disk: 5

tissue_detection:
  min_tissue_percent: 10.0
  focus_threshold: 100.0

color_normalization:
  method: "percentile"
```

---

## Output Structure

```
output/preprocessed/
├── tiles/              # Preprocessed tiles (PNG)
│   ├── tile_000000.png
│   ├── tile_000001.png
│   └── ...
├── masks/              # Tissue masks (PNG)
│   ├── mask_000000.png
│   ├── mask_000001.png
│   └── ...
├── metadata/           # Tile metadata (JSON)
│   ├── tiles.json      # Tile positions, tissue %, etc.
│   ├── processing_stats.json
│   └── report.html
└── rejected/           # Rejected tiles (optional)
    ├── tile_000042.png
    └── ...
```

**Metadata Schema (`tiles.json`):**

```json
[
  {
    "tile_id": 0,
    "x": 0,
    "y": 0,
    "width": 1024,
    "height": 1024,
    "tissue_percentage": 45.2,
    "is_focused": true,
    "focus_score": 1523.4,
    "is_stained": true,
    "filename": "tile_000000.png",
    "mask_filename": "mask_000000.png"
  }
]
```

---

## Usage

### Basic Usage

```bash
cd scripts
python preprocess_pipeline.py
```

### With Command-Line Arguments

```bash
python preprocess_pipeline.py \
    --svs "../image/TCGA-...svs" \
    --level 0 \
    --roi 0 0 10000 10000 \
    --max-tiles 100 \
    --output-dir "../output/pilot"
```

### Resume Interrupted Processing

```bash
python preprocess_pipeline.py --resume-from 5000
```

---

## Parallel Processing (Hadoop)

The pipeline is designed for parallel execution:

**Why Parallel?**

- Each tile is independent (no dependencies)
- Tiles can be processed on different nodes
- Enables horizontal scaling

**Hadoop Integration Strategy:**

1. **Split:** Generate tile coordinates and save to file
2. **Map:** Each mapper processes one or more tiles
3. **Reduce:** Merge metadata files

**Example Split File (`tile_coordinates.json`):**

```json
[
  {"tile_id": 0, "x": 0, "y": 0, "x_end": 1024, "y_end": 1024},
  {"tile_id": 1, "x": 896, "y": 0, "x_end": 1920, "y_end": 1024},
  ...
]
```

**Mapper Script:**

```python
import json
import sys
from preprocess_pipeline import PreprocessingPipeline

# Load tile coordinates
tile_id = int(sys.argv[1])
coords = load_coords(tile_id)

# Process single tile
pipeline = PreprocessingPipeline()
tile = pipeline.tile_extractor.extract_tile(...)
metadata = pipeline.process_single_tile(tile, coords, tile_id)

# Output JSON
print(json.dumps(metadata))
```

---

## Dependencies

```bash
pip install:
- tifffile
- imagecodecs
- opencv-python
- scikit-image
- numpy
- pyyaml
- tqdm
```

---

## Performance

**Estimated Processing Times (Level 0, 141K×89K):**

| Mode               | Hardware | Time        |
| ------------------ | -------- | ----------- |
| Full Slide GPU     | RTX 4080 | ~25-40 min  |
| Full Slide CPU     | 16 cores | ~8-14 hours |
| ROI (10K×10K) GPU | RTX 4080 | ~2-4 min    |
| ROI (10K×10K) CPU | 16 cores | ~25-40 min  |

**Memory Usage:**

- Peak RAM: ~4-8 GB (with batch size 1)
- GPU VRAM: ~3-4 GB (StarDist 512×512)

---

## Troubleshooting

| Issue                 | Solution                                            |
| --------------------- | --------------------------------------------------- |
| Out of Memory         | Reduce`patch_size` to 512 or use `batch_size=1` |
| Slow Processing       | Use Level 1 instead of Level 0 (4× faster)         |
| Poor Tissue Detection | Adjust`min_tissue_percent` in config              |
| Unfocused Tiles       | Increase`focus_threshold`                         |
| Color Variation       | Try`method: "macenko"` for better normalization   |

---

## References

1. Macenko, M., et al. (2009). A method for normalizing histology slides for quantitative analysis. *ISBI*.
2. Reinhard, E., et al. (2001). Color transfer between images. *IEEE CGA*.
3. Otsu, N. (1979). A threshold selection method from gray-level histograms. *IEEE TSMC*.

---

---

## Alternative Pipeline: Direct Macenko Normalization (`patch_normalization_pipeline.py` + `macenko_normalizer.py`)

**Purpose:** Lightweight alternative that skips noise/background/tissue filtering and directly normalizes patches via Macenko (from `122_normalizing_HnE_images.py`). Keeps **raw** and **normalized** patches in **separate folders** for easier downstream processing.

**When to use vs original pipeline:**

| Pipeline                                  | Steps                                                       | Best for                                                                          |
| ----------------------------------------- | ----------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `preprocess_pipeline.py` (original)     | Extract → Noise → BG → TissueDetect → ColorNorm         | Clean, filtered dataset (tissue-only, QC)                                         |
| `patch_normalization_pipeline.py` (new) | Extract → Save raw → Macenko normalize → Save normalized | Fast, unbiased patches; keep raw+normalized paired; ML training with augmentation |

**Architecture:**

```
SVS --(TileExtractor)--> raw patch --(MacenkoNormalizer)--> normalized patch
                          |                                   |
                          v                                   v
                    output/patches/raw/              output/patches/normalized/
                          \                                   /
                           ===== metadata/tiles.json =====
```

**Modules:**

1. `macenko_normalizer.py` – Refactored `122_normalizing_HnE_images.py` into `MacenkoNormalizer` class + `normalizeStaining()` function. Handles transparent-tile edge case (returns original if OD pixels < 10).
2. `patch_normalization_pipeline.py` – Orchestrates extraction + normalization with:

   - Separate output folders (`raw/`, `normalized/`, `metadata/`)
   - `max_tiles` (default 10 for quick test) vs `--all` for full WSI (15800 tiles)
   - Same filename in both folders (`patch_000006.png`) for easy pairing
   - JSON metadata linking coords to both paths

**Config (`config.yaml` additions):**

```yaml
macenko_normalization:
  Io: 240
  alpha: 1
  beta: 0.15

macenko_output:
  base_dir: "../output/patches"
  raw_dir: "raw"
  normalized_dir: "normalized"
```

**Usage:**

```bash
cd scripts

# Quick test: first 10 patches (default)
python patch_normalization_pipeline.py
python patch_normalization_pipeline.py --max-tiles 10

# Custom count
python patch_normalization_pipeline.py --max-tiles 100 --output-dir ../output/pilot

# Tissue-rich ROI demo (first 10 tissue patches)
python patch_normalization_pipeline.py --roi 12544 896 5000 5000 --max-tiles 10 --output-dir ../output/patches_tissue_demo

# FULL WSI (15800 patches, ~15-20 min on CPU)
python patch_normalization_pipeline.py --all
python patch_normalization_pipeline.py --max-tiles 0   # 0 also means all

# Different pyramid level or SVS
python patch_normalization_pipeline.py --svs ../image/other.svs --level 1 --all
```

**Output Structure:**

```
output/patches/
├── raw/                      # Raw extracted patches
│   ├── patch_000000.png
│   └── ...
├── normalized/               # Macenko-normalized patches (same filenames)
│   ├── patch_000000.png
│   └── ...
└── metadata/
    ├── tiles.json            # [{patch_id, x, y, raw_filename, normalized_filename}, ...]
    └── stats.json            # {total_available, processed, duration, raw_dir, normalized_dir}
```

**Verification (already run for first 10):**

```bash
ls output/patches/raw | wc -l          # 10
ls output/patches/normalized | wc -l   # 10
# Tissue demo
ls output/patches_tissue_demo/raw      # 10 tissue-rich patches, avg raw [226 216 228] -> normalized [220 214 229]
```

---

*Generated for Pathology Image Processing Project - Hadoop Parallel Pipeline*
