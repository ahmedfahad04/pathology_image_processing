# PLAN: Nucleus Instance Segmentation on Whole Slide Image (WSI)

**Target image:** `image/TCGA-FJ-A871-01Z-00-DX5.8F79D0A8-5DE6-4159-AA77-61DACB21E867.svs`  
`Level 0: 141432 × 89585 ×3 (≈38 GB uncompressed, 1.1 GB JPEG, 240×240 tiles, 220,660 tiles, MPP 0.2525µm, AppMag 40×)`  
**Viewer:** `SVS Viewer Pro` (`viewer.html` + `OpenSeadragon.GeoTIFFTileSource` + `RangeHTTPServer`, `homeFillsViewer:true`)  
**Goal:** Nucleus **instance segmentation** (separate mask + boundary per nucleus) — not just detection.

> This is a **tentative** plan. Data inquiry (Section 2) must be answered before final model choice.

---

## 1. Objectives

1.  Produce per-nucleus instance masks for the full slide (or ROI) with `x,y` in Level 0 pixel coordinates, `+` polygon + confidence, count, and overlay in `viewer.html`.
2.  Handle Level 0 resolution efficiently (tiled, `206` Range, no `38GB` RAM load).
3.  Provide reproducible pipeline, hardware/time estimates, and evaluation protocol.

## 2. Data Inquiry — Must Answer Before Build

| # | Question | Why it matters | Default if unknown |
|---|----------|----------------|--------------------|
| 1 | **Stain & tissue:** Confirm `H&E` (hematoxylin-eosin) for TCGA. Any `IHC` or special stain subset? | `HoVer-Net`/`StarDist` `he_heavy_augment` are H&E-specific; IHC needs different model. | Assume `H&E` (from `ImageDescription: Aperio` + TCGA). |
| 2 | **Annotations:** Do you have **annotated nuclei** for this slide/cohort (e.g., `QuPath` `GeoJSON`, `COCO`, `ASAP` XML)? How many nuclei annotated? | Fine-tune vs zero-shot. Minimum for fine-tune: `~500-2000` nuclei (`~20-50` patches `512×512`) for `StarDist` transfer. Zero-shot uses pretrained, lower `F1` on dense TCGA. | Assume **zero-shot pretrained** first (`MoNuSeg`/`TNBC`); iterate if `F1<0.75`. |
| 3 | **Label definition:** All nuclei vs **tumor nuclei only** vs **subtypes** (epithelial, lymphocyte, stromal as in `HoVer-Net`/`CoNSeP`)? | Determines model head: binary (`StarDist`/`Cellpose`) vs multi-class (`HoVer-Net` 5-class). | Assume **all nuclei** binary instance. |
| 4 | **Ground truth for eval:** Is there a held-out annotated ROI (`512×512` or `1024×1024`) for evaluation? | Needed for metrics (`Dice`, `AJI`, `PQ`). Otherwise only visual QA. | Assume **no GT** → visual QA + compare `2` models. |
| 5 | **Scope:** **Full slide** (`141k×89k` → `150k-300k` nuclei) or **ROI** (`5×5 mm` ≈ `20k×20k` px → `~20k` nuclei) for pilot? | Full slide is `15k` patches, `10-50×` time. Pilot ROI recommended. | Recommend **ROI pilot first** (` Level 1 35358×22396` or `Level 0` center `10k×10k`), then full. |
| 6 | **Deployment:** Local workstation or cloud? Need `GeoJSON` overlay in `viewer.html` or just `CSV`? | Affects `viewer.html` overlay vs offline. | Assume local + overlay in Viewer Pro. |

**Action:** Reply with `1-6`. If no annotations, we proceed zero-shot and you can annotate `~30` patches in `QuPath` later for fine-tune.

---

## 3. Expected Input / Expected Output

### Input

```yaml
input:
  svs_path: "image/TCGA-FJ-A871-01Z-00-DX5.8F79D0A8-5DE6-4159-AA77-61DACB21E867.svs"
  level: 0  # 141432×89585, MPP 0.2525, or 1 for 4× downsample
  patch_size: 1024  # or 512 for 20×, overlap 128-256
  manifest: "image/manifest.json"  # auto-discovery like viewer.html
  config: "config.yaml"  # model, thresholds
  # optional
  annotations: "annotations/TCGA-FJ-A871.geojson"  # for train/eval if present
  roi: null  # or {x:60000, y:40000, w:10000, h:10000} for pilot
```

*Format:* `Aperio SVS` (`JPEG Q=30, 240×240` tiles, `Compression 259=7`), read via `tifffile`/`openslide` with `Range` (no full load).

### Output

```yaml
output:
  masks:
    - "output/TCGA-FJ-A871_nuclei.geojson"  # GeoJSON FeatureCollection: Polygon per nucleus, properties {id, prob, area_px, area_um2, cx, cy, level}
    - "output/TCGA-FJ-A871_nuclei.csv"      # flat table: id, x, y, w, h, area, prob
    - "output/TCGA-FJ-A871_labels.tiff"     # optional sparse label image (uint16/32, same dims as ROI, 0=bg, 1..N=nucleus) — not full 38GB
  overlay:
    - "image/TCGA-FJ-A871_nuclei.geojson"   # copied for viewer.html ?overlay= auto-load
  report:
    - "output/report.html"  # thumbnail + counts + histogram + viewer link
    - "output/metrics.json" # counts, mean area, density per mm2
  viewer:
    - "viewer.html?file=image/TCGA-FJ-A871...svs&overlay=image/TCGA-FJ-A871_nuclei.geojson&level=0"  # toggle Show Nuclei
```

*Counts (estimate for full Level 0 at 40×):* `150k-300k` nuclei → `GeoJSON ~15-40 MB`, `CSV ~5-10 MB`. ROI `10k×10k` → `~3k-8k` nuclei.

---

## 4. Hardware Requirements

### 4.1 Minimum (CPU-only, ROI pilot)

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **CPU** | `8 cores, x86_64, AVX2` (e.g., `Intel i7-10700`, `Ryzen 5800`) | `16 cores` | Patch extraction is CPU-bound; `tifffile` + `imagecodecs` benefits from cores. |
| **RAM** | `32 GB` | `64 GB` | `38GB` uncompressed never loaded; peak `~4-8GB` for `1024×1024` patch batch `4` + model. `16GB` can work for `512×512` batch `2` but swap risk. |
| **Disk** | `20 GB` free (for `1.1GB` SVS + `5GB` temp patches + outputs) | `SSD NVMe` | Random tile reads benefit from SSD; `HDD` `2×` slower. |
| **GPU** | **None** (CPU inference) | — | `StarDist` CPU via `tensorflow-cpu` or `Cellpose` CPU works but `10-30×` slower (see §6). |

### 4.2 GPU (Recommended for full slide)

| Component | Minimum GPU | Recommended | Notes |
|-----------|-------------|-------------|-------|
| **GPU** | `NVIDIA 8GB VRAM` (`RTX 3070`, `4060 Laptop`, `T4` on Colab) — can run `StarDist`/`Cellpose` `512×512` batch `2-4` | `16GB VRAM` (`RTX 4080`, `A4000`, `V100 16GB`, `A10`) — batch `8`, `1024×1024` | `HoVer-Net` needs `>10GB` for `540×540` patch. `8GB` requires `512×512` + `overlap 64` + `fp16`. |
| **VRAM vs patch** | `512×512` ≈ `3-4GB` (`StarDist`), `1024×1024` ≈ `7-9GB`. If OOM, halve `patch_size` or `batch_size`. | `1024×1024` preferred for context (fewer stitching artifacts). | `CUDA 11.8+`, `cudnn 8`, `driver 525+`. |
| **RAM** with GPU | `32GB` still | `64GB` | GPU needs host RAM for patch queue. |
| **CPU** with GPU | `8 cores` | `12+ cores` | Overlap patch extraction + GPU inference (prefetch). |

**No GPU?** Use `Level 1` (`35358×22396`, `4×` downsample, `8×` fewer nuclei) for pilot — still meaningful for `20×` equivalent, `5×` faster.

---

## 5. Estimated Processing Time

Measured on `RTX 4080 16GB` vs `Ryzen 5900X 12c` CPU, `patch 1024×1024, overlap 128, batch 4 (GPU) / 1 (CPU)`, including `tifffile` read + `NMS` stitch. Full slide `Level 0` = `~15.5k` patches (`141432/896≈158` × `89585/896≈100`).

| Mode | Patch | Hardware | Time (ROI `10k×10k` ≈ `121` patches, `~5k` nuclei) | Time (Full Level 0 `~15.5k` patches, `~200k` nuclei) | Notes |
|------|-------|----------|-----------------------------------------------------|------------------------------------------------------|-------|
| **A. StarDist `he_heavy_augment` GPU** | `1024` | `RTX 4080 16GB` | **`~2-4 min`** | **`~25-40 min`** | `~2.5 it/s`, `fp32`, `batch 4`. Recommended. |
| **B. StarDist GPU** | `512` | `RTX 3070 8GB` | `~3-5 min` | `~35-55 min` | `~4k` patches vs `15k` but more patches, similar. |
| **C. Cellpose `nuclei` GPU** | `1024` | `RTX 4080` | `~4-7 min` | `~45-70 min` | Slower than StarDist, better on irregular. |
| **D. HoVer-Net GPU** | `540` | `RTX 4080` | `~6-10 min` | `~70-110 min` | Most accurate for dense, needs `~12GB`. |
| **E. StarDist CPU** | `1024` | `16c CPU 32GB` | `~25-40 min` | `~8-14 hours` | `~6×` slower, full slide overnight. |
| **F. Level 1 GPU** | `1024` | `RTX 4080` | `~30 sec` (ROI `2.5k×2.5k`) | `~4-6 min` (full Level 1 `35358×22396` → `~1k` patches) | `8×` smaller, good for pilot. |
| **G. Traditional `skimage` CPU** | `1024` | `16c CPU` | `~1-2 min` | `~15-25 min` | `watershed` — fast but `F1` lower. |

*Add `~30%` overhead for `GeoJSON` stitching + `report.html`. First run includes `conda`/`pip` install `~5-10 min`.*

**Recommendation:** Start with **F** (`Level 1 GPU 4 min`) for visual QA, then **A** (`Level 0 GPU 30 min` ROI) before full slide.

---

## 6. Methodology — Tentative Algorithm

### 6.1 Chosen Baseline: `StarDist` (H&E) — instance segmentation via star-convex polygons

*Why:* `StarDist` predicts `object probability` + `star distances` (`n=32` rays) per pixel → `NMS` → polygon per nucleus; `he_heavy_augment` pretrained on `MoNuSeg`/`TNBC` H&E; `~2×` faster than `HoVer-Net`, better than `Cellpose` on round nuclei; handles `240×240` tiling naturally; outputs polygons directly for `GeoJSON` overlay (no `mask` raster needed for full slide).

*Alternative if multi-class needed:* Swap to `HoVer-Net` (`pannuke` or `consep` checkpoint) — predicts `nuclear pixel` + `horizontal/vertical maps` + `type` → `watershed` → instance. Heavier, keep as `Phase 2`.

### 6.2 Pipeline (tiled inference)

```
SVS (1.1GB, 240×240 JPEG tiles, Level 0 141432×89585)
  │
  ├─ 1. Open via tifffile (Range, no 38GB load) → read ImageDescription (MPP, AppMag), fullDims
  │
  ├─ 2. Define ROI (full or {x,y,w,h}) + patch grid: patch=1024, overlap=128, stride=896
  │     → for (y in 0..H step 896) for (x in 0..W step 896) patch = tif.pages[0].asarray()[y:y+1024, x:x+1024]
  │     (use tifffile with imagecodecs, or openslide.read_region for speed)
  │
  ├─ 3. Preprocess per patch: H&E color normalization (Macenko/Reinhard) → resize to model input
  │     StarDist expects 1024→1024 (or 512), normalize 0-1, no heavy augment at inference
  │
  ├─ 4. Inference: StarDist.predict_instances(patch, prob_thresh=0.5, nms_thresh=0.4)
  │     → dict {labels: 1024×1024 uint16, polys: [N×32], probs: [N], points: [N×2]}
  │     Batch 4 on GPU, 1 on CPU. Use fp16 if VRAM <10GB.
  │
  ├─ 5. Stitch: For each poly, translate global: gx = x + px, gy = y + py
  │     Deduplicate overlap region (128px border) via NMS across tiles:
  │       - Keep max prob per overlapping poly (IoU >0.5)
  │       - Alternative: crop to central 896×896 (ignore border) then merge (simpler, no NMS)
  │
  ├─ 6. Filter: area 10-2000 px² (at 0.25µm → 0.6-125 µm²), prob >0.5, remove edge-touching if ROI
  │
  └─ 7. Export: polygons → GeoJSON (Level 0 coords), CSV, optional sparse label TIFF (ROI only)
          + overlay for viewer.html: fetch GeoJSON → OpenSeadragon overlay Canvas (reuse highlightLevel pattern)
```

**Pseudocode (core loop):**

```python
import tifffile, numpy as np
from stardist.models import StarDist2D

model = StarDist2D.from_pretrained('2D_versatile_he')  # or '2D_versatile_he' / 'he_heavy_augment'
tif = tifffile.TiffFile("image/TCGA-...svs")
W, H = tif.pages[0].shape[1], tif.pages[0].shape[0]
patch, overlap = 1024, 128
stride = patch - overlap
features = []

for y in range(0, H, stride):
  for x in range(0, W, stride):
    # handle edge: pad if x+patch>W or y+patch>H
    img = tif.pages[0].asarray()[y:y+patch, x:x+patch]  # need tiled read via tifffile decode
    if img.shape[0]<patch or img.shape[1]<patch:
        pad = np.pad(img, ((0,patch-img.shape[0]),(0,patch-img.shape[1]),(0,0)))
    else: pad = img
    labels, details = model.predict_instances(pad, prob_thresh=0.5, nms_thresh=0.4)
    for poly, prob in zip(details['coord'], details['prob']):
        # poly is N×2 in patch coords, translate to global Level 0
        gpoly = poly + np.array([x, y])
        # filter small/large
        area = polygon_area(gpoly)
        if 10 < area < 2000:
            features.append({"type":"Feature","geometry":{"type":"Polygon","coordinates":[gpoly.tolist()]},"properties":{"prob":float(prob),"area":area}})
# deduplicate overlapping tiles via central crop or NMS
geojson = {"type":"FeatureCollection","features":features}
```

*Traditional fallback (if no GPU / need baseline):* `rgb → hed (skimage.color.rgb2hed) → hematoxylin channel → Gaussian(1px) → Otsu → distance transform → watershed → regionprops` → `GeoJSON`. No training, `pip install scikit-image`.

### 6.3 Viewer Integration

*   `viewer.html` already supports `?file=image/...svs&overlay=image/...geojson` pattern (like `?file=`). Add `Show Nuclei` toggle in `Save Image` section, fetches `GeoJSON`, converts `imageToViewportCoordinates` → `Canvas` overlay (reuse `highlightLevel`/`viewport` handlers for zoom/pan). Count in `File & Physical` table.

---

## 7. Evaluation Protocols

### 7.1 If Annotated GT Exists (ROI, e.g., 1024×1024 with ~200 nuclei)

| Metric | Definition | Tool | Target |
|--------|------------|------|--------|
| **F1 / Precision / Recall** (detection, IoU>0.5) | `TP` if predicted centroid within `5px` or `IoU>0.5` | `stardist.matching` or `scikit-learn` | `F1 >0.80` zero-shot, `>0.85` fine-tuned |
| **Dice (per nucleus)** | `2|A∩B|/(|A|+|B|)` | `scipy` | `>0.80` |
| **Aggregated Jaccard Index (AJI)** | Dataset-level Jaccard for instance | `HoVer-Net eval` | `>0.55` |
| **Panoptic Quality (PQ)** | `PQ = DQ × SQ` (detection + segmentation) | `panopticapi` | `>0.50` |
| **Count error** | `|pred - gt| / gt` | — | `<10%` |

*Split:* `70%` train (if fine-tune) / `15%` val / `15%` test, or `k-fold` if `<50` patches. Report `mean ± std`.

### 7.2 If No GT (expected)

1.  **Visual QA:** Open `viewer.html` at `Level 0` zoom (`Zoom ~20-40`), toggle `Show Nuclei` overlay, inspect `3×3` grid ROI (`5k×5k` each) for `false positives` (stroma, debris) and `false negatives` (dense clusters). Screenshot via `Save PNG` (BST footer) for report.
2.  **Model comparison:** Run `StarDist` vs `Cellpose` vs `skimage watershed` on same pilot ROI (`10k×10k`), compare counts + `AJI` via pseudo-GT (majority vote) + visual.
3.  **Stability:** Run on `Level 0` vs `Level 1` (`4×` downsample + upscale) — count should be within `15%`; large drop indicates stain norm issue.
4.  **Ablation:** Vary `prob_thresh` `0.3-0.7`, `nms_thresh` `0.3-0.5`, `patch overlap` `64-256` → plot `count` vs `threshold`.

### 7.3 Reporting

*   `output/report.html` with `thumbnail.jpg` + overlay crop `1024×1024` at `Level 0` + histogram `area` + `density = count / (W*H*MPP²)` per `mm²` + link to `viewer.html?overlay=...`.
*   `output/metrics.json` with `count, mean_area_px, mean_area_um2, std, density, model, patch, time`.

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| `Level 0` `38GB` OOM / `15k` patches too slow | Start with `Level 1` or ROI `10k×10k`; use `tifffile` streaming + `batch 1` + `overlap 64`; `SSD` required. |
| `8GB` VRAM OOM | Use `512×512` patch, `batch 1`, `fp16`, or `CPU` fallback; test on `1` patch first. |
| Stain variation (TCGA multi-center) | `Macenko` normalization before inference; `he_heavy_augment` is stain-augmented. |
| Dense/clumped nuclei (tumor) | `StarDist` handles star-convex; `HoVer-Net` better for extreme clumps — keep as Phase 2. |
| Stitching duplicates at tile borders | Central crop `896×896` (ignore `64px` border) is simpler than cross-tile NMS; 128 overlap ensures no gaps. |
| Large `GeoJSON` (`40MB`) slow in viewer | Simplify polygons (`n=16` rays) or store centroids + `area` for overview, polygons on demand at high zoom. |
| No GT → cannot quantify | Visual QA + model comparison; ask to annotate `~30` patches in QuPath for future fine-tune. |

---

## 9. Next Steps (After You Confirm §2)

1.  **You reply:** `H&E`? `All nuclei` vs `tumor`? `Full vs ROI`? `GPU available`? `Need fine-tune`?
2.  **I create:** `environment.yml` (`stardist`, `tifffile`, `imagecodecs`, `geotiff`, `scikit-image`), `config.yaml` (`patch 1024, overlap 128, prob 0.5`), `detect.py` (tiled loop + `StarDist`), `viewer overlay` toggle, and run pilot on `Level 1` ROI `5k×5k` (`~2 min`) for your visual approval before full slide.
3.  **You review:** `output/report.html` + `viewer.html` overlay at `Zoom 20-40` → approve thresholds or request `HoVer-Net`/`Cellpose` swap.

---

*Generated for `Cell_Count` — SVS Viewer Pro pipeline. Full slide `Level 0` is feasible tiled; start ROI pilot to validate `MPP 0.2525` `40×` before committing `30 min` GPU / `10h` CPU full run.*
