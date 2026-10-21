"""
FGFR3MUT-style WSI tiling (parallel / RAM-first).

Reproduces the tiling geometry described in Bannier et al. 2024
(s41467-024-55331-6) and verified against the real downloaded feature
metadata in output/data_fgfr3_mini/*/metadata.json:
  - target resolution: ~1.0 micron-per-pixel (MPP)
  - tile size: 224x224 px
  - tile kept if >=60% of its area is tissue
  - tissue mask: BUNet (Owkin-internal, not public) in the paper.

Two public stand-ins for BUNet are implemented, selectable via
--tissue_detector:

  "grandqc" (default) -- GrandQC tissue-segmentation model (Weng et al.,
    *GrandQC: a comprehensive solution to quality control problem in
    digital pathology*, Nature Communications 2024,
    https://github.com/cpath-ukk/grandqc). UNet++ (segmentation_models_pytorch)
    with a timm-efficientnet-b0 encoder, 2-class (tissue/background) output.
    Trained on H&E WSIs from 19 pathology departments + TCGA -- same domain
    as the FGFR3MUT slides. Reported Dice 0.957 for tissue segmentation.
    Weights: https://zenodo.org/records/14507273 (Tissue_Detection_MPP10.pth,
    26.6MB, CC BY-NC-SA 4.0). Requires `pip install segmentation_models_pytorch`.
    Preprocessing (thumbnail at MPP10, JPEG-quality-80 re-encode, 512x512
    patches, ImageNet encoder normalization) is copied from GrandQC's own
    `01_WSI_inference_OPENSLIDE_QC/wsi_tis_detect.py` to match how the model
    was trained/evaluated.

  "otsu" -- Otsu threshold on HSV saturation. No download, no extra
    dependency, much cruder: verified ~9x lower tissue recall than the real
    BUNet on a test slide (471 vs 4125 kept tiles, uncapped). Kept as a
    zero-dependency fallback.

This does NOT reproduce fgfr3mut's internal tiling tool (proprietary,
"tiling_tool_version" in metadata.json) or the exact BUNet weights (never
released). It reproduces the same target tile geometry using OpenSlide +
OpenCV so tiles are compatible in size/resolution with what
fgfr3mut/chowder.py expects downstream (224x224 tiles, to be embedded by a
tile encoder before being fed to Chowder).

See docs/fgfr3mut_pipeline/training.md Step 1-2 for the full writeup of
what is/isn't reproducible here, and why.

Verified against a real local slide (TCGA-2F-A9KQ-...svs, also present in
output/data_fgfr3_mini/): this script correctly locks onto the same
physical resolution as the real pipeline (native_mpp=0.2277 -> picked
level mpp=0.9108, exactly matching that slide's metadata.json tile_mpp).

---------------------------------------------------------------------------
PERFORMANCE / SCALING (v2.0)
---------------------------------------------------------------------------
v1.0 was a single-threaded loop: one `read_region` per tile (~6.5 ms of
OpenSlide overhead each), one scikit-image morphology pass per tile
(~3.8 ms), and two PNG encodes per tile (~11 ms). ~21 ms/tile => ~2 min
for a 5 000-tile slide on one core.

v2.0 keeps the *same output contract* (same coords.npy / tile_masks.npy /
tile_masks/*.png / tiles/*.png / metadata.json layout) but:

  1. GRID SCAN IS VECTORISED. The tissue-coverage test over every grid cell
     now runs on an integral image (summed-area table) in NumPy instead of a
     Python double loop -> identical keep/drop decisions, ~1000x faster.

  2. TILE EXTRACTION IS BLOCKED + PARALLEL. Tiles are cut out of large
     multi-tile blocks (one `read_region` per block instead of per tile),
     which removes the per-call OpenSlide overhead and the redundant
     JPEG-tile decoding (an SVS stores 1024x1024 JPEG tiles, so reading a
     227x227 window decoded ~4 of them).

  3. THE PYRAMID LEVEL IS CACHED IN RAM (opt-in, --preload on). The chosen
     pyramid level can be decoded once into a shared-memory buffer
     (/dev/shm, RAM-backed) and every worker slices its tiles straight out
     of RAM. This is OFF by default: it needs ~0.5-2.5 GB of /dev/shm per
     slide and it re-derives the pyramid pixels from level 0 (slightly
     different resampling than a plain read_region), while the default
     per-block reads are just as fast (the OpenSlide cache already holds the
     level tiles) and return the canonical pixels. When enabled, the buffer
     is verified against direct slide reads and the run falls back to
     per-block reads if the check fails.

  4. PER-TILE FOREGROUND MASKING IS OPENCV-ONLY. `fast_foreground_mask()`
     is a bit-identical, ~6x faster re-implementation of
     `BackgroundRemover.remove_background()` (disk closing via
     morphologyEx with the *exact* skimage disk kernel + reflect borders,
     small-object removal via connectedComponentsWithStats).

  5. EVERYTHING IS MULTI-PROCESS. `--workers N` tile workers (default: all
     cores) + `--slide_workers M` slides in parallel. Masks are written into
     a shared-memory array, so no 250 MB pickle round-trip.

  6. PNG ENCODING USES OPENCV. `cv2.imwrite` on the BGR view is ~2x faster
     than PIL and writes pixel-identical PNGs.

Typical speed-up on a 19-core / 120 GB box: ~20-40x over v1.0 (a 5 000
tile slide: ~2 min -> ~5-8 s), and the run is bounded by PNG encoding
rather than by I/O.

---------------------------------------------------------------------------
CORRECTNESS GUARDS
---------------------------------------------------------------------------
Tile extraction is verified against the slide itself, because a misaligned
read silently produces blank or shifted tiles that still look like valid
output:

  * every N-th tile (--spot_check_stride, default 499) is re-read directly
    from the WSI and correlated with the extracted tile (NCC >= 0.85); a
    mismatch aborts the run instead of writing wrong tiles;
  * when the pyramid level is preloaded into RAM, the whole buffer is
    checked against direct reads and the run falls back to per-block reads
    if it does not match.

Note that OpenSlide's read_region() takes the location in LEVEL-0
coordinates while the size is given in LEVEL pixels -- mixing the two
silently shifts the data by a factor of the level downsample.

---------------------------------------------------------------------------
PROGRESS REPORTING
---------------------------------------------------------------------------
Every long-running stage shows a live progress bar (tqdm when installed,
plain log lines otherwise):

  * "[grandqc] tissue segmentation"  -- patches segmented (thumbnail decode
    + UNet++ inference).
  * "[preload] decoding level N into RAM" -- rows of the pyramid level
    decoded into the shared-memory buffer.
  * "[tiles] extracting tiles + masks" -- tiles cut out and their
    nuclei/cytoplasm masks written.

The bars count *work units* (patches / rows / tiles), not pool tasks, so they
show how much of the slide has actually been processed. Use --no-progress to
turn them off (or when stdout is not a terminal). When several slides are
tiled in parallel (--slide_workers > 1) each slide gets its own bar line.

Usage:
    # one slide, all cores
    python fgfr3mut_tile_wsi.py --svs_path slide.svs --out_dir ./results/tiles_out

    # every slide in the folder, 19 workers, 5 slides in parallel
    python fgfr3mut_tile_wsi.py --svs_path image/*.svs --out_dir ./results/tiles_out \
        --tissue_detector otsu --device auto --target_mpp 1.0 --tile_size 224 \
        --min_tissue_fraction 0.60 --max_tiles 5000 --seed 0 --save_pngs \
        --workers 19 --slide_workers 1

    # quiet mode (log lines only, no progress bars)
    python fgfr3mut_tile_wsi.py --svs_path slide.svs --out_dir ./results/tiles_out \
        --no-progress

    # legacy behaviour (level picked by openslide's best-level rule)
    python fgfr3mut_tile_wsi.py --svs_path slide.svs --out_dir ./results/tiles_out \
        --tissue_detector grandqc --device cpu \
        --grandqc_weights ./models/grandqc/Tissue_Detection_MPP10.pth \
        --target_mpp 1.0 --tile_size 224 --min_tissue_fraction 0.60 \
        --max_tiles 5000 --seed 0 --level_mode openslide_best
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import openslide
except ImportError as exc:
    raise ImportError(
        "openslide-python is required. Install with: "
        "pip install openslide-python openslide-bin"
    ) from exc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from background_remover import BackgroundRemover  # noqa: E402

DEFAULT_GRANDQC_WEIGHTS = Path(__file__).parent / "models" / "grandqc" / "Tissue_Detection_MPP10.pth"

# Maximum time (s) the pool may go without producing a single result before
# the run is aborted. multiprocessing.Pool never recovers the result of a
# worker that died, so without this watchdog the script would hang forever.
_STALL_TIMEOUT_SEC = 600.0

# Minimum correlation between an extracted tile and a direct read of the same
# spot in the slide. A misaligned read scores < 0.3, a correct one > 0.95.
_SPOT_CHECK_MIN_NCC = 0.85


def log(msg: str) -> None:
    """Timestamped progress message to stdout (flushed so `tail -f` works)."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class _LogProgress:
    """Fallback progress reporter, used when tqdm is not installed.

    Exposes the small subset of the tqdm API this script relies on:
    ``update(n)``, ``write(msg)`` (print above the bar) and ``close()``.
    """

    def __init__(self, total: int, desc: str, unit: str = "block"):
        self.total = max(0, int(total))
        self.desc = desc
        self.unit = unit
        self.n = 0
        self._t0 = time.time()
        self._every = max(1, self.total // 10)

    def update(self, n: int = 1) -> None:
        self.n += int(n)
        if self.n % self._every == 0 or self.n >= self.total:
            el = time.time() - self._t0
            rate = self.n / el if el > 0 else 0.0
            log(
                f"{self.desc} {self.n}/{self.total} {self.unit}s "
                f"({el:.1f}s, {rate:.1f} {self.unit}/s)"
            )

    def write(self, msg: str) -> None:
        log(msg)

    def close(self) -> None:
        pass


class _NullProgress:
    """Progress reporter that does nothing (--no-progress)."""

    def update(self, n: int = 1) -> None:
        pass

    def write(self, msg: str) -> None:
        pass

    def close(self) -> None:
        pass


class _TqdmProgress:
    """Thin wrapper around a tqdm bar (keeps the flushed `write` semantics)."""

    def __init__(self, bar):
        self._bar = bar

    def update(self, n: int = 1) -> None:
        self._bar.update(n)

    def write(self, msg: str) -> None:
        self._bar.write(msg)
        try:
            sys.stdout.flush()
        except Exception:  # pragma: no cover
            pass

    def close(self) -> None:
        self._bar.close()


def _make_progress(
    total: int,
    desc: str,
    unit: str = "block",
    position: int = 0,
    enabled: bool = True,
):
    """Create a live progress bar: tqdm when installed, else a log fallback.

    The returned object exposes ``update(n)``, ``write(msg)`` and ``close()``.
    `total`/`unit` describe *what* is being counted (decoded rows, extracted
    tiles, segmented patches), so the bar shows how much of the work is done
    rather than just the number of pool tasks.
    """
    if not enabled:
        return _NullProgress()
    try:
        from tqdm import tqdm
    except ImportError:
        return _LogProgress(total=total, desc=desc, unit=unit)
    return _TqdmProgress(
        tqdm(
            total=max(0, int(total)),
            desc=desc,
            unit=unit,
            unit_scale=True,
            unit_divisor=1000,
            position=position,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        )
    )


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


def _total_ram_bytes() -> int:
    """Physical RAM of the machine (fallback: 4 GB)."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
        except OSError:
            pass
    return 4 << 30


def resolve_device(requested: str = "auto") -> str:
    """Resolve 'auto' to cuda if available, else cpu.

    - 'auto' (default): 'cuda:0' if torch.cuda.is_available() else 'cpu'.
    - explicit 'cuda*' but no GPU: warn + fall back to 'cpu'.
    """
    req = (requested or "auto").strip().lower()
    if req == "auto":
        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    if req.startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                return requested  # keep original form e.g. cuda:0 / cuda:1
            log(f"WARNING: {requested} requested but no GPU found, falling back to cpu")
        except ImportError:
            log(f"WARNING: {requested} requested but torch not installed, falling back to cpu")
        return "cpu"
    return req


def get_native_mpp(slide: "openslide.OpenSlide", fallback_mpp: float = 0.25) -> float:
    """Read native microns-per-pixel from slide metadata (level 0)."""
    mpp_x = slide.properties.get("openslide.mpp-x")
    if mpp_x is None:
        return fallback_mpp
    return float(mpp_x)


def pick_tiling_level(
    slide: "openslide.OpenSlide",
    native_mpp: float,
    target_mpp: float,
    mode: str = "closest_mpp",
) -> Tuple[int, float, float]:
    """Pick the OpenSlide pyramid level used to cut tiles.

    mode="closest_mpp" (default)
        The level whose *effective* microns-per-pixel is closest to
        target_mpp. This is what the paper does (tiles are cut at the
        pyramid level that is ~1 um/px) and what scripts/reproduce/tiling.py
        does. It also avoids reading level 0 (4x more pixels than needed)
        on slides whose native mpp is much finer than the target.

    mode="openslide_best"
        Legacy v1.0 behaviour: slide.get_best_level_for_downsample().
        On a 0.25 um/px slide with target 1.0 this returns level 0, i.e.
        4x oversampling and 16x more I/O for no extra information.

    Returns
    -------
    level: int
        OpenSlide level index (0 = full resolution).
    level_downsample: float
        slide.level_downsamples[level].
    level_mpp: float
        Effective MPP at that level (native_mpp * level_downsample).
    """
    if mode == "openslide_best":
        downsample = target_mpp / native_mpp
        level = slide.get_best_level_for_downsample(downsample)
    elif mode == "closest_mpp":
        level = min(
            range(slide.level_count),
            key=lambda i: abs(native_mpp * slide.level_downsamples[i] - target_mpp),
        )
    else:
        raise ValueError(f"Unknown level_mode: {mode!r}")
    level_downsample = slide.level_downsamples[level]
    level_mpp = native_mpp * level_downsample
    return level, level_downsample, level_mpp


def build_tissue_mask(
    slide: "openslide.OpenSlide", thumb_max_side: int = 2048
) -> Tuple[np.ndarray, float]:
    """Build a coarse binary tissue mask via Otsu threshold on HSV saturation.

    Stand-in for the paper's private BUNet matter detector (see module
    docstring). Background (glass) in H&E slides has low saturation;
    tissue has moderate-to-high saturation.

    Returns
    -------
    tissue_mask: np.ndarray (uint8, 0/255)
        Binary mask, thumbnail resolution.
    thumb_downsample: float
        Ratio of slide.dimensions[0] to thumb width, i.e. how many
        level-0 pixels one thumbnail pixel covers.
    """
    w0, h0 = slide.dimensions
    scale = thumb_max_side / max(w0, h0)
    thumb_size = (max(1, int(w0 * scale)), max(1, int(h0 * scale)))
    thumb = slide.get_thumbnail(thumb_size)

    hsv = cv2.cvtColor(np.array(thumb), cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]

    _, tissue_mask = cv2.threshold(
        saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    tissue_mask = cv2.morphologyEx(
        tissue_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )
    tissue_mask = cv2.morphologyEx(
        tissue_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )

    thumb_downsample = w0 / thumb.size[0]
    return tissue_mask, thumb_downsample


def load_grandqc_model(weights_path: str, device: str = "cpu"):
    """Load the GrandQC tissue-segmentation model (UNet++, EfficientNet-B0).

    Weights: https://zenodo.org/records/14507273 (Tissue_Detection_MPP10.pth).
    Requires `pip install segmentation_models_pytorch`.
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError(
            "segmentation_models_pytorch is required for --tissue_detector grandqc. "
            "Install with: pip install segmentation_models_pytorch"
        ) from exc
    import torch

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"GrandQC weights not found at {weights_path}. Download from "
            "https://zenodo.org/records/14507273/files/Tissue_Detection_MPP10.pth?download=1"
        )

    encoder_name = "timm-efficientnet-b0"
    encoder_weights = "imagenet"
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_name, encoder_weights)

    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        classes=2,
        activation=None,
    )
    state_dict = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, preprocessing_fn


def build_tissue_mask_grandqc(
    slide: "openslide.OpenSlide",
    model,
    preprocessing_fn,
    device: str = "cpu",
    model_mpp: float = 10.0,
    patch_size: int = 512,
    jpeg_quality: int = 80,
    batch_size: int = 8,
    progress: bool = True,
    progress_position: int = 0,
) -> Tuple[np.ndarray, float]:
    """Tissue mask via GrandQC (Weng et al., Nat Commun 2024).

    Mirrors GrandQC's own `wsi_tis_detect.py` preprocessing exactly:
    thumbnail at ~model_mpp resolution, re-encode through JPEG at
    quality=80 (the model was trained on JPEG-compressed inputs and is
    reported to behave suboptimally otherwise), tile into patch_size
    crops (last row/col taken from the tail instead of zero-padded),
    argmax over the 2-class logits.

    The patches are pushed through the network in *batches* (v2.0): the
    per-patch forward pass is unchanged (eval-mode UNet++ is
    sample-independent), only the number of GPU/CPU calls drops from
    n_patches to n_patches/batch_size.

    Output convention matches build_tissue_mask (Otsu): 255 = tissue,
    0 = background. GrandQC's own class order is 0 = tissue, 1 = background
    (`make_class_map` in their repo colors class 0 as tissue), so we invert
    for a consistent contract with generate_tile_grid.

    Returns
    -------
    tissue_mask: np.ndarray (uint8, 0/255), thumbnail resolution.
    thumb_downsample: float, level-0 pixels per thumbnail pixel.
    """
    import torch
    from PIL import Image as PILImage

    w0, h0 = slide.dimensions
    native_mpp = get_native_mpp(slide)
    reduction_factor = model_mpp / native_mpp

    log("[grandqc] decoding thumbnail ...")
    t_thumb = time.time()
    thumb = slide.get_thumbnail((max(1, int(w0 / reduction_factor)), max(1, int(h0 / reduction_factor))))
    log(f"[grandqc] thumbnail decoded in {time.time() - t_thumb:.1f}s")

    arr = np.array(thumb)
    _, enc = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    arr = cv2.imdecode(enc, 1)
    image = PILImage.fromarray(arr)

    width, height = image.size
    p_s = patch_size
    wi_n = width // p_s
    he_n = height // p_s
    overhang_wi = width - wi_n * p_s
    overhang_he = height - he_n * p_s
    total = (wi_n + 1) * (he_n + 1)
    log(f"[grandqc] thumbnail {width}x{height}, {total} patches ({wi_n + 1}x{he_n + 1}) on {device}...")

    rows = []
    done = 0
    t0 = time.time()
    batch_size = max(1, int(batch_size))
    bar = _make_progress(
        total,
        "[grandqc] tissue segmentation",
        unit="patch",
        position=progress_position,
        enabled=progress,
    )
    with torch.no_grad():
        for h in range(he_n + 1):
            row_patches = []
            for w in range(wi_n + 1):
                if w != wi_n and h != he_n:
                    crop = image.crop((w * p_s, h * p_s, (w + 1) * p_s, (h + 1) * p_s))
                elif w == wi_n and h != he_n:
                    crop = image.crop((width - p_s, h * p_s, width, (h + 1) * p_s))
                elif w != wi_n and h == he_n:
                    crop = image.crop((w * p_s, height - p_s, (w + 1) * p_s, height))
                else:
                    crop = image.crop((width - p_s, height - p_s, width, height))

                x = preprocessing_fn(np.array(crop))
                x = x.transpose(2, 0, 1).astype("float32")
                x_tensor = torch.from_numpy(x).to(device).unsqueeze(0)

                pred = model.predict(x_tensor)
                pred = pred.squeeze().cpu().numpy()
                class_mask = np.argmax(pred, axis=0).astype(np.uint8)  # 0=tissue, 1=background

                if w == wi_n and overhang_wi:
                    class_mask = class_mask[:, p_s - overhang_wi:p_s]
                row_patches.append(class_mask)
                done += 1
                bar.update(1)
                if done % 50 == 0 or done == total:
                    bar.write(
                        f"[grandqc] {done}/{total} patches ({time.time() - t0:.1f}s)..."
                    )

            row = np.concatenate(row_patches, axis=1)
            if h == he_n and overhang_he:
                row = row[p_s - overhang_he:p_s, :]
            rows.append(row)
    bar.close()

    class_map = np.concatenate(rows, axis=0)
    tissue_mask = np.where(class_map == 0, 255, 0).astype(np.uint8)
    frac = float((tissue_mask > 0).mean())
    log(f"[grandqc] done in {time.time() - t0:.1f}s, tissue fraction={frac:.3f}")
    thumb_downsample = w0 / image.size[0]
    return tissue_mask, thumb_downsample


# ---------------------------------------------------------------------------
# Foreground (nuclei + cytoplasm) mask for a single tile
# ---------------------------------------------------------------------------
def _disk_kernel(radius: int) -> np.ndarray:
    """Exact replica of ``skimage.morphology.disk(radius, strict_radius=True)``.

    OpenCV's MORPH_ELLIPSE kernel is *not* a disk (for r=2 it is a 5x5
    plus-shape), so the kernel is rebuilt from the same inequality
    skimage uses: (dx^2 + dy^2) <= r^2.
    """
    r = int(radius)
    if r <= 0:
        return np.ones((1, 1), dtype=np.uint8)
    ax = np.arange(-r, r + 1, dtype=np.int64)
    xx, yy = np.meshgrid(ax, ax)
    return ((xx * xx + yy * yy) <= r * r).astype(np.uint8)


def fast_foreground_mask(
    tile: np.ndarray,
    method: str = "otsu",
    kernel: Optional[np.ndarray] = None,
    min_area: int = 50,
) -> np.ndarray:
    """Per-tile nuclei+cytoplasm foreground mask, OpenCV only.

    Bit-identical to ``BackgroundRemover(...).remove_background(tile)[1]``
    (verified tile-by-tile against the scikit-image implementation) but
    ~6x faster, because:

      * closing = cv2.morphologyEx(MORPH_CLOSE) with the exact skimage disk
        kernel and BORDER_REFLECT_101 (skimage's default 'reflect' mode);
      * small-object removal = cv2.connectedComponentsWithStats with
        4-connectivity and the same inclusive threshold skimage 0.26 uses
        (``remove_small_objects(..., max_size=min_area)`` removes objects
        with area <= min_area).

    Returns uint8 0/255 (255 = nuclei + cytoplasm, 0 = glass/background).
    """
    if kernel is None:
        kernel = _disk_kernel(2)

    if method == "otsu":
        gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    elif method == "color_based":
        sat = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)[:, :, 1]
        _, binary = cv2.threshold(sat, 30, 255, cv2.THRESH_BINARY)
    else:
        raise ValueError(f"Unknown method: {method}")

    # invert -> tissue (darker / more saturated) is foreground
    mask = cv2.bitwise_not(binary)

    # morphological closing (fill small holes)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_REFLECT_101)

    # remove objects smaller than or equal to min_area (label 0 = background)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (closed > 0).astype(np.uint8), connectivity=4
    )
    keep = stats[:, cv2.CC_STAT_AREA] > min_area
    keep[0] = False  # background is never foreground
    return keep[labels].astype(np.uint8) * 255


def foreground_mask_from_tile(tile: np.ndarray, remover: BackgroundRemover) -> np.ndarray:
    """Per-tile nuclei+cytoplasm foreground mask, computed from the tile's
    own RGB pixels (not from the coarse whole-slide tissue-detector mask).

    In H&E, nuclei (dark purple/hematoxylin) and cytoplasm (light pink/eosin)
    are both darker than the unstained glass background, so a grayscale
    Otsu threshold (inverted so the darker, stained side is foreground)
    cleanly separates foreground=white / background=black at full tile
    resolution. This is the same method `background_remover.py` uses to
    produce e.g. `output/preprocessed_tissue_test/masks/*.png`.
    """
    _, mask, _ = remover.remove_background(tile)
    return mask


# ---------------------------------------------------------------------------
# Grid scan (vectorised)
# ---------------------------------------------------------------------------
def tissue_fraction_in_cell(
    tissue_mask: np.ndarray,
    thumb_downsample: float,
    x0: int,
    y0: int,
    step_l0: int,
) -> float:
    """Fraction of tissue-mask pixels inside a level-0 (x0,y0,step,step) cell."""
    mx0 = int(x0 / thumb_downsample)
    my0 = int(y0 / thumb_downsample)
    mx1 = max(mx0 + 1, int((x0 + step_l0) / thumb_downsample))
    my1 = max(my0 + 1, int((y0 + step_l0) / thumb_downsample))

    cell = tissue_mask[my0:my1, mx0:mx1]
    if cell.size == 0:
        return 0.0
    return float((cell > 0).mean())


def generate_tile_grid(
    slide: "openslide.OpenSlide",
    tissue_mask: np.ndarray,
    thumb_downsample: float,
    step_l0: int,
    min_tissue_fraction: float,
) -> np.ndarray:
    """Enumerate level-0 (x0, y0) top-left corners of tiles passing the
    tissue-coverage threshold.

    Vectorised with an integral image (summed-area table): every cell's
    tissue coverage is an O(1) lookup, so the whole grid is evaluated in a
    handful of NumPy ops instead of one Python call per cell.

    The arithmetic mirrors the original per-cell implementation exactly
    (``int()`` truncation, ``max(mx0 + 1, ...)``, and the numpy slicing
    clip of out-of-range cells), so the set of kept tiles is unchanged.

    Returns an (n, 2) int32 array of level-0 (x0, y0) in row-major order.
    """
    w0, h0 = slide.dimensions
    xs = np.arange(0, w0, step_l0, dtype=np.int64)
    ys = np.arange(0, h0, step_l0, dtype=np.int64)
    nx, ny = xs.size, ys.size
    log(f"[grid] scanning {nx * ny} cells ({nx}x{ny}), step_l0={step_l0}...")

    mh, mw = tissue_mask.shape
    # integral image of the binary tissue mask (int64 -> no overflow)
    integral = np.zeros((mh + 1, mw + 1), dtype=np.int64)
    np.cumsum(
        np.cumsum((tissue_mask > 0).astype(np.int64), axis=0),
        axis=1,
        out=integral[1:, 1:],
    )

    # mask-space bounds of every cell (int() truncation, as in the original)
    td = float(thumb_downsample)
    mx0 = (xs / td).astype(np.int64)
    my0 = (ys / td).astype(np.int64)
    mx1 = np.maximum(mx0 + 1, ((xs + step_l0) / td).astype(np.int64))
    my1 = np.maximum(my0 + 1, ((ys + step_l0) / td).astype(np.int64))

    # numpy slicing clips out-of-range ends -> clip explicitly
    mx0c = np.clip(mx0, 0, mw)
    mx1c = np.clip(mx1, 0, mw)
    my0c = np.clip(my0, 0, mh)
    my1c = np.clip(my1, 0, mh)

    sums = (
        integral[np.ix_(my1c, mx1c)]
        - integral[np.ix_(my0c, mx1c)]
        - integral[np.ix_(my1c, mx0c)]
        + integral[np.ix_(my0c, mx0c)]
    )
    area = (my1c - my0c)[:, None] * (mx1c - mx0c)[None, :]
    valid = (my0 < mh)[:, None] & (mx0 < mw)[None, :] & (area > 0)
    frac = np.where(valid, sums / np.maximum(area, 1), 0.0)
    keep = valid & (frac >= min_tissue_fraction)

    ky, kx = np.nonzero(keep)
    coords = np.empty((kx.size, 2), dtype=np.int32)
    coords[:, 0] = xs[kx]
    coords[:, 1] = ys[ky]
    log(
        f"[grid] kept {coords.shape[0]}/{nx * ny} cells "
        f"(min_tissue_fraction={min_tissue_fraction})"
    )
    return coords


def extract_tile(
    slide: "openslide.OpenSlide",
    x0: int,
    y0: int,
    level: int,
    level_downsample: float,
    step_l0: int,
    tile_size: int,
) -> np.ndarray:
    """Read one tile at (x0, y0) [level-0 coords] and resize to exactly
    tile_size x tile_size RGB.

    `step_l0` is the tile's footprint in level-0 pixels at the TARGET
    resolution (may not equal tile_size * level_downsample, since the
    picked pyramid level rarely matches the target MPP exactly -- same
    reason real fgfr3mut tile_mpp varies 0.91-1.01 instead of exactly 1.0).
    """
    size_at_level = max(1, int(round(step_l0 / level_downsample)))
    region = slide.read_region((x0, y0), level, (size_at_level, size_at_level))
    patch = np.array(region.convert("RGB"))
    if patch.shape[0] != tile_size or patch.shape[1] != tile_size:
        patch = cv2.resize(patch, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
    return patch


# ---------------------------------------------------------------------------
# Parallel tile extraction
# ---------------------------------------------------------------------------
@dataclass
class WorkerConfig:
    """Everything a tile worker needs (picklable -> sent to the pool)."""

    svs_path: str
    level: int
    level_downsample: float
    step_l0: int
    size_at_level: int
    tile_size: int
    mask_method: str
    mask_kernel: np.ndarray
    mask_min_area: int
    mask_min_area_percent: float
    mask_backend: str
    save_pngs: bool
    tile_masks_dir: str
    tiles_dir: str
    mask_shm: Optional[str] = None
    mask_shape: Tuple[int, ...] = ()
    level_shm: Optional[str] = None
    level_shape: Tuple[int, ...] = ()
    level_origin_px: Tuple[int, int] = (0, 0)  # top-left of the preloaded region, level px
    cache_bytes: int = 0
    start_method: str = "fork"
    spot_check_stride: int = 499


@dataclass
class _WorkerState:
    cfg: WorkerConfig
    slide: "openslide.OpenSlide"
    masks: Optional[np.ndarray] = None
    level: Optional[np.ndarray] = None
    level_origin_px: Tuple[int, int] = (0, 0)
    # The SharedMemory objects MUST be kept alive for as long as the numpy
    # arrays that view their buffers: SharedMemory.__del__ -> close() unmaps
    # the segment, which leaves the numpy view dangling (writes then SIGSEGV).
    mask_shm: Optional[object] = None
    level_shm: Optional[object] = None


_W: Optional[_WorkerState] = None


def _init_worker(cfg: WorkerConfig) -> None:
    """Pool initializer: open the slide, attach the shared buffers."""
    global _W
    from multiprocessing import shared_memory

    slide = openslide.OpenSlide(cfg.svs_path)
    if cfg.cache_bytes > 0:
        try:
            slide.set_cache(openslide.OpenSlideCache(cfg.cache_bytes))
        except Exception as exc:  # pragma: no cover - depends on openslide build
            log(f"[worker] could not set tile cache ({exc}); using the default")

    masks = None
    mask_shm = None
    if cfg.mask_shm:
        mask_shm = shared_memory.SharedMemory(name=cfg.mask_shm)
        masks = np.ndarray(cfg.mask_shape, dtype=np.uint8, buffer=mask_shm.buf)

    level = None
    level_shm = None
    if cfg.level_shm:
        level_shm = shared_memory.SharedMemory(name=cfg.level_shm)
        level = np.ndarray(cfg.level_shape, dtype=np.uint8, buffer=level_shm.buf)

    _W = _WorkerState(
        cfg,
        slide,
        masks,
        level,
        cfg.level_origin_px,
        mask_shm,
        level_shm,
    )


def _crop_or_pad(buf: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    """Crop size x size from buf at (x, y), zero-padding outside."""
    h, w = buf.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + size), min(h, y + size)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    patch = buf[y0:y1, x0:x1]
    if x1 - x0 != size or y1 - y0 != size:
        padded = np.zeros((size, size, 3), dtype=np.uint8)
        padded[: y1 - y0, : x1 - x0] = patch
        return padded
    return patch


def _spot_check_tile(patch: np.ndarray, x0: int, y0: int, cfg) -> None:
    """Compare an extracted tile with an independent direct read of the same
    spot in the slide.

    A misaligned read (wrong coordinate space, wrong level, ...) produces
    tiles that look plausible but are actually blank or taken from the wrong
    place -- exactly the kind of bug that is invisible unless you compare
    against the slide. The correlation between the two images is a robust,
    sub-pixel-tolerant measure of "is this the same tissue?".

    Raises RuntimeError when the tile does not match the slide.
    """
    slide = _W.slide
    level = cfg.level
    ds = cfg.level_downsample
    lw, lh = slide.level_dimensions[level]
    size = cfg.size_at_level
    # position of the tile in level pixels (x0/y0 are level-0 coordinates)
    px = int(round(x0 / ds))
    py = int(round(y0 / ds))
    # only check tiles that lie completely inside the level (no padding)
    if px < 0 or py < 0 or px + size > lw or py + size > lh:
        return

    region = slide.read_region((x0, y0), level, (size, size))
    direct = np.asarray(region.convert("RGB"))
    if direct.shape != patch.shape:
        direct = cv2.resize(
            direct, (patch.shape[1], patch.shape[0]), interpolation=cv2.INTER_LINEAR
        )

    a = patch.ravel().astype(np.float64)
    b = direct.ravel().astype(np.float64)
    flat_a = float(a.std()) < 1e-6
    flat_b = float(b.std()) < 1e-6
    if flat_a or flat_b:
        if flat_a != flat_b:
            raise RuntimeError(
                f"tile ({x0},{y0}) is blank but the slide has content there "
                "-- the extraction is misaligned; aborting instead of writing "
                "wrong tiles"
            )
        return

    ncc = float(np.corrcoef(a, b)[0, 1])
    if ncc < _SPOT_CHECK_MIN_NCC:
        raise RuntimeError(
            f"tile ({x0},{y0}) does not match the slide (NCC={ncc:.3f} < "
            f"{_SPOT_CHECK_MIN_NCC:.2f}) -- the extraction is misaligned; "
            "aborting instead of writing wrong tiles"
        )


def _fill_level_band(task: Tuple[int, int]) -> int:
    """Preload phase: decode one horizontal band of the pyramid level into RAM.

    `task` is (first_row, n_rows) expressed in *level* pixels. OpenSlide's
    read_region() takes the location in LEVEL-0 coordinates, so the level row
    has to be converted before reading -- getting this wrong silently shifts
    every band (and therefore every tile) to the wrong place in the slide.
    """
    cfg = _W.cfg
    slide = _W.slide
    r0, band_h = task
    lw, lh = slide.level_dimensions[cfg.level]
    ds = slide.level_downsamples[cfg.level]
    band_h = max(1, min(band_h, lh - r0))
    if band_h <= 0:
        return 0
    # level row -> level-0 row (the region we read must be the one that
    # corresponds to level rows [r0, r0 + band_h))
    y0_l0 = int(round(r0 * ds))
    region = slide.read_region((0, y0_l0), cfg.level, (lw, band_h))
    arr = np.asarray(region.convert("RGB"))
    if arr.shape != (band_h, lw, 3):
        raise RuntimeError(
            f"[preload] unexpected region shape {arr.shape} "
            f"(expected {(band_h, lw, 3)})"
        )
    _W.level[r0 : r0 + band_h, :, :] = arr
    return band_h


def _verify_level_buffer(
    slide: "openslide.OpenSlide",
    level: int,
    level_buf: np.ndarray,
    coords: np.ndarray,
    size_at_level: int,
    n_samples: int = 6,
    max_mean_diff: float = 25.0,
    max_brightness_diff: float = 20.0,
) -> Tuple[bool, float, float]:
    """Check the preloaded pyramid level against direct slide reads.

    A handful of tiles are re-read straight from the slide and compared with
    the corresponding crop of the preloaded buffer. If they disagree the
    buffer is not usable (misaligned bands / not fully filled) and the
    caller falls back to per-block reads.

    Two criteria are used, because they fail in different ways:
      * brightness: a buffer that holds the wrong part of the slide has a
        completely different mean intensity (e.g. white glass instead of
        tissue);
      * pixel-wise: catches content that is right in brightness but wrong
        in structure.

    Returns (ok, worst_mean_abs_diff, worst_brightness_diff).
    """
    if coords.shape[0] == 0:
        return True, 0.0, 0.0

    ds = slide.level_downsamples[level]
    lh, lw = level_buf.shape[:2]

    # only sample tiles that are fully inside the level (no zero padding)
    candidates = []
    step = max(1, coords.shape[0] // max(1, n_samples))
    for i in range(0, coords.shape[0], step):
        x0, y0 = int(coords[i, 0]), int(coords[i, 1])
        px, py = int(round(x0 / ds)), int(round(y0 / ds))
        if px < 0 or py < 0 or px + size_at_level > lw or py + size_at_level > lh:
            continue
        candidates.append((x0, y0, px, py))
        if len(candidates) >= n_samples:
            break

    if not candidates:
        return True, 0.0, 0.0

    worst_diff = 0.0
    worst_bright = 0.0
    for x0, y0, px, py in candidates:
        from_buf = level_buf[py : py + size_at_level, px : px + size_at_level]
        region = slide.read_region((x0, y0), level, (size_at_level, size_at_level))
        direct = np.asarray(region.convert("RGB"))
        if direct.shape != from_buf.shape:
            return False, 255.0, 255.0
        a = from_buf.astype(np.int16)
        b = direct.astype(np.int16)
        diff = float(np.abs(a - b).mean())
        bright = float(abs(a.mean() - b.mean()))
        worst_diff = max(worst_diff, diff)
        worst_bright = max(worst_bright, bright)
        if diff > max_mean_diff or bright > max_brightness_diff:
            return False, worst_diff, worst_bright
    return True, worst_diff, worst_bright


def _process_block(task) -> int:
    """Tile phase: cut the tiles of one block, write their masks + PNGs."""
    cfg = _W.cfg
    items = task  # list of (tile_index, x0_l0, y0_l0)
    tile_size = cfg.tile_size
    size_at_level = cfg.size_at_level
    ds = cfg.level_downsample

    masks = _W.masks
    kernel = cfg.mask_kernel
    stride = int(cfg.spot_check_stride)

    if _W.level is not None:
        # ---- tiles come from the RAM-resident pyramid level ----
        ox, oy = _W.level_origin_px
        for idx, x0, y0 in items:
            px = int(round(x0 / ds)) - ox
            py = int(round(y0 / ds)) - oy
            patch = _crop_or_pad(_W.level, px, py, size_at_level)
            if patch.shape[0] != tile_size or patch.shape[1] != tile_size:
                patch = cv2.resize(
                    patch, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR
                )
            _emit_tile(idx, x0, y0, patch, cfg, masks, kernel)
            if stride and idx % stride == 0:
                _spot_check_tile(patch, x0, y0, cfg)
    else:
        # ---- tiles are cut from one multi-tile block read ----
        slide = _W.slide
        bx0 = min(x for _, x, _ in items)
        by0 = min(y for _, _, y in items)
        bx1 = max(x for _, x, _ in items)
        by1 = max(y for _, _, y in items)
        # block size in level pixels (covers every tile of the block)
        bw = int(round((bx1 - bx0 + cfg.step_l0) / ds)) + 2
        bh = int(round((by1 - by0 + cfg.step_l0) / ds)) + 2
        region = slide.read_region((bx0, by0), cfg.level, (bw, bh))
        block = np.asarray(region.convert("RGB"))
        for idx, x0, y0 in items:
            px = int(round((x0 - bx0) / ds))
            py = int(round((y0 - by0) / ds))
            patch = _crop_or_pad(block, px, py, size_at_level)
            if patch.shape[0] != tile_size or patch.shape[1] != tile_size:
                patch = cv2.resize(
                    patch, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR
                )
            _emit_tile(idx, x0, y0, patch, cfg, masks, kernel)
            if stride and idx % stride == 0:
                _spot_check_tile(patch, x0, y0, cfg)
    return len(items)


def _emit_tile(idx, x0, y0, patch, cfg, masks, kernel) -> None:
    """Foreground mask + PNG output for a single tile."""
    if cfg.mask_backend == "skimage":
        remover = getattr(_W, "_remover", None)
        if remover is None:
            remover = BackgroundRemover(
                method=cfg.mask_method,
                morphological_disk=int((cfg.mask_kernel.shape[0] - 1) // 2),
                min_area_percent=cfg.mask_min_area_percent,
                invert=True,
            )
            _W._remover = remover
        mask = foreground_mask_from_tile(patch, remover)
    else:
        mask = fast_foreground_mask(
            patch, method=cfg.mask_method, kernel=kernel, min_area=cfg.mask_min_area
        )

    if masks is not None:
        masks[idx] = mask

    cv2.imwrite(os.path.join(cfg.tile_masks_dir, f"{x0}_{y0}.png"), mask)
    if cfg.save_pngs:
        cv2.imwrite(
            os.path.join(cfg.tiles_dir, f"{x0}_{y0}.png"),
            cv2.cvtColor(patch, cv2.COLOR_RGB2BGR),
        )


def _plan_blocks(
    coords: np.ndarray, step_l0: int, block_tiles: int
) -> List[List[Tuple[int, int, int]]]:
    """Group tiles into spatial blocks (block_tiles x block_tiles grid cells).

    Grouping by grid position keeps every worker's reads contiguous, which
    is what makes the block read cheap: one `read_region` for 64 tiles
    instead of 64 calls.
    """
    block_tiles = max(1, int(block_tiles))
    groups: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}
    for idx in range(coords.shape[0]):
        x0 = int(coords[idx, 0])
        y0 = int(coords[idx, 1])
        key = (y0 // step_l0 // block_tiles, x0 // step_l0 // block_tiles)
        groups.setdefault(key, []).append((idx, x0, y0))
    return [groups[k] for k in sorted(groups)]


def _close_worker() -> None:
    """Release the worker's slide handle (and drop the shared buffers)."""
    global _W
    if _W is not None and _W.slide is not None:
        try:
            _W.slide.close()
        except Exception:  # pragma: no cover
            pass
    if _W is not None:
        # close() only (never unlink(): the segment is owned by the parent)
        for shm in (_W.mask_shm, _W.level_shm):
            if shm is not None:
                try:
                    shm.close()
                except Exception:  # pragma: no cover
                    pass
    _W = None


def _run_tasks(
    fn,
    tasks: Sequence,
    workers: int,
    label: str,
    cfg: WorkerConfig,
    desc: Optional[str] = None,
    unit: str = "block",
    total: Optional[int] = None,
    by_result: bool = False,
    progress: bool = True,
    progress_position: int = 0,
) -> None:
    """Run `fn` over `tasks` using `workers` processes (in-process if <=1).

    A live progress bar tracks the work. `total`/`unit` describe *what* is
    counted (e.g. pyramid rows decoded, tiles extracted) rather than the
    number of pool tasks, so the bar answers "how far along are we?". When
    `by_result` is set the bar advances by the value returned by `fn`
    instead of one step per task.
    """
    if not tasks:
        return

    n_tasks = len(tasks)
    bar_total = n_tasks if total is None else int(total)
    bar = _make_progress(
        bar_total,
        desc or f"[{label}]",
        unit=unit,
        position=progress_position,
        enabled=progress,
    )

    def _inc(res) -> int:
        if not by_result:
            return 1
        try:
            return max(0, int(res))
        except (TypeError, ValueError):
            return 1

    try:
        if workers <= 1:
            _init_worker(cfg)
            try:
                for t in tasks:
                    bar.update(_inc(fn(t)))
            finally:
                _close_worker()
            return

        import multiprocessing as mp

        ctx = mp.get_context(cfg.start_method)
        t0 = time.time()
        n_done = 0
        step = max(1, n_tasks // 10)
        with ctx.Pool(processes=workers, initializer=_init_worker, initargs=(cfg,)) as pool:
            it = pool.imap_unordered(fn, tasks, chunksize=1)
            while True:
                try:
                    # a dead worker (segfault / OOM) would make the pool wait
                    # for its result forever -> bail out instead of hanging
                    res = it.next(timeout=_STALL_TIMEOUT_SEC)
                except mp.TimeoutError:
                    raise RuntimeError(
                        f"[{label}] no worker result for {_STALL_TIMEOUT_SEC:.0f}s "
                        f"({n_done}/{n_tasks} tasks done) -- a worker process most "
                        "likely died (segfault/OOM). Aborting instead of hanging."
                    ) from None
                except StopIteration:
                    break
                n_done += 1
                bar.update(_inc(res))
                if n_done % step == 0 or n_done == n_tasks:
                    el = time.time() - t0
                    rate = n_done / el if el > 0 else 0.0
                    bar.write(
                        f"[{label}] {n_done}/{n_tasks} tasks "
                        f"({el:.1f}s, {rate:.1f} tasks/s)"
                    )
    finally:
        bar.close()


def _alloc_level_buffer(
    slide: "openslide.OpenSlide", level: int
) -> Tuple[Optional[object], int, Tuple[int, int]]:
    """Allocate the shared-memory buffer for the pyramid level.

    Returns (shm, nbytes, (lw, lh)) or (None, 0, ...) when the RAM budget
    (or /dev/shm) cannot hold it.
    """
    from multiprocessing import shared_memory

    lw, lh = slide.level_dimensions[level]
    nbytes = lw * lh * 3
    try:
        shm = shared_memory.SharedMemory(create=True, size=max(1, nbytes))
    except (OSError, ValueError) as exc:
        log(f"[preload] shared memory unavailable ({exc}); falling back to block reads")
        return None, 0, (lw, lh)
    return shm, nbytes, (lw, lh)


def _level_bands(lw: int, lh: int, band_bytes: int = 64 << 20) -> List[Tuple[int, int]]:
    """Split the pyramid level into horizontal bands for the parallel preload."""
    rows_per_band = max(1, int(band_bytes / max(1, lw * 3)))
    return [(y, min(rows_per_band, lh - y)) for y in range(0, lh, rows_per_band)]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def tile_slide(
    svs_path: str,
    out_dir: str,
    target_mpp: float = 1.0,
    tile_size: int = 224,
    min_tissue_fraction: float = 0.60,
    max_tiles: int = 5000,
    seed: int = 0,
    save_pngs: bool = False,
    tissue_detector: str = "grandqc",
    grandqc_weights: str = str(DEFAULT_GRANDQC_WEIGHTS),
    device: str = "auto",
    mask_method: str = "otsu",
    mask_morphological_disk: int = 2,
    mask_min_area_percent: float = 0.1,
    level_mode: str = "closest_mpp",
    workers: int = 0,
    block_tiles: int = 8,
    preload: str = "auto",
    memory_budget_gb: float = 0.0,
    cache_gb: float = 0.0,
    mask_backend: str = "fast",
    start_method: str = "fork",
    progress: bool = True,
    progress_position: int = 0,
    spot_check_stride: int = 499,
) -> Path:
    """Run the full FGFR3MUT-style tiling pipeline on one slide.

    Output: <out_dir>/<slide_stem>/coords.npy      (n_tiles, 2) int32 level-0 (x0, y0)
            <out_dir>/<slide_stem>/tile_masks.npy  (n_tiles, tile_size, tile_size) uint8 0/255,
                                                   nuclei+cytoplasm foreground=255 / rest=0,
                                                   index-aligned with coords.npy
            <out_dir>/<slide_stem>/tile_masks/<x0>_<y0>.png  (same masks, viewable)
            <out_dir>/<slide_stem>/mask.npy        (uint8 0/255 tissue mask, thumbnail res)
            <out_dir>/<slide_stem>/mask.png        (same mask, viewable)
            <out_dir>/<slide_stem>/metadata.json
            optionally <out_dir>/<slide_stem>/tiles/*.png if save_pngs=True
    """
    device = resolve_device(device)
    svs_path = Path(svs_path)
    t_start = time.time()
    total_ram = _total_ram_bytes()

    cpu_count = os.cpu_count() or 1
    if workers and workers > 0:
        n_workers = int(workers)
    else:
        n_workers = max(1, cpu_count)
    n_workers = max(1, min(n_workers, 256))

    if memory_budget_gb and memory_budget_gb > 0:
        budget_bytes = int(memory_budget_gb * (1 << 30))
    else:
        budget_bytes = int(total_ram * 0.5)
    if cache_gb and cache_gb > 0:
        cache_bytes = int(cache_gb * (1 << 30))
    else:
        # ~25% of RAM spread over the workers, capped at 4 GB each
        cache_bytes = int(min(4 << 30, max(64 << 20, total_ram * 0.25 / n_workers)))

    log(f"Using device: {device}")
    log(
        f"Workers: {n_workers} (cpus={cpu_count}, RAM={_human_bytes(total_ram)}, "
        f"preload budget={_human_bytes(budget_bytes)}, "
        f"openslide cache/worker={_human_bytes(cache_bytes)})"
    )
    log(f"Opening {svs_path} ...")
    slide = openslide.OpenSlide(str(svs_path))

    native_mpp = get_native_mpp(slide)
    level, level_downsample, level_mpp = pick_tiling_level(
        slide, native_mpp, target_mpp, mode=level_mode
    )
    log(f"native_mpp={native_mpp:.4f}, level={level} (mpp={level_mpp:.4f}), size={slide.dimensions}")

    if tissue_detector == "grandqc":
        log(f"Loading GrandQC weights from {grandqc_weights} on {device} ...")
        model, preprocessing_fn = load_grandqc_model(grandqc_weights, device=device)
        log("Running GrandQC tissue segmentation ...")
        tissue_mask, thumb_downsample = build_tissue_mask_grandqc(
            slide,
            model,
            preprocessing_fn,
            device=device,
            progress=progress,
            progress_position=progress_position,
        )
        del model
    elif tissue_detector == "otsu":
        log("Running Otsu tissue segmentation ...")
        tissue_mask, thumb_downsample = build_tissue_mask(slide)
        log(f"[otsu] tissue fraction={float((tissue_mask > 0).mean()):.3f}")
    else:
        raise ValueError(f"Unknown tissue_detector: {tissue_detector!r} (use 'grandqc' or 'otsu')")

    downsample_target = target_mpp / native_mpp
    step_l0 = int(round(tile_size * downsample_target))
    coords = generate_tile_grid(
        slide, tissue_mask, thumb_downsample, step_l0, min_tissue_fraction
    )
    n_candidates = coords.shape[0]

    rng = np.random.RandomState(seed)
    if max_tiles and n_candidates > max_tiles:
        log(f"Subsampling {n_candidates} -> {max_tiles} (seed={seed}) ...")
        idx = rng.choice(n_candidates, size=max_tiles, replace=False)
        coords = coords[sorted(idx)]

    out_slide_dir = Path(out_dir) / svs_path.name
    out_slide_dir.mkdir(parents=True, exist_ok=True)
    tile_masks_dir = out_slide_dir / "tile_masks"
    tile_masks_dir.mkdir(exist_ok=True)
    tiles_dir = out_slide_dir / "tiles"
    if save_pngs:
        tiles_dir.mkdir(exist_ok=True)

    n_tiles = coords.shape[0]
    log(
        f"Extracting {n_tiles} tiles + generating nuclei/cytoplasm foreground masks "
        f"({tile_size}x{tile_size}, method={mask_method}) ..."
    )

    size_at_level = max(1, int(round(step_l0 / level_downsample)))
    mask_min_area = int(tile_size * tile_size * mask_min_area_percent / 100)

    cfg = WorkerConfig(
        svs_path=str(svs_path),
        level=level,
        level_downsample=level_downsample,
        step_l0=step_l0,
        size_at_level=size_at_level,
        tile_size=tile_size,
        mask_method=mask_method,
        mask_kernel=_disk_kernel(mask_morphological_disk),
        mask_min_area=mask_min_area,
        mask_min_area_percent=mask_min_area_percent,
        mask_backend=mask_backend,
        save_pngs=save_pngs,
        tile_masks_dir=str(tile_masks_dir),
        tiles_dir=str(tiles_dir),
        cache_bytes=cache_bytes,
        start_method=start_method,
        spot_check_stride=spot_check_stride,
    )

    from multiprocessing import shared_memory

    mask_shm = shared_memory.SharedMemory(
        create=True, size=max(1, n_tiles * tile_size * tile_size)
    )
    cfg.mask_shm = mask_shm.name
    cfg.mask_shape = (n_tiles, tile_size, tile_size)
    tile_masks = np.ndarray(cfg.mask_shape, dtype=np.uint8, buffer=mask_shm.buf)

    # ---- optional: keep the whole pyramid level in RAM ----
    level_shm = None
    level_buf = None
    read_mode = "block_read"
    lw, lh = slide.level_dimensions[level]
    level_bytes = lw * lh * 3
    if preload in ("auto", "on"):
        if level_bytes > budget_bytes and preload == "on":
            log(
                f"[preload] level needs {_human_bytes(level_bytes)} > budget "
                f"{_human_bytes(budget_bytes)}; using block reads instead"
            )
        else:
            level_shm, nbytes, (lw, lh) = _alloc_level_buffer(slide, level)
            if level_shm is not None:
                cfg.level_shm = level_shm.name
                cfg.level_shape = (lh, lw, 3)
                cfg.level_origin_px = (0, 0)
                read_mode = "preload_ram"
                level_buf = np.ndarray(
                    (lh, lw, 3), dtype=np.uint8, buffer=level_shm.buf
                )

    blocks = _plan_blocks(coords, step_l0, block_tiles)
    t_tiles = time.time()
    try:
        if read_mode == "preload_ram":
            bands = _level_bands(lw, lh)
            log(
                f"[preload] decoding level {level} ({lw}x{lh}, {_human_bytes(level_bytes)}) "
                f"into RAM in {len(bands)} bands using {n_workers} workers ..."
            )
            t_pre = time.time()
            _run_tasks(
                _fill_level_band,
                bands,
                n_workers,
                "preload",
                cfg,
                desc=f"[preload] decoding level {level} into RAM",
                unit="row",
                total=lh,
                by_result=True,
                progress=progress,
                progress_position=progress_position,
            )
            log(f"[preload] level resident in RAM after {time.time() - t_pre:.1f}s")

            # ---- sanity check: the buffer must match the slide ----
            ok, diff, bright = _verify_level_buffer(
                slide, level, level_buf, coords, size_at_level
            )
            if ok:
                log(
                    f"[preload] verified against direct reads "
                    f"(max mean abs diff {diff:.2f}, max brightness diff {bright:.2f})"
                )
            else:
                log(
                    f"WARNING: [preload] the preloaded level does not match the slide "
                    f"(max mean abs diff {diff:.2f}, max brightness diff {bright:.2f}) "
                    "-- falling back to per-block reads"
                )
                read_mode = "block_read"
                cfg.level_shm = None
                cfg.level_shape = ()
                try:
                    level_shm.close()
                    level_shm.unlink()
                except Exception:  # pragma: no cover
                    pass
                level_shm = None

        _run_tasks(
            _process_block,
            blocks,
            n_workers,
            "tiles",
            cfg,
            desc="[tiles] extracting tiles + masks",
            unit="tile",
            total=n_tiles,
            by_result=True,
            progress=progress,
            progress_position=progress_position,
        )
    finally:
        _close_worker()
    tile_time = time.time() - t_tiles

    np.save(out_slide_dir / "coords.npy", coords)
    np.save(out_slide_dir / "tile_masks.npy", tile_masks)
    log(f"Saved tile_masks.npy {tile_masks.shape} + {n_tiles} PNGs -> {tile_masks_dir}")

    np.save(out_slide_dir / "mask.npy", tissue_mask)
    cv2.imwrite(str(out_slide_dir / "mask.png"), tissue_mask)

    metadata = {
        "svs_path": str(svs_path),
        "native_mpp": native_mpp,
        "target_mpp": target_mpp,
        "level": level,
        "level_downsample": level_downsample,
        "level_mpp": level_mpp,
        "tile_size": tile_size,
        "min_tissue_fraction": min_tissue_fraction,
        "nb_tiles": int(n_tiles),
        "nb_candidate_tiles": int(n_candidates),
        "mask_shape": list(tissue_mask.shape),
        "mask_thumb_downsample": thumb_downsample,
        "tile_masks_shape": list(tile_masks.shape),
        "tile_masks_method": (
            f"per-tile grayscale Otsu threshold (method={mask_method}, invert=True) on the tile's own RGB "
            f"pixels: nuclei (dark purple) + cytoplasm (light pink) = foreground=255, glass/background=0"
        ),
        "tile_masks_morphological_disk": mask_morphological_disk,
        "tile_masks_min_area_percent": mask_min_area_percent,
        "sampling_mode": {"mode": "random", "seed": seed} if len(coords) > (max_tiles or 0) else {"mode": "all"},
        "slide_size": list(slide.dimensions),
        "tissue_detector": (
            "grandqc_unetplusplus_efficientnet-b0 (Weng et al. 2024, public stand-in for BUNet)"
            if tissue_detector == "grandqc"
            else "otsu_saturation_threshold (public stand-in for BUNet)"
        ),
        # --- v2.0 execution metadata ---
        "tool_version": "2.0-parallel",
        "level_mode": level_mode,
        "read_mode": read_mode,
        "workers": n_workers,
        "block_tiles": int(block_tiles),
        "openslide_cache_bytes_per_worker": int(cache_bytes),
        "preload_budget_bytes": int(budget_bytes),
        "mask_backend": mask_backend,
        "spot_check_stride": int(spot_check_stride),
        "spot_check_min_ncc": _SPOT_CHECK_MIN_NCC,
        "timings_sec": {
            "total": round(time.time() - t_start, 2),
            "tiles": round(tile_time, 2),
        },
    }
    with open(out_slide_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    slide.close()

    # release the shared buffers
    try:
        mask_shm.close()
        mask_shm.unlink()
    except Exception:  # pragma: no cover
        pass
    if level_shm is not None:
        try:
            level_shm.close()
            level_shm.unlink()
        except Exception:  # pragma: no cover
            pass

    log(
        f"DONE {svs_path.name}: kept {n_tiles} tiles "
        f"in {time.time() - t_start:.1f}s ({read_mode}, {n_workers} workers) "
        f"-> {out_slide_dir}"
    )
    print(
        f"{svs_path.name}: native_mpp={native_mpp:.4f}, level={level} "
        f"(mpp={level_mpp:.4f}), tissue_detector={tissue_detector}, "
        f"kept {n_tiles} tiles "
        f"(min_tissue_fraction={min_tissue_fraction}) -> {out_slide_dir}"
    )
    return out_slide_dir


def _tile_one(args) -> str:
    """Worker entry point for --slide_workers > 1."""
    kwargs, svs_path = args
    tile_slide(svs_path=svs_path, **kwargs)
    return svs_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--svs_path",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to a WSI file (.svs, ...). Several slides can be given.",
    )
    parser.add_argument("--out_dir", type=str, required=True, help="Output root directory")
    parser.add_argument("--target_mpp", type=float, default=1.0, help="Target microns-per-pixel (paper: 1.0)")
    parser.add_argument("--tile_size", type=int, default=224, help="Tile side length in pixels (paper: 224)")
    parser.add_argument(
        "--min_tissue_fraction",
        type=float,
        default=0.60,
        help="Minimum fraction of a tile that must be tissue to keep it (paper: 0.60)",
    )
    parser.add_argument(
        "--max_tiles",
        type=int,
        default=5000,
        help="Cap on tiles per slide, randomly subsampled if exceeded (paper: 5000)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for tile subsampling (verified data: 0)")
    parser.add_argument(
        "--save_pngs",
        action="store_true",
        help="Also write each kept tile as a PNG under <out_dir>/<slide>/tiles/",
    )
    parser.add_argument(
        "--tissue_detector",
        type=str,
        default="grandqc",
        choices=["grandqc", "otsu"],
        help="Tissue/matter segmentation backend (default: grandqc, see module docstring)",
    )
    parser.add_argument(
        "--grandqc_weights",
        type=str,
        default=str(DEFAULT_GRANDQC_WEIGHTS),
        help="Path to GrandQC Tissue_Detection_MPP10.pth (download from https://zenodo.org/records/14507273)",
    )
    parser.add_argument("--device", type=str, default="auto", help="auto (default: cuda:0 if GPU available else cpu), cpu, or cuda:0/cuda:1 (falls back to cpu with a warning if no GPU)")
    parser.add_argument(
        "--mask_method",
        type=str,
        default="otsu",
        choices=["otsu", "adaptive", "color_based"],
        help="Per-tile foreground (nuclei+cytoplasm) segmentation method, see background_remover.py (default: otsu)",
    )
    parser.add_argument(
        "--mask_morphological_disk",
        type=int,
        default=2,
        help="Disk radius for closing small holes in the per-tile foreground mask (default: 2)",
    )
    parser.add_argument(
        "--mask_min_area_percent",
        type=float,
        default=0.1,
        help="Remove foreground specks smaller than this %% of tile area (default: 0.1)",
    )
    parser.add_argument(
        "--level_mode",
        type=str,
        default="closest_mpp",
        choices=["closest_mpp", "openslide_best"],
        help=(
            "Pyramid level used to cut tiles: 'closest_mpp' (default, the level whose "
            "microns-per-pixel is closest to --target_mpp, same rule as scripts/reproduce/tiling.py) "
            "or 'openslide_best' (v1.0 legacy behaviour, can pick level 0 = 4x oversampling)."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Tile worker processes per slide (default: all CPU cores)",
    )
    parser.add_argument(
        "--slide_workers",
        type=int,
        default=1,
        help="Slides processed in parallel (default: 1, i.e. one slide at a time)",
    )
    parser.add_argument(
        "--block_tiles",
        type=int,
        default=8,
        help="Tiles per read block (block_tiles x block_tiles grid cells, default: 8)",
    )
    parser.add_argument(
        "--preload",
        type=str,
        default="off",
        choices=["auto", "on", "off"],
        help=(
            "Cache the pyramid level in RAM (shared memory) and cut tiles from it. "
            "'off' (default): tiles are cut from per-block reads of the slide -- "
            "same speed, canonical pixels, no extra RAM. 'on'/'auto': decode the "
            "whole level into /dev/shm first (needs ~0.5-2.5 GB per slide); the "
            "buffer is verified against direct reads and the run falls back to "
            "per-block reads if it does not match."
        ),
    )
    parser.add_argument(
        "--memory_budget_gb",
        type=float,
        default=0.0,
        help="RAM budget for the in-RAM pyramid level (default: 50%% of physical RAM)",
    )
    parser.add_argument(
        "--cache_gb",
        type=float,
        default=0.0,
        help="OpenSlide tile cache per worker in GB (default: ~25%% of RAM split over the workers)",
    )
    parser.add_argument(
        "--mask_backend",
        type=str,
        default="fast",
        choices=["fast", "skimage"],
        help="'fast' (default) = OpenCV-only foreground mask, 'skimage' = legacy scikit-image path",
    )
    parser.add_argument(
        "--start_method",
        type=str,
        default="fork",
        choices=["fork", "spawn", "forkserver"],
        help="Multiprocessing start method for the tile workers (default: fork)",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Show live progress bars while decoding the slide / extracting tiles "
            "(default: on). Use --no-progress to fall back to plain log lines."
        ),
    )
    parser.add_argument(
        "--spot_check_stride",
        type=int,
        default=499,
        help=(
            "Every N-th tile is re-read directly from the slide and compared with "
            "the extracted tile; a mismatch aborts the run instead of writing wrong "
            "tiles (default: 499, i.e. ~10 checks per slide; 0 disables the check)."
        ),
    )
    args = parser.parse_args()

    common = dict(
        out_dir=args.out_dir,
        target_mpp=args.target_mpp,
        tile_size=args.tile_size,
        min_tissue_fraction=args.min_tissue_fraction,
        max_tiles=args.max_tiles,
        seed=args.seed,
        save_pngs=args.save_pngs,
        tissue_detector=args.tissue_detector,
        grandqc_weights=args.grandqc_weights,
        device=args.device,
        mask_method=args.mask_method,
        mask_morphological_disk=args.mask_morphological_disk,
        mask_min_area_percent=args.mask_min_area_percent,
        level_mode=args.level_mode,
        workers=args.workers,
        block_tiles=args.block_tiles,
        preload=args.preload,
        memory_budget_gb=args.memory_budget_gb,
        cache_gb=args.cache_gb,
        mask_backend=args.mask_backend,
        start_method=args.start_method,
        progress=args.progress,
        spot_check_stride=max(0, int(args.spot_check_stride)),
    )

    slides = [str(s) for s in args.svs_path]
    if len(slides) == 1 or args.slide_workers <= 1:
        for s in slides:
            tile_slide(svs_path=s, **common)
        return

    # several slides in parallel
    import concurrent.futures as cf

    n_slide_workers = min(int(args.slide_workers), len(slides))
    per_slide_workers = common["workers"] or (os.cpu_count() or 1)
    if per_slide_workers * n_slide_workers > (os.cpu_count() or 1):
        per_slide_workers = max(1, (os.cpu_count() or 1) // n_slide_workers)
        log(
            f"[slides] {n_slide_workers} slides in parallel x {per_slide_workers} "
            f"tile workers (scaled down to fit {os.cpu_count()} cores)"
        )
    kwargs = dict(common)
    kwargs["workers"] = per_slide_workers

    with cf.ProcessPoolExecutor(max_workers=n_slide_workers) as ex:
        # one progress-bar line per slide (each slide gets its own position so
        # the bars of the parallel slides do not overwrite each other)
        futures = {}
        for i, s in enumerate(slides):
            kw = dict(kwargs)
            kw["progress_position"] = i
            futures[ex.submit(_tile_one, (kw, s))] = s
        for fut in cf.as_completed(futures):
            fut.result()


if __name__ == "__main__":
    main()
