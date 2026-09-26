"""
FGFR3MUT-style WSI tiling.

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

Usage:
    python fgfr3mut_tile_wsi.py --svs_path slide.svs --out_dir ./tiles_out
    python fgfr3mut_tile_wsi.py --svs_path slide.svs --out_dir ./tiles_out \
        --tissue_detector grandqc --device cpu \
        --grandqc_weights ./models/grandqc/Tissue_Detection_MPP10.pth \
        --target_mpp 1.0 --tile_size 224 --min_tissue_fraction 0.60 \
        --max_tiles 5000 --seed 0
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

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


def log(msg: str) -> None:
    """Timestamped progress message to stdout (flushed so `tail -f` works)."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _tqdm(iterable, **kwargs):
    """tqdm if installed, else a plain iterator (no new dependency)."""
    try:
        from tqdm import tqdm

        return tqdm(iterable, **kwargs)
    except ImportError:
        return iterable


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
    return requested


def get_native_mpp(slide: "openslide.OpenSlide", fallback_mpp: float = 0.25) -> float:
    """Read native microns-per-pixel from slide metadata (level 0)."""
    mpp_x = slide.properties.get("openslide.mpp-x")
    if mpp_x is None:
        return fallback_mpp
    return float(mpp_x)


def pick_tiling_level(
    slide: "openslide.OpenSlide", native_mpp: float, target_mpp: float
) -> Tuple[int, float, float]:
    """Pick the OpenSlide pyramid level closest to target_mpp.

    Returns
    -------
    level: int
        OpenSlide level index (0 = full resolution).
    level_downsample: float
        slide.level_downsamples[level].
    level_mpp: float
        Effective MPP at that level (native_mpp * level_downsample).
    """
    downsample = target_mpp / native_mpp
    level = slide.get_best_level_for_downsample(downsample)
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
) -> Tuple[np.ndarray, float]:
    """Tissue mask via GrandQC (Weng et al., Nat Commun 2024).

    Mirrors GrandQC's own `wsi_tis_detect.py` preprocessing exactly:
    thumbnail at ~model_mpp resolution, re-encode through JPEG at
    quality=80 (the model was trained on JPEG-compressed inputs and is
    reported to behave suboptimally otherwise), tile into patch_size
    crops (last row/col taken from the tail instead of zero-padded),
    argmax over the 2-class softmax-free logits.

    Output convention matches build_tissue_mask (Otsu): 255 = tissue,
    0 = background. GrandQC's own class order is 0 = tissue, 1 = background
    (`make_class_map` in their repo colors class 0 as tissue), so we invert
    for a consistent contract with tissue_fraction_in_cell / generate_tile_grid.

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

    thumb = slide.get_thumbnail((max(1, int(w0 / reduction_factor)), max(1, int(h0 / reduction_factor))))

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
    with torch.no_grad():
        for h in _tqdm(range(he_n + 1), desc="[grandqc] rows", unit="row"):
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
                if done % 50 == 0 or done == total:
                    log(f"[grandqc] {done}/{total} patches ({time.time() - t0:.1f}s)...")

            row = np.concatenate(row_patches, axis=1)
            if h == he_n and overhang_he:
                row = row[p_s - overhang_he:p_s, :]
            rows.append(row)

    class_map = np.concatenate(rows, axis=0)
    tissue_mask = np.where(class_map == 0, 255, 0).astype(np.uint8)
    frac = float((tissue_mask > 0).mean())
    log(f"[grandqc] done in {time.time() - t0:.1f}s, tissue fraction={frac:.3f}")
    thumb_downsample = w0 / image.size[0]
    return tissue_mask, thumb_downsample


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
) -> List[Tuple[int, int]]:
    """Enumerate level-0 (x0, y0) top-left corners of tiles passing the
    tissue-coverage threshold."""
    w0, h0 = slide.dimensions
    nx, ny = len(range(0, w0, step_l0)), len(range(0, h0, step_l0))
    log(f"[grid] scanning {nx * ny} cells ({nx}x{ny}), step_l0={step_l0}...")
    kept = []
    for y0 in _tqdm(range(0, h0, step_l0), desc="[grid] rows", unit="row"):
        for x0 in range(0, w0, step_l0):
            frac = tissue_fraction_in_cell(tissue_mask, thumb_downsample, x0, y0, step_l0)
            if frac >= min_tissue_fraction:
                kept.append((x0, y0))
    log(f"[grid] kept {len(kept)}/{nx * ny} cells (min_tissue_fraction={min_tissue_fraction})")
    return kept


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
    picked pyramid level rarely matches the target MPP exactly — same
    reason real fgfr3mut tile_mpp varies 0.91-1.01 instead of exactly 1.0).
    """
    size_at_level = max(1, int(round(step_l0 / level_downsample)))
    region = slide.read_region((x0, y0), level, (size_at_level, size_at_level))
    patch = np.array(region.convert("RGB"))
    if patch.shape[0] != tile_size or patch.shape[1] != tile_size:
        patch = cv2.resize(patch, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
    return patch


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
    log(f"Using device: {device}")
    log(f"Opening {svs_path} ...")
    slide = openslide.OpenSlide(str(svs_path))

    native_mpp = get_native_mpp(slide)
    level, level_downsample, level_mpp = pick_tiling_level(slide, native_mpp, target_mpp)
    log(f"native_mpp={native_mpp:.4f}, level={level} (mpp={level_mpp:.4f}), size={slide.dimensions}")

    if tissue_detector == "grandqc":
        log(f"Loading GrandQC weights from {grandqc_weights} on {device} ...")
        model, preprocessing_fn = load_grandqc_model(grandqc_weights, device=device)
        log("Running GrandQC tissue segmentation ...")
        tissue_mask, thumb_downsample = build_tissue_mask_grandqc(
            slide, model, preprocessing_fn, device=device
        )
    elif tissue_detector == "otsu":
        log("Running Otsu tissue segmentation ...")
        tissue_mask, thumb_downsample = build_tissue_mask(slide)
        log(f"[otsu] tissue fraction={float((tissue_mask > 0).mean()):.3f}")
    else:
        raise ValueError(f"Unknown tissue_detector: {tissue_detector!r} (use 'grandqc' or 'otsu')")

    downsample_target = target_mpp / native_mpp
    step_l0 = int(round(tile_size * downsample_target))
    candidates = generate_tile_grid(
        slide, tissue_mask, thumb_downsample, step_l0, min_tissue_fraction
    )

    rng = np.random.RandomState(seed)
    if max_tiles and len(candidates) > max_tiles:
        log(f"Subsampling {len(candidates)} -> {max_tiles} (seed={seed}) ...")
        idx = rng.choice(len(candidates), size=max_tiles, replace=False)
        candidates = [candidates[i] for i in sorted(idx)]

    out_slide_dir = Path(out_dir) / svs_path.name
    out_slide_dir.mkdir(parents=True, exist_ok=True)
    log(f"Saving coords.npy + tile_masks + mask.npy + mask.png + metadata.json -> {out_slide_dir} ...")

    coords = np.array(candidates, dtype=np.int32)
    np.save(out_slide_dir / "coords.npy", coords)

    fg_remover = BackgroundRemover(
        method=mask_method,
        morphological_disk=mask_morphological_disk,
        min_area_percent=mask_min_area_percent,
        invert=True,
    )

    tile_masks_dir = out_slide_dir / "tile_masks"
    tile_masks_dir.mkdir(exist_ok=True)
    tiles_dir = out_slide_dir / "tiles"
    if save_pngs:
        from PIL import Image

        tiles_dir.mkdir(exist_ok=True)

    log(
        f"Extracting {len(candidates)} tiles + generating nuclei/cytoplasm "
        f"foreground masks ({tile_size}x{tile_size}, method={mask_method}) ..."
    )
    tile_masks = np.zeros((len(candidates), tile_size, tile_size), dtype=np.uint8)
    for i, (x0, y0) in enumerate(_tqdm(candidates, desc="[tiles+masks]", unit="tile")):
        patch = extract_tile(slide, int(x0), int(y0), level, level_downsample, step_l0, tile_size)
        tile_masks[i] = foreground_mask_from_tile(patch, fg_remover)
        cv2.imwrite(str(tile_masks_dir / f"{x0}_{y0}.png"), tile_masks[i])
        if save_pngs:
            Image.fromarray(patch).save(tiles_dir / f"{x0}_{y0}.png")
        if (i + 1) % 500 == 0:
            log(f"[tiles+masks] {i + 1}/{len(candidates)} ...")
    np.save(out_slide_dir / "tile_masks.npy", tile_masks)
    log(f"Saved tile_masks.npy {tile_masks.shape} + {len(candidates)} PNGs -> {tile_masks_dir}")

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
        "nb_tiles": len(candidates),
        "mask_shape": list(tissue_mask.shape),
        "mask_thumb_downsample": thumb_downsample,
        "tile_masks_shape": list(tile_masks.shape),
        "tile_masks_method": (
            f"per-tile grayscale Otsu threshold (method={mask_method}, invert=True) on the tile's own RGB "
            f"pixels: nuclei (dark purple) + cytoplasm (light pink) = foreground=255, glass/background=0"
        ),
        "tile_masks_morphological_disk": mask_morphological_disk,
        "tile_masks_min_area_percent": mask_min_area_percent,
        "sampling_mode": {"mode": "random", "seed": seed} if len(candidates) > (max_tiles or 0) else {"mode": "all"},
        "slide_size": list(slide.dimensions),
        "tissue_detector": (
            "grandqc_unetplusplus_efficientnet-b0 (Weng et al. 2024, public stand-in for BUNet)"
            if tissue_detector == "grandqc"
            else "otsu_saturation_threshold (public stand-in for BUNet)"
        ),
    }
    with open(out_slide_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    slide.close()

    log(
        f"DONE {svs_path.name}: kept {len(candidates)} tiles "
        f"in {time.time() - t_start:.1f}s -> {out_slide_dir}"
    )
    print(
        f"{svs_path.name}: native_mpp={native_mpp:.4f}, level={level} "
        f"(mpp={level_mpp:.4f}), tissue_detector={tissue_detector}, "
        f"kept {len(candidates)} tiles "
        f"(min_tissue_fraction={min_tissue_fraction}) -> {out_slide_dir}"
    )
    return out_slide_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--svs_path", type=str, required=True, help="Path to a WSI file (.svs, ...)")
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
        help="Also write each kept tile as a PNG under <out_dir>/<slide>/tiles/ (slow, disk-heavy)",
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
    args = parser.parse_args()

    tile_slide(
        svs_path=args.svs_path,
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
    )


if __name__ == "__main__":
    main()
