"""Replicate the paper's tiling pipeline on a local SVS -> (n_tiles, 1539) matrix.

Correct flow (paper p.9, "Preprocessing of whole-slide images"):
    WSI -> tissue_mask = UNet_segmentation(WSI)            # full slide FIRST
          -> tiles = split_into_grid(WSI, tile_size=224, mpp=1.0)
          -> kept = [t for t in tiles if fraction_matter >= 0.60]
          -> features = H_optimus_0(kept)                  # 1536-dim each
          -> slide_representation = stack -> (n_tiles, 1539) with coords

Reader backend (author versions: openslide lib 3.4.1, openslide-python 1.3.1):
- openslide (default): tile windows come from read_region at level 0, so the
  full slide is never decoded whole; mask runs on the lowest-res level.
  SVS stores only ~4 real levels; the author's 0-17 metadata table is their
  tool's internal dyadic pyramid (dims = ceil(full / 2**(17-k))), rebuilt here.
- tifffile (--reader tifffile): legacy path, decodes the ~1.0-MPP page fully.

Honest substitutions (documented, tweak later):
- UNet_segmentation -> Otsu matter mask on a low-res level.
  Paper's BUNet weights are private (Owkin internal; 460 slides, Dice 0.96).
  Same contract (bool mask) and same 60% rule; only fold/pen/blur rejection
  differs. Drop-in point: replace UNet_segmentation body.
- H_optimus_0 weights are gated: accept conditions at
  https://huggingface.co/bioptimus/H-optimus-0 then `huggingface-cli login`.
  Output 1536-dim (model card) + 3 coord cols = 1539 cols, same as repo .npy.

Run with torch-env (has timm+torch+openslide):
    # smoke test (16 tiles, ~10 min CPU):
    conda run -n torch-env python reproduce/tiling.py --svs image/<file>.svs --out output/tiling_test --max_tiles 16
    # COMPLETE single-image vector (all kept tiles, hours on CPU):
    conda run -n torch-env python reproduce/tiling.py --svs image/<file>.svs --out output/tiling_full --max_tiles 0 --compare_dir output/data_fgfr3_mini
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

# H-optimus-0 input spec (model card): 224x224, these stats.
H0_MEAN = np.array([0.707223, 0.578729, 0.703617], dtype=np.float32)
H0_STD = np.array([0.211883, 0.230117, 0.177517], dtype=np.float32)

TILE_PX = 224
TARGET_MPP = 1.0
KEEP_FRAC = 0.60
DYADIC_TOP = 17  # author table: level 17 = full resolution, 0 = smallest


def read_mpp(svs_path: Path) -> float:
    """Microns-per-pixel of level 0 from the SVS ImageDescription tag."""
    import tifffile
    with tifffile.TiffFile(svs_path) as t:
        desc = t.pages[0].tags.get("ImageDescription").value
    m = re.search(r"MPP\s*=\s*([0-9.]+)", desc)
    return float(m.group(1)) if m else 0.25


def UNet_segmentation(page_rgb: np.ndarray, max_width: int = 2048):
    """STAND-IN for the paper's private BUNet. Otsu on grayscale -> tissue=True.

    Same contract as BUNet for the pipeline: bool mask, same HxW as input
    (when scale == 1, else caller maps coordinates). max_width caps RAM.
    Returns (mask, otsu_threshold).
    Tweak point: replace body with a trained U-Net (e.g. milesial/Pytorch-UNet).
    """
    from PIL import Image
    h, w = page_rgb.shape[:2]
    scale = min(1.0, max_width / max(h, w))
    small = page_rgb if scale == 1.0 else np.asarray(
        Image.fromarray(page_rgb).resize((int(w * scale), int(h * scale)), Image.BILINEAR))
    gray = (0.299 * small[..., 0] + 0.587 * small[..., 1]
            + 0.114 * small[..., 2]).astype(np.float32)
    del small
    hist, _ = np.histogram(gray, bins=256, range=(0, 255))
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu) ** 2 / np.maximum(omega * (1 - omega), 1e-9)
    thresh = int(np.nanargmax(sigma_b))
    tissue = gray < thresh  # H&E tissue is darker than glass
    if tissue.mean() > 0.9:  # degenerate (all tissue) -> be strict
        thresh = int(gray.mean())
        tissue = gray < thresh
    otsu_thresh = thresh
    del gray
    if scale != 1.0:  # nearest-neighbour upscale back to input size
        tissue = np.asarray(Image.fromarray(tissue).resize((w, h), Image.NEAREST), dtype=bool)
    return tissue, otsu_thresh


def split_into_grid(page_w: int, page_h: int, footprint: int):
    """Non-overlapping grid of 224um tissue footprints. Yields (tx, ty)."""
    for ty in range(0, page_h // footprint):
        for tx in range(0, page_w // footprint):
            yield tx, ty


def extract_tile(page_rgb: np.ndarray, tx: int, ty: int, footprint: int,
                 tile_size: int = TILE_PX) -> np.ndarray:
    """Crop one footprint from a decoded page and resize to tile_size (tifffile path)."""
    tile = page_rgb[ty * footprint:(ty + 1) * footprint, tx * footprint:(tx + 1) * footprint]
    if footprint == tile_size:
        return tile
    from PIL import Image
    return np.asarray(Image.fromarray(tile).resize((tile_size, tile_size), Image.BILINEAR))


def extract_tile_oslide(slide, tx: int, ty: int, abs_tile: int,
                        tile_size: int = TILE_PX) -> np.ndarray:
    """Window-read one level-0 box via openslide and resize to tile_size.

    Peak RAM ~ abs_tile^2 pixels (2.5MB at 908px), never the whole slide.
    """
    from PIL import Image
    img = slide.read_region((tx * abs_tile, ty * abs_tile), 0, (abs_tile, abs_tile))
    img = img.convert("RGB")
    if abs_tile != tile_size:
        img = img.resize((tile_size, tile_size), Image.BILINEAR)
    return np.asarray(img)


def fraction_matter(tx: int, ty: int, tissue_mask: np.ndarray, footprint: int) -> float:
    """Fraction of the tile footprint flagged as matter (mask at same scale)."""
    m = tissue_mask[ty * footprint:(ty + 1) * footprint, tx * footprint:(tx + 1) * footprint]
    return float(m.mean())


def fraction_matter_box(tissue_mask: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    """Fraction of matter inside a mask-pixel box (openslide path: mask is low-res)."""
    return float(tissue_mask[y0:y1, x0:x1].mean())


def H_optimus_0(tiles_rgb, device: str, batch_size: int):
    """Batched H-optimus-0 embeddings -> (n, 1536) float32. tiles_rgb: list of 224x224x3 uint8."""
    import torch
    import timm
    import gc
    model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True,
                              init_values=1e-5, dynamic_img_size=False)
    model.to(device).eval()
    feats = []
    with torch.inference_mode():
        for i in range(0, len(tiles_rgb), batch_size):
            batch = np.stack(tiles_rgb[i:i + batch_size]).astype(np.float32) / 255.0
            batch = (batch - H0_MEAN) / H0_STD
            batch = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
            feats.append(model(batch).cpu().numpy().astype(np.float32))
            del batch
            gc.collect()  # release each batch: ViT-Giant activations are large on CPU
    return np.concatenate(feats, axis=0)


def pick_extraction_page(svs_path: Path, target_mpp: float = TARGET_MPP):
    """Pyramid page whose mpp is closest to target (tifffile path).

    Returns (page_idx, mpp0, page_mpp). footprint = round(224 * page_mpp / 1.0).
    """
    import tifffile
    mpp0 = read_mpp(svs_path)
    with tifffile.TiffFile(svs_path) as t:
        shapes = [(p.shape[1], p.shape[0]) for p in t.pages]  # (W, H)
    w0 = shapes[0][0]
    best, best_err, best_mpp = 0, float("inf"), mpp0
    for i, (w, _h) in enumerate(shapes):
        mpp_i = mpp0 * w0 / max(1, w)
        err = abs(mpp_i - target_mpp)
        if err < best_err:
            best, best_err, best_mpp = i, err, mpp_i
    return best, mpp0, best_mpp


def dyadic_table(w0: int, h0: int, mpp0: float, top: int = DYADIC_TOP):
    """Author-style pyramid table: level `top` = full res, 0 = smallest.

    Reverse-engineered from the released metadata: dims(k) = ceil(full / 2**(top-k)),
    mpp(k) = mpp0 * 2**(top-k). Verified exact on TCGA-4Z-AA7S (all 18 entries).
    """
    dims, mpps = {}, {}
    for k in range(top + 1):
        f = 2 ** (top - k)
        dims[str(k)] = [math.ceil(w0 / f), math.ceil(h0 / f)]
        mpps[str(k)] = round(mpp0 * f, 4)
    return dims, mpps


def author_level(extraction_w: int, w0: int, top: int = DYADIC_TOP) -> int:
    """Dyadic-table level whose width best matches the extraction width."""
    k = top - int(round(math.log2(max(1, w0) / max(1, extraction_w))))
    return max(0, min(top, k))


def build_metadata(svs_path: Path, mpp0: float, extraction_w: int,
                   tile_mpp: float, nb_tiles: int, seed: int, otsu_thresh: int,
                   reader: str) -> dict:
    """Repo-compatible metadata (same keys as Owkin tiling_tool 11.6.0 output)."""
    import hashlib
    import platform as pf
    try:
        import openslide
        os_py, os_lib = openslide.__version__, openslide.__library_version__
    except ImportError:
        os_py, os_lib = "not-installed", "not-installed"
    try:
        import torch
        torch_v = torch.__version__
    except ImportError:
        torch_v = "not-installed"
    try:
        import timm
        timm_v = timm.__version__
    except ImportError:
        timm_v = "not-installed"
    if reader == "openslide":
        import openslide as _os
        s = _os.OpenSlide(svs_path)
        w0, h0 = s.dimensions
        s.close()
    else:
        import tifffile
        with tifffile.TiffFile(svs_path) as t:
            w0, h0 = t.pages[0].shape[1], t.pages[0].shape[0]
    dims, mpps = dyadic_table(w0, h0, mpp0)
    h = hashlib.md5()
    with open(svs_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {
        "version": "3.0.0",
        "tiling_tool_version": "repro-1.0 (schema-compatible with owkin 11.6.0)",
        "nb_tiles": nb_tiles,
        "total_number_of_tiles": nb_tiles,
        "sampling_mode": {"mode": "random", "seed": seed},
        "matter_detector": {"name": "Otsu-standin-for-BUNet",
                            "dilatation": 1, "threshold": otsu_thresh},
        "features_extractor": "H-optimus-0",
        "level": author_level(extraction_w, w0),
        "tile_size": TILE_PX,
        # Pyramid-exact like the authors: 224 extraction-px * (w0 / ext_w).
        # (mpp-derived round(224/mpp0) suffers from rounded mpp0: 887 vs 896.)
        "absolute_tile_size": round(TILE_PX * w0 / max(1, extraction_w)),
        "slide_size": [w0, h0],
        "tile_mpp": round(tile_mpp, 4),
        "level_dimensions": dims,
        "level_mpp_mapping": mpps,
        "environment": {"python_version": pf.python_version(),
                        "openslide_version": os_lib,
                        "openslide_python_version": os_py,
                        "torch_version": torch_v, "timm_version": timm_v,
                        "platform": pf.platform()},
        "slide_hash": h.hexdigest(),
    }


def main():
    ap = argparse.ArgumentParser(description="Paper tiling replication -> (n_tiles,1539)")
    ap.add_argument("--svs", required=True)
    ap.add_argument("--out", default="output/tiling_test")
    ap.add_argument("--max_tiles", type=int, default=16,
                    help="Cap kept tiles (16=smoke test ~10min CPU; 0=all, hours on 1.1B ViT)")
    ap.add_argument("--batch_size", type=int, default=1,
                    help="Tiles per forward pass (1 = safest on CPU, ViT-Giant is ~4.4GB fp32)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0, help="Uniform sampling seed (paper: random)")
    ap.add_argument("--reader", default="auto", choices=["auto", "openslide", "tifffile"],
                    help="openslide = window reads from level 0 (default if installed)")
    ap.add_argument("--compare_dir", default=None,
                    help="Downloaded features dir to compare tile counts/coords against")
    args = ap.parse_args()

    svs_path = Path(args.svs)
    reader = args.reader
    if reader == "auto":
        try:
            import openslide  # noqa: F401
            reader = "openslide"
        except ImportError:
            reader = "tifffile"
    print(f"Reader backend: {reader}")

    kept_lvl0 = None  # (tx, ty) in level-0 pixels for the openslide path
    if reader == "openslide":
        import openslide
        slide = openslide.OpenSlide(svs_path)
        mpp0 = float(slide.properties.get("openslide.mpp-x", 0.25))
        w0, h0 = slide.dimensions
        ds = list(slide.level_downsamples)
        # STEP 0: extraction level = downsample closest to 1.0/mpp0 (paper: 1.0 MPP)
        ext_lvl = min(range(slide.level_count),
                      key=lambda i: abs(mpp0 * ds[i] - TARGET_MPP))
        tile_mpp = mpp0 * ds[ext_lvl]
        ext_w, ext_h = slide.level_dimensions[ext_lvl]
        # Author parity: tiles are exactly TILE_PX px at the extraction level
        # (nominal 224um); level-0 box is pyramid-exact, no mpp rounding.
        footprint = TILE_PX
        abs_tile = round(TILE_PX * ds[ext_lvl])  # level-0 px per tile (896, not 887)
        print(f"SVS: {svs_path.name}  mpp0={mpp0:.4f}  ext_level={ext_lvl} "
              f"(tile_mpp={tile_mpp:.4f}, footprint={footprint}px, abs_tile={abs_tile}px)")

        # STEP 1: mask on the lowest-res level (small), never the full slide
        mask_lvl = slide.level_count - 1
        mw, mh = slide.level_dimensions[mask_lvl]
        mask_rgb = np.asarray(slide.read_region((0, 0), mask_lvl, (mw, mh)).convert("RGB"))
        tissue_mask, otsu_thresh = UNet_segmentation(mask_rgb)
        print(f"STEP 1 tissue mask: level={mask_lvl} shape={mw}x{mh} "
              f"fraction={tissue_mask.mean():.3f} (Otsu thresh={otsu_thresh})")

        # STEP 2+3: level-0 grid, keep tiles with >=60% matter (mask coords mapped)
        sx, sy = mw / w0, mh / h0  # mask px per level-0 px
        nx, ny = w0 // abs_tile, h0 // abs_tile
        grid_n = nx * ny
        kept_lvl0 = [(tx, ty) for ty in range(ny) for tx in range(nx)
                     if fraction_matter_box(
                         tissue_mask,
                         math.floor(tx * abs_tile * sx), math.floor(ty * abs_tile * sy),
                         math.ceil((tx + 1) * abs_tile * sx),
                         math.ceil((ty + 1) * abs_tile * sy)) >= KEEP_FRAC]
        print(f"STEP 2 grid={grid_n}  STEP 3 kept={len(kept_lvl0)} (>=60% rule)")
        extraction_w = ext_w
    else:
        import tifffile
        # STEP 0: choose the pyramid page at ~1.0 MPP (paper: 224px == 224um there)
        page_idx, mpp0, page_mpp = pick_extraction_page(svs_path)
        tile_mpp = page_mpp
        footprint = max(1, round(TILE_PX * page_mpp / TARGET_MPP))
        print(f"SVS: {svs_path.name}  mpp0={mpp0:.4f}  page={page_idx} "
              f"(mpp={page_mpp:.4f}, footprint={footprint}px -> resized to {TILE_PX})")
        with tifffile.TiffFile(svs_path) as t:
            page = t.pages[page_idx].asarray()
            w0, h0 = t.pages[0].shape[1], t.pages[0].shape[0]  # level-0 dims for coords/table
        if page.ndim == 2:
            page = np.stack([page] * 3, axis=-1)
        page = page[:, :, :3]
        ph, pw = page.shape[:2]
        print(f"Page shape: {pw}x{ph}")

        # STEP 1: tissue_mask = UNet_segmentation(WSI) — full slide FIRST
        tissue_mask, otsu_thresh = UNet_segmentation(page)
        print(f"STEP 1 tissue mask: fraction={tissue_mask.mean():.3f} (Otsu thresh={otsu_thresh})")

        # STEP 2+3: grid + keep tiles with >=60% matter
        grid_n = (ph // footprint) * (pw // footprint)
        kept = [(tx, ty) for tx, ty in split_into_grid(pw, ph, footprint)
                if fraction_matter(tx, ty, tissue_mask, footprint) >= KEEP_FRAC]
        print(f"STEP 2 grid={grid_n}  STEP 3 kept={len(kept)} (>=60% rule)")
        extraction_w = pw
        slide = None

    if reader == "openslide":
        kept = kept_lvl0
    rng = np.random.default_rng(args.seed)
    if args.max_tiles and len(kept) > args.max_tiles:
        kept = [kept[i] for i in sorted(rng.choice(len(kept), args.max_tiles, replace=False))]
        print(f"Sampled down to max_tiles={args.max_tiles} (seed {args.seed}); "
              f"use --max_tiles 0 for the COMPLETE vector")
    else:
        print(f"COMPLETE run: all {len(kept)} kept tiles (no sampling cap)")

    # STEP 4: tile pixels (each exactly 224um, resized to 224px for H-0)
    if reader == "openslide":
        tiles_rgb = [extract_tile_oslide(slide, tx, ty, abs_tile) for tx, ty in kept]
        slide.close()
    else:
        tiles_rgb = [extract_tile(page, tx, ty, footprint) for tx, ty in kept]
        del page

    # STEP 5: features = H_optimus_0(kept) — 1536-dim each
    print(f"STEP 5 embedding {len(tiles_rgb)} tiles with H-optimus-0 ({args.device})...")
    emb = H_optimus_0(tiles_rgb, args.device, args.batch_size)
    print(f"Embeddings: {emb.shape}")
    del tiles_rgb

    # STEP 6: slide_representation = stack -> (n_tiles, 1539)
    if reader == "openslide":  # coords in extraction-level grid units, like the repo
        coords = np.array([[author_level(extraction_w, w0),
                            (tx * abs_tile) // footprint,
                            (ty * abs_tile) // footprint] for tx, ty in kept],
                          dtype=np.float32)
    else:
        coords = np.array([[author_level(extraction_w, w0), tx, ty] for tx, ty in kept],
                          dtype=np.float32)
    slide_rep = np.concatenate([coords, emb], axis=1)
    print(f"STEP 6 slide_representation: {slide_rep.shape}")

    out_dir = Path(args.out) / (svs_path.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "features.npy", slide_rep)
    np.save(out_dir / "mask.npy", tissue_mask.astype(np.float32))
    meta = build_metadata(svs_path, mpp0, extraction_w, tile_mpp,
                          len(kept), args.seed, otsu_thresh, reader)
    json.dump(meta, open(out_dir / "metadata.json", "w"), indent=2)
    print(f"Saved: {out_dir}/features.npy, mask.npy, metadata.json")

    # 7) test against downloaded reference feature set
    if args.compare_dir:
        ref = Path(args.compare_dir) / "features" / svs_path.name / "features.npy"
        if ref.exists():
            r = np.load(ref, mmap_mode="r")
            print(f"\nReference: {r.shape} (nb_tiles={r.shape[0]})")
            print(f"Replicated: {slide_rep.shape} (nb_tiles={slide_rep.shape[0]})")
            print(f"Tile-count ratio: {slide_rep.shape[0] / r.shape[0]:.2f}")
            print("Note: counts differ (Otsu-vs-BUNet mask, max_tiles cap).")
        else:
            print(f"\nNo reference for {svs_path.name} in {args.compare_dir} "
                  f"(e.g. TSA slide was never uploaded).")


if __name__ == "__main__":
    main()
