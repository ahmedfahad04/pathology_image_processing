"""Attention heatmap for FGFR3MUT pilot — overlay tile attention on the real SVS thumbnail.

Paper Fig.2e: tiles aggregated back into slide space; each tile colored by
P(mutant) (blue=WT, red=MUT), overlaid on the slide image.

How it works:
1. Load features.npy -> coords [:,:3] = (level, tx, ty) grid indices + embeddings.
2. Run Chowder.score_model per tile -> sigmoid -> P(mutant); average over ensemble.
3. Extract background from the real SVS with tifffile (same IFD/pyramid scheme
   as viewer.html: page0=full, page1=thumbnail stripped, page3~=5k px tiled).
   Default page3 (5533x4835) — sharp yet fits RAM; matches mask.npy resolution.
4. Map each tile to background pixels:
    level-L dims (WL,HL) + tile_size T from metadata.json
    tile (tx,ty) -> level-L rect (tx*T, ty*T, T, T) -> scaled by (bg_w/WL, bg_h/HL)
   Paint rect with coolwarm(score), alpha-blended over the thumbnail.
5. Print top/bottom tiles in (tx, ty) + score.

Usage:
    # fgfr3mut_env ONLY (torch-env's imagecodecs lacks JPEG decode -> onsvs crashes):
    PYTHONPATH=fgfr3mut conda run -n fgfr3mut_env python reproduce/heatmap.py --data_dir output/data_fgfr3_mini --slide TCGA-CF-A5U8 --out output/heatmaps
    python reproduce/heatmap.py --data_dir output/data_fgfr3_mini --slide TCGA-FJ-A871 --slide_idx 4 --svs image/TCGA-FJ-A871-01Z-00-DX5.8F79D0A8-5DE6-4159-AA77-61DACB21E867.svs --out output/heatmaps

Outputs in --out:
    attention_<slide>.png   # grid heatmap (reconstructed slide, NaN=background)
    overlay_<slide>.png     # heatmap on tissue mask (no SVS needed)
    onsvs_<slide>.png       # heatmap on the real SVS thumbnail  <-- what you asked for
    tilescores_<slide>.npz  # coords, scores, grid

Tweak: TOP_K, ALPHA, cmap, --bg_page, --n_models (10=fast pilot, 0=all 125).
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from fgfr3mut.chowder import Chowder
from fgfr3mut.utils import load_ckpt, sigmoid

TOP_K = 9
ALPHA = 0.75  # high so cmap blue dominates pink H&E (paper Fig.2e is near-opaque)
# Tile scores cluster mid-range (~0.3-0.5 even on WT slides); gamma Them
# toward blue so only true hotspots glow yellow/red like paper Fig.2e.
GAMMA = 2.0
# Paper Fig.2e style: tissue blue by default, yellow->red where P(mutant) is
# high. 'fgfr3_paper' custom map below; --cmap overrides (turbo/jet/inferno…).
CMAP = "fgfr3_paper"
_PAPER_COLORS = ["#2b5cab", "#5dade2", "#f7e731", "#f4511e", "#c21807"]


def _register_paper_cmap():
    """Blue (WT) -> yellow -> red (MUT), mimicking paper Fig.2e colorbar."""
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib
    import matplotlib.pyplot as plt
    if "fgfr3_paper" not in plt.colormaps():
        cmap = LinearSegmentedColormap.from_list("fgfr3_paper", _PAPER_COLORS, N=256)
        matplotlib.colormaps.register(cmap, name="fgfr3_paper")  # matplotlib >= 3.9


_register_paper_cmap()


def find_slide_dirs(features_dir: Path, slide_substr: str) -> List[Path]:
    cands = sorted(d for d in features_dir.iterdir() if d.is_dir() and slide_substr in d.name)
    if not cands:
        raise FileNotFoundError(f"No slide matching '{slide_substr}' in {features_dir}")
    return cands


def tile_scores(features: np.ndarray, weights_dir: Path, device: str, n_models: Optional[int]):
    """Mean sigmoid tile-scores across ensemble. Returns (n_tiles,) array."""
    X = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(device)
    model_paths = sorted(weights_dir.glob("split_*/*.pt"))
    if n_models:
        model_paths = model_paths[:n_models]
    print(f"  averaging tile scores over {len(model_paths)} model(s)")
    model = Chowder(in_features=features.shape[1], n_extreme=100)
    acc = None
    with torch.inference_mode():
        for mp in model_paths:
            model.load_state_dict(load_ckpt(str(mp), device))
            model.to(device).eval()
            s = model.score_model(X).squeeze(0).squeeze(1).cpu().numpy()
            s = sigmoid(s)
            acc = s if acc is None else acc + s
    return acc / len(model_paths)


def build_grid(tx: np.ndarray, ty: np.ndarray, scores: np.ndarray):
    """Pivot scattered tiles into dense (H, W) grid; missing = NaN (background)."""
    ux, uy = np.unique(tx), np.unique(ty)
    x_to_j = {v: j for j, v in enumerate(ux)}
    y_to_i = {v: i for i, v in enumerate(uy)}
    grid = np.full((len(uy), len(ux)), np.nan, dtype=np.float32)
    for x, y, s in zip(tx, ty, scores):
        grid[y_to_i[y], x_to_j[x]] = s
    return grid, ux, uy


def extract_background(svs_path: Path, bg_page: int):
    """Read one pyramid level as RGB. Same IFD scheme as viewer.html (GeoTIFF pages).

    SVS layout (tifffile pages): 0=full, 1=thumbnail stripped 878x768 (what the
    viewer shows first), 2..4=downsampled tiled levels. Default 3 (~5k px) is
    sharp and matches mask.npy, still small enough for RAM.
    """
    import tifffile
    with tifffile.TiffFile(svs_path) as t:
        if bg_page >= len(t.pages):
            bg_page = 1
        arr = t.pages[bg_page].asarray()
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr[:, :, :3], bg_page


def overlay_on_thumbnail(bg: np.ndarray, tx: np.ndarray, ty: np.ndarray,
                         scores: np.ndarray, tile_size: int, wl: int, hl: int,
                         alpha: float):
    """Paint each tile rect with cmap(score), alpha-blended over bg RGB."""
    h, w = bg.shape[:2]
    sx, sy = w / wl, h / hl
    fw, fh = max(1, int(round(tile_size * sx))), max(1, int(round(tile_size * sy)))
    cmap = plt.get_cmap(CMAP)
    # Grayscale base: pink eosin + blue makes purple; gray + blue stays blue
    # like paper Fig.2e. Alpha grows with score: blue wash on tissue,
    # near-opaque yellow/red hotspots.
    gray = (0.299 * bg[..., 0] + 0.587 * bg[..., 1] + 0.114 * bg[..., 2]) / 255.0
    canvas = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    for x, y, s in zip(tx, ty, scores):
        px, py = int(round(x * tile_size * sx)), int(round(y * tile_size * sy))
        x1, y1 = min(w, px + fw), min(h, py + fh)
        if x1 <= px or y1 <= py:
            continue
        v = float(np.clip(s, 0, 1)) ** GAMMA
        a = 0.5 + 0.5 * v
        rgb = np.array(cmap(v)[:3], dtype=np.float32)
        patch = canvas[py:y1, px:x1]
        canvas[py:y1, px:x1] = (1 - a) * patch + a * rgb
    return (np.clip(canvas, 0, 1) * 255).astype(np.uint8)


def save_attention(grid: np.ndarray, title: str, out_png: Path):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(np.nan_to_num(grid, nan=0.0) ** GAMMA, cmap=CMAP, vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("tile x")
    ax.set_ylabel("tile y")
    plt.colorbar(im, ax=ax, label="P(FGFR3-mutant) per tile")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_overlay_mask(grid: np.ndarray, mask: np.ndarray, title: str, out_png: Path):
    try:
        from PIL import Image
        h, w = mask.shape
        g = np.nan_to_num(grid, nan=0.0).astype(np.float32)
        heat = np.asarray(Image.fromarray(g).resize((w, h), resample=Image.BILINEAR), dtype=np.float32)
        valid = np.asarray(Image.fromarray((~np.isnan(grid)).astype(np.uint8) * 255).resize((w, h), resample=Image.NEAREST)) > 127
    except ImportError:
        h, w = mask.shape
        gh, gw = grid.shape
        ry, rx = max(1, h // gh), max(1, w // gw)
        heat = np.kron(np.nan_to_num(grid, nan=0.0), np.ones((ry, rx), dtype=np.float32))[:h, :w]
        valid = np.kron((~np.isnan(grid)).astype(np.uint8), np.ones((ry, rx), dtype=np.uint8))[:h, :w] > 0
    bg = mask.astype(np.float32)
    bg = (bg - bg.min()) / max(1e-6, bg.max() - bg.min())
    rgb = plt.get_cmap(CMAP)(np.clip(heat, 0, 1) ** GAMMA)[..., :3]
    canvas = np.stack([bg, bg, bg], axis=-1)
    m = valid[..., None] & (bg[..., None] > 0.02)
    canvas = np.where(m, (1 - ALPHA) * canvas + ALPHA * rgb, canvas)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(canvas)
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    global CMAP
    ap = argparse.ArgumentParser(description="Attention heatmap on real SVS thumbnail")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--slide", required=True, help="Substring, e.g. TCGA-CF-A5U8")
    ap.add_argument("--slide_idx", type=int, default=0, help="Which match if several (FJ-A871 has DX1-5)")
    ap.add_argument("--svs", default=None, help="Path to local .svs (default: image/<slide_dir_name>)")
    ap.add_argument("--bg_page", type=int, default=3, help="tifffile page: 1=thumbnail 878px, 3=~5k px (default)")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--cmap", default=CMAP, help="matplotlib colormap (fgfr3_paper, turbo, jet, inferno, coolwarm)")
    ap.add_argument("--out", default="output/heatmaps")
    ap.add_argument("--n_models", type=int, default=10, help="10=fast, 0=all 125")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    _register_paper_cmap()  # custom map must exist before plt.get_cmap(CMAP)
    CMAP = args.cmap

    data_dir = Path(args.data_dir)
    cands = find_slide_dirs(data_dir / "features", args.slide)
    print(f"Matches for '{args.slide}':")
    for i, c in enumerate(cands):
        print(f"  [{i}] {c.name}")
    slide_dir = cands[args.slide_idx]
    print(f"Using: {slide_dir.name}")

    md = json.load(open(slide_dir / "metadata.json"))
    level = md["level"]
    wl, hl = md["level_dimensions"][str(level)]
    tile_size = md["tile_size"]
    print(f"Extraction level {level}: {wl}x{hl}, tile {tile_size} (mpp {md['tile_mpp']})")

    arr = np.load(slide_dir / "features.npy", mmap_mode="r", allow_pickle=True).astype(np.float32)
    coords, feats = arr[:, :3], arr[:, 3:].copy()
    tx, ty = coords[:, 1].astype(int), coords[:, 2].astype(int)
    print(f"Tiles: {feats.shape[0]}, dim: {feats.shape[1]}")

    n_models = None if args.n_models == 0 else args.n_models
    scores = tile_scores(feats, data_dir / "models", args.device, n_models)

    order_top = np.argsort(scores)[::-1][:TOP_K]
    order_bot = np.argsort(scores)[:TOP_K]
    print(f"\nTop-{TOP_K} MUT tiles (tx, ty, score):")
    for i in order_top:
        print(f"  ({tx[i]}, {ty[i]}) score={scores[i]:.3f}")
    print(f"\nTop-{TOP_K} WT tiles (tx, ty, score):")
    for i in order_bot:
        print(f"  ({tx[i]}, {ty[i]}) score={scores[i]:.3f}")

    grid, ux, uy = build_grid(tx, ty, scores)
    print(f"\nReconstructed grid: {grid.shape} (tissue tiles: {(~np.isnan(grid)).sum()})")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_attention(grid, f"{slide_dir.name}\nattention: mean tile P(mutant)",
                   out_dir / f"attention_{slide_dir.name}.png")
    mask = np.load(slide_dir / "mask.npy", mmap_mode="r").astype(np.float32)
    save_overlay_mask(grid, np.asarray(mask), f"{slide_dir.name}\noverlay on tissue mask",
                      out_dir / f"overlay_{slide_dir.name}.png")

    svs_path = Path(args.svs) if args.svs else Path("image") / slide_dir.name
    if svs_path.exists():
        bg, used_page = extract_background(svs_path, args.bg_page)
        print(f"Background: {svs_path.name} page {used_page} -> {bg.shape[1]}x{bg.shape[0]}")
        painted = overlay_on_thumbnail(bg, tx, ty, scores, tile_size, wl, hl, args.alpha)
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.imshow(painted)
        ax.set_title(f"{slide_dir.name}\nattention on SVS thumbnail (page {used_page})")
        ax.axis("off")
        out_svs = out_dir / f"onsvs_{slide_dir.name}.png"
        fig.savefig(out_svs, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_svs}")
    else:
        print(f"WARNING: SVS not found at {svs_path} — skipped onsvs_*.png (mask overlay still saved).")

    np.savez(out_dir / f"tilescores_{slide_dir.name}.npz",
             coords=coords, scores=scores, grid=grid)
    print(f"Saved to {out_dir}/: attention_*.png, overlay_*.png, onsvs_*.png, tilescores_*.npz")


if __name__ == "__main__":
    main()
