"""Turn the FGFR3 tiler's output into CLAM-format patch files for E2E-ABMILX.

The FGFR3 experiment tiled every slide with ``fgfr3mut_tile_wsi.py`` into
``<tiles_root>/<slide>.svs/{coords.npy, metadata.json}``: level-0 (x0, y0) of
224 px tiles at ~1.0 um/px, read from pyramid level ``metadata["level"]``.
E2E's own ``CLAM/save_patches.py`` reads CLAM ``patches/<slide_id>.h5`` files
(a ``coords`` dataset of level-0 (x, y) with ``patch_level`` / ``patch_size``
attributes) and crops ``read_region(coord, patch_level, (patch_size,)*2)``.

This script writes those .h5 files so E2E crops EXACTLY the tiles Chowder was
trained on, with the same geometry as ``train_fgfr3mut.py:embed_slide()``:

    step_l0    = round(224 * target_mpp / native_mpp)   # tile footprint, level-0 px
    patch_size = max(1, round(step_l0 / level_downsample))

Two scales are written, because the E2E configs train with multi-scale input:

  main/  the FGFR3 tiles themselves                  (~1.0 um/px after resize)
  sub/   the 2x2 parent blocks containing >=1 tile    (~2.0 um/px after resize)

The sub scale only feeds E2E training (it is never used at test time) and covers
the same tissue as the main tiles. Each output dir also gets the
``process_list.csv`` (column ``slide_id`` = file name with extension) that
save_patches.py expects.

Only slides listed in the FGFR3 ``train_manifest.json`` are converted, so the
E2E slide set is identical to the Chowder one.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np


def tile_geometry(meta):
    """Return (patch_level, step_l0, patch_size) exactly as embed_slide() reads tiles."""
    step_l0 = int(round(224 * meta["target_mpp"] / meta["native_mpp"]))
    patch_size = max(1, int(round(step_l0 / meta["level_downsample"])))
    return int(meta["level"]), step_l0, patch_size


def parent_blocks(coords, step_l0, slide_size):
    """Level-0 origins of the 2x2 tile blocks that contain at least one tile.

    The tiler's grid starts at (0, 0) with stride step_l0, so block origins are
    multiples of 2*step_l0. Blocks are shifted back inside the slide when they
    would overrun its right/bottom edge.
    """
    block = 2 * step_l0
    parents = np.unique((coords // block) * block, axis=0)
    w0, h0 = slide_size
    parents[:, 0] = np.clip(parents[:, 0], 0, max(0, w0 - block))
    parents[:, 1] = np.clip(parents[:, 1], 0, max(0, h0 - block))
    return np.unique(parents, axis=0)


def write_h5(path, coords, patch_level, patch_size, slide_name):
    """Write a CLAM-style patch file (only coords + patch_level/patch_size are read)."""
    tmp = path.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as f:
        dset = f.create_dataset("coords", data=coords.astype(np.int64))
        dset.attrs["patch_level"] = int(patch_level)
        dset.attrs["patch_size"] = int(patch_size)
        dset.attrs["name"] = slide_name
    tmp.replace(path)


def h5_matches(path, coords, patch_level, patch_size):
    """True when an existing .h5 already holds these exact coords/attrs (rerun = no-op)."""
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as f:
            d = f["coords"]
            return (int(d.attrs["patch_level"]) == patch_level
                    and int(d.attrs["patch_size"]) == patch_size
                    and np.array_equal(d[:], coords.astype(np.int64)))
    except (OSError, KeyError):
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="FGFR3 train_manifest.json (defines the slide set)")
    ap.add_argument("--tiles_root", required=True, help="FGFR3 tiler output: <tiles_root>/<slide>.svs/coords.npy")
    ap.add_argument("--out_dir", required=True, help="writes <out_dir>/main and <out_dir>/sub")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    slides = manifest["slides"]
    tiles_root = Path(args.tiles_root)
    out_main = Path(args.out_dir) / "main"
    out_sub = Path(args.out_dir) / "sub"
    for d in (out_main, out_sub):
        (d / "patches").mkdir(parents=True, exist_ok=True)

    missing = [s for s in slides if not (tiles_root / s / "coords.npy").exists()
               or not (tiles_root / s / "metadata.json").exists()]
    if missing:
        sys.exit(f"error: {len(missing)} manifest slide(s) have no tiles under {tiles_root}, e.g. {missing[0]}\n"
                 "       re-run the FGFR3 tiler for them (run_e2e_fgfr3.sh does this unless SKIP_TILE=1)")

    n_main = n_sub = written = 0
    levels = {}
    for i, slide in enumerate(slides):
        stem = slide[:-len(".svs")] if slide.lower().endswith(".svs") else slide
        meta = json.loads((tiles_root / slide / "metadata.json").read_text())
        coords = np.load(tiles_root / slide / "coords.npy").astype(np.int64)
        if len(coords) == 0:
            sys.exit(f"error: {slide} has 0 tiles but is in the manifest")

        level, step_l0, psize = tile_geometry(meta)
        parents = parent_blocks(coords, step_l0, meta["slide_size"])
        psize_sub = max(1, int(round(2 * step_l0 / meta["level_downsample"])))

        for path, c, ps in ((out_main / "patches" / f"{stem}.h5", coords, psize),
                            (out_sub / "patches" / f"{stem}.h5", parents, psize_sub)):
            if not h5_matches(path, c, level, ps):
                write_h5(path, c, level, ps, stem)
                written += 1

        n_main += len(coords)
        n_sub += len(parents)
        levels[(level, psize)] = levels.get((level, psize), 0) + 1
        if (i + 1) % 50 == 0 or i + 1 == len(slides):
            print(f"  {i + 1}/{len(slides)} slides")

    for d in (out_main, out_sub):
        with open(d / "process_list.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["slide_id"])
            for s in slides:
                w.writerow([s])

    print(f"slides: {len(slides)} | main tiles: {n_main} | sub blocks: {n_sub} | h5 files (re)written: {written}")
    print("main-scale read geometry (patch_level, patch_size) -> #slides: "
          + ", ".join(f"{k}: {v}" for k, v in sorted(levels.items())))


if __name__ == "__main__":
    main()
