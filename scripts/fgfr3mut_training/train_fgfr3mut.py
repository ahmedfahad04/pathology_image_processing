"""FGFR3MUT training pipeline: SVS tiles -> H-optimus-0 features -> Chowder MIL.

=====================================================================================
BEGINNER'S GUIDE — read this if you're new to PyTorch / image ML pipelines
=====================================================================================

THE BIG PICTURE
----------------
We want to predict whether a cancer patient's tumor has an "FGFR3 mutation" (MUT)
or not (WT = wild-type), just by looking at a microscope image of their tissue
(a "whole-slide image", or WSI, stored as a giant .svs file — often 50,000x50,000
pixels or bigger, way too large to feed into a neural network directly).

We solve the "too large" problem in 3 stages:

  Stage 1 (a DIFFERENT script: scripts/preprocessing/fgfr3mut_tile_wsi.py)
      Cut the giant image into thousands of small 224x224 pixel squares
      ("tiles"), keeping only squares that contain actual tissue (not empty
      glass background). This is standard image preprocessing, no ML yet.

  Stage 2 (this script, functions embed_slide / build_features_from_tiles)
      Run each small tile through a PRETRAINED, FROZEN neural network called
      "H-optimus-0" (a Vision Transformer trained by Bioptimus on millions of
      pathology images). "Frozen" means we do NOT train it further — we just
      use it to turn each 224x224x3 pixel image into a single list of 1536
      numbers (a "feature vector" / "embedding") that summarizes what's in
      that tile. A slide with N tiles becomes an (N, 1536) matrix.

  Stage 3 (this script, functions train_one / Chowder model)
      Train a SMALL model (Chowder) that looks at all N of a slide's feature
      vectors at once and outputs one prediction: P(this slide is MUT).
      We only have ONE label per slide (not per tile!), so this is a
      "Multiple Instance Learning" (MIL) problem: a "bag" of N instances
      (tiles) shares one bag-level label (the patient's mutation status).

WHY IS THIS "WEAKLY SUPERVISED"?
---------------------------------
We never told the model *which* tiles look mutant — only "this whole bag of
~5000 tiles came from a MUT patient". Chowder has to figure out on its own
which tiles are informative. It does this by scoring every tile individually
with a tiny per-tile network, then only paying attention to the most extreme
scores (the tiles that look "most MUT-like" and "most WT-like").

KEY PYTORCH CONCEPTS YOU'LL SEE BELOW
---------------------------------------
  tensor            PyTorch's version of a numpy array; can live on CPU or GPU
                     and remembers how it was computed (for backpropagation).
  device             Either "cpu" or "cuda:0" (a GPU). `.to(device)` moves a
                     tensor/model there. GPUs do the same matrix math as CPUs
                     but thousands of times in parallel, so training is faster.
  model.eval()        Puts a model in "inference mode" (turns off things like
                     dropout that only make sense during training).
  model.train()       The opposite: turns training-only behavior back on.
  torch.inference_mode()   Tells PyTorch "don't bother tracking gradients for
                     anything in this block" — much faster / less memory when
                     you're only doing a forward pass (no learning happening).
  gradient / backward()   To train a model we compute a "loss" (how wrong the
                     prediction was) and call `.backward()`, which uses calculus
                     (the chain rule) to figure out how to nudge every learnable
                     number in the model to make the loss a little smaller next
                     time. This is called "backpropagation".
  optimizer.step()   Actually applies those nudges to the model's parameters.
  logit               A model's raw, unbounded output number before it's been
                     squashed into a 0..1 probability. `sigmoid(logit)` does
                     that squashing. We train on logits (numerically nicer)
                     and only apply sigmoid when we want an actual probability.
  state_dict          A plain dictionary of {parameter_name: tensor_of_numbers}
                     representing everything a model has "learned". Saving/
                     loading this is how we save/load trained models to disk.
  DataLoader          A PyTorch helper that automatically groups your dataset
                     into shuffled mini-batches for training.
  batch                Instead of training on 1 example at a time (slow, noisy)
                     or the whole dataset at once (too much memory), we train
                     on small groups ("batches") of examples at a time.

Paper: Bannier et al., Nat Commun 15:10914 (2024).
Model is Chowder (Courtiol et al. 2018), NOT attention-MIL (Ilse et al.):
  (N,1536) -> TilesMLP[1536->128 Sigmoid ->1] -> per-tile scores
  -> ExtremeLayer(top100+bottom100) -> (200,1)
  -> MLP[200->128 Sigmoid ->64 Sigmoid ->1] -> logit -> sigmoid -> P(MUT)

Usage (run from the project root; all generated output is written under ./results/):
  # A. Preprocessing (ONLY scripts/preprocessing/fgfr3mut_tile_wsi.py):
  for S in image/*.svs; do
    python scripts/preprocessing/fgfr3mut_tile_wsi.py --svs_path "$S" \\
      --out_dir ./results/tiles_out --tissue_detector grandqc --device auto \\
      --target_mpp 1.0 --tile_size 224 --min_tissue_fraction 0.60 \\
      --max_tiles 5000 --seed 0
  done
  # B. Embed + train from tiles (--feat_dir/--out_dir default to ./results/*, override if needed):
  python scripts/fgfr3mut_training/train_fgfr3mut.py --tiles_root ./results/tiles_out --slides_dir ./image \\
    --labels_csv ./results/labels.csv --epochs 30
  # C. Train directly from precomputed (N,1539) features.npy (paper's HuggingFace dump):
  python scripts/fgfr3mut_training/train_fgfr3mut.py --features_dir ./output/data_fgfr3_mini/features \\
    --labels_csv ./results/labels.csv --epochs 30
  # D. Smoke test (no GPU/SVS needed):
  python scripts/fgfr3mut_training/train_fgfr3mut.py --smoke_test --out_dir /tmp/fgfr3_smoke --epochs 1

labels.csv: slide,label  (slide = "<name>.svs" folder/file name, label 0/1)

Requirements: see requirements.txt next to this script (pip install -r scripts/fgfr3mut_training/requirements.txt).
"""

# ---- Standard library imports -------------------------------------------------
import argparse   # parses command-line flags like --epochs 30 into a Python object
import json       # reads the metadata.json files written by the tiling script
import subprocess  # used below to `git clone` the fgfr3mut dependency if it's missing
import sys        # lets us edit Python's module search path (sys.path), see below
import time         # used to measure embedding speed and print a live time estimate/ETA
from pathlib import Path  # modern, cross-platform way to build file paths

# ---- Third-party imports -------------------------------------------------------
import numpy as np    # numerical arrays (CPU only, no GPU/gradient support)
import torch           # PyTorch: tensors, neural network layers, autograd, GPU support
from sklearn.metrics import roc_auc_score            # computes the AUC evaluation metric
from sklearn.model_selection import StratifiedKFold   # splits data into train/val folds
from torch.utils.data import DataLoader                # batches + shuffles a Dataset for training

# ---------------------------------------------------------------------------------
# The `fgfr3mut` package (Chowder model, bootstrap metrics, etc. — Bannier et al.'s
# own paper code) is an EXTERNAL dependency this project builds on top of, not code
# we wrote or vendor in git (see .gitignore) — like a `pip install`-able dependency,
# except it isn't published on PyPI, so there's nothing for `pip install` to fetch.
# Instead, we expect (or fetch) a plain `git clone` of it living right next to this
# script, at scripts/fgfr3mut_training/fgfr3mut/fgfr3mut/chowder.py, and add that
# folder to `sys.path` (Python's list of "places to look for importable code") so
# a plain `import fgfr3mut` below finds it.
# ---------------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent          # scripts/fgfr3mut_training/
FGFR3MUT_DIR = SCRIPT_DIR / "fgfr3mut"                  # .../fgfr3mut_training/fgfr3mut/
FGFR3MUT_REPO_URL = "https://github.com/PABannier/fgfr3mut"


def ensure_fgfr3mut_repo():
    """Clone the fgfr3mut dependency into FGFR3MUT_DIR if it isn't already there.

    Every train_fgfr3mut.py / infer_fgfr3mut.py run calls this before trying to
    `import fgfr3mut`, so a completely fresh checkout of THIS repo "just works"
    on first run — no manual clone-this-other-repo-yourself setup step needed.
    """
    if (FGFR3MUT_DIR / "fgfr3mut" / "chowder.py").exists():
        return  # already present (either cloned by an earlier run, or by hand) — nothing to do

    print(f"fgfr3mut/ not found at {FGFR3MUT_DIR} — cloning it from {FGFR3MUT_REPO_URL} ...")
    try:
        # `subprocess.run([...], check=True)` runs `git clone <url> <dest>` as if
        # typed in a terminal, and raises an exception if git exits with an error
        # (e.g. no network) instead of silently continuing with a half-done clone.
        subprocess.run(["git", "clone", FGFR3MUT_REPO_URL, str(FGFR3MUT_DIR)], check=True)
    except FileNotFoundError:
        # Raised when the "git" program itself can't be found at all (not installed).
        raise RuntimeError(
            "git does not appear to be installed (or is not on PATH). Install git, "
            f"or manually clone {FGFR3MUT_REPO_URL} into {FGFR3MUT_DIR}"
        ) from None
    except subprocess.CalledProcessError as e:
        # Raised when git itself ran but reported failure (bad network, etc.).
        raise RuntimeError(
            f"git clone of {FGFR3MUT_REPO_URL} failed (exit code {e.returncode}). "
            f"Check your network connection, or manually clone it into {FGFR3MUT_DIR}"
        ) from None

    if not (FGFR3MUT_DIR / "fgfr3mut" / "chowder.py").exists():
        # Defensive check: the clone "succeeded" per git's exit code, but the
        # expected file still isn't there — e.g. the upstream repo's layout
        # changed. Fail loudly now rather than a confusing ImportError later.
        raise RuntimeError(
            f"cloned {FGFR3MUT_REPO_URL} into {FGFR3MUT_DIR}, but "
            "fgfr3mut/fgfr3mut/chowder.py still doesn't exist — has the upstream repo's layout changed?"
        )
    print("done.")


ensure_fgfr3mut_repo()
if str(FGFR3MUT_DIR) not in sys.path:
    sys.path.insert(0, str(FGFR3MUT_DIR))  # "search here first" when we `import fgfr3mut` below

# `noqa: E402` just tells linters "yes, I know these imports aren't at the very
# top of the file — that's intentional, because sys.path had to be patched first".
from fgfr3mut.chowder import Chowder  # noqa: E402  (the MIL model architecture)
from fgfr3mut.utils import SlideFeaturesDataset, pad_collate_fn  # noqa: E402

# ---------------------------------------------------------------------------------
# H0_MEAN / H0_STD: the H-optimus-0 model was trained on images normalized with
# these specific per-channel (R, G, B) mean and standard-deviation values (this
# is standard practice for vision models — similar to ImageNet's famous
# [0.485, 0.456, 0.406] mean). We MUST normalize our tiles the exact same way
# before feeding them in, or the model's predictions will be meaningless.
# ---------------------------------------------------------------------------------
H0_MEAN = np.array([0.707223, 0.578729, 0.703617], dtype=np.float32)
H0_STD = np.array([0.211883, 0.230117, 0.177517], dtype=np.float32)
IN_FEATURES = 1536   # length of the feature vector H-optimus-0 outputs per tile
N_EXTREME = 100       # Chowder keeps the top-100 and bottom-100 scoring tiles per slide


def resolve_device(requested="auto"):
    """Decide which PyTorch device ("cpu" or "cuda:0") to run on.

    - "auto" (the default): use a GPU if one is available, else fall back to CPU.
    - an explicit "cuda*" request on a machine with no GPU: warn and use CPU
      anyway instead of crashing.
    - anything else (e.g. "cpu"): use exactly what was asked for.
    """
    req = (requested or "auto").strip().lower()
    if req == "auto":
        # torch.cuda.is_available() checks whether PyTorch can see an NVIDIA GPU.
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if req.startswith("cuda") and not torch.cuda.is_available():
        print(f"warning: --device={requested} requested but no CUDA GPU is available; falling back to cpu")
        return "cpu"
    return requested


def load_h_optimus_0(device):
    """Download (or load from local cache) the frozen H-optimus-0 feature extractor.

    `timm` ("PyTorch Image Models") is a library of ready-made vision model
    architectures + pretrained weights. `create_model(..., pretrained=True)`
    downloads the weights from the given Hugging Face Hub repo the first time,
    then reuses a local cache on later runs.

    We load this ONCE and reuse the same in-memory model object for every
    slide — it has ~1.1 BILLION parameters, so re-loading it from disk for
    every single slide would waste minutes each time for no benefit (its
    weights never change; we never train it).

    `.to(device)` moves all of the model's weight tensors onto the chosen
    device (CPU RAM or GPU memory).
    `.eval()` puts it in inference mode (see the BEGINNER'S GUIDE above).
    We never call `.train()` on this model because we are never updating its
    weights — it stays "frozen" for the whole script.
    """
    import timm  # imported lazily here so --smoke_test doesn't need timm installed

    return timm.create_model(
        "hf-hub:bioptimus/H-optimus-0", pretrained=True,
        init_values=1e-5, dynamic_img_size=False,
    ).to(device).eval()


def embed_slide(svs_path, coords, level, level_ds, step_l0, device, model, batch=8, progress_cb=None):
    """Turn one slide's kept tiles into a (N_tiles, 1536) matrix of feature vectors.

    Parameters (all come from the tiling script's coords.npy / metadata.json):
        svs_path   path to the original whole-slide image file
        coords     (N, 2) array of (x, y) pixel coordinates — one per kept tile,
                   measured at the WSI's full-resolution "level 0"
        level      which zoom level of the WSI pyramid to actually read pixels
                   from (WSIs store the same image at multiple resolutions,
                   like a Google-Maps zoom pyramid, so we don't have to
                   downsample a full-res crop ourselves)
        level_ds   the "downsample factor" of that level versus level 0
        step_l0    tile width in level-0 pixels (so we crop the right amount
                   of real tissue, then resize down to exactly 224x224)
        model      the already-loaded, frozen H-optimus-0 model (see above)
        batch      how many tiles to feed through the model at once (a bigger
                   batch is faster per-tile but uses more memory)
        progress_cb  optional callback, called as progress_cb(n_tiles_just_embedded)
                   after every batch — lets the caller print a running
                   ETA without embed_slide needing to know anything about
                   how that ETA is computed/printed (see build_features_from_tiles).

    Returns:
        np.ndarray of shape (N_tiles, 1536), one feature vector per input tile.
    """
    import cv2         # OpenCV: used here only to resize images
    import openslide     # library for reading regions out of giant WSI files

    # OpenSlide gives us random-access reads into the pyramid file without ever
    # loading the whole (often 10+ GB) image into memory at once.
    slide = openslide.OpenSlide(str(svs_path))

    embs = []   # will collect one (batch, 1536) numpy array per flush() call
    cur = []    # the tile pixel-arrays we've accumulated but not yet embedded

    def flush():
        """Run everything currently sitting in `cur` through the model at once."""
        # 1) Stack the list of (224, 224, 3) uint8 images into one array,
        #    convert to float32, and rescale pixel values from [0, 255] to
        #    [0, 1] — this is the standard first step of image normalization.
        # 2) Subtract the per-channel mean and divide by the per-channel std
        #    (H0_MEAN / H0_STD) — see the comment above those constants.
        x = (np.stack(cur).astype(np.float32) / 255.0 - H0_MEAN) / H0_STD

        # PyTorch's convolution/transformer layers expect images in
        # "channels-first" order: (batch, channels, height, width), i.e. NCHW.
        # numpy/OpenCV instead give us "channels-last": (batch, height, width,
        # channels), i.e. NHWC. `.permute(0, 3, 1, 2)` just reorders those axes
        # — no data is changed, only how the axes are labeled/strided.
        x = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)

        # `torch.inference_mode()`: we're only doing a forward pass through a
        # frozen model, never calling .backward() on it, so we tell PyTorch not
        # to bother recording the computation graph needed for backprop. This
        # saves both memory and time.
        with torch.inference_mode():
            # model(x) runs the forward pass -> (batch, 1536) tensor of features.
            # .cpu() moves the result off the GPU (if it was on one) back to
            # regular CPU memory so we can convert it to a plain numpy array.
            embs.append(model(x).cpu().numpy().astype(np.float32))
        n_this_batch = len(cur)
        cur.clear()
        if progress_cb is not None:
            progress_cb(n_this_batch)

    # Walk through every kept-tile coordinate for this slide.
    for x0, y0 in coords:
        # `step_l0` is the tile size in level-0 pixels; `level_ds` tells us how
        # much smaller `level` is than level 0, so dividing gives us the tile
        # size *in level's own pixel units* — i.e. how big a square to read.
        s = max(1, round(step_l0 / level_ds))

        # Read an (s, s) pixel square starting at (x0, y0) [level-0 coordinates]
        # from the given pyramid `level`, convert from RGBA to plain RGB.
        px = np.array(slide.read_region((int(x0), int(y0)), level, (s, s)).convert("RGB"))

        # H-optimus-0 expects exactly 224x224 input images. If the level we
        # read from didn't give us exactly that (rounding, different pyramid
        # levels per slide, etc.), resize it now with bilinear interpolation.
        if px.shape[:2] != (224, 224):
            px = cv2.resize(px, (224, 224), interpolation=cv2.INTER_LINEAR)

        cur.append(px)
        if len(cur) == batch:   # once we have a full batch, embed it and reset
            flush()
    if cur:   # embed whatever's left over (a partial final batch)
        flush()

    slide.close()  # release the file handle
    # Stick all the per-batch (batch, 1536) arrays together into one
    # (N_tiles, 1536) array, in the same order the tiles were given to us.
    return np.concatenate(embs, axis=0)


def load_labels(labels_csv):
    """Read labels.csv (columns: slide,label) into a slide_name -> 0.0/1.0 mapping."""
    import pandas as pd  # imported lazily; only this function needs it

    df = pd.read_csv(labels_csv)
    assert {"slide", "label"} <= set(df.columns), "labels.csv needs columns: slide,label"
    # Use the "slide" column as the row index, keep only the "label" column,
    # and make sure labels are floats (0.0 / 1.0) since PyTorch losses want floats.
    return df.set_index("slide")["label"].astype(float)


def build_features_from_tiles(tiles_root, slides_dir, feat_dir, labels, device):
    """Turn each slide's saved tile coordinates into a cached (N,1539) feature matrix.

    Expects, for every slide_name in `labels`:
        tiles_root/<slide_name>/coords.npy      (N, 2) kept-tile (x, y) coordinates
        tiles_root/<slide_name>/metadata.json    tiling parameters (mpp, level, ...)
        slides_dir/<slide_name>                  the original .svs file

    Writes/reads a cache file at feat_dir/<slide_name>.npy so that re-running
    this script later (e.g. after fixing a bug in `train_one`) doesn't require
    re-embedding every tile through the expensive H-optimus-0 model again.

    Each cached .npy row is [level, x, y, <1536 embedding numbers...>] = 1539
    columns total — the first 3 are bookkeeping (matches the paper's own
    "1539" feature-file layout), the rest are the actual embedding used for training.

    Returns:
        X: a Python list of length n_slides, each element an (n_tiles_i, 1536)
           numpy array (slides can have DIFFERENT numbers of tiles — that's
           why this is a list of arrays and not one big stacked array).
        y: a (n_slides,) numpy array of 0.0/1.0 labels, same order as X.
        names: a Python list of the n_slides slide names, same order as X/y.
           NOTE: this can be SHORTER than `labels` if any slides were skipped
           (0 kept tiles) — always zip X/y against `names`, never against the
           original `labels`, or a skipped slide will silently misalign
           everything after it.
    """
    tiles_root, slides_dir, feat_dir = Path(tiles_root), Path(slides_dir), Path(feat_dir)
    feat_dir.mkdir(parents=True, exist_ok=True)  # create the cache folder if missing

    # ---- Pass 1: figure out what's already cached vs. what needs embedding ---
    # We do this as a separate pass (instead of deciding + embedding slide by
    # slide) so we can print an upfront "here's how many tiles total, here's
    # roughly how long that'll take" estimate BEFORE committing to what can be
    # a many-hour CPU job — giving you the chance to Ctrl-C out early instead
    # of finding out an hour in.
    slide_plan = []   # (slide_name, lab, tdir, meta, coords, out_path, needs_embed), in label order
    for slide_name, lab in labels.items():
        tdir = tiles_root / slide_name
        meta = json.load(open(tdir / "metadata.json"))
        coords = np.load(tdir / "coords.npy")
        if len(coords) == 0:
            print(f"skip {slide_name}: 0 kept tiles")
            continue

        out = feat_dir / f"{slide_name}.npy"
        # BUG THIS FIXES: originally this only checked `not out.exists()` — if
        # you re-run the TILING script with a different --max_tiles (or a
        # different tissue detector/seed), coords.npy changes, but a stale
        # feat_dir/<slide>.npy cached from an OLDER tiling run would still
        # "exist" and get silently reused as-is. That's not just "fewer tiles
        # than you'd expect" — the cached embeddings' (x, y) coordinates don't
        # even match the current coords.npy anymore, so training would run on
        # tiles that no longer correspond to what's actually in tiles_out/,
        # with zero warning. So: also re-embed whenever the cached file's own
        # stored coordinates (columns 1:3, see the comment below) don't
        # exactly match the coords.npy we just loaded above.
        needs_embed = True
        if out.exists():
            cached = np.load(out, mmap_mode="r")
            cached_coords = np.asarray(cached[:, 1:3], dtype=np.int64)
            if cached_coords.shape == coords.shape and np.array_equal(cached_coords, coords.astype(np.int64)):
                needs_embed = False
            else:
                print(f"note: cached features for {slide_name} ({cached_coords.shape[0]} tiles) don't match "
                      f"current coords.npy ({len(coords)} tiles) -- tiles_out/ was probably re-tiled since "
                      f"the cache was built; re-embedding.")
        slide_plan.append((slide_name, lab, tdir, meta, coords, out, needs_embed))

    tiles_to_embed = [coords for (_name, _lab, _tdir, _meta, coords, _out, needs_embed) in slide_plan if needs_embed]
    total_tiles_to_embed = sum(len(c) for c in tiles_to_embed)
    n_slides_to_embed = len(tiles_to_embed)
    if n_slides_to_embed:
        print(f"{n_slides_to_embed} slide(s) need embedding through H-optimus-0: "
              f"{total_tiles_to_embed} tiles total on device={device}. This is the slow step "
              "(see this file's BEGINNER'S GUIDE) -- a live ETA will print as it goes; Ctrl-C anytime to stop.")

    # ---- Pass 2: actually embed whatever needs it, with a live ETA ----------
    model = None
    tiles_done = 0          # cumulative tiles embedded so far, across all slides this run
    embed_clock_start = None  # wall-clock time.time() of the very first batch embedded this run
    last_eta_print = 0.0     # time.time() of the last ETA line we printed, for throttling

    def on_batch_embedded(n_tiles_in_batch):
        """Called by embed_slide() after every batch; prints a throttled running ETA."""
        nonlocal tiles_done, last_eta_print
        tiles_done += n_tiles_in_batch
        now = time.time()
        # Throttle to ~ once every 10 real seconds so this doesn't spam the
        # console on a fast machine/GPU where batches finish in milliseconds.
        if now - last_eta_print < 10.0 and tiles_done < total_tiles_to_embed:
            return
        last_eta_print = now
        elapsed = now - embed_clock_start
        rate = elapsed / tiles_done  # seconds per tile, averaged over everything embedded so far this run
        remaining = total_tiles_to_embed - tiles_done
        eta_sec = remaining * rate
        print(f"  ...{tiles_done}/{total_tiles_to_embed} tiles embedded "
              f"({rate:.2f}s/tile avg) -- ETA for remaining {remaining} tiles: "
              f"~{eta_sec / 60:.1f} min ({eta_sec / 3600:.2f} h)")

    X, y, names = [], [], []
    for slide_name, lab, tdir, meta, coords, out, needs_embed in slide_plan:
        if needs_embed:
            if model is None:
                # Only pay the (slow, ~30s+) H-optimus-0 load cost the first
                # time we actually need to embed something in this run.
                print("loading H-optimus-0 (once, reused across all slides)...")
                model = load_h_optimus_0(device)
            if embed_clock_start is None:
                embed_clock_start = time.time()  # start the ETA clock on the very first real batch

            # The tiling script recorded the tile size at its native scan
            # resolution ("native_mpp") and the resolution we actually wanted
            # ("target_mpp", e.g. 1.0 micron-per-pixel). This converts our
            # desired 224px tile size back into "how many level-0 pixels wide
            # is that", accounting for that resolution difference.
            step_l0 = round(224 * meta["target_mpp"] / meta["native_mpp"])

            t_slide_start = time.time()
            e = embed_slide(slides_dir / slide_name, coords, meta["level"],
                            meta["level_downsample"], step_l0, device, model,
                            progress_cb=on_batch_embedded)
            slide_dt = time.time() - t_slide_start
            print(f"embedded {slide_name}: {len(coords)} tiles in {slide_dt:.0f}s "
                  f"({slide_dt / max(1, len(coords)):.2f}s/tile)")

            # Build the [level, x, y, ...1536 embedding values...] rows and
            # save them to disk so future runs can skip re-embedding.
            lvl = np.full((len(e), 1), meta["level"], np.float32)
            np.save(out, np.concatenate([lvl, coords.astype(np.float32), e], axis=1))

        # Whether we just wrote `out` or it already existed, load it back.
        # `mmap_mode="r"` memory-maps the file instead of reading it all into
        # RAM immediately — cheap even for many/large cached files.
        arr = np.load(out, mmap_mode="r").astype(np.float32)
        X.append(np.array(arr[:, 3:]))   # drop the first 3 bookkeeping columns
        y.append(float(lab))
        names.append(slide_name)
    return X, np.array(y), names


def build_features_from_dir(features_dir, labels, max_tiles=5000):
    """Load ALREADY-embedded (N,1539) features.npy files (no H-optimus-0 needed here).

    This is the fast path used when you have the paper's own precomputed
    HuggingFace feature dump instead of raw .svs files — skips tiling AND
    embedding entirely, going straight to "load numbers off disk".

    Returns (X, y, names) — see build_features_from_tiles()'s docstring for
    the same "names can be shorter than labels" caveat.
    """
    features_dir = Path(features_dir)
    X, y, names = [], [], []
    for slide_name, lab in labels.items():
        p = features_dir / slide_name / "features.npy"
        if not p.exists():
            p = features_dir / f"{slide_name}.npy"   # alternate on-disk layout
        if not p.exists():
            print(f"skip {slide_name}: no features.npy")
            continue
        # [:max_tiles] caps how many tile-rows we use per slide (the paper
        # caps at 5000 tiles/slide for speed/memory — see docs).
        arr = np.load(p, mmap_mode="r").astype(np.float32)[:max_tiles]
        X.append(np.array(arr[:, 3:]))   # same "drop first 3 columns" as above
        y.append(float(lab))
        names.append(slide_name)
    return X, np.array(y), names


def train_one(Xtr, ytr, Xva, yva, device, args, save_path, n_extreme=N_EXTREME):
    """Train ONE Chowder model on a train/validation split and save its checkpoint.

    Xtr/Xva: lists of (n_tiles_i, 1536) numpy arrays (training / validation slides)
    ytr/yva: matching (n_slides,) arrays of 0.0/1.0 labels
    save_path: where to write the trained model's weights (a .pt file)

    Returns the validation-set AUC (Area Under the ROC Curve — a standard
    0.5=random-guessing to 1.0=perfect score for binary classifiers).
    """
    # Chowder's ExtremeLayer wants to pick `n_extreme` top-scoring AND
    # `n_extreme` bottom-scoring tiles (2*n_extreme total) from EVERY slide.
    # If our smallest slide has fewer tiles than that, there aren't enough
    # tiles to pick from, so we shrink n_extreme to fit (only matters for
    # small demo/smoke datasets — real runs have thousands of tiles/slide).
    min_tiles = min(len(x) for x in Xtr + Xva)
    if min_tiles < 2 * n_extreme:
        n_extreme = max(1, min_tiles // 2)
        print(f"note: n_extreme reduced to {n_extreme} (smallest slide has {min_tiles} tiles)")

    # A `Dataset` is just an object PyTorch knows how to index (`dataset[i]`
    # returns one training example). `SlideFeaturesDataset` (from fgfr3mut/utils.py)
    # wraps our list-of-arrays + labels into that interface.
    dtr = SlideFeaturesDataset(Xtr, ytr)
    dva = SlideFeaturesDataset(Xva, yva)

    # A `DataLoader` wraps a Dataset and automatically produces shuffled
    # mini-batches of size `batch_size` for us during training.
    #
    # PROBLEM: different slides have different numbers of tiles (a "ragged"
    # / variable-length batch), but PyTorch tensors in one batch must all be
    # the same shape. `pad_collate_fn` solves this by padding every slide in
    # a batch up to the length of the longest slide in that batch (with
    # zeros) and also returning a boolean "mask" tensor marking which rows
    # are real tiles vs. padding, so the model knows to ignore the padding.
    ltr = DataLoader(dtr, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate_fn)
    lva = DataLoader(dva, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate_fn)

    # Build a fresh Chowder model (random initial weights) and move it to
    # the chosen device. Unlike H-optimus-0, THIS model IS trained below.
    model = Chowder(in_features=IN_FEATURES, n_extreme=n_extreme).to(device)

    # Adam is a popular optimization algorithm that decides HOW to update
    # each of the model's learnable numbers, given their gradients, on every
    # training step. `lr` (learning rate) controls how big each update step
    # is; `weight_decay` is a mild penalty against very large weights (helps
    # avoid overfitting).
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Class imbalance handling: if MUT slides are rarer than WT slides (or
    # vice versa) in this fold's training set, we upweight the rarer class's
    # loss contribution so the model doesn't just learn to always predict the
    # majority class. pos_weight = (#negatives / #positives).
    n_pos = float((ytr == 1).sum())
    pos_weight = torch.tensor([(len(ytr) - n_pos) / max(1.0, n_pos)]).to(device)

    # BCEWithLogitsLoss = "Binary Cross-Entropy with Logits": the standard
    # loss function for binary (yes/no) classification. It internally applies
    # sigmoid to the model's raw logit AND computes cross-entropy in one
    # numerically-stable step (more stable than doing sigmoid, then BCE,
    # as two separate operations).
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ---- THE TRAINING LOOP ----------------------------------------------
    # We show the model the entire training set `args.epochs` times over
    # (one full pass through the data = one "epoch"), and within each epoch
    # we process it in shuffled mini-batches.
    for _ in range(args.epochs):
        model.train()   # enable training-mode behavior (see BEGINNER'S GUIDE)
        for f, mk, lb in ltr:   # f=padded features, mk=padding mask, lb=labels
            # Move this batch's tensors onto the training device.
            f, mk, lb = f.to(device), mk.to(device), lb.float().to(device)

            # 1) Reset gradients from the previous step to zero (PyTorch
            #    accumulates gradients by default, so we must clear them
            #    before computing new ones for this batch).
            opt.zero_grad()

            # 2) Forward pass: run the batch of (padded) tile-feature bags
            #    through Chowder to get one raw logit per slide. The model
            #    also returns per-tile scores (`_`, unused here).
            logit, _ = model(f, mask=mk)

            # 3) Compute the loss (how wrong were these predictions?) and
            #    call .backward() to compute gradients for every learnable
            #    parameter via backpropagation.
            #    .squeeze(1) just removes a redundant size-1 dimension so the
            #    shapes match what BCEWithLogitsLoss expects.
            crit(logit.squeeze(1), lb.squeeze(1)).backward()

            # 4) Apply the computed gradients to actually update the model's
            #    weights (this is where "learning" happens).
            opt.step()

    # ---- EVALUATION ON THE HELD-OUT VALIDATION SLIDES --------------------
    model.eval()   # disable training-mode behavior
    P, T = [], []   # P = predicted probabilities, T = true labels
    with torch.inference_mode():   # no gradients needed, just inference
        for f, mk, lb in lva:
            logit, _ = model(f.to(device), mask=mk.to(device))
            # sigmoid() squashes the raw logit into a 0..1 probability.
            P += torch.sigmoid(logit).cpu().flatten().tolist()
            T += lb.flatten().tolist()

    # AUC needs both classes present in the validation set to be meaningful;
    # with a tiny validation set that might not hold, hence the guard.
    auc = roc_auc_score(T, P) if len(set(T)) > 1 else float("nan")
    print(f"{save_path.name}: val AUC={auc:.3f} (n_tr={len(Xtr)}, n_va={len(Xva)})")

    # Save the trained model's weights to disk. `state_dict()` is a plain
    # dict of {layer_name: tensor_of_numbers} — everything needed to
    # recreate this exact trained model later with `model.load_state_dict(...)`.
    torch.save(model.state_dict(), save_path)
    return auc


def predict_ensemble(features, checkpoints, device, batch_size=16, n_extreme=N_EXTREME):
    """Run a list of slides through EVERY saved checkpoint and average the results.

    Why average multiple models ("ensembling")? Each checkpoint was trained
    on a different train/val split with a different random initialization,
    so each one makes slightly different mistakes. Averaging their predicted
    probabilities usually gives a more accurate and more stable final answer
    than trusting any single model. This matches how fgfr3mut/run_inference.py
    combines its 125 trained checkpoints at inference time.
    """
    # Labels are unknown at prediction time, so we pass dummy -1.0 placeholders
    # (SlideFeaturesDataset needs *some* label value per slide, even if unused).
    ds = SlideFeaturesDataset(features, [-1.0] * len(features))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)
    model = Chowder(in_features=IN_FEATURES, n_extreme=n_extreme).to(device)

    # One column per checkpoint; we'll average across columns at the end.
    probs = np.zeros((len(features), len(checkpoints)), np.float32)

    with torch.inference_mode():
        for ckpt_i, ckpt in enumerate(checkpoints):
            # Load this checkpoint's saved weights into our (currently
            # randomly-initialized) model, replacing its parameters in place.
            # weights_only=True: only allow loading plain tensors (what
            # torch.save(model.state_dict(), ...) actually wrote) rather than
            # arbitrary pickled Python objects — safer, and silences a
            # deprecation warning about the old, less-safe default.
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            model.eval()
            i = 0
            for f, mk, _ in dl:
                logit, _ = model(f.to(device), mask=mk.to(device))
                p = torch.sigmoid(logit).cpu().numpy().flatten()
                probs[i:i + len(p), ckpt_i] = p
                i += len(p)

    # Average each slide's probability across all checkpoints (axis=1 = across columns).
    return probs.mean(axis=1)


def main():
    # argparse turns command-line flags like `--epochs 30` into `args.epochs == 30`.
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # A "mutually exclusive group" means the user must pick AT MOST ONE of
    # --features_dir / --tiles_root (they're two different ways to get input
    # data — doesn't make sense to specify both).
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--features_dir", default=None, help="precomputed features/<slide>.svs/features.npy")
    src.add_argument("--tiles_root", default=None, help="output of fgfr3mut_tile_wsi.py")
    ap.add_argument("--slides_dir", default=None)
    ap.add_argument("--labels_csv", default=None, help="CSV with slide,label")
    ap.add_argument("--feat_dir", default="./results/features_cache", help="cache dir for embedded (N,1539) features (default: ./results/features_cache)")
    ap.add_argument("--out_dir", default="./results/models_trained", help="where trained checkpoints are saved (default: ./results/models_trained)")
    ap.add_argument("--epochs", type=int, default=30)        # how many full passes over the training data
    ap.add_argument("--lr", type=float, default=2e-4)        # Adam's learning rate (step size)
    ap.add_argument("--weight_decay", type=float, default=0.0)  # Adam's L2 regularization strength
    ap.add_argument("--batch_size", type=int, default=16)     # slides per mini-batch
    ap.add_argument("--splits", type=int, default=5)          # number of cross-validation folds
    ap.add_argument("--max_tiles", type=int, default=5000)    # cap on tiles/slide (only --features_dir path)
    ap.add_argument("--device", default="auto", help="auto (default: cuda:0 if GPU available else cpu), cpu, or cuda:0/cuda:1 (falls back to cpu with a warning if no GPU)")
    ap.add_argument("--smoke_test", action="store_true", help="synthetic-data end-to-end check, no SVS needed")
    args = ap.parse_args()
    args.device = resolve_device(args.device)
    print(f"Using device: {args.device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)   # make sure the checkpoint folder exists

    if args.smoke_test:
        # ---- SYNTHETIC DATA PATH ------------------------------------------
        # Generates fake (N, 1536) "feature" matrices with random numbers,
        # just to exercise the full training loop's code paths (Dataset,
        # DataLoader, Chowder forward/backward, checkpoint saving, AUC calc)
        # without needing any real .svs files, GPU, or downloaded models.
        # Useful as a quick "did I break something" sanity check.
        rng = np.random.default_rng(0)   # a seeded random-number generator (reproducible)
        X = [rng.normal(size=(40 + 10 * (i % 3), IN_FEATURES)).astype(np.float32) for i in range(8)]
        for i in range(4):  # nudge half the fake slides' features upward...
            X[i + 4] += 1.5   # ...so the two classes are separable and AUC is meaningful
        y = np.array([0] * 4 + [1] * 4, dtype=float)   # 4 fake WT slides, 4 fake MUT slides
        print(f"smoke: {len(X)} slides, dim={X[0].shape}")
    else:
        # ---- REAL DATA PATH -------------------------------------------------
        assert args.labels_csv, "need --labels_csv (or --smoke_test)"
        labels = load_labels(args.labels_csv)
        if args.features_dir:
            # Fast path: features already embedded, just load numpy arrays.
            X, y, _names = build_features_from_dir(args.features_dir, labels, args.max_tiles)
        else:
            # Slow path: still need to run tiles through H-optimus-0.
            assert args.tiles_root and args.slides_dir, "need --tiles_root + --slides_dir or --features_dir"
            X, y, _names = build_features_from_tiles(args.tiles_root, args.slides_dir,
                                             args.feat_dir, labels, args.device)
        print(f"loaded: {len(X)} slides, {int((y == 1).sum())} MUT / {int((y == 0).sum())} WT")
        assert len(X) >= 2 and len(set(y.tolist())) > 1, "need >=2 slides with both classes"

    # ---- K-FOLD CROSS-VALIDATION SETUP --------------------------------------
    # StratifiedKFold splits our slides into `n_splits` groups ("folds"),
    # keeping the MUT/WT ratio roughly the same in every fold (that's what
    # "stratified" means — as opposed to a totally random split, which could
    # accidentally put almost all the MUT slides in one fold). Each fold in
    # turn gets used as the validation set while the rest train the model, so
    # we end up training `n_splits` independent models total (an ensemble).
    #
    # StratifiedKFold requires n_splits <= (the number of slides in the SMALLER
    # class), otherwise it can't guarantee that ratio in every fold, so we cap
    # the requested --splits down to fit whatever data we actually loaded.
    min_class_count = int(min(np.bincount(y.astype(int))))
    n_splits = min(args.splits, min_class_count)
    if n_splits < args.splits:
        print(f"note: --splits reduced {args.splits} -> {n_splits} (smallest class has {min_class_count} slides)")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    ckpts = []
    # skf.split(...) yields (train_indices, validation_indices) once per fold.
    # We pass np.zeros(len(y)) as a dummy "X" because StratifiedKFold only
    # actually looks at `y` to decide the split — it doesn't touch the features.
    for k, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        ckpt = out_dir / f"model_{k}_{args.epochs}_ep.pt"
        train_one([X[i] for i in tr], y[tr], [X[i] for i in va], y[va],
                  args.device, args, ckpt)
        ckpts.append(ckpt)

    print(f"saved {len(ckpts)} checkpoints -> {out_dir}")
    print("Inference: predict_ensemble() = mean(sigmoid(logit)) over checkpoints.")


if __name__ == "__main__":
    # This guard means `main()` only runs when this file is executed directly
    # (`python train_fgfr3mut.py ...`), not when it's imported from another
    # script (e.g. `import train_fgfr3mut` to reuse `predict_ensemble`).
    main()
