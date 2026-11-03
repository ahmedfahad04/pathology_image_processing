"""FGFR3MUT training: tiled WSIs -> frozen H-optimus-0 features -> Chowder MIL.

Input modes:
  --tiles_root + --slides_dir : embed tiles with H-optimus-0, then train.
  --features_dir             : load precomputed (N, 1539) features.npy files.

Output: one Chowder checkpoint per (repeat, fold) under --out_dir, plus a
``train_manifest.json`` describing exactly how the folds were built (infer_
fgfr3mut.py reads it back so its --oof mode can rebuild the SAME split).

CHECKPOINT LAYOUT (mirrors the paper's own ``models/split_<r>/model_<k>_30_ep.pt``):
  --repeats 1 (default)  ->  model_<fold>_<epochs>_ep.pt            (flat, as before)
  --repeats N            ->  split_<repeat>/model_<fold>_<epochs>_ep.pt
  N=25 and --splits 5 is the FGFR3MUT paper's recipe: 25 x 5 = 125 models,
  30 epochs each.

HOW THE FOLDS ARE BUILT (see make_folds(), the single source of truth shared
with infer_fgfr3mut.py's --oof mode):
  * Patient-grouped: every slide of one patient (TCGA-XX-XXXX) stays in the SAME
    fold, because all of a patient's slides share the patient's genotype —
    splitting them across train/val would leak the answer. This is the paper's
    own unit (its metrics are computed per patient).
  * Stratified by label, shuffled with ``random_state = repeat`` so repeat 0
    reproduces the historical split and every later repeat is a different
    partition of the same slides.
  * n_splits = min(--splits, smallest class count in slides AND in patients),
    so the script never silently builds an invalid fold; if that is < 2 it stops
    with an actionable error instead of training nonsense.

Chowder is a multiple-instance learning (MIL) model: it scores every tile,
keeps the top-100 and bottom-100 scores, and predicts one label per slide.
The tile encoder (H-optimus-0) is frozen; only Chowder is trained.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Dependency: vendored fgfr3mut package (Chowder architecture + dataset utils)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
FGFR3MUT_DIR = SCRIPT_DIR / "fgfr3mut"
FGFR3MUT_REPO_URL = "https://github.com/PABannier/fgfr3mut"


def ensure_fgfr3mut_repo():
    """Clone the fgfr3mut dependency if it is not already present."""
    if (FGFR3MUT_DIR / "fgfr3mut" / "chowder.py").exists():
        return

    print(f"fgfr3mut/ not found at {FGFR3MUT_DIR} — cloning it from {FGFR3MUT_REPO_URL} ...")
    try:
        subprocess.run(["git", "clone", FGFR3MUT_REPO_URL, str(FGFR3MUT_DIR)], check=True)
    except FileNotFoundError:
        raise RuntimeError(
            "git does not appear to be installed (or is not on PATH). Install git, "
            f"or manually clone {FGFR3MUT_REPO_URL} into {FGFR3MUT_DIR}"
        ) from None
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"git clone of {FGFR3MUT_REPO_URL} failed (exit code {e.returncode}). "
            f"Check your network connection, or manually clone it into {FGFR3MUT_DIR}"
        ) from None

    if not (FGFR3MUT_DIR / "fgfr3mut" / "chowder.py").exists():
        raise RuntimeError(
            f"cloned {FGFR3MUT_REPO_URL} into {FGFR3MUT_DIR}, but "
            "fgfr3mut/fgfr3mut/chowder.py still doesn't exist — has the upstream repo's layout changed?"
        )
    print("done.")


ensure_fgfr3mut_repo()
if str(FGFR3MUT_DIR) not in sys.path:
    sys.path.insert(0, str(FGFR3MUT_DIR))

from fgfr3mut.chowder import Chowder  # noqa: E402
from fgfr3mut.utils import SlideFeaturesDataset, pad_collate_fn  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
H0_MEAN = np.array([0.707223, 0.578729, 0.703617], dtype=np.float32)
H0_STD = np.array([0.211883, 0.230117, 0.177517], dtype=np.float32)
IN_FEATURES = 1536   # H-optimus-0 embedding size
N_EXTREME = 100      # Chowder keeps top-100 + bottom-100 tiles per slide


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def resolve_device(requested="auto"):
    """Resolve 'auto'/'cuda' requests to an available device."""
    req = (requested or "auto").strip().lower()
    if req == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if req.startswith("cuda") and not torch.cuda.is_available():
        print(f"warning: --device={requested} requested but no CUDA GPU is available; falling back to cpu")
        return "cpu"
    return requested


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def load_h_optimus_0(device):
    """Load the frozen H-optimus-0 feature extractor."""
    import timm

    return timm.create_model(
        "hf-hub:bioptimus/H-optimus-0", pretrained=True,
        init_values=1e-5, dynamic_img_size=False,
    ).to(device).eval()


def embed_slide(svs_path, coords, level, level_ds, step_l0, device, model, batch=8, progress_cb=None):
    """Embed one slide's kept tiles into an (N_tiles, 1536) array."""
    import cv2
    import openslide

    slide = openslide.OpenSlide(str(svs_path))
    embs = []
    cur = []

    def flush():
        # Normalize with H-optimus-0 statistics and convert NHWC -> NCHW.
        x = (np.stack(cur).astype(np.float32) / 255.0 - H0_MEAN) / H0_STD
        x = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)

        # Frozen inference: no autograd graph.
        with torch.inference_mode():
            embs.append(model(x).cpu().numpy().astype(np.float32))

        n_this_batch = len(cur)
        cur.clear()
        if progress_cb is not None:
            progress_cb(n_this_batch)

    for x0, y0 in coords:
        # Tile size in the selected pyramid level's pixel units.
        s = max(1, round(step_l0 / level_ds))
        px = np.array(slide.read_region((int(x0), int(y0)), level, (s, s)).convert("RGB"))

        # H-optimus-0 expects exactly 224x224 input.
        if px.shape[:2] != (224, 224):
            px = cv2.resize(px, (224, 224), interpolation=cv2.INTER_LINEAR)

        cur.append(px)
        if len(cur) == batch:
            flush()
    if cur:
        flush()

    slide.close()
    return np.concatenate(embs, axis=0)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_labels(labels_csv):
    """Read labels.csv into a slide_name -> 0.0/1.0 mapping."""
    import pandas as pd

    df = pd.read_csv(labels_csv)
    assert {"slide", "label"} <= set(df.columns), "labels.csv needs columns: slide,label"
    return df.set_index("slide")["label"].astype(float)


# ---------------------------------------------------------------------------
# Fold layout — shared by train_fgfr3mut.py and infer_fgfr3mut.py (--oof)
# ---------------------------------------------------------------------------
def patient_id(slide):
    """``TCGA-XX-XXXX-01Z-00-DX1.<uuid>.svs`` -> ``TCGA-XX-XXXX``.

    Every slide of one patient carries that patient's FGFR3 status, so slides
    must be grouped by patient when splitting — otherwise the model trains on
    one slide of a patient and is evaluated on another of the SAME patient
    (textbook leakage). Non-TCGA file names map to their own stem, i.e. one
    slide counts as one patient.
    """
    name = Path(str(slide)).name
    m = re.match(r"(TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4})", name)
    if m:
        return m.group(1)
    return name[: -len(".svs")] if name.lower().endswith(".svs") else name


def make_folds(y, splits, repeat=0, groups=None):
    """Rebuild the CV fold layout. SINGLE SOURCE OF TRUTH for train + infer --oof.

    Parameters
    ----------
    y      : (n,) label array (1.0 = MUT, 0.0 = WT), in slide order.
    splits : the --splits value the user asked for (both scripts default to 5,
             the paper's fold count).
    repeat : which repeat to build (0-based). Fold membership depends on it
             (``random_state = repeat``), so repeat 0 == the historical split.
    groups : per-slide patient id (from ``patient_id()``) or None for
             slide-level splitting (``--split_unit slide``).

    Returns
    -------
    (n_splits, [(train_idx, val_idx), ...])

    Raises
    ------
    ValueError with an actionable message when fewer than 2 folds are possible
    (needs >= 2 slides AND >= 2 patients in the smallest class; >= 5 of each for
    the paper's 5-fold recipe). Callers surface this as a clean error instead of
    letting sklearn raise its own cryptic message halfway through training.
    """
    y = np.asarray(y)
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("no slides to split")
    classes, counts = np.unique(y.astype(int), return_counts=True)
    if len(classes) < 2:
        raise ValueError(
            f"only one class present ({classes.tolist()}) — need slides of BOTH classes"
        )

    # A fold's held-out set must contain >=1 member of every class, so the fold
    # count can never exceed the smallest class — counted in patients too when
    # grouping, since a whole patient (all its slides) lands in one fold.
    limit = int(counts.min())
    unit = "slide"
    if groups is not None:
        groups = np.asarray(groups)
        gpc = [len(np.unique(groups[y.astype(int) == c])) for c in classes]
        limit = min(limit, int(min(gpc)))
        unit = "patient"

    n_splits = min(int(splits), limit)
    if n_splits < 2:
        detail = (
            f"smallest class has {limit} {unit}(s)"
            + (f" across classes {dict(zip(classes.tolist(), [int(c) for c in counts]))} slides"
               if groups is None else "")
        )
        raise ValueError(
            f"cannot cross-validate: {detail}, so not even 2 folds are possible. "
            "Need >= 2 slides AND >= 2 patients per class to train at all, and >= 5 per "
            "class for the paper's 5-fold x 25-repeat (125-model) recipe. Add slides of "
            "the missing class."
        )

    if groups is not None:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=repeat)
        folds = list(sgkf.split(np.zeros(len(y)), y, groups))
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=repeat)
        folds = list(skf.split(np.zeros(len(y)), y))
    return n_splits, folds


def build_features_from_tiles(tiles_root, slides_dir, feat_dir, labels, device):
    """Embed tiles and cache each slide as an (N, 1539) .npy file.

    Row layout: [level, x, y, <1536 embedding values>].
    Returns (X, y, names), where X is a list of (n_tiles_i, 1536) arrays.
    """
    tiles_root, slides_dir, feat_dir = Path(tiles_root), Path(slides_dir), Path(feat_dir)
    feat_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: decide what is cached and what needs embedding.
    slide_plan = []
    for slide_name, lab in labels.items():
        tdir = tiles_root / slide_name
        meta = json.load(open(tdir / "metadata.json"))
        coords = np.load(tdir / "coords.npy")
        if len(coords) == 0:
            print(f"skip {slide_name}: 0 kept tiles")
            continue

        out = feat_dir / f"{slide_name}.npy"
        needs_embed = True
        if out.exists():
            cached = np.load(out, mmap_mode="r")
            cached_coords = np.asarray(cached[:, 1:3], dtype=np.int64)
            # Re-embed if the cache came from a different tiling run.
            if cached_coords.shape == coords.shape and np.array_equal(cached_coords, coords.astype(np.int64)):
                needs_embed = False
            else:
                print(f"note: cached features for {slide_name} ({cached_coords.shape[0]} tiles) don't match "
                      f"current coords.npy ({len(coords)} tiles) -- re-embedding.")
        slide_plan.append((slide_name, lab, tdir, meta, coords, out, needs_embed))

    tiles_to_embed = [coords for (_name, _lab, _tdir, _meta, coords, _out, needs_embed) in slide_plan if needs_embed]
    total_tiles_to_embed = sum(len(c) for c in tiles_to_embed)
    n_slides_to_embed = len(tiles_to_embed)
    if n_slides_to_embed:
        print(f"{n_slides_to_embed} slide(s) need embedding through H-optimus-0: "
              f"{total_tiles_to_embed} tiles total on device={device}.")

    # Pass 2: embed the missing slides, printing a throttled ETA.
    model = None
    tiles_done = 0
    embed_clock_start = None
    last_eta_print = 0.0

    def on_batch_embedded(n_tiles_in_batch):
        nonlocal tiles_done, last_eta_print
        tiles_done += n_tiles_in_batch
        now = time.time()
        if now - last_eta_print < 10.0 and tiles_done < total_tiles_to_embed:
            return
        last_eta_print = now
        elapsed = now - embed_clock_start
        rate = elapsed / tiles_done
        remaining = total_tiles_to_embed - tiles_done
        eta_sec = remaining * rate
        print(f"  ...{tiles_done}/{total_tiles_to_embed} tiles embedded "
              f"({rate:.2f}s/tile avg) -- ETA for remaining {remaining} tiles: "
              f"~{eta_sec / 60:.1f} min ({eta_sec / 3600:.2f} h)")

    X, y, names = [], [], []
    for slide_name, lab, tdir, meta, coords, out, needs_embed in slide_plan:
        if needs_embed:
            if model is None:
                print("loading H-optimus-0 (once, reused across all slides)...")
                model = load_h_optimus_0(device)
            if embed_clock_start is None:
                embed_clock_start = time.time()

            # Convert 224 px at target MPP back to level-0 pixels.
            step_l0 = round(224 * meta["target_mpp"] / meta["native_mpp"])

            t_slide_start = time.time()
            e = embed_slide(slides_dir / slide_name, coords, meta["level"],
                            meta["level_downsample"], step_l0, device, model,
                            progress_cb=on_batch_embedded)
            slide_dt = time.time() - t_slide_start
            print(f"embedded {slide_name}: {len(coords)} tiles in {slide_dt:.0f}s "
                  f"({slide_dt / max(1, len(coords)):.2f}s/tile)")

            lvl = np.full((len(e), 1), meta["level"], np.float32)
            np.save(out, np.concatenate([lvl, coords.astype(np.float32), e], axis=1))

        arr = np.load(out, mmap_mode="r").astype(np.float32)
        X.append(np.array(arr[:, 3:]))   # drop level, x, y
        y.append(float(lab))
        names.append(slide_name)
    return X, np.array(y), names


def build_features_from_dir(features_dir, labels, max_tiles=5000):
    """Load precomputed (N, 1539) features.npy files; no ViT needed."""
    features_dir = Path(features_dir)
    X, y, names = [], [], []
    for slide_name, lab in labels.items():
        p = features_dir / slide_name / "features.npy"
        if not p.exists():
            p = features_dir / f"{slide_name}.npy"
        if not p.exists():
            print(f"skip {slide_name}: no features.npy")
            continue
        arr = np.load(p, mmap_mode="r").astype(np.float32)[:max_tiles]
        X.append(np.array(arr[:, 3:]))
        y.append(float(lab))
        names.append(slide_name)
    return X, np.array(y), names


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_one(Xtr, ytr, Xva, yva, device, args, save_path, n_extreme=N_EXTREME):
    """Train one Chowder model; returns (validation AUC, n_extreme actually used).

    The returned n_extreme matters: Chowder's ExtremeLayer size depends on it,
    so a checkpoint trained with a reduced value can only be loaded again with
    the matching --n_extreme (infer_fgfr3mut.py reads it from the manifest).
    """
    # Ensure every slide has at least 2 * n_extreme tiles.
    min_tiles = min(len(x) for x in Xtr + Xva)
    if min_tiles < 2 * n_extreme:
        n_extreme = max(1, min_tiles // 2)
        print(f"note: n_extreme reduced to {n_extreme} (smallest slide has {min_tiles} tiles)")

    dtr = SlideFeaturesDataset(Xtr, ytr)
    dva = SlideFeaturesDataset(Xva, yva)

    # Bags have variable tile counts; pad_collate_fn pads them and returns a mask.
    ltr = DataLoader(dtr, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate_fn)
    lva = DataLoader(dva, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate_fn)

    # Trainable model: Chowder only (the ViT is not in this graph).
    model = Chowder(in_features=IN_FEATURES, n_extreme=n_extreme).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Upweight the minority class in this fold.
    n_pos = float((ytr == 1).sum())
    pos_weight = torch.tensor([(len(ytr) - n_pos) / max(1.0, n_pos)]).to(device)
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # One pass over the training set per epoch.
    for _ in range(args.epochs):
        model.train()
        for f, mk, lb in ltr:
            f, mk, lb = f.to(device), mk.to(device), lb.float().to(device)
            opt.zero_grad()
            logit, _ = model(f, mask=mk)
            crit(logit.squeeze(1), lb.squeeze(1)).backward()
            opt.step()

    # Validation: one probability per slide.
    model.eval()
    P, T = [], []
    with torch.inference_mode():
        for f, mk, lb in lva:
            logit, _ = model(f.to(device), mask=mk.to(device))
            P += torch.sigmoid(logit).cpu().flatten().tolist()
            T += lb.flatten().tolist()

    auc = roc_auc_score(T, P) if len(set(T)) > 1 else float("nan")
    print(f"{save_path.name}: val AUC={auc:.3f} (n_tr={len(Xtr)}, n_va={len(Xva)})")

    torch.save(model.state_dict(), save_path)
    return auc, n_extreme


def predict_ensemble(features, checkpoints, device, batch_size=16, n_extreme=N_EXTREME):
    """Average sigmoid probabilities across all checkpoints (one per slide)."""
    ds = SlideFeaturesDataset(features, [-1.0] * len(features))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)
    model = Chowder(in_features=IN_FEATURES, n_extreme=n_extreme).to(device)

    probs = np.zeros((len(features), len(checkpoints)), np.float32)
    with torch.inference_mode():
        for ckpt_i, ckpt in enumerate(checkpoints):
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            model.eval()
            i = 0
            for f, mk, _ in dl:
                logit, _ = model(f.to(device), mask=mk.to(device))
                p = torch.sigmoid(logit).cpu().numpy().flatten()
                probs[i:i + len(p), ckpt_i] = p
                i += len(p)

    return probs.mean(axis=1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--features_dir", default=None, help="precomputed features/<slide>.svs/features.npy")
    src.add_argument("--tiles_root", default=None, help="output of fgfr3mut_tile_wsi.py")
    ap.add_argument("--slides_dir", default=None)
    ap.add_argument("--labels_csv", default=None, help="CSV with slide,label")
    ap.add_argument("--feat_dir", default="./results/features_cache", help="cache dir for embedded (N,1539) features")
    ap.add_argument("--out_dir", default="./results/models_trained", help="where trained checkpoints are saved")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--splits", type=int, default=5,
                    help="folds per repeat (paper: 5). Capped at the smallest class, counted in "
                         "--split_unit units — see make_folds()")
    ap.add_argument("--repeats", type=int, default=1,
                    help="how many CV repeats to train (paper: 25, i.e. 25 x 5 = 125 models). Every repeat "
                         "re-splits the slides with random_state=repeat and writes split_<repeat>/model_<fold>_"
                         "<epochs>_ep.pt, matching the paper's own models/split_<r>/model_<k>_30_ep.pt layout")
    ap.add_argument("--split_unit", choices=("patient", "slide"), default="patient",
                    help="indivisible splitting unit: 'patient' (default — the paper's unit; all slides of one "
                         "patient share one genotype, so they must stay in the same fold) or 'slide' (old "
                         "behaviour, which leaks a patient across folds when it has 2+ slides)")
    ap.add_argument("--max_tiles", type=int, default=5000, help="cap on tiles/slide (only --features_dir path)")
    ap.add_argument("--device", default="auto", help="auto, cpu, or cuda:0/cuda:1")
    ap.add_argument("--smoke_test", action="store_true", help="synthetic-data end-to-end check, no SVS needed")
    args = ap.parse_args()
    args.device = resolve_device(args.device)
    print(f"Using device: {args.device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        # 10 synthetic slides, 5 WT + 5 MUT — deliberately the paper's minimum
        # for its real recipe (5 patients per class), so --smoke_test --splits 5
        # --repeats 25 builds the full 125-model layout without needing a SVS.
        rng = np.random.default_rng(0)
        X = [rng.normal(size=(40 + 10 * (i % 3), IN_FEATURES)).astype(np.float32)
             for i in range(10)]
        for i in range(5):
            X[i + 5] += 1.5   # make the two classes separable
        y = np.array([0] * 5 + [1] * 5, dtype=float)
        names = [f"smoke_slide_{i}.svs" for i in range(len(X))]
        print(f"smoke: {len(X)} slides, dim={X[0].shape}")
    else:
        assert args.labels_csv, "need --labels_csv (or --smoke_test)"
        labels = load_labels(args.labels_csv)
        if args.features_dir:
            X, y, names = build_features_from_dir(args.features_dir, labels, args.max_tiles)
        else:
            assert args.tiles_root and args.slides_dir, "need --tiles_root + --slides_dir or --features_dir"
            X, y, names = build_features_from_tiles(args.tiles_root, args.slides_dir,
                                                    args.feat_dir, labels, args.device)
        print(f"loaded: {len(X)} slides, {int((y == 1).sum())} MUT / {int((y == 0).sum())} WT")
        assert len(X) >= 2 and len(set(y.tolist())) > 1, "need >=2 slides with both classes"

    # Patient-level splitting (the paper's unit): a patient's slides share one
    # genotype, so they must never be spread across train/validation folds.
    groups = [patient_id(n) for n in names] if args.split_unit == "patient" else None
    if groups is not None and len(set(groups)) != len(names):
        print(f"note: {len(names)} slides = {len(set(groups))} patients — folds are grouped by patient")
    try:
        n_splits, _ = make_folds(y, args.splits, repeat=0, groups=groups)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")

    if n_splits < args.splits:
        print(f"note: --splits reduced {args.splits} -> {n_splits} folds "
              f"(capped by the smallest class, counted in {args.split_unit}s)")
    total = n_splits * args.repeats
    print(f"recipe: {args.split_unit}-level {n_splits}-fold CV x {args.repeats} repeat(s) "
          f"x {args.epochs} epochs = {total} model(s)")
    if n_splits < 5 or args.repeats < 25:
        print(f"note: the FGFR3MUT paper's recipe is 25 repeats x 5 folds = 125 models "
              f"(yours: {args.repeats} x {n_splits} = {total}) — 5-fold needs >= 5 patients per class")

    # One checkpoint per (repeat, fold). Repeat 0 with --repeats 1 keeps the
    # historical flat layout, so earlier runs stay loadable as they are.
    ckpt_records = []
    for r in range(args.repeats):
        _, folds = make_folds(y, args.splits, repeat=r, groups=groups)
        sub_dir = out_dir if args.repeats == 1 else out_dir / f"split_{r}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        for k, (tr, va) in enumerate(folds):
            # Distinct, reproducible init per model: same command -> same
            # checkpoints, repeat/fold -> different (so the ensemble members
            # really are independent models rather than clones).
            torch.manual_seed(100000 * r + 1000 * k + 7)
            ckpt = sub_dir / f"model_{k}_{args.epochs}_ep.pt"
            auc, n_extreme_used = train_one([X[i] for i in tr], y[tr],
                                            [X[i] for i in va], y[va],
                                            args.device, args, ckpt)
            ckpt_records.append({
                "path": str(ckpt.relative_to(out_dir)),
                "repeat": r, "fold": k,
                "n_extreme": int(n_extreme_used),
                "val_auc": None if not np.isfinite(auc) else round(float(auc), 4),
            })
        print(f"repeat {r + 1}/{args.repeats}: {len(folds)} checkpoint(s) -> {sub_dir}")

    # train_manifest.json: everything infer_fgfr3mut.py needs to rebuild this
    # exact split (so --oof pairs every slide with the right checkpoint) and to
    # know which --n_extreme each checkpoint was actually trained with.
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "labels_csv": "smoke_test" if args.smoke_test else str(args.labels_csv),
        "slides": list(names),
        "patients": [patient_id(n) for n in names],
        "labels": [int(v) for v in np.asarray(y).tolist()],
        "split_unit": args.split_unit,
        "splits_requested": int(args.splits),
        "n_splits": int(n_splits),
        "repeats": int(args.repeats),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "max_tiles": int(args.max_tiles),
        "seed_rule": {
            "fold_membership": "random_state = repeat "
                               "(StratifiedGroupKFold when split_unit=patient, else StratifiedKFold)",
            "model_init": "torch.manual_seed(100000*repeat + 1000*fold + 7)",
        },
        "checkpoints": ckpt_records,
        "n_extreme_values": sorted({c["n_extreme"] for c in ckpt_records}),
    }
    (out_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2))

    layout = " (split_<repeat>/ subfolders)" if args.repeats > 1 else ""
    print(f"saved {len(ckpt_records)} checkpoints -> {out_dir}{layout}")
    print(f"wrote {out_dir / 'train_manifest.json'} — infer_fgfr3mut.py reads it to rebuild this split")
    print("Inference: predict_ensemble() = mean(sigmoid(logit)) over checkpoints.")


if __name__ == "__main__":
    main()
