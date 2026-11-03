#!/usr/bin/env python3
"""Build synthetic features + labels that EXACTLY match train_fgfr3mut.py --smoke_test.

WHAT THIS IS FOR
----------------
--smoke_test makes train_fgfr3mut.py invent 10 separable slides (5 WT / 5 MUT =
the paper's 5-per-class minimum) instead of reading any SVS, so the real recipe
can be exercised without data:

    25 repeats x 5 folds x 30 epochs = 125 checkpoints  (the paper's layout)

This script writes the SAME synthetic slides as (N, 1539) feature files — the
layout infer_fgfr3mut.py --features_dir expects ([level, x, y, <1536 values>],
same np.random.default_rng(0) construction training uses) — plus labels.csv, so
the inference half of the pipeline (manifest read, split_<repeat>/ checkpoint
discovery, out-of-fold over 25 repeats, patient-level metrics, calibrated
threshold, predictions.csv/metrics.json/ROC) can be run end to end too:

    python scripts/fgfr3mut_training/make_smoke_features.py
    python scripts/fgfr3mut_training/train_fgfr3mut.py --smoke_test \
        --splits 5 --repeats 25 --epochs 30 --out_dir ./results/smoke_pipeline/models
    python scripts/fgfr3mut_training/infer_fgfr3mut.py \
        --features_dir ./results/smoke_pipeline/features \
        --labels_csv   ./results/smoke_pipeline/labels.csv \
        --checkpoints_dir ./results/smoke_pipeline/models \
        --oof --out_csv ./results/smoke_pipeline/predictions.csv \
        --metrics_out ./results/smoke_pipeline/metrics.json \
        --roc_out ./results/smoke_pipeline/roc_curve.png

Metrics on this data prove the MECHANICS, never the biology: it is synthetic.

It also writes one slide training never saw (``brand_new_slide.svs``) and a
labels file containing it (labels_with_new_slide.csv), to check infer's guard:
``--oof`` must refuse a slide no checkpoint was trained without and point at the
default ensemble mode instead — that is exactly the situation of scoring the
untouched test slides in image/others.

Usage:
  python scripts/fgfr3mut_training/make_smoke_features.py [--out ./results/smoke_pipeline]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

IN_FEATURES = 1536      # H-optimus-0 embedding width (must match train's IN_FEATURES)
N_SLIDES = 10           # 5 WT + 5 MUT -> 5-fold CV is possible, like the paper


def synthetic_bags() -> list[np.ndarray]:
    """The exact data train_fgfr3mut.py --smoke_test invents (same seed, same shapes)."""
    rng = np.random.default_rng(0)
    X = [rng.normal(size=(40 + 10 * (i % 3), IN_FEATURES)).astype(np.float32)
         for i in range(N_SLIDES)]
    for i in range(N_SLIDES // 2):
        X[i + N_SLIDES // 2] += 1.5      # make the two classes separable
    return X


def save_bag(path: Path, arr: np.ndarray) -> None:
    """(n_tiles, 1536) -> (n_tiles, 1539) file: [level, x, y, <features>]."""
    n = len(arr)
    full = np.zeros((n, 3 + IN_FEATURES), np.float32)
    full[:, 1] = np.arange(n)             # level=0, x, y (coords are irrelevant here)
    full[:, 3:] = arr
    np.save(path, full)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="./results/smoke_pipeline",
                    help="where to write features/, labels.csv, labels_with_new_slide.csv "
                         "(default: ./results/smoke_pipeline)")
    args = ap.parse_args()

    out = Path(args.out)
    feat = out / "features"
    feat.mkdir(parents=True, exist_ok=True)

    X = synthetic_bags()
    labels = [0] * (N_SLIDES // 2) + [1] * (N_SLIDES // 2)

    for i, arr in enumerate(X):
        save_bag(feat / f"smoke_slide_{i}.svs.npy", arr)

    # A slide with features that no checkpoint was ever trained on (see module doc).
    save_bag(feat / "brand_new_slide.svs.npy",
             np.random.default_rng(1).normal(size=(50, IN_FEATURES)).astype(np.float32))

    with open(out / "labels.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["slide", "label"])
        for i, lab in enumerate(labels):
            w.writerow([f"smoke_slide_{i}.svs", lab])

    with open(out / "labels_with_new_slide.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["slide", "label"])
        for i, lab in enumerate(labels):
            w.writerow([f"smoke_slide_{i}.svs", lab])
        w.writerow(["brand_new_slide.svs", 1])

    n_feat = len(list(feat.glob("*.npy")))
    print(f"wrote {n_feat} feature file(s) -> {feat}")
    print(f"wrote {out / 'labels.csv'}  ({N_SLIDES} training slides, {sum(labels)} MUT / "
          f"{N_SLIDES - sum(labels)} WT)")
    print(f"wrote {out / 'labels_with_new_slide.csv'}  (+1 never-trained-on slide for the "
          "--oof guard check)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
