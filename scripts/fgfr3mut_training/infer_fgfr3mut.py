"""FGFR3MUT inference: run trained Chowder checkpoints on slides and check performance.

WHAT THIS SCRIPT DOES (for beginners — see train_fgfr3mut.py's own top-of-file
"BEGINNER'S GUIDE" for the full background on tiles / embeddings / MIL / Chowder)
------------------------------------------------------------------------------------
train_fgfr3mut.py trains several independent Chowder models (one per
cross-validation fold) and saves each one's learned weights as a checkpoint
file (results/models_trained/model_*.pt). Training already reports a
validation AUC for each individual fold, but that's only ONE model's score on
ONE held-out slice of data.

This script instead:
  1. Loads ALL of those saved checkpoints back into memory.
  2. Runs every requested slide through EVERY checkpoint (a forward pass only
     — no training, no gradients — see train_fgfr3mut.py's glossary for what
     "inference" / "eval mode" mean).
  3. Averages the checkpoints' predicted probabilities together per slide
     ("ensembling" — see predict_ensemble()'s docstring in train_fgfr3mut.py).
  4. If you give it known ground-truth labels (labels.csv), it also computes
     accuracy metrics so you can see how good the trained model actually is
     — not just "did training run". This includes both a simple fixed-
     threshold report (accuracy, confusion matrix @ --threshold, default
     0.5) AND the exact metric set the FGFR3MUT paper itself reports, computed
     the exact same way as the paper's own code (see WHICH METRICS below).
  5. Saves ALL of the above (not just prints them) to --metrics_out (default
     ./results/metrics.json) as JSON, plus the per-slide predictions to
     --out_csv (default ./results/predictions.csv), so you have a permanent
     record of "how good was this checkpoint set" instead of only scrollback.

This reuses the exact same data-loading + model code as train_fgfr3mut.py
(imported from it directly below) so predictions are guaranteed to be
computed the same way features were computed during training.

WHICH METRICS, AND WHERE THEY COME FROM
------------------------------------------
Two different reports are computed, because the paper and its own code use a
DIFFERENT (better) approach than "just threshold at 0.5":

  A) Simple fixed-threshold report (this script's own code):
     accuracy + confusion matrix (TN/FP/FN/TP) at whatever --threshold you
     pick (default 0.5). Easy to understand, but 0.5 is an arbitrary cutoff.

  B) Paper-style bootstrapped report (reuses fgfr3mut/bootstrap.py's own
     `compute_ci_with_bootstrap()` — the EXACT function the paper's own
     `run_inference.py` calls to produce Table 1 / Fig. 2's numbers):
       - AUC (Area Under the ROC Curve) and AP (Average Precision) — overall,
         threshold-independent ranking quality.
       - Instead of an arbitrary 0.5 cutoff, the paper picks its operating
         threshold as "the LOWEST prediction score that still catches at
         least `target_sens` (default 92%) of true MUT cases" (see the
         paper's Methods, "Performance assessment and statistical methods",
         and Table 1's "Operating threshold" column). This matters clinically:
         missing a treatable MUT patient (a false negative) is worse than
         over-testing a WT patient (a false positive), so the threshold is
         deliberately biased toward high sensitivity rather than raw accuracy.
       - Sensitivity (Se, = Recall = TPR), Specificity (Sp, = TNR), PPV
         (Positive Predictive Value = Precision), NPV (Negative Predictive
         Value) — computed at that chosen threshold.
       - 95% confidence intervals for every one of the above, via
         non-parametric bootstrap: resample the slides WITH REPLACEMENT
         1000 times (by default), recompute every metric on each resample,
         and take the 2.5th/97.5th percentiles across those 1000 values as
         the CI. This is exactly what the paper's Methods section describes
         ("Confidence intervals at 95% confidence level were obtained by
         bootstrapping experiment results with 1000 repeats").
     NOTE: with only a handful of slides (as in this repo's demo dataset),
     bootstrap CIs are statistically weak/wide — they're meaningful once you
     have dozens+ of slides, same as in the paper's own hundreds-of-patient
     validation cohorts. Pass --n_bootstrap 0 to skip this and only get (A).

Usage (run from the project root):
  # Evaluate the ensemble in results/models_trained/ against known labels,
  # reusing the tile features already cached during training:
  python scripts/fgfr3mut_training/infer_fgfr3mut.py \\
    --tiles_root ./results/tiles_out --slides_dir ./image \\
    --labels_csv ./results/labels.csv

  # Same, but reading precomputed (N,1539) features.npy files instead of tiles:
  python scripts/fgfr3mut_training/infer_fgfr3mut.py \\
    --features_dir ./output/data_fgfr3_mini/features \\
    --labels_csv ./results/labels.csv

  # Point at a different checkpoint folder / save predictions elsewhere:
  python scripts/fgfr3mut_training/infer_fgfr3mut.py \\
    --tiles_root ./results/tiles_out --slides_dir ./image \\
    --labels_csv ./results/labels.csv \\
    --checkpoints_dir ./results/models_trained --out_csv ./results/predictions.csv

Output: prints a per-slide prediction table + summary metrics to the console,
writes the same per-slide table to --out_csv (default:
./results/predictions.csv) with columns: slide,label,pred_prob,pred_label,
writes every computed metric to --metrics_out (default
./results/metrics.json), and — mirroring the paper's own Fig. 2a-c — plots
and saves an ROC curve to --roc_out (default ./results/roc_curve.png), with
AUC/NPV/PPV annotated the same way the paper's figure does.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

# ---------------------------------------------------------------------------------
# Reuse train_fgfr3mut.py's own functions instead of re-implementing them: this
# guarantees predictions are computed EXACTLY the same way features were built
# and embedded during training (same H-optimus-0 normalization, same tile
# feature caching, same Chowder architecture). We add this script's own folder
# to sys.path so a plain `import train_fgfr3mut` can find it as a module.
# ---------------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_fgfr3mut import (  # noqa: E402
    N_EXTREME,
    build_features_from_dir,
    build_features_from_tiles,
    load_labels,
    predict_ensemble,
    resolve_device,
)

# Importing train_fgfr3mut (above) already added the vendored fgfr3mut/ package
# folder to sys.path as a side effect (see its own SCRIPT_DIR search loop), so
# this import of the paper's OWN metrics/bootstrap code just works from here too.
from fgfr3mut.bootstrap import compute_ci_with_bootstrap  # noqa: E402
from loguru import logger  # noqa: E402  (bootstrap.py logs skipped resamples through this)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Same two ways of supplying tile features as train_fgfr3mut.py — see that
    # script's comments for the difference between the two modes.
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--features_dir", default=None, help="precomputed features/<slide>.svs/features.npy")
    src.add_argument("--tiles_root", default=None, help="output of fgfr3mut_tile_wsi.py")
    ap.add_argument("--slides_dir", default=None)
    ap.add_argument("--labels_csv", required=True, help="CSV with slide,label — the ground truth to score predictions against")
    ap.add_argument("--feat_dir", default="./results/features_cache", help="cache dir for embedded (N,1539) features, reused from training if present (default: ./results/features_cache)")
    ap.add_argument("--checkpoints_dir", default="./results/models_trained", help="folder of trained model_*.pt checkpoints to ensemble (default: ./results/models_trained)")
    ap.add_argument("--out_csv", default="./results/predictions.csv", help="where to save the per-slide prediction table (default: ./results/predictions.csv)")
    ap.add_argument("--metrics_out", default="./results/metrics.json", help="where to save AUC/accuracy/confusion-matrix + the paper-style bootstrapped metrics as JSON (default: ./results/metrics.json)")
    ap.add_argument("--roc_out", default="./results/roc_curve.png", help="where to save the ROC curve figure (paper Fig. 2a-c style), or 'none' to skip plotting (default: ./results/roc_curve.png)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_tiles", type=int, default=5000)
    ap.add_argument("--threshold", type=float, default=0.5, help="probability cutoff for turning P(MUT) into a predicted 0/1 label, used ONLY for the simple fixed-threshold accuracy/confusion-matrix report")
    ap.add_argument("--n_bootstrap", type=int, default=1000, help="bootstrap resamples for the paper-style AUC/AP/Se/Sp/PPV/NPV confidence intervals (paper default: 1000). Set to 0 to skip this report entirely.")
    ap.add_argument("--target_sens", type=float, default=0.92, help="the paper picks its operating threshold as the lowest score that still achieves this sensitivity (paper default: 0.92) -- see WHICH METRICS above")
    ap.add_argument("--n_extreme", type=int, default=N_EXTREME, help=(
        "MUST MATCH whatever n_extreme the checkpoints were actually trained with "
        "(default: the paper's 100). Chowder's ExtremeLayer picks the top-N and "
        "bottom-N scoring tiles, so its next layer's input size depends on N — "
        "loading a checkpoint with the wrong --n_extreme fails with a tensor "
        "'size mismatch' error. train_fgfr3mut.py auto-shrinks n_extreme (and "
        "prints 'note: n_extreme reduced to X') whenever the smallest training "
        "slide has fewer than 2*n_extreme tiles — e.g. with only 80 tiles/slide "
        "you'd need --n_extreme 40 here to match."))
    ap.add_argument("--device", default="auto", help="auto (default: cuda:0 if GPU available else cpu), cpu, or cuda:0/cuda:1")
    args = ap.parse_args()
    args.device = resolve_device(args.device)
    print(f"Using device: {args.device}")

    # ---- 1. Find the trained checkpoints we're going to ensemble -------------
    checkpoints_dir = Path(args.checkpoints_dir)
    # sorted() makes the checkpoint order (and therefore any printed ordering)
    # deterministic/reproducible between runs instead of depending on the
    # filesystem's arbitrary directory-listing order.
    checkpoints = sorted(checkpoints_dir.glob("*.pt"))
    assert checkpoints, f"no *.pt checkpoints found in {checkpoints_dir} — run train_fgfr3mut.py first"
    print(f"found {len(checkpoints)} checkpoint(s) to ensemble:")
    for c in checkpoints:
        print(f"  - {c.name}")

    # ---- 2. Load slide features (same helpers train_fgfr3mut.py uses) --------
    labels = load_labels(args.labels_csv)   # slide_name -> 0.0/1.0 ground truth
    if args.features_dir:
        X, y, slide_names = build_features_from_dir(args.features_dir, labels, args.max_tiles)
    else:
        assert args.tiles_root and args.slides_dir, "need --tiles_root + --slides_dir or --features_dir"
        # NOTE: if a slide's features were already cached under --feat_dir
        # during training, this will just reuse that cache instead of
        # re-running the slow H-optimus-0 embedding step again.
        X, y, slide_names = build_features_from_tiles(args.tiles_root, args.slides_dir, args.feat_dir, labels, args.device)
    # `slide_names` (returned alongside X/y) is the authoritative, correctly-
    # aligned list of which slides actually made it into X/y — some slides in
    # labels.csv can get skipped (e.g. 0 kept tiles), so we must NOT assume
    # X/y line up with the original labels.csv order.
    print(f"loaded: {len(X)} slides, {int((y == 1).sum())} MUT / {int((y == 0).sum())} WT")

    # ---- 3. Run every slide through every checkpoint and average the results -
    # predict_ensemble() (defined in train_fgfr3mut.py) does the actual work:
    # for each checkpoint, load its weights into a fresh Chowder model, run a
    # forward pass (torch.inference_mode(), no training) over all slides, take
    # sigmoid(logit) to get a 0..1 probability, then average those
    # probabilities across all checkpoints for each slide.
    probs = predict_ensemble(X, checkpoints, args.device, batch_size=args.batch_size, n_extreme=args.n_extreme)

    # A probability >= threshold becomes a predicted "1" (MUT), else "0" (WT).
    # .astype(int) turns the True/False comparison result into 1/0.
    pred_labels = (probs >= args.threshold).astype(int)

    # ---- 4. Print + save a per-slide table -----------------------------------
    print("\nslide,label,pred_prob,pred_label")
    for name, true_lab, prob, pred_lab in zip(slide_names, y, probs, pred_labels):
        print(f"{name},{int(true_lab)},{prob:.4f},{pred_lab}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w") as f:
        f.write("slide,label,pred_prob,pred_label\n")
        for name, true_lab, prob, pred_lab in zip(slide_names, y, probs, pred_labels):
            f.write(f"{name},{int(true_lab)},{prob:.6f},{pred_lab}\n")
    print(f"\nsaved predictions -> {out_csv}")

    # ---- 5. Summary metrics ---------------------------------------------------
    both_classes_present = len(set(y.tolist())) > 1  # AUC/bootstrap need both MUT and WT to exist

    # (A) Simple fixed-threshold report ----------------------------------------
    # AUC (Area Under the ROC Curve) needs both classes present to be defined.
    auc = roc_auc_score(y, probs) if both_classes_present else float("nan")
    # Accuracy = fraction of slides where the thresholded prediction matched
    # the true label (a simpler, threshold-dependent metric — unlike AUC,
    # which considers all possible thresholds at once).
    acc = accuracy_score(y, pred_labels)

    print(f"\n** Ensemble performance ({len(checkpoints)} checkpoints, {len(X)} slides) **")
    print(f"AUC      = {auc:.3f}")
    print(f"Accuracy = {acc:.3f}  (threshold={args.threshold})")

    confusion = None
    if both_classes_present:
        # A confusion matrix breaks accuracy down into 4 counts:
        #   tn = predicted WT, actually WT   (correct)
        #   fp = predicted MUT, actually WT   (false alarm)
        #   fn = predicted WT, actually MUT   (missed mutation)
        #   tp = predicted MUT, actually MUT (correct)
        # `labels=[0, 1]` fixes the row/column order so it's always WT-then-MUT,
        # regardless of which labels happen to appear in this particular run.
        tn, fp, fn, tp = confusion_matrix(y, pred_labels, labels=[0, 1]).ravel()
        confusion = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        print(f"Confusion matrix @ threshold={args.threshold}:")
        print(f"  true WT,  pred WT  (TN) = {tn}")
        print(f"  true WT,  pred MUT (FP) = {fp}")
        print(f"  true MUT, pred WT  (FN) = {fn}")
        print(f"  true MUT, pred MUT (TP) = {tp}")

    # (B) Paper-style bootstrapped report ---------------------------------------
    # Reuses the FGFR3MUT paper's OWN metrics code (fgfr3mut/bootstrap.py) so
    # these numbers are computed exactly the way Table 1 / Fig. 2 were: AUC,
    # AP, and Se/Sp/PPV/NPV at a threshold chosen for --target_sens sensitivity,
    # each with a 95% bootstrap confidence interval. See the WHICH METRICS
    # section at the top of this file for what each one means and why the
    # threshold isn't just --threshold here.
    paper_metrics = None
    if both_classes_present and args.n_bootstrap > 0:
        print(f"\n** Paper-style bootstrapped metrics (n_bootstrap={args.n_bootstrap}, target_sens={args.target_sens}) **")
        skip_reason = None
        if len(X) < 20:
            skip_reason = (
                f"note: only {len(X)} slides — many of the {args.n_bootstrap} bootstrap resamples will "
                "accidentally contain zero MUT or zero WT slides (expected with this few slides); "
                "fgfr3mut/bootstrap.py silently skips those resamples and still reports CIs over "
                "whatever resamples succeeded (per-resample warnings suppressed below to avoid flooding "
                "the console — nothing is hidden, this is just explaining it once instead of hundreds of times)."
            )
            print(skip_reason)
        # Small-sample bootstrap resamples routinely hit "only one class present"
        # edge cases (see the note above): sklearn raises UndefinedMetricWarning/
        # UserWarning, and fgfr3mut/bootstrap.py itself logs one loguru WARNING
        # per skipped resample — both are expected noise here (already explained
        # to the user above), not a real problem, so we suppress both for the
        # duration of this call only (logger.disable/enable + warnings filters),
        # restored immediately afterward.
        logger.disable("fgfr3mut.bootstrap")
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
                paper_metrics = compute_ci_with_bootstrap(
                    y_true=y, y_pred=probs, y_pred_bin=None,
                    conf_level=0.95, n_repeats=args.n_bootstrap, target_sens=args.target_sens,
                )
        finally:
            logger.enable("fgfr3mut.bootstrap")
        print(f"AUC = {paper_metrics['med_auc']:.3f} [{paper_metrics['ci_lower_bound_auc']:.3f}-{paper_metrics['ci_upper_bound_auc']:.3f}]")
        print(f"AP  = {paper_metrics['med_ap']:.3f} [{paper_metrics['ci_lower_bound_ap']:.3f}-{paper_metrics['ci_upper_bound_ap']:.3f}]")
        print(f"Operating threshold (for target sensitivity {args.target_sens}) = {paper_metrics['med_threshold']:.3f} "
              f"[{paper_metrics['ci_lower_bound_threshold']:.3f}-{paper_metrics['ci_upper_bound_threshold']:.3f}]")
        print(f"Se  = {paper_metrics['med_se']:.3f} [{paper_metrics['ci_lower_bound_se']:.3f}-{paper_metrics['ci_upper_bound_se']:.3f}]")
        print(f"Sp  = {paper_metrics['med_sp']:.3f} [{paper_metrics['ci_lower_bound_sp']:.3f}-{paper_metrics['ci_upper_bound_sp']:.3f}]")
        print(f"PPV = {paper_metrics['med_ppv']:.3f} [{paper_metrics['ci_lower_bound_ppv']:.3f}-{paper_metrics['ci_upper_bound_ppv']:.3f}]")
        print(f"NPV = {paper_metrics['med_npv']:.3f} [{paper_metrics['ci_lower_bound_npv']:.3f}-{paper_metrics['ci_upper_bound_npv']:.3f}]")
    elif not both_classes_present:
        print("\n(skipping paper-style bootstrapped metrics: need both MUT and WT slides present)")

    # ---- 6. ROC curve figure (mirrors paper Fig. 2a-c) -------------------------
    # A ROC (Receiver Operating Characteristic) curve plots Sensitivity (TPR)
    # against Specificity as the classification threshold sweeps from 1 down
    # to 0 — it visualizes ranking quality across EVERY possible threshold at
    # once, which is exactly what the single AUC number above summarizes into
    # one value ("area under" this curve). The paper's Fig. 2a-c plots exactly
    # this, with the AUC and the NPV/PPV at its chosen operating threshold
    # annotated on the plot — we reproduce that here from our own predictions.
    roc_point_estimate = None
    if both_classes_present and args.roc_out.lower() != "none":
        from sklearn.metrics import roc_curve  # lazy: only needed for this plot

        # roc_curve gives us, for every distinct score in `probs`, the resulting
        # (false-positive-rate, true-positive-rate) pair if we used that score
        # as the classification threshold. Specificity = 1 - FPR by definition.
        fpr, tpr, roc_thresholds = roc_curve(y, probs)
        specificity = 1 - fpr

        # Pick the SAME kind of operating threshold the paper uses (see WHICH
        # METRICS above): the lowest score that still catches >= target_sens
        # of true MUT cases. This is a plain point estimate on the FULL dataset
        # (no bootstrap resampling), unlike paper_metrics['med_threshold']
        # above (which is the median across bootstrap resamples) — the two
        # will usually be close but aren't required to match exactly.
        idx_target = np.where(tpr >= args.target_sens)[0][0]
        op_threshold = float(roc_thresholds[idx_target])
        op_pred_labels = (probs >= op_threshold).astype(int)
        op_tn, op_fp, op_fn, op_tp = confusion_matrix(y, op_pred_labels, labels=[0, 1]).ravel()
        # PPV (precision) and NPV need at least one predicted-positive /
        # predicted-negative to be defined; guard against divide-by-zero on
        # tiny datasets where the operating threshold predicts only one class.
        op_ppv = op_tp / (op_tp + op_fp) if (op_tp + op_fp) > 0 else float("nan")
        op_npv = op_tn / (op_tn + op_fn) if (op_tn + op_fn) > 0 else float("nan")
        roc_point_estimate = {
            "target_sens": args.target_sens, "threshold": op_threshold,
            "sensitivity": float(op_tp / (op_tp + op_fn)) if (op_tp + op_fn) > 0 else float("nan"),
            "specificity": float(op_tn / (op_tn + op_fp)) if (op_tn + op_fp) > 0 else float("nan"),
            "ppv": float(op_ppv), "npv": float(op_npv),
        }

        import matplotlib
        matplotlib.use("Agg")   # headless/non-interactive backend: just render to a file, no display needed
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5))
        # drawstyle="steps-post": connects the ROC points as a staircase
        # (matches how a ROC curve built from a finite set of scores actually
        # looks — see the paper's own Fig. 2a-c, which has the same jagged
        # stairstep shape rather than a smooth curve).
        ax.plot(specificity, tpr, drawstyle="steps-post", color="#1f5fa8", linewidth=2)
        ax.plot([1, 0], [0, 1], linestyle="--", color="gray", linewidth=1)  # diagonal = random-guessing baseline
        ax.scatter([1 - fpr[idx_target]], [tpr[idx_target]], color="crimson", zorder=5,
                   label=f"operating point (target Se={args.target_sens})")

        ax.set_xlim(1.02, -0.02)   # reversed x-axis, 1.0 (left) -> 0.0 (right), matching paper Fig. 2a-c
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Specificity")
        ax.set_ylabel("Sensitivity")
        ax.set_title(f"ROC curve — {len(X)} slides ({int((y == 1).sum())} MUT / {int((y == 0).sum())} WT)")
        # Text annotations in the same style/position as the paper's Fig. 2a-c
        # (AUC top-left; NPV/PPV near the marked operating point).
        # ha="right" right-aligns each string so it grows LEFTWARD from x=0.90,
        # comfortably clear of the right edge of the axes box.
        ax.text(0.90, 0.06, f"AUC={auc:.2f}", transform=ax.transAxes, fontsize=11, ha="right")
        ax.text(0.90, 0.16, f"NPV={op_npv:.2f}", transform=ax.transAxes, fontsize=11, ha="right")
        ax.text(0.90, 0.26, f"PPV={op_ppv:.2f}", transform=ax.transAxes, fontsize=11, ha="right")
        ax.legend(loc="lower left", fontsize=8)
        fig.tight_layout()

        roc_out = Path(args.roc_out)
        roc_out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(roc_out, dpi=150, bbox_inches="tight")  # bbox_inches="tight": never clip content that sticks out past the axes
        plt.close(fig)   # free the figure's memory; harmless but good practice in scripts/loops
        print(f"saved ROC curve -> {roc_out}")
    elif not both_classes_present:
        print("\n(skipping ROC curve: need both MUT and WT slides present)")

    # ---- 7. Save every metric computed above to one JSON file -----------------
    # json.dump can't serialize numpy floats/ints directly, so cast everything
    # to plain Python float/int first (this is a common numpy<->JSON gotcha).
    metrics_record = {
        "n_checkpoints": len(checkpoints),
        "checkpoints": [c.name for c in checkpoints],
        "n_slides": len(X),
        "n_mut": int((y == 1).sum()),
        "n_wt": int((y == 0).sum()),
        "fixed_threshold_report": {
            "threshold": args.threshold,
            "auc": float(auc),
            "accuracy": float(acc),
            "confusion_matrix": confusion,   # {tn, fp, fn, tp}, or null if only one class present
        },
        "paper_style_bootstrap_report": (
            {k: float(v) for k, v in paper_metrics.items()} if paper_metrics is not None else None
        ),
        "roc_operating_point": roc_point_estimate,   # non-bootstrapped point estimate used for the ROC figure; null if skipped
    }
    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w") as f:
        json.dump(metrics_record, f, indent=2)
    print(f"\nsaved metrics -> {metrics_out}")


if __name__ == "__main__":
    main()
