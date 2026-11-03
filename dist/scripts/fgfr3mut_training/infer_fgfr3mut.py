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
     — not just "did training run". This includes both a threshold report
     (accuracy, confusion matrix at the PREDICTION threshold — see CALIBRATED
     PREDICTION LABEL below) AND the exact metric set the FGFR3MUT paper
     itself reports, computed the exact same way as the paper's own code
     (see WHICH METRICS below). By default every metric is computed per
     PATIENT rather than per slide — see SLIDE vs PATIENT below.
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

ENSEMBLE vs OUT-OF-FOLD (--oof): WHICH NUMBER SHOULD YOU BELIEVE?
-----------------------------------------------------------------------
The default scores every slide with the AVERAGE of all checkpoints
("ensembling"). That is the right thing when the slides you are scoring are
brand-new — none of the checkpoints ever saw them.

It is the WRONG thing for measuring how well training worked on the slides you
trained with. Each slide is held out by exactly one CV fold, but it is TRAINED
ON by every other fold — so at least one checkpoint has memorized it, and that
checkpoint's (memorized) vote drags the average toward the correct label. With
a handful of slides this inflates the metrics a lot: on this repo's 5-slide
demo, the ensemble reports AUC 1.000 while the honest out-of-fold score is
AUC 0.667 (accuracy 0.80 -> 0.60).

Pass --oof to score each slide ONLY with the checkpoint that was trained
without it: the standard "out-of-fold" cross-validation prediction, where
every single number comes from a model that never saw that slide. Use --oof
whenever the slides being scored are the same ones used for training (i.e.
when you want an honest generalization estimate); leave it off only when you
are scoring slides that were never trained on.

--oof has to know the fold layout. It reads it from the
``train_manifest.json`` that train_fgfr3mut.py writes next to the checkpoints
(the exact slides, labels, patient grouping, fold count and per-repeat seed
used at training time) — and, for checkpoint sets without a manifest, falls
back to re-creating the split from --splits the way this script always did.
Repeat-aware too: checkpoints saved as ``split_<repeat>/model_<fold>_*.pt``
(the paper's 25 x 5 = 125-model layout) are matched to the fold that held each
slide out IN THAT REPEAT, so a slide's out-of-fold score averages the one
model per repeat that never trained on it. Both modes' headline numbers are
always printed and both are saved to --metrics_out, so you can compare them.

SLIDE vs PATIENT (--metric_unit, default: patient)
---------------------------------------------------
The paper does not score slides: it averages every slide of one patient into a
single probability (``run_inference.py``: ``df.groupby(patient).mean()``) and
computes AUC/Se/Sp/PPV/NPV on those patient-level scores, because the clinical
question is "does THIS PATIENT have an FGFR3-mutant tumour?" and a patient with
two slides must not count twice. This script therefore groups by patient by
default: ``--metric_unit patient`` (mean probability per patient, label taken
from the patient — all slides of a patient share one genotype) or
``--metric_unit slide`` to score each slide separately. The chosen unit is
recorded in --metrics_out, and --out_csv always carries both identifiers
(columns: ``slide,patient,label,pred_prob,pred_label``).

CALIBRATED PREDICTION LABEL (--threshold, default: auto)
--------------------------------------------------------
``pred_label`` (the 0/1 column of --out_csv) is what the model actually
DECIDES, and the paper never decides at 0.5: it picks an operating threshold
calibrated to miss at most (1 - target_sens) of the true MUT cases, because a
missed FGFR3-mutant patient (no erdafitinib) is worse than an over-call. With
the default ``--threshold auto`` this script applies exactly the paper's rule
(the same code path as fgfr3mut/bootstrap.py's ``_compute_metrics``: the ROC
threshold whose sensitivity first reaches --target_sens, default 0.92) to the
probabilities being reported, and records the value used in --metrics_out.
Pass a number (e.g. ``--threshold 0.5``) for a fixed cutoff instead; accuracy
at 0.5 is always printed alongside for reference. Calibration needs both
classes present — with a single-class labels file the script falls back to 0.5
and says so.

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

  # Honest cross-validation numbers — score each slide ONLY with the fold
  # checkpoint that held it out. Use this whenever the slides you are scoring
  # are the ones you trained on (the default ensemble then averages in a model
  # that memorized each slide, e.g. AUC 1.000 vs the honest 0.667 on the
  # 5-slide demo above):
  python scripts/fgfr3mut_training/infer_fgfr3mut.py \\
    --oof --tiles_root ./results/tiles_out --slides_dir ./image \\
    --labels_csv ./results/labels.csv

Output: prints the prediction table + summary metrics to the console, writes
the same table to --out_csv (default ./results/predictions.csv) with columns
``slide,patient,label,pred_prob,pred_label`` — one row per --metric_unit unit
(a patient row lists that patient's slide(s), semicolon-separated) — writes
every computed metric to --metrics_out (default ./results/metrics.json), and —
mirroring the paper's own Fig. 2a-c — plots and saves an ROC curve to --roc_out
(default ./results/roc_curve.png), with AUC/NPV/PPV annotated the same way the
paper's figure does.
"""

import argparse
import json
import re
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
    make_folds,
    patient_id,
    predict_ensemble,
    resolve_device,
)

# Importing train_fgfr3mut (above) already added the vendored fgfr3mut/ package
# folder to sys.path as a side effect (see its own SCRIPT_DIR search loop), so
# this import of the paper's OWN metrics/bootstrap code just works from here too.
from fgfr3mut.bootstrap import compute_ci_with_bootstrap  # noqa: E402
from loguru import logger  # noqa: E402  (bootstrap.py logs skipped resamples through this)


def parse_checkpoint_split(ckpt, checkpoints_dir):
    """(repeat, fold) index encoded in a checkpoint's path.

    train_fgfr3mut.py saves ``model_<fold>_<epochs>_ep.pt``, optionally inside a
    ``split_<repeat>/`` subfolder when it was run with --repeats > 1 (the paper's
    ``models/split_<r>/model_<k>_30_ep.pt`` layout). A flat checkpoint (a
    --repeats 1 run, i.e. everything trained before --repeats existed) counts as
    repeat 0.
    """
    m = re.search(r"model_(\d+)_", ckpt.name)
    if m is None:
        raise ValueError(
            f"checkpoint '{ckpt.name}' does not look like train_fgfr3mut.py's "
            "'model_<fold>_<epochs>_ep.pt' name, so its fold is unknown — it cannot "
            "be matched to a held-out fold"
        )
    try:
        rel_parent = str(ckpt.parent.relative_to(checkpoints_dir))
    except ValueError:
        rel_parent = ckpt.parent.name
    m_rep = re.search(r"split_(\d+)", rel_parent)
    return (int(m_rep.group(1)) if m_rep else 0), int(m.group(1))


def operating_threshold(y_true, scores, target_sens):
    """The paper's operating threshold for a target sensitivity.

    Exactly what fgfr3mut/bootstrap.py's ``_compute_metrics`` does: sweep the ROC
    and take the first (i.e. highest) threshold whose sensitivity reaches
    ``target_sens`` — the most conservative cutoff that still misses no more than
    ``1 - target_sens`` of the true MUT cases. Needs both classes present.
    """
    from sklearn.metrics import roc_curve

    _fpr, tpr, thresholds = roc_curve(y_true, scores)
    return float(thresholds[np.where(tpr >= target_sens)[0][0]])


def aggregate_units(slide_names, y, probs, unit):
    """Group slide-level scores into the unit metrics are reported on.

    ``unit="patient"`` averages every slide of one patient into ONE probability
    (the paper's ``run_inference.py`` does ``groupby(patient).mean()`` before any
    metric, because the clinical question is about the patient and a patient with
    two slides must not count twice). ``unit="slide"`` leaves rows untouched.

    Returns ``(unit_y, unit_probs, rows)`` with one ``{"patient", "slides"}`` row
    per scored unit, in a stable order (patients sorted, slides kept in the
    order they were loaded).
    """
    if unit != "patient":
        rows = [{"patient": patient_id(n), "slides": [n]} for n in slide_names]
        return np.asarray(y, dtype=float), np.asarray(probs, dtype=float), rows

    order: dict[str, list[int]] = {}
    for i, name in enumerate(slide_names):
        order.setdefault(patient_id(name), []).append(i)

    rows, ys, ps = [], [], []
    for pat, idxs in order.items():
        labs = {int(y[i]) for i in idxs}
        if len(labs) > 1:
            print(f"WARNING: {pat} has slides with different labels {sorted(labs)} — "
                  f"using {int(y[idxs[0]])} for the patient (all slides of a patient share one genotype)")
        ys.append(float(y[idxs[0]]))
        ps.append(float(np.mean([probs[i] for i in idxs])))
        rows.append({"patient": pat, "slides": [slide_names[i] for i in idxs]})
    return np.asarray(ys), np.asarray(ps), rows


def out_of_fold_probs(ckpt_probs, checkpoints, scored_names, scored_y, checkpoints_dir,
                      manifest=None, splits=5):
    """Score every slide with ONLY the checkpoint(s) that were trained WITHOUT it.

    WHY THIS EXISTS
    ---------------
    train_fgfr3mut.py divides the slides into cross-validation folds and trains
    one Chowder per fold on the OTHER folds. Averaging every checkpoint (this
    script's default) is correct when you score brand-new slides — none of the
    checkpoints ever saw them. But when the slides being scored are the SAME
    slides used for training, the average is quietly optimistic: a slide is held
    out by exactly one fold yet trained on by all the others, so at least one
    checkpoint has memorized it, and that vote pulls the average toward the true
    label.

    This rebuilds train_fgfr3mut.py's exact fold split and, for each slide, keeps
    only the checkpoint column(s) trained on the other folds. The result is the
    standard "out-of-fold" prediction: every number came from a model that never
    trained on that slide.

    The fold layout is taken from ``train_manifest.json`` when the checkpoint dir
    carries one (slides, labels, patient grouping, fold count — no guessing), and
    falls back to re-creating the historical slide-level split from ``splits``
    otherwise. Checkpoints in ``split_<repeat>/`` subfolders (the paper's 25 x 5
    recipe) are matched per repeat: a slide's score averages the one model per
    repeat that held it out.

    Parameters
    ----------
    ckpt_probs : (n_scored, n_checkpoints) array — one column per checkpoint.
    checkpoints : the checkpoint Paths (fold/Repeat parsed from their paths).
    scored_names, scored_y : the slides being scored, in row order of ckpt_probs.
    checkpoints_dir : where the checkpoints were found (needed for path parsing).
    manifest : train_manifest.json contents, or None when there is none.
    splits : --splits fallback when there is no manifest (both scripts default 5).

    Returns
    -------
    (probs, desc_lines, fold_cols)
        probs     : (n_scored,) out-of-fold probabilities
        desc_lines: printable description of the fold layout
        fold_cols : {(repeat, fold) -> [checkpoint column, ...]}

    Raises ValueError with a human-readable message when the checkpoints and the
    fold split cannot be lined up (unparseable name, missing fold, a scored slide
    the models never trained on, too few folds to rebuild the split, ...).
    """
    if len(scored_y) == 0:
        raise ValueError("no slides to score")

    # 1) Which (repeat, fold) did each checkpoint column come from?
    fold_cols: dict[tuple[int, int], list[int]] = {}
    for col, ckpt in enumerate(checkpoints):
        fold_cols.setdefault(parse_checkpoint_split(ckpt, checkpoints_dir), []).append(col)

    # 2) Rebuild the split EXACTLY as train_fgfr3mut.main() built it.
    if manifest is not None:
        tr_names = list(manifest.get("slides") or [])
        tr_y = np.asarray(manifest.get("labels") or [], dtype=float)
        split_unit = manifest.get("split_unit", "patient")
        n_splits = int(manifest.get("n_splits") or 0)
        if not tr_names or len(tr_y) != len(tr_names) or n_splits < 2:
            raise ValueError(
                f"train_manifest.json is unusable (slides={len(tr_names)}, "
                f"labels={len(tr_y)}, n_splits={n_splits})"
            )
        # Out-of-fold only exists for slides the training run actually saw.
        unscored = [n for n in scored_names if n not in set(tr_names)]
        if unscored:
            raise ValueError(
                f"{len(unscored)} of the {len(scored_names)} scored slide(s) were never trained on "
                f"(e.g. {unscored[0]}) — out-of-fold scoring applies only to the TRAINING slides. "
                "This is the brand-new-slide case, for which this script's default ensemble mode is "
                "the right choice."
            )
        splits = n_splits   # training-time fold count: no guessing
    else:
        # No manifest (checkpoints from before it existed): assume the scored
        # slides ARE the trained slides and rebuild the historical slide-level
        # split from --splits, exactly like older versions of this script did.
        tr_names = list(scored_names)
        tr_y = np.asarray(scored_y, dtype=float)
        split_unit = "slide"

    groups = [patient_id(n) for n in tr_names] if split_unit == "patient" else None
    scored_idx = {name: i for i, name in enumerate(scored_names)}

    # 3) Rebuild every repeat's folds; each present repeat must be complete.
    folds_by_repeat: dict[int, list] = {}
    for rep in sorted({r for r, _f in fold_cols}):
        try:
            n_this, folds = make_folds(tr_y, splits, repeat=rep, groups=groups)
        except ValueError as exc:
            raise ValueError(
                f"cannot rebuild repeat {rep}'s fold split from the training labels: {exc}"
            ) from None
        have = sorted(f for (r, f) in fold_cols if r == rep)
        missing = sorted(set(range(n_this)) - set(have))
        if missing:
            raise ValueError(
                f"repeat {rep} has no checkpoint for fold(s) {missing} (found folds {have}) — "
                "is --checkpoints_dir holding ONE complete training run's checkpoints? "
                f"({', '.join(c.name for c in checkpoints)})"
            )
        folds_by_repeat[rep] = folds

    # 4) A slide is held out by exactly one fold PER REPEAT: average those.
    votes: list[list[float]] = [[] for _ in scored_names]
    for rep, folds in folds_by_repeat.items():
        for fold, (_tr, va) in enumerate(folds):
            cols = fold_cols[(rep, fold)]
            for j in va:                      # j = row in the TRAINED layout
                i = scored_idx.get(tr_names[j])
                if i is None:
                    continue                  # trained slide not being scored this run
                votes[i].append(float(np.mean(ckpt_probs[i, cols])))
    oof = np.full(len(scored_names), np.nan, np.float64)
    for i, v in enumerate(votes):
        if v:
            oof[i] = float(np.mean(v))
    if np.isnan(oof).any():
        n_blank = sum(1 for v in votes if not v)
        raise ValueError(
            f"{n_blank} of {len(votes)} scored slide(s) were never held out by any fold — "
            "the checkpoints and the rebuilt split don't line up"
        )
    return oof, _describe_folds(folds_by_repeat, fold_cols, checkpoints, tr_names), fold_cols


def _describe_folds(folds_by_repeat, fold_cols, checkpoints, trained_names):
    """Printable lines: which slides each fold held out, which checkpoints score them.

    Only the first repeat's membership is spelled out (later repeats are the
    same structure with different slides), plus a summary line for the rest.
    """
    rep0 = min(folds_by_repeat)
    lines = []
    for fold, (_tr, va) in enumerate(folds_by_repeat[rep0]):
        held = [trained_names[i] for i in va]
        shown = ", ".join(n[:30] for n in held[:3])
        if len(held) > 3:
            shown += f", +{len(held) - 3} more"
        cks = ", ".join(checkpoints[c].name for c in fold_cols[(rep0, fold)])
        lines.append(f"  repeat {rep0} / fold {fold}: holds out {len(va)} slide(s) [{shown}] -> scored by {cks}")
    if len(folds_by_repeat) > 1:
        lines.append(
            f"  ... + {len(folds_by_repeat) - 1} more repeat(s), each a different partition, so every "
            f"slide is scored by {len(folds_by_repeat)} model(s) that never trained on it"
        )
    return lines


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
    ap.add_argument("--threshold", default="auto", help=(
        "cutoff turning P(MUT) into the predicted 0/1 label. 'auto' (default) = calibrate it the way "
        "the paper does: the ROC threshold whose sensitivity reaches --target_sens (missing a MUT "
        "patient is worse than over-calling); the value used is recorded in --metrics_out. Pass a "
        "number (e.g. 0.5) for a fixed cutoff. Accuracy at 0.5 is always printed for reference."))
    ap.add_argument("--metric_unit", choices=("patient", "slide"), default="patient", help=(
        "what one scored row is: 'patient' (default) averages a patient's slides into one "
        "probability before ANY metric, exactly like the paper's run_inference.py "
        "groupby(patient).mean(); 'slide' scores each slide separately"))
    ap.add_argument("--n_bootstrap", type=int, default=1000, help="bootstrap resamples for the paper-style AUC/AP/Se/Sp/PPV/NPV confidence intervals (paper default: 1000). Set to 0 to skip this report entirely.")
    ap.add_argument("--target_sens", type=float, default=0.92, help="the paper picks its operating threshold as the lowest score that still achieves this sensitivity (paper default: 0.92) -- see WHICH METRICS above")
    ap.add_argument("--n_extreme", type=int, default=None, help=(
        "MUST MATCH whatever n_extreme the checkpoints were actually trained with. Chowder's "
        "ExtremeLayer picks the top-N and bottom-N scoring tiles, so its next layer's input size "
        "depends on N — loading a checkpoint with the wrong value fails with a tensor 'size "
        "mismatch'. Default: read from train_manifest.json (train_fgfr3mut.py records the real "
        "value per checkpoint, including when it auto-shrunk it to fit a small slide), else the "
        "paper's 100."))
    ap.add_argument("--oof", action="store_true",
                    help="HONEST cross-validation mode: score each slide ONLY with the checkpoint "
                         "that was trained without it, instead of averaging every checkpoint. "
                         "Use this whenever the slides being scored are the ones used for training "
                         "(the default ensemble average then includes a model that memorized each "
                         "slide and inflates AUC) -- see ENSEMBLE vs OUT-OF-FOLD above")
    ap.add_argument("--splits", type=int, default=None,
                    help="only used with --oof when there is NO train_manifest.json next to the "
                         "checkpoints (with one, the training-time fold count is used, which is "
                         "always right). Fallback default: 5, the paper's fold count")
    ap.add_argument("--device", default="auto", help="auto (default: cuda:0 if GPU available else cpu), cpu, or cuda:0/cuda:1")
    args = ap.parse_args()
    args.device = resolve_device(args.device)
    print(f"Using device: {args.device}")

    # ---- 1. Find the trained checkpoints we're going to ensemble -------------
    checkpoints_dir = Path(args.checkpoints_dir)
    # sorted() makes the checkpoint order (and therefore any printed ordering)
    # deterministic/reproducible between runs instead of depending on the
    # filesystem's arbitrary directory-listing order. rglob (not glob) because
    # train_fgfr3mut.py --repeats puts them in split_<repeat>/ subfolders
    # (the paper's models/split_<r>/model_<k>_30_ep.pt layout); flat runs from
    # before --repeats existed are found at the top level as before.
    checkpoints = sorted(checkpoints_dir.rglob("*.pt"))
    assert checkpoints, f"no *.pt checkpoints found in {checkpoints_dir} — run train_fgfr3mut.py first"
    print(f"found {len(checkpoints)} checkpoint(s) to ensemble:")
    for c in checkpoints:
        print(f"  - {c.relative_to(checkpoints_dir)}")

    # train_manifest.json (written by train_fgfr3mut.py): the exact training-time
    # truth — which slides/labels/patients, how many folds, which seed rule — so
    # --oof rebuilds the SAME split and this script can tell training slides from
    # brand-new ones instead of guessing.
    manifest_path = checkpoints_dir / "train_manifest.json"
    manifest = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        print(f"read {manifest_path} "
              f"(training: {manifest.get('n_splits')} fold(s) x {manifest.get('repeats')} repeat(s), "
              f"split_unit={manifest.get('split_unit')}, epochs={manifest.get('epochs')})")
    else:
        print(f"note: no train_manifest.json in {checkpoints_dir} — falling back to --splits for "
              "--oof (older run; retrain to get an exact split record)")

    # Which --n_extreme do these checkpoints actually need? The manifest knows
    # per checkpoint (train_fgfr3mut.py shrinks it for small slides, which
    # changes Chowder's tensor shapes).
    manifest_n_extreme = None
    if manifest is not None:
        vals = sorted(set(int(v) for v in (manifest.get("n_extreme_values") or [])))
        if len(vals) == 1:
            manifest_n_extreme = vals[0]
        elif len(vals) > 1:
            raise SystemExit(
                f"error: the {len(vals)} checkpoints in {checkpoints_dir} were trained with "
                f"MIXED n_extreme values {vals} — they cannot be ensembled (different tensor "
                "shapes). Retrain so every slide has >= 200 tiles (or drop train_fgfr3mut.py's "
                "auto-reduction, which the paper — always 100 — does not have)."
            )
    if args.n_extreme is None:
        args.n_extreme = manifest_n_extreme if manifest_n_extreme is not None else N_EXTREME
        if manifest_n_extreme is not None:
            print(f"n_extreme = {args.n_extreme} (from train_manifest.json)")
    elif manifest_n_extreme is not None and args.n_extreme != manifest_n_extreme:
        print(f"WARNING: --n_extreme {args.n_extreme} but the manifest says the checkpoints were "
              f"trained with {manifest_n_extreme} — loading will most likely fail with a size mismatch")
    if args.splits is None:
        args.splits = 5 if manifest is None else int(manifest.get("n_splits") or 5)

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

    # ---- 3. Score every slide with every checkpoint ---------------------------
    # predict_ensemble() (defined in train_fgfr3mut.py) does the actual work:
    # for each checkpoint, load its weights into a fresh Chowder model, run a
    # forward pass (torch.inference_mode(), no training) over all slides, and
    # take sigmoid(logit) to get a 0..1 probability. We call it once PER
    # CHECKPOINT rather than once for the whole list so that we keep the
    # individual scores as columns: the ensemble average is simply their mean
    # (identical numbers to before), and --oof needs to pick a column per slide.
    ckpt_probs = np.column_stack([
        predict_ensemble(X, [c], args.device, batch_size=args.batch_size, n_extreme=args.n_extreme)
        for c in checkpoints
    ])
    probs_ensemble = ckpt_probs.mean(axis=1)

    # Out-of-fold scores are computed even when --oof is off, so both numbers
    # can always be shown side by side (and saved) for comparison.
    oof_probs, oof_desc, oof_fold_cols, oof_error = None, None, None, None
    try:
        oof_probs, oof_desc, oof_fold_cols = out_of_fold_probs(
            ckpt_probs, checkpoints, slide_names, y, checkpoints_dir,
            manifest=manifest, splits=args.splits,
        )
    except (ValueError, AssertionError) as exc:
        oof_error = str(exc)

    if args.oof:
        if oof_probs is None:
            raise SystemExit(f"error: --oof was requested but out-of-fold scoring is impossible: {oof_error}")
        probs, eval_mode = oof_probs, "out_of_fold"
        mode_title = "OUT-OF-FOLD — each slide scored only by the model(s) that held it out"
    else:
        probs, eval_mode = probs_ensemble, "ensemble_average"
        mode_title = "ENSEMBLE average of all checkpoints"

    # ---- 4. Aggregate to the reporting unit (per patient, as the paper does) ---
    # aggregate_units() is where a patient's multiple slides collapse into one
    # probability; every metric below (and --out_csv) works on those units.
    y_u, probs_u, rows = aggregate_units(slide_names, y, probs, args.metric_unit)
    ens_u = aggregate_units(slide_names, y, probs_ensemble, args.metric_unit)[1]
    oof_u = (aggregate_units(slide_names, y, oof_probs, args.metric_unit)[1]
             if oof_probs is not None else None)
    unit_word = "patient" if args.metric_unit == "patient" else "slide"
    if args.metric_unit == "patient" and len(rows) != len(slide_names):
        print(f"aggregated {len(slide_names)} slides -> {len(rows)} patient(s)")

    both_classes_present = len(set(y_u.tolist())) > 1  # AUC/bootstrap need both MUT and WT

    def _summary(p_u):
        """(AUC, accuracy @ 0.5) for one candidate set of unit-level probabilities."""
        auc_v = float(roc_auc_score(y_u, p_u)) if both_classes_present else float("nan")
        acc_v = float(accuracy_score(y_u, (p_u >= 0.5).astype(int)))
        return auc_v, acc_v

    print(f"\n** Evaluation mode: {mode_title} **")
    if args.oof:
        for line in (oof_desc or []):
            print(line)
    elif oof_probs is not None:
        print("  note: every slide below was also TRAINED on by some other fold's checkpoint, so the "
              "ensemble numbers are optimistic — re-run with --oof for the honest held-out figure")
    else:
        print(f"  note: out-of-fold scoring unavailable ({oof_error})")
    ens_auc_ref, ens_acc05_ref = _summary(ens_u)
    if oof_probs is not None:
        oof_auc_ref, oof_acc05_ref = _summary(oof_u)
        print(f"  out-of-fold (honest)  AUC={oof_auc_ref:.3f}  acc@0.5={oof_acc05_ref:.3f}")
        print(f"  ensemble              AUC={ens_auc_ref:.3f}  acc@0.5={ens_acc05_ref:.3f}")
    else:
        oof_auc_ref = oof_acc05_ref = float("nan")

    # ---- 5. The prediction threshold (paper: calibrated, not 0.5) --------------
    if str(args.threshold).lower() == "auto":
        if both_classes_present:
            eval_thr = operating_threshold(y_u, probs_u, args.target_sens)
            thr_mode = f"calibrated: lowest ROC threshold reaching Se >= {args.target_sens:g} (paper's operating rule)"
        else:
            eval_thr = 0.5
            thr_mode = "fallback 0.5: only one class present, cannot calibrate"
    else:
        eval_thr = float(args.threshold)
        thr_mode = "fixed"
    print(f"prediction threshold = {eval_thr:.4f}  ({thr_mode})")

    # A probability >= threshold becomes a predicted "1" (MUT), else "0" (WT).
    pred_labels = (probs_u >= eval_thr).astype(int)

    # ---- 6. Print + save the prediction table ---------------------------------
    # One row per scored unit (patient by default); the slide column lists every
    # slide that was averaged into that row, semicolon-separated.
    header = "slide,patient,label,pred_prob,pred_label"
    print(f"\n{header}   (unit={unit_word}, threshold={eval_thr:.4f})")
    for row, true_lab, prob, pred_lab in zip(rows, y_u, probs_u, pred_labels):
        print(f"{';'.join(row['slides'])},{row['patient']},{int(true_lab)},{prob:.4f},{pred_lab}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w") as f:
        f.write(header + "\n")
        for row, true_lab, prob, pred_lab in zip(rows, y_u, probs_u, pred_labels):
            f.write(f"{';'.join(row['slides'])},{row['patient']},{int(true_lab)},{prob:.6f},{pred_lab}\n")
    print(f"\nsaved predictions -> {out_csv}")

    # ---- 7. Summary metrics ---------------------------------------------------
    # `both_classes_present`, `y_u` and `probs_u` come from step 4 (the unit
    # aggregation): AUC / the bootstrap report / the ROC curve all need both MUT
    # and WT units to be defined.

    # (A) Threshold report at the PREDICTION threshold chosen in step 5 --------
    # AUC (Area Under the ROC Curve) needs both classes present to be defined.
    auc = roc_auc_score(y_u, probs_u) if both_classes_present else float("nan")
    # Accuracy = fraction of units where the thresholded prediction matched
    # the true label (a simpler, threshold-dependent metric — unlike AUC,
    # which considers all possible thresholds at once).
    acc = accuracy_score(y_u, pred_labels)
    acc_at_05 = accuracy_score(y_u, (probs_u >= 0.5).astype(int))

    mode_word = "Out-of-fold" if eval_mode == "out_of_fold" else "Ensemble"
    print(f"\n** {mode_word} performance ({len(checkpoints)} checkpoints, {len(rows)} "
          f"{unit_word}(s) from {len(slide_names)} slides) **")
    print(f"AUC      = {auc:.3f}")
    print(f"Accuracy = {acc:.3f}  (threshold={eval_thr:.4f}: {thr_mode})")
    if abs(eval_thr - 0.5) > 1e-9:
        print(f"Accuracy = {acc_at_05:.3f}  (reference: fixed threshold=0.5)")

    confusion = None
    if both_classes_present:
        # A confusion matrix breaks accuracy down into 4 counts:
        #   tn = predicted WT, actually WT   (correct)
        #   fp = predicted MUT, actually WT   (false alarm)
        #   fn = predicted WT, actually MUT   (missed mutation)
        #   tp = predicted MUT, actually MUT (correct)
        # `labels=[0, 1]` fixes the row/column order so it's always WT-then-MUT,
        # regardless of which labels happen to appear in this particular run.
        tn, fp, fn, tp = confusion_matrix(y_u, pred_labels, labels=[0, 1]).ravel()
        confusion = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        print(f"Confusion matrix @ threshold={eval_thr:.4f} ({unit_word}-level):")
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
        print(f"\n** Paper-style bootstrapped metrics (n_bootstrap={args.n_bootstrap}, target_sens={args.target_sens}, unit={unit_word}) **")
        skip_reason = None
        if len(y_u) < 20:
            skip_reason = (
                f"note: only {len(y_u)} {unit_word}(s) — many of the {args.n_bootstrap} bootstrap resamples will "
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
                # numpy's 0/0 for npv/specificity in the resamples that end up
                # with no predicted-negative (or no true-negative) slide —
                # expected with this few slides, same category of noise as the
                # two filters above (sklearn's metrics already return NaN there;
                # only numpy's scalar divide complains about it).
                warnings.filterwarnings("ignore", category=RuntimeWarning,
                                        message="invalid value encountered in scalar divide")
                paper_metrics = compute_ci_with_bootstrap(
                    y_true=y_u, y_pred=probs_u, y_pred_bin=None,
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

        # roc_curve gives us, for every distinct score in `probs_u`, the resulting
        # (false-positive-rate, true-positive-rate) pair if we used that score
        # as the classification threshold. Specificity = 1 - FPR by definition.
        fpr, tpr, roc_thresholds = roc_curve(y_u, probs_u)
        specificity = 1 - fpr

        # The SAME operating threshold the paper uses (see WHICH METRICS above),
        # picked on the FULL evaluation set (no bootstrap resampling), unlike
        # paper_metrics['med_threshold'] above (median across resamples). With
        # --threshold auto this is exactly the threshold pred_label used in step 5.
        idx_target = np.where(tpr >= args.target_sens)[0][0]
        op_threshold = float(roc_thresholds[idx_target])
        op_pred_labels = (probs_u >= op_threshold).astype(int)
        op_tn, op_fp, op_fn, op_tp = confusion_matrix(y_u, op_pred_labels, labels=[0, 1]).ravel()
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
        ax.set_title(f"ROC curve — {len(rows)} {unit_word}(s) "
                     f"({int((y_u == 1).sum())} MUT / {int((y_u == 0).sum())} WT)")
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
        "checkpoints": [str(c.relative_to(checkpoints_dir)) for c in checkpoints],
        "train_manifest": str(manifest_path) if manifest is not None else None,
        "n_slides": len(X),
        "n_units": len(rows),          # patients by default (--metric_unit)
        "metric_unit": args.metric_unit,
        "n_mut": int((y_u == 1).sum()),
        "n_wt": int((y_u == 0).sum()),
        # WHICH probabilities produced the reports below: "ensemble_average"
        # (every checkpoint votes on every slide — the default) or
        # "out_of_fold" (--oof: each slide scored only by the checkpoint that
        # was trained without it). Both summaries are always stored so the two
        # can be compared without re-running; see ENSEMBLE vs OUT-OF-FOLD at
        # the top of this file for why they differ.
        "eval_mode": eval_mode,
        "ensemble_reference": {"auc": float(ens_auc_ref), "accuracy_at_0.5": float(ens_acc05_ref)},
        "out_of_fold_reference": (
            {"auc": float(oof_auc_ref), "accuracy_at_0.5": float(oof_acc05_ref)}
            if oof_probs is not None else None
        ),
        "out_of_fold_error": oof_error,   # null when out-of-fold scoring worked
        # What pred_label was decided with (paper: a calibrated operating
        # threshold, never an arbitrary 0.5) — see CALIBRATED PREDICTION LABEL.
        "threshold": {
            "mode": thr_mode,
            "value": float(eval_thr),
            "target_sens": float(args.target_sens) if str(args.threshold).lower() == "auto" else None,
        },
        "prediction_report": {
            "auc": float(auc),
            "accuracy": float(acc),                 # at the threshold above
            "accuracy_at_0.5": float(acc_at_05),    # always recorded for reference
            "confusion_matrix": confusion,          # {tn, fp, fn, tp}, or null if one class only
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
