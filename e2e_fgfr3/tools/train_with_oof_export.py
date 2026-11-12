"""Run E2E-ABMILX ``main.py`` unchanged and also save per-slide test predictions.

E2E's ``main.py`` only logs aggregate metrics. At the end of every fold it scores
the test set with the best-on-val checkpoint(s) via
``engines.base_engine.validate() -> get_metric_val(bag_logit, ...)``. This wrapper
wraps that ``get_metric_val`` reference at runtime (no file in the E2E code is
edited), records the slide logits of every ``status == "test"`` call, then hands
control to ``main.py`` exactly as ``python main.py <args>`` would.

The last test call of a fold is the final best-checkpoint evaluation, so after
training ``--oof_csv`` holds one row per (fold, test slide):

    fold,slide,patient,label,prob_main,prob_ema

``prob_*`` = P(FGFR3 mutant) = softmax(logits)[:, 1] for the main model and, when
the config evaluates it (test_type both_ema), the EMA model. Test folds are the
FGFR3 held-out folds, so together the rows are out-of-fold predictions for every
slide.

Usage (same arguments as main.py, plus --oof_csv):
    python train_with_oof_export.py --e2e_root third_party/E2E-WSI-ABMILX \
        --oof_csv results/.../oof_predictions.csv -- <main.py arguments>
"""

import argparse
import csv
import os
import re
import runpy
import sys
from pathlib import Path


def patient_of(slide):
    m = re.match(r"(TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4})", slide)
    return m.group(1) if m else slide


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--e2e_root", required=True)
    ap.add_argument("--oof_csv", required=True)
    ap.add_argument("--expect_folds", type=int, default=0, help="exit non-zero unless this many folds were scored")
    ap.add_argument("main_args", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    main_args = args.main_args[1:] if args.main_args[:1] == ["--"] else args.main_args

    e2e_root = Path(args.e2e_root).resolve()
    oof_csv = Path(args.oof_csv).resolve()
    sys.path.insert(0, str(e2e_root))
    os.chdir(e2e_root)

    import torch
    import engines.base_engine as base_engine

    original = base_engine.get_metric_val
    records = {}   # (fold, slide) -> row
    fresh = set()  # folds scored by this process

    # Keep rows of folds scored by an earlier run (resuming with --fold_start).
    if oof_csv.exists():
        with open(oof_csv, newline="") as fh:
            for row in csv.DictReader(fh):
                row["fold"] = int(row["fold"])
                records[(row["fold"], row["slide"])] = row

    def write_csv():
        rank = int(os.environ.get("RANK", "0"))
        if rank != 0 or not records:
            return
        oof_csv.parent.mkdir(parents=True, exist_ok=True)
        tmp = oof_csv.with_suffix(".tmp")
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["fold", "slide", "patient", "label", "prob_main", "prob_ema"])
            w.writeheader()
            for key in sorted(records):
                w.writerow(records[key])
        tmp.replace(oof_csv)

    def recording_get_metric_val(margs, bag_logit, bag_labels, *a, **kw):
        if kw.get("status") == "test" and bag_logit is not None and torch.is_tensor(bag_logit):
            logits = bag_logit.detach().float().cpu()
            probs = (torch.softmax(logits, dim=-1)[:, 1] if logits.shape[-1] > 1
                     else torch.sigmoid(logits.view(-1)))
            labels = bag_labels.detach().cpu().view(-1).tolist()
            slides = list(getattr(margs, "all_imgs_test", {}).keys())
            if len(slides) != len(probs):
                raise RuntimeError(f"test slide list ({len(slides)}) and logits ({len(probs)}) differ")
            col = "prob_ema" if kw.get("suffix") == "ema" else "prob_main"
            if margs.fold_curr not in fresh:
                for key in [k for k in records if k[0] == margs.fold_curr]:
                    del records[key]
                fresh.add(margs.fold_curr)
            for s, lab, p in zip(slides, labels, probs.tolist()):
                row = records.setdefault((margs.fold_curr, s), {
                    "fold": margs.fold_curr, "slide": s, "patient": patient_of(s),
                    "label": int(lab), "prob_main": "", "prob_ema": ""})
                row[col] = f"{p:.6f}"
            write_csv()
        return original(margs, bag_logit, bag_labels, *a, **kw)

    base_engine.get_metric_val = recording_get_metric_val

    sys.argv = [str(e2e_root / "main.py")] + main_args
    runpy.run_path(str(e2e_root / "main.py"), run_name="__main__")

    # main.py swallows exceptions (traceback.print_exc()), so check the outcome here.
    if int(os.environ.get("RANK", "0")) == 0:
        folds = sorted({k[0] for k in records})
        print(f"[oof] {len(records)} test predictions from fold(s) {folds} "
              f"(scored now: {sorted(fresh)}) -> {oof_csv}")
        if args.expect_folds and len(folds) != args.expect_folds:
            sys.exit(f"[oof] error: expected {args.expect_folds} scored folds, got {len(folds)} "
                     "(training failed? see the traceback above)")


if __name__ == "__main__":
    main()
