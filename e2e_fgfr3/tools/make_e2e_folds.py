"""Write E2E-ABMILX fold CSVs that reproduce the FGFR3 experiment's CV split.

Source of truth:
  * slides / labels / patients : the FGFR3 ``train_manifest.json``
  * fold layout                : ``make_folds()`` from the FGFR3 training script,
                                 run with the manifest's split settings and
                                 ``repeat`` (default 0 = Chowder's split_0)

``make_folds`` and ``patient_id`` are lifted out of ``train_fgfr3mut.py`` with
``ast`` and executed as-is, so the split is built by the very same code without
importing that script (importing it would load torch and the fgfr3mut package).

Each FGFR3 fold k has (train, held-out). For E2E:
  test  = FGFR3's held-out fold k                    (identical slides)
  val   = ~val_frac of FGFR3's train part, patient-grouped + label-stratified
          (E2E selects its best epoch and early-stops on it; Chowder had no val)
  train = the rest of FGFR3's train part

Output: ``<out_dir>/fold_{k}.csv`` with columns ``case,slide,label,fold``
(E2E renames them positionally to Case,ID,Label,Split; ``slide`` = LMDB slide id
= file name without ``.svs``), plus ``folds_summary.json`` with the checks.

NOTE: StratifiedGroupKFold's shuffling can differ across scikit-learn versions.
Run this with the same Python environment that trained the FGFR3 models to get
the identical split; the scikit-learn version used is recorded in the summary.
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def load_fgfr3_functions(train_script):
    """Exec only the make_folds/patient_id definitions from the FGFR3 train script."""
    tree = ast.parse(Path(train_script).read_text())
    wanted = {"make_folds", "patient_id"}
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    found = {d.name for d in defs}
    if found != wanted:
        sys.exit(f"error: {train_script} lacks {sorted(wanted - found)}")
    ns = {"np": np, "re": re, "Path": Path,
          "StratifiedGroupKFold": StratifiedGroupKFold, "StratifiedKFold": StratifiedKFold}
    exec(compile(ast.Module(body=defs, type_ignores=[]), str(train_script), "exec"), ns)
    return ns["make_folds"], ns["patient_id"]


def inner_val_split(y, groups, val_frac, seed):
    """Patient-grouped, label-stratified val split of a training part.

    Takes the first fold of StratifiedGroupKFold(n_splits=round(1/val_frac)) whose
    held-out part contains both classes.
    """
    n_splits = max(2, int(round(1.0 / val_frac)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in sgkf.split(np.zeros(len(y)), y, groups):
        if len(set(y[va].tolist())) == 2:
            return tr, va
    sys.exit("error: could not build a validation split with both classes")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="FGFR3 train_manifest.json")
    ap.add_argument("--train_script", required=True, help="FGFR3 train_fgfr3mut.py (provides make_folds)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--repeat", type=int, default=0, help="FGFR3 CV repeat to reproduce (0 = split_0)")
    ap.add_argument("--val_frac", type=float, default=0.125, help="share of each train part used as E2E val")
    ap.add_argument("--val_seed", type=int, default=0, help="seed of the inner val split (fold k uses seed+k)")
    args = ap.parse_args()

    make_folds, patient_id = load_fgfr3_functions(args.train_script)
    m = json.loads(Path(args.manifest).read_text())

    slides = list(m["slides"])
    y = np.asarray(m["labels"], dtype=float)
    patients = [patient_id(s) for s in slides]
    if m.get("patients") and list(m["patients"]) != patients:
        sys.exit("error: manifest patients do not match patient_id(slides)")
    split_unit = m.get("split_unit", "patient")
    groups = patients if split_unit == "patient" else None
    if args.repeat >= int(m.get("repeats", 1)):
        print(f"warning: manifest has {m.get('repeats')} repeat(s); repeat {args.repeat} has no Chowder counterpart")

    # Same call infer_fgfr3mut.py --oof uses to rebuild the training split.
    n_splits, folds = make_folds(y, int(m["n_splits"]), repeat=args.repeat, groups=groups)
    if n_splits != int(m["n_splits"]):
        sys.exit(f"error: rebuilt {n_splits} folds but the manifest says {m['n_splits']}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stems = [s[:-len(".svs")] if s.lower().endswith(".svs") else s for s in slides]
    pat = np.asarray(patients)
    y_int = y.astype(int)

    summary = {
        "manifest": str(args.manifest),
        "fgfr3_repeat": args.repeat,
        "split_unit": split_unit,
        "n_folds": n_splits,
        "sklearn_version": sklearn.__version__,
        "val_frac": args.val_frac,
        "val_seed_rule": f"{args.val_seed} + fold",
        "n_slides": len(slides),
        "n_patients": len(set(patients)),
        "folds": [],
    }
    seen_test = np.zeros(len(slides), dtype=int)
    for k, (tr, te) in enumerate(folds):
        tr_rel, va_rel = inner_val_split(y_int[tr], pat[tr], args.val_frac, args.val_seed + k)
        tr_idx, va_idx = tr[tr_rel], tr[va_rel]
        seen_test[te] += 1

        sets = {"train": set(pat[tr_idx]), "val": set(pat[va_idx]), "test": set(pat[te])}
        if (sets["train"] & sets["val"]) or (sets["train"] & sets["test"]) or (sets["val"] & sets["test"]):
            sys.exit(f"error: fold {k} shares a patient between splits")
        if len(set(y_int[te].tolist())) < 2:
            sys.exit(f"error: fold {k} test part has a single class")

        split = np.empty(len(slides), dtype=object)
        split[tr_idx], split[va_idx], split[te] = "train", "val", "test"
        df = pd.DataFrame({"case": patients, "slide": stems, "label": y_int, "fold": split})
        df.to_csv(out / f"fold_{k}.csv", index=False)

        info = {"fold": k}
        for name, idx in (("train", tr_idx), ("val", va_idx), ("test", te)):
            info[name] = {"slides": int(len(idx)), "patients": len(set(pat[idx])),
                          "mut": int(y_int[idx].sum()), "wt": int((y_int[idx] == 0).sum())}
        info["test_patients"] = sorted(sets["test"])
        summary["folds"].append(info)
        print(f"fold {k}: " + " | ".join(
            f"{n} {info[n]['slides']} slides/{info[n]['patients']} pts (MUT {info[n]['mut']})"
            for n in ("train", "val", "test")))

    if not np.all(seen_test == 1):
        sys.exit("error: test folds do not partition the slide set")
    (out / "folds_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {n_splits} fold CSVs + folds_summary.json -> {out} (scikit-learn {sklearn.__version__})")


if __name__ == "__main__":
    main()
