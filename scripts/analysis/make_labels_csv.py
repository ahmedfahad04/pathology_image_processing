#!/usr/bin/env python3
"""Build ``results/labels.csv`` (slide,label) for the slides in ``results/tiles_out``.

WHAT THIS SCRIPT DOES
----------------------
``train_fgfr3mut.py`` and ``infer_fgfr3mut.py`` need a ground-truth file with
exactly two columns::

    slide,label
    TCGA-2F-A9KQ-01Z-00-DX1.1C8CB2DD-....svs,1

``slide`` is the slide's filename (``<name>.svs``, i.e. the name of the folder
that ``fgfr3mut_tile_wsi.py`` created under ``results/tiles_out/``) and
``label`` is ``1`` for FGFR3-mutant and ``0`` for FGFR3 wild-type.

This script generates that file automatically from the cBioPortal metadata that
was already downloaded into ``data/``, so labels never have to be typed by hand
and can never drift away from the genomics that was actually fetched.

WHERE THE LABEL COMES FROM
---------------------------
FGFR3 status is a property of the *sample* (e.g. ``TCGA-2F-A9KQ-01``), and the
sample id is encoded in every slide's GDC barcode, so the lookup is a join on
the slide filename. Sources are consulted in this order, and the source that
answered is reported for every slide:

  1. ``data/data_manifest.json``          per-slide ``FGFR3_status`` ("MUT"/"WT")
  2. ``data/cbioportal_summary_all.csv``  per-slide ``FGFR3_status`` ("MUTANT"/"WILD-TYPE")
  3. ``data/<slide>/cbioportal_<sample>_mutations.json``
                                           raw cBioPortal mutation records:
                                           an FGFR3 record means MUT
  4. sample-level fallback                status of any other slide of the
                                           same sample (all slides of a sample
                                           share the patient's genotype)

Usage:
  # regenerate ./results/labels.csv from ./results/tiles_out + ./data
  python scripts/analysis/make_labels_csv.py

  # explicit paths
  python scripts/analysis/make_labels_csv.py \
    --tiles_root ./results/tiles_out --data_dir ./data --out_csv ./results/labels.csv

  # preview without touching the file on disk
  python scripts/analysis/make_labels_csv.py --dry_run

Exit codes:
  0  labels.csv written (or would be, with --dry_run)
  1  at least one slide could not be labeled (nothing written unless --skip_unlabeled)
  2  bad input (missing tiles dir, no slides found, ...)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

GENE = "FGFR3"

# Status spellings seen across cBioPortal exports / this repo's own scripts.
MUTANT_TOKENS = {"MUT", "MUTANT", "MUTATED", "MUTATION", "MUTATIONAL", "1", "YES", "TRUE"}
WILD_TYPE_TOKENS = {"WT", "WILD-TYPE", "WILDTYPE", "WILD_TYPE", "WILD TYPE", "0", "NO", "FALSE", ""}

# GDC barcode: TCGA-XX-XXXX-01Z-00-DX1.<uuid>.svs
#   group(1) = patient id  (TCGA-XX-XXXX)
#   group(2) = sample code  (01)
BARCODE_RE = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-([0-9]{2})")


def log(msg: str) -> None:
    """Timestamped progress message to stdout (flushed so `tail -f` works)."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Slide / sample helpers
# ---------------------------------------------------------------------------
def parse_sample_id(slide: str) -> str | None:
    """``TCGA-2F-A9KQ-01Z-00-DX1.<uuid>.svs`` -> ``TCGA-2F-A9KQ-01``.

    The 4th barcode field ("01Z", "01A", ...) carries the two-character sample
    code; everything before it is the patient id.  Returns None for names that
    do not look like a TCGA barcode.
    """
    stem = Path(slide).name
    if stem.lower().endswith(".svs"):
        stem = stem[: -len(".svs")]
    m = BARCODE_RE.match(stem)
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}"


def status_to_label(status: str) -> int | None:
    """Map a status string to 1 (mutant) / 0 (wild-type) / None (unknown)."""
    s = (status or "").strip().upper()
    if s in MUTANT_TOKENS:
        return 1
    if s in WILD_TYPE_TOKENS:
        return 0
    return None


# ---------------------------------------------------------------------------
# Label sources
# ---------------------------------------------------------------------------
def load_manifest(data_dir: Path) -> dict[str, tuple[int, str]]:
    """file_name -> (label, sample_id) from data/data_manifest.json."""
    path = data_dir / "data_manifest.json"
    out: dict[str, tuple[int, str]] = {}
    if not path.exists():
        return out
    for rec in json.loads(path.read_text()):
        fn = rec.get("file_name")
        if not fn:
            continue
        label = status_to_label(rec.get("FGFR3_status", ""))
        if label is None:
            continue
        sample = rec.get("sampleId") or parse_sample_id(fn) or ""
        out[fn] = (label, sample)
    return out


def load_summary_csv(data_dir: Path) -> dict[str, tuple[int, str]]:
    """file_name -> (label, sample_id) from data/cbioportal_summary_all.csv."""
    path = data_dir / "cbioportal_summary_all.csv"
    out: dict[str, tuple[int, str]] = {}
    if not path.exists():
        return out
    with open(path, newline="") as fh:
        for rec in csv.DictReader(fh):
            fn = rec.get("file_name")
            if not fn:
                continue
            label = status_to_label(rec.get("FGFR3_status", ""))
            if label is None:
                continue
            sample = rec.get("sample_id") or parse_sample_id(fn) or ""
            out[fn] = (label, sample)
    return out


def label_from_mutation_json(folder: Path, sample_id: str | None) -> int | None:
    """Derive FGFR3 status from a slide's raw cBioPortal mutation records.

    Prefers ``cbioportal_<sample>_mutations.json`` (all genes) and falls back
    to ``cbioportal_<sample>_FGFR3_mutation.json`` (FGFR3 only).  An FGFR3
    record => mutant (1); a record set without FGFR3 => wild-type (0).
    """
    candidates: list[Path] = []
    if sample_id:
        candidates.append(folder / f"cbioportal_{sample_id}_mutations.json")
    candidates += [p for p in sorted(folder.glob("cbioportal_*_mutations.json"))
                   if p not in candidates]
    candidates += sorted(folder.glob("cbioportal_*_FGFR3_mutation.json"))

    for path in candidates:
        if not path.exists():
            continue
        try:
            records = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(records, list):
            continue

        if path.name.endswith("_FGFR3_mutation.json"):
            # FGFR3-only projection: any record at all means mutant.
            return 1 if records else 0

        for rec in records:
            gene = rec.get("gene")
            symbol = gene.get("hugoGeneSymbol") if isinstance(gene, dict) else None
            if symbol is None:
                symbol = rec.get("hugoGeneSymbol")
            if symbol == GENE:
                return 1
        # The all-genes file exists and holds no FGFR3 record -> wild-type.
        return 0
    return None


def find_slides(tiles_root: Path) -> list[str]:
    """Slide folder names under tiles_root, sorted.

    A folder counts as a slide when it looks like one produced by
    ``fgfr3mut_tile_wsi.py`` (it holds metadata.json / coords.npy).
    """
    slides = []
    for d in sorted(p for p in tiles_root.iterdir() if p.is_dir()):
        if (d / "metadata.json").exists() or (d / "coords.npy").exists():
            slides.append(d.name)
    return slides


# ---------------------------------------------------------------------------
# Label resolution
# ---------------------------------------------------------------------------
def resolve_labels(slides: list[str], data_dir: Path) -> tuple[list[dict], list[str]]:
    """Resolve one label per slide.  Returns (rows, unresolved_slide_names)."""
    manifest = load_manifest(data_dir)
    summary = load_summary_csv(data_dir)

    # Sample-level fallback table, built from the per-slide sources.
    sample_status: dict[str, int] = {}
    for fn, (label, sample) in list(manifest.items()) + list(summary.items()):
        if sample:
            sample_status.setdefault(sample, label)

    rows: list[dict] = []
    unresolved: list[str] = []

    for slide in slides:
        label = None
        source = ""

        if slide in manifest:
            label, _ = manifest[slide]
            source = "data_manifest.json"
        elif slide in summary:
            label, _ = summary[slide]
            source = "cbioportal_summary_all.csv"
        else:
            # Raw cBioPortal records stored per slide in data/<slide>/.
            folder = data_dir / Path(slide).stem
            if folder.is_dir():
                sample = parse_sample_id(slide)
                label = label_from_mutation_json(folder, sample)
                if label is not None:
                    source = f"{folder.name}/cbioportal_*_mutations.json"

        if label is None:
            # Last resort: the genotype of any other slide of the same sample.
            sample = parse_sample_id(slide)
            if sample and sample in sample_status:
                label = sample_status[sample]
                source = f"sample-level fallback ({sample})"

        if label is None:
            unresolved.append(slide)
            continue

        rows.append(
            {
                "slide": slide,
                "label": label,
                "status": "MUTANT" if label else "WILD-TYPE",
                "sample": parse_sample_id(slide) or "",
                "source": source,
            }
        )

    return rows, unresolved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tiles_root", default="./results/tiles_out",
                    help="folder holding one sub-folder per tiled slide (default: ./results/tiles_out)")
    ap.add_argument("--data_dir", default="./data",
                    help="folder holding the cBioPortal metadata (default: ./data)")
    ap.add_argument("--out_csv", default="./results/labels.csv",
                    help="where to write the labels (default: ./results/labels.csv)")
    ap.add_argument("--slides_dir", default=None,
                    help="optional folder of .svs slides; only used to warn about missing files")
    ap.add_argument("--skip_unlabeled", action="store_true",
                    help="write the slides that could be labeled and skip the rest (instead of failing)")
    ap.add_argument("--dry_run", action="store_true",
                    help="print what would be written, without touching --out_csv")
    args = ap.parse_args()

    tiles_root = Path(args.tiles_root)
    data_dir = Path(args.data_dir)
    out_csv = Path(args.out_csv)

    if not tiles_root.is_dir():
        log(f"ERROR: tiles root not found: {tiles_root}")
        return 2
    if not data_dir.is_dir():
        log(f"ERROR: data dir not found: {data_dir}")
        return 2

    slides = find_slides(tiles_root)
    if not slides:
        log(f"ERROR: no slide folders found under {tiles_root} "
            f"(expected one folder per slide, each with metadata.json)")
        return 2
    log(f"found {len(slides)} slide(s) under {tiles_root}")

    rows, unresolved = resolve_labels(slides, data_dir)

    # Report
    log("")
    log(f"{'slide':<70} {'label':<6} {'status':<11} source")
    for r in rows:
        log(f"{r['slide']:<70} {r['label']:<6} {r['status']:<11} {r['source']}")

    n_mut = sum(r["label"] for r in rows)
    log("")
    log(f"labeled: {len(rows)}  (MUTANT={n_mut}, WILD-TYPE={len(rows) - n_mut})")

    if unresolved:
        log(f"ERROR: could not determine FGFR3 status for {len(unresolved)} slide(s):")
        for s in unresolved:
            log(f"         {s}")
        log("       (run scripts/analysis/fetch_cbioportal_data.py for those slides, or "
            "pass --skip_unlabeled to write the rest)")
        if not args.skip_unlabeled:
            log(f"nothing written to {out_csv}")
            return 1

    if args.skip_unlabeled and unresolved:
        log(f"WARNING: skipping {len(unresolved)} unlabeled slide(s) (--skip_unlabeled)")

    if args.dry_run:
        log(f"dry run: {out_csv} not written")
        return 0

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")  # plain "\n", like the rest of the repo's CSVs
        w.writerow(["slide", "label"])
        for r in rows:
            w.writerow([r["slide"], r["label"]])
    log(f"wrote {out_csv} ({len(rows)} rows)")

    if args.slides_dir:
        sdir = Path(args.slides_dir)
        missing = [r["slide"] for r in rows if not (sdir / r["slide"]).exists()]
        if missing:
            log(f"WARNING: {len(missing)} slide(s) not found in {sdir}:")
            for s in missing:
                log(f"         {s}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
