"""
Build data/data_manifest.json for the SVS viewer (viewer.html Inspect tab).

The viewer loads data/data_manifest.json (an ARRAY of per-slide entries) and
then fetches detailed JSONs on demand from data/<folder>/cbioportal_*.json.
This script builds that manifest from:

  1. output/mutation_analysis/per_slide_table.csv   (cohort-wide table: one row
     per slide, FGFR3 status / hotspot / VAF / RSEM / clinical demographics)
  2. data/<folder>/cbioportal_<sample>_summary.json (freshly downloaded by
     fetch_cbioportal_data.py — overrides/refreshes the CSV values when present)

Important: the manifest "folder" field must be the folder NAME ONLY (no "data/"
prefix) because viewer.html builds paths as `data/${entry.folder}/...`.

Usage:
  python scripts/analysis/build_data_manifest.py
  python scripts/analysis/build_data_manifest.py --csv output/mutation_analysis/per_slide_table.csv --out data/data_manifest.json
"""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse_num(v):
    try:
        f = float(v)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return v if v not in (None, "") else None


def main():
    p = argparse.ArgumentParser(description="Build data/data_manifest.json for viewer.html")
    p.add_argument("--csv", default=str(ROOT / "output" / "mutation_analysis" / "per_slide_table.csv"),
                   help="Per-slide table CSV (from analyze_fgfr3_mutations.py)")
    p.add_argument("--data-dir", default=str(ROOT / "data"), help="Data directory with per-image folders")
    p.add_argument("--out", default=None, help="Output JSON path (default: <data-dir>/data_manifest.json)")
    args = p.parse_args()

    csv_path = Path(args.csv)
    data_dir = Path(args.data_dir)
    out_path = Path(args.out) if args.out else data_dir / "data_manifest.json"

    if not csv_path.exists():
        raise SystemExit(f"[error] per-slide table not found: {csv_path}")

    entries = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            folder = (row.get("folder") or "").strip().removeprefix("data/").strip("/\\")
            if not folder:
                continue
            entry = {
                "folder": folder,
                "file_name": (row.get("file_name") or f"{folder}.svs").strip(),
                "sampleId": (row.get("sampleId") or "").strip(),
                "patientId": (row.get("patientId") or "").strip(),
                "FGFR3_status": (row.get("FGFR3_status") or "WT").strip(),
                "hotspot": (row.get("hotspot") or "").strip(),
                "proteinChange_raw": (row.get("proteinChange_raw") or "").strip(),
                "VAF": parse_num(row.get("VAF")),
                "tumorAltCount": parse_num(row.get("tumorAltCount")),
                "tumorRefCount": parse_num(row.get("tumorRefCount")),
                "FGFR3_RSEM": parse_num(row.get("FGFR3_RSEM")),
                "FGFR3_CNA": (row.get("FGFR3_CNA") or "").strip(),
                "total_mutation_count": parse_num(row.get("total_mutation_count")),
                "AGE": parse_num(row.get("AGE")),
                "SEX": (row.get("SEX") or "").strip(),
                "STAGE": (row.get("STAGE") or "").strip(),
                "GRADE": (row.get("GRADE") or "").strip(),
                "TMB_CAT": (row.get("TMB_CAT") or "").strip(),
            }
            # parse fgfr3_mutations JSON blob if present
            raw = (row.get("fgfr3_mutations") or "").strip()
            if raw:
                try:
                    entry["fgfr3_mutations"] = json.loads(raw)
                except json.JSONDecodeError:
                    entry["fgfr3_mutations"] = raw

            # Refresh from the freshly downloaded per-folder summary (if any)
            summary_path = data_dir / folder / f"cbioportal_{entry['sampleId']}_summary.json"
            if summary_path.exists():
                try:
                    s = json.loads(summary_path.read_text(encoding="utf-8"))
                    entry["has_local_data"] = True
                    rsem = s.get("FGFR3_expression_RSEM")
                    if rsem is not None:
                        entry["FGFR3_RSEM"] = rsem
                    tot = s.get("total_mutations_in_sample")
                    if tot is not None:
                        entry["total_mutation_count"] = tot
                    stage = s.get("sample_clinical_STAGE")
                    if stage:
                        entry["STAGE"] = stage
                    st = (s.get("FGFR3_status") or "").upper()
                    if st.startswith("MUTANT"):
                        entry["FGFR3_status"] = "MUT"
                    elif st.startswith("WILD"):
                        entry["FGFR3_status"] = "WT"
                    details = s.get("FGFR3_mutation_details")
                    if details:
                        entry["fgfr3_mutations"] = details
                        if isinstance(details, list) and details:
                            entry["hotspot"] = details[0].get("proteinChange", entry["hotspot"])
                            entry["proteinChange_raw"] = details[0].get("proteinChange", entry["proteinChange_raw"])
                except Exception as e:  # keep CSV values if summary unreadable
                    print(f"  [warn] could not read {summary_path}: {e}")
            entries.append(entry)

    mut = sum(1 for e in entries if e["FGFR3_status"] == "MUT")
    wt = sum(1 for e in entries if e["FGFR3_status"] == "WT")
    local = sum(1 for e in entries if e.get("has_local_data"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} — {len(entries)} entries (MUT {mut} / WT {wt}, {local} with local per-folder JSONs)")


if __name__ == "__main__":
    main()
