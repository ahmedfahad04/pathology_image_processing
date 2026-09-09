"""Build paper lookup JSON for viewer.html "paper only" filter.

Reads the 3 official paper files in output/paper/:
  - filtered_slides_tcga.xlsx  (MIBC / NMIBC review per patient)
  - loeffler_tcga.xlsx         (Loeffler et al. cases: AI score, prediction, histology)
  - mutations_blca_tcga_pancancer_atlas_cbioportal.txt (FGFR3 status per sample)

plus data/data_manifest.json (926 folders) to map patients -> manifest DX slides.

Writes output/paper/paper_lookup.json:
{
  "meta": {...counts, definitions...},
  "patients": { "TCGA-XX-XXXX": {...merged fields, flags...} },
  "paper_dx_files": [ manifest file_name, ... ]   # DX slides of union paper patients
}

"paper only" in the viewer = manifest entries whose file_name is in
paper_dx_files (union of MIBC-379 and Loeffler-391 experiments, restricted
to DX slides present in this repo's manifest). Both MUT and WT included.
"""
import json
from pathlib import Path

import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "output" / "paper"
MANIFEST = ROOT / "data" / "data_manifest.json"
OUT = PAPER / "paper_lookup.json"


def main():
    manifest = json.loads(MANIFEST.read_text())
    manifest_dx = [d for d in manifest if "-DX" in d["file_name"]]

    # ---- 1. filtered slides (407 patients) ----
    filt = pd.read_excel(PAPER / "filtered_slides_tcga.xlsx")
    filt = filt.rename(columns={"PATIENT_ID": "patient_id"})
    filt["DX_REV_Summary_norm"] = filt["DX_REV_Summary"].replace(
        {"NMIBC - pT1": "NMIBC", "NMIBC - pTa": "NMIBC"}
    )
    mibc_patients = set(
        filt.loc[filt["DX_REV_Summary_norm"] == "MIBC", "patient_id"]
    )

    # ---- 2. mutations (411 samples) ----
    mut = pd.read_csv(
        PAPER / "mutations_blca_tcga_pancancer_atlas_cbioportal.txt", sep="\t"
    )
    mut["patient_id"] = mut["SAMPLE_ID"].str.slice(0, 12)

    # ---- 3. loeffler (328 patients, incl. 1 row without label) ----
    wb = openpyxl.load_workbook(PAPER / "loeffler_tcga.xlsx", data_only=True)
    ws = wb.active
    loeffler_rows = {}
    for row in ws.iter_rows(min_row=4, values_only=True):
        sid = row[1]
        if not sid:
            continue
        sid = str(sid).strip()
        pid = sid[:12]
        loeffler_rows[pid] = {
            "loeffler_sample": sid,
            "age": row[2],
            "gender": row[3],
            "pT_stage": row[4],
            "grade_2004": row[5],
            "nodal_status": row[6],
            "tumor_other_histologic_subtype": row[7],
            "histological_subtype": row[8],
            "sample_type": row[9],
            "molecular_subtype": row[10],
            "noninvasive_therapy": row[11],
            "neoadjuvant_history": row[12],
            "papillary_morphology": row[13],
            "irregular_nuclei": row[14],
            "perinuclear_clearing": row[15],
            "distinct_cell_borders": row[16],
            "fgfr3_molecular_status": row[17],
            "pathologist_judgement": row[18],
            "ai_score": row[19],
            "ai_classification": row[20],
        }

    # ---- 4. experiment flags (HF feature sets, from previous analysis) ----
    # MIBC-379: patient in mibc AND has mutation record (replicates dataset.py)
    mut_pats = set(mut["patient_id"])
    mibc_experiment_pats = (mibc_patients & mut_pats)
    # Loeffler-391: patient with valid loeffler FGFR3 label
    loeffler_labeled = {
        pid
        for pid, r in loeffler_rows.items()
        if r["fgfr3_molecular_status"] in ("wt", "mut", "WT", "MUT")
    }

    # ---- 5. merge per patient ----
    filt_map = {r["patient_id"]: r for _, r in filt.iterrows()}
    mut_map = {r["patient_id"]: r for _, r in mut.iterrows()}
    patients = {}
    for pid in sorted(set(filt_map) | set(mut_map) | set(loeffler_rows)):
        fr = filt_map.get(pid, {})
        mr = mut_map.get(pid, {})
        lr = loeffler_rows.get(pid, {})
        patients[pid] = {
            "patient_id": pid,
            # filtered_slides_tcga.xlsx
            "cryo_section_rev": _s(fr.get("Cryo_Section_REV")),
            "cryo_summary": _s(fr.get("Cryo_Summary")),
            "dx_rev": _s(fr.get("DX_Rev")),
            "dx_rev_summary": _s(fr.get("DX_REV_Summary")),
            "exclusion_summary": _s(fr.get("Exclusion_Summary_Slides")),
            "is_mibc": pid in mibc_patients,
            # mutations file
            "mutation_sample": _s(mr.get("SAMPLE_ID")),
            "fgfr3_paper": _s(mr.get("FGFR3")),
            # loeffler file
            "in_loeffler": pid in loeffler_rows,
            **{k: _num(v) for k, v in lr.items() if k != "loeffler_sample"},
            "loeffler_sample": _s(lr.get("loeffler_sample")),
            # experiment flags
            "in_mibc_experiment": pid in mibc_experiment_pats,
            "in_loeffler_experiment": pid in loeffler_labeled,
        }

    # ---- 6. manifest DX files for union paper patients ----
    union_pats = mibc_experiment_pats | loeffler_labeled
    paper_dx_files = sorted(
        d["file_name"]
        for d in manifest_dx
        if d["patientId"] in union_pats
    )

    out = {
        "meta": {
            "source": "Bannier et al. Nat Commun 2024 + PABannier/fgfr3mut official repo/HF dataset",
            "paper_only_definition": (
                "Manifest DX (FFPE diagnostic) slides whose patient is in "
                "MIBC-379 (filtered MIBC + mutation record) OR Loeffler-391 "
                "(valid Loeffler FGFR3 label). Both MUT and WT included."
            ),
            "n_manifest_total": len(manifest),
            "n_manifest_dx": len(manifest_dx),
            "n_paper_dx_files": len(paper_dx_files),
            "n_mibc_experiment_patients": len(mibc_experiment_pats),
            "n_loeffler_experiment_patients": len(loeffler_labeled),
            "n_union_patients": len(union_pats),
            "note_uuid": (
                "2 HF feature slides (GU-A42P/GU-A42Q DX1) carry a different "
                "file UUID than the GDC copy in this repo; patient-level "
                "matching covers them via the TSV-UUID DX slides."
            ),
        },
        "patients": patients,
        "paper_dx_files": paper_dx_files,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT} ({len(paper_dx_files)} paper DX files, "
          f"{len(patients)} patients)")


def _s(v):
    if v is None:
        return None
    s = str(v).strip()
    return s if s not in ("", "nan", "None", "NA") else None


def _num(v):
    if isinstance(v, float) and pd.isna(v):
        return None
    if isinstance(v, str) and v.strip() in ("", "NA", "nan", "None"):
        return None
    return v


if __name__ == "__main__":
    main()
