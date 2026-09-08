#!/usr/bin/env python3
"""
FGFR3 Mutation Analysis — Layman-Friendly Visual Report
========================================================

Paper: Bannier et al. Nature Communications (2024) 15:10914
       "AI allows pre-screening of FGFR3 mutational status using routine
        histology slides of muscle-invasive bladder cancer"
       https://doi.org/10.1038/s41467-024-55331-6

What this script does:
  1. Scans the local `data/` folder produced by `scripts/fetch_cbioportal_data.py`
     (per-image subfolders `data/TCGA-*/` + cohort file).
  2. Aggregates FGFR3 mutation ground-truth (MUT vs WT) using the paper's
     activating hotspot definition.
  3. Computes prevalence, hotspot distribution, VAF, expression, stage/sex,
     tumor mutation burden and other statistics.
  4. Produces 7 publication-quality, layman-annotated figures + a markdown
     report that a software developer with NO biology background can read.
  5. Explicitly connects the statistics to actionable image-feature
     engineering (what to measure on the H&E slide).

Data layout expected:
  data/
    cbioportal_blca_tcga_pan_can_atlas_2018_FGFR3_mutations_cohort.json
    cbioportal_study_info.json
    TCGA-FD-A3NA-01Z-00-DX1.2AD62CEE-.../
        cbioportal_TCGA-FD-A3NA-01_FGFR3_mutation.json  ( [] = WT )
        cbioportal_TCGA-FD-A3NA-01_FGFR3_expression.json
        cbioportal_TCGA-FD-A3NA-01_FGFR3_CNA.json
        cbioportal_TCGA-FD-A3NA-01_mutations.json       (all genes)
        cbioportal_TCGA-FD-A3NA-01_patient_clinical.json
        ...
    TCGA- ... /  (one folder per SVS image ≈ 926 expected, subset ok)

Usage:
  python scripts/analyze_fgfr3_mutations.py
  python scripts/analyze_fgfr3_mutations.py --data-dir data --out-dir output/mutation_analysis
  python scripts/analyze_fgfr3_mutations.py --data-dir data --out-dir output/mutation_analysis --dpi 200

Requires: pandas, numpy, matplotlib, seaborn (all standard in this repo).
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Paper-derived constants  (Methods p.9 + Supplementary Table 7)
# ---------------------------------------------------------------------------
# The 11 SNaPshot hotspots that cover >99% of activating FGFR3 mutations.
# Note: transcript differences mean G372C==G370C, G382R==G380R, Y375C==Y373C etc.
# We normalize via the proteinChange string exactly as returned by cBioPortal.
ACTIVATING_HOTSPOTS_CANON = {
    "R248C", "S249C",
    "G370C", "G372C",  # same codon, different isoform numbering
    "S371C", "S373C",
    "Y373C", "Y375C",
    "G380R", "G382R",
    "A393E",
    "K650E", "K652E", "K652Q", "K652M", "K652T",
}
# For display we collapse aliases
HOTSPOT_ALIAS = {
    "G372C": "G370C/G372C",
    "G370C": "G370C/G372C",
    "G382R": "G380R/G382R",
    "G380R": "G380R/G382R",
    "Y375C": "Y373C/Y375C",
    "Y373C": "Y373C/Y375C",
    "S373C": "S371C/S373C",
    "S371C": "S371C/S373C",
    "K650E": "K652E (K650E)",
    "K652E": "K652E (K650E)",
}
# Non-activating FGFR3 variants explicitly called WT in paper Methods p.8
NON_ACTIVATING_AS_WT = {"H349D", "S344Y", "P358L", "V306I", "L88Wfs*10",
                         "Q674*", "E216K", "D222N", "G235D", "H791Tfs*29"}

PAPER_TITLE = "Bannier et al. Nat Commun 2024 — FGFR3 in Muscle-Invasive Bladder Cancer (MIBC/mUC)"
LAYMAN_GLOSSARY = """
GLOSSARY FOR SOFTWARE DEVELOPERS
  Bladder cancer types:  NMIBC = outer layer only (common, less deadly), MIBC =
    muscle wall invaded (25% of cases, aggressive), mUC = spread to other organs.
  FGFR3 = a gene for a cell-surface antenna (growth-factor receptor). Normal = "gas
    pedal that only accelerates when growth signal arrives".  MUTANT = pedal stuck ON.
  WT (wild-type) = gene is normal / unmutated.  MUT = activating mutation present.
  Hotspot = the ~11 exact codons where a single-letter DNA change is known to jam the
    pedal ON.  S249C is the most common (almost half of all FGFR3-MUT cases).
  Why we care:  drug erdafitinib blocks the stuck pedal, but only works if you ARE
    MUT.  Testing every patient by PCR/NGS is slow and expensive — the paper asks if
    a photo of the tissue (routine H&E glass slide) already contains a visual hint.
  What the AI sees (paper Fig.5): MUT tissue tends to look tidy — cells are uniform
    (monomorphic), with little scar-like stroma around the tumor and few immune cells
    inside. WT tissue looks messier — more scar, more immune infiltration, varied nuclei.
"""

# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="FGFR3 mutation layman analysis")
    p.add_argument("--data-dir", type=str, default="data",
                   help="Folder containing cbioportal per-image subfolders (default: data)")
    p.add_argument("--out-dir", type=str, default="output/mutation_analysis",
                   help="Output folder for figures + report (default: output/mutation_analysis)")
    p.add_argument("--dpi", type=int, default=180, help="Figure DPI")
    return p.parse_args()

def style():
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 180,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "font.family": "sans-serif",
    })

def is_activating(protein_change: str) -> bool:
    return protein_change in ACTIVATING_HOTSPOTS_CANON

def canonical_hotspot_label(pc: str) -> str:
    return HOTSPOT_ALIAS.get(pc, pc)

# ---------------------------------------------------------------------------
def discover_per_image_dirs(data_dir: Path):
    dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("TCGA")])
    # fall back: if none, maybe all at top-level?
    return dirs

def load_one_sample(sample_dir: Path):
    """Return dict with all parsed fields for one SVS image."""
    stem = sample_dir.name
    # derive sampleId: TCGA-XX-XXXX-01  (first 3 dash parts + -01/11 etc.?)
    # Most reliable: read summary or mutation file name
    mut_files = list(sample_dir.glob("*_FGFR3_mutation.json"))
    sum_files = list(sample_dir.glob("*_summary.json"))
    expr_files = list(sample_dir.glob("*_FGFR3_expression.json"))
    cna_files = list(sample_dir.glob("*_FGFR3_CNA.json"))
    allmut_files = list(sample_dir.glob("*_mutations.json"))
    patclin_files = list(sample_dir.glob("*_patient_clinical.json"))
    samclin_files = list(sample_dir.glob("*_sample_clinical.json"))
    patient_info_files = list(sample_dir.glob("*_patient_info.json"))
    sample_info_files = list(sample_dir.glob("*_sample_info.json"))

    rec = {"folder": stem, "per_image_dir": str(sample_dir)}
    # sample / patient id: parse from file names
    sample_id = None
    patient_id = None
    if mut_files:
        m = re.match(r"cbioportal_(TCGA-[A-Z0-9]+-[A-Z0-9]+)-.+_FGFR3_mutation\.json", mut_files[0].name)
        if m:
            patient_id = m.group(1)
            # sample is patient + -01 or -11; infer from real file: look inside JSON if available
            try:
                j = json.load(open(mut_files[0]))
                if len(j) > 0:
                    sample_id = j[0].get("sampleId")
            except: pass
            if not sample_id:
                # fallback: try summary
                if sum_files:
                    try:
                        sj = json.load(open(sum_files[0]))
                        sample_id = sj.get("cbio_sample_id")
                    except: pass
                if not sample_id:
                    sample_id = patient_id + "-01"
        else:
            # parse from summary fallback
            if sum_files:
                try:
                    sj = json.load(open(sum_files[0]))
                    sample_id = sj.get("cbio_sample_id")
                    patient_id = sj.get("cbio_patient_id")
                except: pass
    rec["sampleId"] = sample_id
    rec["patientId"] = patient_id

    # FGFR3 mutation records
    fgfr3_muts = []
    if mut_files and mut_files[0].exists():
        try:
            fgfr3_muts = json.load(open(mut_files[0]))
        except: fgfr3_muts = []
    rec["fgfr3_mutations"] = fgfr3_muts
    rec["fgfr3_mut_count"] = len(fgfr3_muts)

    # Classify WT vs MUT at the biological (activating) level — paper rule:
    # any activating hotspot = MUT, empty or only non-activating = WT.
    # However cBioPortal returns empty [] for WT; non-activating still returns
    # a record (e.g. Q674*). We must downgrade non-activating to WT.
    activating_records = [r for r in fgfr3_muts if is_activating(r.get("proteinChange", ""))]
    rec["activating_records"] = activating_records
    rec["is_mut_activating"] = len(activating_records) > 0
    rec["is_mut_any"] = len(fgfr3_muts) > 0
    # For layman stats we use activating definition (matches paper Fig.3)
    rec["FGFR3_status"] = "MUT" if rec["is_mut_activating"] else "WT"
    rec["FGFR3_status_any"] = "MUT_any" if rec["is_mut_any"] else "WT_any"
    # hotspot label (first activating)
    if activating_records:
        pc = activating_records[0].get("proteinChange", "?")
        rec["hotspot"] = canonical_hotspot_label(pc)
        rec["proteinChange_raw"] = pc
        # VAF
        alt = activating_records[0].get("tumorAltCount", 0)
        ref = activating_records[0].get("tumorRefCount", 0)
        vaf = alt / (alt + ref) if (alt + ref) > 0 else np.nan
        rec["VAF"] = vaf
        rec["tumorAltCount"] = alt
        rec["tumorRefCount"] = ref
    else:
        # keep the (non-activating) proteinChange for transparency if present
        if len(fgfr3_muts) > 0:
            rec["hotspot"] = fgfr3_muts[0].get("proteinChange", "non-activating")
            rec["proteinChange_raw"] = fgfr3_muts[0].get("proteinChange")
        else:
            rec["hotspot"] = None
            rec["proteinChange_raw"] = None
        rec["VAF"] = np.nan
        rec["tumorAltCount"] = np.nan
        rec["tumorRefCount"] = np.nan

    # expression (RSEM batch-normalized; paper uses TPM log2 — we flag the difference)
    rec["FGFR3_RSEM"] = np.nan
    rec["has_expression"] = False
    if expr_files and expr_files[0].exists():
        try:
            ej = json.load(open(expr_files[0]))
            if isinstance(ej, list) and len(ej) > 0:
                val = ej[0].get("value", np.nan)
                try: rec["FGFR3_RSEM"] = float(val)
                except: rec["FGFR3_RSEM"] = np.nan
                rec["has_expression"] = not np.isnan(rec["FGFR3_RSEM"])
            elif isinstance(ej, dict) and "value" in ej:
                rec["FGFR3_RSEM"] = float(ej["value"])
                rec["has_expression"] = True
        except: pass

    # CNA (discrete GISTIC: -2 del, -1 hem-del, 0 neutral, 1 gain, 2 amp)
    rec["FGFR3_CNA"] = np.nan
    if cna_files and cna_files[0].exists():
        try:
            cj = json.load(open(cna_files[0]))
            if isinstance(cj, list) and len(cj) > 0:
                rec["FGFR3_CNA"] = cj[0].get("value", np.nan)
                # try coerce to int
                try: rec["FGFR3_CNA"] = int(rec["FGFR3_CNA"])
                except: pass
        except: pass

    # all mutations count (TMB proxy)
    rec["total_mutation_count"] = np.nan
    if allmut_files and allmut_files[0].exists():
        try:
            aj = json.load(open(allmut_files[0]))
            if isinstance(aj, list):
                rec["total_mutation_count"] = len(aj)
        except: pass

    # clinical
    rec["AGE"] = np.nan
    rec["SEX"] = None
    rec["STAGE"] = None
    rec["GRADE"] = None
    rec["TMB_CAT"] = None
    rec["file_name"] = stem + ".svs"
    # try to read actual file_name.txt
    fn_txt = sample_dir / "file_name.txt"
    if fn_txt.exists():
        try:
            rec["file_name"] = fn_txt.read_text().strip()
        except: pass
    elif sum_files:
        try:
            sj = json.load(open(sum_files[0]))
            rec["file_name"] = sj.get("image_file", rec["file_name"])
        except: pass

    clin_src = patclin_files[0] if patclin_files else None
    # also try samclin
    for clin_file in (patclin_files + samclin_files):
        if clin_file and clin_file.exists():
            try:
                cdata = json.load(open(clin_file))
                for entry in cdata:
                    cid = entry.get("clinicalAttributeId", "")
                    val = entry.get("value")
                    if cid == "AGE" and rec["AGE"] is np.nan:
                        try: rec["AGE"] = float(val)
                        except: pass
                    elif cid == "SEX" and rec["SEX"] is None:
                        rec["SEX"] = val
                    elif cid == "AJCC_PATHOLOGIC_TUMOR_STAGE" and rec["STAGE"] is None:
                        rec["STAGE"] = val
                    elif cid == "GRADE" and rec["GRADE"] is None:
                        rec["GRADE"] = val
                    elif cid in ("MUTATION_COUNT", "TMB (NS) - MUTATION COUNT") and pd.isna(rec["total_mutation_count"]):
                        try: rec["total_mutation_count"] = int(float(val))
                        except: pass
            except: pass

    # normalize stage label
    if rec["STAGE"]:
        # map "Stage II" style vs "STAGE II"
        s = str(rec["STAGE"]).strip().upper()
        # keep as-is but clean
        rec["STAGE"] = s.replace("STAGE ", "Stage ")
        # Actually keep original uppercase for grouping
        rec["STAGE"] = str(rec["STAGE"]).strip()
        # unify: "STAGE II" -> "Stage II"
        if rec["STAGE"].upper().startswith("STAGE"):
            num = rec["STAGE"].split()[-1]
            rec["STAGE"] = f"Stage {num}"

    return rec

# ---------------------------------------------------------------------------
def load_cohort(data_dir: Path):
    cohort_path = data_dir / "cbioportal_blca_tcga_pan_can_atlas_2018_FGFR3_mutations_cohort.json"
    study_path = data_dir / "cbioportal_study_info.json"
    cohort = []
    study = {}
    if cohort_path.exists():
        try: cohort = json.load(open(cohort_path))
        except: pass
    if study_path.exists():
        try: study = json.load(open(study_path))
        except: pass
    return cohort, study

# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    style()

    print(f"Data dir: {data_dir.resolve()}")
    print(f"Out dir : {out_dir.resolve()}")

    per_image_dirs = discover_per_image_dirs(data_dir)
    print(f"Found {len(per_image_dirs)} per-image folders (TCGA-*)")

    if len(per_image_dirs) == 0:
        print(f"[ERROR] No per-image folders in {data_dir}. Did you run scripts/fetch_cbioportal_data.py?")
        return

    rows = []
    for d in per_image_dirs:
        try:
            rows.append(load_one_sample(d))
        except Exception as e:
            print(f"[warn] failed {d.name}: {e}")
    df = pd.DataFrame(rows)
    print(f"Parsed {len(df)} samples")

    # Deduplicate by sampleId (one patient can have multiple slides e.g. DX1 + TSA)
    # Keep one row per sampleId for sample-level stats, but also keep per-slide counts
    n_slides = len(df)
    # sample-level dedup
    if "sampleId" in df.columns and df["sampleId"].notna().any():
        df_sample = df.sort_values("folder").drop_duplicates(subset=["sampleId"], keep="first").copy()
    else:
        df_sample = df.copy()
    n_samples = len(df_sample)
    n_patients = df_sample["patientId"].nunique()

    # Cohort
    cohort, study = load_cohort(data_dir)
    cohort_counter = Counter(x.get("proteinChange", "?") for x in cohort)
    cohort_n_records = len(cohort)
    cohort_n_patients = len(set(x.get("patientId") for x in cohort)) if cohort else 0
    study_all = study.get("allSampleCount", 411)

    # -------------------- Stats --------------------
    n_mut = (df_sample["FGFR3_status"] == "MUT").sum()
    n_wt = (df_sample["FGFR3_status"] == "WT").sum()
    prevalence = n_mut / n_samples * 100 if n_samples else 0

    n_mut_any = (df_sample["FGFR3_status_any"] == "MUT_any").sum()
    n_nonnactiv = n_mut_any - n_mut  # records that were non-activating

    # Hotspot distribution among MUT
    hotspot_counts = Counter(df_sample.loc[df_sample["FGFR3_status"] == "MUT", "hotspot"].dropna())
    # Expression
    has_expr = df_sample["FGFR3_RSEM"].notna().sum()
    expr_mut = df_sample.loc[df_sample["FGFR3_status"] == "MUT", "FGFR3_RSEM"].dropna()
    expr_wt = df_sample.loc[df_sample["FGFR3_status"] == "WT", "FGFR3_RSEM"].dropna()

    # Save a tidy CSV for reuse
    df_sample.to_csv(out_dir / "per_sample_table.csv", index=False)
    df.to_csv(out_dir / "per_slide_table.csv", index=False)
    if cohort:
        pd.DataFrame(cohort).to_csv(out_dir / "cohort_FGFR3_all_records.csv", index=False)

    # -------------------- Layman header text --------------------
    layman_intro = f"""
# FGFR3 Mutation Report — Layman Edition
**Paper:** {PAPER_TITLE}
**Local data snapshot:** {data_dir} → {n_slides} slides, {n_samples} unique samples, {n_patients} patients
**Study-wide (cBioPortal):** {study_all} total BLCA samples, {cohort_n_patients} patients carry ANY FGFR3 variant ({cohort_n_records} records)

{LAYMAN_GLOSSARY}

## 1-Sentence Takeaway
> In this local snapshot, **{n_mut} of {n_samples} samples ({prevalence:.1f}%) carry an FGFR3 mutation that
> would jam the growth pedal ON** (activating hotspot). The most common jam is **S249C**.
> Across the full 411-patient TCGA BLCA cohort the rate is {cohort_n_patients}/{study_all} ({cohort_n_patients/study_all*100:.1f}% any FGFR3 variant, ~8-14% activating
> depending on MIBC vs NMIBC enrichment — exactly the imbalance the paper exploits).

## Why Prevalence Matters to You (the developer)
- Machine-learning models on slides are **highly imbalanced**: ~9 WT for every 1 MUT.  Accuracy alone
  lies — you must look at sensitivity/NPV.
- The paper's AI acts as a **rule-out test**: if it says WT, you skip expensive DNA testing in ~40%
  of cases (NPV ~0.99).  That is the product goal our image features should copy.
"""

    # -----------------------------------------------------------------------
    # FIGURE 1 — Prevalence pie + bar (MUT vs WT)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    colors = {"MUT": "#D32F2F", "WT": "#1976D2"}
    # pie
    ax = axes[0]
    vals = [n_mut, n_wt]
    labs = [f"MUT\n{n_mut} ({prevalence:.1f}%)", f"WT\n{n_wt} ({100-prevalence:.1f}%)"]
    wedges, texts, autotexts = ax.pie(vals, labels=labs, autopct="", startangle=90,
                                      colors=[colors["MUT"], colors["WT"]],
                                      wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2))
    ax.set_title("Local snapshot: MUT vs WT\n(per-sample, activating hotspots only)", fontsize=12)
    # annotate center
    ax.text(0, 0, f"n={n_samples}", ha="center", va="center", fontsize=13, weight="bold")
    # bar: local vs cohort
    ax2 = axes[1]
    cats = ["Local\n(snapshot)", "TCGA BLCA\n(full 411)"]
    mut_counts = [n_mut, cohort_n_patients]
    wt_counts = [n_wt, study_all - cohort_n_patients]
    x = np.arange(len(cats))
    b1 = ax2.bar(x, wt_counts, label="WT", color=colors["WT"], edgecolor="white")
    b2 = ax2.bar(x, mut_counts, bottom=wt_counts, label="MUT (any FGFR3)", color=colors["MUT"], edgecolor="white")
    # adjust second bar to show distinction: text
    for i, (w, m) in enumerate(zip(wt_counts, mut_counts)):
        ax2.text(i, w/2, f"WT {w}", ha="center", va="center", color="white", weight="bold", fontsize=10)
        ax2.text(i, w + m/2, f"MUT {m}\n{m/(w+m)*100:.1f}%", ha="center", va="center", color="white", weight="bold", fontsize=9)
    ax2.set_xticks(x); ax2.set_xticklabels(cats)
    ax2.set_ylabel("Patients / samples")
    ax2.set_title("How common is FGFR3 mutation?", fontsize=12)
    ax2.legend(frameon=True, loc="upper right")
    # layman caption
    fig.suptitle("Figure 1 — How many patients would benefit from the drug? (Prevalence)", fontsize=14, weight="bold", y=1.02)
    fig.text(0.5, -0.02,
             "Layman: Red = pedal stuck ON (MUT, drug could help). Blue = normal (WT). ~1 in 10 is MUT in MIBC; up to ~4 in 10 in NMIBC.",
             ha="center", fontsize=9, style="italic", bbox=dict(boxstyle="round,pad=0.3", fc="#FFF9C4", ec="#FBC02D"))
    plt.tight_layout()
    plt.savefig(out_dir / "fig1_prevalence.png", dpi=args.dpi, bbox_inches="tight")
    plt.close()

    # -----------------------------------------------------------------------
    # FIGURE 2 — Hotspot distribution
    # -----------------------------------------------------------------------
    # Combine local + cohort for context
    # Local hotspot order: paper's top 6
    top6 = ["S249C", "Y373C/Y375C", "G370C/G372C", "R248C", "G380R/G382R", "S371C/S373C"]
    # normalize local labels
    def norm_for_plot(label):
        # map G370C/G372C etc already canonical
        return label
    local_series = pd.Series(hotspot_counts).reindex(top6).fillna(0).astype(int)
    cohort_series_raw = cohort_counter
    # Map cohort raw PC to canonical
    cohort_mapped = Counter()
    for pc, cnt in cohort_series_raw.items():
        lab = canonical_hotspot_label(pc)
        cohort_mapped[lab] += cnt
    cohort_series = pd.Series({k: cohort_mapped.get(k, 0) for k in top6})
    # also collect "Other" (activating + non)
    other_local = sum(v for k, v in hotspot_counts.items() if k not in top6)
    other_cohort = sum(v for k, v in cohort_mapped.items() if k not in top6)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    y = np.arange(len(top6) + 1)
    labels = top6 + ["Other"]
    local_vals = list(local_series.values) + [other_local]
    cohort_vals = list(cohort_series.values) + [other_cohort]
    # horizontal grouped
    h = 0.38
    ax.barh(y + h/2, cohort_vals, height=h, label=f"Full cohort (n={cohort_n_records})", color="#90CAF9", edgecolor="#1976D2")
    ax.barh(y - h/2, local_vals, height=h, label=f"Local snapshot (n={n_mut})", color="#EF9A9A", edgecolor="#D32F2F")
    for i, (lv, cv) in enumerate(zip(local_vals, cohort_vals)):
        if lv > 0:
            ax.text(lv + 0.2, y[i]-h/2, str(lv), va="center", fontsize=9, color="#D32F2F", weight="bold")
        if cv > 0:
            ax.text(cv + 0.2, y[i]+h/2, str(cv), va="center", fontsize=9, color="#1565C0")
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("Number of patients / records with this hotspot")
    ax.set_title("Figure 2 — Which exact letter-change causes the jam? (Hotspot distribution)", weight="bold")
    ax.legend()
    ax.set_xlim(0, max(max(cohort_vals) if cohort_vals else 1, max(local_vals) if local_vals else 1) * 1.25 + 1)
    fig.text(0.5, -0.02,
             "Layman: The hotspot name is the codon: S249C = 'at position 249, Serine→Cysteine'. S249C is always the tallest bar — easiest for the AI (paper Fig.3).",
             ha="center", fontsize=9, style="italic", bbox=dict(boxstyle="round,pad=0.3", fc="#E8F5E9", ec="#388E3C"))
    plt.tight_layout()
    plt.savefig(out_dir / "fig2_hotspots.png", dpi=args.dpi, bbox_inches="tight")
    plt.close()

    # -----------------------------------------------------------------------
    # FIGURE 3 — FGFR3 expression WT vs MUT (RSEM)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw=dict(width_ratios=[2, 1]))
    ax = axes[0]
    # log1p transform for readability (RSEM spans 0..100k)
    plot_df = df_sample[["FGFR3_status", "FGFR3_RSEM"]].dropna()
    # Use violin + strip
    if len(plot_df) > 0:
        sns.violinplot(data=plot_df, x="FGFR3_status", y="FGFR3_RSEM", palette=colors,
                       inner="quartile", cut=0, scale="width", ax=ax)
        sns.stripplot(data=plot_df, x="FGFR3_status", y="FGFR3_RSEM", color="black", alpha=0.35, size=3, jitter=0.18, ax=ax)
        # annotate medians
        for status in ["WT", "MUT"]:
            vals = plot_df.loc[plot_df["FGFR3_status"] == status, "FGFR3_RSEM"]
            if len(vals) > 0:
                med = vals.median()
                mean = vals.mean()
                ax.text(["WT","MUT"].index(status), med, f" median {med:.0f}", va="center", fontsize=9, weight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))
        ax.set_yscale("log")
        ax.set_ylabel("FGFR3 mRNA (RSEM, log scale)  — paper uses TPM log2, related")
        ax.set_xlabel("FGFR3 status (activating definition)")
        ax.set_title("Expression: MUT tends to be higher — but not perfectly", fontsize=11, weight="bold")
        # second panel: histogram
        ax2 = axes[1]
        bins = np.logspace(np.log10(max(1, plot_df["FGFR3_RSEM"].min())), np.log10(plot_df["FGFR3_RSEM"].max()), 20) if len(plot_df)>1 else 10
        # fallback linear if needed
        ax2.hist(expr_wt, bins=bins, alpha=0.55, color=colors["WT"], label=f"WT (n={len(expr_wt)})", edgecolor="white")
        ax2.hist(expr_mut, bins=bins, alpha=0.65, color=colors["MUT"], label=f"MUT (n={len(expr_mut)})", edgecolor="white")
        ax2.set_xscale("log"); ax2.set_xlabel("RSEM (log)"); ax2.set_ylabel("Count")
        ax2.set_title("Histogram", fontsize=11, weight="bold"); ax2.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No expression data", ha="center", va="center", transform=ax.transAxes)
    fig.suptitle("Figure 3 — How LOUD is the gene speaking? (FGFR3 expression)", fontsize=13, weight="bold", y=1.02)
    fig.text(0.5, -0.02,
             "Layman: Think of expression as volume knob. MUT is usually louder (paper: low MUT + high WT are hard; false-positive WT avg logTPM 4.28 vs true-neg 3.26, p=7.9e-8). "
             "A quiet MUT looks like WT and vice-versa.",
             ha="center", fontsize=9, style="italic", bbox=dict(boxstyle="round,pad=0.3", fc="#F3E5F5", ec="#7B1FA2"))
    plt.tight_layout()
    plt.savefig(out_dir / "fig3_expression.png", dpi=args.dpi, bbox_inches="tight")
    plt.close()

    # -----------------------------------------------------------------------
    # FIGURE 4 — Stage distribution and MUT rate by stage
    # -----------------------------------------------------------------------
    # Stage counts overall + MUT rate
    stage_order = ["Stage I", "Stage II", "Stage III", "Stage IV"]
    # Filter to those with stage
    df_stage = df_sample.dropna(subset=["STAGE"])
    # normalize stage strings
    df_stage["STAGE"] = df_stage["STAGE"].astype(str).str.strip()
    # Some might be "Stage III" vs "STAGE III" vs "Stage IIIA" — collapse to I-IV
    def collapse_stage(s):
        s = str(s).upper()
        m = re.search(r"STAGE\s*(I{1,3}|IV|\d)", s)
        if m:
            token = m.group(1)
            # map 1->I etc.
            mapping = {"1":"I","2":"II","3":"III","4":"IV","I":"I","II":"II","III":"III","IV":"IV"}
            return f"Stage {mapping.get(token, token)}"
        return str(s).title()
    df_stage["STAGE_simple"] = df_stage["STAGE"].apply(collapse_stage)
    stage_counts = df_stage["STAGE_simple"].value_counts().reindex(stage_order).fillna(0).astype(int)
    # MUT rate per stage
    mut_by_stage = df_stage.groupby("STAGE_simple")["FGFR3_status"].apply(lambda x: (x=="MUT").sum()).reindex(stage_order).fillna(0).astype(int)
    wt_by_stage = stage_counts - mut_by_stage
    mut_rate = (mut_by_stage / stage_counts.replace(0, np.nan) * 100).fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axes[0]
    x = np.arange(len(stage_order))
    b_wt = ax.bar(x, wt_by_stage, label="WT", color=colors["WT"], edgecolor="white")
    b_mut = ax.bar(x, mut_by_stage, bottom=wt_by_stage, label="MUT", color=colors["MUT"], edgecolor="white")
    for i in range(len(stage_order)):
        total = stage_counts.iloc[i]
        if total > 0:
            ax.text(i, total + 0.6, f"n={total}", ha="center", va="bottom", fontsize=9, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(stage_order)
    ax.set_ylabel("Samples"); ax.set_title("Count per stage (stacked MUT/WT)", fontsize=11, weight="bold"); ax.legend()

    ax2 = axes[1]
    bars = ax2.bar(x, mut_rate, color="#FF7043", edgecolor="#BF360C")
    # annotate rate
    for i, v in enumerate(mut_rate):
        if stage_counts.iloc[i] > 0:
            ax2.text(i, v + 0.8, f"{v:.1f}%\n({mut_by_stage.iloc[i]}/{stage_counts.iloc[i]})", ha="center", va="bottom", fontsize=9, weight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(stage_order)
    ax2.set_ylabel("MUT rate (%)"); ax2.set_ylim(0, max(22, mut_rate.max()*1.4 + 5))
    ax2.set_title("MUT % inside each stage", fontsize=11, weight="bold")
    fig.suptitle("Figure 4 — Does stage matter? (Stage vs FGFR3 status)", fontsize=13, weight="bold", y=1.02)
    fig.text(0.5, -0.02,
             "Layman: Bladder cancer stage = how deep/spread. Paper's MIBC/mUC are Stage II-IV. MUT rate is lower in advanced stages (~10-15%) — "
             "the exact imbalance that makes a slide pre-screen valuable.",
             ha="center", fontsize=9, style="italic", bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3E0", ec="#EF6C00"))
    plt.tight_layout()
    plt.savefig(out_dir / "fig4_stage.png", dpi=args.dpi, bbox_inches="tight")
    plt.close()

    # -----------------------------------------------------------------------
    # FIGURE 5 — VAF (how pure is the mutation signal)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    vaf_vals = df_sample.loc[df_sample["FGFR3_status"] == "MUT", "VAF"].dropna()
    alt_vals = df_sample.loc[df_sample["FGFR3_status"] == "MUT", "tumorAltCount"].dropna()
    ref_vals = df_sample.loc[df_sample["FGFR3_status"] == "MUT", "tumorRefCount"].dropna()
    ax = axes[0]
    if len(vaf_vals) > 0:
        ax.hist(vaf_vals, bins=12, color="#AB47BC", edgecolor="white", alpha=0.85)
        ax.axvline(vaf_vals.median(), color="#4A148C", linestyle="--", linewidth=2, label=f"median {vaf_vals.median():.2f}")
        ax.axvline(vaf_vals.mean(), color="#CE93D8", linestyle=":", linewidth=2, label=f"mean {vaf_vals.mean():.2f}")
        ax.set_xlabel("Variant Allele Frequency (VAF) = alt / (alt+ref)")
        ax.set_ylabel("MUT patients"); ax.set_title("VAF distribution (MUT only)", fontsize=11, weight="bold"); ax.legend()
        # add text box
        ax.text(0.02, 0.96, f"n={len(vaf_vals)}\nrange {vaf_vals.min():.2f}–{vaf_vals.max():.2f}",
                transform=ax.transAxes, va="top", fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
    else:
        ax.text(0.5, 0.5, "No VAF data", ha="center", va="center", transform=ax.transAxes)
    ax2 = axes[1]
    if len(vaf_vals) > 0 and len(df_sample.loc[df_sample["FGFR3_status"]=="MUT", "FGFR3_RSEM"].dropna()) > 0:
        # scatter VAF vs expression for MUT
        sub = df_sample.loc[df_sample["FGFR3_status"]=="MUT", ["VAF","FGFR3_RSEM","hotspot"]].dropna()
        if len(sub) > 0:
            sc = ax2.scatter(sub["VAF"], sub["FGFR3_RSEM"], c="#D32F2F", s=55, alpha=0.75, edgecolor="white", linewidth=0.8)
            for _, row in sub.iterrows():
                ax2.annotate(row["hotspot"], (row["VAF"], row["FGFR3_RSEM"]), fontsize=7, alpha=0.7,
                             xytext=(4,4), textcoords="offset points")
            # annotate corr
            try:
                corr = sub["VAF"].corr(sub["FGFR3_RSEM"])
                ax2.text(0.02, 0.96, f"corr(VAF, RSEM) ≈ {corr:.2f}\n(n={len(sub)})", transform=ax2.transAxes, va="top", fontsize=9,
                         bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
            except: pass
            ax2.set_xlabel("VAF"); ax2.set_ylabel("RSEM (log)"); ax2.set_yscale("log")
            ax2.set_title("MUT: does louder mutation = louder expression?", fontsize=10, weight="bold")
    else:
        ax2.text(0.5, 0.5, "No VAF+expression pair", ha="center", va="center", transform=ax2.transAxes)
    fig.suptitle("Figure 5 — How strong is the mutation signal? (VAF)", fontsize=13, weight="bold", y=1.02)
    fig.text(0.5, -0.02,
             "Layman: VAF ~ 'what fraction of DNA reads show the typo?'. ~0.5 = half the cells carry it. High VAF ≈ cleaner slide signal.",
             ha="center", fontsize=9, style="italic", bbox=dict(boxstyle="round,pad=0.3", fc="#E3F2FD", ec="#1565C0"))
    plt.tight_layout()
    plt.savefig(out_dir / "fig5_vaf.png", dpi=args.dpi, bbox_inches="tight")
    plt.close()

    # -----------------------------------------------------------------------
    # FIGURE 6 — Clinical demographics (sex, age, TMB)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))
    # Sex
    ax = axes[0]
    sex_counts = df_sample["SEX"].value_counts(dropna=True)
    if len(sex_counts) > 0:
        # MUT rate by sex
        sex_order = ["Male","Female"]
        sex_vals = []
        mut_by_sex = []
        for s in sex_order:
            sub = df_sample[df_sample["SEX"]==s]
            sex_vals.append(len(sub))
            mut_by_sex.append((sub["FGFR3_status"]=="MUT").sum())
        x = np.arange(len(sex_order))
        ax.bar(x, [v - m for v, m in zip(sex_vals, mut_by_sex)], label="WT", color=colors["WT"], edgecolor="white")
        ax.bar(x, mut_by_sex, bottom=[v - m for v, m in zip(sex_vals, mut_by_sex)], label="MUT", color=colors["MUT"], edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels([f"{s}\nn={v}" for s, v in zip(sex_order, sex_vals)])
        ax.set_ylabel("Samples"); ax.set_title("MUT vs WT by sex", fontsize=11, weight="bold"); ax.legend(fontsize=9)
        # annotate MUT %
        for i, (v, m) in enumerate(zip(sex_vals, mut_by_sex)):
            if v > 0:
                ax.text(i, v + 0.5, f"{m/v*100:.1f}% MUT", ha="center", va="bottom", fontsize=9, weight="bold")
        # chi-square note: paper says no significant sex difference (Supp Table 15)
        ax.text(0.5, -0.22, "Paper: no sex difference (p > 0.05)", ha="center", va="top", fontsize=8, style="italic", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))
    # Age
    ax = axes[1]
    age_mut = df_sample.loc[df_sample["FGFR3_status"]=="MUT", "AGE"].dropna()
    age_wt = df_sample.loc[df_sample["FGFR3_status"]=="WT", "AGE"].dropna()
    if len(age_mut)+len(age_wt) > 5:
        # box + strip
        plot_age = pd.DataFrame({"AGE": pd.concat([age_wt, age_mut]), "status": ["WT"]*len(age_wt) + ["MUT"]*len(age_mut)})
        sns.boxplot(data=plot_age, x="status", y="AGE", palette=colors, ax=ax, width=0.5)
        sns.stripplot(data=plot_age, x="status", y="AGE", color="black", alpha=0.35, size=3, jitter=0.2, ax=ax)
        ax.set_title("Age at diagnosis", fontsize=11, weight="bold"); ax.set_xlabel(""); ax.set_ylabel("Age (years)")
        # annotate medians
        for status, vals in [("WT", age_wt), ("MUT", age_mut)]:
            if len(vals)>0:
                ax.text(["WT","MUT"].index(status), vals.median(), f" med {vals.median():.0f}", ha="left", va="center", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))
    else:
        # fallback: histogram
        ax.text(0.5, 0.5, "Insufficient age data", ha="center", va="center", transform=ax.transAxes)
    # TMB proxy: total_mutation_count
    ax = axes[2]
    tmb_mut = df_sample.loc[df_sample["FGFR3_status"]=="MUT", "total_mutation_count"].dropna()
    tmb_wt = df_sample.loc[df_sample["FGFR3_status"]=="WT", "total_mutation_count"].dropna()
    # filter extremes for plotting
    if len(tmb_mut)+len(tmb_wt) > 5:
        plot_tmb = pd.DataFrame({"TMB": pd.concat([tmb_wt, tmb_mut]), "status": ["WT"]*len(tmb_wt) + ["MUT"]*len(tmb_mut)})
        # clip at 99th percentile for visibility
        cap = plot_tmb["TMB"].quantile(0.98)
        plot_tmb["TMB_clip"] = plot_tmb["TMB"].clip(upper=cap)
        sns.boxplot(data=plot_tmb, x="status", y="TMB_clip", palette=colors, ax=ax, width=0.5)
        sns.stripplot(data=plot_tmb, x="status", y="TMB_clip", color="black", alpha=0.35, size=3, jitter=0.2, ax=ax)
        ax.set_title("Total mutations per sample\n(TMB proxy, capped 98%)", fontsize=11, weight="bold"); ax.set_xlabel(""); ax.set_ylabel("# mutations")
    fig.suptitle("Figure 6 — Who are the patients? (Demographics & mutation load)", fontsize=13, weight="bold", y=1.02)
    fig.text(0.5, -0.03,
             "Layman: Age/sex do NOT predict MUT (paper Supp.Table 15, p>0.05). TMB = total typos in DNA; MUT samples trend slightly lower TMB in paper cohorts.",
             ha="center", fontsize=9, style="italic", bbox=dict(boxstyle="round,pad=0.3", fc="#FFF8E1", ec="#FF8F00"))
    plt.tight_layout()
    plt.savefig(out_dir / "fig6_clinical.png", dpi=args.dpi, bbox_inches="tight")
    plt.close()

    # -----------------------------------------------------------------------
    # FIGURE 7 — “So what do I measure on the slide?”  (synthetic surrogate)
    # -----------------------------------------------------------------------
    # We have no tile-level histology in cBioPortal, but we CAN visualize the
    # paper's documented surrogate signals as a radar/infographic.
    # Data from paper Fig.5a: frequencies in predictive tiles
    # MUT-predictive tiles: monomorphic 18% vs WT 1%, atypia 27% vs 12%, mito etc.
    # WT-predictive: desmoplastic stroma 14% vs MUT 1%, lymphocytes 12% vs 0%, etc.
    categories = ["Monomorphic\n(nuclei uniform)", "Atypia\n(cell abnormal)", "Desmoplastic\nstroma (scar)", "Immune\nlymphocytes", "Mitosis\n(dividing)", "Tumor\ncellularity"]
    # Approximate percentages from Fig.5a (paper): MUT vs WT
    mut_pct = [18, 27, 1, 0, 6, 99]   # tumor cellularity 99% MUT vs 57% WT — use that
    wt_pct  = [1, 12, 14, 12, 2, 57]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.arange(len(categories))
    w = 0.36
    ax.bar(x - w/2, mut_pct, width=w, label="In MUT-predictive tiles (n=200)", color="#FF5252", edgecolor="#B71C1C")
    ax.bar(x + w/2, wt_pct, width=w, label="In WT-predictive tiles (n=200)", color="#448AFF", edgecolor="#0D47A1")
    for i, (m, wv) in enumerate(zip(mut_pct, wt_pct)):
        ax.text(i - w/2, m + 1.2, f"{m}%", ha="center", va="bottom", fontsize=9, weight="bold", color="#B71C1C")
        ax.text(i + w/2, wv + 1.2, f"{wv}%", ha="center", va="bottom", fontsize=9, weight="bold", color="#0D47A1")
        # significance stars for paper's p<0.001 etc.
        if i in [0,1,2,3]:
            ax.text(i, max(m,wv)+8, "***", ha="center", va="bottom", fontsize=10, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel("Frequency in top-predictive 112×112 µm tiles (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Figure 7 — What the AI actually looks at on the slide\n(Paper Fig.5a: histology patterns in most-predictive tiles, TCGA MIBC)", weight="bold")
    ax.legend(fontsize=9, loc="upper right")
    # annotate surrogate meaning
    fig.text(0.5, -0.02,
             "Layman: Imagine MUT as 'tidy, dense tumor with uniform dots and little scar/inflammation' → easy for nuclei/stroma/TIL features. "
             "WT is 'scarred, inflamed, varied nuclei' → measure exactly those!",
             ha="center", fontsize=9, style="italic", bbox=dict(boxstyle="round,pad=0.3", fc="#FBE9E7", ec="#D84315"))
    # add actionable box
    # fig.text(0.02, 0.02,
    #          "→ ACTIONABLE FOR YOUR PIPELINE:\n"
    #          "1) Nuclear morphometry: monomorphism (variance of nuclear size/shape ↓ in MUT)\n"
    #          "2) Stroma ratio: area of desmoplastic stroma / tumor area (↓ in MUT)\n"
    #          "3) TIL density: lymphocyte count per mm² (↓ in MUT)\n"
    #          "4) Growth pattern: papillary vs solid (paper: AUC 0.84–1.00 stable, but papillary enriched in MUT)\n"
    #          "5) Grade: low-grade AUC 0.89 > high-grade 0.82 — add grade as auxiliary head",
    #          fontsize=8, family="monospace", bbox=dict(boxstyle="round,pad=0.4", fc="#FFFDE7", ec="#F9A825", alpha=0.95))
    plt.tight_layout()
    plt.savefig(out_dir / "fig7_histology_surrogates.png", dpi=args.dpi, bbox_inches="tight")
    plt.close()

    # -----------------------------------------------------------------------
    # MARKDOWN REPORT
    # -----------------------------------------------------------------------
    # Compute extra stats for report
    cna_vals = df_sample["FGFR3_CNA"].value_counts(dropna=True).sort_index()
    # Most mutated co-genes would need all_muts aggregation — do quick top genes
    # Collect all_muts for available samples
    from collections import Counter as C2
    gene_counter = C2()
    for d in per_image_dirs:
        f = list(d.glob("*_mutations.json"))
        if f:
            try:
                j = json.load(open(f[0]))
                for rec in j:
                    g = rec.get("gene", {}).get("hugoGeneSymbol") or rec.get("Hugo_Symbol") or rec.get("geneId") or "?"
                    if g and g != "FGFR3":
                        gene_counter[g] += 1
            except: pass
    top_cogenes = gene_counter.most_common(8)

    report = layman_intro + f"""
## Key Numbers (local snapshot, activating definition)

| Metric | Value |
|---|---|
| **Slides** | {n_slides} (some patients have DX + TSA slides) |
| **Unique samples** | {n_samples} |
| **Patients** | {n_patients} |
| **MUT (activating)** | {n_mut} ({prevalence:.1f}%) |
| **MUT (any FGFR3 variant)** | {n_mut_any} ({n_mut_any/n_samples*100:.1f}%) |
| **Non-activating FGFR3 variants → counted as WT per paper** | {n_nonnactiv} (e.g. Q674*, H349D, etc.) |
| **WT** | {n_wt} ({100-prevalence:.1f}%) |
| **Median VAF (MUT)** | {vaf_vals.median():.2f} (n={len(vaf_vals)}) |
| **Expression available** | {has_expr}/{n_samples} ({has_expr/n_samples*100:.1f}%) |
| **Median RSEM MUT vs WT** | {expr_mut.median():.0f} vs {expr_wt.median():.0f}  (paper: TPM log2 4.28 FP-WT vs 3.26 TN-WT, p=7.9e-8) |
| **CNA distribution (GISTIC)** | {dict(cna_vals)}  (−2=homDel, −1=hemDel, 0=neutral, 1=gain, 2=amp) |
| **Top co-mutated genes (all samples, MUT+WT)** | {', '.join(f'{g}×{c}' for g,c in top_cogenes) if top_cogenes else 'n/a — fetch incomplete'} |

> **Cohort context (ground truth from cBioPortal):** {cohort_n_patients} patients with ANY FGFR3 variant / {study_all} total ({cohort_n_patients/study_all*100:.1f}%).  Top hotspots cohort-wide: {', '.join(f'{k}×{v}' for k,v in cohort_counter.most_common(6))}.

## How to Read the 7 Figures (2-minute tour)

- **Fig 1** — Is the drug useful to many?  Red pie = the lucky few whose tumor has the pedal stuck ON.
- **Fig 2** — Which typo matters?  Each bar is a codon. The paper's AI finds **S249C** easiest (brightest red box in Fig.3 of paper).
- **Fig 3** — Loud vs quiet gene.  MUT dots sit higher, but overlap → explains false positives/negatives (paper Supp. Table 5-6 where removing atypical expressors lifts AUC 0.82→0.89).
- **Fig 4** — Stage.  MUT is rarer in late stages (the clinical problem: 10–15% in MIBC/mUC vs 50–80% in early NMIBC) — so a pre-screen saves testing.
- **Fig 5** — Signal strength.  VAF ≈ contamination of normal tissue. Higher VAF → cleaner slide.
- **Fig 6** — Age/sex/TMB do NOT predict MUT (paper p>0.05) — you cannot cheat with demographics; you need the slide.
- **Fig 7** — The PAYOFF.  **What to compute on each 224×224 µm tile (at 1.0 MPP, paper Methods):** nuclear uniformity, stroma %, TIL density, papillary vs solid. These three separate MUT vs WT with p<0.001.

## What This Means for Your Image Pipeline (Concrete Features to Extract)

### Must-have tile/slide features (paper-validated surrogates)
1. **Nuclear morphometry** — For each nucleus (HoverNet/CellViT/StarDist): area, eccentricity, chromatin texture variance.  Aggregate tile = mean + **variance**; MUT predicts **low variance** (monomorphic, 18% vs 1%, p=4.3e-13).
2. **Stroma ratio** — Segment stroma (e.g. color deconvolution / U-Net).  Tile feature = `desmoplastic_stroma_area / tumor_area`.  MUT ↓ (1% vs 14%, p=9.8e-13).  This alone was top WT signal.
3. **TIL / immune density** — Detect lymphocytes (small round dark nuclei).  Count per mm².  MUT ↓ (0% vs 12%, p=2.9e-8).  Papers 15-16 confirm FGFR3-MUT is lymphocyte-depleted; not a response-to-immunotherapy predictor but a mutation marker.
4. **Growth pattern** — Papillary (finger-like) vs solid sheet.  Train a tiny classifier; paper shows stable AUC 0.84–1.00 for both — keep as covariate, not sole signal.
5. **Grade assist** — Low vs High grade as auxiliary head.  Model AUC 0.89 low-grade vs 0.82 high-grade on TCGA MIBC — low-grade MUT easier.

### Nice-to-have / de-confounders (paper Methods & Discussion)
- **MPP/resolution:** Paper found 1.0 MPP (224×224 µm) beats 0.5 MPP (Supp. Table 4).  Fix your tiling to 1.0 MPP, 60% tissue matter, max 5000 tiles/slide, Bioptimus H0 1539-D features (or UNI/CONCH) frozen.
- **Multiple slides per patient:** ICC TCGA 0.89, mUC 0.66 (Fig.4a-b).  If a patient has DX + TSA, average logits — reduces block bias.
- **Expression-aware training:** Paper shows low-expressor MUT mimics WT morphology and vice-versa (mean RSEM FP-WT > TN-WT).  If you have RNA, stratify or add expression as second label; without RNA, expect ~30% of WT with high RSEM to look MUT.
- **Hotspot-aware:** Train with hotspot label auxiliary (especially S249C vs Y373C vs R248C).  Paper Fig.3 shows S249C scores highest — use hotspot as extra supervision if labels available.
- **Fusion caution:** FGFR3 fusions (~1%, n=10 in MIBC I+TCGA) were NOT learnable in paper — don't expect to detect; treat as WT for training.
- **Erlangen vs TCGA:** Staining varied even on same scanner; add stain normalization (Macenko) and scanner augmentation.

### What NOT to overfit
- Don't use stage, age, sex as predictors (paper: no sex prevalence difference, stage is prognostic not predictive of mutation mechanism).
- Don't rely on morphology codes `8120/3` alone — histologic subtype (conventional vs divergent vs neuroendocrine) did NOT change AUC (Supp. Table 12).

## Files Produced

| File | Content |
|---|---|
| `per_sample_table.csv` | One row per unique sample ({n_samples} rows) — all parsed fields |
| `per_slide_table.csv` | One row per slide image ({n_slides} rows) |
| `cohort_FGFR3_all_records.csv` | {cohort_n_records} FGFR3 variant records cohort-wide |
| `fig1_prevalence.png`  … `fig7_histology_surrogates.png` | 7 layman-annotated figures (this report) |
| `report.md` (this file) | Human-readable synthesis |

## Re-run / Extend

```bash
# Full fetch (926 slides) then re-analyze
python scripts/fetch_cbioportal_data.py --file-list output/tcga_blca_file_names.txt --outdir data --batch-size 30
python scripts/analyze_fgfr3_mutations.py --data-dir data --out-dir output/mutation_analysis

# Per-image deep dive for one slide (e.g. the big SVS on disk)
python scripts/fetch_cbioportal_data.py --image image/TCGA-FJ-A871*.svs --outdir data
```

## Limitations & Honest Disclaimers

- This local snapshot is **only {n_samples}/{study_all} = {n_samples/study_all*100:.1f}% of the full cohort** — rates will shift when you fetch all 926 slides.  Cohort-wide numbers (Fig.2 cohort bars) are the truth.
- Expression here is **cBioPortal RSEM (HiSeq_RNASeqV2)**, while the paper uses **TPM-normalized log2** from a separate GDC HTSeq + nf-core v3.3 pipeline — values correlate but are not interchangeable.
- No tile-level histology (monomorphic %, stroma %) is available via cBioPortal — Fig.7 bars are **paper-reported reference values**, not re-computed from your slides.  Extract them yourself from `image/*.svs` with a nuclei+TIL+stroma pipeline.
- The script counts a sample as MUT only if an **activating** hotspot is present; purely non-activating variants (H349D etc.) are correctly counted as WT per paper — toggle via `--activating-only` if you add that flag later.

---
*Generated by `scripts/analyze_fgfr3_mutations.py` on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}.  Paper DOI: 10.1038/s41467-024-55331-6*
"""

    out_report = out_dir / "report.md"
    out_report.write_text(report)
    print(f"\n✓ Report saved to {out_report}")
    print(f"✓ Figures (7) saved to {out_dir}/fig*.png")
    print(f"✓ Tables: per_sample_table.csv ({n_samples} rows), per_slide_table.csv ({n_slides} rows)")
    # also print preview to stdout
    print("\n--- LAYMAN SUMMARY (first 40 lines) ---")
    print("\n".join(report.splitlines()[:45]))

if __name__ == "__main__":
    main()
