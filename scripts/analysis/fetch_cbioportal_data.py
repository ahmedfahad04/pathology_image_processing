"""
Fetch cBioPortal Data for Pathology Slide Images

Purpose:
  Download clinically + genomically relevant cBioPortal records for any
  SVS image(s) and save to ``data/``. Handles three input modes:
    1) single image  (--image)
    2) directory scan (--image-dir, default: image/)
    3) file-list TSV/CSV/TXT (--file-list data/tcga_blca_slides.tsv or data/tcga_blca_file_names.txt)

  When --file-list is used, the script reads ONLY the ``file_name`` column
  (e.g. TCGA-FD-A3NA-01Z-00-DX1...svs) and creates a per-image subfolder
  ``{outdir}/{file_stem}/`` for each entry, as requested:
    data/TCGA-FJ-A871-01Z-00-DX5.8F79D0A8-.../cbioportal_*.json

  Paper reference: Bannier et al. Nat Commun 2024 15:10914
    - Study: ``blca_tcga_pan_can_atlas_2018`` (TCGA BLCA PanCan Atlas 2018)
      FGFR3 status via cBioPortal, expression via GDC portal (Methods p.8)
    - Data availability: https://www.cbioportal.org + https://portal.gdc.cancer.gov

  API docs: https://docs.cbioportal.org/web-api-and-clients/
    Swagger: https://www.cbioportal.org/api/swagger-ui/index.html
    Base:    https://www.cbioportal.org/api
    Uses plain ``requests`` (``pip install requests``).  Also works with
    bravado/gget/cBioPortalData if you prefer.

What it downloads (per slide / per sample):
  1. Study + molecular profiles metadata (once per study)
  2. Patient / sample clinical data  (AGE, stage, GRADE, TMB, etc.)
  3. All somatic mutations for sample (projection=DETAILED)
  4. FGFR3-specific mutation (entrez 2261)  -> [] means WILD-TYPE
  5. FGFR3 mRNA expression (RSEM, batch-normalized HiSeq_RNASeqV2)
  6. FGFR3 discrete CNA (GISTIC)
  7. Cohort-level FGFR3 mutations (for prevalence / hotspot distribution)
  8. Summary JSON linking file ↔ GDC case ↔ cBio sample ↔ paper terminology

Barcode parsing:
  TCGA-##-####-##X-##-DX#-UUID.svs
    e.g. TCGA-FJ-A871-01Z-00-DX5.8F79D0A8-5DE6-4159-AA77-61DACB21E867.svs
    -> sample_id  = TCGA-FJ-A871-01
    -> patient_id = TCGA-FJ-A871
  See: https://docs.gdc.cancer.gov/Encyclopedia/pages/TCGA_Barcode/

Usage:
  # single image (legacy)
  python scripts/analysis/fetch_cbioportal_data.py --image image/TCGA-FJ-A871-01Z-00-DX5.8F79D0A8-5DE6-4159-AA77-61DACB21E867.svs

  # whole folder (legacy)
  python scripts/analysis/fetch_cbioportal_data.py --image-dir image/ --study blca_tcga_pan_can_atlas_2018

  # NEW: from TSV that lists all images (as discovered at data/tcga_blca_slides.tsv)
  # Step A: create file_names list (already done):
  #   cut -f4 data/tcga_blca_slides.tsv | tail -n +2 > data/tcga_blca_file_names.txt
  # Step B: download per-image folder:
  python scripts/analysis/fetch_cbioportal_data.py --file-list data/tcga_blca_file_names.txt --outdir data
  python scripts/analysis/fetch_cbioportal_data.py --file-list data/tcga_blca_file_names.csv --outdir data --study blca_tcga_pan_can_atlas_2018
  python scripts/analysis/fetch_cbioportal_data.py --file-list data/tcga_blca_slides.tsv --outdir data/per_image --per-image-dir

  # limit to first X files (your request: x = number of files)
  python scripts/analysis/fetch_cbioportal_data.py --file-list data/tcga_blca_file_names.txt --outdir data --limit 10
  python scripts/analysis/fetch_cbioportal_data.py --file-list data/tcga_blca_file_names.txt --limit 5 --batch-size 5
  python scripts/analysis/fetch_cbioportal_data.py --image-dir image/ --limit 3
  python scripts/analysis/fetch_cbioportal_data.py -n 20 --file-list data/tcga_blca_slides.tsv

  # custom study / gene
  python scripts/analysis/fetch_cbioportal_data.py --file-list data/tcga_blca_file_names.txt --gene EGFR --entrez 1956

  # autodetect study
  python scripts/analysis/fetch_cbioportal_data.py --file-list data/tcga_blca_file_names.txt --study auto

Dependencies:
  pip install requests
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from collections import Counter, defaultdict

try:
    import requests
except ImportError:
    raise ImportError("requests is required. Install with: pip install requests")

BASE = "https://www.cbioportal.org/api"
DEFAULT_STUDY = "blca_tcga_pan_can_atlas_2018"
DEFAULT_GENE = "FGFR3"
DEFAULT_ENTREZ = 2261  # FGFR3


# ---------------------------------------------------------------------------
def parse_tcga_barcode(filename: str):
    base = os.path.basename(filename)
    m = re.match(r"^(TCGA-[A-Z0-9]+-[A-Z0-9]+)-([0-9]{2})[A-Z]?-.*", base)
    if not m:
        parts = base.split("-")
        if len(parts) >= 3 and parts[0] == "TCGA":
            patient = "-".join(parts[:3])
            sample = f"{patient}-01"
            return patient, sample, "01"
        raise ValueError(f"Cannot parse TCGA barcode from filename: {filename}")
    patient = m.group(1)
    code = m.group(2)
    sample = f"{patient}-{code}"
    return patient, sample, code


def discover_study_for_patient(patient_id: str):
    try:
        studies = requests.get(f"{BASE}/studies", params={"pageSize": 500, "projection": "SUMMARY"}, timeout=30).json()
    except Exception:
        return DEFAULT_STUDY
    for s in studies:
        sid = s["studyId"]
        if "blca" not in sid.lower() and "pancan" not in sid.lower():
            continue
        r = requests.get(f"{BASE}/studies/{sid}/patients/{patient_id}", timeout=10)
        if r.status_code == 200:
            return sid
    return DEFAULT_STUDY


def api_get(url, params=None):
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def api_post(url, payload):
    r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=60)
    r.raise_for_status()
    return r.json()


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  saved {path} ({len(json.dumps(obj))} bytes)")


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def read_file_list(path: Path):
    """
    Reads file names from TSV/CSV/TXT.
    - If file is data/tcga_blca_slides.tsv (5 cols, header contains file_name), extracts col 4.
    - If file is csv with header file_name, extracts that column.
    - If file is txt one per line, reads lines.
    Returns list of file_name strings (deduped, order preserved, header skipped).
    """
    text = path.read_text(encoding='utf-8', errors='ignore')
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if len(lines) == 0:
        return []

    # Detect header
    header = lines[0]
    # TSV with 5 cols
    if '\t' in header:
        # Use csv.DictReader with tab
        file_names = []
        reader = csv.DictReader(lines, delimiter='\t')
        # DictReader expects header as first line; check if 'file_name' in fieldnames
        if reader.fieldnames and 'file_name' in reader.fieldnames:
            for row in reader:
                fn = row.get('file_name', '').strip()
                if fn and fn != 'file_name':
                    file_names.append(fn)
        else:
            # fallback: 4th column
            for line in lines[1:]:
                cols = line.split('\t')
                if len(cols) >= 4:
                    file_names.append(cols[3].strip())
        # Dedupe preserve order
        seen = set()
        uniq = []
        for fn in file_names:
            if fn not in seen:
                seen.add(fn)
                uniq.append(fn)
        return uniq

    # CSV with comma
    if ',' in header and 'file_name' in header:
        file_names = []
        reader = csv.DictReader(lines)
        for row in reader:
            fn = row.get('file_name', '').strip()
            if fn:
                file_names.append(fn)
        seen = set()
        uniq = []
        for fn in file_names:
            if fn not in seen:
                seen.add(fn)
                uniq.append(fn)
        return uniq

    # Plain txt: one file_name per line
    # Check if first line is header 'file_name'
    if lines[0].strip() == 'file_name':
        lines = lines[1:]
    # Dedupe preserve order, filter empty and ensure .svs
    seen = set()
    uniq = []
    for fn in lines:
        fn = fn.strip()
        if fn and fn not in seen:
            seen.add(fn)
            uniq.append(fn)
    return uniq


def fetch_for_slide(svs_path: Path, study: str, gene_symbol: str, entrez: int, outdir: Path, fetch_cohort_once: set, per_image_dir: bool = False):
    """
    Legacy per-slide fetcher (sequential). Used when --image / --image-dir.
    If per_image_dir True, saves into outdir / stem / instead of flat outdir.
    """
    patient, sample, code = parse_tcga_barcode(svs_path.name)
    print(f"\n=== {svs_path.name} ===")
    print(f"  patient={patient}  sample={sample}  study={study}  gene={gene_symbol}({entrez})")

    if study == "auto":
        study = discover_study_for_patient(patient)
        print(f"  auto-discovered study: {study}")

    # Determine output directory for this slide
    if per_image_dir:
        # use stem without .svs as folder
        stem = svs_path.stem  # TCGA-FJ-A871-01Z-00-DX5.8F79...
        # also handle .svs double extension? stem is without .svs
        slide_outdir = outdir / stem
    else:
        slide_outdir = outdir
    slide_outdir.mkdir(parents=True, exist_ok=True)

    prefix = f"cbioportal_{sample}"

    if study not in fetch_cohort_once:
        try:
            study_info = api_get(f"{BASE}/studies/{study}")
            save_json(study_info, outdir / "cbioportal_study_info.json")
            # also copy to slide dir if per_image
            if per_image_dir:
                save_json(study_info, slide_outdir / "cbioportal_study_info.json")
        except Exception as e:
            print(f"  [warn] study_info failed: {e}")
            study_info = {"studyId": study, "allSampleCount": None}
        try:
            profiles = api_get(f"{BASE}/studies/{study}/molecular-profiles")
            save_json(profiles, outdir / "cbioportal_molecular_profiles.json")
            if per_image_dir:
                save_json(profiles, slide_outdir / "cbioportal_molecular_profiles.json")
        except Exception as e:
            print(f"  [warn] profiles failed: {e}")
    else:
        try:
            study_info = json.load(open(outdir / "cbioportal_study_info.json"))
        except Exception:
            study_info = {"studyId": study, "allSampleCount": None}

    try:
        samp_info = api_get(f"{BASE}/studies/{study}/samples/{sample}")
        save_json(samp_info, slide_outdir / f"{prefix}_sample_info.json")
    except Exception as e:
        print(f"  [warn] sample_info: {e}")
    try:
        pat_info = api_get(f"{BASE}/studies/{study}/patients/{patient}")
        save_json(pat_info, slide_outdir / f"{prefix}_patient_info.json")
    except Exception as e:
        print(f"  [warn] patient_info: {e}")

    try:
        pat_clin = api_get(f"{BASE}/studies/{study}/patients/{patient}/clinical-data")
        save_json(pat_clin, slide_outdir / f"{prefix}_patient_clinical.json")
    except Exception as e:
        print(f"  [warn] patient clinical: {e}")
        pat_clin = []
    try:
        samp_clin = api_get(f"{BASE}/studies/{study}/samples/{sample}/clinical-data")
        save_json(samp_clin, slide_outdir / f"{prefix}_sample_clinical.json")
    except Exception as e:
        print(f"  [warn] sample clinical: {e}")
        samp_clin = []

    try:
        all_muts = api_post(f"{BASE}/molecular-profiles/{study}_mutations/mutations/fetch?projection=DETAILED",
                            {"sampleIds": [sample]})
        save_json(all_muts, slide_outdir / f"{prefix}_mutations.json")
        print(f"  total mutations: {len(all_muts)}")
    except Exception as e:
        print(f"  [warn] all mutations: {e}")
        all_muts = []

    try:
        fgfr3_mut = api_post(f"{BASE}/molecular-profiles/{study}_mutations/mutations/fetch?projection=DETAILED",
                             {"entrezGeneIds": [entrez], "sampleIds": [sample]})
        save_json(fgfr3_mut, slide_outdir / f"{prefix}_{gene_symbol}_mutation.json")
        fgfr3_status = "MUTANT" if len(fgfr3_mut) > 0 else "WILD-TYPE"
        print(f"  {gene_symbol} status: {fgfr3_status} ({len(fgfr3_mut)} records)")
    except Exception as e:
        print(f"  [warn] FGFR3 mutation: {e}")
        fgfr3_mut = []
        fgfr3_status = "UNKNOWN"

    try:
        expr_pid = f"{study}_rna_seq_v2_mrna"
        fgfr3_expr = api_post(f"{BASE}/molecular-profiles/{expr_pid}/molecular-data/fetch?projection=DETAILED",
                              {"entrezGeneIds": [entrez], "sampleIds": [sample]})
        save_json(fgfr3_expr, slide_outdir / f"{prefix}_{gene_symbol}_expression.json")
        expr_val = fgfr3_expr[0]["value"] if len(fgfr3_expr) > 0 else None
        print(f"  {gene_symbol} expression: {expr_val}")
    except Exception as e:
        print(f"  [warn] expression: {e}")
        fgfr3_expr = []
        expr_val = None

    try:
        cna_pid = f"{study}_gistic"
        fgfr3_cna = api_post(f"{BASE}/molecular-profiles/{cna_pid}/discrete-copy-number/fetch?projection=DETAILED",
                             {"entrezGeneIds": [entrez], "sampleIds": [sample]})
        save_json(fgfr3_cna, slide_outdir / f"{prefix}_{gene_symbol}_CNA.json")
    except Exception as e:
        print(f"  [warn] CNA (often no call): {e}")
        fgfr3_cna = []
        save_json([], slide_outdir / f"{prefix}_{gene_symbol}_CNA.json")

    if study not in fetch_cohort_once:
        try:
            try:
                cohort = api_post(f"{BASE}/molecular-profiles/{study}_mutations/mutations/fetch?projection=DETAILED",
                                  {"entrezGeneIds": [entrez], "sampleListId": f"{study}_all"})
            except Exception:
                cohort = api_get(f"{BASE}/molecular-profiles/{study}_mutations/mutations",
                                 params={"sampleListId": f"{study}_all", "entrezGeneId": entrez, "projection": "DETAILED"})
            save_json(cohort, outdir / f"cbioportal_{study}_{gene_symbol}_mutations_cohort.json")
            if per_image_dir:
                save_json(cohort, slide_outdir / f"cbioportal_{study}_{gene_symbol}_mutations_cohort.json")
            c = Counter(x.get("proteinChange", "?") for x in cohort)
            print(f"  cohort {gene_symbol} mutated records: {len(cohort)} distinct patients {len(set(x['patientId'] for x in cohort))}")
            print(f"  hotspots: {c.most_common(6)}")
            fetch_cohort_once.add(study)
        except Exception as e:
            print(f"  [warn] cohort fetch: {e}")

    try:
        def get_attr(clin_list, attr):
            for x in clin_list:
                if x.get("clinicalAttributeId") == attr:
                    return x.get("value")
            return None

        summary = {
            "image_file": svs_path.name,
            "svs_path": str(svs_path),
            "cbio_sample_id": sample,
            "cbio_patient_id": patient,
            "study": study,
            "gene": gene_symbol,
            "entrez": entrez,
            f"{gene_symbol}_status": fgfr3_status + (" (no FGFR3 mutation = WT per paper: non-activating considered WT)" if fgfr3_status == "WILD-TYPE" else ""),
            f"{gene_symbol}_mutation_details": fgfr3_mut,
            f"{gene_symbol}_expression_RSEM": expr_val,
            "note_expression": "Paper uses TPM-normalized log2; cBioPortal RSEM batch-normalized HiSeq_RNASeqV2 - related but not identical",
            "total_mutations_in_sample": len(all_muts),
            "sample_clinical_STAGE": get_attr(pat_clin, "AJCC_PATHOLOGIC_TUMOR_STAGE") or get_attr(pat_clin, "AJCC_STAGING_EDITION"),
            "sample_clinical_T": get_attr(pat_clin, "PATH_T_STAGE"),
            "sources": {
                "study_info": f"{BASE}/studies/{study}",
                "mutations": f"{BASE}/molecular-profiles/{study}_mutations/mutations/fetch",
                "rna": f"{BASE}/molecular-profiles/{study}_rna_seq_v2_mrna/molecular-data/fetch",
                "docs": "https://docs.cbioportal.org/web-api-and-clients/",
            },
        }
        save_json(summary, slide_outdir / f"{prefix}_summary.json")
    except Exception as e:
        print(f"  [warn] summary: {e}")

    return {"patient": patient, "sample": sample, "status": fgfr3_status, "expr": expr_val, "file": svs_path.name, "outdir": str(slide_outdir)}


def fetch_batch_file_list(file_list_path: Path, study: str, gene_symbol: str, entrez: int, outdir: Path, batch_size: int = 50, limit: int = None):
    """
    Efficient batch mode for --file-list with 926 entries.
    - Reads file names from TSV/CSV/TXT
    - Derives patient/sample per file
    - Fetches study info once, then batch-fetches mutations/expression in chunks
    - Writes per-image subfolder {outdir}/{file_stem}/ with JSONs
    - If limit is set, only processes first `limit` entries (e.g., --limit 10)
    """
    file_names = read_file_list(file_list_path)
    print(f"Read {len(file_names)} file names from {file_list_path}")
    if limit is not None:
        file_names = file_names[:limit]
        print(f"  -> limited to first {limit} entries")
    if len(file_names) == 0:
        print(f"[error] no file names in {file_list_path}", file=sys.stderr)
        sys.exit(1)

    # Parse barcodes, keep mapping file_name -> (patient, sample, stem)
    entries = []
    sample_to_files = defaultdict(list)
    for fn in file_names:
        try:
            patient, sample, code = parse_tcga_barcode(fn)
            stem = Path(fn).stem  # without .svs
            entries.append((fn, patient, sample, stem))
            sample_to_files[sample].append(fn)
        except Exception as e:
            print(f"[warn] skip unparsable {fn}: {e}")

    unique_samples = sorted(set(s for _, _, s, _ in entries))
    unique_patients = sorted(set(p for _, p, _, _ in entries))
    print(f"  unique samples: {len(unique_samples)}  unique patients: {len(unique_patients)}  entries: {len(entries)}")

    if study == "auto":
        # pick first patient to discover
        study = discover_study_for_patient(unique_patients[0])
        print(f"  auto-discovered study: {study}")

    outdir.mkdir(parents=True, exist_ok=True)

    # 1 study info + profiles (once)
    try:
        study_info = api_get(f"{BASE}/studies/{study}")
        save_json(study_info, outdir / "cbioportal_study_info.json")
    except Exception as e:
        print(f"[warn] study_info: {e}")
        study_info = {"studyId": study, "allSampleCount": None}
    try:
        profiles = api_get(f"{BASE}/studies/{study}/molecular-profiles")
        save_json(profiles, outdir / "cbioportal_molecular_profiles.json")
    except Exception as e:
        print(f"[warn] profiles: {e}")

    # 2 Cohort-level FGFR3
    try:
        try:
            cohort = api_post(f"{BASE}/molecular-profiles/{study}_mutations/mutations/fetch?projection=DETAILED",
                              {"entrezGeneIds": [entrez], "sampleListId": f"{study}_all"})
        except Exception:
            cohort = api_get(f"{BASE}/molecular-profiles/{study}_mutations/mutations",
                             params={"sampleListId": f"{study}_all", "entrezGeneId": entrez, "projection": "DETAILED"})
        save_json(cohort, outdir / f"cbioportal_{study}_{gene_symbol}_mutations_cohort.json")
        c = Counter(x.get("proteinChange", "?") for x in cohort)
        print(f"  cohort {gene_symbol}: {len(cohort)} records, {len(set(x['patientId'] for x in cohort))} patients, hotspots {c.most_common(6)}")
        # Build quick lookup sample -> FGFR3 mut list
        cohort_by_sample = defaultdict(list)
        for rec in cohort:
            cohort_by_sample[rec.get("sampleId")].append(rec)
    except Exception as e:
        print(f"[warn] cohort: {e}")
        cohort = []
        cohort_by_sample = defaultdict(list)

    # 3 Batch fetch FGFR3 expression for all unique samples in chunks
    expr_by_sample = {}
    print(f"  fetching {gene_symbol} expression for {len(unique_samples)} samples in batches of {batch_size}...")
    for chunk in chunked(unique_samples, batch_size):
        try:
            expr_pid = f"{study}_rna_seq_v2_mrna"
            chunk_res = api_post(f"{BASE}/molecular-profiles/{expr_pid}/molecular-data/fetch?projection=DETAILED",
                                 {"entrezGeneIds": [entrez], "sampleIds": chunk})
            for rec in chunk_res:
                expr_by_sample[rec["sampleId"]] = rec
        except Exception as e:
            print(f"    [warn] expr chunk {chunk[:2]}: {e}")
        time.sleep(0.2)

    # 4 Batch fetch FGFR3 mutations for all unique samples (to get WT vs MUT per sample)
    fgfr3_by_sample = defaultdict(list)
    print(f"  fetching {gene_symbol} mutations for {len(unique_samples)} samples in batches of {batch_size}...")
    for chunk in chunked(unique_samples, batch_size):
        try:
            chunk_res = api_post(f"{BASE}/molecular-profiles/{study}_mutations/mutations/fetch?projection=DETAILED",
                                 {"entrezGeneIds": [entrez], "sampleIds": chunk})
            for rec in chunk_res:
                fgfr3_by_sample[rec["sampleId"]].append(rec)
        except Exception as e:
            print(f"    [warn] fgfr3 chunk: {e}")
        time.sleep(0.2)

    # 5 Batch fetch all mutations for all unique samples (this can be large; chunk smaller)
    allmuts_by_sample = defaultdict(list)
    print(f"  fetching all mutations for {len(unique_samples)} samples in batches of {batch_size} (may take a while)...")
    for chunk in chunked(unique_samples, 30):  # smaller chunk for all-muts (each sample ~30-200 muts)
        try:
            chunk_res = api_post(f"{BASE}/molecular-profiles/{study}_mutations/mutations/fetch?projection=DETAILED",
                                 {"sampleIds": chunk})
            for rec in chunk_res:
                allmuts_by_sample[rec["sampleId"]].append(rec)
        except Exception as e:
            print(f"    [warn] allmuts chunk: {e}")
        time.sleep(0.3)

    # 6 For each entry, create per-image folder and write files
    # We still need clinical data per patient/sample - fetch per sample but we can also batch via clinical-data/fetch
    # Use GET per sample for simplicity but with batch we already have expr/mut; clinical we fetch per sample sequentially but it's only 926 *2 = 1852 calls.
    # To save time, we batch clinical via POST /clinical-data/fetch if available, fallback to GET.
    # Try batch clinical fetch:
    clin_by_sample = {}
    clin_by_patient = {}
    try:
        # Batch patient clinical: POST /api/clinical-data/fetch with ids = patientIds
        # Endpoint expects {ids: [...], studyId?} but we use study-specific endpoint
        # Use generic /clinical-data/fetch with studyId filter
        for chunk in chunked(unique_patients, batch_size):
            try:
                # Try study-specific clinical-data/fetch
                res = api_post(f"{BASE}/studies/{study}/clinical-data/fetch?projection=DETAILED",
                               {"ids": chunk})
                for rec in res:
                    # rec has patientId or sampleId
                    pid = rec.get("patientId")
                    sid = rec.get("sampleId")
                    if sid:
                        clin_by_sample.setdefault(sid, []).append(rec)
                    elif pid:
                        clin_by_patient.setdefault(pid, []).append(rec)
            except Exception as e:
                print(f"    [warn] clinical batch (study) {e}, falling back to per-sample GET")
                break
        # Also fetch sample-level clinical in same way if not already
        for chunk in chunked(unique_samples, batch_size):
            try:
                res = api_post(f"{BASE}/studies/{study}/clinical-data/fetch?projection=DETAILED",
                               {"ids": chunk})
                for rec in res:
                    sid = rec.get("sampleId")
                    if sid:
                        clin_by_sample.setdefault(sid, []).append(rec)
            except Exception:
                pass
    except Exception as e:
        print(f"  [warn] batch clinical fetch failed: {e}")

    # If batch clinical didn't populate enough, fallback to per-sample GET for missing
    # We'll do lazy fetching per entry below if not present

    print(f"\n  writing per-image folders to {outdir} ...")
    results = []
    for fn, patient, sample, stem in entries:
        slide_outdir = outdir / stem
        slide_outdir.mkdir(parents=True, exist_ok=True)

        # Reuse pre-fetched batch results
        fgfr3_mut = fgfr3_by_sample.get(sample, [])
        # also check cohort_by_sample for extra (should be same)
        expr_rec = expr_by_sample.get(sample)
        fgfr3_expr = [expr_rec] if expr_rec else []
        expr_val = expr_rec["value"] if expr_rec else None
        all_muts = allmuts_by_sample.get(sample, [])
        fgfr3_status = "MUTANT" if len(fgfr3_mut) > 0 else "WILD-TYPE"

        # Clinical: use batch cache or fetch GET if missing
        pat_clin = clin_by_patient.get(patient, [])
        samp_clin = clin_by_sample.get(sample, [])
        if not pat_clin:
            try:
                pat_clin = api_get(f"{BASE}/studies/{study}/patients/{patient}/clinical-data")
            except Exception:
                pat_clin = []
            time.sleep(0.05)
        if not samp_clin:
            try:
                samp_clin = api_get(f"{BASE}/studies/{study}/samples/{sample}/clinical-data")
            except Exception:
                samp_clin = []
            time.sleep(0.05)

        # Fetch sample/patient info (lightweight, per entry)
        try:
            samp_info = api_get(f"{BASE}/studies/{study}/samples/{sample}")
        except Exception:
            samp_info = {"sampleId": sample, "patientId": patient, "studyId": study}
        try:
            pat_info = api_get(f"{BASE}/studies/{study}/patients/{patient}")
        except Exception:
            pat_info = {"patientId": patient, "studyId": study}

        prefix = f"cbioportal_{sample}"

        # Save per-image files
        save_json(samp_info, slide_outdir / f"{prefix}_sample_info.json")
        save_json(pat_info, slide_outdir / f"{prefix}_patient_info.json")
        save_json(pat_clin, slide_outdir / f"{prefix}_patient_clinical.json")
        save_json(samp_clin, slide_outdir / f"{prefix}_sample_clinical.json")
        save_json(all_muts, slide_outdir / f"{prefix}_mutations.json")
        save_json(fgfr3_mut, slide_outdir / f"{prefix}_{gene_symbol}_mutation.json")
        save_json(fgfr3_expr, slide_outdir / f"{prefix}_{gene_symbol}_expression.json")
        # CNA: batch not pre-fetched, fetch per sample (small)
        try:
            cna_pid = f"{study}_gistic"
            fgfr3_cna = api_post(f"{BASE}/molecular-profiles/{cna_pid}/discrete-copy-number/fetch?projection=DETAILED",
                                 {"entrezGeneIds": [entrez], "sampleIds": [sample]})
        except Exception:
            fgfr3_cna = []
        save_json(fgfr3_cna, slide_outdir / f"{prefix}_{gene_symbol}_CNA.json")

        # Copy cohort file into each folder for self-containment
        try:
            import shutil
            cohort_src = outdir / f"cbioportal_{study}_{gene_symbol}_mutations_cohort.json"
            if cohort_src.exists():
                shutil.copy(cohort_src, slide_outdir / cohort_src.name)
        except Exception:
            pass

        # Also copy study info
        try:
            import shutil
            for fname in ["cbioportal_study_info.json", "cbioportal_molecular_profiles.json"]:
                src = outdir / fname
                if src.exists():
                    shutil.copy(src, slide_outdir / fname)
        except Exception:
            pass

        # Summary
        def get_attr(clin_list, attr):
            for x in clin_list:
                if x.get("clinicalAttributeId") == attr:
                    return x.get("value")
            return None

        summary = {
            "image_file": fn,
            "svs_path": fn,  # file_name as listed, not local path
            "cbio_sample_id": sample,
            "cbio_patient_id": patient,
            "study": study,
            "gene": gene_symbol,
            "entrez": entrez,
            f"{gene_symbol}_status": fgfr3_status + (" (no FGFR3 mutation = WT per paper: non-activating considered WT)" if fgfr3_status == "WILD-TYPE" else ""),
            f"{gene_symbol}_mutation_details": fgfr3_mut,
            f"{gene_symbol}_expression_RSEM": expr_val,
            "note_expression": "Paper uses TPM-normalized log2; cBioPortal RSEM batch-normalized HiSeq_RNASeqV2 - related but not identical",
            "total_mutations_in_sample": len(all_muts),
            "sample_clinical_STAGE": get_attr(pat_clin, "AJCC_PATHOLOGIC_TUMOR_STAGE") or get_attr(pat_clin, "AJCC_STAGING_EDITION"),
            "sample_clinical_T": get_attr(pat_clin, "PATH_T_STAGE"),
            "per_image_folder": str(slide_outdir),
            "sources": {
                "study_info": f"{BASE}/studies/{study}",
                "mutations": f"{BASE}/molecular-profiles/{study}_mutations/mutations/fetch",
                "rna": f"{BASE}/molecular-profiles/{study}_rna_seq_v2_mrna/molecular-data/fetch",
                "docs": "https://docs.cbioportal.org/web-api-and-clients/",
            },
        }
        save_json(summary, slide_outdir / f"{prefix}_summary.json")

        # Also write a tiny file_name echo for traceability
        (slide_outdir / "file_name.txt").write_text(fn + "\n")

        results.append({"file": fn, "patient": patient, "sample": sample, "status": fgfr3_status, "expr": expr_val, "outdir": str(slide_outdir)})

    # Final global summary CSV
    try:
        import csv as csvm
        with open(outdir / "cbioportal_summary_all.csv", "w", newline="") as csvfile:
            w = csvm.writer(csvfile)
            w.writerow(["file_name", "patient_id", "sample_id", f"{gene_symbol}_status", f"{gene_symbol}_RSEM", "total_mutations", "AJCC_stage", "per_image_folder"])
            for r in results:
                # Find stage from its summary file
                sf = Path(r["outdir"]) / f"cbioportal_{r['sample']}_summary.json"
                stage = ""
                try:
                    j = json.load(open(sf))
                    stage = j.get("sample_clinical_STAGE", "")
                except Exception:
                    pass
                w.writerow([r["file"], r["patient"], r["sample"], r["status"], r["expr"], "", stage, r["outdir"]])
        print(f"  global summary {outdir / 'cbioportal_summary_all.csv'}")
    except Exception as e:
        print(f"  [warn] global csv: {e}")

    return results


def main():
    p = argparse.ArgumentParser(description="Download cBioPortal data for SVS slide(s) (generalized fetcher, supports --file-list per-image folders)")
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument("--image", type=str, help="Path to single SVS file (e.g. image/foo.svs)")
    g.add_argument("--image-dir", type=str, help="Directory containing SVS files")
    g.add_argument("--file-list", type=str, help="Path to TSV/CSV/TXT listing file_name per line (e.g. data/tcga_blca_slides.tsv or data/tcga_blca_file_names.txt). Creates per-image subfolder {outdir}/{file_stem}/")
    p.add_argument("--study", type=str, default=DEFAULT_STUDY,
                   help=f"cBioPortal studyId (default: {DEFAULT_STUDY}). Use 'auto' to discover via patient API.")
    p.add_argument("--gene", type=str, default=DEFAULT_GENE, help="Hugo symbol (default: FGFR3)")
    p.add_argument("--entrez", type=int, default=DEFAULT_ENTREZ, help="Entrez gene ID (default: 2261 for FGFR3)")
    p.add_argument("--outdir", type=str, default="data", help="Output directory (default: data/). For --file-list, each image gets {outdir}/{file_stem}/")
    p.add_argument("--batch-size", type=int, default=50, help="Batch size for POST fetch (default: 50)")
    p.add_argument("--per-image-dir", action="store_true", help="Force per-image subfolders even for --image/--image-dir mode")
    p.add_argument("--limit", "-n", "--num-files", "--max-files", type=int, default=None, dest="limit",
                   help="Limit to first X file names / images (e.g., --limit 10 or -n 10 downloads only first 10). Works with --file-list and --image-dir.")
    args = p.parse_args()

    # Default if no input specified: try image/ dir
    if not args.image and not args.image_dir and not args.file_list:
        args.image_dir = "image"

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.file_list:
        fl = Path(args.file_list)
        if not fl.exists():
            print(f"[error] file-list not found: {fl}", file=sys.stderr)
            sys.exit(1)
        lim_msg = f" (limit {args.limit})" if args.limit else ""
        print(f"Mode: --file-list {fl}{lim_msg} -> per-image folders in {outdir}")
        results = fetch_batch_file_list(fl, args.study, args.gene, args.entrez, outdir, batch_size=args.batch_size, limit=args.limit)
        print("\n=== SUMMARY (first 20) ===")
        for r in results[:20]:
            print(f" {r['file']} -> {r['sample']} : {r['status']} expr={r['expr']}  => {r['outdir']}")
        if len(results) > 20:
            print(f" ... and {len(results)-20} more")
        print(f"\nAll outputs in {outdir.resolve()}  (each image: {{outdir}}/{{file_stem}}/ )")
        print("Also flat cohort files: cbioportal_study_info.json etc.")
        return

    # Legacy single / dir mode
    svs_list = []
    if args.image:
        svs_list = [Path(args.image)]
    elif args.image_dir:
        imgdir = Path(args.image_dir)
        if not imgdir.exists():
            print(f"[error] image dir does not exist: {imgdir}", file=sys.stderr)
            sys.exit(1)
        svs_list = sorted(imgdir.glob("*.svs")) + sorted(imgdir.glob("*.SVS"))
        if len(svs_list) == 0:
            svs_list = sorted(imgdir.rglob("*.svs"))
        if len(svs_list) == 0:
            print(f"[error] no .svs files found in {imgdir}", file=sys.stderr)
            sys.exit(1)

    if args.limit is not None:
        svs_list = svs_list[:args.limit]
        print(f"  -> limited to first {args.limit} slides")
    print(f"Found {len(svs_list)} slide(s)")
    for s in svs_list:
        try:
            sz = s.stat().st_size / 1e9
            print(f"  - {s} ({sz:.2f} GB)")
        except Exception:
            print(f"  - {s}")

    fetch_cohort_once = set()
    results = []
    for svs in svs_list:
        try:
            r = fetch_for_slide(svs, args.study, args.gene, args.entrez, outdir, fetch_cohort_once, per_image_dir=args.per_image_dir)
            results.append(r)
        except Exception as e:
            print(f"[error] failed for {svs}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    print("\n=== SUMMARY ===")
    for r in results:
        print(f" {r['sample']} ({r['patient']}): {r['status']} expr={r['expr']}")
    print(f"\nAll outputs in {outdir.resolve()}")
    print("Relevant docs: https://docs.cbioportal.org/web-api-and-clients/ and paper Data availability p.9")


if __name__ == "__main__":
    main()
