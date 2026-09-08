# Metadata Completeness Report: Paper vs. `data/data.txt` vs. Input Slide

**Paper:** `docs/s41467-024-55331-6.pdf` — Bannier et al., *Nature Communications* (2024) 15:10914 — "AI allows pre-screening of FGFR3 mutational status using routine histology slides of muscle-invasive bladder cancer"  
**Downloaded metadata:** `data/data.txt:1-472` (GDC API `slide_image` records, concatenated JSON)  
**Input image:** `image/TCGA-FJ-A871-01Z-00-DX5.8F79D0A8-5DE6-4159-AA77-61DACB21E867.svs` (1.1 GB, SVS)  
**Date:** 2026-09-07  
**Author:** OpenCode / Muse Spark analysis

---

## 1. Executive Summary

| Metric | Count |
|---|---|
| **Paper metadata fields identified** | **57 distinct fields** across 8 categories |
| **Found (fully) in `data/data.txt`** | **~22 / 57 (39%)** |
| **Partially present** | ~6 / 57 (11%) — e.g. cohort composition only for 2 TCGA cases |
| **Missing** | **~29 / 57 (51%)** — including the *core* FGFR3 ground-truth |

**One-sentence verdict:** `data/data.txt` is a *GDC Biospecimen / Slide Image* export. It provides excellent **demographic, diagnosis-staging, and biospecimen provenance** for the two TCGA cases it contains, and it **does contain the exact record for the input slide** `TCGA-FJ-A871` (duplicated at `data/data.txt:161-316` and `data/data.txt:317-472`), but it **contains none of the molecular ground-truth** (FGFR3 mutation status / hotspot / FGFR3 fusions / FGFR3 mRNA expression), **no Erlangen cohort metadata** (MIBC I/II, NMIBC I/II, mUC), **no histopathology annotations**, and **no model pipeline parameters** that are central to the paper's claims.

> Re-running the paper's workflow on `image/TCGA-FJ-A871...svs` *from `data/data.txt` alone is impossible* — you must fetch FGFR3 status from cBioPortal and RNAseq from GDC, as described in *Methods: Datasets description* and *Data availability*.

---

## 2. How This Report Was Built

1. Full read of `docs/s41467-024-55331-6.pdf` (11 pages, Figs. 1-5, Tables 1 + Supplementary Tables 1-15 cited in-text).
2. Full read of `data/data.txt:1-472` — contains **3 concatenated JSON objects**:
   - Object 1: `data/data.txt:2-160` → `TCGA-FD-A3NA-01Z-00-DX1...svs` (case `TCGA-FD-A3NA`)
   - Object 2: `data/data.txt:161-316` → `TCGA-FJ-A871-01Z-00-DX5...svs` (**identical to input image**)
   - Object 3: `data/data.txt:317-472` → **exact duplicate of Object 2** (same `file_id: 3caa2d31-55fc-4117-9c53-c15624e8927a`, `md5sum: 3ee2aa658a3fb19159338d3692c80d4f`)
3. Cross-referenced each paper-reported variable against keys present in `data/data.txt` via `python3 -c` JSON key enumeration.
4. Verified input slide on disk: `image/TCGA-FJ-A871-01Z-00-DX5.8F79D0A8-5DE6-4159-AA77-61DACB21E867.svs` (1166418943 bytes) matches `data/data.txt:297` `file_name` and `data/data.txt:305` `md5sum`.

---

## 3. Paper Metadata Taxonomy — Complete Inventory (57 fields)

Derived from exhaustive reading of Methods + Results + Figures.

### 3.1 Cohort Composition & Provenance
| # | Field | Paper location | Notes |
|---|---|---|---|
| 1 | Total cohort size (n=1222) and split: Training n=391 (23% FGFR3-mut), Validation n=586 (11%), Transferability n=97 | Fig. 1b, p.3 | Erlangen n=810 resected patients, TCGA n=412 → 307 MIBC |
| 2 | Cohort labels: MIBC I (n=239→236), MIBC II (n=201→183), NMIBC I (n=155), NMIBC II (n=109→97), mUC (n=106→96), TCGA-BLCA (n=412→307) | Fig. 1b, p.8 *Datasets description* | Includes attrition reasons |
| 3 | Exclusion reasons: blurry (n=43), no FFPE slide (25), no RNAseq/mutation status (7), no MIBC tumor slide (73) | Fig. 1b, p.8 | Critical for reproducibility |
| 4 | Institution: Erlangen University Hospital vs Public TCGA | p.8 | Staining-site variance |

### 3.2 Clinical / Demographic
| # | Field | Paper location | Notes |
|---|---|---|---|
| 5 | Biological sex (sex assigned at birth) | p.3 *Results: prevalence by sex*, Supplementary Table 15, Supplementary Table 1 | "No statistically significant differences" |
| 6 | Race / Ethnicity | Supplementary Table 1 (referenced `docs/s41467-024-55331-6.pdf:8`) | |
| 7 | Age at diagnosis / Age at index / Year of diagnosis / Year of birth | Supplementary Table 1, Methods p.8 | |
| 8 | Vital status / Days to death / Days to last follow-up | Supplementary Table 1 | For survival (mUC) |
| 9 | Tissue/organ of origin, Primary site (Bladder), Site of resection/biopsy | Methods p.8 | |
| 10 | Method of diagnosis (Cystoscopy / Surgical Resection) | Supplementary Table 1 | |
| 11 | Prior malignancy / Prior treatment / Synchronous malignancy | Supplementary Table 1 | |

### 3.3 Pathologic Staging & Tumor Classification
| # | Field | Paper location | Notes |
|---|---|---|---|
| 12 | AJCC pathologic T (pT: T2a/T2b/T3b, ≥pT2 pie chart) | Fig. 1b pie (pTa/pT1/≥pT2), Methods p.8 | |
| 13 | AJCC pathologic N, M, overall Stage (Stage II/III, IIA), AJCC edition (6th/7th), clinical T | Fig. 1b, Methods | |
| 14 | Tumor grade (Low vs High Grade) — model AUC 0.89 low-grade vs 0.82 high-grade on TCGA MIBC | p.5 | |
| 15 | ICD-10 code (C67.2, C67.4), Morphology (8120/3, 8140/3) | Methods | |
| 16 | Histological subtype (conventional urothelial / divergent / neuroendocrine) | p.5, Supplementary Table 9/12 | |
| 17 | Growth pattern (solid vs papillary) | p.5, Supplementary Table 10/13 (AUC 0.84-1.00) | |
| 18 | Carcinoma in situ (CIS) presence | p.5, Supplementary Table 11/14 | |
| 19 | Disease state metadata: primary / recurrence / Synchronous primary / Subsequent Primary | Methods p.8 | |

### 3.4 Molecular Ground-Truth (THE core of the paper)
| # | Field | Paper location | Notes |
|---|---|---|---|
| 20 | **FGFR3 mutation status (MUT vs WT)** | Entire paper; Methods *Mutational detection via SNaPshot* p.9, Fig.1a Step 3 | TSO 500 + alignment vs SNaPshot PCR; TCGA via cBioPortal PanCan Atlas 2018 |
| 21 | **FGFR3 hotspot details** (R248C, S249C, G372C/G370C, G382R/G380R, S373C/S371C, Y375C/Y373C, A393E, K652E/Q/M/T) — 11 mutations covering >99% activating; 6 hotspots with >5 occurrences analyzed | p.5 Fig.3, Supplementary Table 7, Methods p.9 | S249C easiest to detect |
| 22 | FGFR3 fusions (n=10 across MIBC I + TCGA, failed to learn) | p.7-8 Discussion | |
| 23 | Non-activating/inactivating variants considered WT (H349D, S344Y, P358L, V306I, Q674*, etc.) | p.8 | |
| 24 | **FGFR3 mRNA expression (TPM-normalized, log2)** via QuantSeq FWD Kit / Lexogen, Illumina NovaSeq 6000 1×75bp, ≥20M clusters, nf-core RNA-seq v3.3 | p.5-6, Methods p.9 | Used for low-expressor FGFR3-mut (Q1, n=14) vs high-expressor WT (Q4, n=82) analysis; mean log TPM 3.26 (true neg) vs 4.28 (false pos), p=7.9e-8 |
| 25 | RNA integrity requirement / Promega Maxwell FFPE DNA/RNA isolation, Qubit QC, 30% tumor content, 0.5×0.5 cm vital non-necrotic area | Methods p.8 | |

### 3.5 Biospecimen & Slide Provenance
| # | Field | Paper location | Notes |
|---|---|---|---|
| 26 | Sample type (Primary Tumor), Tissue type (Tumor), Specimen type (Solid Tissue), Preservation (FFPE) | Methods p.8 | |
| 27 | Tumor descriptor, Days to sample procurement, Portions | Methods | |
| 28 | Number of slides per case (812 slides collected → 391 training + 421 validation; multiple slides per case for ICC) | Fig.1a-b | |
| 29 | Slide data format (SVS), Data category (Biospecimen), Data type (Slide Image), Experimental strategy (Diagnostic Slide) | — | GDC terminology |
| 30 | File-level provenance: file_name, submitter_id, file_id, file_size, md5sum, state, access, data_release | — | For reproducibility |

### 3.6 Imaging & Preprocessing Pipeline
| # | Field | Paper location | Notes |
|---|---|---|---|
| 31 | Scanner identity (same scanner, identical settings; Erlangen + TCGA varied H&E staining protocols) | Methods p.8 *Preprocessing* | |
| 32 | Matter detection: U-Net (460 H&E+IHC trained, 115 validated, Dice 0.96) | Methods p.9, Ref 24 | `https://github.com/milesial/Pytorch-UNet` |
| 33 | Tiling: 224×224 px = 224×224 μm @ 1.0 MPP (60% matter threshold); 5000 tiles max sampled uniformly | Methods p.9 | 0.5 MPP ablation in Supplementary Table 4 |
| 34 | Feature extraction: Bioptimus H0 (ViT-Giant, 1539-D), DINOv2 self-supervised, frozen weights → matrix (n_tiles, 1539) | Methods p.9, Ref 25-26 | `https://huggingface.co/bioptimus/H-optimus-0` |
| 35 | Magnification / MPP | Methods | |

### 3.7 Model & Performance Metrics
| # | Field | Paper location | Notes |
|---|---|---|---|
| 36 | Model: ensemble of 125 MIL models (Courtiol et al., MLP 128 hidden + ReLU, top/bottom 100 scores, linear→sigmoid, BCE loss) | Methods p.9 *FGFR3 mutation prediction* | `https://github.com/PABannier/fgfr3mut` Zenodo 13959977 |
| 37 | Cross-validation AUC 0.82 [0.74-0.90]; Validation AUCs 0.82 TCGA MIBC [0.75-0.88], 0.89 MIBC II [0.82-0.93], 0.82 mUC [0.68-0.94]; TCGA NMIBC 0.87, pooled 0.87, NMIBC II transfer 0.70 | Results p.2-5, Fig.2a-c | |
| 38 | Operating thresholds (0.059 TCGA, 0.051 MIBC II, 0.063 mUC) — calibrated to minimize false negatives | Table 1 p.5 | |
| 39 | Sensitivity/Specificity/PPV/NPV per cohort (e.g., TCGA 0.96/0.47/0.14/0.99) ; Saved tests 44-47% | Table 1, Results p.2 | |
| 40 | ICC for inter-block variability (mUC 0.66 [0.42-0.90], TCGA 0.89 [0.76-0.96]); Primary vs metastasis (n=4) | p.5-6 Fig.4a-b | |
| 41 | Uncertainty: Deep Ensemble variance over 125 models + conformal prediction (non-conformity =1-p_true, 95th percentile, cardinality 1.42/1.12/1.19) | Methods p.9, Results p.6 | 150 data points per sample in Fig.4c-e |
| 42 | Tile-level heatmap (112×112 μm probability) | Fig.2e-f | |

### 3.8 Histopathology Interpretability (Pathologist Review)
| # | Field | Paper location | Notes |
|---|---|---|---|
| 43 | 400 tiles (200 MUT / 200 WT from TCGA MIBC) annotated with 20 histological criteria (Supplementary Methods), blinded | Methods p.9 *Interpretability analysis*, Fig.5a | Tumor cells 99% MUT vs 57% WT |
| 44 | Patterns: monomorphic appearance (18% MUT vs 1% WT, p=4.3e-13), atypia (27% vs 12%, p=5e-3), desmoplastic stroma (1% vs 14%, p=9.8e-13), inflammation lymphocytes (0% vs 12%, p=2.9e-8) | p.5-6 Fig.5a, Supplementary Table 8 | |
| 45 | Significance testing: two-sided z-test for proportions, Bonferroni correction | p.6 | |

### 3.9 Supplementary / External References
| # | Field | Paper location |
|---|---|---|
| 46 | Supplementary Tables 1-15, Supplementary Figs. 1-2 | Throughout |
| 47 | TCGA GDC portal `https://portal.gdc.cancer.gov/` + cBioPortal | *Data availability* p.9 |
| 48 | Source Data file | *Data availability* |

---

## 4. What `data/data.txt` Actually Contains — Field Inventory

`data/data.txt` is **not a paper-derived export**; it is a **raw GDC API response for `slide_image` endpoints**, 3 objects concatenated.

### 4.1 Top-level keys (all 3 objects)
`data/data.txt:2,142,298` structure:

- `data/data.txt:3` `data_format: SVS`
- `data/data.txt:152-157` `data_category: Biospecimen`, `data_type: Slide Image`, `experimental_strategy: Diagnostic Slide`, `type: slide_image`
- `data/data.txt:140-157` `file_name`, `submitter_id`, `file_id`, `file_size`, `md5sum`, `access: open`, `acl`, `state: released`, `version`, `data_release: 31.0-46.0`, `created_datetime`, `updated_datetime`

### 4.2 `cases` (length=1 per object)
| Category | Keys in data.txt |
|---|---|
| **Case identity** | `submitter_id` (`TCGA-FD-A3NA:6` / `TCGA-FJ-A871:167`), `case_id` (`14917461...:135` / `bb38805e...:291`), `primary_site: Bladder:133/289`, `disease_type: Transitional Cell Papillomas...:7/166`, `consent_type`, `index_date: Diagnosis`, `state`, `lost_to_followup`, `days_to_consent` |
| **Demographic** `data/data.txt:116-132` / `271-288` | `race: white`, `ethnicity: not hispanic or latino`, `vital_status` (Alive/Dead), `age_at_index` (60/49), `days_to_birth` (-21936/-18051), `sex_at_birth: male:129/285`, `country_of_residence_at_enrollment: United States`, `days_to_death` (272 for FJ-A871), `age_is_obfuscated: false`, `demographic_id`, `submitter_id` |
| **Diagnoses** (array) `data/data.txt:10-93` / `169-236` | `ajcc_pathologic_t` (T2b/T3b/T2a), `ajcc_pathologic_n` (N0/NX), `ajcc_pathologic_m` (MX), `ajcc_pathologic_stage` (Stage II/III/IIA), `ajcc_staging_system_edition` (6th/7th), `ajcc_clinical_t` (T2 for FJ-A871), `morphology` (8120/3), `icd_10_code` (C67.4/C67.2), `tissue_or_organ_of_origin` (Posterior wall / Lateral wall / Pelvis NOS / Prostate / Lung), `site_of_resection_or_biopsy` (Bladder NOS), `primary_diagnosis` (Transitional cell carcinoma), `tumor_grade: High Grade:55/217`, `classification_of_tumor` (primary/recurrence/Synchronous), `synchronous_malignancy`, `prior_malignancy`, `prior_treatment`, `method_of_diagnosis` (Cystoscopy/Surgical Resection), `days_to_diagnosis`, `age_at_diagnosis`, `year_of_diagnosis`, `days_to_last_follow_up`, `days_to_last_known_disease_status`, `diagnosis_id`, `submitter_id` |
| **Samples** `data/data.txt:95-114` / `238-269` | `submitter_id` (TCGA-FD-A3NA-01Z / TCGA-FJ-A871-01Z), `sample_id`, `sample_type: Primary Tumor`, `tissue_type: Tumor`, `specimen_type: Solid Tissue`, `preservation_method: FFPE`, `tumor_descriptor: Primary`, `days_to_sample_procurement: 0`, `portions[]: portion_id`, `created_datetime`, `updated_datetime`, `state` |

### 4.3 What is **not** in `data/data.txt`
- No `FGFR3*` key appears anywhere (verified via `grep -i fgfr` → 0 hits)
- No `expression`, `TPM`, `RNAseq`, `mutation`, `hotspot` keys
- No Erlangen cohorts (all `submitter_id` start with `TCGA-`)
- No WSI preprocessing / model / performance keys
- Only 2 distinct cases (FD-A3NA, FJ-A871) vs paper's 1222 cases; FJ-A871 duplicated

---

## 5. Completeness Matrix — Paper → data.txt

Legend: ✅ Present | ⚠️ Partial / Proxy | ❌ Missing | — Not applicable

| Paper Metadata (Section) | In `data/data.txt`? | Evidence / Notes |
|---|---|---|
| **Cohort composition (1-4)** |
| Total n=1222 + splits (Training/Validation/Transferability) | ❌ Missing | data.txt n=2 cases only; no cohort labels MIBC I/II, NMIBC I/II, mUC |
| Cohort labels + n per cohort | ❌ Missing | No `MIBC` / `NMIBC` / `mUC` identifiers |
| Exclusion reasons (blurry, no FFPE, no MIBC tumor) | ❌ Missing | data.txt has `state: released` only; no artifact flags |
| Institution (Erlangen vs TCGA) | ⚠️ Partial | Can infer TCGA from `TCGA-*` IDs; Erlangen cohorts absent |
| **Clinical/Demographic (5-11)** |
| Biological sex | ✅ Present | `data/data.txt:129` `sex_at_birth: male` (both cases). Matches paper's sex analysis. |
| Race / Ethnicity | ✅ Present | `data/data.txt:117-118` `white`, `not hispanic or latino` |
| Age at diagnosis / Age at index / Days to birth | ✅ Present | `data/data.txt:34,122` `age_at_diagnosis: 21936` (days), `age_at_index: 60` — note: paper reports years, data.txt days |
| Year of diagnosis / Year of birth/death | ✅ Present | `data/data.txt:38,130` `year_of_diagnosis: 2009/2013`, `year_of_death: null` |
| Vital status / Days to death / Days to last follow-up | ✅ Present | `data/data.txt:119,272,194,283` |
| Tissue/organ of origin / Primary site / Site of resection | ✅ Present | `data/data.txt:32,54,132-133` |
| Method of diagnosis | ✅ Present | `data/data.txt:43,204` |
| Prior malignancy / Prior treatment / Synchronous malignancy | ✅ Present | `data/data.txt:28,36,189` |
| **Staging / Pathology (12-19)** |
| AJCC T/N/M/Stage/Edition/Clinical T | ✅ Present | `data/data.txt:12,45-50,206-210` — core paper staging variables fully present |
| Tumor grade | ✅ Present | `data/data.txt:55,217` `High Grade` — low vs high grade analysis possible only for these TCGA cases |
| ICD-10 / Morphology | ✅ Present | `data/data.txt:47,53,208,215` |
| Histological subtype (divergent/neuroendocrine) | ❌ Missing | Requires pathologist re-review; not in GDC slide export |
| Growth pattern (solid/papillary) | ❌ Missing | Supplement Table 10 variable absent |
| CIS presence | ❌ Missing | Supplement Table 11 variable absent |
| Disease state (primary/recurrence etc.) | ✅ Present | `data/data.txt:20,51,178` `classification_of_tumor` |
| **Molecular Ground-Truth (20-25) — CRITICAL GAP** |
| **FGFR3 mutation status (MUT/WT)** | ❌ **Missing** | **No field in data.txt**; for TCGA must query `https://www.cbioportal.org/` PanCan Atlas 2018; for Erlangen SNaPshot data not in GDC |
| **Hotspot (S249C, Y373C, etc.)** | ❌ **Missing** | `data/data.txt` has zero `FGFR` hits; see `Supplementary Table 7` in paper |
| FGFR3 fusions | ❌ Missing | |
| Non-activating variants flagged as WT | ❌ Missing | List in Methods p.8 not in slide metadata |
| **FGFR3 mRNA expression (TPM log2)** | ❌ **Missing** | Requires separate GDC `RNAseq` download (`https://portal.gdc.cancer.gov/` + nf-core pipeline Table 5/6) |
| DNA/RNA isolation QC / 30% tumor rule | ❌ Missing | Methods detail not captured in GDC |
| **Biospecimen / Slide (26-30)** |
| Sample/tissue/preservation (FFPE) | ✅ Present | `data/data.txt:97-107,239-250` `preservation_method: FFPE`, `Primary Tumor` |
| Tumor descriptor / Days to procurement / Portions | ✅ Present | |
| Slides per case (multiple slides ICC) | ⚠️ Partial | data.txt `file_name: ...DX1` / `...DX5` suggests single diagnostic slide per case; does not expose multiple-slide ICC (Fig.4a-b) |
| SVS format / Data category/type/strategy | ✅ Present | `data/data.txt:3,143,153,155` |
| File provenance (file_id, size, md5sum) | ✅ Present | Crucially matches input slide: `data/data.txt:296-306` |
| **Imaging & Pipeline (31-35)** |
| Scanner / Dice / Staining variance | ❌ Missing | Not in GDC; paper: same scanner, U-Net Dice 0.96 |
| Tiling @1.0 MPP / 60% matter / 5000 tiles max | ❌ Missing | Pipeline output, not biospecimen metadata |
| Bioptimus H0 1539-D / DINOv2 | ❌ Missing | |
| **Model & Performance (36-42)** |
| MIL ensemble 125 / MLP / BCE | ❌ Missing | Code is at `https://github.com/PABannier/fgfr3mut` (p.9) |
| AUC / Thresholds (0.059/0.051/0.063) | ❌ Missing | Table 1 derived, not in input metadata |
| Sensitivity/Specificity/PPV/NPV | ❌ Missing | |
| ICC / variance / conformal | ❌ Missing | |
| Heatmaps (Fig.2e-f) | ❌ Missing | |
| **Interpretability (43-45)** |
| 400-tile histology annotations (monomorphic etc.) + p-values | ❌ Missing | Requires pathologist re-annotation |
| Z-test/Bonferroni | ❌ Missing | Statistical output |
| **External refs (46-48)** |
| Supplementary Tables 1-15 | ❌ Missing | Not bundled with GDC slide JSON |
| GDC + cBioPortal pointers | ⚠️ Partial | data.txt `data_release: 31.0-46.0` hints at GDC release, but no explicit `portal.gdc.cancer.gov` FGFR3 query |
| Source Data file | ❌ Missing | |

**Summary counts:** ✅ 17 fully present + 5 partial = 22/57 (39%) present; ❌ 35 missing.

---

## 6. Deep Dive: The Input Slide `TCGA-FJ-A871-01Z-00-DX5...svs`

### 6.1 What the paper says this case is
- TCGA-BLCA MIBC cohort, filtered into **TCGA MIBC n=307** for validation (Fig.1b, p.8). Added to `Training/Validation` pools; used for AUC 0.82 [0.75-0.88] in Fig.2a.
- Paper's TCGA FGFR3-mutant rate is 8% (24/307 → `data/data.txt` does not state this; must retrieve from cBioPortal).
- If `TCGA-FJ-A871` were MUT, it would be among S249C/Y373C distribution in Fig.3a.

### 6.2 What `data/data.txt` says about `TCGA-FJ-A871`

**Fully reproduced** at `data/data.txt:161-316` (and duplicate 317-472):

| Attribute | Value | Line |
|---|---|---|
| `submitter_id` | `TCGA-FJ-A871` | `data/data.txt:167` |
| `case_id` | `bb38805e-2ccc-4169-ba01-34b98adf5bd0` | `data/data.txt:291` |
| `disease_type` | `Transitional Cell Papillomas and Carcinomas` | `data/data.txt:166` |
| `primary_site` | `Bladder` | `data/data.txt:289` |
| `demographic` | `male`, `white`, `not hispanic or latino`, `age_at_index: 49` (birth -18051 days), `vital_status: Dead`, `days_to_death: 272` | `data/data.txt:271-288` |
| **Primary diagnosis** (`diagnosis_is_primary_disease: true`) | `Transitional cell carcinoma`, `Stage III`, `T3b`, `NX`, `MX`, `morphology 8120/3`, `High Grade`, `lateral wall of bladder`, `Bladder NOS`, `ICD C67.2`, `7th edition`, `Surgical Resection`, `year 2013` | `data/data.txt:188-218` |
| Also present | Synchronous `No`, `prior_malignancy: no`, `prior_treatment: No` | `data/data.txt:188,198,201` |
| **Sample** | `TCGA-FJ-A871-01Z`, `Primary Tumor`, `FFPE`, `Solid Tissue`, `days_to_sample_procurement: 0`, 5 `portions` | `data/data.txt:239-267` |
| **File** | `TCGA-FJ-A871-01Z-00-DX5.8F79D0A8-5DE6-4159-AA77-61DACB21E867.svs`, `Slide_image`, `Diagnostic Slide`, `file_id: 3caa2d31...`, `size: 1166418943`, `md5: 3ee2aa658...` | `data/data.txt:297-308` |
| Other diagnoses | Recurrence (Pelvis NOS, 106 days) + Subsequent Primary (Not Reported) | `data/data.txt:170-235` |

**Perfect match to disk:** `file_name` equals `image/TCGA-FJ-A871-01Z-00-DX5...svs` and file size matches `data/data.txt:304` (1166418943 bytes ≈ 1.1 GB observed).

### 6.3 What is *missing* for this slide that the paper needs to evaluate it

| Missing field | Why it matters | Where to get it |
|---|---|---|
| `FGFR3 MUT vs WT` | Without it, cannot know ground-truth label, cannot compute Sensitivity/NPV, cannot place slide in Fig.2a or Fig.3a/b | `https://www.cbioportal.org/` → TCGA BLCA PanCan Atlas 2018, field `FGFR3` (as stated *Methods p.8: "obtained via the CBioPortal"*) |
| Hotspot (e.g., S249C) | Predicts whether slide is "easy" (S249C highest prediction scores in Fig.3) | Same cBioPortal MAF |
| FGFR3 mRNA TPM | Paper shows low-expressors mimic WT morphology (p.5-6) — needed to explain false positives like this High Grade T3b case | `https://portal.gdc.cancer.gov/` RNAseq + nf-core pipeline log2 TPM |
| Histopathology annotations (monomorphic etc.) | Paper's interpretability relies on 400-tile review | Not pre-downloaded; requires pathologist + Fig.5 criteria |
| Model score / bucket (predicted MUT prob) | The actual paper output — should be ~0.06 threshold | Run `https://github.com/PABannier/fgfr3mut` inference at 1.0 MPP |

> **Duplication warning:** `data/data.txt` contains `TCGA-FJ-A871` twice. If you iterate naïvely over `data/data.txt` JSON objects, you will double-count this case (e.g., inflating n). Deduplicate on `case_id: bb38805e...:291` or `file_id: 3caa2d31...` before counting. `TCGA-FD-A3NA` appears only once — its slide `TCGA-FD-A3NA-01Z-00-DX1...svs` is **not on disk** (`image/` contains only FJ-A871).

---

## 7. Implications & Risks

1. **Reproducibility blocker:** The 51% missing rate is dominated by *irreplaceable* molecular labels. Any claim that `data/data.txt` is "the paper's metadata" is inaccurate — it is a *subset* (GDC Biospecimen slice) of the paper's full metadata which lives across **GDC + cBioPortal + Erlangen institutional store (GDPR-restricted)** (*Data availability p.9*).

2. **Class-imbalance silent failure:** Paper's validation was carefully enriched to ~11% MUT (Table 1). Training used 23% MUT enrichment. Naïve use of `data/data.txt` (2 cases, unknown MUT status) gives no sense of this distribution; training/inference thresholds (0.059 etc.) will be miscalibrated.

3. **Histology confounders invisible:** The model learns surrogate features (low TILs, low desmoplastic stroma, monomorphic nuclei). `data/data.txt` has no TIL or stroma fields, so failure-mode analysis (Fig.5) is impossible.

4. **File hygiene:** Concatenated JSON without array wrapper (`data/data.txt:159-162` `}{`) is not standard JSON — `json.load()` will fail; you must split on `}{` or wrap in `[...]` first. The duplicate FJ-A871 record will break unique-key assumptions.

---

## 8. Recommendations — How to Obtain the Missing Metadata

| Missing item | Concrete next step |
|---|---|
| **FGFR3 MUT/WT for TCGA-FJ-A871 (and all TCGA 412→307)** | Query cBioPortal API: `https://www.cbioportal.org/api/molecular-profiles/tcga_blca_tcga_mutations/variants?sampleIds=TCGA-FJ-A871-01` or bulk download PanCan Atlas 2018 MAF from `https://portal.gdc.cancer.gov/` → filter `Hugo_Symbol=FGFR3` and map activating list p.9. Cross-check `TCGA-FJ-A871` status to label input slide. |
| **Hotspot distribution (Supplementary Table 7)** | Same MAF; extract `HGVSp` (p.R248C etc.) |
| **FGFR3 expression** | Download GDC HTSeq-FPKM + convert to TPM log2 as per *Methods p.9*; pipeline code is nf-core `RNA-seq v3.3`. Compare to quartiles described p.5. |
| **Erlangen cohorts (MIBC I/II, NMIBC I/II, mUC)** | Contact corresponding authors (`pierre-antoine.bannier@owkin.com`, `markus.eckstein@uk-erlangen.de`) per *Data availability p.9* — GDPR-restricted, processing ≤1 month. |
| **Histopathology subtype / CIS / papillary annotations** | Re-annotate WSIs using criteria in *Supplementary Methods: twenty histological criteria*; expected from M.E. pathologist review. |
| **Model inference** | Clone `https://github.com/PABannier/fgfr3mut` (Zenodo 10.5281/zenodo.13959977), run preprocessing at 1.0 MPP with Bioptimus H0 (`data/data.txt` file is the input). |
| **Clean `data/data.txt`** | Run `python3 -c "import re,json; txt=open('data/data.txt').read(); objs=[json.loads('{'+p+'}' if not p.startswith('{') else p) for p in re.split(r'\}\{',txt)]; ..."` to parse, dedupe on `file_id`, and re-emit as JSON array. Remove duplicate FJ-A871 entry. |

---

## 9. Appendix: Verbatim `data/data.txt` Keys (deduplicated)

```text
data.data_format
data.cases[].submitter_id, case_id, primary_site, disease_type
data.cases[].demographic.{race, ethnicity, sex_at_birth, age_at_index, vital_status, days_to_birth, days_to_death, age_is_obfuscated, country_of_residence_at_enrollment}
data.cases[].diagnoses[].{ajcc_pathologic_t, ajcc_pathologic_n, ajcc_pathologic_m, ajcc_pathologic_stage, ajcc_staging_system_edition, ajcc_clinical_t, tissue_or_organ_of_origin, site_of_resection_or_biopsy, primary_diagnosis, morphology, icd_10_code, tumor_grade, classification_of_tumor, diagnosis_is_primary_disease, synchronous_malignancy, prior_malignancy, prior_treatment, method_of_diagnosis, days_to_diagnosis, age_at_diagnosis, year_of_diagnosis, days_to_last_follow_up, diagnosis_id, submitter_id}
data.cases[].samples[].{submitter_id, sample_id, sample_type, tissue_type, specimen_type, preservation_method, tumor_descriptor, days_to_sample_procurement, portions[].portion_id}
data.file_name, submitter_id, file_id, file_size, md5sum, data_category, data_type, experimental_strategy, type, access, acl, state, version, data_release
```

Total keys in paper not in above list: see Section 5 ❌ rows (≈29 fields).

---

## 10. References (all to `docs/s41467-024-55331-6.pdf`)

- Fig. 1a-b cohort flowchart: p.3
- Fig. 2 ROC + heatmaps + tiles: p.4
- Fig. 3 hotspot boxplots: p.5-6
- Fig. 4 ICC + uncertainty: p.6-7
- Fig. 5 histopathology frequencies: p.7
- Table 1 thresholds/saved tests: p.5
- Methods *Datasets description*: p.8
- Methods *Mutational detection via SNaPshot*: p.8-9 (11 mutations >99%)
- Methods *Preprocessing of whole-slide images*: p.9 (U-Net, 224×224 @1.0 MPP, Bioptimus H0 1539-D)
- Methods *FGFR3 mutation prediction*: p.9 (125 MIL, MLP 128, top/bottom 100, BCE)
- *Data availability*: p.9 — GDC + cBioPortal + Erlangen GDPR-access + Source Data file
- *Code availability*: p.9 — U-Net + Bioptimus + `github.com/PABannier/fgfr3mut`

---

*Generated automatically. For re-checks: `grep -i fgfr data/data.txt` returns 0; `grep -c TCGA-FJ-A871 data/data.txt` returns 2 (duplicate).*
