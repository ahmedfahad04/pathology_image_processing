# Ground Truth Availability — What We TRULY Have vs. What We Don't

**Purpose:** Before starting any grade / stage / mutation prediction task, this file records
exactly which labels actually exist in this repository, where they come from, and which are
**missing**. Everything below was verified directly from the data files (not assumed).

**Verified on:** 2026-10-26 · data fetched from cBioPortal study `blca_tcga_pan_can_atlas_2018`
(TCGA BLCA PanCan Atlas) + the Bannier et al. Nat Commun 2024 paper supplementary files.

---

## 0. Data Sources (where the labels live)

| File | What it contains | Size |
| --- | --- | --- |
| `output/mutation_analysis/per_slide_table.csv` | One row per DX slide: FGFR3 status, hotspot, VAF, RSEM expression, total mutation count, AGE, SEX, STAGE, GRADE | 582 slides |
| `data/data_manifest.json` | Same per-slide labels, viewer-ready copy | 582 entries |
| `data/<slide-folder>/cbioportal_*.json` | Deep per-slide records (FGFR3 mutation detail, expression, CNA, clinical) | 1 local slide |
| `output/paper/paper_lookup.json` | Paper's own patient-level records: **pathologist histology features**, molecular subtype, pT stage, nodal status, AI score | 413 patients / 440 DX slides |
| `image/*.svs` | The actual slide images | **1 slide only** (TCGA-FJ-A871 — FGFR3 **WT**) |

> ⚠️ **Key limitation:** we have **labels for 582 slides but only 1 slide image locally**.
> More SVS files must be downloaded via GDC (`scripts/utils/download_svs.sh`) before any
> slide-based training can happen.

---

## 1. Your Grading-Feature Table — Availability Check

Status meanings: ✅ = truly available as ground truth · ⚠️ = weak/proxy label only · ❌ = nothing exists

| Feature (from your table) | Status | What we actually have |
| --- | --- | --- |
| **Architecture** (papillary vs solid/branching) | ✅ | `papillary_morphology` = positive/negative → 73 pos / 254 neg (paper_lookup). Also `tumor_other_histologic_subtype`: Papillary 103 / Non-Papillary 220 |
| **Nuclear size** (large, unequal 2–3×) | ⚠️ | Only `irregular_nuclei` = positive/negative → **12 pos** / 315 neg. Extremely rare positives — too weak to train on, usable only as a sanity/evaluation hint |
| **Nuclear polarity** (loss, disarray) | ❌ | No label anywhere in cBioPortal clinical or paper files |
| **Chromatin** (coarse, hyperchromatic) | ❌ | No label anywhere |
| **Nucleoli** (large, prominent) | ❌ | No label anywhere |
| **Mitosis** (count, atypical forms) | ❌ | No label anywhere (mitotic count is not in TCGA clinical data) |
| **Necrosis** | ❌ | No label anywhere |
| *(bonus)* **Perinuclear clearing** | ✅ | `perinuclear_clearing` → 24 pos / 303 neg |
| *(bonus)* **Distinct cell borders** (plasmacytoid-ish look) | ✅ | `distinct_cell_borders` → 22 pos / 305 neg |

**Bottom line for grading features:** only **architecture (papillary)** has enough positives to
be meaningfully used. Nuclear size/polarity/chromatin/nucleoli/mitosis/necrosis have **no usable
ground truth** — they must either be annotated by a pathologist on the images, or predicted
unsupervised / weakly-supervised (that is exactly the research problem, not an input).

All these labels are **patient-level** (from paper review sheets), linked to DX slide files via
`paper_dx_files` (440 slides). They are **not pixel/region annotations**.

---

## 2. Grade (High vs Low)

| Source | High Grade | Low Grade | Missing |
| --- | --- | --- | --- |
| `per_slide_table.csv` → `GRADE` | **535** | **23** | 24 |
| `paper_lookup.json` → `grade_2004` | 305 | 20 | 88 |

**Verdict: ⚠️ technically available but NOT usable as a classification target.**
~92–94% of TCGA BLCA slides are High Grade (the cohort is muscle-invasive bladder cancer by
design; low-grade papillary tumours are mostly non-muscle-invasive and were not collected).
A classifier trained on this would just learn "always predict High Grade".
**Recommendation:** skip grade as a primary task; if needed, use it only as a secondary/context
feature, or collect low-grade NMIBC slides separately.

---

## 3. Stage

| Source | Distribution | Missing |
| --- | --- | --- |
| `per_slide_table.csv` → `STAGE` (AJCC pathologic) | Stage II = 116, Stage III = 118, Stage IV = 124, Stage I = 7 | 217 |
| `paper_lookup.json` → `pT_stage` | T2(x/a/b)=98, T3(x/a/b)=159, T4(x/a/b)=44, T1=1 | 109 |
| `paper_lookup.json` → `nodal_status` | N0 = 185, N1 = 35, N2 = 66, N3 = 5, NX = 31 | 91 |

**Verdict: ✅ usable.** ~365 slides have a stage label, roughly balanced across Stage II/III/IV.
Stage I has only 7 cases (merge into II or drop). pT-stage and nodal status give finer labels
for ~300 patients. Good candidate for a secondary prediction task.

---

## 4. Mutation Ground Truth (your mutation table checked gene by gene)

| Gene (your table) | Status | What we actually have |
| --- | --- | --- |
| **FGFR3** | ✅ **ready to use** | 76 MUT / 506 WT per slide + exact hotspot (S249C 35, Y373C/Y375C 13, G380R/G382R 11, G370C/G372C 7, R248C 4, K652E 4, others…) + **VAF for all 76 mutants** + mRNA RSEM expression (555 slides) + total mutation count (581 slides) |
| **PIK3CA** | ❌ not downloaded | cBioPortal has it — fetch with `--gene PIK3CA --entrez 5290` |
| **TP53** | ❌ not downloaded | fetch with `--gene TP53 --entrez 7157` |
| **RB1** | ❌ not downloaded | fetch with `--gene RB1 --entrez 5925` |
| **CDH1** | ❌ not downloaded | fetch with `--gene CDH1 --entrez 999` |
| **KDM6A / ARID1A / ERCC2** | ❌ not downloaded | fetch with `--gene KDM6A --entrez 7403` (ARID1A 8289, ERCC2 1905) |
| TMB (total mutation count) | ✅ | `total_mutation_count` per slide (TMB_CAT column exists but is empty — derive a threshold yourself) |

Example command to add any gene for the whole cohort:

```bash
.venv\Scripts\python.exe scripts\analysis\fetch_cbioportal_data.py \
  --file-list output\misc\tcga_blca_file_names.txt \
  --outdir data --gene TP53 --entrez 7157 --batch-size 30
```

**Verdict:** FGFR3 mutant-vs-WT slide classification is the **only mutation task ready to train
today** (76/506, use class weighting). The mutation→phenotype links in your table
(FGFR3 = finger-like papillary, TP53 = large irregular nuclei + necrosis, CDH1 = plasmacytoid…)
are biologically correct, but **the phenotype side of those links has no per-slide labels**
except the few histology flags listed in section 1 — so you can correlate, not supervise.

---

## 5. Molecular Subtype (Basal / Luminal)

From `paper_lookup.json` → `molecular_subtype` (consensus subtype, pathologist/paper-assigned):

| Subtype | Patients |
| --- | --- |
| Luminal papillary | 113 |
| Basal squamous | 110 |
| Luminal infiltrated | 60 |
| Luminal (unspecified) | 23 |
| Neuronal | 17 |
| missing | 90 |

**Verdict: ✅ usable** for ~323 patients (attach to their DX slides via `paper_dx_files`).
5-class with imbalance — collapsing to Basal-vs-Luminal gives ~170 vs ~196, a reasonable binary task.

---

## 6. Variant Histologies (your last rows: squamous, plasmacytoid, small cell)

From `paper_lookup.json` → `histological_subtype`:

| Label | Count |
| --- | --- |
| urothelial (pure) | 281 |
| urothelial **with squamous differentiation** | 39 |
| **Small cell / neuroendocrine** | 3 |
| **Plasmacytoid carcinoma** | 2 |
| micropapillary features | 2 |

**Verdict: ⚠️ labels exist but counts are tiny.** Keratin-pearl/squamous (39) can support a
small exploratory experiment; plasmacytoid (2) and small cell (3) are case studies only —
not trainable. No per-region annotation of the differentiation exists.

---

## 7. Extra Data We Have Beyond Your Tables (also available as ground truth / features)

Everything below is **already downloaded** in this repo — most of it useful as extra clinical
context, auxiliary labels, or survival endpoints for the prediction tasks.

### 7.1 Patient identity & demographics

| Data | Where it lives | Coverage / example |
| --- | --- | --- |
| Patient / sample barcodes (`TCGA-FJ-A871`, `TCGA-FJ-A871-01`) | `per_slide_table.csv`, `data_manifest.json`, every JSON | all 582 slides |
| Slide file ↔ **GDC file_id** mapping (to download the actual SVS) | `output/misc/tcga_blca_slides.tsv` (+ `.txt`/`.csv` name lists) | all 926 TCGA-BLCA slides |
| **AGE** | `per_slide_table.csv` (`AGE`) + `paper_lookup.json` (`age`) | 369 slides / 327 patients |
| **SEX** | same | 369 slides / 327 patients |
| RACE, ETHNICITY, GENETIC_ANCESTRY_LABEL | `data/<slide>/cbioportal_*_patient_clinical.json` | per-slide deep records |
| DAYS_TO_BIRTH, DAYS_TO_INITIAL_PATHOLOGIC_DIAGNOSIS | same | per-slide |
| ICD-10 code, ICD-O-3 histology + site | same | per-slide |

### 7.2 Survival & treatment outcome (enables **prognosis** tasks!)

| Data | Where it lives | Meaning |
| --- | --- | --- |
| `OS_MONTHS` / `OS_STATUS` | `*_patient_clinical.json` | overall survival (months + dead/alive) |
| `DSS_MONTHS` / `DSS_STATUS` | same | disease-specific survival |
| `PFS_MONTHS` / `PFS_STATUS` | same | progression-free survival |
| `NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT` | same | recurrence flag |
| `RADIATION_THERAPY`, `HISTORY_NEOADJUVANT_TRTYN` | same | treatment history |
| `neoadjuvant_history`, `noninvasive_therapy` | `paper_lookup.json` | paper-annotated therapy (327 patients) |
| `PERSON_NEOPLASM_CANCER_STATUS`, `PRIOR_DX` | `*_patient_clinical.json` | cancer status at collection |

### 7.3 Detailed staging (finer than the STAGE column)

| Data | Where it lives | Values seen |
| --- | --- | --- |
| `PATH_T_STAGE`, `PATH_N_STAGE`, `PATH_M_STAGE` | `*_patient_clinical.json` | TNM components per patient |
| `AJCC_PATHOLOGIC_TUMOR_STAGE`, `AJCC_STAGING_EDITION` | same | stage + edition used |
| `pT_stage` (T2a…T4b) | `paper_lookup.json` | 304 patients labeled |
| `nodal_status` (N0–N3) | `paper_lookup.json` | N0=185, N1=35, N2=66, N3=5 |

### 7.4 Sample / tissue metadata & genomics-quality scores

| Data | Where it lives | Meaning |
| --- | --- | --- |
| `SAMPLE_TYPE` / `sampleType` | `*_sample_info.json` | e.g. "Primary Solid Tumor" |
| `TUMOR_TISSUE_SITE`, `TUMOR_TYPE`, `CANCER_TYPE_DETAILED`, `ONCOTREE_CODE` | `*_sample_clinical.json` | tissue/tumour classification |
| `TISSUE_SOURCE_SITE` (+ code) | same | which hospital the tissue came from (batch-effect check!) |
| Prospective/retrospective collection indicators | same | collection quality flag |
| `GRADE` | same | cBioPortal's own grade column |
| `MUTATION_COUNT`, `TMB_NONSYNONYMOUS` | same | true TMB values (better than the empty TMB_CAT) |
| `FRACTION_GENOME_ALTERED`, `ANEUPLOIDY_SCORE` | same | CNA burden |
| `MSI_SCORE_MANTIS`, `MSI_SENSOR_SCORE`, `TBL_SCORE` | same | MSI / mutational-signature scores |
| `sequenced`, `copyNumberSegmentPresent` | `*_sample_info.json` | data-availability flags |

### 7.5 Mutation-level genomics (per slide, deeper than the FGFR3 columns)

| Data | Where it lives | Example |
| --- | --- | --- |
| **All somatic mutations** of a sample: gene, protein change, chromosome, start/end, ref→alt allele, variant type (SNP/INS/DEL), tumor Alt/Ref read counts, validation status | `data/<slide>/cbioportal_*_mutations.json` | local slide: 45 records (e.g. `S1778*`, chr16, Alt count 19/54 reads) |
| Cohort-wide FGFR3 mutation table (hotspot, VAF, patients) | `data/cbioportal_blca_tcga_pan_can_atlas_2018_FGFR3_mutations_cohort.json` | 68 records / 59 patients |
| FGFR3 mRNA **RSEM expression** | `*_FGFR3_expression.json` + `FGFR3_RSEM` column | 555 slides |
| FGFR3 discrete **CNA** (GISTIC) | `*_FGFR3_CNA.json` | fetchable per sample |
| Full molecular-profile catalog (every expression/CNA/mutation profile of the study) | `data/cbioportal_molecular_profiles.json` | lets you fetch any gene's data |

### 7.6 PanCancer-Atlas special scores

| Data | Where it lives | Meaning |
| --- | --- | --- |
| `BUFFA_HYPOXIA_SCORE`, `RAGNUM_HYPOXIA_SCORE`, `WINTER_HYPOXIA_SCORE` | `*_patient_clinical.json` | 3 published hypoxia gene-signature scores per patient |
| `SUBTYPE` | same | cBioPortal subtype attribute |
| `WEIGHT` | same | patient weight |

### 7.7 Paper-specific annotations (from `paper_lookup.json`, 327 fully-annotated patients)

| Data | Meaning |
| --- | --- |
| `is_mibc`, `in_loeffler`, `in_mibc_experiment`, `in_loeffler_experiment` | which paper experiment each patient belongs to |
| `ai_score` (0–1) + `ai_classification` (mut/wt) | the paper's **AI FGFR3 prediction** — the reference system to beat/compare |
| `pathologist_judgement` (mut/wt, 25 mut / 302 wt) | a **pathologist's slide-based FGFR3 call** — human reference |
| `fgfr3_paper` (hotspot string) + `fgfr3_molecular_status` (wt/mut) | DNA-based truth as published in the paper |
| `dx_rev`, `cryo_summary`, `exclusion_summary` | review notes: which slides are good/excluded and why |
| `grade_2004`, `pT_stage`, `nodal_status`, `molecular_subtype`, histology flags | the paper's clinical table (detailed in sections 1–6) |

### 7.8 Analysis outputs already generated (context, not labels)

| Output | Content |
| --- | --- |
| `output/mutation_analysis/fig1–7*.png` + `report.md` | FGFR3 prevalence, hotspots, expression, stage, VAF, clinical figures |
| `output/mutation_analysis/per_sample_table.csv` | one row per unique **sample** (278 rows) vs per-slide table |
| `output/mutation_analysis/cohort_FGFR3_all_records.csv` | the 68 cohort FGFR3 variant records |
| `output/mutant_locations/WHICH_IMAGES_ARE_MUTANT.csv` | MUT/WT label + hotspot + VAF per SVS |

---

## 8. What We Do NOT Have At All (be honest about this)

1. **Mitotic figures, necrosis, chromatin texture, nucleoli, nuclear polarity** — no labels in
   any file. To use these as prediction targets you need new pathologist annotation on the
   images (or accept unsupervised/weakly-supervised proxies).
2. **Pixel/region-level ground truth** — none. A mutation is a DNA-level call on the patient's
   tumour bulk (see `todo.md` note 4: chr4:1806089 G→T is a DNA address, **not** an x,y on the
   slide). Any "mutant region" heatmap is a model prediction, never ground truth.
3. **Slide images** — labels exist for 582 slides, but only **1 SVS** is downloaded locally.
   All WSI-based training needs more downloads first.
4. **Paper AI outputs are predictions, not truth** — `ai_score` / `ai_classification`
   (457-slide Loeffler experiment) and `pathologist_judgement` (FGFR3 call made by a
   pathologist from the slides) are useful as *reference systems to compare against*, while
   `fgfr3_molecular_status` / `fgfr3_paper` (DNA-based) is the real FGFR3 ground truth.

---

## 9. Practical Verdict — Which Tasks Can We Start Now?

| Task | Ground truth ready? | Usable labels | Notes |
| --- | --- | --- | --- |
| **FGFR3 MUT vs WT from WSI** | ✅ Yes | 76 / 506 | Primary ready task. Needs more SVS downloads + class weighting |
| **Stage (II/III/IV) from WSI** | ✅ Yes | ~360 | Balanced enough; drop Stage I (7) |
| **Molecular subtype (Basal vs Luminal)** | ✅ Yes | ~320 | From paper_lookup; collapse 5 classes to 2 |
| **Survival / prognosis (OS, DSS, PFS)** | ✅ Yes | per-slide clinical JSONs | `OS_MONTHS`/`OS_STATUS` etc. — enables time-to-event models |
| **Grade High vs Low** | ❌ No (92% High) | 23 Low only | Imbalanced to the point of uselessness |
| **Histology flags (papillary, irregular nuclei…)** | ⚠️ Weak | 12–73 positives | Aux/eval only, not primary training targets |
| **Mitosis / necrosis / chromatin / nucleoli / polarity** | ❌ None | 0 | Needs new annotation |
| **Other-gene mutation status (TP53, PIK3CA…)** | ⚠️ One fetch away | ~500+ each | Run `fetch_cbioportal_data.py --gene …` |
| **Mutant-region localization on the slide** | ❌ No pixel truth | 0 | Prediction/heatmap only — validate against paper AI score |

---

## 10. One-Page Summary — Everything in a Single Table

Legend: ✅ = truly available · ⚠️ = weak / needs work · ❌ = not available

| # | Item | Have? | Coverage | Use it for |
| --- | --- | --- | --- | --- |
| 1 | **FGFR3 mutation (MUT/WT)** | ✅ | 76 MUT / 506 WT | **Primary task — ready now** |
| 2 | FGFR3 hotspot + VAF | ✅ | all 76 mutants (S249C, Y373C…) | fine-grained mutant subclass analysis |
| 3 | FGFR3 expression (RSEM) | ✅ | 555 slides | correlate morphology ↔ expression |
| 4 | TMB (mutation count) | ✅ | 581 slides | covariate / stratification |
| 5 | Other genes (TP53, PIK3CA, RB1, CDH1, KDM6A, ARID1A, ERCC2) | ⚠️ | 0 downloaded — **one API fetch away** | multi-gene tasks after running `fetch_cbioportal_data.py --gene …` |
| 6 | **Stage (AJCC II/III/IV)** | ✅ | ~365 slides, balanced | secondary prediction task |
| 7 | pT stage + nodal status | ✅ | ~300 patients | finer staging / N0-vs-N+ task |
| 8 | Grade (High/Low) | ❌ | 535 High vs **23 Low** | too imbalanced — **skip** |
| 9 | **Molecular subtype** | ✅ | ~323 (Basal ~170 / Luminal ~196 after collapsing) | Basal-vs-Luminal task |
| 10 | **Survival (OS / DSS / PFS months+status)** | ✅ | per-slide clinical JSONs | prognosis / time-to-event models |
| 11 | Age, Sex, Race, ancestry, ICD codes | ✅ | 369+ slides, extra in JSONs | covariates, subgroup analysis |
| 12 | Hypoxia signatures (Buffa/Ragnum/Winter) | ✅ | per patient JSONs | extra biological covariates |
| 13 | Treatment history (radiation, neoadjuvant) | ✅ | per patient JSONs | confounder control |
| 14 | Papillary architecture flag | ⚠️ | 73 pos / 254 neg | weak label — aux/eval only |
| 15 | Irregular nuclei / perinuclear clearing / cell borders | ⚠️ | 12 / 24 / 22 positives | too rare to train — eval hints only |
| 16 | Squamous / plasmacytoid / small-cell variants | ⚠️ | 39 / 2 / 3 cases | squamous = small experiment; others = case studies |
| 17 | Nuclear polarity, chromatin, nucleoli, mitosis, necrosis | ❌ | **0 labels** | needs new pathologist annotation |
| 18 | Pixel / region-level truth (mutant area on slide) | ❌ | **0** | heatmaps are predictions, never ground truth |
| 19 | **WSI slide images (.svs)** | ❌ | **1 of 582 labeled slides** | download more via GDC before any image training |
| 20 | Paper AI score + pathologist FGFR3 call | ✅ | 327 patients | **reference systems to compare your model against** (not ground truth) |

**One-line takeaway:** *Labels are rich and ready for **FGFR3 status, stage, subtype, and survival** — but the slide images themselves (only 1 downloaded) and the microscopic grading features (mitosis, necrosis, chromatin…) are the two hard gaps to close before model training.*

---

*Generated for the Pathology Image Processing project — TCGA BLCA FGFR3 pipeline, 2026.*
