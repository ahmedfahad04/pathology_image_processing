
# FGFR3 Mutation Report — Layman Edition
**Paper:** Bannier et al. Nat Commun 2024 — FGFR3 in Muscle-Invasive Bladder Cancer (MIBC/mUC)
**Local data snapshot:** data → 582 slides, 278 unique samples, 257 patients
**Study-wide (cBioPortal):** 411 total BLCA samples, 59 patients carry ANY FGFR3 variant (68 records)


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


## 1-Sentence Takeaway
> In this local snapshot, **35 of 278 samples (12.6%) carry an FGFR3 mutation that
> would jam the growth pedal ON** (activating hotspot). The most common jam is **S249C**.
> Across the full 411-patient TCGA BLCA cohort the rate is 59/411 (14.4% any FGFR3 variant, ~8-14% activating
> depending on MIBC vs NMIBC enrichment — exactly the imbalance the paper exploits).

## Why Prevalence Matters to You (the developer)
- Machine-learning models on slides are **highly imbalanced**: ~9 WT for every 1 MUT.  Accuracy alone
  lies — you must look at sensitivity/NPV.
- The paper's AI acts as a **rule-out test**: if it says WT, you skip expensive DNA testing in ~40%
  of cases (NPV ~0.99).  That is the product goal our image features should copy.

## Key Numbers (local snapshot, activating definition)

| Metric | Value |
|---|---|
| **Slides** | 582 (some patients have DX + TSA slides) |
| **Unique samples** | 278 |
| **Patients** | 257 |
| **MUT (activating)** | 35 (12.6%) |
| **MUT (any FGFR3 variant)** | 39 (14.0%) |
| **Non-activating FGFR3 variants → counted as WT per paper** | 4 (e.g. Q674*, H349D, etc.) |
| **WT** | 243 (87.4%) |
| **Median VAF (MUT)** | 0.52 (n=35) |
| **Expression available** | 254/278 (91.4%) |
| **Median RSEM MUT vs WT** | 14107 vs 2124  (paper: TPM log2 4.28 FP-WT vs 3.26 TN-WT, p=7.9e-8) |
| **CNA distribution (GISTIC)** | {}  (−2=homDel, −1=hemDel, 0=neutral, 1=gain, 2=amp) |
| **Top co-mutated genes (all samples, MUT+WT)** | TTN×503, TP53×350, MUC16×271, KMT2D×239, ARID1A×194, MACF1×162, KDM6A×159, KMT2C×157 |

> **Cohort context (ground truth from cBioPortal):** 59 patients with ANY FGFR3 variant / 411 total (14.4%).  Top hotspots cohort-wide: S249C×32, Y373C×8, G370C×5, R248C×3, G380R×2, S371C×2.

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
| `per_sample_table.csv` | One row per unique sample (278 rows) — all parsed fields |
| `per_slide_table.csv` | One row per slide image (582 rows) |
| `cohort_FGFR3_all_records.csv` | 68 FGFR3 variant records cohort-wide |
| `fig1_prevalence.png`  … `fig7_histology_surrogates.png` | 7 layman-annotated figures (this report) |
| `report.md` (this file) | Human-readable synthesis |

## Re-run / Extend

```bash
# Full fetch (926 slides) then re-analyze
python scripts/analysis/fetch_cbioportal_data.py --file-list output/tcga_blca_file_names.txt --outdir data --batch-size 30
python scripts/analysis/analyze_fgfr3_mutations.py --data-dir data --out-dir output/mutation_analysis

# Per-image deep dive for one slide (e.g. the big SVS on disk)
python scripts/analysis/fetch_cbioportal_data.py --image image/TCGA-FJ-A871*.svs --outdir data
```

## Limitations & Honest Disclaimers

- This local snapshot is **only 278/411 = 67.6% of the full cohort** — rates will shift when you fetch all 926 slides.  Cohort-wide numbers (Fig.2 cohort bars) are the truth.
- Expression here is **cBioPortal RSEM (HiSeq_RNASeqV2)**, while the paper uses **TPM-normalized log2** from a separate GDC HTSeq + nf-core v3.3 pipeline — values correlate but are not interchangeable.
- No tile-level histology (monomorphic %, stroma %) is available via cBioPortal — Fig.7 bars are **paper-reported reference values**, not re-computed from your slides.  Extract them yourself from `image/*.svs` with a nuclei+TIL+stroma pipeline.
- The script counts a sample as MUT only if an **activating** hotspot is present; purely non-activating variants (H349D etc.) are correctly counted as WT per paper — toggle via `--activating-only` if you add that flag later.

---
*Generated by `scripts/analysis/analyze_fgfr3_mutations.py` on 2026-09-08 20:17.  Paper DOI: 10.1038/s41467-024-55331-6*
