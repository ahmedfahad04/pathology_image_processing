# FGFR3MUT Pipeline — Full Reproduction Guide

**Paper:** Bannier et al., *AI allows pre-screening of FGFR3 mutational status using routine histology slides of muscle-invasive bladder cancer*, Nature Communications 2024 (s41467-024-55331-6)
**Repo:** https://github.com/PABannier/fgfr3mut
**Weights + data:** https://huggingface.co/datasets/PABannier/fgfr3mut
**Goal of this doc:** explain every step like you're a CS student with no pathology background, so you can reproduce the pipeline one command at a time.

---

## 1. Run this first: reproduce the paper in 3 commands (~10 min CPU)

This is the only part the public repo lets you fully rerun (inference on TCGA). Do this before reading the rest.

```bash
# 1. Clone and install
git clone https://github.com/PABannier/fgfr3mut.git
cd fgfr3mut
conda create -n fgfr3mut_env python=3.8 -y
conda activate fgfr3mut_env
pip install .

# 2. Download weights (~50 MB) + TCGA features (>200 GB!)
python download.py --out_dir ./data_fgfr3
# layout after download:
# ./data_fgfr3/
#   models/split_0/*.pt ... split_4/*.pt  # 125 Chowder checkpoints total
#   features/<slidename>/features.npy     # pre-extracted tile embeddings
#   filtered_slides_tcga.xlsx             # which TCGA slides are MIBC vs NMIBC
#   mutations_blca_tcga_pancancer_atlas_cbioportal.txt  # ground-truth labels
#   loeffler_tcga.xlsx                    # subset used by Loeffler et al. 2022

# 3a. External validation on TCGA MIBC only (paper Fig.2a, Table 1)
python fgfr3mut/run_inference.py \
  --data_dir ./data_fgfr3 \
  --device cpu \
  --n_tiles 5000 \
  --keep_tcga_cases MIBC
# Expected: n_slides=379, n_patients=308
# Expected: AUC=0.82 [0.75-0.88], Se=0.95 [0.92-1.00], Sp=0.47 [0.30-0.72], PPV=0.14, NPV=0.99

# 3b. Same but on exact Loeffler et al. cases (for fair comparison)
python fgfr3mut/run_inference.py \
  --data_dir ./data_fgfr3 \
  --device cpu \
  --n_tiles 5000 \
  --loeffler_tcga_cases

python fgfr3mut/run_inference.py \
	--data_dir ../output/data_fgfr3_mini \
    --device cpu \
    --n_tiles 5000 \
    --keep_tcga_cases MIBC NMIBC
# Expected: n_slides=391, n_patients=327
# Expected: AUC=0.83 [0.77-0.88] vs Loeffler 0.70
```

Output: `./predictions/preds_keep_['MIBC']_cases.csv` or `preds_loeffler_cases.csv` with columns `slidename, patient_id, pred, model_filename, label` where `pred` is sigmoid-averaged ensemble probability of FGFR3-mutant.

> Tip: use `--device cuda:0` if you have a GPU — seconds instead of minutes. Use `--n_tiles` smaller for a smoke test.

**Step 1 of 7 done: you reproduced inference. Next: understand what each stage actually does.**

---

## 2. Big picture: what problem does this solve?

Think spam filter, but for cancer slides:

- **Input:** routine H&E glass slide (pink/purple stained bladder tumor, scanned to a ~100,000 x 100,000 pixel image).
- **Output:** probability `P(FGFR3-mutant)`. If low → skip expensive DNA test. If high → send for PCR/NGS confirmation.
- **Why:** 10-15% of muscle-invasive / metastatic bladder cancers have activating FGFR3 hotspot mutations → eligible for erdafitinib (FDA/EMA approved). PCR/NGS is slow, expensive, needs good tissue/RNA. H&E slide already exists for every patient.
- **Result in paper:** sensitivity 0.96-1.00, NPV 0.99-1.00, saves ~40% of molecular tests (144/307 TCGA, 72/183 MIBC II, 35/96 mUC).

Analogy: model is a **triage nurse**, not a final doctor. It is tuned to almost never miss a mutant (high sensitivity), even if it over-calls some wild-types as mutant (low specificity ~0.3-0.47 is OK).

---

## 3. End-to-end pipeline map (paper Fig.1 + Methods)

```
Step 1 SURGERY → Step 2 SLIDE → Step 3 GROUND TRUTH → Step 4 PREPROCESS → Step 5 TRAIN → Step 6 PREDICT → Step 7 EVALUATE/EXPLAIN
```

| # | Stage                                                     | Paper section              | Repo file                                                | Rerunnable?              |
| - | --------------------------------------------------------- | -------------------------- | -------------------------------------------------------- | ------------------------ |
| 1 | Cohort collection (1222 cases)                            | Datasets description       | — (private except TCGA)                                 | Partial: TCGA 307 public |
| 2 | DNA ground truth (SNaPshot / cBioPortal)                  | Mutational detection       | `dataset.py:load_fgfr3_status()`                       | Yes for TCGA             |
| 3 | WSI preprocessing: U-Net → tiles → H-Optimus-0 features | Preprocessing of WSIs      | Precomputed`.npy` (not live code)                      | Inference only           |
| 4 | MIL model: Chowder + 125-ensemble                         | FGFR3 mutation prediction  | `chowder.py`, `run_inference.py:infer()`             | Yes (forward pass)       |
| 5 | Metrics + bootstrap CIs                                   | Performance assessment     | `bootstrap.py`, `run_inference.py:compute_metrics()` | Yes                      |
| 6 | Uncertainty + conformal sets                              | Uncertainty quantification | Described, not in repo script                            | Manual                   |
| 7 | Heatmaps + pathology review                               | Interpretability           | Tile scores via`score_model()`                         | Manual                   |

**Step 2 of 7 done: you see the map. Next: each step in plain English.**

---

## 4. Step-by-step breakdown

### Step 1 — Get patients and slides (why 1222 → 391 train + 586 val)

- **Erlangen hospital (private):** 810 resected patients → 812 slides scanned on same scanner (covers stain variation).
  - MIBC I n=239 + NMIBC I n=155 = **391 training (23% mutant-enriched)**.
  - MIBC II n=183 + mUC n=96 = validation. NMIBC II n=97 = transfer experiment only.
  - Excluded 105: no FFPE slide (25), no RNA/mutation (7), no MIBC tumor on slide (73).
- **TCGA-BLCA (public):** 412 → filter to 307 MIBC with FFPE + mutation + RNAseq. Excluded 43 blurry + 73 without MIBC tumor (pathologist M.E. reviewed).
- **Key trick:** training mixes NMIBC + MIBC even though biologists call this "original sin" — NMIBC has 42-65% mutants so it shows the model many more positive examples (monomorphic tumor, low stroma). Paper Suppl. Table 3 shows this boosts TCGA-MIBC AUC vs MIBC-only training.

For you: you only need TCGA. The `.xlsx` + `.txt` in the download already encode the filtering.

### Step 2 — Ground-truth labels (how they know MUT vs WT)

Think of this as labeling your dataset:

- **Erlangen:** SNaPshot PCR — multiplex PCR of FGFR3 exons 7, 10, 15 + single-base extension with fluorescent ddNTPs, read on sequencer, double-checked. Detects 11 activating hotspots covering >99% of bladder FGFR3 mutations: `R248C, S249C, G372C, G382R, S373C, Y375C, A393E, K652E/Q/M/T`. 5x10µm FFPE sections, ≥50% tumor, Maxwell extraction, Qubit QC.
- **TCGA:** download from cBioPortal `blca_tcga_pan_can_atlas_2018`, column `FGFR3`. **Important nuance in `dataset.py:load_fgfr3_status()`:** 6 patients with non-activating variants (`TCGA-XF-A9SL, TCGA-UY-A78N, ...`) are forced back to `WT` because they wouldn't respond to FGFR inhibitor. Then binarize: `label = (mutation != "WT")`.
- **What repo does:** `TCGADataset.get_ids_to_keep()` reads `filtered_slides_tcga.xlsx`, normalizes `NMIBC-pT1/pTa → NMIBC`, keeps `["MIBC"]` by default. `load_fgfr3_status(binarize=True)` returns a `pd.Series` indexed by patient ID (first 12 chars of slide name, e.g. `TCGA-FJ-A871`).

Most common hotspots: S249C (easiest to detect), Y373C, R248C, G380R, G370C, S371C.

### Step 3 — Turn a gigapixel slide into a matrix (the part CS students must get)

A WSI is ~100k x 100k pixels — can't feed to a CNN. Three sub-steps:

**3a. Matter detection (U-Net).**

- U-Net segments "usable tissue" vs folds, pen marks, blur, glass. Trained on 460 H&E+IHC slides, Dice 0.96 on 115 val slides. Implementation ref: https://github.com/milesial/Pytorch-UNet.
- Mental model: foreground mask. Everything downstream only sees masked regions.

**3b. Tiling.**

- Cut masked regions into `224x224 px` tiles at **1.0 µm-per-pixel (MPP)** → each tile = `224x224 µm` of tissue (~112µm tiles in figures are zoomed crops for display). Keep tile if ≥60% tissue.
- Paper tested 0.5 MPP (higher mag) vs 1.0 MPP — **1.0 wins** (more context per tile, less noise). Suppl. Table 4.
- For training they cap at **5000 tiles/slide** randomly sampled (memory/speed). Same `n_tiles=5000` flag at inference.

**3c. Feature extraction (frozen foundation model).**

- Each tile → vector via **Bioptimus H-optimus-0**, a Vision Transformer Giant pretrained self-supervised with DINOv2 on millions of H&E tiles, no labels. Weights frozen.
- Paper says 1539-dim, repo code uses `in_features=1536` (`run_inference.py:get_predictions_for_mpp()`). Treat as ~1.5k embedding; mismatch is version drift, doesn't affect reproduction since you use precomputed `.npy`.
- Each `.npy` row = `[x, y, z-coords (3 cols) + embedding]`. `utils.py:load_raw_features_slide()` splits into `X_coords = [:,:3]`, `X = [:,3:]`. `load_features_to_mem()` loads all slides with `ThreadPoolExecutor`.
- End result per slide: matrix `(n_tiles, ~1536)`. Whole dataset fits in RAM as list of arrays.

You don't need to rerun 3a-3c — the HuggingFace download already gives you the matrices (>200 GB). If you had raw `.svs`, you'd need OpenSlide + U-Net + H-optimus-0 (GPU-heavy).

### Step 4 — The model: Chowder MIL + 125 ensemble (the core)

**Why MIL?** You have one label per *slide* (MUT/WT) but thousands of *tiles*. Which tiles show mutation? Unknown. MIL = "bag of instances with one bag label."

Chowder (Courtiol et al. 2018, `chowder.py:Chowder`) in 3 lines:

1. **Score every tile:** `TilesMLP`: `1536 → 128 (MaskedLinear + Sigmoid) → 1` score per tile. MaskedLinear forces padded tiles to `-inf` so sigmoid→0 and they never get picked.
2. **Keep extremes:** `ExtremeLayer(n_top=100, n_bottom=100)` sorts scores, concatenates top-100 (most mutant-looking) + bottom-100 (most WT-looking) → 200-dim slide descriptor. Intuition: diagnosis hinges on most suspicious + most normal regions, middle is ignored.
3. **Slide decision:** `MLP: 200 → 128 → 64 → 1` logit → `sigmoid` → probability. Loss: binary cross-entropy vs slide label.

Training details (paper only, no train script in repo):

- 125 separate Chowders trained (different splits/inits), predictions averaged at logit level — classic **deep ensemble** for robustness + uncertainty.
- Checkpoints stored as `models/split_{0..4}/*.pt` (25 per split × 5 splits = 125). `run_inference.py:infer()` loops over all, loads via `load_ckpt()`, runs `model(features, mask)` in `torch.inference_mode()`, applies `sigmoid()`.

**What to look at to understand code:**

- `chowder.py:TilesMLP`, `ExtremeLayer`, `MLP` — read top to bottom.
- `utils.py:pad_collate_fn()` + `SlideFeaturesDataset` — pads variable-length slides in a batch to same `N_TILES` + boolean mask (`True`=padded). Batch default 16.
- `run_inference.py:infer()` — the double loop: outer over slide batches, inner over 125 models.

### Step 5 — From slide scores to patient scores to metrics

- One patient can have multiple slides. Repo groups by `patient_id` and takes **mean** (reported) and median. `compute_metrics()` does this.
- **Operating threshold:** chosen for high sensitivity, not max accuracy. Paper thresholds: 0.059 (TCGA), 0.051 (MIBC II), 0.063 (mUC) — very low because mutants are rare (~10%) and missing one is worse than extra PCR. Code's `bootstrap.py:_compute_metrics(target_sens=0.92)` auto-picks threshold achieving ≥92% sensitivity on ROC curve.
- **Metrics:** AUC, sensitivity, specificity, PPV, NPV. CIs via **1000x bootstrap** (resample patients with replacement, recompute, take 5th/95th percentiles). See `bootstrap.py:compute_ci_with_bootstrap()`.
- **Agreement:** ICC2k (Pingouin) across slides of same patient: 0.89 TCGA, 0.66 mUC — primary vs metastasis give same call.
- **Uncertainty (paper, not in script):** variance across 125 models per patient (low variance on mutants = confident) + conformal prediction: split val set in half, non-conformity = `1 - P(true class)`, 95th percentile → prediction sets of size 1 (confident) or 2 (unsure). Expected size 1.12-1.42 → mostly confident.

### Step 6 — What does the model look at? (interpretability)

- For each slide, rank tiles by Chowder tile-score, make heatmap (Fig.2e: blue=WT, red=MUT).
- Pathologist reviewed top-100 tiles: MUT → monomorphic conventional urothelial cells, low desmoplastic stroma, lymphocyte-depleted; WT → pleomorphic, desmoplastic stroma, inflammation (all p<0.001, 400 tiles).
- Low-FGFR3-expression mutants look like WT and vice versa (RNAseq TPM check) — explains false positives (mean log TPM 4.28 FP vs 3.26 TN).
- Fusions (n=10) excluded — too rare to learn, model only claims point mutations.

### Step 7 — Limitations you must cite if reproducing

- Only point mutations, not fusions. Erlangen-only training, TCGA-only public val. NMIBC transfer drops to AUC 0.70. Needs large blind prospective + threshold calibration before clinical use.

**Step 3 of 7 done: you understand all stages. Next: run + verify.**

---

## 5. Reproduce checklist (copy-paste)

1. `python download.py --out_dir ./data_fgfr3` — verify `features/`, `models/`, `*.xlsx`, `*.txt` exist.
2. `python fgfr3mut/run_inference.py --data_dir ./data_fgfr3 --device cpu --n_tiles 5000 --keep_tcga_cases MIBC` — check `n_slides=379, n_patients=308`, AUC ~0.82.
3. Same with `--loeffler_tcga_cases` — check AUC ~0.83.
4. Open `./predictions/*.csv` — histogram `pred` by `label`; confirm threshold ~0.05-0.06 separates with high NPV.
5. (Optional) Load one `features.npy`, run `Chowder(in_features=1536, n_extreme=100).score_model` to get tile scores → overlay heatmap like Fig.2e.

Dependencies: `torch==2.3.0, numpy==1.23.5, pandas==1.1.3, scikit-learn==1.3.0, huggingface-hub==0.22.2` (`setup.py`). Python 3.8-3.9.

---

## 6. Key files cheat-sheet

| File                          | What it is in one sentence                                                                                 |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `download.py`               | One-liner`snapshot_download(PABannier/fgfr3mut)` from HuggingFace.                                       |
| `fgfr3mut/run_inference.py` | Main: load features + labels → ensemble infer → bootstrap metrics → save CSV. Start here.               |
| `fgfr3mut/dataset.py`       | `TCGADataset`: maps slide files ↔ patient IDs, filters MIBC/NMIBC, loads + binarizes labels.            |
| `fgfr3mut/chowder.py`       | Model:`TilesMLP` → `ExtremeLayer(100,100)` → `MLP` → logit.                                       |
| `fgfr3mut/utils.py`         | I/O + batching:`load_features_to_mem`, `pad_collate_fn` (mask), `SlideFeaturesDataset`, `sigmoid`. |
| `fgfr3mut/bootstrap.py`     | Eval: ROC threshold at 92% sens, confusion matrix → Se/Sp/PPV/NPV, 1000x bootstrap CIs.                   |
| `setup.py`                  | Pinned env.                                                                                                |

Paper↔code anchor examples: tiling `224x224 @1.0MPP` → `n_tiles` arg; `5000 tiles` → `get_features(n_tiles=...)`; `125 MIL` → `models_paths` loop; `BCE + sigmoid` → `sigmoid()` + Chowder forward; `AUC + NPV` → `compute_metrics()`.

---

## 7. Glossary for non-bio students

- **H&E:** pink/purple dye every tumor gets; only input the model needs.
- **FFPE:** wax block preserving tissue; source of both slide and DNA.
- **WSI:** scanned slide, gigapixels.
- **MIBC/mUC/NMIBC:** muscle-invasive / metastatic / non-muscle-invasive bladder cancer. Model targets MIBC/mUC.
- **MPP:** microns per pixel; 1.0 MPP = zoomed out vs 0.5 MPP.
- **MIL/Chowder:** learn from bag label when you don't know which tile matters.
- **NPV:** if model says WT, how often right? 0.99 = safe to skip PCR.
- **SNaPshot:** Erlangen PCR assay for hotspots. **cBioPortal:** where TCGA labels come from.

---

*Sources: paper Methods (pp.8-9), Fig.1-2, Table 1, Suppl. Tables 2-4, repo README + `run_inference.py`, `chowder.py`, `dataset.py`, `utils.py`, `bootstrap.py`, `download.py`, `setup.py` as fetched 2026-09-12.*
