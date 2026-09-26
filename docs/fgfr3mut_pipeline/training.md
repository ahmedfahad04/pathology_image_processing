# FGFR3MUT — Training Pipeline (Exact Reproduction Spec)

**Source paper:** Bannier et al., *AI allows pre-screening of FGFR3 mutational status using routine histology slides of muscle-invasive bladder cancer*, Nature Communications 15:10914 (2024), doi:10.1038/s41467-024-55331-6 — Methods section, pp.8–9 ("Preprocessing of whole-slide images", "FGFR3 mutation prediction"), Fig. 1a.
**Source code:** `fgfr3mut/` (this repo), `fgfr3mut/chowder.py`, `fgfr3mut/dataset.py`, `fgfr3mut/utils.py`.

> **Read this first:** the public `fgfr3mut` repo ships **inference only**. There is no `train.py` / training loop anywhere in `fgfr3mut/`, `fgfr3mut/build/`, or `fgfr3mut.egg-info/`. Everything below step 5 (model architecture) is verified against `chowder.py` source. The optimizer, learning rate, and exact cross-validation fold construction are **not published** by the paper or the repo; the epoch count **is** recoverable from the downloaded checkpoint filenames — see §6 "Undisclosed hyperparameters."
>
> **Verified against the actual downloaded data** (`output/data_fgfr3_mini/`, a small HuggingFace subset pulled 2026-09-25 — 6 TCGA slides across 4 patients, full `models/` + label files intact): several details below were corrected or sharpened against this real artifact rather than the paper text alone. Each such item is marked **[verified from data]**.

---

## 1. Training cohort (paper Fig. 1a, Methods "Datasets description")

| Cohort                   | n slides      | Role                 |
| ------------------------ | ------------- | -------------------- |
| MIBC I                   | 239           | Training (discovery) |
| NMIBC I                  | 155           | Training (discovery) |
| **Total training** | **391** | 23% FGFR3-mutant     |

Both cohorts are private (Erlangen hospital, FFPE H&E). They are **not** included in this repo or its HuggingFace download — only the 125 resulting model checkpoints and the public TCGA validation features are distributed. If reproducing training from scratch, you must substitute your own labeled WSI cohort (e.g. TCGA-BLCA training split) with the same file layout described below.

Validation cohorts (not used for gradient updates): TCGA MIBC (n=307 after QC), MIBC II (n=183), mUC (n=96).

---

## 2. Exact input files, per stage

### 2a. Raw input (per patient/slide)

| File                                                   | Type                                                                   | Produced by                                                                         | Notes                                                                                                                                                                   |
| ------------------------------------------------------ | ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `<slidename>.svs` (or equivalent WSI format)         | Whole-slide image, ~100,000×100,000 px, RGB, H&E-stained FFPE section | Scanner (paper: all slides digitized on the same scanner to control stain variance) | Not shipped in this repo. TCGA slides are`.svs`, downloadable from `https://portal.gdc.cancer.gov/`.                                                                |
| SNaPshot PCR trace / genotype call                     | Lab assay output → binary label                                       | Wet-lab (multiplex PCR, exons 7/10/15, 9 SNaPshot primers)                          | Ground truth for Erlangen cohorts. Detects 11 activating hotspots:`R248C, S249C, G372C, G382R, S373C, Y375C, A393E, K652E, K652Q, K652M, K652T`.                      |
| `mutations_blca_tcga_pancancer_atlas_cbioportal.txt` | Tab-separated text (`pd.read_csv(sep="\t")`)                         | Downloaded from cBioPortal`blca_tcga_pan_can_atlas_2018`                          | Ground truth for TCGA. Column`SAMPLE_ID` (first 12 chars = patient ID), column `FGFR3` (mutation string, renamed to `fgfr3_mutation` in `dataset.py`).          |
| `filtered_slides_tcga.xlsx`                          | Excel,`index_col=0`                                                  | Internal pathology review (M.E.)                                                    | Column`DX_REV_Summary` ∈ {`MIBC`, `NMIBC - pT1`, `NMIBC - pTa`}; used to keep only MIBC (or NMIBC) TCGA cases — `dataset.py:TCGADataset.get_ids_to_keep()`. |

### 2b. Preprocessing intermediate outputs (per slide)

| File                                                                      | Type                                                                   | Shape / format                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ------------------------------------------------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `features/<slidename>.svs/mask.npy` **[verified from data]**      | NumPy float32 array, continuous tissue-probability map (not binarized) | e.g.`(4832, 6912)` for one mini-dataset slide — matches the WSI's pyramid `level_dimensions["13"]`, i.e. the mask is stored at ~1.82 MPP, not at tile resolution. Ships in the HuggingFace download alongside `features.npy` but is **not read anywhere in `fgfr3mut/` public code** (`dataset.py` only globs `features.npy`). Useful for visualizing/recomputing the ≥60%-tissue tile-keep rule yourself.                                                                                                                                                                                                                                                  |
| `features/<slidename>.svs/metadata.json` **[verified from data]** | JSON                                                                   | Per-slide preprocessing record — see Step 1–2 below, all fields pulled from this file.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `features/<slidename>.svs/features.npy`                                 | NumPy float array,`mmap_mode="r"`, `astype(np.float32)`            | `(n_tiles, 1539)`, confirmed by direct inspection (`(4125, 1539) float32` for one slide). Columns `[:, :3]` = tile `(x, y, z)` coordinates; columns `[:, 3:]` = **1536-dim** feature embedding (so 1539 = 3 coord cols + 1536 feature dims — this resolves the paper's "1539" figure vs. the code's `in_features=1536`, they are the same thing, not a version mismatch). Parsed by `fgfr3mut/utils.py:load_raw_features_slide()`. Folder name is the **full original filename including `.svs`** (e.g. `TCGA-2F-A9KQ-01Z-00-DX1.1C8CB2DD-....svs/`), matching `TCGADataset._extract_slidename()` which reads the parent-folder name verbatim. |

### 2c. Training-time label vector

Produced by `fgfr3mut/dataset.py:TCGADataset.load_fgfr3_status(binarize=True)`: `pd.Series` indexed by 12-character patient ID, `label = float(mutation_string != "WT")`. Six TCGA patients with **non-activating** FGFR3 variants (`TCGA-XF-A9SL, TCGA-UY-A78N, TCGA-FJ-A3ZF, TCGA-XF-A9SJ, TCGA-4Z-AA81, TCGA-DK-A3IS`) are forced to `WT` before binarizing, since they would not respond to an FGFR inhibitor.

### 2d. Final output artifact

| File                                                                          | Type                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `models/split_{0..24}/model_{0..4}_30_ep.pt` **[verified from data]** | PyTorch`state_dict` checkpoint, one per trained Chowder model. **25 split folders × 5 checkpoints per folder = 125 checkpoints total.** Confirmed by directly counting the downloaded `models/` tree: `split_0` … `split_24` (25 dirs), each containing `model_0_30_ep.pt` … `model_4_30_ep.pt` (5 files). Loaded with `torch.load()` / `model.load_state_dict()` (`utils.py:load_ckpt`, `run_inference.py:infer()`). State-dict keys/shapes match `Chowder(in_features=1536, n_extreme=100)` exactly: `score_model.hidden_layers.0.weight (128,1536)`, `.2.weight (1,128)`, `mlp.0.0.weight (128,200)`, `mlp.1.0.weight (64,128)`, `mlp.2.weight (1,64)`. |

---

## 3. Step-by-step pipeline, scratch → final model

### Step 1 — Matter (tissue) detection: **U-Net**

- **Algorithm, as named in the paper:** U-Net (Ronneberger, Fischer & Brox, *MICCAI* 2015). Reference implementation named in paper/README: `https://github.com/milesial/Pytorch-UNet`.
- **Algorithm, as actually recorded on the downloaded features [verified from data]:** every `metadata.json` in the mini download stamps `"matter_detector": {"name": "BUNet", "dilatation": 1, "threshold": 0.5}` — i.e. the real internal component is called **BUNet**, not the public `milesial/Pytorch-UNet`. It is part of Owkin's internal `classic_algos` package (`"classic_algos_version": "1.19.0"` in the same file), which is **not open-sourced** and is not in `fgfr3mut/`. Treat "U-Net" in the paper as a description of the *architecture family*, not a pointer to the exact trained weights/repo used to produce the released features.
- Training data (paper): 460 H&E + IHC slides, manually annotated tissue masks; validated on 115 held-out slides, **Dice score = 0.96**.
- Output: a continuous tissue-probability mask per WSI, persisted as `mask.npy` at a coarse pyramid level (§2b) — thresholded at **0.5** (`"threshold": 0.5` in metadata) to decide tissue vs. background, then further filtered by the ≥60% tile-coverage rule in Step 2.

### Step 2 — Tiling

- Cut masked tissue regions into tiles of **224×224 px**, nominally at **1.0 µm-per-pixel (MPP)**.
- **[verified from data]** The *actual* per-tile MPP recorded in `metadata.json` varies slightly by slide because it snaps to the nearest available pyramid level of that slide's scan: observed values across the 6 mini-dataset slides were `0.9108`, `1.01`, `0.9824`, and `1.01` µm/px (field `tile_mpp`; pyramid level used: `"level": 15` in all 6 cases). Treat "1.0 MPP" as the *target*, not an exact per-slide guarantee.
- Keep a tile only if **≥60%** of its pixels were flagged as tissue by the matter-detector mask.
- **[verified from data]** Tile sampling is **not exhaustive-then-capped only** — `metadata.json` records `"sampling_mode": {"mode": "random", "seed": 0}` for every slide, i.e. tiles are randomly subsampled from the tissue-passing candidates using a **fixed seed of 0**. `nb_tiles` observed per slide (4125, 3440, 3206, 2496, 2635, 2759, 2495, 2548) were all below the 5000 cap, so for these particular slides the cap never bound — but the seeded-random mechanism is what "5000 tiles uniformly sampled" refers to when a slide does have more tissue tiles than the cap.
- Paper Suppl. Table 4: 1.0 MPP outperforms 0.5 MPP for this task (more spatial context per tile).
- No code for this step ships in `fgfr3mut/` — it is upstream of the released `.npy` feature files, and the tiling tool itself is proprietary (`"tiling_tool_version": "11.6.0"` in metadata).

### Step 3 — Tile feature extraction: **H-optimus-0** (paper) / **iBOTViTGiant** (actual metadata)

- **Algorithm, as named in the paper:** H-optimus-0 (Saillard et al., Bioptimus), a Vision Transformer-Giant, pretrained self-supervised with **DINOv2** (Oquab et al., arXiv:2304.07193) on H&E tiles, no labels. Public weights: `https://huggingface.co/bioptimus/H-optimus-0`.
- **⚠️ Discrepancy [verified from data]:** every `metadata.json` in the downloaded features stamps `"features_extractor": "iBOTViTGiant"` — Owkin's internal name for a ViT-Giant pretrained with the **iBOT** self-supervised framework, not "H-optimus-0" and not explicitly DINOv2. This is consistent across all 6 inspected slides, so it is not a one-off labeling error. It is plausible that `iBOTViTGiant` is an internal/pre-release codename for what was later published as H-optimus-0 (both are Owkin/Bioptimus ViT-Giant H&E foundation models), but this is **not confirmed** by any public document. If you extract features yourself from H-optimus-0 on HuggingFace, your embeddings are **not guaranteed to be numerically identical** to the ones shipped in `features.npy` — validate by comparing embeddings on a shared tile before assuming interchangeability.
- **Frozen** either way — not fine-tuned for FGFR3 prediction, used purely as a feature extractor for both training and inference.
- Not included in `fgfr3mut/` repo — external dependency, regardless of which of the two names is accurate.
- Output per tile: 1536-dim embedding vector, confirmed by direct array inspection (`features.npy` columns `3:` = 1536 floats, e.g. row 0 starts `[-0.105, -0.749, 0.078, -0.743, 1.211, ...]`). Per slide: `(n_tiles, 1536)` embeddings + `(n_tiles, 3)` coordinates → saved as `features.npy` (§2b).
- **You do not need to rerun Steps 1–3** if reproducing on TCGA: the HuggingFace dataset `PABannier/fgfr3mut` (via `download.py`) already ships precomputed `features.npy` (+ `mask.npy`, `metadata.json`) for all TCGA slides (>200 GB total). Steps 1–3 are only required if extracting features from your own raw `.svs` files, and if so you should first try to confirm which extractor name is the accurate one before trusting cross-comparability with the shipped checkpoints.

### Step 4 — Load features + labels into memory

- `fgfr3mut/dataset.py:TCGADataset.get_features(n_tiles, num_workers, features_as="list")` → calls `fgfr3mut/utils.py:load_features_to_mem()`, which parallel-loads every `features.npy` via `ThreadPoolExecutor`, truncating/capping each slide to `n_tiles` rows (paper: **max 5000 tiles per slide**, uniformly sampled, for speed/memory).
- `TCGADataset.load_fgfr3_status()` loads and binarizes labels (§2c).
- Intersect slide IDs with label IDs to build the aligned `(X, y)` training set.
- `fgfr3mut/utils.py:SlideFeaturesDataset` wraps `(features, labels)` as a `torch.utils.data.Dataset`; `fgfr3mut/utils.py:pad_collate_fn` pads variable-tile-count slides in a batch to the same length and emits a boolean padding mask `(B, N_TILES, 1)` (`True` = padded). Default batch size in the repo's inference script is 16 — no training batch size is published.

### Step 5 — Model architecture: **Chowder** (MIL)

- **Algorithm:** Chowder (Courtiol, Tramel, Sanselme & Wainrib, arXiv:1802.02212, 2018) — a weakly-supervised multiple-instance-learning (MIL) architecture for classifying a "bag" of tile instances with only one slide-level label.
- Implementation: `fgfr3mut/chowder.py:Chowder`. Exact forward pass, reading the source top to bottom:

  1. **`TilesMLP`** (`chowder.py:176-229`) scores every tile independently:
     - `MaskedLinear(in_features=1536, out_features=128, mask_value="-inf")` — a `torch.nn.Linear` that fills padded-tile outputs with `-inf` so they vanish after sigmoid (`chowder.py:MaskedLinear`, line 114-173).
     - `torch.nn.Sigmoid()` activation.
     - `torch.nn.Linear(128, out_features=1, bias=True)` — final per-tile scalar score.
     - **Note (paper vs. code discrepancy):** the paper's Methods text describes this as "an MLP with 128 hidden neurons followed by one neuron and a ReLU activation." The released code (`chowder.py`) uses `torch.nn.Sigmoid()`, not `ReLU`, as the hidden-layer activation. For exact reproduction, **follow the code** (`Sigmoid`), since it is the artifact that actually produced the published checkpoints.
  2. **`ExtremeLayer`** (`chowder.py:232-349`), instantiated as `ExtremeLayer(n_top=100, n_bottom=100)`: sorts the per-tile scores along the tile dimension and keeps the **top-100** (most FGFR3-mutant-looking) and **bottom-100** (most wild-type-looking) tiles, masking out padded tiles with ±inf before the `topk` so they are never selected unless a slide has fewer than 100 real tiles. Output: `(B, 200, 1)` slide descriptor.
  3. **`MLP`** (`chowder.py:81-111`), instantiated as `MLP(in_features=200, out_features=1)`: `Linear(200,128) → Sigmoid → Linear(128,64) → Sigmoid → Linear(64,1)` → one logit per slide.
  4. Final probability: `sigmoid(logit)` — applied outside the model, in `fgfr3mut/utils.py:sigmoid()` / `run_inference.py`.
- **Weight initialization:** `Chowder.weight_initialization` (`chowder.py:47-53`) applies `torch.nn.init.xavier_uniform_` to every `torch.nn.Linear` weight and zero-fills its bias, applied via `.apply()` to both `score_model` (`TilesMLP`) and `mlp` (`MLP`) submodules at construction time.
- **Loss:** binary cross-entropy between the sigmoid-activated slide logit and the binarized SNaPshot/cBioPortal label (paper Methods, "FGFR3 mutation prediction"). No loss computation is present in the repo (inference-only), so use `torch.nn.BCEWithLogitsLoss()` against the raw `Chowder` logit output (mathematically equivalent to sigmoid + BCE, and numerically preferable).
- **Model instantiation used at inference** (and presumably matching training): `Chowder(in_features=1536, n_extreme=100)` (`run_inference.py:175`).

### Step 6 — Deep ensemble training: **125 independent Chowder models**

- **Algorithm:** Deep Ensembles for predictive uncertainty (Lakshminarayanan, Pritzel & Blundell, NeurIPS 2017) — train many independently-initialized models and aggregate at inference for both a stronger point estimate and an uncertainty signal.
- Paper: **125 Chowder models**. **[verified from data]** The actual checkpoint layout is **25 split folders × 5 checkpoints each** (`models/split_0/` … `models/split_24/`, each holding `model_0_30_ep.pt` … `model_4_30_ep.pt`) — confirmed by counting directories/files in the downloaded `models/` tree (25 × 5 = 125). This is the reverse of a naive "5 splits × 25 models" reading of the paper's ensemble-size claim; `run_inference.py:infer()` is agnostic to the split count either way, since it just globs `weights_path.glob("split_*")` then `*.pt` inside each.
- **[verified from data]** Checkpoint filenames encode the training length: `model_{0..4}_30_ep.pt` → each model was trained for **30 epochs**. This is the one training hyperparameter directly recoverable without the training script.
- The paper does not disclose whether the 25 splits are 25-fold cross-validation folds of the 391-slide training cohort, or 25 independent random splits, nor how the 5 models within a split differ (random seed only, or also data subsampling / different tile samples). This detail is **not recoverable from the repo** — no training script ships.
- **Aggregation used at inference:** each of the 125 models produces one raw logit per slide; the code applies `sigmoid()` to every individual model's per-slide prediction, then **averages the resulting probabilities** across models (and across a patient's multiple slides) via a `groupby(patient_id).mean()` (`run_inference.py:compute_metrics()`). Note the paper's Methods text says "predictions of each model were aggregated by averaging all the output logits" (logit-space averaging) — this **also disagrees with the shipped code**, which averages post-sigmoid probabilities. Follow the code (`sigmoid` → `mean`) to match the repo's actual reproducible numbers; if attempting to match the paper's stated method exactly, average the raw logits per patient/slide instead, then apply a single `sigmoid` at the end.

---

## 4. Output of the training pipeline

- 125 trained `Chowder(in_features=1536, n_extreme=100)` checkpoints, one `state_dict` `.pt` file each, arranged in `models/split_0/` … `models/split_24/` (5 files per folder, each trained for 30 epochs — **[verified from data]**, see §2d and Step 6).
- This ensemble **is** the final model. There is no further distillation/calibration step baked into the repo; the operating threshold used at deployment time is chosen post hoc on validation data (see `docs/fgfr3mut_pipeline/inference.md` §Threshold selection), not during training.

---

## 5. Algorithm reference table (name → paper citation → code location)

| Stage                                | Algorithm                                                                                                                | Paper reference                                                             | Code location                                                                                                                                                  |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tissue detection                     | U-Net (paper) /**BUNet** (actual metadata field — [verified from data])                                           | Ronneberger et al., MICCAI 2015 (ref. 24)                                   | External —`github.com/milesial/Pytorch-UNet` referenced by paper; actual extractor ships inside Owkin's private `classic_algos` package, not in this repo |
| Tile feature extraction              | H-optimus-0 (paper) /**iBOTViTGiant** (actual metadata field — [verified from data], see Step 3 discrepancy note) | Saillard et al. (ref. 25); Oquab et al., DINOv2, arXiv:2304.07193 (ref. 26) | External —`huggingface.co/bioptimus/H-optimus-0` referenced by paper (not in this repo)                                                                     |
| Per-tile scoring + slide aggregation | Chowder (MIL)                                                                                                            | Courtiol, Tramel, Sanselme & Wainrib, arXiv:1802.02212 (ref. 27)            | `fgfr3mut/chowder.py` (`TilesMLP`, `ExtremeLayer`, `MLP`, `Chowder`)                                                                                 |
| Multi-model aggregation              | Deep Ensembles                                                                                                           | Lakshminarayanan, Pritzel & Blundell, NeurIPS 2017 (ref. 28)                | `fgfr3mut/run_inference.py:infer()` (125-checkpoint loop)                                                                                                    |
| Loss                                 | Binary cross-entropy                                                                                                     | Methods, "FGFR3 mutation prediction"                                        | Not shipped; use`torch.nn.BCEWithLogitsLoss()` on `Chowder`'s raw logit output                                                                             |
| Batching/masking                     | Padded-sequence MIL batching                                                                                             | —                                                                          | `fgfr3mut/utils.py:pad_collate_fn`, `SlideFeaturesDataset`                                                                                                 |

---

## 6. Undisclosed hyperparameters — decide before retraining

Neither the paper nor the `fgfr3mut` repo states: optimizer (Adam/AdamW/SGD), learning rate / schedule, weight decay, early-stopping criterion, train/val split inside the 391-slide discovery cohort, or the exact procedure that produced 25 splits × 5 models (fold definition vs. random seed only). **Number of epochs is no longer undisclosed: 30**, recovered directly from the `model_{i}_30_ep.pt` checkpoint filenames (§2d, Step 6) — **[verified from data]**. §7 below narrows the rest further using a second, independent Owkin repo. If retraining from scratch for experimental reproduction:

- Treat what's left as free hyperparameters to tune against a held-out slice of your own training cohort.
- Document whatever choices you make in your own experiment log, since they are not attributable to Bannier et al.

---

## 7. A real Chowder training loop exists — `owkin/HistoSSLscaling` **[verified from data, 2026-09-25]**

`https://github.com/owkin/HistoSSLscaling` is the companion code for a different Owkin paper (Filiot et al., *Scaling Self-Supervised Learning for Histopathology with Masked Image Modeling*, MedRxiv 2023 — the paper behind the public **Phikon** feature extractor). It is **not** `fgfr3mut`-specific and ships no FGFR3 data or checkpoints, but it contains two things directly useful for closing gaps in §6, confirmed by cloning and reading the source directly:

### 7a. It has the same Chowder implementation — and independently confirms the paper-vs-code discrepancy

`rl_benchmarks/models/slide_models/chowder.py` is essentially the same architecture as `fgfr3mut/chowder.py` (`TilesMLP`, `ExtremeLayer`, `MLP`, Xavier-uniform init, same docstring citing Courtiol et al.), generalized with configurable `n_top`/`n_bottom` and hidden-layer sizes. Its default config (`conf/slide_level_task/cross_validation/model/chowder.yaml`) sets:

```yaml
mlp_activation:
  _target_: torch.nn.Sigmoid
```

This is a **second, independent Owkin codebase** using `Sigmoid` (not `ReLU`) as the Chowder MLP activation — corroborating the §Step 5 conclusion that the paper's "ReLU activation" phrase is imprecise and the actual family of Owkin Chowder implementations uses `Sigmoid`.

### 7b. It has an actual working MIL training loop — `TorchTrainer`

`rl_benchmarks/trainers/torch_trainer.py:TorchTrainer.train()` is a complete, runnable train/val loop for exactly this class of model (Chowder or other slide-MIL aggregators on precomputed tile features): standard PyTorch loop, `DataLoader(shuffle=True, drop_last=True)` for train / `drop_last=False` for val, forward/backward per epoch, metrics computed every epoch, no early stopping (runs the full fixed `num_epochs`). Defaults and an example config (`conf/slide_level_task/cross_validation/test_ncv.yaml`, for a different task — TCGA-COAD overall survival):

| Hyperparameter | `TorchTrainer` class default                                                                           | Example task config value                                                                                                                                                                                                                             |
| -------------- | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Optimizer      | `torch.optim.Adam`                                                                                     | `torch.optim.Adam`                                                                                                                                                                                                                                  |
| Learning rate  | `1.0e-3`                                                                                               | `2.0e-4`                                                                                                                                                                                                                                            |
| Weight decay   | `0.0`                                                                                                  | `0.0`                                                                                                                                                                                                                                               |
| Batch size     | `16`                                                                                                   | `16`                                                                                                                                                                                                                                                |
| Epochs         | `10`                                                                                                   | `5`                                                                                                                                                                                                                                                 |
| Loss           | — (passed in)                                                                                           | `CoxLoss` for survival; `rl_benchmarks/losses/bce_with_logits_loss.py:BCEWithLogitsLoss` (thin wrapper on `torch.nn.BCEWithLogitsLoss`) exists in the same repo for classification tasks — this is the exact loss type to use for FGFR3 MUT/WT |
| Splitting      | `rl_benchmarks.val_schemes.NestedCrossValidation`, `split_mode: patient_split`, `stratified: True` | `n_splits_outer=2, n_repeats_outer=1, n_splits_inner=2, n_repeats_inner=1` in the example (task-specific, not fgfr3mut's 25/5)                                                                                                                      |

**None of these numeric values are confirmed to be what produced the 125 FGFR3MUT checkpoints** (`batch_size=16` matches `fgfr3mut/run_inference.py`'s inference default, which is suggestive but not proof). What this *does* establish: the Adam-optimizer / patient-level-stratified-nested-CV / BCEWithLogitsLoss recipe isn't a guess extrapolated from the unrelated 2018 Chowder paper — it's Owkin's own standard training harness for this exact model family, reusable as-is. To retrain FGFR3MUT from scratch with a real loop instead of writing one from zero: adapt `TorchTrainer` + `Chowder` + `BCEWithLogitsLoss` from this repo, plug in a `Dataset` wrapping your own `features.npy`/label files (same shapes as `fgfr3mut/utils.py:SlideFeaturesDataset`), and set `num_epochs=30` (§6) and `n_top=n_bottom=100` (§Step 5) to match the known FGFR3MUT-specific parameters.

**License note:** `HistoSSLscaling` ships under Owkin's own non-commercial research license (`LICENSE.txt`, "limited to grant non-profit entities with the rights defined hereunder") — consistent with `fgfr3mut`'s CC-BY-NC-ND license (§ "Code availability" in the paper). Fine for the stated experimental/research reproduction goal; re-check the license text before any commercial use.

### 7c. It confirms `iBOTViTGiant` is real — but only the *Base* variant is public

`rl_benchmarks/models/feature_extractors/ibot_vit.py:iBOTViT` implements exactly the `iBOTViT<Size><Dataset>` naming pattern seen in the FGFR3MUT feature `metadata.json` (§Step 3) — but its `architecture` argument only accepts `"vit_base_pancan"`, and `constants.py:MODEL_WEIGHTS` ships only one entry: `"iBOTViTBasePANCAN": ibot_vit_base_pancan.pth` (this is the public **Phikon** model, `huggingface.co/owkin/phikon`, ViT-**Base**, 224px tiles per `TILE_SIZES["iBOTViTBasePANCAN"] = 224`). No "Giant" variant, weights, or config exists anywhere in this repo (`grep -rn giant` returns nothing) or in `fgfr3mut/`. So: **`iBOTViTGiant` is a real, systematically-named member of Owkin's internal model family (iBOT self-supervised ViT, PanCancer-trained), scaled up from the publicly-released Phikon (`iBOTViTBasePANCAN`) — but the Giant-scale checkpoint itself was never open-sourced**, under that name or (as far as either repo shows) as H-optimus-0 either. Treat the extractor used for the shipped `features.npy` as **unreproducible from public weights**; the closest public stand-ins, in descending order of likely similarity, are H-optimus-0 (same "Giant" scale, DINOv2 not iBOT, Bioptimus/Owkin lineage) and Phikon (`iBOTViTBasePANCAN`, same iBOT training method, much smaller scale, fully public and directly runnable via the snippet in `HistoSSLscaling/README.md`).

---

*Sources: paper Methods pp.8–9 ("Preprocessing of whole-slide images", "FGFR3 mutation prediction"), Fig. 1a, References 24–28; `fgfr3mut/chowder.py`, `fgfr3mut/dataset.py`, `fgfr3mut/utils.py`, `fgfr3mut/run_inference.py`, `README.md` as present in this repo on 2026-09-25; downloaded HuggingFace artifact `output/data_fgfr3_mini/` (features.npy, mask.npy, metadata.json, model checkpoints, filtered_slides_tcga.xlsx, mutations_blca_tcga_pancancer_atlas_cbioportal.txt) inspected directly on 2026-09-25; `github.com/owkin/HistoSSLscaling` (chowder.py, torch_trainer.py, ibot_vit.py, constants.py, test_ncv.yaml, README.md, LICENSE.txt) cloned and inspected directly on 2026-09-25.*

# Extension goal: mutation-subtype + WHO-grade classification (proposed — not yet implemented)

This section is a **plan**, not a verified reproduction like §1–7 above. Target: extend the existing binary FGFR3 MUT/WT Chowder pipeline into a system that also predicts (a) **which** FGFR3 hotspot is present, and (b) the tumor's **WHO grade** (1/2/3, or the WHO'04 low/high-grade scheme TCGA actually uses) — both directly from the WSI, reusing the tiling + feature-extraction stages already built in this repo.

**Ground-truth constraint that shapes everything below:** FGFR3 status/hotspot comes from bulk SNaPshot PCR / cBioPortal calls — **one label per slide/sample**, never per-cell or per-pixel (§2a). No public dataset gives per-nucleus genotype ground truth for this task. Every paper cited below (including the one this repo already reproduces) trains and validates at the **slide level** with **weak supervision** (MIL) and treats the resulting attention map as a correlational visualization, not a verified per-cell call. WHO grade labels are also recorded per sample (TCGA clinical fields, already parsed into this repo's `data/*/cbioportal_*_sample_clinical.json`, surfaced in `viewer.html`'s "Grade" field) — but unlike genotype, grade *is* a directly visible histomorphological property (nuclear pleomorphism, architecture, mitoses), so it is the better-grounded of the two targets.

> **Correction (2026-09-26):** an earlier draft of this section read "which mutant" as **which hotspot/protein-change** (e.g. `Y373C` vs `S249C`). That is wrong. Per the user's clarification (screenshot of `viewer.html`'s FGFR3 card), "mutation type" means the **MAF-style variant-consequence class** — the `mutationType`/`variantType` cBioPortal fields (`Missense_Mutation`, `Nonsense_Mutation`, `Frame_Shift_Del`, …; `SNP`, `DEL`, `INS`, …), **not** the specific hotspot/protein-change identity. §8a/§8c/§9 below are corrected to target this field. The hotspot/protein-change identity (`Y373C`, `S249C`, …) is a *different, harder* target — see the feasibility note in §8c, Stage 2b.

### 8a. Which paper covers which step

| Pipeline step                                                               | Paper to follow                                                                                                                                        | Why this one                                                                                                                                                                                                                                                                                               |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tissue detection + tiling                                                   | *(unchanged — already implemented, §Step 1–2 above)*                                                                                              | No new paper needed;`fgfr3mut_tile_wsi.py` already reproduces this stage.                                                                                                                                                                                                                                |
| Frozen tile-embedding backbone                                              | Saillard et al. (H-optimus-0, ref. 25) — already used; benchmark against UNI/CONCH per the foundation-model MIL survey¹ if H-optimus-0 underperforms | Survey¹ shows UNI/CONCH tile embeddings measurably improve downstream MIL accuracy over smaller backbones for grading/biomarker tasks — worth an ablation, not a mandatory swap.                                                                                                                         |
| Binary FGFR3 MUT/WT (existing task)                                         | Bannier et al. 2024² (already reproduced, §1–7)                                                                                                     | No change — this is the working baseline.                                                                                                                                                                                                                                                                 |
| **FGFR3 mutation TYPE** (new — corrected target, see note above)     | Hierarchical Deep MIL,*Medical Image Analysis*³                                                                                                     | Same clinical setting (bladder WSI) and same core problem — predicting**multiple distinct mutation classes** from one MIL model instead of one binary model. Directly transferable: replace "5 genes" with "N MAF `mutationType` classes." Class set + feasibility numbers are in §8c, Stage 2b. |
| **WHO grade** (new)                                                   | NMGrad⁴ (primary) + "A Novel Self-Learning Framework for Bladder Cancer Grading"⁵ + multi-scale pyramidal CNN⁶                                      | See §8b for what each contributes specifically.                                                                                                                                                                                                                                                           |
| Joint mutation+grade model (optional, if you want one model instead of two) | PA-MIL⁷                                                                                                                                               | Purpose-built for coupling a genotype task with a phenotype/morphology task in one MIL model with a shared attention backbone — exactly this repo's mutation+grade combination.                                                                                                                           |
| Multi-branch, per-class attention/interpretability                          | CLAM⁸                                                                                                                                                 | Gives one attention branch*per output class* (per hotspot, per grade) out of the box — a natural drop-in replacement for Chowder's single-branch scorer when the output is no longer binary.                                                                                                            |

### 8b. What NMGrad actually does, concretely (since "weakly-supervised MIL for grading" alone under-specifies it)

NMGrad targets non-muscle-invasive bladder cancer (NMIBC) WHO'04 grading and adds two things beyond a plain Chowder-style model:

1. **Tissue-type filtering before grading tiles are chosen.** It first segments the WSI into urothelium vs. other tissue (stroma, muscle, blood), and *only* urothelium tiles are fed to the grading model — grade is a property of the urothelial lining, and diluting the bag with stromal tiles hurts the attention mechanism's ability to find genuinely diagnostic regions. This repo has no urothelium-specific segmenter yet (see §8c caveat).
2. **Nested/hierarchical attention aggregation**, not a single flat tile→slide pooling step: tiles are first grouped into *location-dependent regions*, region-level attention scores are computed, and regions are aggregated into the final slide-level WHO'04 grade — an extra hierarchy level between Chowder's flat `TilesMLP → ExtremeLayer → MLP`.
3. Its attention scores were checked against pathologist-verified high-grade regions and correlated with them — i.e. the same "attention heatmap as interpretability, not ground truth" caveat that applies to this repo's existing FGFR3 heatmap (`reproduce/heatmap.py`) applies here too, but NMGrad is one of the few papers that actually validated the correlation rather than just asserting it.

### 8c. Proposed end-to-end pipeline

```
WSI
 │
 ├─ Stage 0 (existing, unchanged): tissue tiling
 │    fgfr3mut_tile_wsi.py → 224×224 px tiles @ ~1.0 MPP, ≥60% tissue (Bannier et al. 2024, §1–7 above)
 │
 ├─ Stage 0.5 (NEW, gap — see caveat below): urothelium-vs-other tissue-type filter
 │    Only needed for the grade head (NMGrad §8b point 1). No public urothelium-specific
 │    segmenter ships anywhere in this repo or cited papers; closest available stand-in is
 │    the existing per-tile nuclei/cytoplasm foreground mask (tile_masks.npy, this
 │    session's earlier fix) as a weak proxy for "cellular tissue" vs. stroma/muscle —
 │    NOT validated as urothelium-specific. Treat as an open research gap, not solved.
 │
 ├─ Stage 1 (existing, unchanged): frozen foundation-model tile embeddings
 │    H-optimus-0 (Saillard et al.²) → 1536-dim/tile, exactly as in fgfr3mut/chowder.py.
 │    Optional ablation: swap in UNI or CONCH per survey¹ if accuracy needs a boost.
 │
 ├─ Stage 2a (existing, unchanged): FGFR3 MUT/WT — Chowder MIL, binary, Bannier et al.²
 │
 ├─ Stage 2b (NEW): FGFR3 mutation TYPE — Chowder/CLAM MIL, softmax, N classes
 │    Target = the MAF `mutationType`/`variantType` field (Missense_Mutation, Nonsense_Mutation,
 │    Frame_Shift_Del, …; SNP, DEL, INS, …) — i.e. exactly the "Mutation type"/"Variant type"
 │    fields boxed in the user's viewer.html screenshot — NOT the hotspot/protein-change
 │    identity (Y373C, S249C, …; that's a separate field, "Hotspot / Protein").
 │
 │    [verified from data, 2026-09-26] Directly counted from this repo's own
 │    data/cbioportal_blca_tcga_pan_can_atlas_2018_FGFR3_mutations_cohort.json (68 FGFR3-mutated
 │    TCGA samples, the same cohort behind this repo's labels):
 │      mutationType: Missense_Mutation=65, Frame_Shift_Del=2, Nonsense_Mutation=1
 │      variantType:  SNP=66, DEL=2
 │    i.e. 95%+ of real samples are the single class Missense_Mutation/SNP. A balanced
 │    3-class (or even 2-class) classifier is NOT feasible on TCGA alone — only 3 total
 │    non-missense examples exist in the whole public cohort. Two honest paths forward:
 │      (a) reframe as binary "canonical missense/SNP" vs. "disruptive/other" (nonsense +
 │          frameshift + indel + splice pooled into one rare class) — still severely
 │          imbalanced (3 vs 65) but at least a defined 2-class problem; or
 │      (b) pool in the paper's private 391-slide Erlangen discovery cohort (§1) and/or
 │          additional cBioPortal bladder studies beyond blca_tcga_pan_can_atlas_2018 to
 │          collect enough non-missense examples before attempting real multi-class training.
 │    Hotspot/protein-change identity (Y373C, S249C, …) is a separate, harder target if
 │    ever wanted later: the same 68-sample cohort has 21 unique protein-change values,
 │    most singletons (S249C alone is 32/68 = 47%) — effectively infeasible as supervised
 │    multi-class without a much larger cohort.
 │
 ├─ Stage 2c (NEW): WHO grade — nested-attention MIL, softmax, grade classes
 │    Follow NMGrad⁴'s nested tile→region→slide aggregation (§8b point 2). Labels already
 │    available per-slide in this repo's data/*/cbioportal_*_sample_clinical.json.
 │    Optional accuracy add-on: fuse PanNuke-derived per-tile nuclear morphometrics
 │    (pleomorphism, mitotic density) into the tile embedding before pooling, per the
 │    multi-scale pyramidal CNN paper⁶'s approach of adding finer-scale nuclear detail.
 │
 └─ Stage 3 (optional): joint mutation+grade model
      PA-MIL⁷-style shared attention backbone with two task-specific heads (2b + 2c),
      trained jointly instead of as two independent models — more data-efficient given
      how few MUT slides exist, at the cost of extra implementation complexity. Start
      with 2b and 2c as two separate models; only merge into one PA-MIL-style model if
      the separate versions show they'd benefit from shared representations.
```

**Validation**: same k-fold, patient-stratified scheme this repo already documents in §1–7 (`TorchTrainer` + `NestedCrossValidation`, §7b) — report per-class AUROC/F1, and (per NMGrad's own validation approach) sanity-check attention heatmaps against any available pathologist-marked high-grade regions before trusting them as more than a visualization.

### 8d. References

1. Foundation-model + MIL benchmarking survey — multi-cancer comparison of tile-embedding backbones (CTransPath, PathoDuet, PLIP, CONCH, UNI) across MIL methods for grading/biomarker/genotype tasks. "When multiple instance learning meets foundation models: Advancing histological whole slide image analysis," *Medical Image Analysis* (ScienceDirect).
2. Bannier et al., *AI allows pre-screening of FGFR3 mutational status using routine histology slides of muscle-invasive bladder cancer*, **Nature Communications** 15:10914 (2024), doi:10.1038/s41467-024-55331-6 — the paper already reproduced in §1–7.
3. "Histopathological bladder cancer gene mutation prediction with hierarchical deep multiple-instance learning," *Medical Image Analysis* (ScienceDirect) — multi-gene (ATM, PIK3CA, ERBB2, FGFR3, ERCC2) MIL from bladder WSIs; template for the multi-class mutation-TYPE head (Stage 2b).
4. "NMGrad: Advancing Histopathological Bladder Cancer Grading with Weakly Supervised Deep Learning," arXiv:2405.15275 / *PMC11428615* — nested urothelium-filtered, region-hierarchical attention-MIL for WHO'04 NMIBC grading; template for Stage 0.5 + Stage 2c.
5. "A Novel Self-Learning Framework for Bladder Cancer Grading Using Histopathological Images," arXiv:2106.13559 — self-training with pseudo-labels for grading under limited annotation; relevant if the labeled-grade cohort turns out too small for direct supervision.
6. "Precise grading of non-muscle invasive bladder cancer with multi-scale pyramidal CNN," *Scientific Reports* (2024), doi:10.1038/s41598-024-77101-6 — multi-scale/finer-detail fusion for grading; template for the optional PanNuke nuclear-morphometrics add-on in Stage 2c.
7. "PA-MIL: Phenotype-Aware Multiple Instance Learning Guided by Language Prompting and Genotype-to-Phenotype Relationships," arXiv:2602.02558 — joint genotype+phenotype MIL architecture; template for the optional Stage 3 joint model.
8. Lu, Williamson, Chen, Chen, Barbieri & Mahmood, *Data Efficient and Weakly Supervised Computational Pathology on Whole Slide Images* (CLAM), arXiv:2004.09666, **Nature Biomedical Engineering** (2021) — multi-branch attention-MIL architecture; usable as a drop-in multi-class/multi-branch alternative to Chowder for Stages 2b/2c.

*This section written 2026-09-26, based on a literature search conducted the same day; none of steps 8c has been implemented or validated in this repo yet.*

---

## 9. Is this actually a novel idea?

**Verdict: partially.** Each individual piece already has a published precedent — the *specific combination* proposed in §8 does not, as far as a targeted search on 2026-09-26 could find.

**Not novel (already solved, cited above):**

- Binary FGFR3 MUT/WT prediction from bladder WSI — Bannier et al.² (this repo's baseline), Loeffler et al.¹¹, Woerl et al.¹².
- WHO-grade prediction from bladder WSI via MIL — NMGrad⁴, self-learning framework⁵, multi-scale pyramidal CNN⁶.
- Multi-class *specific-mutation-subtype* classification from WSI as a general technique — demonstrated for **other** genes/cancers: papillary thyroid carcinoma driver-mutation subtyping (BRAF/RET/etc.)¹³, EGFR exon-level subtype classification in lung adenocarcinoma¹⁴.

**Appears genuinely novel (no matching published pipeline found in this search) — corrected 2026-09-26 to the actual target (MAF `mutationType`, not hotspot identity, see §8 correction note):**

1. Classifying **FGFR3 mutation TYPE** (`Missense_Mutation` / `Nonsense_Mutation` / `Frame_Shift_Del` / etc. — the variant-consequence class, not which gene or which hotspot) directly from bladder-cancer WSI. A dedicated search for this exact framing ("mutation type" as a classification *target*, distinct from "which gene is mutated") found no matching paper for **any** cancer type, let alone bladder/FGFR3 — existing WSI-genomics literature (Bannier/Loeffler/Woerl for FGFR3; the NSCLC¹⁵, breast¹⁶, and AML¹⁷ mutation-prediction papers) all predict **presence of a mutation in a given gene**, not its MAF consequence class. The closest adjacent precedent is FLT3-ITD detection in AML¹⁷ (predicting one specific structural-variant *type*, an internal tandem duplication/insertion, vs. wild-type) — conceptually closer to "variant type" classification than any bladder/FGFR3 paper, but still a different gene, cancer, and framing.
2. A **single pipeline predicting both FGFR3 mutation type and WHO grade** from the same WSI tiling/embedding backbone. No cited paper does both — NMGrad⁴ only grades, Bannier/Loeffler/Woerl only call FGFR3 binary status. Nothing found couples a genotype-consequence-class head with a grade head for bladder cancer.

**Feasibility caveat that qualifies the novelty claim:** §8c, Stage 2b shows this repo's own TCGA cohort has only 3 non-missense examples out of 68 FGFR3-mutant samples. Part of *why* this specific classification appears unpublished may simply be that it is barely trainable on public data as a balanced multi-class problem — "unpublished" here is not strong evidence the idea is good, just that the exact framing doesn't already exist.

**Proof standard, honestly stated:** this is a live web-search check (§ conducted 2026-09-26), not a systematic literature review (PubMed/arXiv/Google Scholar exhaustive query + patent search). It is evidence the *specific* combination in §8 is currently unpublished, not proof no one is doing it privately or that it would work — the ground-truth-granularity caveat in §8 (bulk, slide-level labels only) still applies, and going from "unpublished" to "clinically validated" is the actual hard part.

11. Loeffler et al., *Artificial Intelligence–based Detection of FGFR3 Mutational Status Directly from Routine Histology in Bladder Cancer*, ScienceDirect / *Eur Urol Focus* (2021).
12. Woerl et al., *Using deep learning to identify bladder cancers with FGFR-activating mutations from histology images*, *Cancer Medicine* (2021), PMC8290253.
13. *Classifying driver mutations of papillary thyroid carcinoma on whole slide image: an automated workflow applying deep convolutional neural network*, PMC11573888. (Multi-class *which-gene* precedent, not mutation-type-class — kept for context.)
14. *Predicting EGFR Mutation in LUAD from Histopathological Whole-Slide Images Using Pretrained Foundation Model and Transfer Learning: An Indian Cohort Study*, arXiv:2508.01352. (Same caveat as 13.)
15. Coudray et al., *Classification and mutation prediction from non–small cell lung cancer histopathology images using deep learning*, *Nature Medicine* (2018) — predicts gene-mutation presence, not MAF consequence class.
16. *Genetic mutation and biological pathway prediction based on whole slide images in breast carcinoma using deep learning*, PMC8460699 — same, breast cancer.
17. *Annotation-free deep learning for predicting gene mutations from whole slide images of acute myeloid leukemia*, *npj Precision Oncology* (2025) — includes FLT3-ITD (an insertion/duplication *type*) prediction; closest adjacent precedent for "variant type" as the target, but different gene/cancer/framing.
