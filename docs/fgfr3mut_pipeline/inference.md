# FGFR3MUT — Inference Pipeline (Exact Reproduction Spec)

**Source paper:** Bannier et al., *Nature Communications* 15:10914 (2024), doi:10.1038/s41467-024-55331-6 — Results, Methods "Performance assessment and statistical methods", Table 1, Fig. 2a–c.
**Source code:** `fgfr3mut/run_inference.py`, `fgfr3mut/bootstrap.py`, `fgfr3mut/dataset.py`, `fgfr3mut/utils.py`, `fgfr3mut/chowder.py`.
**Prerequisite:** the trained-model artifact from `docs/fgfr3mut_pipeline/training.md` — 125 `Chowder` checkpoints under `models/split_{0..24}/model_{0..4}_30_ep.pt` (this repo provides these pretrained; you don't have to run training yourself to run inference).

> **Verified against the actual downloaded data** (`output/data_fgfr3_mini/`, a small HuggingFace subset pulled 2026-09-25): confirms the file layout below, including that `models/` has **25 split folders × 5 checkpoints each** (not 5×25), and that each `features/<slidename>.svs/` folder also ships a `mask.npy` and `metadata.json` that `run_inference.py` never reads.

---

## 1. Exact input files

| File | Type | Role |
|---|---|---|
| `<data_dir>/features/<slidename>.svs/features.npy` | NumPy float array `(n_tiles, 1539)` = 3 coord cols + 1536-dim embedding (metadata calls the extractor `iBOTViTGiant`; paper calls it H-optimus-0 — see training.md §Step 3) | Per-slide tile features (see training.md §2b for how these are produced). Folder name includes the `.svs` extension. |
| `<data_dir>/models/split_{0..24}/model_{0..4}_30_ep.pt` | PyTorch `state_dict` checkpoints, **25 folders × 5 files = 125 total** | Trained Chowder ensemble (each model trained 30 epochs, per filename) |
| `<data_dir>/filtered_slides_tcga.xlsx` | Excel, `index_col=0`, column `DX_REV_Summary` | MIBC/NMIBC case filter for TCGA |
| `<data_dir>/mutations_blca_tcga_pancancer_atlas_cbioportal.txt` | Tab-separated text | Ground-truth FGFR3 labels (TCGA, from cBioPortal) |
| `<data_dir>/loeffler_tcga.xlsx` | Excel, `index_col=1` | Optional: restricts to the exact case set used by Loeffler et al. 2022, for head-to-head AUC comparison |

All of the above are fetched in one shot by `python download.py --out_dir <data_dir>` (HuggingFace dataset `PABannier/fgfr3mut`).

---

## 2. Step-by-step pipeline

### Step 1 — Load features + build slide list

`fgfr3mut/run_inference.py:get_predictions_for_mpp()`:
- `TCGADataset(data_dir)` globs all `features/**/features.npy`.
- `dataset.get_features(n_tiles=parser_args.n_tiles, num_workers=8, features_as="list")` → calls `utils.load_features_to_mem()`, truncating every slide to the first `n_tiles` rows (paper: **5000**), returns `X` (list of `(n_tiles_i, 1536)` arrays), `X_slidenames`, `X_ids` (12-char patient IDs, `slidename[:12]`).

### Step 2 — Load ground-truth labels

`TCGADataset.load_fgfr3_status(binarize=True, keep_cases=parser_args.keep_tcga_cases, loeffler_cases=parser_args.loeffler_tcga_cases)` (see training.md §2c for the exact label-derivation logic, including the 6 forced-WT patients).

### Step 3 — Align features ↔ labels

`get_predictions_for_mpp()` intersects `X_ids` with `labels.index`; drops any slide/patient without a match on either side. Prints `n_slides=..., n_patients=...` — this is the number you check against the paper's reported cohort sizes (e.g. TCGA MIBC: `n_slides=379, n_patients=308`).

### Step 4 — Instantiate the model

```python
model = Chowder(in_features=1536, n_extreme=100)
```
(`run_inference.py:175`) — must match the architecture in training.md §Step 5 exactly, since weights are loaded by `state_dict`.

### Step 5 — Ensemble forward pass

`fgfr3mut/run_inference.py:infer()`:
1. Wrap `(X, dummy_labels=-1.0, metadata={slidename, patient_id})` in `SlideFeaturesDataset`, batch with `DataLoader(batch_size=16, shuffle=False, collate_fn=pad_collate_fn)` — `pad_collate_fn` pads each batch's slides to the batch's max tile count and emits the boolean padding mask.
2. Discover all 125 checkpoints: loop `split_0 … split_{k-1}` (k=25 in the actual download), glob `*.pt` inside each.
3. For every batch, loop over **all 125 checkpoints**: `model.load_state_dict(load_ckpt(path, device))`, `model.eval()`, then under `torch.inference_mode()`: `preds = model(features, mask=mask)[0]` — the raw logit output (index `[0]` of `Chowder.forward`, since index `[1]` is the extreme-tile scores, unused here).
4. Record one row per `(slidename, patient_id, raw_logit, model_filename)` — i.e. **125 rows per slide**, one per ensemble member.
5. After the full loop: `all_preds["pred"] = all_preds["pred"].apply(sigmoid)` — converts every model's raw logit to a probability (`utils.py:sigmoid`, plain `1/(1+exp(-x))`).

### Step 6 — Aggregate to one prediction per patient

`fgfr3mut/run_inference.py:compute_metrics()`:
- Attach the ground-truth `label` to each row via `patient_id`.
- `set_index("patient_id")`, then `groupby(index).mean(numeric_only=True)` on the `pred` column — this averages the sigmoid-activated probability **across all 125 ensemble members and across all of a patient's slides simultaneously**. (A `median` aggregate is also computed but not used for the headline metrics.)
- Drop patients with no label.
- This mean-probability value is the model's final **P(FGFR3-mutant)** for that patient.

### Step 7 — Metrics + bootstrap confidence intervals

`fgfr3mut/bootstrap.py:compute_ci_with_bootstrap(y_true, y_pred, n_repeats=1000)`:
- `sklearn.metrics.roc_auc_score` → AUC; `average_precision_score` → AP.
- Threshold selection: `roc_curve()`, then pick the **lowest threshold achieving sensitivity ≥ `target_sens` (default 0.92)** — `idx_target_se = np.where(tpr_ >= target_sens)[0][0]`. This is why the paper's operating thresholds are very low (0.051–0.063): the model is tuned for near-miss-proof sensitivity, not accuracy.
- Binarize predictions at that threshold, compute confusion matrix → Sensitivity, Specificity, PPV, NPV.
- Repeat the whole computation **1000×** on patient-level bootstrap resamples (`np.random.choice(range(n), size=n)`, with replacement) to get 5th/95th-percentile confidence intervals (`conf_level=0.95`).

### Step 8 — Save predictions

`run_inference.py:main()` writes `./predictions/preds_{suffix}.csv`, columns: `slidename, patient_id, pred, model_filename, label` (`pred` = per-patient sigmoid-averaged ensemble probability after Step 6 has been merged back onto the per-slide rows — the CSV as saved still carries the original per-slide/per-model granularity from Step 5 plus the `label` column attached in `compute_metrics()`; the headline metrics printed to stdout are the patient-level aggregate from Step 6/7). Suffix is `_keep_{cases}_cases` or `_loeffler_cases` depending on CLI flags.

---

## 3. Exact command (reproduces paper Table 1 / Fig. 2a)

```bash
# TCGA MIBC only — paper: AUC=0.82 [0.75-0.88], Se=0.95, Sp=0.47, PPV=0.14, NPV=0.99
python fgfr3mut/run_inference.py \
  --data_dir ./data_fgfr3 \
  --device cpu \
  --n_tiles 5000 \
  --keep_tcga_cases MIBC
```

Full CLI (`run_inference.py` argparse):

| Flag | Default | Meaning |
|---|---|---|
| `--data_dir` | required | Root containing `features/`, `models/`, `*.xlsx`, `*.txt` |
| `--weights_dir` | `<data_dir>/models` | Override checkpoint location |
| `--device` | `cuda:0` | `cpu` or `cuda:0` |
| `--n_tiles` | `None` (all) | Cap tiles/slide loaded (paper: 5000) |
| `--keep_tcga_cases` | `["MIBC"]` | `MIBC`, `NMIBC`, or both |
| `--loeffler_tcga_cases` | `False` | Restrict to Loeffler et al. case set; overrides `--keep_tcga_cases` |
| `--batch_size` | `16` | DataLoader batch size |

---

## 4. Algorithm reference table

| Stage | Algorithm | Code location |
|---|---|---|
| Ensemble forward pass | Chowder MIL model × 125 (Deep Ensemble) | `fgfr3mut/run_inference.py:infer()`, `fgfr3mut/chowder.py:Chowder` |
| Patient-level aggregation | Probability-space mean over slides + models | `run_inference.py:compute_metrics()` |
| Operating threshold | ROC-curve threshold at target sensitivity (default 0.92) | `fgfr3mut/bootstrap.py:_compute_metrics()` |
| Confidence intervals | Non-parametric bootstrap, patient-level resampling, 1000 repeats | `fgfr3mut/bootstrap.py:compute_ci_with_bootstrap()` |
| Inter-slide agreement (paper only, not in repo) | ICC(2,k), Pingouin | Methods, "Performance assessment" |
| Uncertainty / conformal sets (paper only, not in repo) | Deep-ensemble variance; split conformal prediction | Methods, "Uncertainty quantification and conformal prediction"; refs. 29–31 |

---

*Sources: `fgfr3mut/run_inference.py`, `fgfr3mut/bootstrap.py`, `fgfr3mut/dataset.py`, `fgfr3mut/utils.py`, `README.md`, paper Methods pp.9, Table 1, Fig. 2a–c, as present in this repo on 2026-09-25; downloaded HuggingFace artifact `output/data_fgfr3_mini/` inspected directly on 2026-09-25.*
