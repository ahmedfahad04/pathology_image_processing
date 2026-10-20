# FGFR3MUT Training Code — Short Audit

**Scope:** `scripts/fgfr3mut_training/train_fgfr3mut.py`, the vendored
`scripts/fgfr3mut_training/fgfr3mut/` package, and
`scripts/preprocessing/fgfr3mut_tile_wsi.py`, checked against Bannier et al.,
*Nat Commun* 15:10914 (2024), Methods pp. 8–9.

## Final result

| Question                  | Verdict                                                                                                                                                                                   |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Is the ViT frozen?        | **Functionally frozen.** H-optimus-0 is used only to precompute embeddings; only Chowder is trained.                                                                                |
| How many tiles per slide? | **Up to 5000** kept at tiling. Training uses all saved tiles (or the first 5000 precomputed rows). The classifier sees **200**: top-100 + bottom-100.                         |
| Tile-wise or WSI-level?   | **WSI/slide-level classification.** One label per slide/patient, one logit per slide. Per-tile scores are intermediate MIL/interpretability signals, not a trained tile classifier. |

---

## 1. ViT is frozen (H-optimus-0)

`train_fgfr3mut.py:235-238` — loaded pretrained and put in eval mode:

```python
return timm.create_model(
    "hf-hub:bioptimus/H-optimus-0", pretrained=True,
    init_values=1e-5, dynamic_img_size=False,
).to(device).eval()
```

`.eval()` alone does **not** freeze weights. The actual freeze is structural:

- `train_fgfr3mut.py:296-300` — forward pass runs under `torch.inference_mode()`
  (no autograd graph):
  ```python
  with torch.inference_mode():
      embs.append(model(x).cpu().numpy().astype(np.float32))
  ```
- `train_fgfr3mut.py:476` — embeddings are saved to disk.
- `train_fgfr3mut.py:481-482` — training reloads only the 1536-d embedding
  columns, not the ViT.
- `train_fgfr3mut.py:556` — the trainable model is `Chowder(...)`.
- `train_fgfr3mut.py:563` — optimizer receives **Chowder** parameters only:
  ```python
  opt = torch.optim.Adam(model.parameters(), ...)
  ```
- `train_fgfr3mut.py:597-604` — forward/backward only through Chowder.

**Nuance:** there is no explicit `requires_grad=False` loop. Parameters still
default to `requires_grad=True`, but no gradient path exists because the ViT is
never in the training graph or optimizer.

**Paper:** p. 9 — “For training and inference, the feature extractor weights
were frozen.”

---

## 2. Tile count and filtering

### Tiling defaults — `fgfr3mut_tile_wsi.py:562-576`

| Parameter                 | Value    |
| ------------------------- | -------- |
| `--target_mpp`          | `1.0`  |
| `--tile_size`           | `224`  |
| `--min_tissue_fraction` | `0.60` |
| `--max_tiles`           | `5000` |
| `--seed`                | `0`    |

### Filtering

`fgfr3mut_tile_wsi.py:375-377` — keep only tiles with ≥60% tissue:

```python
frac = tissue_fraction_in_cell(tissue_mask, thumb_downsample, x0, y0, step_l0)
if frac >= min_tissue_fraction:
    kept.append((x0, y0))
```

`fgfr3mut_tile_wsi.py:466-470` — if more than 5000 candidates, seeded random
subsample without replacement:

```python
rng = np.random.RandomState(seed)
if max_tiles and len(candidates) > max_tiles:
    idx = rng.choice(len(candidates), size=max_tiles, replace=False)
    candidates = [candidates[i] for i in sorted(idx)]
```

This is uniform random sampling, not a uniform spatial grid. The filter removes
background, folds, and artifacts — **not** non-tumour tissue.

### Training usage

- `train_fgfr3mut.py:466-476` — raw-tiles path embeds **all** saved coordinates;
  there is no second cap.
- `train_fgfr3mut.py:509` — precomputed-features path uses the **first**
  `max_tiles` rows:
  ```python
  arr = np.load(p, mmap_mode="r").astype(np.float32)[:max_tiles]
  ```
- `train_fgfr3mut.py:691` — the training `--max_tiles` flag applies **only** to
  the `--features_dir` path.
- `train_fgfr3mut.py:193` — `N_EXTREME = 100`.
- `chowder.py:41-43` — `ExtremeLayer(n_top=100, n_bottom=100)` and
  `MLP(200, 1)`.
- `chowder.py:70-76` — all tiles are scored, but only the 200 extreme scores
  reach the slide classifier.

**Per-slide numbers:** ≤5000 stored → all scored → **200** used for the final
logit.

---

## 3. WSI-level, not tile-level

The model is multiple-instance learning: one label per bag of tiles, one output
per slide.

- `train_fgfr3mut.py:99` — labels are per slide:
  ```
  labels.csv: slide,label
  ```
- `train_fgfr3mut.py:335-343` — one label loaded per slide.
- `utils.py:243-257` — dataset returns `(N_TILES, 1536)` features plus one
  scalar label.
- `chowder.py:55-78` — forward returns `(B, 1)` logits, one per slide.
- `train_fgfr3mut.py:604` — loss is computed against the slide label.
- `train_fgfr3mut.py:617-622` — AUC is computed over slides.
- `run_inference.py:105-108` — the paper’s official code averages slide
  predictions **per patient**:
  ```python
  all_preds_tmp.set_index("patient_id", inplace=True)
  mean_preds = all_preds_tmp.groupby(all_preds_tmp.index).mean(numeric_only=True)
  ```

**Paper:** p. 9 — the model is an ensemble of 125 MIL models; each tile gets a
score, the 100 top and 100 bottom scores are selected, and a final logit gives
“whether the case is FGFR3-mutant or wild-type.” Labels come from DNA
sequencing, not tile annotations.

**Per-tile scores:** produced by `TilesMLP` (`chowder.py:70`) and used for
top/bottom selection and heatmaps (Fig. 2e, Fig. 5). There is no tile-level
loss, label, or metric. The Fig. 2e heatmap is a visualisation of the learned
scorer, not an independently validated tile classifier. (Note: Fig. 2e says
112 µm tiles while Methods and code use 224 µm.)

---

## 4. Paper vs. code caveats

| Topic               | Paper                    | This code                                                                                                              |
| ------------------- | ------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| Ensemble size       | 125 models               | Default `--splits 5` → **5 checkpoints** (`train_fgfr3mut.py:690, 750-754`)                                 |
| Per-tile activation | ReLU                     | **Sigmoid** (`chowder.py:99, 198`)                                                                             |
| Aggregation         | Average **logits** | Sigmoid, then average **probabilities** (`run_inference.py:80, 108`; `train_fgfr3mut.py:668-669`)            |
| Tile sampling       | “Uniformly sampled”    | Seeded random subset (`fgfr3mut_tile_wsi.py:466-470`); precomputed path takes first 5000 (`train_fgfr3mut.py:509`) |

---

## Bottom line

The pipeline is a **frozen ViT feature extractor + trainable Chowder MIL
classifier**. Tiles are filtered once at tiling (≥60% tissue, 224 px at ~1.0
MPP, ≤5000 per slide), and Chowder aggregates the 200 most extreme tile scores
into **one WSI-level prediction**. Anything described as a “tile prediction” is
an intermediate score for pooling or visualisation, not a supervised
tile-level output.
