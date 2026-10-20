# How Chowder Turns Tiles Into One Slide Prediction (Plain Language)

A beginner's guide to the tile-scoring step in this repo, with code references.

Source: `scripts/fgfr3mut_training/fgfr3mut/fgfr3mut/chowder.py` (the model) and
`scripts/fgfr3mut_training/train_fgfr3mut.py` (the training loop).

---

## 1. The words you need first

| Word                  | Meaning here                                                                                                                                        |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **slide / WSI** | One whole-slide image of a patient’s tissue (`.svs`). Too big for a neural net, so it is cut into tiles.                                         |
| **tile**        | One small 224×224 pixel image cut out of a slide. A typical slide yields a few thousand tiles.                                                     |
| **embedding**   | The 1536 numbers that describe a tile. Produced by the frozen ViT (H-optimus-0). The ViT is never trained.                                          |
| **bag**         | All the tiles of one slide, treated as one group. A bag of ~4000 tiles = one slide.                                                                 |
| **bag label**   | The one answer we have for the bag: 1 = FGFR3 mutated (MUT), 0 = wild-type (WT). It comes from DNA sequencing, **not** from looking at tiles. |
| **tile score**  | One number the model gives each tile, saying “this tile looks more MUT-like” or “more WT-like”. It is *not* a probability.                    |
| **logit**       | The model’s raw output number for the slide, before it is squashed into a 0–1 probability.                                                        |

**The core problem:** we have 4000 tiles and exactly **one** answer per slide.
Nobody told us which tiles are the mutant ones. The model has to work that out
for itself, and the only grade it ever receives is the final slide-level answer.

---

## 2. The whole thing in one picture

```
slide (one .svs)
  └── ~4000 tiles
        └── each tile -> 1536 numbers (frozen ViT)           <- no learning happens here
              └── each tile -> 1 "tile score"                <- step 1: score every tile
                    └── keep the 100 highest + 100 lowest    <- step 2: keep the extremes
                          └── 200 numbers -> 1 logit         <- step 3: one number per slide
                                └── sigmoid -> P(MUT)        <- the final answer
```

Four thousand tiles in, **one** number out. That is the whole trick.

---

## 3. Step 1 — Score every tile (`chowder.py:70`)

```python
# chowder.py:70
scores = self.score_model(x=x, mask=mask)     # (B, N_TILES, 1)
```

`score_model` is a tiny 3-layer network (`chowder.py:190-208`):

```python
# chowder.py:200-208
self.hidden_layers = torch.nn.ModuleList()
for h in [128]:
    self.hidden_layers.append(MaskedLinear(in_features, h, mask_value="-inf"))
    self.hidden_layers.append(activation)          # Sigmoid
    in_features = h
self.hidden_layers.append(torch.nn.Linear(in_features, out_features, bias=True))
```

In plain words:

| Step                          | What it does                                  | Result                                |
| ----------------------------- | --------------------------------------------- | ------------------------------------- |
| `MaskedLinear(1536 → 128)` | squeezes the tile’s 1536 numbers down to 128 | `(B, N, 128)`                       |
| `Sigmoid`                   | squashes each of those 128 to between 0 and 1 | `(B, N, 128)`                       |
| `Linear(128 → 1)`          | squeezes those 128 down to a single number    | `(B, N, 1)` = **tile scores** |

The same weights are used for every tile of every slide. A tile’s score depends
**only on that tile** — not on where it sits, not on its neighbours.

**About the mask:** slides have different numbers of tiles, so batches are
zero-padded to the longest slide in the batch (`utils.py:116-180`). The mask
marks which rows are padding. `MaskedLinear` writes `-inf` into padded rows
(`chowder.py:165-168`); `sigmoid(-inf) = 0`, so padded tiles score a harmless
constant and produce no gradient. You can ignore padding entirely.

**Key point:** because the ViT is frozen, the only thing being learned here is
the 1536 → 128 → 1 “lens” that decides what counts as MUT-like. It can
re-weight what the ViT already sees, but it cannot learn new visual features.

---

## 4. Step 2 — Keep only the most extreme tiles (`chowder.py:71-73`)

```python
# chowder.py:71-73
extreme_scores = self.extreme_layer(x=scores, mask=mask)   # (B, 2*N_EXTREME, 1)
```

`ExtremeLayer` is just a ranking + selection step (`chowder.py:301-333`):

```python
# chowder.py:303-316
top, top_idx = x.masked_fill(mask, float("-inf")).topk(k=self.n_top, sorted=True, dim=self.dim)
...
bottom, bottom_idx = x.masked_fill(mask, float("inf")).topk(k=self.n_bottom, largest=False, sorted=True, dim=self.dim)
```

- `topk(k=100)` → the **100 highest** scoring tiles (most MUT-like).
- `topk(largest=False)` → the **100 lowest** scoring tiles (most WT-like).
- The two lists are glued together in that order (`chowder.py:332`).

So out of ~4000 tiles, **200 survive** (100 top + 100 bottom).
`N_EXTREME = 100` is set in `train_fgfr3mut.py:193` and passed in at
`train_fgfr3mut.py:556`:

```python
# train_fgfr3mut.py:193
N_EXTREME = 100       # Chowder keeps the top-100 and bottom-100 scoring tiles per slide
```

If a slide has fewer than 200 tiles, the number is shrunk to fit
(`train_fgfr3mut.py:531-534`).

**Why keep the bottom ones too?** For a mutated slide, the *most MUT-looking*
tiles carry the evidence. For a wild-type slide, the *most WT-looking* tiles
do. The classifier sees both lists, so it can use whichever half is
informative for that particular slide.

---

## 5. Step 3 — 200 numbers into one number (`chowder.py:76`)

```python
# chowder.py:76
y = self.mlp(extreme_scores.transpose(1, 2))    # (B, 1, 1)
```

`self.mlp` is a small network over the 200 numbers (`chowder.py:91-111`):

```
200 -> Linear(128) -> Sigmoid -> Linear(64) -> Sigmoid -> Linear(1)   = 1 logit
```

`chowder.py:78` returns both outputs:

```python
# chowder.py:78
return y.squeeze(2), extreme_scores       # (B, 1) slide logits, (B, 200, 1) extreme scores
```

The logit becomes a probability in the training/eval code
(`train_fgfr3mut.py:615-617`):

```python
logit, _ = model(f.to(device), mask=mk.to(device))
P += torch.sigmoid(logit).cpu().flatten().tolist()
```

**One logit per slide.** That is the final answer.

---

## 6. The surprising part: how are tile scores learned if no tile is labelled?

This is the part that trips everyone up. The model is only ever graded on the
final slide answer (`train_fgfr3mut.py:597-604`):

```python
# train_fgfr3mut.py:597-604
logit, _ = model(f, mask=mk)
crit(logit.squeeze(1), lb.squeeze(1)).backward()
```

The loss is computed on the slide label, *after* the top/bottom selection has
already thrown most tiles away. So:

> **A tile that was not selected into the top-100/bottom-100 contributes
> nothing to the loss. It cannot be improved. It gets no feedback at all.**

Only the 200 selected tiles get a gradient, and only through the final
classifier. The model is therefore pushed to do two things:

1. Make the tiles it currently considers extreme as *useful* as possible.
2. Over time, learn to push genuinely informative tiles **up** (toward the
   top-100) or **down** (toward the bottom-100), so that the extreme lists
   become full of evidence.

A good analogy: a teacher has 4000 exam sheets from one student, one final
grade for the student, and no marks on individual sheets. The teacher learns to
*rank* the sheets — the best 100 and the worst 100 — and then writes one report
based on those 200. Sheets in the middle teach the teacher nothing, because
they never make it into the report.

This is why the score should be read as a **ranking**, not a probability.

---

## 7. What the tile score is, and what it is not

|                     |                                                                                                                                                                                  |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **It IS**     | A per-tile ranking signal: “compared with the other tiles in this slide, this one looks more MUT-like.” Comparable across slides because the same weights are used everywhere. |
| **It is NOT** | A calibrated probability that the tile is mutant. Nothing calibrates it — there is no tile-level truth to calibrate against.                                                    |

Evidence that the score carries real meaning comes from the paper: a
pathologist reviewed the 200 most and least predictive tiles and found that
top-scoring tiles show monomorphic tumour morphology with little stroma,
while bottom-scoring tiles show pleomorphic morphology, desmoplastic stroma
and inflammation (Bannier et al., *Nat Commun* 15:10914 (2024), Methods
“Interpretability analysis”, p. 9; Figs. 2f, 5a).

---

## 8. Getting the tile scores yourself

`Chowder.forward` only returns the 200 selected scores, not all of them
(`chowder.py:78`). To get every tile’s score, call the scorer directly, as
this repo’s own heatmap script does (`scripts/reproduce/heatmap.py:90-93`):

```python
s = model.score_model(X).squeeze(0).squeeze(1).cpu().numpy()   # (N_TILES,)
s = sigmoid(s)                                                 # for display only
acc = acc + s                                                  # average over the ensemble
```

---

## 9. Quick reference — where each piece lives

| Step                          | File and lines                                                    |
| ----------------------------- | ----------------------------------------------------------------- |
| Score every tile              | `chowder.py:70` → `chowder.py:176-229` (`TilesMLP`)        |
| Padding is neutralised        | `chowder.py:114-173` (`MaskedLinear`), `utils.py:116-180`   |
| Keep top-100 + bottom-100     | `chowder.py:71-73` → `chowder.py:232-349` (`ExtremeLayer`) |
| 200 scores → 1 logit         | `chowder.py:76` → `chowder.py:81-111` (`MLP`)              |
| Slide probability             | `train_fgfr3mut.py:615-617`                                     |
| Where the learning happens    | `train_fgfr3mut.py:592-608` (only Chowder is trained)           |
| `n_extreme = 100`           | `train_fgfr3mut.py:193`, used at `train_fgfr3mut.py:556`      |
| Tile count per slide          | `fgfr3mut_tile_wsi.py:562-576` (≤5000), filter at `:375-377` |
| Per-tile scores for a heatmap | `scripts/reproduce/heatmap.py:90-93`                            |

---

## 10. One-paragraph summary

Each slide is cut into up to 5000 tiles; a frozen ViT turns each tile into 1536
numbers; a **tiny trainable network** turns those 1536 numbers into one *tile
score* per tile; the 100 highest and 100 lowest scores are kept; a **second
small network** turns those 200 numbers into a single logit, which becomes
P(slide is FGFR3-mutant). The tile scores are never supervised directly — they
are shaped indirectly, because only the tiles that make it into the 200
survivors can influence the slide-level loss. That is the whole idea behind
Chowder, and it is what makes it a *multiple instance learning* model: one
label for a bag of thousands of unlabelled instances.
