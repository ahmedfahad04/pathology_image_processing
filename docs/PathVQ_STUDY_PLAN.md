# PathVQ Study Plan

**Paper:** *PathVQ: Reforming Computational Pathology Foundation Model for Whole Slide Image
Analysis via Vector Quantization* — Li, Shui, Zhang, Zhu, Yang (Westlake Univ. / ZJU), NeurIPS 2025.
`docs/papers/PathVQ.pdf`

**Goal:** be able to answer, without notes, any question about (a) the methodology, (b) vector
quantization, (c) what/why they did, (d) how good it is.

**Total time:** ~16–20 focused hours (about 3 days at 2–3 h/day, or 5 days at 1 h + commute reading).

**Rule that makes this work:** every phase ends with a **Gate**. A gate is a question you must be
able to answer *in writing, from memory, with the paper closed*. If you fail a gate, redo that
phase — do not move on. Passing all gates is the actual study; reading the paper is just phase 7.

**Files in this package**

| File | What it is |
|---|---|
| `docs/PathVQ_STUDY_PLAN.md` | this file — the 8 phases, the gates, the cheat sheet |
| `docs/PathVQ_QA_BANK.md` | 45 questions a professor could ask, with model answers (Parts A–E) |
| `scripts/learning/vq_attention_intuition.py` | runnable demo: attention in numpy (verified against torch), a real VQ-VAE with straight-through, codebook collapse, and MSVQ multi-scale residual. ~15 s on CPU. |
| `docs/papers/PathVQ.pdf` | the paper itself |
| `docs/HISTOPATHOLOGY_GUIDE.md` | domain primer already in this repo — read it in Phase 6 |
| `docs/TRANSFORMERS_ON_IRIS.md` | transformer theory notes already in this repo — skim in Phase 1 |

---

## 0. The one-paragraph summary of what you are about to learn

A pathology foundation model (like UNI) turns a 224×224 tile into **197 vectors**: one `[CLS]`
summary token + **196 patch tokens** (14×14 grid, 16×16 px patches). A whole slide has 5,000+
tiles. Every WSI method to date throws away the 196 patch tokens and keeps only the `[CLS]` →
1,000,000 patch tokens are discarded, and the slide model never sees spatial structure.
PathVQ says: *don't discard them, but you can't afford 196× more tokens either.* So it **compresses
each patch token with vector quantization** (1024-d → 16-d, 64×) using a learned codebook, then
reconstructs a *tile-level* feature from the compressed grid with a small conv stack, and uses the
codebook indices as **discrete self-supervision targets** for slide-level pretraining.

Everything else in the paper is an implementation of that sentence.

---

## Phase 1 — Attention: the "how is attention calculated" question (4 h)

You said you froze on this. This is the single highest-value 4 hours of the plan. Do not skip it.

### 1.1 Must know (write these from memory)

1. **Why a transformer at all.** A transformer is a function over a *set of vectors*:
   `T: R^{n×d} → R^{n×d}`. It has no inherent notion of order (that's what positional encoding
   supplies) and it lets any token attend to any other token in O(1) path length. For pathology this
   matters: a slide is an unordered bag of tiles (MIL) and a tile is an unordered grid of patches.

2. **The three projections.** For input `X ∈ R^{n×d}`:
   `Q = XW_Q`, `K = XW_K`, `V = XW_V`. Semantics: *query* = "what am I looking for",
   *key* = "what do I contain", *value* = "what do I actually hand over".
   `W_Q, W_K, W_V ∈ R^{d×d_k}` are learned.

3. **Scaled dot-product attention (the actual formula — be able to write it and expand it):**

   ```
   Attention(Q,K,V) = softmax( Q Kᵀ / √d_k ) V
   ```

   Step by step, with shapes:
   - `QKᵀ ∈ R^{n×n}` — every token scores every token.
   - `√d_k` in the denominator — critical. If `q,k` have i.i.d. variance 1 per component, then
     `q·k` has variance `d_k`. Without dividing, logits grow like `√d_k`, softmax saturates,
     gradients vanish. Dividing by `√d_k` keeps logits O(1) regardless of `d_k`. **This is the
     answer to "why is it scaled".** You can derive it in one line; be able to.
   - `softmax` over the **row** (over keys j) → each row is a probability distribution over "which
     token do I attend to". Rows sum to 1.
   - multiply by `V` → output is a **weighted average of values**, weights = attention distribution.
     So **attention output is a convex combination of value vectors.** A professor may ask:
     "what is the range of the output vector?" → it lies in the convex hull of the value vectors.

4. **Causality mask.** In NLP, mask `j > i` → `−∞` before softmax → 0 probability. In this paper
   masks are NOT used (bidirectional), and *position* is supplied by RoPE instead. Know the
   difference: decoder-only (causal) vs encoder (bidirectional).

5. **Multi-head attention.**
   ```
   MultiHead(X) = Concat(head_1,…,head_h) W_O ,  head_i = Attention(XW_Q^i, XW_K^i, XW_V^i)
   ```
   Each head has its own projection (usually `d_k = d/h`). Why multiple: one head computes a single
   similarity function; different heads can specialize (e.g. head 1 attends to "same cell type",
   head 2 to "adjacent tile"). **Head count does not change the math of a single head** — know the
   single-head math cold.

6. **The full block.**
   ```
   X' = LayerNorm(X + MultiHead(X))          # residual + pre-norm
   Y  = LayerNorm(X' + FFN(X'))              # FFN(x) = W₂ σ(W₁ x + b₁) + b₂, d→d_ff→d
   ```
   Residual = keeps gradient flowing, makes it possible to stack. Pre-LN (norm before sublayer) is
   what modern ViTs use for training stability. FFN = position-wise MLP, the "thinking" happens
   here; attention is the "routing".

7. **Positional encoding.** Attention is permutation-equivariant — without position info, token 1
   and token 2 are interchangeable. Options: absolute (learned, ViT) vs **RoPE** (rotary; rotate
   Q,K by angle proportional to absolute position; a dot product of two rotated vectors depends
   only on their *relative* offset). PathVQ uses **RoPE**, including 2-D RoPE for slides
   (x-coord and y-coord each get their own rotation). Know RoPE's one-line idea:
   `q_i' = R_i q_i`, `k_j' = R_j k_j`, then `q_i'·k_j'` depends on `i−j` only.

### 1.2 Resources (pick one path, not all)

| Resource | Why | Length |
|---|---|---|
| Jay Alammar, *The Illustrated Transformer* | fastest correct mental model | 1 h read |
| 3Blue1Brown, *Attention in transformers, step-by-step* (2 videos) | makes the QKV intuition stick | 1 h watch |
| *Attention Is All You Need* (Vaswani et al. 2017), §3.2.1 only | the source of truth for the formula | 30 min skim |

### 1.3 Do it (this is what makes it stick)

Implement single-head attention in **numpy**, no torch, from the formula above, and check it
matches `torch.nn.functional.scaled_dot_product_attention` on random input.
→ `scripts/learning/vq_attention_intuition.py` (Part 1) does exactly this, with printed
attention maps you can eyeball. Run it and read the output.

### ✅ Gate 1
1. Write the scaled dot-product formula from memory, with all three shapes.
2. In one sentence each: what is Q, what is K, what is V.
3. Explain the `√d_k` scaling in terms of logit variance. (If you say "to keep gradients stable",
   that's only half — you must mention *logit magnitude → softmax saturation*.)
4. Given a 2-token, `d_k=2` example with numbers, compute the attention weights by hand.
5. What is the output of a single attention head, in terms of V? → a convex combination.

---

## Phase 2 — Vision Transformer and what a `[CLS]` token actually is (2 h)

### 2.1 Must know

- **Patch embedding.** 224×224 image, patch size 16×16 → 14×14 = **196 patches**. Each patch
  (16·16·3 = 768 raw pixels) is flattened and linearly projected to `d=1024` → 196 tokens.
  A conv with stride 16 is exactly this, done in one op.
- **`[CLS]` token.** One extra *learned* token is prepended → 197 tokens total. Through the
  network it attends to all patches; the final `[CLS]` is therefore a learned **summary of the tile
  as a whole** (this is the "summary [CLS] of the spatial tokens of all patches" phrasing in
  PathVQ §2.1).
- **Positional embeddings.** ViT adds a learned position embedding per patch.
- **What the PathVQ paper actually consumes:**
  - `s` = the `[CLS]` token (1024-d) → what everyone else uses.
  - `ST = [h₁ … h₁₉₆]` = the 196 spatial patch tokens (1024-d each) → what PathVQ wants.
  - `h_i` is a *middle-layer* representation, not the final one — remember this, it explains
    §2.4's "the encoding direction of ST is not so controllable".

### 2.2 Resources
- *An Image is Worth 16x16 Words* (Dosovitskiy et al. 2021), §3.1–3.2. Read this once, properly.
- Your repo already has `docs/TRANSFORMERS_ON_IRIS.md` — it covers tokenization/positional ideas
  at the right level; skim §1–2 to reconnect.

### 2.3 Do it
Compute by hand: tokens for a 224 tile at patch 16; the ratio of storage between keeping
`[CLS]` only vs all 196 tokens (this is exactly the paper's "200×" claim — you now understand where
it comes from).

### ✅ Gate 2
1. How many tokens does a 224×224 tile produce with 16×16 patches? Why is the `[CLS]` token
   "a summary of the spatial tokens"?
2. Why is 196× (not 200× exactly) the number the paper quotes as the cost multiplier? *(It is
   196 patch tokens replacing 1 CLS token → 196×.)*
3. Where does the information in the 196 tokens "go" if you keep only `[CLS]`? *(It is averaged/
   attended into `[CLS]`; fine-grained spatial detail is irreversibly mixed/lost.)*

---

## Phase 3 — Vector Quantization: the "how is quantization calculated" question (4 h)

This is the paper's core. If you master this phase, you can answer every method question.

### 3.1 Must know — plain VQ layer

1. **Codebook.** `V = [v₁, …, v_C] ∈ R^{C×d}`. `C` learned embedding vectors ("codewords",
   "prototypes", "visual tokens"). PathVQ: `C = 8192`, `d = 16`.
2. **The quantization step (the answer to "how is quantization calculated"):**

   ```
   z_i = arg min_j ‖ ℓ₂(e_i) − ℓ₂(v_j) ‖₂        (PathVQ Eq. 2)
   ```

   Read it as: L2-normalize the encoder output and every codeword, then pick the nearest
   codeword. Because `‖â−b̂‖² = 2 − 2·â·b̂` for unit vectors, **this is exactly nearest-neighbour
   in cosine similarity.** That is *why* they L2-normalize: cosine similarity is the metric they
   want (direction of the feature matters; magnitude is nuisance), and normalizing makes the
   argmin equal to a max-cosine argmax.
   - `z_i` = the **index** (an integer in `[0, C)`).
   - `e_{z_i}` = the selected codeword (the "quantized" vector).
3. **Why is this hard to train.** `arg min` is a *discrete* selection → zero gradient w.r.t. `e`.
4. **Straight-through estimator (STE).** Treat the forward pass as the argmin, but in the backward
   pass pretend it was the identity:
   ```
   forward:  e → z = argmin → e_z
   backward: ∂L/∂e  :=  ∂L/∂e_z     (identity Jacobian)
   ```
   So the encoder receives the gradient that would have flowed to the codebook lookup. This is the
   trick from VQ-VAE (van den Oord 2017). Be able to state it in one sentence and write the
   3-line PyTorch implementation (`y_q = e + (e_q - e).detach()`).
5. **The loss (VQ-VAE).** Three parts, and know which is which:
   ```
   L = ||x − x̂||²  +  || sg[e] − e_q ||²  +  β·|| e − sg[e_q] ||²
             reconstruction      codebook loss          commitment loss
   ```
   - **codebook loss** moves the *codebook* toward the encoder outputs.
   - **commitment loss** moves the *encoder* toward its assigned codeword (so the encoder doesn't
     drift; β = 0.25 in the original VQ-VAE).
   - PathVQ writes it (Eq. 3) as `max Σᵢ [ cos(oᵢ, hᵢ) − ‖ℓ₂(eᵢ) − ℓ₂(v_i)‖² ]` — i.e.
     **cosine reconstruction** + **codebook term**, and it *says* it omits the stop-gradient
     notation and the straight-through path. **Note: a commitment term is not explicitly visible
     in Eq. 3 — that is a legitimate question to raise.** (See the QA bank, Q17.)
6. **Cosine reconstruction instead of MSE.** PathVQ maximizes `cos(o_i, h_i)` — they care about
   the *direction* of the feature vector, consistent with L2-normalized quantization.

### 3.2 Must know — Residual VQ (needed for MSVQ)

- **Idea (RQ-VAE / SoundStream / residual quantization):** don't try to approximate `e` with one
  codeword. Approximate it with a **sum of codewords**:
  `e ≈ c₁ + c₂ + … + c_K`, where `c₁` is chosen to capture the coarse/global part, and each
  subsequent `c_k` is chosen to capture the **residual** `e − (c₁+…+c_{k−1})`.
- **Algorithm (residual quantization):**
  ```
  r_0 = e
  for k = 1..K:
      z_k = argmin_j ‖ℓ₂(r_{k−1}) − ℓ₂(v_j)‖      # quantize the residual
      c_k = v_{z_k}
      r_k = r_{k−1} − c_k
  ```
- **Why it matters here:** a coarse scale (1×1 grid) captures tile-level/global semantics, and
  finer scales (14×14) capture local patch detail — and a *single shared codebook* across scales
  means one vocabulary for both.

### 3.3 Resources
- van den Oord et al., *Neural Discrete Representation Learning* (VQ-VAE 2017), §2.1–2.2 — read
  slowly, this is the paper PathVQ's VQ comes from.
- *Taming Transformers* (Esser et al. 2021) — for VQ-VAE as a tokenizer and codebook collapse.
- *SoundStream* (Zeghidour et al. 2021) or *FiLM* (2022) for residual VQ. Or simply RQ-VAE
  (Residual Quantized VAE, 2022) — the cleanest short description of the residual idea.
- 3Blue1Brown, *A gentle introduction to neural vector quantization* (or *But what is a
  convolutional neural network?* follow-up) if you want the visual intuition.

### 3.4 Do it
- Implement a VQ layer with STE in PyTorch and train it on synthetic 2-D data; watch
  (a) reconstruction improve, (b) codebook usage collapse if you only use the codebook loss with
  a high LR, (c) the effect of adding the commitment term.
- Implement residual VQ with K=2 and show that sum of two codewords beats one.
→ `scripts/learning/vq_attention_intuition.py` (Parts 2–3) does all of this in ~20 s on CPU and
prints the numbers you need to answer "how is quantization calculated".

**Expected output you should see (so you know the demo is working):**
- Part 1: `max |numpy - torch.scaled_dot_product_attention| = 2.2e-16` and the empirical logit
  std table showing `std(raw) ≈ sqrt(d_k)` (4.0, 8.0, 16.0, 32.0) while `std(scaled) ≈ 1.0`.
- Part 2: identity error `2 - 2cos == ||a_hat - b_hat||^2` at ~1e-7; gradient through the hard argmin
  = `0.000e+00`; gradient through STE = non-zero with an identical forward value.
- Part 2b: a trained VQ layer reaching cosine ≈ 0.97 on the synthetic tile.
- Part 2c: an oversized codebook using only a handful of codes — **collapse**.
- Part 3 (the important one): reconstruction error falling as scales are added —
  `1x1: 0.85  →  +2x2: 0.58  →  +4x4: 0.24  →  +7x7: 0.16  →  +14x14: 0.05`.
  That curve **is** MSVQ's mechanism, and it is the number to remember from Phase 3.

### ✅ Gate 3
1. Write PathVQ Eq. 2 from memory and explain every symbol (`e_i`, `v_j`, `ℓ₂`, `C`, `z_i`).
2. Why is `argmin` non-differentiable, and what is the straight-through estimator?
3. What is the difference between the codebook loss and the commitment loss? Which one moves the
   codebook?
4. Why does the paper L2-normalize before computing distance? *(→ nearest neighbour under cosine
   similarity.)*
5. Explain residual quantization in 4 lines, with the loop.
6. Why can a codebook "collapse", and what two mitigations do you know? *(dead codes; mitigations:
   EMA codebook updates, k-means re-init of dead codes, codebook reset, L2-normalized +
   small-ε codebook.)*

---

## Phase 4 — MIL & attention pooling (2 h)

PathVQ's downstream model is MIL; its slide-level SSL objective is literally "attention pooling →
softmax over 8192 token classes". You cannot read §2.3 without this.

### 4.1 Must know
- **MIL setup.** Slide = *bag* of N tile instances; slide label = *bag* label; tile labels unknown.
  Learning signal: "somewhere in this bag there is tumour."
- **ABMIL (Ilse et al. 2018).**
  ```
  a_i = softmax(wᵀ tanh(V h_i) ⊙ sigmoid(U h_i))       # gated attention
  z   = Σ_i a_i h_i                                     # attention pooling
  ŷ   = softmax(W z)
  ```
  Plus an **attention loss** forcing the top-k highest-attention instances to be positive (a
  supervision trick for interpretability/attention quality).
- **Why PathVQ's ABMIL SSL makes sense:** in supervised ABMIL you pool tile features to get a slide
  prediction. PathVQ reuses exactly that pooling, but trains the head to predict the *token
  histogram* of the region (Eq. 4) — the soft target `q_c(x)` is the **frequency of token class c
  among the region's tiles**, normalized over the C=8192 classes. This is a "bag of visual words"
  objective; it is a *self*-supervised target because the "words" come from the frozen tokenizer.
- **TransMIL**: adds a `[CLS]` token to the instance sequence and uses **Nyström-approximated**
  self-attention for O(n) cost; also uses **correlative self-attention** (cosine similarity between
  instance embeddings) which is important for pathology.
- **LoRA** (used for fine-tuning PathVQ's pretrained transformer): freeze base weights, learn
  low-rank `ΔW = BA` with `B ∈ R^{r×d}`, `A ∈ R^{d×r}`, `r=16` in this paper.

### 4.2 Resources
- Ilse et al., *Attention-based Deep MIL* (2018) — 10 min.
- Shao et al., *TransMIL* (2021) — abstract + §3.
- *RoPE* (Su et al. 2021) §3 — and 2-D RoPE from ref [41] of the paper (the authors' own prior
  work: *Rethinking Transformer for Long Contextual Histopathology WSI Analysis*).

### ✅ Gate 4
1. Write ABMIL's attention pooling in 3 lines of pseudo-code. What is the output? *(weighted
   average of instance features.)*
2. What is the "attention loss" for and why is it needed if the bag label is already known?
3. In PathVQ Eq. 4, what exactly is `q_c(x)`? Where does the label come from? *(No human label —
   it is computed from the frozen tokenizer's own outputs: a self-consistent target.)*
4. Why is a *frozen offline* tokenizer a more stable SSL target than iBOT's *online* tokenizer?
   *(An online tokenizer co-adapts with the encoder, so the target moves during training — a
   moving-target / collapse risk. A frozen tokenizer gives a fixed label space.)*

---

## Phase 5 — Self-supervised learning & masked image modeling (2 h)

PathVQ's slide-level objective is MIM, but applied to *features* instead of pixels, and BEiT-style
(discrete targets) instead of MAE-style (pixel targets).

### 5.1 Must know
- **MAE (2021):** mask ~75% of patches, encoder sees only visible patches, a *lightweight decoder*
  reconstructs the **pixel values** of masked patches. Asymmetric: encoder heavy, decoder light.
- **BEiT (2021):** same masking, but the decoder predicts a **discrete visual token index**
  produced by an **offline tokenizer** (dVAE) instead of pixels. (This is literally PathVQ's move:
  swap the dVAE image tokenizer for a VQ feature tokenizer.)
- **iBOT (2021):** adds a **masked image modeling** head *and* an **online tokenizer** (an EMA
  teacher) producing targets, plus a contrastive objective. PathVQ's argument against iBOT: the
  online tokenizer's targets are **unstable** because teacher and student co-evolve.
- **DINO (2021):** self-distillation with no labels; mention it as context for "why SSL works".
- **Why predict discrete tokens instead of pixels?** Because pixel reconstruction is dominated by
  low-level texture and is a very hard target; a discrete semantic code is a *smoother, more
  semantic* target, and the model is forced to learn semantics, not texture.

### 5.2 Resources
- He et al., *Masked Autoencoders Are Scalable Vision Learners* (2021) — §3.
- Bao et al., *BEiT: BERT Pre-Training of Image Transformers* (2021) — §3.1, Fig. 1.
- Zhou et al., *iBOT* (2021) — abstract + the online-tokenizer paragraph.

### ✅ Gate 5
1. Difference between MAE and BEiT targets, in one sentence each.
2. What exactly is an "offline tokenizer" vs an "online tokenizer"? Why is offline more stable?
3. PathVQ's masked modeling (Eq. 7) — what is the label `z_i` for a masked tile, and where does it
   come from? Is the target tile-level or patch-level? *(Careful: the paper is ambiguous here.
   See QA bank Q18 — you should be able to state your reading AND the evidence.)*
4. Why does PathVQ need RoPE if it's a "Transformer-based MIL model"? *(Because tiles in a region
   have a spatial arrangement, and attention is permutation-equivariant.)*

---

## Phase 6 — Pathology-specific context (1 h)

You already have `docs/HISTOPATHOLOGY_GUIDE.md` in this repo. Read it. Then:
- **WSI basics:** gigapixel image, pyramid of levels; a slide is tiled at 224×224 (or 256) at
  ~20× magnification; a slide yields 5k–100k+ tiles.
- **Why WSI is hard:** (i) size — billions of pixels; (ii) label scarcity — patient-level labels
  only, no tile-level annotations (weak supervision); (iii) class imbalance; (iv) tissue
  heterogeneity (one slide contains many distinct morphologies).
- **The two pretraining families:** vision-only SSL (scalable, unlabeled data) vs vision-language
  (strong, but needs image-text pairs which are scarce in pathology).

### ✅ Gate 6
1. Why are WSIs tiled at all?
2. Why is WSI learning "weakly supervised" — what is the bag, what is the instance, what is the label?
3. Name one vision-only pathology FM and one vision-language pathology FM, and the data bottleneck
   of each.

---

## Phase 7 — Read PathVQ (three passes, 4–5 h)

Now — and only now — open the PDF. Use the passes; do not read it linearly front to back.

### Pass 1 — 30 min: the skeleton
Read: Abstract, §1 Introduction, Figure 1, Figure 3, the 3 bullet contributions, §4 Conclusion.
Do **not** read any equation. After this pass, write down:
- The problem in one sentence.
- The method in one sentence.
- The three claimed contributions (verbatim).
- The single most important number in the paper (your answer: **196×**, the cost of keeping all
  patch tokens vs `[CLS]` only).

### Pass 2 — 2.5 h: the method, equation by equation
Read: §2 in order, with Figure 2 open next to Algorithm 1.
For **each** of Eq. (1)–(7), fill this table in your own notes:

| Eq | What it computes | Why it exists | Which Phase-3/4/5 concept it uses |
|----|------------------|---------------|--------------------------------|
| 1  | `Enc: R^1024 → R^16` per patch | shrink to codebook dim | — |
| 2  | nearest codeword (L2-normed) | discrete index | VQ |
| 3  | VQ training objective | train enc/codebook/dec | STE + codebook loss |
| 4  | ABMIL SSL cross-entropy | slide-level self-supervision | ABMIL |
| 5  | `softmax(classifier[AttnPool(x)])` | predicted token distribution | ABMIL |
| 6  | corrupted input with `e[M]` | mask tiles | MIM |
| 7  | masked token CE | MIM loss | BEiT-style offline target |

Then read §2.4 (downstream fine-tuning) and §2.5 (framework summary), and be able to trace
**Figure 3 end-to-end**:
tile → frozen FM ViT → 196 patch tokens (1024-d) → VQ encoder → 16-d → codebook indices →
decoder reconstructs 1024-d (for training) / conv stack 128→256→512→1024 stride 2 → (1×1, 1024)
per tile → treated as a tile `[CLS]`-equivalent → MIL model.

### Pass 3 — 2 h: the evidence
Read: §3.1 (implementation), Tables 1–2, Figure 5, §3.3 ablations, and Appendix A.2/A.3/A.4/A.6.
Answer these with numbers:
- BRACS: ABMIL F1 → +PathVQ → +PathVQ+SPT? (0.692 → 0.730 → 0.747). AUC (0.875 → 0.902 → 0.906)
- LGG-GBM mutation: F1 (0.685 → 0.723 → 0.752), AUC (0.852 → 0.871 → 0.879)
- Roformer+PathVQ+SPT on BRACS: F1 0.754, AUC 0.910 (best in table)
- Survival (c-index, 5-fold): ABMIL-reproduce → +PathVQ → +SPT across 5 TCGA cohorts —
  BRCA 0.633→0.655→0.674, CRC 0.612→0.649→0.659, BLCA 0.540→0.608→0.616,
  UCEC 0.671→0.721→0.748, KIRC 0.691→0.760→0.778
- The UNI-2 argument: UNI-2+ABMIL improves only ~0.6–1.4% over UNI+ABMIL, while PathVQ
  (with only 1M tiles) improves 3.8–4.8% → *the bottleneck was information, not capacity.*
- Figure 5 ablations: PCA ≈ 0.70 F1; avg-pool 4×4/7×7 ≈ CLS; `[CLS]`+avg ≈ +1 point; TopK-64 ≈
  0.72 but heavy; TopK-16 ≈ 0.70; **VQ ≈ 0.73** (best).
- Compute cost of the VQ stage: 1M tiles, 4×RTX-3090, 50 epochs, ~22 h.

### Pass 4 — 30 min: be adversarial (this is where marks are won)
Re-read §1 and §4 and list **limitations and weaknesses** in the paper's own terms, then your own:
- The VQ stage needs extra training data (1M tiles) — the paper's stated limitation.
- VQ is lossy; some information is lost (paper's stated limitation).
- Does not beat Cox-loss-trained slide FMs (TITAN, CHIEF) on survival — stated in A.6.
- Only compared against UNI; the backbone ablation (does PathVQ work on CONCH / Virchow / H-1K?)
  is not in the main table.
- 8192-entry codebook at 16 dims is a very low information rate for 1024-d features — how do you
  reconcile "64× compression" with "preserving fidelity"? (Answer: the *decoder* re-inflates using
  learned priors; the claim is about downstream utility, not bit-exactness.)
- The [CLS]-bottleneck claim is supported by the UNI-2 comparison, but UNI-2 is one data point.
- Evaluation is on TCGA-dominated data; external validation is limited.

---

## Phase 8 — Rehearsal (1 h)

- Open `docs/PathVQ_QA_BANK.md` and answer **every** question out loud, in under 2 minutes each,
  without the paper. Then check the answers and note any you got wrong.
- Re-answer any failed question the next day.
- Then read the paper's own figures one final time, with the PDF closed, and be able to redraw
  Figures 2 and 3 from memory.

---

## Where to get the paper's other references (already in `docs/papers/`)
- `CALM.pdf` — *CALM* (computational pathology foundation model, MHV-style) — for context on what
  a "pathology FM" is.
- `NMGrad.pdf` — for normalization/robustness context.
- `s41467-024-55331-6.pdf` — a Nature Communications pathology paper; skim for how these papers
  structure methods and ablations.
- Also in repo: `docs/MIL_SCALE_PARADIGMS_2024_2026.md` (MIL taxonomy),
  `docs/HISTOPATHOLOGY_GUIDE.md` (domain primer) — read these *before* Phase 6.

---

## One-page cheat sheet to keep beside you while reading

```
ATTENTION
  Q=XWq, K=XWk, V=XWv
  head = softmax(QKᵀ/√dk) V          # (n,n) scores, row-softmax, weighted avg of V
  multihead = Concat(heads) Wo
  block: X ← LN(X + MHA(X)); X ← LN(X + FFN(X))

ViT
  224px, patch 16 → 14×14 = 196 patches → +[CLS] = 197 tokens, d=1024 (UNI ViT-L)

VQ
  codebook V ∈ R^{C×d}
  z = argmin_j ||ℓ2(e) − ℓ2(vj)||   ≡ nearest in COSINE space
  STE: forward argmin, backward identity
  L = recon + codebook-loss + commitment-loss  (PathVQ: cosine recon + codebook term)

RESIDUAL VQ (MSVQ)
  r0 = e;  z_k = argmin_j ||ℓ2(r_{k-1}) − Ⅎ(vj)||;  r_k = r_{k-1} − v_{z_k}

MIL
  ABMIL: a = softmax(wᵀ(tanh(Vh)⊙sigmoid(Uh))); z = Σ a_i h_i

PATHVQ numbers
  C = 8192 codes, d = 16, 1024→16 = 64× compression
  scales {1×1, 2×2, 4×4, 7×7, 14×14}, shared codebook
  VQ data: 1M TCGA tiles, 4×3090, 50 ep, 22 h
  SSL: 3584² regions → ~250k regions, 256 tiles/region, indices ~65 MB
  conv stack: 4 layers stride 2, 128→256→512→1024 → (N,1,1,1024)
  BRACS F1: 0.692 → 0.730 → 0.747 (ABMIL → +PathVQ → +SPT)
  LGG-GBM F1: 0.685 → 0.723 → 0.752
```

---

## Timeline (3-day version)

| Day | Morning (2 h) | Evening (1.5 h) |
|-----|---------------|-----------------|
| 1 | Phase 1: attention (read) | Phase 1: numpy implementation + Gate 1 |
| 2 | Phase 2: ViT + Phase 3: VQ-VAE | Phase 3: implement VQ + residual VQ, Gate 3 |
| 3 | Phase 4 (MIL) + Phase 5 (SSL) + Phase 6 (pathology) | Phase 7 Pass 1–2 (method) |
| 4 | Phase 7 Pass 3–4 (experiments + critique) | Phase 8 rehearsal |
| 5 | redo failed gates | — |

If you only have **one day**: Phase 1 (4 h) → Phase 3 (4 h) → Phase 7 Pass 1 (30 min) →
Phase 8 with the QA bank (1.5 h). That gets you "I can explain attention and quantization, and
here is what the paper does."

---

## A note on the paper itself

PathVQ is a **well-motivated engineering paper with a clean, standard method** — VQ-VAE +
residual VQ + BEiT-style MIM, all applied to pathology patch tokens. There is no exotic math
beyond what is in Phases 1–5. The paper's real contribution is the *framing* (the `[CLS]`
information bottleneck at WSI scale) and the *empirical* demonstration that compressing patch
tokens beats scaling the backbone. If you master the five phases above, you will be able to
answer every methodological question about it — and you will be able to say clearly which parts
are theirs and which parts are borrowed from VQ-VAE / BEiT / ABMIL, which is exactly what a
sharp examiner wants to hear.
