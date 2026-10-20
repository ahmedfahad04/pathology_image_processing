# PathVQ — Professor Question Bank

Companion to `docs/PathVQ_STUDY_PLAN.md`. Every answer should be given **out loud, in under 2
minutes, with the PDF closed**. Answers are written the way a strong candidate would say them:
first the direct answer, then the justification, then (if invited) the caveat.

Part A = the basics they asked you. Part B = methodology. Part C = VQ specifics. Part D =
experiments & numbers. Part E = critical thinking (this is where extra marks are).

---

## Part A — Basics (the questions that got you last time)

**A1. How is attention calculated?**
Three learned projections of the input: `Q = XW_Q`, `K = XW_K`, `V = XW_V`, each `R^{n×d_k}`.
Then `Attention = softmax(QKᵀ/√d_k) V`. `QKᵀ` is an `n×n` matrix of pairwise compatibility
scores; dividing by `√d_k` keeps the logits O(1) so the softmax doesn't saturate (each component
of `q·k` has variance `d_k`); softmax over each row turns that row into a probability distribution
over which tokens to read; multiplying by `V` gives a convex combination of the value vectors.
Multi-head just runs this `h` times with separate projections and concatenates. Add residual
connections, LayerNorm, and a position-wise FFN to get a block.

**A2. Why scale by `√d_k`?**
Because if `q` and `k` have roughly unit variance per component, then `q·k` (a dot product over
`d_k` dimensions) has variance `d_k` and therefore standard deviation `√d_k`. Dividing by `√d_k`
normalizes the logit magnitude back to ~1, so the softmax operates in its well-conditioned regime
regardless of head width. Without it, wide heads produce near-one-hot attention with vanishing
gradients. (Answering only "numerical stability" is half credit; mention softmax saturation.)

**A3. What is the output of one attention head?**
A weighted average of the value vectors, with weights given by that head's attention distribution
over positions. So it lies in the convex hull of the value vectors. Attention is a *soft* lookup.

**A4. What is a `[CLS]` token and what is it good for?**
A single extra learnable vector prepended to the token sequence. Through the layers it attends to
all other tokens, so at the end it holds a learned global summary of the input. In a ViT it gives
you one fixed-size vector per image/tile, which is why it is convenient as a slide-level instance
feature. The cost: producing that summary **mixes away** the per-patch detail, so any spatial
information you didn't explicitly keep is not recoverable from `[CLS]` alone.

**A5. How is quantization (VQ) calculated?**
You keep a learned codebook `V = [v₁..v_C] ∈ R^{C×d}`. An encoder maps a continuous feature `e`
to a continuous vector, and then the **discrete index** is chosen by nearest neighbour:
`z = arg min_j ‖ℓ₂(e) − ℓ₂(v_j)‖`. The decoder replaces `e` with the selected codeword
`v_z`. The paper L2-normalizes both, and the reason is that for unit vectors
`‖â − b̂‖² = 2 − 2 â·b̂`, so the argmin over distance is exactly the argmax over **cosine
similarity** — the codebook is effectively a set of 8192 semantic directions.

**A6. Why is quantization hard to train, and what fixes it?**
`arg min` is a discrete, non-differentiable operation — the gradient of the loss w.r.t. the
encoder output is zero. The fix is the **straight-through estimator**: in the forward pass use the
argmin, in the backward pass pretend the operation was the identity, so the encoder receives
`∂L/∂e_q` directly. Codebook vectors are updated by a separate codebook loss term.

**A7. What is a Transformer, in one sentence?**
A function over a set of vectors, built from alternating attention (token-to-token mixing) and
position-wise feed-forward layers (per-token transformation), with residual connections,
LayerNorm, and positional encoding to break permutation equivariance.

**A8. Why is a Transformer a reasonable fit for a WSI?**
Because a slide is an *unordered set* of instances and a tile is an *unordered grid* of patches;
attention handles variable-length sets with a constant path length between any two elements, and
MIL needs permutation-invariant pooling. The catch: attention is O(n²) and a slide can have 5k+
instances, hence the whole "Nyström attention", "long-context transformer", and "compress the
tokens" line of work — PathVQ sits in that last camp.

---

## Part B — Methodology

**B1. What is the problem PathVQ solves, in one sentence?**
Pathology foundation models are used on whole slides by taking the `[CLS]` token of each tile as
that tile's feature; this throws away the 196 spatial patch tokens per tile, and using all of them
would cost ~196× more storage and compute, so the field has been forced into a lossy
"one vector per tile" bottleneck that caps how much slide-level information the model can exploit.

**B2. Where does the "200×" number come from?**
From the token count, precisely: a 224×224 tile with 16×16 patches gives 14×14 = 196 patch
tokens; the standard approach keeps 1 `[CLS]` token instead. 196 patch tokens vs 1 → 196× more
tokens (rounded to ~200× in the paper), and since cost is roughly linear in token count, that's
~200× storage and training cost. With ~5,000 tiles per slide, keeping everything means ~1M patch
tokens per slide.

**B3. What is the core idea?**
Compress the patch tokens with **vector quantization** instead of discarding them: an MLP encoder
maps each 1024-d patch token to 16-d, a codebook of C=8192 entries snaps each one to its nearest
codeword, and the index is what you keep (plus a 16-d reconstruction). That is a 64× reduction in
feature dimension, and a decoder can inflate the 16-d codes back to 1024-d for reconstruction
training, which is how they show the compression is not destroying the information.

**B4. Walk me through the full pipeline.**
1. WSI → N tiles of 224×224.
2. Frozen FM (UNI / ViT-L) encodes each tile → `[CLS]` `s` (1024-d) + 196 patch tokens `ST`
   (1024-d each).
3. VQ encoder: `e = MLP(ST)`, 1024 → 16, two linear layers with tanh.
4. Quantizer: nearest codeword in L2-normalized space → index `z ∈ [0, 8192)`, embedding `e_z`.
5. Decoder: linear 16→768, three Transformer blocks, linear 768→1024; reconstructs `o`; trained
   with cosine similarity to the original `h` plus a codebook loss, using STE.
6. For downstream: the compressed grid `(N, 14, 14, 16)` is passed through 4 conv layers with
   stride 2 and channels 128→256→512→1024 → `(N, 1, 1, 1024)`, i.e. **one 1024-d vector per
   tile** — but this one is *spatially informed*, unlike `[CLS]`. It is then reshaped to look like
   the original tile-level feature and fed to a MIL model (ABMIL or WSI-Transformer).
7. Slide-level SSL: the level-1 (1×1) MSVQ index of each tile is a discrete label; the model
   predicts either the region's token histogram (ABMIL) or the masked tiles' indices (Transformer).

**B5. What exactly is the "progressive convolutional module"?**
Four stride-2 conv layers that collapse the 14×14 spatial grid of the decoded/quantized features
into a single 1×1 vector while expanding the channel dimension 128→256→512→1024. Its job is
"upsampling" in the channel sense and "downsampling" in the spatial sense: it converts
spatially-rich compressed features back into a per-tile summary that lives in the same
1024-d space the MIL model expects, but built from *all* the patch information rather than from a
single `[CLS]` vector.

**B6. Why does the conv module need pretraining / alignment?**
Because the VQ encoder's output is a mid-layer representation — the paper says the "encoding
direction of ST is not so controllable" since the features are from intermediate ViT layers. If the
conv stack is randomly initialized, the downstream feature space is meaningless relative to the
frozen FM's space. So during VQ training the conv output is aligned to the **level-0 tile feature**
(the original `[CLS]`-like tile embedding). Appendix Fig. 6 ablates this: random-init + fine-tune,
pretrain + fine-tune, and pretrain + freeze; pretraining+aligning is best.

**B7. What is SPT and why does it help?**
SPT = slide-level pre-training. The offline MSVQ tokenizer produces discrete labels for tiles
without any human annotation, so you can pretrain the slide-level model (ABMIL or 6-layer WSI
Transformer) on ~250k unlabeled TCGA regions. This acts like a large-scale, pathology-specific
pretraining stage, and it consistently adds on top of PathVQ (e.g. BRACS F1 0.730 → 0.747).
It also fixes the overfitting seen with the randomly-initialized Roformer on survival prediction.

**B8. Why is the tokenizer called *offline*?**
Because MSVQ is trained separately and then **frozen**; slide-level SSL only uses its stored
indices. Contrast with iBOT's *online* tokenizer, where a teacher network is updated by EMA
during pretraining and produces the MIM targets — those targets drift as training proceeds, which
the paper argues makes supervision unstable.

**B9. How is the WSI-Transformer given positional information?**
Rotary Position Embedding (RoPE), including a 2-D variant where the x- and y-coordinates of each
tile each contribute their own rotation. Coordinates of tiles are stored during preprocessing.
RoPE is used because it encodes *relative* offsets and extrapolates better than absolute learned
embeddings.

**B10. How is the model fine-tuned downstream?**
Batch size 1 (WSI-level), 20 epochs, AdamW, lr 1e-4, weight decay 1e-4. Both the conv module and
the MIL model are trained. For a transformer that was SSL-pretrained, **LoRA** with rank 16 on all
`nn.Linear` layers is used to limit overfitting; for a transformer trained from scratch, full
fine-tuning.

**B11. Why do you need LoRA here?**
Batch size 1 per slide and huge bags means little gradient signal per step; fully fine-tuning a
6-layer transformer on a few hundred slides overfits. LoRA freezes the pretrained weights and
learns low-rank updates (`ΔW = BA`, `r=16`), which is far fewer parameters and acts as a
regularizer.

**B12. What are the two SSL objectives exactly?**
*ABMIL variant (Eq. 4–5):* for a region, compute the soft target `q_c(x)` = the fraction of the
region's tiles whose level-1 MSVQ index is `c`, normalized over the 8192 classes. Pass the region
through ABMIL → attention pooling → classifier → softmax → `p_θ(x)_c`. Loss = cross-entropy
`−Σ_c q_c log p_c`. It is a soft-label, bag-of-words objective.
*Transformer variant (Eq. 6–7):* mask a subset of tiles in a region (96 of 256), replace them with
a learnable mask embedding, and predict each masked tile's level-1 MSVQ index with a softmax
classifier over the 8192 codes. Loss = cross-entropy over masked tiles. This is BEiT-style MIM
with a feature tokenizer instead of a pixel decoder.

**B13. Why is the *tile-level* (1×1) index used for SSL, while the *patch-level* is used for
compression?**
Because the slide-level model consumes one vector per tile; a tile-level index is a natural
discrete label for a tile, and a region of 256 tiles gives 256 labels for a 8192-way
classification (enough signal, and the frequency profile of the region is itself meaningful). The
patch-level (14×14) codes are the ones that carry spatial detail for downstream WSI modeling.

**B14. What is the "residual paradigm" in MSVQ?**
Residual quantization: at each scale, quantize the *remaining* (residual) feature, then subtract
the retrieved codeword from it, so the next scale only has to model what is left. Coarse scales
capture global/tile-level semantics, fine scales capture local detail, and a **single shared
codebook** across all scales means both live in one vocabulary.

**B15. What are the exact MSVQ scales and what does each degenerate to?**
`{1×1, 2×2, 4×4, 7×7, 14×14}`. With `R = (1×1)` only, MSVQ reduces to standard VQ applied to the
average-pooled patch-token representation — i.e. essentially the tile-level `[CLS]`-like feature.
With `R = (14×14)` only, it is exactly the patch-level VQ of §2.2.1. The full list interpolates
between tile-level and patch-level semantics.

**B16. Why is the decoder a Transformer and not an MLP?**
Because the tokens of a tile are spatially arranged and the reconstruction of one patch token
benefits from its neighbours (a cell's nucleus spans several patches). Self-attention over the
196 patch positions gives the decoder spatial context that a position-wise MLP would not.

**B17. What is the "reconstruction fidelity" claim, and how is it demonstrated?**
Figure 1: they train a decoder to reconstruct the full set of patch tokens from the compressed
representation and measure reconstruction quality. Using only the `[CLS]` token gives a large
reconstruction error (big information loss); the VQ-compressed representation reconstructs much
better. Figure 7 shows MSVQ reconstructs better than single-scale VQ. Figures 8/9 show
reconstruction is stable for codebook size 8k vs 16k and embedding dim 16 vs 32.

**B18. What are the stated limitations?**
(1) VQ learning needs extra training data (1M tiles) beyond the FM's own pretraining.
(2) VQ still loses some information, though the authors consider it acceptable.
(3) From A.6: the method does not surpass slide-level FMs trained with a Cox loss (TITAN, CHIEF),
because with WSI batch size 1 the Cox loss cannot be used; the paper plans to combine them.
(4) Scope: the comparison uses a single backbone (UNI) for the main ablations, and the
Cox-loss comparison is not apples-to-apples.

---

## Part C — VQ specifics (the deep ones)

**C1. Why L2-normalize before the distance?**
To make the nearest-neighbour search a *cosine* (directional) search. For unit vectors
`‖â−b̂‖² = 2 − 2·cos(θ)`, so minimizing the L2 distance is identical to maximizing cosine
similarity. This also removes the feature-norm nuisance, which matters because the MSE/cosine
reconstruction loss also cares about direction.

**C2. What is the difference between the codebook loss and the commitment loss?**
Codebook loss `‖sg[e] − e_q‖²`: moves the **codeword** toward the encoder output (the stop-grad
`sg` means the encoder is not updated by it). Commitment loss `β‖e − sg[e_q]‖²`: moves the
**encoder** toward its assigned codeword, preventing the encoder from drifting away and reducing
codebook churn. PathVQ's Eq. 3 shows the cosine reconstruction plus the codebook term and says
stop-gradient notation is omitted; the commitment term is not explicitly visible — a legitimate
question to ask, and one you can phrase as "is there an implicit commitment term or is β=0?"

**C3. PathVQ Eq. 3 is written as a `max`. Why?**
Because the codebook term enters with a negative sign: maximizing
`cos(o_i, h_i) − ‖ℓ₂(e_i) − ℓ₂(v_i)‖²` is equivalent to minimizing
`−cos(o_i, h_i) + ‖ℓ₂(e_i) − ℓ₂(v_i)‖²`, i.e. minimizing (negative cosine similarity) plus the
codebook distance. The objective is "make the reconstruction as cosine-similar as possible and the
codeword as close to the encoder output as possible", subject to the straight-through path.

**C4. How is the straight-through estimator implemented in code?**
```python
e_q = codebook[indices]                 # forward: nearest neighbour
e_q = e + (e_q - e).detach()            # backward: identity Jacobian
```
Forward value is `e_q`; gradient w.r.t. `e` is `∂L/∂e_q` unchanged. The codebook gradient comes
from the codebook loss term instead.

**C5. What is "codebook collapse" and how do you prevent it?**
Dead codes: some codewords are never selected and receive no gradient, so they never move; the
effective vocabulary shrinks. Mitigations: EMA-based codebook updates (replacing the codebook loss
altogether), re-initializing dead codes with k-means on recent encoder outputs, codebook
restarts, L2-normalized codes + small ε (as in the paper) to avoid the "all encoders collapse to
one code" failure mode, and lowering the codebook learning rate.

**C6. Why is a 16-dim codebook embedding said to be *better* than 32-dim?**
Lower-dimensional codes are easier to use: with a fixed codebook size C, a lower dimension gives
more effective capacity per code and improves codebook usage, and it compresses harder
(1024→16 is 64×; →32 would be 32×). The paper's own ablation (Fig. 9) shows reconstruction is
stable from 16 to 32, and it cites literature showing dims 8–32 improve codebook usage.

**C7. What is the actual bit-rate of the compression?**
Per patch token, PathVQ keeps either the 16-d quantized embedding (32 bytes at fp16) or the code
index alone (13 bits for C=8192 ≈ 1.6 bytes) — versus 1024-d fp16 = 2048 bytes for the original
token. The headline "64×" is a *dimension* ratio (1024/16); the *index* representation is
dramatically smaller still, which is what makes the 65 MB index dataset possible. Be precise
about which ratio you quote.

**C8. Why is 8192 a good codebook size for this task?**
It is large enough that 196 patches × many tiles populate the codes with genuinely distinct
semantic content, small enough that a 16-d codebook entry stays well-conditioned and the
classifier head over 8192 classes is trainable with 250k regions. Ablation (Fig. 8) shows 8k vs
16k is not sensitive, i.e. the method is robust to this choice in that range.

**C9. Residual quantization: why does it help reconstruction?**
Because a single codeword can only capture one "direction" of the 16-d residual space. Summing
K codewords gives a K-times richer approximation at the cost of K indices — a rate/distortion
trade where you can dial the rate by changing K (here: the number of scales).

**C10. In Algorithm 1, what is `r_k` and what is `z_k`?**
`r_k` is the *index map* at scale k — i.e. the residual is interpolated to resolution
`(H_k, W_k)`, quantized (`Q(...)`), and the result is stored as indices. `z_k = lookup(Z, r_k)`
turns indices back into codeword embeddings, which are then interpolated and subtracted from `f`
(`f ← f − z_k`). So `r` is discrete (indices) and `z` is continuous (embeddings) — the paper uses
`r`/`z` somewhat loosely, and the residual subtraction is written at the interpolated resolution.

**C11. Does the paper train the VQ on patch tokens, tile tokens, or both?**
Both, and that is the point of MSVQ: the 1×1 scale quantizes the average-pooled token map
(tile-level, semantically close to `[CLS]`), and the 14×14 scale quantizes the full patch grid.
The **shared** codebook is what makes the two comparable and lets one tokenizer serve both roles
(compression + SSL label generator).

---

## Part D — Experiments & numbers

**D1. Main results on BRACS (3-class WSI)?**
ABMIL + UNI: F1 0.692, AUC 0.875. +PathVQ: F1 0.730, AUC 0.902. +PathVQ +SPT: F1 0.747, AUC
0.906. Roformer + PathVQ + SPT is the best overall: F1 0.754, AUC 0.910. Δ over the UNI+ABMIL
baseline is +3.8% F1 / +2.7% AUC (and +4.8% / +1.9% on LGG-GBM mutation).

**D2. Mutation prediction (LGG-GBM, R132H)?**
F1 0.685 → 0.723 → 0.752; AUC 0.852 → 0.871 → 0.879 (ABMIL → +PathVQ → +SPT).

**D3. Survival prediction?**
C-index, 5-fold CV, NLL loss, all on UNI features. ABMIL reproduce → +PathVQ → +PathVQ+SPT:
BRCA 0.633 → 0.655 → 0.674; CRC 0.612 → 0.649 → 0.659; BLCA 0.540 → 0.608 → 0.616;
UCEC 0.671 → 0.721 → 0.748; KIRC 0.691 → 0.760 → 0.778. The paper reports Δ over ABMIL of
2.2% / 3.7% / 6.8% / 5.0% / 6.9%. Every cohort improves, and every cohort improves further
with slide-level pretraining. But PathVQ+SPT still does **not** beat the Cox-loss-trained slide
FMs (TITAN, CHIEF) on most cohorts — on survival, the *loss* is the binding constraint, not the
representation (Appendix A.6 says so explicitly).

**D4. What is the single most important comparison in the paper, and why?**
**UNI-2 + ABMIL vs UNI + ABMIL + PathVQ.** UNI-2 is a ViT-Giant backbone trained on far more
data; it improves over UNI by only ~0.6–1.4%. PathVQ, using a *smaller* model plus 1M tiles of
extra data, improves by 3.8–4.8%. Conclusion: the binding constraint was **information loss at
the `[CLS]` token**, not backbone capacity or data scale. This is the paper's thesis, expressed as
a number.

**D5. The Figure 5 trade-off study.**
PCA to a low-rank subspace → F1 ≈ 0.70 (preserves variance, drops discriminative detail).
Average pooling 196 → 4×4 or 7×7 → ≈ same as `[CLS]` (over-smoothing). `[CLS]` + average pooling
→ ~+1 point over `[CLS]` alone. Attention-based Top-K token selection: Top-64 ≈ 0.72 but still
computationally heavy; Top-16 ≈ 0.70 (loses too much). **VQ ≈ 0.73**, the best F1 with a much
lower token cost. Message: naive dimensionality reduction (linear or pooling) is *not* enough —
you need a learned, content-adaptive discretization that preserves what matters.

**D6. How expensive is the VQ stage?**
1M random 224×224 tiles from all TCGA diagnostic WSIs; frozen UNI ViT-L backbone; C=8192,
d=16; 4×RTX-3090; 50 epochs; batch 128 per GPU; lr 2e-4 with 5-epoch warmup then cosine decay to
1e-5; weight decay 1e-4; AdamW β=(0.9, 0.99); ~22 hours total. Cheap relative to the payoff.

**D7. How expensive is the slide-level SSL stage?**
TCGA WSIs cropped into 3584×3584 regions → ~250k regions (256 tiles of 224×224 per region);
the MSVQ indices for all tiles occupy only ~65 MB; 4 GPUs, 20 epochs, lr 5e-4 with 2 warmup
epochs then cosine to 1e-5, AdamW β=(0.9,0.98); ABMIL batch 64, Transformer batch 32 with 96 of
256 tokens masked; cross-entropy over 8192 categories (soft targets for ABMIL).

**D8. What is the ROI/tile-level result (Fig. 4)?**
On CRC-100K and Camelyon-17 WILDS tiles: fine-tuning only the **last block** of UNI improves
substantially over linear probing; PathVQ (which compresses and reconstructs the patch spatial
tokens) gives a comparable improvement to last-block fine-tuning; UNI-2 linear probe is *not*
consistently better. This supports the claim that the patch tokens carry information that the
`[CLS]` token does not.

**D9. What would you check first if PathVQ failed to reproduce?**
(1) Is the backbone actually frozen and are the tile features extracted with the same
preprocessing? (2) Is the conv module being pretrained/aligned to the level-0 tile feature, or is
it randomly initialized (Fig. 6 says this matters a lot)? (3) Are the MSVQ indices computed with
the *frozen* pretrained tokenizer, or were they recomputed with a still-training tokenizer?
(4) Is the reconstruction loss actually cosine similarity (a sign flip or MSE would break it)?
(5) Are the STE and codebook loss both present?

**D10. What are the exact trainable components in each stage?**
*VQ stage:* MLP encoder, codebook, Transformer decoder (and, for downstream, the conv module).
The FM backbone is frozen throughout. *SSL stage:* the conv module and the MIL/Transformer model,
with the VQ module frozen and used to re-embed indices. *Fine-tuning stage:* conv module +
MIL model, with LoRA (r=16) if the transformer was SSL-pretrained.

---

## Part E — Critical thinking (the answers that impress)

**E1. Is 64× compression really "preserving fidelity"?**
Be honest and precise: the reconstruction is *not* information-preserving in the information-
theoretic sense — a 16-d code over 8192 entries is a severe bottleneck for a 1024-d feature, and
the paper itself admits "the VQ still loses some information". What the paper demonstrates is
**task-relevant fidelity**: the decoder can reconstruct the patch-token grid well enough to
support downstream slide-level tasks, and the compressed features beat PCA, pooling, and Top-K
selection on the same downstream metric. So the correct claim is "preserves *task-relevant*
information at 64× lower dimension", not "lossless".

**E2. Isn't the 65 MB / 196-token story internally inconsistent?**
Do the arithmetic yourself. 250k regions × 256 tiles = 64M tiles. 65 MB / 64M ≈ **1 byte per
tile**, i.e. exactly one level-1 (1×1) index per tile — not 196 patch indices per tile (that
would be ~25 GB). So the 65 MB figure refers to the *tile-level* index set used by the SSL
objectives. Meanwhile §3.1 says the indices are re-embedded into a `(256, 14×14, 16)` feature
tensor, i.e. the *patch-level* grid, which cannot be stored that cheaply. The paper is
under-specified here; the honest answer is: "I read the 65 MB as the tile-level index set;
the `(256, 14×14, 16)` re-embedding must therefore be computed on the fly or cached differently,
and I would want the code or a clarifying answer from the authors."

**E3. What exactly is masked in the Transformer SSL objective — tiles or patches?**
Eq. 6–7 describe masking *tiles* ("a subset of tiles indexed by M", "96 masked tokens out of
256"), and the label `z_i` is "the MSVQ token index for the i-th masked tile" — the **level-1**
index. But the input feature tensor is `(256, 14×14, 16)`, i.e. per-tile patch grids. So the
natural reading is: each tile contributes 196 patch embeddings, whole tiles are dropped and
replaced by a learnable mask embedding, and the model predicts that tile's level-1 index. An
alternative reading is that only the level-1 (pooled) tile embedding is the Transformer token
and the 14×14 grid is used elsewhere. State your reading and cite the two sentences that support
it — that is a better answer than pretending the paper is unambiguous.

**E4. Where is the "200×" cost really spent?**
Storage and *pre-extraction* cost. The features are extracted once and stored, then the slide
model consumes them. 196× more tokens per tile means 196× more disk for cached features and
196× more compute in the slide-level model, and (for attention-based models) potentially
quadratic blow-up in the attention itself. PathVQ reduces tokens to 16-d per patch (64×) and then
collapses them back to one vector per tile via the conv stack, so the *slide-level* model sees
N tokens as before while the *information* per token is much richer than `[CLS]`.

**E5. What is the most likely failure mode of this approach in the real world?**
(1) **Codebook domain shift**: the codebook is trained on 1M TCGA tiles; a new scanner, stain
protocol, or tissue type may produce features that fall outside the well-used codes, so the
quantization becomes lossy exactly where it matters. (2) **The conv module is the real
bottleneck**: everything spatial must survive being squeezed from 14×14×16 to 1×1×1024 — the
paper's conv ablation (Fig. 6) shows this stage is not free. (3) **Frozen-FM dependence**: the
whole method inherits the biases of the tile encoder. (4) **Downstream-only validation of
"reconstruction fidelity"**: fidelity is measured by a decoder, but the downstream metric is what
matters, and the two can diverge.

**E6. If you had to criticize the experimental design in one sentence?**
The main claim (the `[CLS]` bottleneck) rests on a *single* backbone family (UNI) and a single
comparative data point (UNI-2), and the ablations for "what to do with patch tokens" (PCA,
pooling, Top-K, VQ) are run on BRACS only, so the generality of the trade-off curve is untested
on the other cohorts.

**E7. What would you try next?**
(1) Vary the number of scales K and the compression ratio to map the rate–distortion–utility
curve (the paper only ablates codebook size and dim). (2) Add a *learned* residual scale
schedule or a VQ-VAE-style EMA codebook to reduce dead codes. (3) Try a patch-level MIM
objective (predict 14×14 codes) instead of tile-level, to exploit the full spatial structure.
(4) Combine with a Cox loss (paper's own stated future work) by using prototype-based
pre-extraction to get batched samples. (5) Validate on a non-TCGA external cohort and on a
different tile encoder (CONCH / Virchow) to show the method is backbone-agnostic.

**E9. Have you spotted anything numerically inconsistent in the paper?**
Yes, one: in Table 1 the row "Δ over UNI + ABMIL" reports 4.8% for LGG-GBM F1, but the table
itself shows 0.685 → 0.723, which is a 3.8-point absolute gain (the other three Δ values, 3.8 /
2.7 / 1.9, *are* the absolute differences). It looks like a typo. It is a small thing, but being
able to say "I checked the arithmetic in the tables and found one inconsistency, here it is" is
exactly the kind of reading that separates a student who read the paper from one who skimmed it.

**E8. One-sentence summary you can end with.**
PathVQ shows that the pathology FM bottleneck is not model scale but the information destroyed by
collapsing each tile to a single `[CLS]` vector, and that a learned 64× discrete compression of
the patch tokens — used both as a compact WSI representation and as a source of free self-
supervised labels — recovers most of that loss at negligible cost, lifting slide-level
classification and survival prediction consistently over ABMIL and Roformer baselines and beating
much larger backbones trained on more data.

---

## Quick self-rating

| Score | Meaning |
|---|---|
| < 40 % | Re-do Phases 1 and 3 before reading again |
| 40–60 % | You can follow the paper; repeat the two hard gates (attention, quantization) |
| 60–80 % | You can defend the methodology; sharpen Part E |
| > 80 % | You can teach it. Try explaining PathVQ to your professor as if you had built it |
