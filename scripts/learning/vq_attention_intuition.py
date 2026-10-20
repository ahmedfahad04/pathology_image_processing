#!/usr/bin/env python3
"""
Intuition builder for the PathVQ study plan:  ATTENTION + VECTOR QUANTIZATION.

Run:
    python3 scripts/learning/vq_attention_intuition.py

Three parts, each printing numbers you can check by hand:

  Part 1  Single-head attention from scratch in NUMPY, verified against PyTorch.
          Shows Q/K/V, the raw logits, WHY the 1/sqrt(d_k) scaling matters
          (empirical logit variance = d_k), and multi-head attention.

  Part 2  A real VQ-VAE layer (the PathVQ Eq. 2 step) with the straight-through
          estimator: nearest-neighbour selection under L2-normalization,
          proof that the argmin == cosine argmax, proof that argmin kills the
          gradient and that STE restores it, codebook usage, and collapse.

  Part 3  Residual VQ (the mechanism behind MSVQ) on a synthetic 14x14 tile:
          1 code vs 2 vs 3 vs 4 -> reconstruction error drops. This is the idea
          behind MSVQ's {1x1, 2x2, 4x4, 7x7, 14x14} scale list.

CPU only, ~40 seconds.
"""

from __future__ import annotations

import math

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

np.random.seed(0)
torch.manual_seed(0)

LINE = "=" * 78


def h1(title: str) -> None:
    print(f"\n{LINE}\n{title}\n{LINE}")


def l2n(x, eps=1e-8):
    """L2-normalize along the last dimension (numpy or torch)."""
    if isinstance(x, np.ndarray):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)
    return x / (x.norm(dim=-1, keepdim=True) + eps)


# --------------------------------------------------------------------------------------
# PART 1 -- ATTENTION
# --------------------------------------------------------------------------------------
def part1_attention() -> None:
    h1("PART 1  --  SCALED DOT-PRODUCT ATTENTION  (by hand, in numpy)")

    n, d_model, d_k = 4, 8, 8
    X = np.random.randn(n, d_model)
    Wq = np.random.randn(d_model, d_k) * 0.3
    Wk = np.random.randn(d_model, d_k) * 0.3
    Wv = np.random.randn(d_model, d_k) * 0.3

    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    print(f"X {X.shape}   Q {Q.shape}   K {K.shape}   V {V.shape}")
    print("Q = X @ Wq ,  K = X @ Wk ,  V = X @ Wv\n")

    logits_raw = Q @ K.T
    logits_scaled = logits_raw / math.sqrt(d_k)

    def softmax(z):
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    attn = softmax(logits_scaled)
    out = attn @ V

    print("STEP 1  scores = Q @ K.T                  ->", logits_raw.shape, "(n x n)")
    print("STEP 2  scores / sqrt(d_k)  [d_k =", d_k, "-> divisor", round(math.sqrt(d_k), 3), "]")
    print("STEP 3  softmax over the ROW (over keys)  -> each row is a distribution")
    print("STEP 4  out = attn @ V                    ->", out.shape, "(convex combination of V rows)")
    print(f"\nrow sums: {attn.sum(axis=1).round(6)}")
    print("\nattention matrix (row = query, col = key):")
    print(np.array2string(attn, precision=3, suppress_small=True))

    # --- why the 1/sqrt(d_k) scaling matters -------------------------------------------
    print("\n--- WHY divide by sqrt(d_k)?  (empirical logit variance) ----------------------")
    print("q, k with i.i.d. N(0,1) components  ->  q.k has VARIANCE d_k, std sqrt(d_k)")
    for dim in (16, 64, 256, 1024):
        q = np.random.randn(4000, dim)
        k = np.random.randn(4000, dim)
        raw = (q * k).sum(axis=1)
        print(f"  d_k={dim:5d}   std(raw logit) = {raw.std():7.2f}  (predicted sqrt(d_k) ="
              f" {math.sqrt(dim):6.2f})   std(scaled) = {(raw/math.sqrt(dim)).std():.3f}")
    print("  -> unscaled logits blow up with head width; softmax saturates to one-hot,")
    print("     gradients vanish. Scaling keeps logits O(1) for every d_k.")
    p_raw = softmax(np.array([[0.0, 20.0]]))
    p_scaled = softmax(np.array([[0.0, 20.0]]) / math.sqrt(64))
    print(f"\n  softmax([0, 20])        = {p_raw.round(6)}   <- saturated, no gradient")
    print(f"  softmax([0, 20]/sqrt64) = {p_scaled.round(6)}   <- healthy")

    # --- verify against pytorch ---------------------------------------------------------
    if HAS_TORCH:
        h1("PART 1b -- VERIFY THE NUMPY IMPLEMENTATION AGAINST PYTORCH")
        tX = torch.tensor(X)
        tq, tk, tv = tX @ torch.tensor(Wq), tX @ torch.tensor(Wk), tX @ torch.tensor(Wv)
        ref = F.scaled_dot_product_attention(tq[None], tk[None], tv[None])[0].numpy()
        print(f"max |numpy - torch.scaled_dot_product_attention| = {np.abs(ref - out).max():.3e}")
        print("(matches => your understanding of the formula is correct)\n")

        h = 4
        d_head = d_model // h
        Wq_h = np.random.randn(d_model, h * d_head) * 0.3
        Wk_h = np.random.randn(d_model, h * d_head) * 0.3
        Wv_h = np.random.randn(d_model, h * d_head) * 0.3
        Wo = np.random.randn(h * d_head, d_model) * 0.3
        heads = []
        for i in range(h):
            q = X @ Wq_h[:, i * d_head : (i + 1) * d_head]
            k = X @ Wk_h[:, i * d_head : (i + 1) * d_head]
            v = X @ Wv_h[:, i * d_head : (i + 1) * d_head]
            heads.append(softmax(q @ k.T / math.sqrt(d_head)) @ v)
        mh = np.concatenate(heads, axis=1) @ Wo
        print(f"multi-head output shape {mh.shape}  (= (n, d_model))")
        print("Each head runs the SAME single-head math with its own projections;")
        print("the only difference is d_k = d_model / h. Multi-head != a different formula.")

    print("\n>>> PathVQ connection: the WSI-Transformer uses RoPE (rotary position embedding)")
    print("    because attention is permutation-equivariant. RoPE rotates Q and K by a")
    print("    position-dependent angle so q_i . k_j depends on the RELATIVE offset (i-j).")


# --------------------------------------------------------------------------------------
# PART 2 -- VECTOR QUANTIZATION
# --------------------------------------------------------------------------------------
def part2_vq() -> None:
    h1("PART 2  --  VECTOR QUANTIZATION  (PathVQ Eq. 2 + straight-through estimator)")

    n_patch, d_in, C = 16, 32, 16
    target = np.random.randn(n_patch, d_in).astype(np.float32)
    init = target[np.random.choice(n_patch, C, replace=False)].copy()
    codebook = init + 0.05 * np.random.randn(C, d_in).astype(np.float32)

    # ---------------------------------------------------- the quantization step (Eq. 2)
    enc = target.copy()
    norm_e, norm_v = l2n(enc), l2n(codebook)
    d2 = (
        np.sum(norm_e**2, axis=1, keepdims=True)
        - 2 * norm_e @ norm_v.T
        + np.sum(norm_v**2, axis=1)[None, :]
    )
    idx = np.argmin(d2, axis=1)
    print(f"codebook V {codebook.shape}   encoder outputs e {enc.shape}   C = {C}")
    print(f"selected indices z = {idx.tolist()}")
    cos_direct = np.sum(norm_e * norm_v[idx], axis=1)
    err = np.abs(2 - 2 * cos_direct - d2[np.arange(n_patch), idx]).max()
    print(f"\n||a_hat - b_hat||^2 == 2 - 2*cos(a,b)   ->  max identity error = {err:.2e}")
    print("=> the argmin over L2 distance on NORMALIZED vectors IS the argmax over COSINE.")
    print("   That is exactly why PathVQ L2-normalizes before Eq. 2.\n")

    # --------------------------------------------- why the hard argmin kills the gradient
    print("--- WHY THE HARD argmin CANNOT BE TRAINED ------------------------------------")
    if HAS_TORCH:
        cb_t = torch.tensor(codebook, requires_grad=True)
        e = torch.tensor(target, requires_grad=True)
        with torch.no_grad():
            idx_t = torch.argmin(
                ((l2n(e)[:, None, :] - l2n(cb_t.detach())[None]) ** 2).sum(-1), dim=1
            )
        # the `+ 0 * e` keeps `e` in the graph so we can read its gradient
        hard = cb_t[idx_t] + 0.0 * e.sum()
        hard.pow(2).sum().backward()
        print(f"  gradient reaching the ENCODER through a hard argmin : {e.grad.abs().sum().item():.3e}")
        print("  -> ZERO. argmin is non-differentiable: the encoder learns nothing.\n")

        # ------------------------------------------------- straight-through estimator
        e2 = torch.tensor(target, requires_grad=True)
        q = cb_t.detach()[idx_t]
        q_ste = e2 + (q - e2).detach()  # forward value = q, backward Jacobian = identity
        q_ste.pow(2).sum().backward()
        print(f"  gradient reaching the ENCODER through STE            : {e2.grad.abs().sum().item():.3e}")
        print(f"  forward value identical to the hard path            : {float((q_ste - q).abs().max()):.3e}")
        print("  -> NON-ZERO gradient, IDENTICAL forward pass.")
        print("\n  STE in one line:   q_ste = e + (q - e).detach()")
        print("  The codebook itself is updated by a separate codebook-loss term.\n")
    else:
        print("  (torch not installed - skipping the gradient demonstration)\n")


# --------------------------------------------------------------------------------------
# Synthetic "tile": a 14x14 grid of 32-dim patch features with spatial smoothness,
# which mimics a ViT's 196 patch tokens far better than iid noise.
# --------------------------------------------------------------------------------------
def make_tile(n=196, d=32, seed=0, n_regions=3):
    """A synthetic 14x14 tile of d-dim 'patch' features with spatial smoothness."""
    rng = np.random.default_rng(seed)
    side = int(math.sqrt(n))
    base = rng.normal(size=(n_regions, d)).astype(np.float32)
    grid = np.zeros((side, side, d), dtype=np.float32)
    for i in range(side):
        for j in range(side):
            band = (i * n_regions) // side          # horizontal bands of tissue types
            grid[i, j] = base[band] + 0.25 * rng.normal(size=d).astype(np.float32)
    return grid.reshape(n, d)


class VQLayer(nn.Module):
    """Minimal VQ-VAE layer with straight-through estimator (PathVQ Eq. 1-3)."""

    def __init__(self, d_in: int, zdim: int, C: int):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, zdim), nn.Tanh())  # PathVQ: 1024 -> 16
        self.dec = nn.Linear(zdim, d_in)
        self.codebook = nn.Embedding(C, zdim)                     # PathVQ: C = 8192
        self.C = C

    def forward(self, x, beta: float = 0.25):
        e = self.enc(x)                                           # (B, zdim)
        # Eq. 2: nearest codeword under L2-normalized distance == max cosine
        sim = l2n(e) @ l2n(self.codebook.weight).T
        idx = sim.argmax(dim=-1)                                  # (B,)
        e_q = self.codebook(idx)                                  # (B, zdim)
        e_q_ste = e + (e_q - e).detach()                          # straight-through
        recon = self.dec(e_q_ste)
        codebook_loss = ((e - e_q.detach()) ** 2).mean()           # moves the CODEBOOK
        commit_loss = ((e_q - e.detach()) ** 2).mean() * beta       # moves the ENCODER
        return recon, codebook_loss, commit_loss, idx, e


def train_vq(data, C=64, zdim=8, steps=1500, beta=0.25, codebook_lr=0.05, seed=0):
    torch.manual_seed(seed)
    model = VQLayer(data.shape[1], zdim, C)
    opt = torch.optim.Adam(
        [
            {"params": model.enc.parameters(), "lr": 1e-3},
            {"params": model.dec.parameters(), "lr": 1e-3},
            {"params": model.codebook.parameters(), "lr": codebook_lr},
        ]
    )
    t = torch.tensor(data)
    history = []
    for step in range(steps):
        recon, cb_loss, commit_loss, idx, _ = model(t)
        loss = ((recon - t) ** 2).mean() + cb_loss + commit_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 300 == 0 or step == steps - 1:
            with torch.no_grad():
                cos = F.cosine_similarity(model.dec(model.enc(t)), t, dim=-1).mean().item()
            history.append((step, loss.item(), cos, len(torch.unique(idx))))
    return model, history


def part2b_training() -> None:
    h1("PART 2b -- TRAINING A REAL VQ LAYER (VQ-VAE, as in PathVQ Sec. 2.2.1)")

    data = make_tile(n_regions=6)
    print(f"synthetic tile: 196 'patch' features of dim {data.shape[1]} (a 14x14 grid,")
    print(f"6 horizontal bands of 'tissue type' + noise)\n")

    model, hist = train_vq(data, C=32, zdim=8)
    print("step |   loss   |  cos(recon, target) | codes used")
    for step, loss, cos, used in hist:
        print(f"{step:5d} | {loss:8.4f} |      {cos:6.4f}      |   {used}/32")

    with torch.no_grad():
        recon, cb_loss, commit_loss, idx, e = model(torch.tensor(data))
        cos = F.cosine_similarity(recon, torch.tensor(data), dim=-1)
    print(f"\nfinal: mean cosine(reconstruction, original) = {cos.mean():.4f}")
    print(f"       codebook entries actually used       = {len(torch.unique(idx))}/32")
    print(f"       codebook loss = {cb_loss:.5f}   commitment loss = {commit_loss:.5f}")
    print("       (the tile only has 6 distinct tissue bands, so ~6-15 live codes is")
    print("        the right answer here -- the toy data is SIMPLE, not a real pathology tile)")
    print("\n  >>> this is EXACTLY PathVQ Eq. 3's structure: cosine reconstruction")
    print("      + a codebook term, trained with a straight-through path.\n")

    # ------------------------------------------------------------------ codebook collapse
    h1("PART 2c -- CODEBOOK COLLAPSE  (the classic VQ failure mode)")
    model_c, hist_c = train_vq(data, C=256, zdim=8, codebook_lr=0.5)
    with torch.no_grad():
        _, _, _, idx_c, _ = model_c(torch.tensor(data))
    used_c = len(torch.unique(idx_c))
    print(f"codebook C = 256 (far too big for 196 samples), codebook lr = 0.5")
    print(f"codes actually used = {used_c}/256   ->  {100*used_c/256:.0f}% of the vocabulary is DEAD")
    print("  (dead codes receive no gradient, so they can never recover)")
    print("\n  Mitigations used in practice: EMA codebook updates, re-initialising dead")
    print("  codes with k-means on recent encoder outputs, codebook restarts, and")
    print("  L2-normalised codes (as PathVQ does) to stop all encoders collapsing to")
    print("  a single direction.")


# --------------------------------------------------------------------------------------
# PART 3 -- RESIDUAL VQ  (MSVQ)
# --------------------------------------------------------------------------------------
def part3_residual() -> None:
    h1("PART 3  --  MULTI-SCALE VQ  (MSVQ, PathVQ Sec. 2.2.2 / Algorithm 1)")

    if not HAS_TORCH:
        print("  (torch required for this part)")
        return

    torch.manual_seed(0)
    side, d, C = 14, 16, 128
    # A synthetic 14x14 map of 16-dim features: smooth low-frequency background
    # + a couple of local "lesions". This mimics a ViT patch-token grid.
    low = torch.randn(1, d, 4, 4)
    grid = F.interpolate(low, size=(side, side), mode="bilinear", align_corners=False)
    grid = grid + 0.35 * torch.randn(1, d, side, side)
    grid[:, :, 5:8, 6:9] += 2.0                       # a local 'lesion'
    target = grid.clone()

    cb = nn.Embedding(C, d)
    nn.init.normal_(cb.weight, std=0.1)
    opt = torch.optim.Adam(cb.parameters(), lr=3e-3)

    SCALES = [(1, 1), (2, 2), (4, 4), (7, 7), (14, 14)]

    def to_vec(t: torch.Tensor) -> torch.Tensor:
        """(1, d, H, W) -> (H*W, d)"""
        return t.permute(0, 2, 3, 1).reshape(-1, d)

    def forward_pass(scale_list, collect=False):
        """MSVQ encode/decode for a given list of scales. Returns (recon, per-scale errs)."""
        f = target.clone()
        recon = torch.zeros_like(target)
        errs = []
        for (H, W) in scale_list:
            f_low = F.interpolate(f, size=(H, W), mode="area")
            sim = l2n(to_vec(f_low)) @ l2n(cb.weight).T
            assign = sim.argmax(dim=-1)
            z = cb(assign).reshape(1, H, W, d).permute(0, 3, 1, 2)
            up = F.interpolate(z, size=(side, side), mode="bilinear", align_corners=False)
            f = f - up                       # residual: remove what this scale explained
            recon = recon + up               # accumulate the reconstruction
            cos = F.cosine_similarity(recon.flatten(0, 1), target.flatten(0, 1), dim=-1)
            errs.append((1 - cos.mean()).item())
        return recon, errs

    print("training a SHARED codebook over all scales ...")
    for step in range(1200):
        f = target.clone()
        loss = 0.0
        for (H, W) in SCALES:
            f_low = F.interpolate(f, size=(H, W), mode="area")
            sim = l2n(to_vec(f_low)) @ l2n(cb.weight).T
            assign = sim.argmax(dim=-1)
            z = cb(assign).reshape(1, H, W, d).permute(0, 3, 1, 2)
            up = F.interpolate(z, size=(side, side), mode="bilinear", align_corners=False)
            f = f - up
            loss = loss + ((to_vec(f_low).detach() - to_vec(z)) ** 2).mean()
        loss = loss / len(SCALES)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print("done.\n")

    print("reconstruction error (1 - cosine) as scales are added, shared codebook:")
    print("  scales used                 |  error")
    for k in range(1, len(SCALES) + 1):
        used = SCALES[:k]
        _, errs = forward_pass(used)
        label = " + ".join(f"{h}x{w}" for h, w in used)
        print(f"  {label:<28} |  {errs[-1]:.4f}")

    _, all_errs = forward_pass(SCALES)
    _, one_errs = forward_pass([(1, 1)])
    print(f"\n  1x1 only (tile-level, ~ a [CLS])  -> error {one_errs[-1]:.4f}")
    print(f"  all 5 scales (MSVQ)              -> error {all_errs[-1]:.4f}")
    print("  => each scale removes the part of the residual it can explain.")
    print("     This is the mechanism behind PathVQ's Fig. 7 (MSVQ reconstructs better")
    print("     than single-scale VQ).")

    print("\n  The MSVQ loop (paper Algorithm 1) in one block:")
    print("     f = Enc(ST)                                  # (14, 14, 16)")
    print("     for k in scales:")
    print("         r_k = Q(interpolate(f, H_k, W_k))        # quantize the RESIDUAL at scale k")
    print("         z_k = lookup(codebook, r_k)              # indices -> embeddings")
    print("         f = f - upsample(z_k)                    # subtract what scale k explained")
    print("     recon = sum_k upsample(z_k)")
    print("  ONE shared codebook across scales => tile-level and patch-level tokens")
    print("  live in ONE vocabulary. That is what lets a single tokenizer serve as BOTH")
    print("  the feature compressor AND the source of free self-supervised labels.")


if __name__ == "__main__":
    part1_attention()
    part2_vq()
    if HAS_TORCH:
        part2b_training()
    part3_residual()
    print("\nDone. Go back to docs/PathVQ_STUDY_PLAN.md and clear Gates 1 and 3.")
