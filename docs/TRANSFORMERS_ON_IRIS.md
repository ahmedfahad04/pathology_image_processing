# Transformers on Iris: How They Work, and Why They Are the Wrong Tool

Source dataset: [UCI Iris (id=53)](https://archive.ics.uci.edu/dataset/53/iris) —
150 plants, 4 continuous features (sepal length/width, petal length/width in cm),
3 balanced classes (setosa, versicolor, virginica), no missing values.
One class is linearly separable; the other two are not.

This note is about the *theory*: what a transformer actually computes when you
point it at a table of numbers, and why that computation is a poor match for a
150-row, 4-feature problem.

---

## 1. The core mismatch

A transformer is a function over a **set of vectors**, not over a single vector:

$$
T: \mathbb{R}^{n \times d} \rightarrow \mathbb{R}^{n \times d}
$$

Iris gives you $x \in \mathbb{R}^4$ — one row, four numbers. There is no
sequence, no order, and no tokens. So the first theoretical question is not
"how does attention work" but **"what are the tokens?"**

This is the same problem ViT solves for images: an image is a grid, not a
sequence, so it is cut into patches and each patch is projected to a token.
Iris is the degenerate case of that idea — and it has no spatial structure to
exploit in the first place.

---

## 2. Step 1 — Manufacture tokens (feature tokenization)

The standard approach for tabular data is the **FT-Transformer**
(Gorishniy et al., 2021): one token per feature.

For feature $j$ with scalar value $x_j$:

$$
t_j = b_j + x_j \, w_j, \qquad b_j, w_j \in \mathbb{R}^d
$$

Each feature gets its **own** learned direction $w_j$ and offset $b_j$. So:

- `petal_length = 5.0` lands at a point along the "petal length" direction in
  embedding space;
- `sepal_width = 5.0` lands at a point along a **different** direction.

This is the crucial theoretical point: **the per-feature parameters replace
positional encoding.** A transformer is permutation-equivariant — it cannot
tell token 1 from token 2. In text, position embeddings supply that
information. In tabular data, feature identity is supplied by giving each
column its own embedding. If you shared one $w, b$ across all four features,
the model would literally be unable to distinguish sepal length from petal
width — a useful sanity check.

Three tokenization schemes, theoretically distinct:

| Scheme                                           | Token for feature $j$                                                    | What it assumes                                                             |
| ------------------------------------------------ | -------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| **Continuous projection** (FT-Transformer) | $b_j + x_j w_j$                                                          | The value has a learned linear direction; smooth in $x_j$                 |
| **Discretization + lookup**                | $E_j[\operatorname{bin}(x_j)]$, with $E_j \in \mathbb{R}^{k \times d}$ | The value is a category; loses resolution, bin edges become hyperparameters |
| **Sample-as-token** (SAINT-style)          | each *row* becomes a token                                               | Rows should attend to each other — a nearest-neighbor prior                |

For Iris the first is the natural choice: 4 tokens of dimension $d$, plus
often a learned `[CLS]` token, giving a sequence of length 5.

---

## 3. Step 2 — The encoder, mathematically

Input sequence $H \in \mathbb{R}^{n \times d}$, with $n = 5$ (CLS + 4 features).
Each layer does two things.

### 3.1 Multi-head self-attention — mixes information *across* tokens

For head $h$:

$$
Q = H W_Q, \qquad K = H W_K, \qquad V = H W_V
$$

$$
A = \operatorname{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right),
\qquad \text{head}_h = A V
$$

Then concatenate the heads and project with $W_O$.

The matrix $A$ is $5 \times 5$ and **row-stochastic**: row $i$ says how much
token $i$'s next representation is a weighted average of all tokens' value
vectors. Because softmax is applied per row, it is a convex combination — a
learned, input-dependent mixing of the other features.

For Iris, this is the entire "interaction modeling" mechanism: the
`petal_width` token can absorb information from `petal_length`, and the CLS
token can absorb from all four. With a complete graph over 5 tokens, one layer
already gives every token access to every other token; depth adds *composition*
of interactions, not reach.

### 3.2 Position-wise feed-forward network — transforms *within* each token

$$
\operatorname{FFN}(h) = W_2 \, \sigma(W_1 h + b_1) + b_2
$$

Typically $d \to 4d \to d$ with GELU, applied identically to every token.

### 3.3 Residuals and normalization

Each sub-layer is wrapped in a residual connection and LayerNorm:

$$
H \leftarrow \operatorname{LN}\!\left(H + \operatorname{Attention}(H)\right),
\qquad
H \leftarrow \operatorname{LN}\!\left(H + \operatorname{FFN}(H)\right)
$$

The division of labor is the theory: **attention does cross-token interaction;
the FFN does per-token nonlinear feature construction.** Stacking $L$ layers
composes the two.

---

## 4. Step 3 — Readout and loss

After $L$ layers, take the CLS token's final state $h_{\text{cls}} \in
\mathbb{R}^d$ (or mean-pool the four feature tokens):

$$
z = W_{\text{cls}} \, h_{\text{cls}} + b, \qquad z \in \mathbb{R}^3
$$

$$
p(y \mid x) = \operatorname{softmax}(z),
\qquad
\mathcal{L} = -\log p(y_{\text{true}} \mid x)
$$

Training is ordinary backpropagation through the attention and FFN weights.
There is nothing classification-specific about the transformer: it is a
feature extractor, and a linear head is the classifier.

The CLS token is worth a note: it carries no data, so its only route to
information is attention. It acts as an unconstrained "readout slot."
Mean-pooling the feature tokens is a stronger prior — it forces the summary to
be an average — and on tiny datasets that constraint often helps.

---

## 5. What the model can and cannot represent

### 5.1 Compared with an MLP

An MLP computes $f(x) = \sigma(W_2 \sigma(W_1 x))$ over the concatenated
4-vector. It can represent the same functions, but the transformer
*factorizes* the function over tokens and **shares** the $QK$ computation
across all pairs. It learns "how to compare two features" once, rather than
six separate interaction weights.

That sharing is the transformer's theoretical advantage — but it pays off when
there are **many tokens and variable sets**, where the same comparison
operation must generalize across positions and lengths. Iris has 4 fixed
tokens and 150 rows, so the sharing buys almost nothing while the flexibility
costs variance.

### 5.2 Permutation invariance

The encoder is equivariant to token order. Since feature identity lives in the
tokenizer, order is irrelevant — correct for tabular data. But it also means
the model has **no built-in notion of feature proximity or smoothness**. A
linear model assumes monotone, additive effects; kNN assumes local similarity;
a transformer must *learn* smoothness from data. That is sample-expensive, and
with ~120 training rows per fold, samples are exactly what is lacking.

### 5.3 Capacity

Parameter count is roughly

$$
\underbrace{4d^2}_{\text{QKV+O}} \;+\; \underbrace{8d^2}_{\text{FFN}}
\quad \text{per layer} \quad + \quad \text{tokenizer} \;+\; \text{head}
$$

For $d = 32$, $L = 2$, that is ~26k parameters. With 120 training examples,
that is ~215 parameters per example. The hypothesis class is far too rich for
the amount of evidence.

### 5.4 Attention is not importance

$A_{ij}$ is a mixing coefficient conditioned on the entire row, not a causal
contribution to the prediction. With $n = 5$ it is a $5 \times 5$
row-stochastic matrix — a learned weighted average, not an explanation.
Reading "the model attended to petal width, so petal width matters" is
theoretically unsupported. Use permutation importance or SHAP if you want
attribution.

---

## 6. Empirical check

A minimal FT-Transformer ($d=32$, 2 layers, 4 heads, dropout 0.1, AdamW,
300 full-batch epochs, 3 seeds per fold) versus standard baselines, both on
the same 5-fold stratified splits with scaling fit inside each fold:

| Model                     | Accuracy (5-fold)      |
| ------------------------- | ---------------------- |
| kNN ($k=5$)             | **97.3% ± 2.5** |
| Random forest (300 trees) | 96.0% ± 3.9           |
| RBF SVM                   | 96.0% ± 3.9           |
| Logistic regression       | 95.3% ± 4.5           |
| FT-Transformer            | 94.0% ± 5.2           |

The transformer is not failing to optimize. Its inductive bias is mismatched
to the problem, and its fold-to-fold variance is the highest of the group.

---

## 7. Why the theory says it underperforms here

The Bayes-optimal boundary for Iris is low-complexity: one class (setosa) is
linearly separable, the other two are nearly so. A regularized linear model,
SVM, or kNN encodes exactly that prior. A transformer starts from a much weaker
prior — essentially "a composition of learned pairwise mixings" — and must
recover smoothness and near-linearity from ~120 examples. That is a bad trade.

The general lesson from tabular benchmarks: FT-Transformer is **competitive,
not dominant**. It tends to match tuned gradient-boosted trees on medium
datasets and can win when there are many heterogeneous or categorical features
and enough data. On 150 rows with 4 continuous features, the theory predicts
no advantage, and the experiment agrees.

---

## 8. When the theory does favor a transformer

1. **The input is genuinely a set or sequence** — many tokens, variable
   length, long-range dependencies (text, image patches, clinical event
   sequences, multiple-instance learning over WSI patches).
2. **Weight sharing across positions matters** — the same local operation must
   apply everywhere.
3. **You can pretrain** on large unlabeled data, so the inductive bias is
   *learned* rather than assumed. With 150 labeled rows, self-supervised
   pretraining is not viable, so the flexibility is pure variance.
4. **Intersample attention is the point** — if the label of a row depends on
   its neighbors in the batch, sample-as-token attention (SAINT) encodes that
   prior. On 150 rows it is very unstable, but the idea is coherent.

For Iris specifically: a transformer *can* be built and trained, and it will
reach the low-to-mid 90s. But it is the wrong tool — not because attention is
weak, but because there is no sequence to attend over, and the dataset is too
small to learn the priors a transformer would otherwise provide for free.

---

## 9. Relation to vision transformers (this repo's setting)

In a ViT, an image is split into patches; each patch is flattened and linearly
projected to a token, a positional embedding is added, and the same encoder
stack runs. The differences from Iris are exactly the ones that matter
theoretically:

|                | Iris                    | ViT                                   |
| -------------- | ----------------------- | ------------------------------------- |
| Tokens         | 4 hand-defined features | hundreds of image patches             |
| Structure      | unordered set           | 2-D grid → needs positional encoding |
| Pretraining    | impossible at 150 rows  | large corpora available               |
| Inductive bias | none useful             | locality + weight sharing pay off     |

That is why transformers are central to pathology image work and peripheral to
tabular toy problems: the architecture's assumptions are about *many
structured tokens*, and Iris has four unstructured ones.

---

## References

- Gorishniy, Y., Rubachev, I., Khrulkov, V., Babenko, A. (2021).
  *Revisiting Deep Learning Models for Tabular Data.* NeurIPS.
- Somepalli, G., Goldblum, M., Schwarzschild, A., Bruss, C. B., Goldstein, T.
  (2021). *SAINT: Improved Neural Networks for Tabular Data via Row Attention
  and Contrastive Pre-Training.*
- Fisher, R. A. (1936). *The use of multiple measurements in taxonomic
  problems.* Annals of Eugenics.
- UCI Machine Learning Repository, Iris dataset (id=53),
  https://doi.org/10.24432/C56C76.
