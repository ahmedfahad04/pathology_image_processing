# Why 2024–2026 MIL Papers Skip Multi-Scale *Pixel* Feature Reconstruction — and What They Do Instead

**Scope:** the 46 papers in `MIL_WSI_papers_2024_2026 (1).xlsx` (2024–2026, whole-slide images / computational pathology).
**Review lenses:** Computation · Efficiency · Accuracy · Reliability.
**Companion artifact:** columns **K** (verified full-text source), **L** (scale actually followed) and **M** (why not multi-scale pixel reconstruction, in plain words) in the same spreadsheet, one row per paper.

---

## 1. The short answer

Recent MIL papers are **not** avoiding multi-scale. They are avoiding **pixel-level** reconstruction.

Two different things are routinely conflated:

| | "Multi-scale" | "Pixel-level reconstruction" |
|---|---|---|
| What it means here | Reading the slide at more than one magnification (20x, 10x, 5x) and fusing the results | Decoding features back into a dense, pixel-accurate map of the whole slide |
| Where it stops | A list of tile features per zoom level, pooled into one slide label | A gigapixel prediction map aligned to the slide's pixels |
| Supervision needed | Still only one label per slide | Pixel-level masks, or a dense region annotation |
| Cost | ~1 extra encoder pass per level, parallelisable | Proportional to slide area, per scale, per case |
| Present in 2024–2026 | Yes — MARBLE, MsaMIL-Net, MAPLE, ViLa-MIL, MSCPT, MoEMambaMIL, ABMILX | **One paper in the table** (Geo-MIL), and it is the exception, not the rule |

**The field now has a name for the mainstream.** FBTA (CVPR 2026) calls it the *"two-stage patch-to-bag paradigm"*:

> "WSIs are divided into patches (instances), from which instance-level features are extracted and cached using a frozen feature extractor f_FE. The frozen features are then aggregated by a trainable MIL network f_MIL for bag-level classification."

The literature's actual move is a **three-way substitution**. Instead of reconstructing the slide at pixel level across scales, it substitutes:

1. **More instances at the same scale** (tens of thousands of tiles per slide — MHIM-MIL Table 1 reports 8,108–14,364 patches/slide; ISBI 2025 reports ~12,000 on CAMELYON16; CLAM reports **24,714** for a lung resection and **41,802–51,426 at 40x**; Prov-GigaPath reports up to **70,121 tiles in a single Providence slide**).
2. **Richer aggregation over those instances** — attention (ABMIL/CLAM), transformers (TransMIL), state-space models (Mamba family), hypergraphs (WSI-HGMamba), tissue graphs (StructMIL), nuclei graphs (NGG-MIL), prototype dictionaries (PTCMIL, StructMIL, PANTHER), prompt-token clusters (PTCMIL), superpatches (SMMILe).
3. **A compressed spatial model instead of a pixel model** — clusters, prototypes, superpatches, nuclei graphs, tissue graphs. These are the closest the field gets to "reconstruction", and every one of them is built out of *tile-sized blocks* at *one zoom level*.

**And where "reconstruction" literally appears, it is in feature space, not pixel space.** PathVQ (NeurIPS 2025) trains a decoder to rebuild a tile's *196 spatial patch tokens* from a 16-dim quantised code; ReconMIL (arXiv:2603.19925) reconstructs a *task-specific manifold* in latent space. Neither ever returns to image space. That distinction is the single most useful answer to the question.

**One caveat to the headline:** pixel-level reconstruction is *not* unheard of at slide scale — it is rare, and the two precedents that matter are the two directions this project can use. Lerousseau et al. (MICCAI 2020) produced a **slide-wide pixel tumour mask from a slide-level label alone** via a MIL-derived proxy-label dense network, and **Geo-MIL (row 33 of this table, npj Digital Medicine 2026)** does the same for gastric cancer. Geo-MIL's own ablation is the cleanest proof of the trade-off in the whole literature:

> "MIL attention, optimized solely for discriminative classification, is insufficient for producing high-fidelity segmentations" — **Dice 0.708 with attention-only output vs 0.789 with a dedicated segmentation head, at essentially unchanged slide-level AUC (0.961).**

So: adding a *spatial head* (not a pixel-level *reconstruction*) is what buys spatial accuracy, and it does **not** cost slide-level accuracy. That is the cheapest, best-supported move available.

---

### 1.1 What each paper actually does, by scale paradigm

*Generated from the spreadsheet's own column L for 46 of 46 papers.*

| Scale paradigm | Papers |
|---|---|
| Multi-scale (token level, >=2 zoom levels) | 5 |
| Region / structure level (clusters, graphs, prototypes, superpatches) | 12 |
| Feature-space reconstruction (no pixel space) | 2 |
| Single-scale tile level (the mainstream) | 24 |
| Survey / benchmark (not a pipeline) | 3 |

| # | Year | Paper | Scale paradigm |
|---|---|---|---|
| 1 | 2024 | Towards a General-Purpose Foundation Model for Computational Pathology (UNI) | Single-scale tile level (the mainstream) |
| 2 | 2025 | FR-MIL — Distribution Re-Calibration-Based MIL with Transformer for WSI Classification | Single-scale tile level (the mainstream) |
| 3 | 2024 | MamMIL — MIL for Whole Slide Images with State Space Models | Region / structure level (clusters, graphs, prototypes, superpatches) |
| 4 | 2024 | E2-MIL — Explainable and Evidential MIL for WSI Classification | Region / structure level (clusters, graphs, prototypes, superpatches) |
| 5 | 2025 | Positional Encoding-Guided Transformer-Based MIL for Histopathology WSI Classification | Single-scale tile level (the mainstream) |
| 6 | 2024 | Advances in MIL for Whole Slide Image Analysis — Techniques, Challenges, and Future Directions | Survey / benchmark (not a pipeline) |
| 7 | 2024 | Rethinking Pre-Trained Feature Extractor Selection in MIL for WSI Classification | Single-scale tile level (the mainstream) |
| 8 | 2024 | MambaMIL — Enhancing Long Sequence Modeling with Sequence Reordering in Computational Pathology | Single-scale tile level (the mainstream) |
| 9 | 2024 | Mamba2MIL — State Space Duality Based MIL for Computational Pathology | Single-scale tile level (the mainstream) |
| 10 | 2024 | DGR-MIL — Exploring Diverse Global Representation in MIL for WSI Classification | Single-scale tile level (the mainstream) |
| 11 | 2024 | ViLa-MIL — Dual-Scale Vision-Language MIL for WSI Classification | Region / structure level (clusters, graphs, prototypes, superpatches) |
| 12 | 2025 | MSCPT — Few-Shot WSI Classification with Multi-Scale and Context-Focused Prompt Tuning | Region / structure level (clusters, graphs, prototypes, superpatches) |
| 13 | 2025 | HMIL — Hierarchical MIL for Fine-Grained WSI Classification | Single-scale tile level (the mainstream) |
| 14 | 2025 | Towards a Generalizable Pathology Foundation Model via Unified Knowledge Distillation (GPFM) | Single-scale tile level (the mainstream) |
| 15 | 2025 | AttriMIL — Revisiting Attention-Based MIL from a Perspective of Instance Attributes | Single-scale tile level (the mainstream) |
| 16 | 2025 | Focus Your Attention — MIL with Attention Modification for WSI Classification | Single-scale tile level (the mainstream) |
| 17 | 2025 | A Cluster Attention-Based MIL Network for Enhancing Histopathological Image Interpretation | Region / structure level (clusters, graphs, prototypes, superpatches) |
| 18 | 2025 | SimMIL — A Universal Weakly Supervised Pre-Training Framework for MIL in WSI | Single-scale tile level (the mainstream) |
| 19 | 2026 | MIL Framework with Masked Hard Instance Mining for Gigapixel Histopathology (MHIM-MIL) | Single-scale tile level (the mainstream) |
| 20 | 2025 | Detection, Localization, and Staging of Breast Cancer Lymph Node Metastasis Using Selective Nei | Single-scale tile level (the mainstream) |
| 21 | 2025 | MsaMIL-Net — End-to-End Multi-Scale Aware MIL for Efficient WSI Classification | Multi-scale (token level, >=2 zoom levels) |
| 22 | 2025 | When MIL Meets Foundation Models — Advancing Histological WSI Analysis | Survey / benchmark (not a pipeline) |
| 23 | 2025 | MMsurv — Multimodal Multi-Instance Multi-Cancer Survival Prediction Model | Single-scale tile level (the mainstream) |
| 24 | 2025 | WeakSupCon — Weakly Supervised Contrastive Learning for Encoder Pre-training | Single-scale tile level (the mainstream) |
| 25 | 2026 | PTCMIL — MIL via Prompt Token Clustering for WSI Analysis | Region / structure level (clusters, graphs, prototypes, superpatches) |
| 26 | 2025 | MIL with Hierarchical Discrimination and Smoothing Attention (HDSA-MIL) | Single-scale tile level (the mainstream) |
| 27 | 2025 | Slide-Level Prompt Learning with VLMs for Few-Shot MIL in Histopathology | Single-scale tile level (the mainstream) |
| 28 | 2025 | ITJP — Image and Text Joint Prompts for Few-Shot WSI Classification | Region / structure level (clusters, graphs, prototypes, superpatches) |
| 29 | 2025 | MAPLE — Multi-Scale Attribute-Enhanced Prompt Learning for Few-Shot WSI Classification | Multi-scale (token level, >=2 zoom levels) |
| 30 | 2025 | Hypergraph Mamba for Efficient Whole Slide Image Understanding (WSI-HGMamba) | Region / structure level (clusters, graphs, prototypes, superpatches) |
| 31 | 2025 | A Survey of Pathology Foundation Model — Progress and Future Directions | Survey / benchmark (not a pipeline) |
| 32 | 2025 | Top-Down Attention-Based MIL for WSI Analysis (TDA-MIL) | Single-scale tile level (the mainstream) |
| 33 | 2026 | Geometric Multi-Instance Learning for Weakly Supervised Gastric Cancer Segmentation | Region / structure level (clusters, graphs, prototypes, superpatches) |
| 34 | 2026 | MILA-MIL — Mamba-Inspired Linear Attention MIL for WSI Survival Prediction | Single-scale tile level (the mainstream) |
| 35 | 2026 | End-to-End Multimodal MIL for Cancer Histopathology Classification with Dual-Attention Fusion | Single-scale tile level (the mainstream) |
| 36 | 2026 | MoEMambaMIL — Structure-Aware Selective State Space Modeling for WSI Analysis | Multi-scale (token level, >=2 zoom levels) |
| 37 | 2026 | Nuclear Graph-Guided MIL for Weakly Supervised Tumor Region Discovery | Region / structure level (clusters, graphs, prototypes, superpatches) |
| 38 | 2026 | Structure-Aware Generalization for Heterogeneous Histopathology via Prototype-Based MIL | Region / structure level (clusters, graphs, prototypes, superpatches) |
| 39 | 2026 | Weakly Supervised Contrastive Learning for Histopathology Patch Embeddings | Single-scale tile level (the mainstream) |
| 40 | 2026 | Improving Multiple Instance Contrastive Learning via Sparse Transformer for WSI Classification  | Single-scale tile level (the mainstream) |
| 41 | 2026 | A Multi-scale Linear-time Encoder for WSI Analysis (MARBLE) | Multi-scale (token level, >=2 zoom levels) |
| 42 | 2026 | ReconMIL — Latent Space Reconstruction with Bi-Stream Mamba for WSI Analysis | Feature-space reconstruction (no pixel space) |
| 43 | 2026 | Contrastive Cross-Bag Augmentation for MIL-Based WSI Classification (C²Aug) | Single-scale tile level (the mainstream) |
| 44 | 2025 | PathVQ — Reforming Computational Pathology Foundation Model for Whole Slide Image Analysis via  | Feature-space reconstruction (no pixel space) |
| 45 | 2025 | Revisiting End-to-End Learning with Slide-level Supervision in Computational Pathology (ABMILX) | Multi-scale (token level, >=2 zoom levels) |
| 46 | 2025 | SMMILe enables accurate spatial quantification in digital pathology using multiple-instance lea | Region / structure level (clusters, graphs, prototypes, superpatches) |

**Reading of the table:** multi-scale *is* present (a minority of papers, at token level); region-level structure is the field's most popular way of getting spatial information; **no paper in the table reconstructs the slide at pixel level**, and the two papers that use the word "reconstruction" do it in feature space, not image space.


---

## 2. The root cause: the label, not the compute

MIL was chosen *because* pixel-level supervision does not exist. ReconMIL states it plainly:

> "their massive size and **scarce pixel-level annotations** make fully supervised learning impractical due to the high cost and variability of manual delineation. Consequently, Multiple Instance Learning (MIL) has become the standard paradigm."
> — ReconMIL, arXiv:2603.19925, §1

Everything downstream follows. If the only ground truth is one label per slide, then:

- A pixel decoder has **no supervision signal** — nothing tells it which pixel is diseased.
- A *region* decoder has none either, so the field fakes one from the bag label (Geo-MIL's node-level "pseudo-segmentation loss"; SMMILe's "instance dropout" and MRF loss). This is exactly what E2-MIL and SMMILe identify as the reason localisation degrades: *"Instance-based MIL typically treats the bag label as a pseudo-label for all instances, leading to inaccurate supervision."* (SMMILe, Nature Cancer 2025)
- A dense output, if produced, is **emergent and post-hoc** — attention weights, upsampled — rather than learned and verifiable.

So "why no pixel-level reconstruction?" is really: **"why would you train a decoder whose output nobody can check?"** MS-CLAM (MedIA 2023) is blunt about the limit of the bag-label regime:

> "it is often difficult to obtain from such models accurate object localization ... as they are mainly designed for slide-level classification."

---

## 3. The four lenses

### 3.1 COMPUTATION — a pixel map is unaffordable by construction

A pixel-accurate map at 1 µm/px over a typical slide (MI-Zero, CVPR 2023: "each image can span up to 100,000 x 100,000 pixels in dimensions"; Mamba2MIL measures TCGA-NSCLC slides up to **191,352 × 97,078 px ≈ 1.9 × 10¹⁰ px**) is a 10-billion-element array *per scale, per case*. Even a single-channel uint8 map is ~19 GB. A 64-dim feature map is ~1.2 TB. Nobody decodes that at inference time.

The tile-level regime exists precisely because it collapses that number. CLAM (Nat BME 2021) reports **820 patches/slide** for a lung biopsy and **24,714 patches/slide** for a resection, and states the design rationale in one sentence:

> "Following this feature extraction, both training and inference can occur in the low-dimensional feature space instead of the high-dimensional pixel space. The volume of the data space is decreased **nearly 200-fold** and we can drastically reduce the subsequent computation required to train supervised deep learning models."

Put the two numbers side by side for a 100k × 100k slide at 20x: the patch bag is **~101 MB**; a 4-level uint8 dense map is **~53 GB** — about **500×**, and ~4×10⁵ times the MIL instance count. The zoom trade-off compounds it: 20x → 5x is **16× fewer tiles but 4× coarser**.

Quadratic attention is the second wall, and it is what drove the 2024–2026 method wave:

> "directly feeding the giga-pixeled WSIs with tens of thousands of instances into the Transformer usually leads to out-of-memory issues due to the quadratic complexity of self-attention" — MamMIL, arXiv:2403.05160

> "the computational complexity of Graph Transformer increases quadratically with the number of tiles (N), as demonstrated in both the FLOPs and GPU memory" — WSI-HGMamba, arXiv:2505.17457

The payoff of escaping that wall is quantified: MamMIL reports GPU memory **−65.5%** (CAMELYON16) and **−71.7%** (BRACS) vs TransMIL. The 2025–2026 papers are the direct descendants of that arithmetic.

FBTA (CVPR 2026) states the hardware ceiling bluntly:

> "WSIs potentially exceed 100,000x100,000 pixels at the commonly used 20x magnification, end-to-end training of the feature extractor and MIL network is infeasible on commodity GPUs (e.g., 24 GB RTX 3090)" … "**no existing method has achieved genuine end-to-end MIL training on 20x WSIs using a single 24 GB GPU.**"

**Where multi-scale shows up in the cost ledger**, it is always at token level and always parallelised, so the bill is set by the finest level. MARBLE (ISBI 2026):

> "many pipelines process a single magnification **for stability and efficiency**, leaving cross-scale dependencies unused. Second, transformer multi-scale fusion often incurs quadratic attention or multi-stage heuristics that hinder end-to-end training."

That sentence is the field's own answer to the question, stated by its most explicitly multi-scale paper.

### 3.2 EFFICIENCY — the frozen encoder is the reason, and the reason it is questioned

The two-stage recipe (extract once, cache, train only the head) exists because end-to-end backprop through an encoder over 10⁴–10⁵ patches is infeasible. SimMIL (arXiv:2505.06710) states it as the definition of the standard protocol:

> "Two-stage bag MIL: A learnable feature extractor is trained over the frozen feature extractor since **end-to-end training is computationally infeasible in MIL on WSIs**."

A frozen, cached, single-scale encoder is *cheap and reusable across dozens of experiments*. A multi-scale end-to-end pixel-level model is the most expensive possible design in this field, and 2025 shows it is not even reliable:

> "E2E learning with sparse-attention MIL performs poorly, falling below SOTA MIL methods using offline features extracted by ResNet-50" — ABMILX, NeurIPS 2025 (prior E2E attempts needed "64 V100 GPUs"; ABMILX's own fix costs ~10 RTX3090-hours)

The cost of the *encoder* dominates, and it scales with the choice of backbone (ABMILX Table 1, TCGA-BRCA subtyping AUC / training FLOPs): ResNet-50 83.80 / 2.12T → UNI 94.05 / 31T → GigaPath 94.39 / 114T. Multi-scale multiplies that by the number of levels, and a pixel-level decoder multiplies it again.

Note the counter-current: ISBI 2025 flags that freezing the encoder *hurts* — "This approach can hinder the aggregator's ability to identify 'key' patches" — and ABMILX shows an end-to-end ResNet beating foundation models. The direction of travel in 2025–2026 is *away* from frozen single-scale, but towards **end-to-end at tile level** (ABMILX), not towards pixel-level decoding.

### 3.3 ACCURACY — pixel-level output is not what raises slide-level accuracy

What actually moved the numbers in 2024–2026, with measured deltas:

| Lever | Measured effect | Source |
|---|---|---|
| Better frozen encoder | AUC 83.80 → 94.05 (ResNet-50 → UNI), same MIL head | ABMILX, NeurIPS 2025 |
| Hard-instance mining | 88.20 (CLAM) / 90.99 (ABMIL) → **95.8 / 96.6** | MHIM-MIL, arXiv:2509.11526 |
| Multi-scale fusion (coarse + fine vs either alone) | **+20.25 pp accuracy, +6.94 pp AUC** on PANDA | MARBLE, ISBI 2026 |
| Recovering discarded spatial tokens (feature space) | ~3% with UNI, ~1% with UNI-2 | PathVQ, arXiv:2503.06482 |
| **Pixel-level reconstruction** | **no measured slide-level gain anywhere in the table** | — |

Two pieces of counter-evidence matter more than the table:

- **Dense supervision is not what makes MIL work.** Max-Min (IEEE TMI 2021) benchmarks eight weakly-supervised-segmentation methods on GlaS reporting *both* image-level error and pixel-level F1, and finds weakly-supervised segmentation driven by classification loss yields "unsatisfying masks" — the methods "lack mechanisms for modeling explicitly non-discriminative regions and reducing false-positive rates". Lerousseau et al. (MICCAI 2020) phrase the underlying failure the same way: "max-pooling based MIL is shown to recall only very discriminative tumor patches, with very high slide-level performance but very poor pixel-level qualitative performance".
- **Most of the sequence is redundant.** MHIM-MIL masks 5% of the easiest instances *plus 70–90% random masking* and accuracy **improves** — direct evidence that "significant redundancy and substantial noise" dominates the slide. A pixel-level model would faithfully reconstruct that redundancy at full cost. This is the strongest single argument: *the pixels you would reconstruct are mostly not the pixels that carry the answer.*

**And the counter-case that opens the door.** Geo-MIL's ablation (row 33 of this table) isolates the effect of adding a *spatial objective* to a MIL model:

| Geo-MIL output mode | Slide-level AUC | Segmentation Dice |
|---|---|---|
| Attention-only (classification loss only, λ = 0) | 0.961 | **0.708** |
| Dedicated segmentation head (λ > 0) | 0.961 | **0.789** |

Spatial quality moves by 8 Dice points; slide-level accuracy does not move at all. SMMILe reports the same decoupling at scale: it "surpasses the second-best method (DSMIL) by only about 1% in spatial AUC" but "substantially outperforms DSMIL by **23.56%** in spatial [accuracy]" — and "the gain from spatially-explicit methods is large in spatial metrics while being ~0 in classification metrics."

Two forces pull the other way, and they explain why the mainstream did not move:

- **Slide-level accuracy is already saturated**, so a decoder buys little: SMMILe's benchmark of 9 methods over 8 datasets found "in binary and multiclass classification tasks, **most methods achieved a macro AUC score of over 97%** across all datasets".
- **The dense output is architecturally incompatible with the frozen-encoder design.** Because the encoder is frozen and only the aggregator trains, the only spatial signal available is an attention score — and ProtoPur (arXiv:2610.15669, Oct 2026) shows those scores "may not reliably identify class-specific local evidence".

So the conclusion is not "detail is useless" — it is **"detail pays off at token level, and at *region* level via a dedicated spatial head, but a full pixel-level reconstruction costs ~500× the storage and buys nothing on the metric the field reports."**

### 3.4 RELIABILITY — the decisive argument, and the risk it creates

Heat maps are the field's universal interpretability artefact, and they are not trustworthy explanations.

> "learned attention weights are frequently uncorrelated with gradient-based measures of feature importance, and one can identify very different attention distributions that nonetheless yield equivalent predictions" — Jain & Wallace, *Attention is not Explanation*, NAACL-HLT 2019

Pathology-specific confirmation, and — importantly — **the founding paper of the paradigm warns you not to trust it**. CLAM (Nat Biomed Eng 2021):

> "caution should be taken not to overly rely on the attention heatmaps with the expectation that they can serve as **pixel-perfect segmentation masks**; intuitively, the attention scores for each region in the slide are relative and simply represent the model's interpretation of which regions are more important (relative to others) in determining the slide-level prediction."

and, in the same paper:

> "Pathologist annotations are only for comparative analysis, **no pixel level annotations were used for training**."

From the strongest current spatial paper (SMMILe, Nature Cancer 2025):

> "attention maps generated by representation-based MIL approaches can only be qualitatively interpreted through manual analysis and are increasingly viewed as suboptimal for spatially aware processing" — SMMILe, Nature Cancer 2025

> "attention scores are **not directly correlated with the slide-level labels** and can be misled by confounding factors" — SMMILe, Nature Cancer 2025

That is *prediction–localisation decoupling*: a model can post a high slide-level AUC and still paint a wrong map. Which is the reliability trap the whole field is in — and which is precisely why nobody ships pixel-level maps: **there is no pixel-level ground truth to audit them against.** A pixel-accurate map that is confidently wrong is worse than an honest tile-weight map, because it invites clinical trust.

The whole field shares this weakness, and the 2024–2026 literature says so out loud: Geo-MIL notes "the localization maps generated by standard attention mechanisms are often incomplete and struggle to accurately segment diffuse or infiltrative tumor regions"; MPGR-Net (BioData Mining 2026) that "CAM-based methods often suffer from incomplete activation, tending to highlight only the most discriminative parts of lesions while ignoring their full spatial extent"; PBIP (CVPR 2025) the same; ProtoPur (arXiv:2610.15669) that instance-level supervision "is often derived from attention scores, model predictions, negative bags, or spat[ial priors]" rather than truth.

Two further reliability facts:

- **Cross-centre collapse is large and measurable.** MHIM-MIL Table 2 (TCGA-NSCLC, ResNet-50): CLAM AUC **90.52**; Table 5, same models on external CPTAC: CLAM **74.9** — a **~15.6-point drop** from a scanner/site change alone. Prov-GigaPath reports "a substantial performance drop ... on out-of-distribution samples"; HistoDomainBed (arXiv:2409.17063) finds "the level of performance for all algorithms drops on MIDOG22" with an 11% cross-domain F1 spread.
- **Instance-level MIL over-concentrates.** SMMILe: instance-based MIL "produces highly skewed attention maps, focusing only on a limited subset of high-discriminative regions within WSI, leading to a decreased recall rate for the relevant tissue regions" — i.e. even the *better* of the two families fails the reliability test, which is why SMMILe had to add superpatch MRF smoothing and instance dropout to fix it.

---

## 4. The exceptions, and what they prove

### 4.1 The slide-scale pixel exceptions (the important ones)

| Paper | Year | Granularity | Supervision | Dense output used for |
|---|---|---|---|---|
| **Lerousseau et al.**, *Weakly supervised MIL histopathological tumor segmentation* (MICCAI 2020, arXiv:2004.05024) | 2020 | **pixel, slide-wide tumour mask** | **slide label only** (pixel masks only for evaluation) | task prediction |
| **Geo-MIL** (row 33 of this table, npj Digital Medicine 2026) | 2026 | **pixel, slide-wide mask** | **slide label only**; expert masks used *only for evaluation* | task prediction |
| **PBIP**, *Prototype-Based Image Prompting* (CVPR 2025) | 2025 | pixel per tile | image-level labels | task prediction |
| **MPGR-Net** (BioData Mining 2026, PMC13505138) | 2026 | pixel per tile, multi-scale patch graph | image-level labels | task prediction |

Geo-MIL is the decisive precedent: *"We effectively bridge the gap between weak supervision and dense prediction, demonstrating that our model can generate accurate and coherent segmentation masks using only slide-level labels."* So slide-wide pixel masks from a bag label are **demonstrably possible** — they are simply not the default, and the table shows why (see 3.3).

### 4.2 The image-scale and fully-supervised neighbours

Pixel reconstruction is **routine at image scale** and only rare at slide scale — that is the scaling insight. HoVer-Net (MICCAI 2019) and DeepLIIF (CVPR 2022) produce per-pixel, per-nucleus instance masks over whole slides, but they are **fully supervised** (MoNuSeg/PanNuke/CoNSeP pixel masks) and operate on 256²/512² tiles that are stitched back together. The weakly-supervised image-level literature (TransCAM, MICCAI 2022) does pixel-level segmentation from a single image label — again, at 512², not 10¹⁰ px.

**So the jump from "image" to "slide" is the jump that breaks it**, and the 10¹⁰-element map is exactly why.

### 4.3 The region-level substitutes (what the field actually chose)

- **PANTHER** (CVPR 2024) — Gaussian-mixture prototypes over patch embeddings; produces a per-patch prototype map, explicitly *"post-hoc interpretability with the quantification and visualization of morphological prototype spread within the tissue"*. Not pixel.
- **SMMILe** (Nature Cancer 2025) — the strongest spatial result in the literature, and it *proves* that instance-level aggregation yields superior spatial quantification without compromising slide-level prediction, then achieves spatial AUC ≈ 90% and spatial F1 5–20 points above nine baselines. It stops at **superpatch (region) resolution, one zoom level** — because region resolution is what a pathologist can use and pixel resolution is neither affordable nor checkable.
- **ProtoPur** (arXiv:2610.15669, Oct 2026) — prototype-purity pseudo-labelling to derive instance-level labels from bag labels; a dense *patch-level* label map, explicitly not pixel.
- **MARBLE** (ISBI 2026) — genuine multi-level encoding with coarse-to-fine token fusion, linear time; stops at tokens, for the efficiency reason it states itself.
- **PathVQ** (NeurIPS 2025) — literally does reconstruction, but of *feature* tokens (64× compression, reconstruction fidelity measured in feature space). Proves how much information a `[CLS]` token discards; cannot produce a spatial map.
- **Geo-MIL**, **NGG-MIL** (IEEE SPL 2026) and **StructMIL** (npj Digital Medicine 2026) all deliberately choose a *readable* spatial product — a graph-guided mask, a nuclei-graph probability map, a prototype dictionary — over a raw heat map.

**Consensus reading:** the field has settled on **region-level reconstruction at one zoom level, trained from a bag label, and treated as an interpretability product rather than a measured output** — with Geo-MIL as the one paper that pushes into slide-wide pixel masks and shows the cost/benefit of adding a dedicated spatial head.

### 4.4 The most relevant new lead

**"From Spots to Pixels: Dense Spatial Gene Expression Prediction from Histology Images" (CVPR 2026)** — listed in the CVPR 2026 open-access index; the PDF was not retrievable at the time of this review, so it is recorded as a **lead, not a verified source**. If it delivers what its title claims, it is the closest published work to a pixel-level, multi-scale, spatially-explicit molecular readout from H&E — i.e. the direct analogue of what an FGFR3-from-H&E project would want. **Read it first.**

---

## 5. What this means for a slide-level-label pipeline (e.g. FGFR3 status from H&E)

Your pipeline is on the right side of this trade-off and can claim the gap explicitly:

1. **The label budget is slide-level** (one SNaPshot/cBioPortal call per sample). So any claim of *tile-level* genotype is a weakly-supervised claim and must be presented as one — the same epistemic position as the attention maps in this literature, and a reason to borrow SMMILe's instance-aggregation design rather than a plain attention map.
2. **Your 1.0 MPP / 224×224 tiling is a single-scale choice.** The literature's own statement (MARBLE) is that single-scale pipelines leave cross-scale dependencies unused, and multi-scale fusion gives real gains (+20 pp accuracy, +6.9 pp AUC on PANDA). If your tiles are 1.0 MPP, a 0.5 MPP (2× downsample, 4× fewer tiles) level is cheap: 4× fewer tiles at 1/4 the linear resolution, and it is the level that carries tissue architecture, which MARBLE's coarse-to-fine fusion exploits.
3. **The genuinely unoccupied niche is *molecular* endpoints at region level.** Almost every paper in the table predicts *morphology-derived* targets (subtype, grade, metastasis, survival). SMMILe gives you the spatial-quantification recipe. Geo-MIL, NGG-MIL and StructMIL give you the region/graph recipes. What no 2024–2026 paper in this table does is **weakly supervise a region-level predictor of a genomic endpoint, and validate the region map against something other than the bag label.** The nearest published work to a pixel-level molecular readout is **"From Spots to Pixels" (CVPR 2026)** — read it before claiming the niche is empty.
4. **Add a spatial head; do not attempt pixel-level reconstruction.** Geo-MIL's ablation is the template: a dedicated segmentation objective moved Dice 0.708 → 0.789 at **no cost to slide-level AUC (0.961 either way)**. That is the cheapest, best-supported improvement available, and it is what makes your per-tile genotype map defensible. SMMILe's superpatch MRF loss and instance dropout are the corresponding region-level regularisers.
5. **Validation is the bottleneck, not modelling.** The reliability section is the real constraint: with no pixel ground truth, any regional map you publish for FGFR3 must be validated *externally* (e.g. against the immunohistochemistry or FISH hotspot that defines the positive call, or against spatial transcriptomics), and reported with the same caution SMMILe applies. Papers that ship a map with no external check are the ones the field is right to distrust — and CLAM, the paper that started the heat-map convention, says so itself.
6. **Redundancy is your friend, not your enemy.** MHIM-MIL's finding that discarding 70–90% of instances *improves* accuracy means a dense multi-scale pixel reconstruction of the whole slide is likely to be mostly noise. A targeted region-level approach, with the per-tile CHOWDER scores you already compute, is better aligned with what actually carries the signal.

---

## 6. Evidence index

| Lens | Source | Key fact |
|---|---|---|
| Computation | Xu et al., *Prov-GigaPath*, Nature 630 (2024) — PMC11153137 | 70,121 tiles in one slide; 1.38B tiles / 171,189 slides; "a larger magnification will quadruple the processing time" |
| Computation | Fang et al., *MamMIL*, BIBM 2024 — arXiv:2403.05160 | "quadratic complexity of self-attention" → GPU memory −65.5% / −71.7% vs TransMIL |
| Computation | Lu et al., *WSI-HGMamba*, arXiv:2505.17457 | Graph-Transformer cost quadratic in tile count N (FLOPs **and** GPU memory) |
| Computation | *PathVQ*, arXiv:2503.06482 | "nearly 200× higher storage and training costs" for all 196 spatial tokens |
| Computation | Dwarampudi et al., *MARBLE*, ISBI 2026 — arXiv:2602.02918 | "many pipelines process a single magnification for stability and efficiency" |
| Computation | *FBTA*, CVPR 2026 | "no existing method has achieved genuine end-to-end MIL training on 20x WSIs using a single 24 GB GPU" |
| Computation | Lu et al., *CLAM*, Nat Biomed Eng 2021 — arXiv:2004.09666 | feature space "instead of the high-dimensional pixel space … decreased nearly 200-fold" |
| Efficiency | Song et al., *SimMIL*, arXiv:2505.06710 | "end-to-end training is computationally infeasible in MIL on WSIs" |
| Efficiency | Tang et al., *ABMILX*, NeurIPS 2025 — arXiv:2506.02408 | E2E sparse-attention MIL "falls below SOTA MIL"; ResNet-50 2.12T vs GigaPath 114T FLOPs |
| Efficiency | Tang et al., *MHIM-MIL*, arXiv:2509.11526 | TransMIL: 4× params, 3× train time, 4.5× memory of ABMIL; masking cuts time 20%, memory 50% |
| Accuracy | *MHIM-MIL*, arXiv:2509.11526 | 95.8 / 96.6 AUC; masking 70–90% of instances *improves* it → "significant redundancy and substantial noise" |
| Accuracy | *MARBLE*, arXiv:2602.02918 | coarse+fine beats either single scale: +20.25 pp acc, +6.94 pp AUC (PANDA) |
| Accuracy | *PathVQ*, arXiv:2503.06482 | spatial tokens ≈ +3% (UNI), ≈ +1% (UNI-2) |
| Accuracy | *Geo-MIL*, npj Digital Medicine 2026 — PMC12860791 | attention-only Dice 0.708 vs 0.789 with a segmentation head, AUC 0.961 either way |
| Accuracy | *SMMILe*, Nature Cancer 2025 — PMC12727519 | +1% spatial AUC but **+23.56% spatial accuracy** vs DSMIL; "most methods achieved a macro AUC over 97%" |
| Accuracy | Bulletin et al., *Max-Min*, IEEE TMI 2021 — arXiv:2011.07221 | WSL from classification loss → "unsatisfying masks" |
| Accuracy | Lerousseau et al., MICCAI 2020 — arXiv:2004.05024 | slide-wide pixel mask from a slide label; but max-pooling MIL "very poor pixel-level qualitative performance" |
| Accuracy | Tourniaire et al., *MS-CLAM*, MedIA 2023 — 10.1016/j.media.2023.102763 | "difficult to obtain from such models accurate object localization" |
| Reliability | Jain & Wallace, NAACL-HLT 2019 — aclanthology.org/N19-1357 | "attention weights are frequently uncorrelated with gradient-based measures of feature importance" |
| Reliability | *CLAM*, Nat Biomed Eng 2021 — arXiv:2004.09666 | "caution should be taken not to overly rely on the attention heatmaps with the expectation that they can serve as pixel-perfect segmentation masks"; "no pixel level annotations were used for training" |
| Reliability | *SMMILe*, Nature Cancer 2025 — PMC12727519 | attention maps "only qualitatively interpreted"; "not directly correlated with the slide-level labels"; instance-based MIL "highly skewed" |
| Reliability | *E2-MIL*, MedIA 2024 — 10.1016/j.media.2024.103294 | weak supervision → "lack robust uncertainty estimation of prediction results" |
| Reliability | *ProtoPur*, arXiv:2610.15669 (Oct 2026) | instance-level supervision "often derived from attention scores … or spatial priors" rather than truth |
| Reliability | Jahanifar et al., *HistoDomainBed*, arXiv:2409.17063 | "the level of performance for all algorithms drops on MIDOG22" |
| Reliability | *MHIM-MIL*, arXiv:2509.11526 | CLAM 90.52 (TCGA-NSCLC) → 74.9 (external CPTAC): ~15.6-point drop |
| Reliability | Kim et al., ISBI 2025 — arXiv:2408.01167 | freezing the encoder "can hinder the aggregator's ability to identify 'key' patches"; ConvNeXt-B < ResNet-50 on CAMELYON16 |
| Supervision | *ReconMIL*, arXiv:2603.19925 | "scarce pixel-level annotations make fully supervised learning impractical … MIL has become the standard paradigm" |
| Framing | *FBTA*, CVPR 2026 | names the mainstream the "two-stage 'patch-to-bag' paradigm" |
| Supervision | *PBIP*, CVPR 2025; *MPGR-Net*, BioData Mining 2026 — PMC13505138 | "obtaining pixel-level annotations for pathological images is extremely labor-intensive" |

---

## 7. Caveats

- Rows marked **abstract-only** in column K of the spreadsheet (E2-MIL, PEGTB-MIL, CAAMIL, HDSA-MIL, ITJP, MILA-MIL, NGG-MIL, MICL++) were verified against publisher abstracts only — their method detail and scale handling are less certain than the rest. FR-MIL's method detail comes from the open-access MICCAI-2022 precursor of the same method, with the TMI-specific addition noted but not read.
- **Row 42 (ReconMIL, arXiv:2603.19925) was withdrawn by its authors on 22 Jun 2026** over problems in its evaluation protocol (verified directly against the arXiv record and the v2 stub). Its numbers should be treated as unverified. It is quoted here only for its framing of the pixel-level-annotation problem.
- *"From Spots to Pixels: Dense Spatial Gene Expression Prediction from Histology Images"* (CVPR 2026) is recorded from the **CVF open-access index only** — the PDF was not retrievable at review time. It is a lead, not a verified source. Same for *H²-MIL* (AAAI 2022), whose full text could not be obtained.
- *MS-CLAM*, *Max-Min*, *E2-MIL*, *TDA-MIL* and *FBTA* were read via abstract or secondary sources where the full text was paywalled; quotations are limited to the passages recorded in column K of the spreadsheet.
- The dense-map byte counts (53 GB vs 101 MB, ~500×) and the "4× downsample → 16× fewer patches" figure are **arithmetic** from the cited slide dimensions and preprocessing definitions, not verbatim sentences in any one paper.
- Several leads commonly cited for this topic could not be located in any index and are therefore **not** used: *CLOCS* (not in CVPR 2022) and *SANFMIL* (not in arXiv/OpenAlex/Crossref/Europe PMC/Semantic Scholar/CVF/GitHub). Two venue claims were corrected from primary sources: *PANTHER* is **CVPR 2024** (not ICCV 2023) and *HGMIL* is **ICME 2024** (not CVPR).
- The taxonomy in §1.1 is generated from column L of the spreadsheet and covers all 46 papers.
