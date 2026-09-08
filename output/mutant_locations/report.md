
# Where is the FGFR3 mutant? — Direct Answer

## 1) Which images *contain* a mutant?

**Short answer:** A mutation is a property of the *whole tumor sample* (the patient), not a single X,Y coordinate.
cBioPortal labels each SVS file’s donor sample as either **MUT** (activating hotspot — drug can help) or **WT** (normal).

### MUT images (the file_names that ARE mutant)
There are **76 MUT slides** in the current metadata snapshot (out of 552 slides / 269 unique samples).
Every one of these 33 MUT samples is listed in `WHICH_IMAGES_ARE_MUTANT.csv` (column `FGFR3_status==MUT`).
Top rows:

| file_name | hotspot | VAF | RSEM | stage | why it matters |
|---|---|---|---|---|---|
| `TCGA-CF-A5U8-01A-01-TSA.CB3B5182-B41C-4499-9769-1720760C914C.svs` | G370C/G372C | 0.45 | 12326 | nan | G370C/G372C is a rarer hotspot |
| `TCGA-CF-A5U8-01Z-00-DX1.D0385BD3-3128-41C6-889D-4EC916B2B228.svs` | G370C/G372C | 0.45 | 12326 | Stage II | G370C/G372C is a rarer hotspot |
| `TCGA-DK-AA6P-01A-01-TSA.E2677252-A0BA-4FB9-8F17-F53B14CABBA5.svs` | G370C/G372C | 0.60 | 2737 | Stage II | G370C/G372C is a rarer hotspot |
| `TCGA-E7-A7DU-01A-01-TSA.6C57A327-FD34-446C-B528-17AB52827881.svs` | G370C/G372C | 0.58 | 19293 | nan | G370C/G372C is a rarer hotspot |
| `TCGA-E7-A7DU-01Z-00-DX1.25C7F59E-1F03-47EF-9F17-DB9ADB16276E.svs` | G370C/G372C | 0.58 | 19293 | nan | G370C/G372C is a rarer hotspot |
| `TCGA-ZF-AA4U-01A-01-TS1.831DB569-494E-45A9-A8D1-4748BF3ED5F6.svs` | G370C/G372C | 0.52 | 19663 | Stage III | G370C/G372C is a rarer hotspot |
| `TCGA-ZF-AA4U-01Z-00-DX1.E7FBFD1C-E974-4B9C-8DA8-D14FEC22D117.svs` | G370C/G372C | 0.52 | 19663 | Stage III | G370C/G372C is a rarer hotspot |
| `TCGA-BT-A20T-01A-01-TSA.aa2eb218-f0b5-49fc-b812-83c09ce3a2fc.svs` | G380R/G382R | 0.42 | 21598 | nan | G380R/G382R is a rarer hotspot |

Full list: `MUT_list.txt` (76 lines) and `WHICH_IMAGES_ARE_MUTANT.csv` (552 rows, filter `FGFR3_status`).

### WT images (normal)
**506 WT slides** — the vast majority. The ONLY file actually present in `image/` is:

| file on disk | status | what it means |
|---|---|---|
| `TCGA-FJ-A871-01Z-00-DX5.8F79...svs` (1.1 GB) | **WT — no activating FGFR3 mutation** (`FGFR3_RSEM 375`, VAF n/a) | No mutant region exists to box. The whole tumor is WT. |

> **Why you see only WT on disk:** the 33 MUT SVS files were never downloaded. Their metadata is in `data/` (cBioPortal JSONs), but the 1-GB SVS binaries must be fetched from GDC via `GDC_fetch_MUT_images.sh`. Disk currently has 1 of 552 slides (0.2%).

## 2) Where, inside an image, is the mutant located?

> **Honest biology:** You CANNOT point to a single nucleus and say “this cell has the typo, that one doesn’t” by looking at H&E pink/purple. The typo is on chromosome 4 in tumor DNA, present diffusely. What *does* vary across the slide is **how mutant-like the morphology looks**:

| Paper Fig.5a (n=200+200 tiles) | MUT-predictive tiles | WT-predictive tiles | p |
|---|---|---|---|
| Monomorphic (uniform nuclei) | **18%** | 1% | p=4.3e-13 |
| Atypia | 27% | 12% | p=5e-3 |
| Desmoplastic stroma (scar) | 1% | **14%** | p=9.8e-13 |
| Lymphocytes (immune) | 0% | **12%** | p=2.9e-8 |
| Tumor cellularity | 99% | 57% | — |

In words (layman): **MUT area = tidy, dense tumor, uniform dots, little scar, no immune swarm.** WT area = scarred, inflamed, mixed nuclei. That is exactly what the paper’s MIL model heatmap paints red (MUT) vs blue (WT) — see `docs/s41467-024-55331-6.pdf:4` Fig.2e/f.

**Therefore:**
- In a **WT slide** (like your `FJ-A871`) there is **no red hotspot to draw — everywhere scores low** (our annotated thumbnail shows a blue grid).
- In a **MUT slide** (e.g. `TCGA-4Z-AA7Y…S249C` or `TCGA-4Z-AA83…S249C`), the red hotspot would sit **inside the densest tumor core**, not at the edge, at roughly **112×112 µm tile resolution** (≈224×224 px @1.0 MPP, paper Methods `scripts/analysis/analyze_fgfr3_mutations.py:133`). The simulated heatmap `simulated_heatmap_MUT_example.png` shows the pattern; the annotated WT thumbnail shows the null case.

## 3) How to get a REAL box/heatmap (not a simulation)

You need tile-level inference. The paper released code and weights:

```bash
git clone https://github.com/PABannier/fgfr3mut && cd fgfr3mut
# installs: U-Net matter detect (Dice 0.96), Bioptimus H0 1539-D, 125× MIL @1.0 MPP
pip install -r requirements.txt
python -m fgfr3mut.infer --slide ../image/TCGA-FJ-A871-01Z-00-DX5.*.svs --out ../output/mutant_locations/heatmap_FJ-A871.png --mpp 1.0
# For a MUT example (after GDC download):
# python -m fgfr3mut.infer --slide image/TCGA-4Z-AA7Y-01A-01-TS1.*.svs --out output/mutant_locations/heatmap_AA7Y.png
```

Output is a PNG heatmap with red (MUT) / blue (WT) overlay + tile scores — the only honest way to “annotate where the mutant is” at 112 µm resolution. Without running this (or an equivalent HoverNet+stroma+TIL surrogate), any box would be made up.

## 4) What the old graphs DID mean (one-sentence each)

- **Fig1 prevalence** — “Out of every ~10 patients, only ~1 is MUT; that rarity is why we need a pre-screen.”
- **Fig2 hotspots** — “Among those few MUTs, half are S249C — the one AI spots easiest.”
- **Fig3 expression** — “MUT genes are louder (high RSEM), but quiet MUTs and loud WTs confuse the AI.”
- **Fig4 stage** — “Later stage = less MUT (10–15%); that’s the sick group where skipping NGS saves time.”
- **Fig5 VAF** — “How pure the DNA typo signal is; 0.5 = half reads show it.”
- **Fig6 sex/age/TMB** — “Demographics don’t predict MUT (p>0.05); you can’t cheat with age.”
- **Fig7 histology** — “The VISUAL recipe: low variance nuclei + low scar + low TIL = MUT.”

## 5) Files you now have

| File | Use |
|---|---|
| `WHICH_IMAGES_ARE_MUTANT.csv` | Ground-truth label for every SVS (MUT/WT, hotspot, VAF, RSEM, stage) |
| `MUT_list.txt` / `WT_list.txt` | Plain lists for scripting |
| `annotated_TCGA-FJ-A871-01Z-00-DX5.png` | Real thumbnail of your disk image, stamped WT with explanation |
| `simulated_heatmap_MUT_example.png` | Educational fake heatmap mimicking paper Fig.2e (disclaimer inside) |
| `GDC_fetch_MUT_images.sh` | Commands to actually download the MUT SVS binaries |
| `../mutation_analysis/` Fig1-7 + `report.md` | Cohort statistics (unchanged) |

*Method refs: `data/data.txt:1` is GDC slide metadata (no FGFR3), cBioPortal PanCan 2018 `blca_tcga_pan_can_atlas_2018_mutations` is FGFR3 truth (`scripts/analysis/fetch_cbioportal_data.py:30`), tiling @1.0 MPP 224×224 µm (`docs/s41467-024-55331-6.pdf:9`).*
