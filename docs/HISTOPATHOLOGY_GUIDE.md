# Bladder Cancer Cohorts + Histopathology Guide (FGFR3 Paper, Plain Language)

Source: `docs/s41467-024-55331-6.pdf` (Bannier et al., Nat. Commun. 2024).

## 1. Cohorts in layman terms

Think of the bladder wall as soil. A tumor is a weed:

- **NMIBC** (non-muscle-invasive): weed on the surface, roots haven't reached
  the muscle. Scraped off, bladder kept (75% of patients).
- **MIBC** (muscle-invasive): roots into the muscle wall. Aggressive:
  bladder removal, chemo/radiation (25% of patients).
- **mUC** (metastatic): seeds escaped to distant organs. Palliative care;
  this is where the drug erdafitinib (needs FGFR3 mutation) is approved.

The **I / II** suffixes are just "first batch / second batch" from Erlangen
hospital — I = used for **training**, II = locked away for **validation**:

| Cohort | n (final) | FGFR3-mutant | Role |
|---|---|---|---|
| MIBC I | 236 (11%) | 11% | training |
| NMIBC I | 155 (42%) | 42% | training |
| Training total | **391 (23%)** | 23% | — |
| MIBC II | 183 (11%) | 11% | validation |
| mUC | 96 (13%) | 13% | validation |
| TCGA (public) | 307 (8%) | 8% | validation |
| Validation total | **586 (11%)** | 11% | — |
| NMIBC II | 97 (65%) | 65% | transferability experiment only |

Note the enrichment: real-world MIBC is only ~10–15% mutant, but training
was boosted to 23% so the model sees enough mutants. NMIBC is naturally
mutant-rich (up to 80% in low-grade papillary) — that is why adding NMIBC I
to training helped ("original sin" that worked, paper p.7).

## 2. Histopathological features considered (Fig. 5a)

A pathologist (M.E.), blinded to scores, examined 400 top-predictive tiles
(200 MUT + 200 WT, split by high/low FGFR3 expression) against 20 criteria
(Supplementary Methods). Six headline patterns, each tagged tumor-associated
(T) or non-tumorous (NT):

1. **Arteries** (NT) — blood vessels in surrounding tissue
2. **Atypia** (T) — how abnormal/irregular tumor cells look
3. **Desmoplastic stroma** (NT) — scar-like fibrous tissue the tumor provokes
4. **Inflammation lymphocytes** (NT) — immune cells attacking the area
5. **Mitosis** (T) — cells caught dividing
6. **Monomorphic appearance** (T) — tumor cells all looking alike (uniform,
   bland, regularly shaped nuclei)

Fig. 5b additionally compares top tiles from **primary vs metastasis** of the
same mUC patients: no significant histomorphological difference found.

## 3. MUT vs WT differences (all significant, z-test + Bonferroni)

FGFR3-mutant tiles look **calm, uniform, lonely**; wild-type tiles look
**chaotic, scarred, and under immune attack**:

| Pattern | MUT (n=200) | WT (n=200) | p |
|---|---|---|---|
| Monomorphic appearance (T) | **18%** | 1% | 4.3e-13 |
| Atypia (T) | **27%** | 12% | 5.0e-3 |
| Desmoplastic stroma (NT) | 1% | **14%** | 9.8e-13 |
| Inflammation lymphocytes (NT) | 0% | **12%** | 2.9e-8 |
| Tiles containing tumor cells | **99%** | 57% | — |

One-line memory hook: **mutant = boring uniform cells, no scar, no immune
fight; wild-type = messy cells, scar tissue, lymphocytes swarming.**

Caveats from the paper:
- Hotspot exception: **G380R** top tiles do NOT follow the pattern (Supp. Fig. 1).
- **Low-expressor mutants mimic WT**: mutants with low FGFR3 RNA look like
  wild-type and get missed; WT with high FGFR3 expression look mutant and
  cause false positives (3.26 vs 4.28 log-TPM, p = 7.9e-8). Removing both
  atypical groups lifts TCGA AUC 0.82 → 0.89.
- Fig. 2f (mUC cohort, 9 sampled top tiles each): same story — mutant regions
  monomorphous + low stroma; WT regions pleomorphic + desmoplastic.
