# cBioPortal Data for `image/TCGA-FJ-A871-01Z-00-DX5...svs`

Fetched via https://docs.cbioportal.org/web-api-and-clients/ (REST API https://www.cbioportal.org/api)

Mapping (as per `docs/s41467-024-55331-6.pdf:8` *Data availability* + *Datasets description*):

- **Study:** `blca_tcga_pan_can_atlas_2018` = TCGA BLCA PanCan Atlas 2018 (the exact study cited in paper)
- **Sample:** `TCGA-FJ-A871-01` (Primary Solid Tumor) maps to slide `TCGA-FJ-A871-01Z-00-DX5.8F79D0A8-5DE6-4159-AA77-61DACB21E867.svs`
- **Patient:** `TCGA-FJ-A871` (case `bb38805e-2ccc-4169-ba01-34b98adf5bd0` in `data/data.txt:291`)

## Files

| File                                                                 | Content                                                                              | Key finding for this slide                                                       |
| -------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| `cbioportal_TCGA-FJ-A871_summary.json`                             | Human-readable summary linking GDC ↔ cBioPortal ↔ paper terminology                | WT, RSEM 375.54                                                                  |
| `cbioportal_TCGA-FJ-A871_FGFR3_mutation.json`                      | `POST /mutations/fetch` with `entrezGeneIds:[2261]` + `sampleIds:[FJ-A871-01]` | `[]` → **FGFR3 WILD-TYPE** (paper: no activating hotspot)               |
| `cbioportal_TCGA-FJ-A871_FGFR3_expression.json`                    | `POST /rna_seq_v2_mrna/molecular-data/fetch`                                       | `value: 375.54` RSEM (paper uses TPM log2 scale)                               |
| `cbioportal_TCGA-FJ-A871_FGFR3_CNA.json`                           | `POST /gistic/discrete-copy-number/fetch`                                          | `[]` → no discrete GISTIC CNA call                                            |
| `cbioportal_TCGA-FJ-A871_mutations.json`                           | All 45 somatic mutations for sample (MSA schema)                                     | e.g. KRAS G12R, CTNNB1 S33F, CREBBP S1778*                                       |
| `cbioportal_TCGA-FJ-A871_patient_clinical.json`                    | Patient-level clinical attributes (AGE 49, STAGE III, T3B, Dead, etc.)               | Matches`data/data.txt:188-218`                                                 |
| `cbioportal_TCGA-FJ-A871_sample_clinical.json`                     | Sample-level (GRADE High Grade, TMB 1.5, MUTATION_COUNT 45)                          |                                                                                  |
| `cbioportal_TCGA-FJ-A871_sample_info.json` / `patient_info.json` | API sample/patient objects                                                           |                                                                                  |
| `cbioportal_blca_FGFR3_mutations_cohort.json`                      | All FGFR3 mutations in study (59 distinct patients, 68 records)                      | S249C 32, Y373C 8, G370C 5, R248C 3 – matches paper Supplementary Table 7 top-6 |
| `cbioportal_study_info.json` / `molecular_profiles.json`         | Study metadata (411 samples, profiles: mutations, rna_seq_v2_mrna, gistic…)         |                                                                                  |
| `cbioportal_TCGA-FD-A3NA_*.json`                                   | Same fetch for second case in`data/data.txt`                                       | Also WT, RSEM 2340.74                                                            |

## Interpretation vs Paper

- Paper Fig.1b TCGA n=307 MIBC (8% FGFR3-mut, ~24/307). Our cohort query returns 59/411 (14.4%) because it includes NMIBC (paper reports 42% in NMIBC I, 65% in NMIBC II). Rate 16.55% if counting records/411.
- Both `FJ-A871` and `FD-A3NA` are FGFR3 wild-type per cBioPortal; paper defines WT to include non-activating variants (H349D etc) as WT.
- Expression: 375.54 vs 2340.74 illustrates paper's observation that WTs can have high expression (false-positive risk, mean 4.28 log TPM).

## How to Reproduce

```bash
curl -s "https://www.cbioportal.org/api/molecular-profiles/blca_tcga_pan_can_atlas_2018_mutations/mutations/fetch?projection=DETAILED" \
  -H "Content-Type: application/json" -d '{"entrezGeneIds":[2261],"sampleIds":["TCGA-FJ-A871-01"]}' | python3 -m json.tool
```

See `/tmp/fetch_cbio.py` for Python `requests` / `bravado` alternative.

## Execution Commands

```Python
python scripts/fetch_cbioportal_data.py --image image/TCGA-FJ-A871-01Z-00-DX5.8F79D0A8-5DE6-4159-AA77-61DACB21E867.svs
python scripts/fetch_cbioportal_data.py --image-dir image/ --outdir data
python scripts/fetch_cbioportal_data.py --image-dir image/ --study auto --gene FGFR3
```
