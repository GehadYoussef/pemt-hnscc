# pemt-hnscc

Code for "Partial EMT in bulk head and neck cancer resolves into a prognostic stromal programme and a tumour-intrinsic EGFR-driven state".

The pipeline trains a three-class classifier on single-cell HNSCC pseudobulks, projects it into four bulk cohorts, and runs the co-expression, survival, network and perturbation analyses in the paper. Running it end to end reproduces every figure and supplementary table.

## Requirements

Python 3.11.

```
pip install -r requirements.txt
```

`lifelines` and `dynamicTreeCut` sometimes fail to build a wheel. If they do, `pip install --no-deps lifelines` and copy `dynamicTreeCut` from source into site-packages.

## Data

Raw data is not in this repository. It is public but large, about 3.7 GB, and the DepMap CRISPR gene effect file alone is 429 MB. Download it into `data/raw/` before running anything. `DATA_DOWNLOAD_CHECKLIST.md` lists every file and its expected path, and `python scripts/00_setup/00_check_inputs.py` reports what is present.

| Source | What to download | Where it goes |
| --- | --- | --- |
| GDC | TCGA-HNSC STAR counts and TPM, clinical, survival | `data/raw/tcga_hnsc/` |
| GDC | CPTAC-3 HNSCC RNA-seq and clinical | `data/raw/external_cohorts/cptac/` |
| GEO | GSE41613 and GSE65858 series matrices, GPL570 and GPL10558 annotations | `data/raw/external_cohorts/` |
| TISCH2 | The six single-cell datasets listed in `config/dataset_registry.tsv` | `data/raw/single_cell/` |
| DepMap 24Q4 | Expression, Model.csv, CRISPRGeneEffect.csv | `data/raw/depmap/` |
| PRISM | Secondary screen dose-response curves | `data/raw/prism_broad/` |
| STRING v11 | Physical links, protein-to-gene map | `data/raw/networks/` |
| DrugBank | Drug and target exports (licence required) | `data/raw/drug_targets/drugbank/` |

The published gene lists, the human interactome edge list and the TCGA subtype calls are small enough to include and are in `data/references/`.

## Running

```
bash run_pipeline.sh          # Linux, macOS, WSL
.\run_pipeline.ps1            # Windows
```

That runs stages 03 to 08. Expect about an hour and a half; the slow steps are drug proximity, which takes roughly ten minutes and runs twice, and the WGCNA correlation matrix. `RERUN_TRAINING=1` also reruns stages 00 to 02, which retrains the classifier and needs another 40 minutes.

Stage 06-05 needs internet access for the LINCS API. Everything else runs offline once the data is downloaded.

## Layout

```
scripts/00_setup                     input check
scripts/01_single_cell_qc            quality control and programme scoring
scripts/02_multinomial_pemt_model    training labels, pseudobulks, classifier
scripts/03_sc_classifier_validation  held-out single-cell validation
scripts/04_tcga_projection           bulk projection, EMT scores, survival, validation
scripts/05_wgcna                     co-expression network
scripts/06_network_analysis          drug proximity, LINCS reversal
scripts/07_depmap_broad_prism_validation   cell line sensitivity and dependency
scripts/08_manuscript_tables         supplementary tables and assembled figures
scripts/_lib                         shared helpers
config/config.yaml                   every parameter and random seed
data/references/                     small published gene lists
```

Every script has a docstring saying what it does, what it reads and what it writes. Figures are written to `manuscript/figures/` and the supplementary workbook to `manuscript/tables/`, both created on the first run. Each `save_figure` call writes the figure's final name in the paper; intermediate panels that are later assembled into a combined figure are written as `panel_*`.

## Notes

Classifier probabilities are cohort-relative, because projection z-scores the classifier genes within each cohort. Absolute proportions are not interpretable and the score cannot be computed for a single sample without a reference cohort.

Two analyses were run and then dropped, and the code for both is still here. A betweenness-based key-protein construct proved unstable to the choice of STRING evidence channel; it is gated behind `RUN_SUBNETWORK=1` and its output is Supplementary Table S9. A mechanism-of-action enrichment test used a null that treats compounds screened on the same cell lines as independent; `scripts/_lib/permtest.py` has the permutation test that replaced it.

## Licence

MIT. See LICENSE.
