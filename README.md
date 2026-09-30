# pemt-hnscc

Analysis code for the article "Malignant and stromal partial EMT programmes and cetuximab outcome in
head and neck cancer" (Alkhatib et al., npj Precision Oncology).

The partial EMT (pEMT) gene list of Puram et al. (2017) is divided by the cell type that expresses
each gene in single-cell data. A classifier trained on single-cell pseudobulks scores the malignant
pEMT state in bulk tumours. Both parts of the gene list are then tested against survival in four
bulk cohorts and against outcome under cetuximab in patients and patient-derived xenografts. The
scripts in `src/` produce every figure and supplementary table of the article.

## Repository layout

```text
config/                   parameters, random seeds, gene signatures, dataset registry
data/
  README.md               where to download every input
  references/             small public reference files used by the code (committed)
  raw/                    downloaded inputs (not committed)
  processed/              intermediate files written by the code (not committed)
src/
  pemt/                   shared functions: configuration, plotting style, survival helpers,
                          MLR-EMT, Firth logistic regression, published gene sets
  00_setup/               input check
  01_single_cell_qc/      import and quality control of the single-cell datasets
  02_multinomial_pemt_model/  training labels, pseudobulk mixtures, elastic-net classifier
  03_sc_classifier_validation/  classifier in the single-cell datasets, label-free recovery
  04_tcga_projection/     bulk cohorts: projection, EMT score panel, survival, meta-analysis
  05_wgcna/               co-expression network of TCGA-HNSC
  06_network_analysis/    network proximity and LINCS signature reversal
  07_depmap_broad_prism_validation/  cell-line drug sensitivity and CRISPR dependency
  09_gene_partition/      partition of the pEMT gene list by cell type and its replication
  10_cetuximab/           cetuximab-treated patients, xenografts and cell lines, MLR-EMT states
  11_composition_and_mechanism/  composition simulation, proteome, ligand-receptor analysis
  12_figures_and_tables/  combined figures and the supplementary workbook
run_all.sh, run_all.ps1   run every stage in order
```

`results/` is created when the code runs. Figures are written to `results/figures/` under the
names used in the article (for example `Figure_6.png` and `Supplementary_Figure_8.png`), and the
supplementary tables to `results/tables/Supplementary_Data_1.xlsx`.

## Installation

Python 3.11 is required.

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

All inputs are public except the DrugBank export, which requires a DrugBank licence.
`data/README.md` lists every file, its source and the path it must be saved under.
`python src/00_setup/00_check_inputs.py` reports which inputs are present.

## Running

```bash
bash run_all.sh                  # on Windows: .\run_all.ps1
```

The scripts run in the order listed in `run_all.sh`. Each script can also be run on its own once
the outputs it reads exist, for example:

```bash
python src/10_cetuximab/01_cetuximab_cohort.py
```

Three scripts need network access. `src/04_tcga_projection/00_prepare_cptac_rnaseq.py` downloads the
CPTAC-3 RNA-seq files from the GDC, and `src/06_network_analysis/05_lincs_signature_reversal.py` and
`src/11_composition_and_mechanism/05_lincs_random_controls.py` query the SigCom LINCS API. All
parameters and random seeds are in `config/config.yaml`.

## Figures and tables

| Item | Script |
|---|---|
| Figure 1 | `src/12_figures_and_tables/03_figure1_study_overview.py` |
| Figure 2 | `src/09_gene_partition/01_pemt_gene_partition.py` |
| Figure 3 | `src/03_sc_classifier_validation/03_validation_figure.py` |
| Figure 4 | `src/12_figures_and_tables/04_figure4_survival_and_scores.py` |
| Figure 5 | `src/04_tcga_projection/10_independent_validation.py` |
| Figure 6, Supplementary Fig. 8 | `src/10_cetuximab/01_cetuximab_cohort.py` |
| Figure 7 | `src/11_composition_and_mechanism/02_cptac_proteome_phospho.py` |
| Supplementary Fig. 1 | `src/09_gene_partition/03_tyler_tirosh_benchmark.py` |
| Supplementary Fig. 2 | `src/05_wgcna/02_run_wgcna.py` |
| Supplementary Fig. 3 | `src/05_wgcna/03_module_trait_correlations.py` |
| Supplementary Figs. 4 and 5 | `src/03_sc_classifier_validation/04_umap_figure.py` |
| Supplementary Fig. 6 | `src/02_multinomial_pemt_model/05_benchmark_classifiers.py` |
| Supplementary Fig. 7 | `src/03_sc_classifier_validation/05_label_free_embedding.py` |
| Supplementary Fig. 9 | `src/11_composition_and_mechanism/03_caf_tumour_ligand_receptor.py` |
| Supplementary Fig. 10 | `src/04_tcga_projection/09_external_cohorts.py` |
| Supplementary Figs. 11 and 15 | `src/12_figures_and_tables/02_combined_figures.py` |
| Supplementary Fig. 12 | `src/04_tcga_projection/11_subgroup_analysis.py` |
| Supplementary Fig. 13 | `src/06_network_analysis/05_lincs_signature_reversal.py` |
| Supplementary Fig. 14 | `src/07_depmap_broad_prism_validation/05_moa_permutation_test.py` |
| Supplementary Figs. 16 and 17 | `src/06_network_analysis/04_network_figures.py` |
| Supplementary Fig. 18 | `src/04_tcga_projection/08_clinical_associations.py` |
| Supplementary Tables S1 to S49 | `src/12_figures_and_tables/01_build_supplementary_tables.py`, `05_extend_supplementary_tables.py` and `06_number_supplementary_tables.py` |

## Citation

Alkhatib DZR, Lunetto S, Chakraborty P, Jolly MK, Philpott M, Biddle A, Youssef G, Han N. Malignant
and stromal partial EMT programmes and cetuximab outcome in head and neck cancer. npj Precision
Oncology (in submission).

## Licence

The code is released under the MIT licence (see `LICENSE`). The reference files in
`data/references/` keep the terms of their sources, which are listed in `data/README.md`.
