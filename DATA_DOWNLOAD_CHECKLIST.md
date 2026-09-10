# Data Download Checklist v4

All paths below are relative to the project root and match
`config/config.yaml`. Run `python scripts/00_setup/00_check_inputs.py`
to verify what is present.

## 1. Single-cell TISCH2

```text
data/raw/single_cell/HNSC_GSE103322/
data/raw/single_cell/OSCC_GSE172577/
data/raw/single_cell/LSCC_GSE150321/
data/raw/single_cell/NPC_GSE150430/
data/raw/single_cell/NPC_GSE162025/
data/raw/single_cell/THCA_GSE148673_outgroup/
```

Each folder must contain `CellMetainfo_table.tsv` plus an expression matrix
(`*expression*.h5` or `.tsv` or `.mtx`). The import script auto-detects.

## 2. TCGA-HNSC

```text
data/raw/tcga_hnsc/expression/TCGA-HNSC.star_counts.tsv
data/raw/tcga_hnsc/expression/TCGA-HNSC.star_tpm.tsv
data/raw/tcga_hnsc/clinical/TCGA-HNSC.clinical.tsv
data/raw/tcga_hnsc/clinical/TCGA-HNSC.survival.tsv
```

STAR TPM is used for classifier projection. STAR counts (log2-transformed and
filtered) is used for WGCNA.

## 3. DepMap

```text
data/raw/depmap/expression/OmicsExpressionProteinCodingGenesTPMLogp1.csv
data/raw/depmap/model_metadata/Model.csv
```

The preprocessor handles the 24Q2+ column format change.

## 4. Broad PRISM

Required:

```text
data/raw/prism_broad/secondary_screen/secondary-screen-cell-line-info.csv
data/raw/prism_broad/secondary_screen/secondary-screen-dose-response-curve-parameters.csv
data/raw/prism_broad/secondary_screen/secondary-screen-replicate-collapsed-logfold-change.csv
data/raw/prism_broad/secondary_screen/secondary-screen-replicate-collapsed-treatment-info.csv
data/raw/prism_broad/compound_metadata/Corsello_supplemental_tables.xlsx
```

Optional primary screen:

```text
data/raw/prism_broad/primary_screen/primary-screen-cell-line-info.csv
data/raw/prism_broad/primary_screen/primary-screen-replicate-collapsed-logfold-change.csv
data/raw/prism_broad/primary_screen/primary-screen-replicate-collapsed-treatment-info.csv
```

## 5. Reference network databases

These ship in `data/raw/networks/` (originally from the Macnetwork repo):

```text
data/raw/networks/9606.protein.links.v11.0.noscores.threshold400.txt
data/raw/networks/list_protein_to_gene.txt
data/raw/networks/COMPARTMENT_membrane_gene_names.txt
data/raw/networks/COMPARTMENT_nucleus_gene_names.txt
data/raw/networks/human_protein_interactome.txt
data/raw/networks/stitch_drugbank_target.v5.1.5_interactions_900_th_400.onlyTarget.Approved.tsv
data/raw/networks/ReactomePathways.gmt
```
