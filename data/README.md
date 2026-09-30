# Data

Paths are relative to the repository root and match `config/config.yaml`.
`python src/00_setup/00_check_inputs.py` reports which inputs are present.

## Committed reference files (`data/references/`)

| File | Content | Source |
|---|---|---|
| `emt_scores/gs76_genes.tsv` | 76-gene EMT signature | Byers et al., Clin Cancer Res 2013 |
| `emt_scores/ks_tumor_signature.tsv` | tumour EMT signature used by the KS score | Tan et al., EMBO Mol Med 2014 |
| `emt_scores/hallmark_emt_genes.txt` | Hallmark epithelial-mesenchymal transition genes | MSigDB, Liberzon et al., Cell Syst 2015 |
| `schinke2022_egfr_emt_genes.txt` | EGFR-induced EMT signature | Schinke et al., Mol Cancer 2022 |
| `zhou2025_cetuximab_response_genes.txt`, `zhou2025_invGRN_59genes.txt`, `zhou2025_cetuximab_predictive_9genes.txt` | EGFR invasion genes, invasive gene network and cetuximab predictors | Zhou et al., Mol Cancer 2025 |
| `ourailidis2026_tumour_budding_28genes.txt` | tumour budding signature | Ourailidis et al., Genome Med 2026 |
| `TCGA_HNSC_4class_expression_subtype.csv` | TCGA-HNSC expression subtypes | Cancer Genome Atlas Network, Nature 2015 |
| `klinghammer2020_table1_cetuximab_RTV.tsv` | cetuximab relative tumour volume per xenograft, transcribed from Table 1 | Klinghammer et al., Oncotarget 2020 |
| `cptac_hnscc_clinical.tsv` | CPTAC-3 HNSCC clinical data and overall survival, one row per case | GDC clinical release and LinkedOmics |
| `cptac_gdc_star_manifest.tsv` | GDC file identifiers of the CPTAC-3 HNSCC STAR gene expression files | GDC |
| `ensembl_to_symbol_mapping.tsv` | Ensembl gene identifier to gene symbol | Ensembl |
| `string_physical_experimental_400_gene_edges.tsv` | STRING v11.0 physical interactions with experimental or curated-database evidence and combined score at least 400, as gene symbols | STRING, Szklarczyk et al., Nucleic Acids Res 2019 (CC BY 4.0) |
| `human_interactome_gene_symbols.edgelist` | human protein interactome as gene symbols | derived from `data/raw/networks/human_protein_interactome.txt` |
| `drug_targets_gene_symbols.csv` | STITCH drug-target pairs of approved drugs, ChEMBL identifier to gene symbol | derived from the STITCH 5 file in `data/raw/networks/` |

## Inputs to download (`data/raw/`)

### Single-cell data

Six datasets from TISCH2 (http://tisch.comp-genomics.org), each folder holding the expression
matrix (`<name>_expression.h5`) and the cell annotation (`<name>_CellMetainfo_table.tsv`):

```text
data/raw/single_cell/HNSC_GSE103322/
data/raw/single_cell/LSCC_GSE150321/
data/raw/single_cell/OSCC_GSE172577/
data/raw/single_cell/NPC_GSE150430/
data/raw/single_cell/NPC_GSE162025/
data/raw/single_cell/THCA_GSE148673_outgroup/
```

GSE181919 from GEO:

```text
data/raw/GSE181919/GSE181919_UMI_counts.txt.gz
data/raw/GSE181919/GSE181919_Barcode_metadata.txt.gz
```

### TCGA-HNSC

From the UCSC Xena GDC hub (STAR counts and TPM, clinical and survival tables) and the UCSC Xena
TCGA hub (legacy clinical matrix). HPV status is taken from the cBioPortal PanCancer Atlas study
(hnsc_tcga_pan_can_atlas_2018).

```text
data/raw/tcga_hnsc/expression/TCGA-HNSC.star_counts.tsv
data/raw/tcga_hnsc/expression/TCGA-HNSC.star_tpm.tsv
data/raw/tcga_hnsc/clinical/TCGA-HNSC.clinical.tsv
data/raw/tcga_hnsc/clinical/TCGA-HNSC.survival.tsv
data/raw/tcga_hnsc/clinical/TCGA-HNSC.hpv_status_pancanatlas_cbioportal.tsv
data/raw/tcga_hnsc/clinical/TCGA.HNSC.sampleMap_HNSC_clinicalMatrix
```

### Microarray cohorts (GEO series matrices and platform annotations)

```text
data/raw/external_cohorts/GSE41613_series_matrix.txt.gz
data/raw/external_cohorts/GSE65858_series_matrix.txt.gz
data/raw/external_cohorts/GSE65021_series_matrix.txt.gz
data/raw/external_cohorts/GPL570.annot.gz
data/raw/external_cohorts/GPL10558.annot.gz
```

### CPTAC-3 HNSCC

RNA-seq is downloaded from the GDC by `src/04_tcga_projection/00_prepare_cptac_rnaseq.py` into
`data/raw/cptac_rnaseq/`. Proteome and phosphoproteome from LinkedOmics
(http://www.linkedomics.org, CPTAC-HNSCC):

```text
data/raw/cptac_proteome/HS_CPTAC_HNSCC_Proteomics_TMT_Gene_level_Tumor.cct
data/raw/cptac_proteome/HS_CPTAC_HNSCC_Phosphoproteomics_TMT_site_level_Tumor.cct
```

### Patient-derived xenografts

GEO series GSE84713 and GSE183881, and the GENCODE v38 HGNC metadata:

```text
data/raw/pdx/GSE84713_series_matrix.txt.gz
data/raw/pdx/GSE183881_series_matrix.txt.gz
data/raw/pdx/GSE183881_expression-human-clsfy.summary.isoforms.TPM.tab.gz
data/raw/pdx/gencode.v38.metadata.HGNC.gz
```

### Cell lines

DepMap 24Q4 (https://depmap.org/portal/download):

```text
data/raw/depmap/expression/OmicsExpressionProteinCodingGenesTPMLogp1.csv
data/raw/depmap/model_metadata/Model.csv
data/raw/depmap/crispr/CRISPRGeneEffect.csv
```

PRISM Repurposing secondary screen (DepMap portal) and the supplementary tables of Corsello et al.,
Nat Cancer 2020:

```text
data/raw/prism_broad/secondary_screen/secondary-screen-cell-line-info.csv
data/raw/prism_broad/secondary_screen/secondary-screen-dose-response-curve-parameters.csv
data/raw/prism_broad/secondary_screen/secondary-screen-replicate-collapsed-logfold-change.csv
data/raw/prism_broad/secondary_screen/secondary-screen-replicate-collapsed-treatment-info.csv
data/raw/prism_broad/compound_metadata/Corsello_supplemental_tables.xlsx
```

GDSC1 fitted dose response, release 8.5 (https://www.cancerrxgene.org/downloads):

```text
data/raw/gdsc/GDSC1_fitted_dose_response_27Oct23.xlsx
```

### MLR-EMT reference files

From https://github.com/Cancer-Systems-Biology-Lab/EMT_Scoring_RNASeq:

```text
data/raw/mlr_reference/genes_for_EMT_score.txt
data/raw/mlr_reference/RelevantData.mat
```

### Networks

```text
data/raw/networks/9606.protein.links.v11.0.noscores.threshold400.txt
data/raw/networks/list_protein_to_gene.txt
data/raw/networks/COMPARTMENT_membrane_gene_names.txt
data/raw/networks/COMPARTMENT_nucleus_gene_names.txt
data/raw/networks/human_protein_interactome.txt
data/raw/networks/stitch_drugbank_target.v5.1.5_interactions_900_th_400.onlyTarget.Approved.tsv
data/raw/networks/ReactomePathways.gmt
```

STRING v11.0 (https://string-db.org), COMPARTMENTS (https://compartments.jensenlab.org), STITCH 5
(http://stitch.embl.de) and Reactome (https://reactome.org).

### DrugBank (licence required)

The DrugBank drug-target analysis needs an export of the DrugBank database made under a DrugBank
licence. The code reads two CSV files:

```text
data/raw/drug_targets/drugbank/drugbank_drug_target_flat_export.csv
    columns: drugbank_id, gene_name, bond_type, organism
data/raw/drug_targets/drugbank/drugbank_drug_summary_export.csv
    columns: drugbank_id, name, type, status_label, approved, withdrawn, investigational,
             atc_codes, atc_titles, category_titles, n_targets, moa
```

DrugBank content is not included in this repository.
