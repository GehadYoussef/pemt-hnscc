#!/usr/bin/env bash
# Run the HNSCC pEMT pipeline in order. Execute from the project root.
# RERUN_TRAINING=1 also reruns stages 00 to 02 and retrains the classifier.
# RUN_SUBNETWORK=1 also runs the STRING subnetwork and centrality null (comparison only).
set -euo pipefail

if [ "${RERUN_TRAINING:-0}" = "1" ]; then
python scripts/00_setup/00_check_inputs.py
python scripts/01_single_cell_qc/01_import_tisch_datasets.py
python scripts/01_single_cell_qc/02_standardise_cell_annotations.py
python scripts/01_single_cell_qc/03_filter_malignant_and_fibroblast_cells.py
python scripts/01_single_cell_qc/04_score_single_cells.py --scope all
python scripts/02_multinomial_pemt_model/01_define_training_labels.py
python scripts/02_multinomial_pemt_model/02_create_pseudobulk_mixtures.py
python scripts/02_multinomial_pemt_model/03_train_multinomial_classifier.py
python scripts/02_multinomial_pemt_model/04_export_classifier_coefficients.py
fi

# optional benchmark of alternative learners (needs data/processed/pseudo_bulk/pseudobulk_expression.tsv from stage 02)
python scripts/02_multinomial_pemt_model/05_benchmark_classifiers.py || true

# stage 03: single-cell validation (consistent scaling, pseudobulk level, Table S1 nodal status)
python scripts/03_sc_classifier_validation/01_puram_pemt_score_characterization.py
python scripts/03_sc_classifier_validation/02_apply_classifier_to_sc_datasets.py
python scripts/03_sc_classifier_validation/03_validation_figure.py
python scripts/03_sc_classifier_validation/04_umap_figure.py

# stage 04: TCGA (no double log, HPV, site fix, EMT score panel, clinical associations)
python scripts/04_tcga_projection/01_preprocess_tcga_expression.py
python scripts/04_tcga_projection/02_prepare_tcga_metadata.py
python scripts/04_tcga_projection/03_project_classifier_to_tcga.py
python scripts/04_tcga_projection/04_compute_pemt_specificity_score.py
python scripts/04_tcga_projection/05_make_tcga_trait_table.py
python scripts/04_tcga_projection/06_survival_analysis.py
python scripts/04_tcga_projection/07_emt_score_panel.py
python scripts/04_tcga_projection/08_clinical_associations.py
python scripts/04_tcga_projection/09_external_cohorts.py   # needs data/raw/external_cohorts (GEO series matrices + GPL annotations)

# independent validation against published programmes and the TCGA 2015 expression subtypes
python scripts/04_tcga_projection/10_independent_validation.py
# CPTAC-3 third external cohort, subgroup robustness, sensitivity, then the meta-analysis
python scripts/04_tcga_projection/12_cptac_validation.py
python scripts/04_tcga_projection/11_subgroup_robustness.py
python scripts/04_tcga_projection/14_survival_sensitivity.py
python scripts/04_tcga_projection/13_survival_meta_analysis.py

# stage 05: WGCNA (log2 TPM, primaries only, signed, scale-free power, PAM, enrichment-based selection)
python scripts/05_wgcna/01_prepare_wgcna_expression.py
python scripts/05_wgcna/02_run_wgcna.py
python scripts/05_wgcna/03_module_trait_correlations.py
python scripts/05_wgcna/04_select_and_export_pemt_module.py

# stage 06: drug proximity of the module genes (Guney 2016; DrugBank curated and STITCH targets; leave-one-anchor-out)   ~10 min
python scripts/06_network_analysis/03_drug_proximity.py
# optional, comparison only (not used in the results): STRING physical subnetwork and per-node centrality null   ~20 min
if [ "${RUN_SUBNETWORK:-0}" = "1" ]; then
python scripts/06_network_analysis/01_build_pemt_subnetwork.py
python scripts/06_network_analysis/02_permutation_centrality.py
python scripts/06_network_analysis/03_drug_proximity.py --disease key_proteins
fi
# strict variant: a drug's targets that are themselves seed genes are dropped before the
# distance is computed (Supplementary Table S21)
python scripts/06_network_analysis/03_drug_proximity.py --exclude-seed-targets
python scripts/06_network_analysis/04_network_figures.py
# LINCS signature reversal (needs network access to the SigCom LINCS API)
python scripts/06_network_analysis/05_lincs_signature_reversal.py

# stage 07: DepMap within lineage, PRISM (QC, per compound x model), CRISPR dependency
python scripts/07_depmap_broad_prism_validation/01_preprocess_depmap_expression.py
python scripts/07_depmap_broad_prism_validation/02_project_classifier_to_depmap.py
python scripts/07_depmap_broad_prism_validation/03_prism_drug_sensitivity.py
python scripts/07_depmap_broad_prism_validation/04_crispr_dependency.py
# mechanism-of-action classes against the cell-line permutation null (supersedes the Mann-Whitney table in 03)
python scripts/07_depmap_broad_prism_validation/05_moa_permutation_test.py

# stage 08: supplementary tables (xlsx + tsv) from the results above
python scripts/08_manuscript_tables/01_build_supplementary_tables.py
python scripts/08_manuscript_tables/02_overview_figures.py
python scripts/08_manuscript_tables/03_combined_figures.py
