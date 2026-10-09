# Run every stage in order from the repository root.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$steps = @(
    "src/00_setup/00_check_inputs.py",
    "src/01_single_cell_qc/01_import_tisch_datasets.py",
    "src/01_single_cell_qc/02_standardise_cell_annotations.py",
    "src/01_single_cell_qc/03_filter_malignant_and_fibroblast_cells.py",
    "src/01_single_cell_qc/04_score_single_cells.py",
    "src/01_single_cell_qc/05_prepare_gse181919.py",
    "src/02_multinomial_pemt_model/01_define_training_labels.py",
    "src/02_multinomial_pemt_model/02_create_pseudobulk_mixtures.py",
    "src/02_multinomial_pemt_model/03_train_multinomial_classifier.py",
    "src/02_multinomial_pemt_model/04_export_classifier_coefficients.py",
    "src/02_multinomial_pemt_model/05_benchmark_classifiers.py",
    "src/03_sc_classifier_validation/01_puram_pemt_score_characterization.py",
    "src/03_sc_classifier_validation/02_apply_classifier_to_sc_datasets.py",
    "src/03_sc_classifier_validation/03_validation_figure.py",
    "src/03_sc_classifier_validation/04_umap_figure.py",
    "src/03_sc_classifier_validation/05_label_free_embedding.py",
    "src/04_tcga_projection/00_prepare_cptac_rnaseq.py",
    "src/04_tcga_projection/01_preprocess_tcga_expression.py",
    "src/04_tcga_projection/02_prepare_tcga_metadata.py",
    "src/04_tcga_projection/03_project_classifier_to_tcga.py",
    "src/04_tcga_projection/04_compute_pemt_specificity_score.py",
    "src/04_tcga_projection/05_make_tcga_trait_table.py",
    "src/04_tcga_projection/06_survival_analysis.py",
    "src/04_tcga_projection/07_emt_score_panel.py",
    "src/04_tcga_projection/08_clinical_associations.py",
    "src/04_tcga_projection/09_external_cohorts.py",
    "src/04_tcga_projection/12_cptac_validation.py",
    "src/04_tcga_projection/10_independent_validation.py",
    "src/04_tcga_projection/11_subgroup_analysis.py",
    "src/04_tcga_projection/14_survival_sensitivity.py",
    "src/04_tcga_projection/13_survival_meta_analysis.py",
    "src/05_wgcna/01_prepare_wgcna_expression.py",
    "src/05_wgcna/02_run_wgcna.py",
    "src/05_wgcna/03_module_trait_correlations.py",
    "src/05_wgcna/04_select_and_export_pemt_module.py",
    "src/05_wgcna/05_plot_module_trait_heatmap.py",
    "src/06_network_analysis/01_build_pemt_subnetwork.py",
    "src/06_network_analysis/02_permutation_centrality.py",
    "src/06_network_analysis/03_drug_proximity.py",
    "src/06_network_analysis/03_drug_proximity.py --exclude-seed-targets",
    "src/06_network_analysis/04_network_figures.py",
    "src/06_network_analysis/05_lincs_signature_reversal.py",
    "src/07_depmap_broad_prism_validation/01_preprocess_depmap_expression.py",
    "src/07_depmap_broad_prism_validation/02_project_classifier_to_depmap.py",
    "src/07_depmap_broad_prism_validation/03_prism_drug_sensitivity.py",
    "src/07_depmap_broad_prism_validation/04_crispr_dependency.py",
    "src/07_depmap_broad_prism_validation/05_moa_permutation_test.py",
    "src/09_gene_partition/01_pemt_gene_partition.py",
    "src/09_gene_partition/02_partition_replication.py",
    "src/09_gene_partition/01_pemt_gene_partition.py",
    "src/09_gene_partition/03_tyler_tirosh_benchmark.py",
    "src/10_cetuximab/01_cetuximab_cohort.py",
    "src/10_cetuximab/02_mlr_emt_states.py",
    "src/10_cetuximab/03_cetuximab_headtohead.py",
    "src/10_cetuximab/04_cetuximab_firth.py",
    "src/10_cetuximab/05_direct_arm_scores.py",
    "src/10_cetuximab/01_cetuximab_cohort.py --figure-only",
    "src/10_cetuximab/06_cetuximab_sensitivity_analyses.py",
    "src/10_cetuximab/07_purity_adjustment.py",
    "src/10_cetuximab/08_pseudobulk_construction_sensitivity.py",
    "src/10_cetuximab/09_pdx_cetuximab.py",
    "src/10_cetuximab/10_gdsc_cetuximab_cell_lines.py",
    "src/10_cetuximab/11_mlr_rnaseq_convention.py",
    "src/11_composition_and_mechanism/01_composition_simulation.py",
    "src/11_composition_and_mechanism/02_cptac_proteome_phospho.py",
    "src/11_composition_and_mechanism/03_caf_tumour_ligand_receptor.py",
    "src/11_composition_and_mechanism/04_malignant_arm_proximity.py",
    "src/11_composition_and_mechanism/05_lincs_random_controls.py",
    "src/13_revision/01_hartung_knapp_meta.py",
    "src/13_revision/02_core_vs_canonical.py",
    "src/13_revision/03_basal_centroid.py",
    "src/13_revision/04_saturation_logodds.py",
    "src/13_revision/05_gavish_meta_programmes.py",
    "src/13_revision/08_single_patient_scoring.py",
    "src/13_revision/09_pdx_basal_centroid.py",
    "src/13_revision/10_external_pemt_controls.py",
    "src/12_figures_and_tables/01_build_supplementary_tables.py",
    "src/12_figures_and_tables/02_combined_figures.py",
    "src/12_figures_and_tables/03_figure1_study_overview.py",
    "src/12_figures_and_tables/04_figure4_survival_and_scores.py",
    "src/12_figures_and_tables/05_extend_supplementary_tables.py",
    "src/12_figures_and_tables/06_number_supplementary_tables.py"
)
foreach ($s in $steps) {
    Write-Host "python $s"
    python @($s -split ' ')
    if ($LASTEXITCODE -ne 0) { throw "failed: $s" }
}

# Optional final stage: BayesPrism deconvolution with GSE181919 as the reference (src/13_revision/06 and 07).
# It needs R with BayesPrism (see README.md), and the TCGA-HNSC run takes about five hours on about 20 cores.
# R writes progress and warnings to stderr, so native errors do not stop the script here and the exit codes are checked instead.
$ErrorActionPreference = "Continue"
$hasBayesPrism = $false
if (Get-Command Rscript -ErrorAction SilentlyContinue) {
    & Rscript -e 'quit(status = as.integer(!suppressWarnings(suppressMessages(requireNamespace(''BayesPrism'', quietly = TRUE)))))' 2>$null | Out-Null
    $hasBayesPrism = ($LASTEXITCODE -eq 0)
}
if ($hasBayesPrism) {
    Write-Host "Rscript src/13_revision/06_bayesprism_deconvolution.R"
    & Rscript src/13_revision/06_bayesprism_deconvolution.R
    if ($LASTEXITCODE -ne 0) { throw "failed: src/13_revision/06_bayesprism_deconvolution.R" }
    Write-Host "python src/13_revision/07_bayesprism_analysis.py"
    python src/13_revision/07_bayesprism_analysis.py
    if ($LASTEXITCODE -ne 0) { throw "failed: src/13_revision/07_bayesprism_analysis.py" }
} else {
    Write-Host "Skipping the optional BayesPrism stage (src/13_revision/06 and 07). Rscript or the R package BayesPrism was not found. README.md lists the R dependencies."
}
