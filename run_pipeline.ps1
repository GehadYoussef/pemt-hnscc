# Windows PowerShell equivalent of run_pipeline.sh. Run from the project root:
#   powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1
# Requires the packages in requirements.txt (pip install -r requirements.txt).
# $env:RERUN_TRAINING = 1 also reruns stages 00 to 02 and retrains the classifier.
# $env:RUN_SUBNETWORK = 1 also runs the STRING subnetwork and centrality null (comparison only).
$ErrorActionPreference = "Stop"

function Invoke-Step($s) {
  Write-Host "=== $s ==="
  $parts = $s -split ' '
  python @parts
  if ($LASTEXITCODE -ne 0) { throw "Failed: $s" }
}

if ($env:RERUN_TRAINING -eq "1") {
  @(
    "scripts/00_setup/00_check_inputs.py",
    "scripts/01_single_cell_qc/01_import_tisch_datasets.py",
    "scripts/01_single_cell_qc/02_standardise_cell_annotations.py",
    "scripts/01_single_cell_qc/03_filter_malignant_and_fibroblast_cells.py",
    "scripts/01_single_cell_qc/04_score_single_cells.py --scope all",
    "scripts/02_multinomial_pemt_model/01_define_training_labels.py",
    "scripts/02_multinomial_pemt_model/02_create_pseudobulk_mixtures.py",
    "scripts/02_multinomial_pemt_model/03_train_multinomial_classifier.py",
    "scripts/02_multinomial_pemt_model/04_export_classifier_coefficients.py"
  ) | ForEach-Object { Invoke-Step $_ }
}

# benchmark of alternative learners; needs the pseudobulk matrix from stage 02
Invoke-Step "scripts/02_multinomial_pemt_model/05_benchmark_classifiers.py"

@(
  "scripts/03_sc_classifier_validation/01_puram_pemt_score_characterization.py",
  "scripts/03_sc_classifier_validation/02_apply_classifier_to_sc_datasets.py",
  "scripts/03_sc_classifier_validation/03_validation_figure.py",
  "scripts/03_sc_classifier_validation/04_umap_figure.py",
  "scripts/04_tcga_projection/01_preprocess_tcga_expression.py",
  "scripts/04_tcga_projection/02_prepare_tcga_metadata.py",
  "scripts/04_tcga_projection/03_project_classifier_to_tcga.py",
  "scripts/04_tcga_projection/04_compute_pemt_specificity_score.py",
  "scripts/04_tcga_projection/05_make_tcga_trait_table.py",
  "scripts/04_tcga_projection/06_survival_analysis.py",
  "scripts/04_tcga_projection/07_emt_score_panel.py",
  "scripts/04_tcga_projection/08_clinical_associations.py",
  "scripts/04_tcga_projection/09_external_cohorts.py",
  "scripts/04_tcga_projection/10_independent_validation.py",
  "scripts/04_tcga_projection/12_cptac_validation.py",
  "scripts/04_tcga_projection/11_subgroup_robustness.py",
  "scripts/04_tcga_projection/14_survival_sensitivity.py",
  "scripts/04_tcga_projection/13_survival_meta_analysis.py",
  "scripts/05_wgcna/01_prepare_wgcna_expression.py",
  "scripts/05_wgcna/02_run_wgcna.py",
  "scripts/05_wgcna/03_module_trait_correlations.py",
  "scripts/05_wgcna/04_select_and_export_pemt_module.py",
  "scripts/06_network_analysis/03_drug_proximity.py"
) | ForEach-Object { Invoke-Step $_ }

if ($env:RUN_SUBNETWORK -eq "1") {
  @(
    "scripts/06_network_analysis/01_build_pemt_subnetwork.py",
    "scripts/06_network_analysis/02_permutation_centrality.py",
    "scripts/06_network_analysis/03_drug_proximity.py --disease key_proteins"
  ) | ForEach-Object { Invoke-Step $_ }
}

@(
  "scripts/06_network_analysis/03_drug_proximity.py --exclude-seed-targets",
  "scripts/06_network_analysis/04_network_figures.py",
  "scripts/06_network_analysis/05_lincs_signature_reversal.py",
  "scripts/07_depmap_broad_prism_validation/01_preprocess_depmap_expression.py",
  "scripts/07_depmap_broad_prism_validation/02_project_classifier_to_depmap.py",
  "scripts/07_depmap_broad_prism_validation/03_prism_drug_sensitivity.py",
  "scripts/07_depmap_broad_prism_validation/04_crispr_dependency.py",
  "scripts/07_depmap_broad_prism_validation/05_moa_permutation_test.py",
  "scripts/08_manuscript_tables/01_build_supplementary_tables.py",
  "scripts/08_manuscript_tables/02_overview_figures.py",
  "scripts/08_manuscript_tables/03_combined_figures.py"
) | ForEach-Object { Invoke-Step $_ }

Write-Host "Pipeline finished."
