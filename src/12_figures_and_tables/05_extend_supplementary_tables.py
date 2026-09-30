"""Extend the supplementary workbook with the tables of the later analyses and format it.

The pipeline workbook written by 01_build_supplementary_tables.py (working ids S1 to S21) is copied
to Supplementary_Data_1.xlsx. One sheet is then appended for each entry of NEW (working ids S22 to
S49), read from a TSV under results/, and listed in the Index sheet with its caption, row count and
column count. A missing source TSV is reported and skipped.

The whole workbook is then formatted.
  - List values in the LIST_COLUMNS columns are separated with commas.
  - The labels robust_hit, robust_hits and robust_approved_hits become seed_stable_hit,
    seed_stable_hits and seed_stable_approved_hits, and a ":robust" suffix becomes ":seed_stable".
  - The DrugBank descriptive text columns (moa, atc_titles, category_titles) are removed from the
    drug proximity sheets (Supplementary Tables S41 and S49), and the column counts in the Index are
    updated. The DrugBank licence does not allow this text to be redistributed.
Free text copied from DrugBank and chemical names are otherwise left as the source gives them.

Sheets keep their working ids here. 06_number_supplementary_tables.py gives them their published
numbers.

Inputs:  results/tables/Supplementary_Tables.xlsx
         the TSVs under results/ named in NEW
Outputs: results/tables/Supplementary_Data_1.xlsx
Usage:   python src/12_figures_and_tables/05_extend_supplementary_tables.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
BASE = ROOT / "results" / "tables" / "Supplementary_Tables.xlsx"
OUT = ROOT / "results" / "tables" / "Supplementary_Data_1.xlsx"

# (sheet name, source tsv, description for the index). Sheet names are capped at 31 characters
# by the file format, so they are written short and the index carries the full description.
NEW = [
    ("S22_pemt_gene_partition", "pemt_partition/puram_pemt_gene_partition.tsv",
     "Each of the 100 Puram pEMT genes resolved by cell type in GSE103322. Columns give the share of total "
     "expression (linear scale) and detection rate in malignant cells, fibroblasts, endothelium and immune "
     "cells, the arm the gene is assigned to, its gene family, the log2 fold change between pEMT-high and "
     "epithelial-like malignant cells, and whether it is among the classifier's top 100 positive coefficients."),
    ("S23_label_free_clusters", "label_free/cluster_composition.tsv",
     "Composition of each Leiden cluster obtained from malignant cells after removing all 533 genes of "
     "the six Puram programmes, with labels used at no point in the clustering."),
    ("S23b_label_free_summary", "label_free/label_free_summary.tsv",
     "Summary statistics for the label-free recovery: genes removed and retained, cluster count, "
     "adjusted Rand index and its permutation p-value, adjusted mutual information, silhouette against "
     "its null, leave-one-patient-out AUROC on non-Puram genes, and the number of Puram genes among the "
     "top 100 classifier coefficients."),
    ("S24_caf_lr_pairs", "cell_communication/caf_to_malignant_pairs.tsv",
     "Ligand-receptor pairs from fibroblasts to each malignant state by LIANA consensus of five methods, "
     "with magnitude and specificity ranks for pEMT-high and epithelial-like targets and the difference "
     "in log10 specificity between them."),
    ("S24b_caf_lr_matched", "cell_communication/caf_to_malignant_matched_control.tsv",
     "The same comparison repeated five times with pEMT-high cells downsampled to the epithelial-like "
     "cell count, giving the per-seed difference, its mean, and the number of runs favouring pEMT-high."),
    ("S24c_ligand_source", "cell_communication/ligand_source_by_celltype.tsv",
     "Share of total expression and detection rate per cell group for EGFR-family ligands and receptors "
     "and for canonical fibroblast-derived ligands, used to test autocrine against paracrine supply."),
    ("S25_cptac_protein", "cptac_proteome/protein_correlations.tsv",
     "Spearman correlation of every score with the abundance of each protein in the five pre-specified "
     "sets across 108 CPTAC-3 tumours, with Benjamini-Hochberg q values."),
    ("S26_cptac_phosphosites", "cptac_proteome/phosphosite_correlations.tsv",
     "Spearman correlation of pEMT specificity with each of 21,057 phosphosites quantified in at least "
     "30 tumours, before and after regressing out the abundance of the site's own parent protein, with q "
     "values and membership of the EGFR to MAPK cascade."),
    ("S27_cptac_pathway_perm", "cptac_proteome/pathway_permutation_test.tsv",
     "Set-level test of the EGFR to MAPK cascade phosphosites against a null in which the score is "
     "permuted across tumours and the whole statistic recomputed 2,000 times."),
    ("S28_tt_compartments", "tyler_tirosh/esg_compartment_assignment.tsv",
     "Each gene of the Tyler and Tirosh style EMT signature set resolved to the cancer or fibroblast "
     "compartment in GSE103322 by share of expression."),
    ("S29_tt_correlations", "tyler_tirosh/score_correlations.tsv",
     "Spearman correlation matrix among the EMT score panel, the malignant arm, and the cancer and "
     "fibroblast compartment scores obtained by applying the Tyler and Tirosh decoupling principle to "
     "the 520 TCGA-HNSC primaries."),
    ("S30_GSE65021_scores", "cetuximab_cohort/GSE65021_scores.tsv",
     "Per-patient scores and recorded covariates for the 40 patients of GSE65021 treated with platinum "
     "plus cetuximab, including progression-free survival group, age, stage, grade, prior radiotherapy, "
     "primary site and whether the specimen was the primary or the recurrence."),
    ("S31_GSE65021_association", "cetuximab_cohort/GSE65021_pfs_association.tsv",
     "Association of every score with prolonged progression-free survival on cetuximab: medians by "
     "group, rank-biserial correlation, one- and two-sided Mann-Whitney p values with Benjamini-Hochberg "
     "q values, bootstrap ROC AUC, and maximum-likelihood odds ratios per standard deviation unadjusted "
     "and adjusted (Firth estimates are in S38). MLR-EMT is the faithful implementation of George et al. "
     "The Zhou nine-gene set is flagged as not independent of this cohort and excluded from the FDR."),
    ("S32_GSE65021_headtohead", "cetuximab_cohort/GSE65021_headtohead.tsv",
     "Head-to-head logistic models in GSE65021 entering pEMT specificity and one other score together, "
     "unadjusted and adjusted, with the correlation between the two scores in this cohort."),
    ("S33_GSE65021_vs_epithelial", "cetuximab_cohort/GSE65021_mlr_vs_epithelial.tsv",
     "Logistic models in GSE65021 entering one score oriented away from the epithelial state "
     "(MLR-EMT or KS) and one epithelial score (76GS or Puram epithelial differentiation) together, "
     "unadjusted and adjusted, with the correlation between the two scores in this cohort. MLR-EMT is the "
     "faithful implementation. Estimates are maximum likelihood, and the Firth estimates are in S38c."),
    ("S34_GSE65021_combined", "cetuximab_cohort/GSE65021_combined_score.tsv",
     "Exploratory combined marker in GSE65021, specified after the single-score results were known. It is "
     "pEMT specificity plus Puram epithelial differentiation, as an equal-weight sum of standardised scores, "
     "with 76GS and sign-reversed MLR-EMT as sensitivity partners. Columns give the AUC with 2,000-resample "
     "bootstrap interval, the paired bootstrap gain over pEMT specificity alone (gain_over_axis), leave-one-out AUC of a fitted two-term logistic model, and AUC in the 31 primary-tumour "
     "specimens."),
    ("S35_malignant_arm_proximity", "network_proximity_malignant_arm/summary.tsv",
     "Network proximity seeded on the 52 malignant-arm pEMT genes of the single-cell partition (linear scale), independent "
     "of the classifier and containing no EGFR-family gene. Columns give drugs tested, hits, approved hits "
     "and hits that survive leaving out any one seed (seed-stable), for DrugBank curated and STITCH targets, in the standard analysis "
     "and with each drug's targets that are themselves seeds excluded, with the leading approved hits."),
    ("S36_arm_gene_sets", "arm_scores/arm_gene_sets.tsv",
     "Gene sets used to score the two arms of the pEMT partition directly. They are the total-share arms "
     "(52 and 43 genes), the cores (12 genes with malignant share at least 80% and 14 with fibroblast share "
     "at least 60%), the consensus arms (16 and 58 genes on the same side of parity by mean expression per "
     "cell in GSE103322, GSE150321 and GSE181919, S37), and sensitivity sets with the Zhou genes removed or "
     "alternative core thresholds."),
    ("S36b_arm_tcga_correlations", "arm_scores/tcga_arm_correlations.tsv",
     "Spearman correlation of each direct arm score with pEMT specificity, the published scores and the "
     "M06 and M13 module eigengenes in the 520 TCGA-HNSC primaries."),
    ("S36c_arm_survival_pooled", "arm_scores/survival_pooled.tsv",
     "Hazard ratio per SD for each direct arm score and comparators, univariable and adjusted, pooled over "
     "TCGA-HNSC, GSE41613, CPTAC-3 and GSE65858 by random and fixed effects."),
    ("S36d_arm_survival_cohorts", "arm_scores/survival_per_cohort.tsv",
     "Per-cohort Cox estimates underlying S36c."),
    ("S36e_tcga_systemic_therapy", "arm_scores/tcga_systemic_therapy_sensitivity.tsv",
     "TCGA-HNSC adjusted hazard ratios after excluding patients whose legacy clinical record marks "
     "systemic (targeted molecular) therapy, and restricted to those recorded as not receiving it."),
    ("S36f_arm_GSE65021", "arm_scores/gse65021_arm_association.tsv",
     "Direct arm scores in GSE65021: bootstrap AUC, rank-biserial correlation, two-sided Mann-Whitney p, "
     "Firth odds ratio per SD with profile-likelihood interval, and correlation with pEMT specificity "
     "and the Puram signature."),
    ("S36g_arm_GSE65021_joint", "arm_scores/gse65021_arm_joint_models.tsv",
     "Joint Firth models in GSE65021: malignant and stromal arm together, and the malignant arm with "
     "each epithelial score."),
    ("S37_partition_replication", "pemt_partition/replication_per_gene.tsv",
     "Each Puram pEMT gene in GSE103322, GSE150321 and GSE181919: malignant share of total expression and the "
     "composition-independent malignant share of mean expression per cell, detection rates, and arm by "
     "the per-cell measure."),
    ("S37b_partition_repl_summary", "pemt_partition/replication_summary.tsv",
     "Per dataset: arm counts by the per-cell measure, agreement and kappa with the published total-share "
     "assignment, correlation with the published shares, and the hemidesmosome versus other comparison."),
    ("S37c_partition_repl_cross", "pemt_partition/replication_cross_dataset.tsv",
     "Agreement of the per-cell measure between each pair of GSE103322, GSE150321 and GSE181919, and the "
     "genes on the same side of parity in all three."),
    ("S37d_partition_GSE181919_pts", "pemt_partition/replication_GSE181919_patient_summary.tsv",
     "GSE181919 per-patient analysis: for each pEMT gene, the number of patients with at least 20 "
     "malignant cells and 20 fibroblasts in tumour tissue, the median per-cell malignant share across "
     "them, and the fraction of patients in which the gene is malignant-predominant."),
    ("S38_GSE65021_firth", "cetuximab_cohort/GSE65021_firth_single.tsv",
     "Firth penalised logistic regression in GSE65021, each score alone, unadjusted and adjusted: odds "
     "ratio of prolonged progression-free survival per SD, profile penalised-likelihood interval and "
     "penalised likelihood-ratio p."),
    ("S38b_GSE65021_firth_h2h", "cetuximab_cohort/GSE65021_firth_headtohead.tsv",
     "Firth head-to-head models, pEMT specificity with each other score."),
    ("S38c_GSE65021_firth_epi", "cetuximab_cohort/GSE65021_firth_vs_epithelial.tsv",
     "Firth models entering MLR-EMT or KS with 76GS or Puram epithelial differentiation."),
    ("S39_mlr_versions_GSE65021", "mlr_faithful/cetuximab_comparison.tsv",
     "MLR-EMT in GSE65021 computed as the earlier approximation and as the faithful implementation of "
     "George et al. (Jolly-group reference code): AUC, rank-biserial, odds ratios and correlations."),
    ("S39b_mlr_versions_cohorts", "mlr_faithful/summary.tsv",
     "Faithful MLR-EMT in each cohort: cohort normalisation offset, epithelial, hybrid and mesenchymal "
     "state counts, and correlation with the approximation and with the other scores."),
    ("S40_GSE65021_fdr_family", "cetuximab_cohort/GSE65021_sensitivity_fdr_family.tsv",
     "One multiplicity family in GSE65021: every independently derived published score and the four arm "
     "scores specified before the cohort was examined, with AUC, two-sided Mann-Whitney p and "
     "Benjamini-Hochberg q."),
    ("S41_GSE65021_transport", "cetuximab_cohort/GSE65021_sensitivity_transportability.tsv",
     "Transportability of pEMT specificity and the malignant core in GSE65021: AUC with within-cohort, "
     "leave-one-out and external-reference (GSE65858) standardisation, and agreement with the published "
     "within-cohort score."),
    ("S42_GSE65021_paired_auc", "cetuximab_cohort/GSE65021_sensitivity_paired_auc.tsv",
     "Paired bootstrap comparison (2,000 resamples) of the AUC of each primary measure with 76GS, Puram "
     "epithelial differentiation, the canonical Puram pEMT signature and the Zhou nine-gene set."),
    ("S43_mlr_emt_states", "cetuximab_cohort/GSE65021_sensitivity_mlr_states.tsv",
     "MLR-EMT read as a three-state model. For GSE65021 and TCGA-HNSC, the range of each state probability "
     "and of the score. For GSE65021, the AUC for prolonged progression-free survival, two-sided Mann-Whitney "
     "p and Firth odds ratio per SD. Spearman correlation with pEMT specificity, the malignant core, 76GS and "
     "the Puram pEMT signature."),
    ("S43b_mlr_emt_state_counts", "cetuximab_cohort/GSE65021_sensitivity_mlr_state_counts.tsv",
     "GSE65021 patients by most probable MLR-EMT state and progression-free survival group, with the "
     "Fisher test of hybrid against mesenchymal."),
    ("S44_pdx_cetuximab", "pdx_cetuximab/pdx_cetuximab_association.tsv",
     "Cetuximab response in two head and neck cancer PDX panels (GSE84713 per patient and over all models, "
     "GSE183881 over tested models): AUC with bootstrap interval, two-sided Mann-Whitney p and Firth odds "
     "ratio per SD for each score, and correlation with pEMT specificity."),
    ("S44b_pdx_pooled", "pdx_cetuximab/pdx_cetuximab_pooled.tsv",
     "Fixed-effect pooling of the two PDX panels: odds ratio per SD with 95% interval."),
    ("S44c_pdx_scores_GSE84713", "pdx_cetuximab/GSE84713_scores.tsv",
     "Per-model scores, response label, patient and molecular subtype for the 28 GSE84713 models."),
    ("S44d_pdx_continuous_RTV", "pdx_cetuximab/GSE84713_continuous_RTV.tsv",
     "GSE84713 models also reported by Klinghammer et al. 2020: Spearman correlation of each score with "
     "cetuximab relative tumour volume transcribed from that paper's Table 1 (lower is better)."),
    ("S44e_pdx_paired_auc", "pdx_cetuximab/pdx_paired_auc.tsv",
     "Paired bootstrap AUC differences in the two PDX panels: malignant core and pEMT specificity against "
     "76GS, Puram epithelial differentiation and the canonical Puram signature."),
    ("S44f_pdx_combined", "pdx_cetuximab/pdx_combined_marker.tsv",
     "The patient-cohort combination (pEMT specificity plus Puram epithelial differentiation, equal "
     "weights) evaluated in the two PDX panels without refitting."),
    ("S45_gdsc_cetuximab", "gdsc_cetuximab/gdsc_cetuximab_association.tsv",
     "Cetuximab sensitivity of cell lines (GDSC1 drug 1114) against each score: Spearman correlation with "
     "the dose-response area and ln IC50 (negative = higher score in more sensitive lines), and AUC for the "
     "most sensitive quartile, in the head and neck panel and in a squamous panel."),
    ("S45b_gdsc_cetuximab_scores", "gdsc_cetuximab/gdsc_cetuximab_scores.tsv",
     "Per-line scores and cetuximab response for both cell-line panels."),
    ("S46_pseudobulk_construction_cv", "pseudobulk_sensitivity/cv_by_construction.tsv",
     "Leave-one-dataset-out cross-validation of the classifier for two constructions of the training "
     "pseudobulks (mean of the cells' log-normalised profiles, as trained, and the bulk-like sum of linear "
     "counts, renormalised and log-transformed), each evaluated with the training-fold scaler and with the "
     "held-out fold standardised within itself, as bulk cohorts are scored."),
    ("S46b_pseudobulk_construction", "pseudobulk_sensitivity/summed_construction_summary.tsv",
     "Classifier refitted on the summed pseudobulks: overlap of its top 100 pEMT-high coefficients with the "
     "published classifier, correlation of its pEMT specificity with the published score in TCGA-HNSC and "
     "GSE65021, and its AUC for prolonged progression-free survival in GSE65021."),
    ("S47_composition_simulation", "composition_simulation/composition_regression.tsv",
     "Simulated tumours built from GSE181919 cells: standardised regression of each score on the share of "
     "pEMT-high malignant cells (q), the fibroblast fraction (f) and the immune fraction (u), with the "
     "partial R squared of each factor."),
    ("S47b_simulated_tumours", "composition_simulation/simulated_tumours.tsv",
     "Design and scores of the 450 simulated tumours."),
    ("S48_lincs_moa_50_sets", "lincs_controls/lincs_moa_enrichment_50sets.tsv",
     "Mechanism-of-action share of reversing L1000 compound signatures for the real query against 50 random "
     "gene sets of the same size, with enrichment, empirical p (add-one) and Benjamini-Hochberg q."),
    ("S48b_lincs_two_sided", "lincs_controls/lincs_l1000_cp_perturbagens_two_sided.tsv",
     "Per-compound sign test of reversing against mimicking signatures, one-sided as in S14 and two-sided, "
     "each with Benjamini-Hochberg false discovery rate."),
    ("S49_GSE65021_purity", "cetuximab_cohort/GSE65021_purity_adjustment.tsv",
     "Firth odds ratio per SD in GSE65021 for the malignant core and pEMT specificity, alone and adjusted "
     "for the stromal core, an 18-gene immune score, or both, with their correlations with these covariates."),
]


# LIST_COLUMNS hold lists, written with commas in the workbook. Free text copied from DrugBank and chemical
# names are left exactly as the source gives them.
LIST_COLUMNS = {"in_published_signature", "anchor_module_genes", "atc_level1", "targets_adjacent_to_module_genes",
                "targets_that_are_module_genes", "targets_at_distance_0", "targets_at_distance_1", "top_approved"}


# DrugBank-authored text columns, dropped from the proximity sheets S8_* and S21_* (published S41 and
# S49) because the DrugBank licence does not allow redistribution
DRUGBANK_TEXT = {"moa", "atc_titles", "category_titles"}


def _clean(wb) -> None:
    import re
    for ws in wb.worksheets:
        if not (ws.title.startswith("S8_") or ws.title.startswith("S21_")):
            continue
        for r in (1, 2):
            vals = [c.value for c in ws[r]]
            drop = [i + 1 for i, h in enumerate(vals) if str(h) in DRUGBANK_TEXT]
            if drop:
                for col in sorted(drop, reverse=True):
                    ws.delete_cols(col)
                index = wb["Index"]
                for ir in range(2, index.max_row + 1):
                    if index.cell(ir, 1).value == ws.title and isinstance(index.cell(ir, 4).value, int):
                        index.cell(ir, 4).value -= len(drop)
                print(f"  {ws.title}: dropped DrugBank text columns {sorted(DRUGBANK_TEXT & set(map(str, vals)))}")
                break
    n_cap = n_cell = 0
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                v = cell.value
                if not isinstance(v, str):
                    continue
                new = v
                new = re.sub(r"\brobust_(hits?|approved_hits)\b", r"seed_stable_\1", new)
                new = re.sub(r":robust\b", ":seed_stable", new)
                if new != v:
                    cell.value = new
                    n_cap += 1
        if ws.title == "Index":
            continue
        header = None
        for r in (1, 2):
            vals = [c.value for c in ws[r]] if ws.max_row >= r else []
            if LIST_COLUMNS & set(map(str, vals)):
                header = (r, vals)
        if header is None:
            continue
        r0, vals = header
        cols = [i + 1 for i, h in enumerate(vals) if str(h) in LIST_COLUMNS]
        for col in cols:
            for (cell,) in ws.iter_rows(min_row=r0 + 1, min_col=col, max_col=col):
                if isinstance(cell.value, str) and ";" in cell.value:
                    cell.value = re.sub(r"\s*;\s*", ", ", cell.value)
                    n_cell += 1
    print(f"  formatting: {n_cap} label cells renamed, {n_cell} list cells separated with commas")


def main() -> None:
    import openpyxl

    shutil.copy(BASE, OUT)
    wb = openpyxl.load_workbook(OUT)
    index = wb["Index"]

    added = []
    for sheet, rel, desc in NEW:
        assert len(sheet) <= 31, f"sheet name too long for xlsx: {sheet}"
        src = RES / rel
        if not src.exists():
            print(f"  MISSING, skipped: {rel}")
            continue
        df = pd.read_csv(src, sep="\t")
        ws = wb.create_sheet(sheet)
        ws.append([str(c) for c in df.columns])
        for row in df.itertuples(index=False, name=None):
            ws.append([None if pd.isna(v) else v for v in row])
        index.append([sheet, desc, len(df), df.shape[1]])
        added.append((sheet, len(df), df.shape[1]))

    _clean(wb)
    wb.save(OUT)
    print(f"\n{len(added)} sheets appended to {OUT.name} ({OUT.stat().st_size/1e6:.1f} MB)")
    for s, r, c in added:
        print(f"  {s:<28} {r:>6} rows  {c:>3} cols")


if __name__ == "__main__":
    main()
