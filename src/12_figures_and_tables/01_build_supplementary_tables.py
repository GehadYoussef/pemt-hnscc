"""Assemble the pipeline supplementary tables into one workbook, one sheet per table.

Reads the results of src/02_multinomial_pemt_model to src/07_depmap_broad_prism_validation
(classifier benchmark, single-cell validation, TCGA and external-cohort scores and survival, WGCNA,
drug proximity, DepMap and PRISM, LINCS and independent validation) and writes one sheet per table
under its working id (S1 to S21), sorted by id, with an Index sheet listing every sheet, its
description and its size. Optional sheets whose source file is absent are skipped. Every sheet
starts with a one-line description. Floats are rounded to 4 decimals in the workbook. Each sheet is
also written as a TSV of the same rounded table.

05_extend_supplementary_tables.py appends the later tables to this workbook and
06_number_supplementary_tables.py gives every sheet its published number.

Inputs:  results/multinomial_classifier/, results/sc_classifier_validation/,
         results/tcga_projection/ (and its external/ folder), results/wgcna/,
         results/network_analysis/, results/depmap_broad_prism/
Outputs: results/tables/Supplementary_Tables.xlsx
         results/tables/tsv/<sheet>.tsv
Usage:   python src/12_figures_and_tables/01_build_supplementary_tables.py
"""

import sys
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pemt import load_config, project_root  # noqa: E402

cfg = load_config()
ROOT = project_root()
RES = ROOT / cfg["paths"]["results_dir"]
OUT = ROOT / "results" / "tables"
(OUT / "tsv").mkdir(parents=True, exist_ok=True)


def tsv(p, **kw):
    return pd.read_csv(p, sep="\t", **kw)


def build() -> list[tuple[str, str, pd.DataFrame]]:
    tables = []

    # Classifier benchmark
    b = RES / "multinomial_classifier" / "classifier_benchmark_folds.tsv"
    if b.exists():
        tables.append(("S1_classifier_benchmark", "Alternative classifiers on the identical pseudobulk training set and leave-one-dataset-out folds: accuracy, AUROC per class, "
                       "log loss, and Spearman correlation of each model's P(pEMT-high) with the elastic net on the held-out fold.", tsv(b)))
        tables.append(("S1b_classifier_benchmark_mean", "Fold means of S1_classifier_benchmark.", tsv(RES / "multinomial_classifier" / "classifier_benchmark_summary.tsv")))
    bd = RES / "multinomial_classifier" / "classifier_benchmark_folds_deployment.tsv"
    if bd.exists():
        tables.append(("S1c_benchmark_deployment", "Deployment-matched evaluation of the same learners on the same folds: the held-out dataset is z-scored gene-wise within "
                       "itself, as every bulk cohort is, and for the linear models only the fitted logistic layer is applied (tree models are refitted on the "
                       "training fold standardised with its own scaler). Columns as in S1_classifier_benchmark.", tsv(bd)))
        tables.append(("S1d_benchmark_deployment_mean", "Fold means of S1c_benchmark_deployment.",
                       tsv(RES / "multinomial_classifier" / "classifier_benchmark_summary_deployment.tsv")))

    # Single-cell validation
    pb = tsv(RES / "sc_classifier_validation" / "pseudobulk_probabilities.tsv", index_col=0)
    pb = pb.reset_index().rename(columns={"index": "pseudobulk"})
    tables.append(("S2_sc_pseudobulk", "Classifier probabilities for every malignant (patient x site) and fibroblast (patient) pseudobulk with >= 20 cells, "
                   "after within-dataset z-scoring of the classifier genes. puram_pemt_signature = mean z of the Puram pEMT genes.", pb))
    for name, f in (("S2b_sc_summary_by_cell_type", "summary_by_cell_type.tsv"), ("S2c_sc_rank_agreement", "rank_agreement_with_puram_signature.tsv"),
                    ("S2d_puram_nodal_status", "puram_nodal_status_test.tsv"), ("S2e_puram_ln_vs_primary", "puram_ln_vs_primary_paired.tsv"),
                    ("S2f_per_cell_score_correlations", "per_cell_score_correlations.tsv")):
        p = RES / "sc_classifier_validation" / f
        if p.exists():
            tables.append((name, f"Single-cell validation: {f}", tsv(p)))

    # TCGA scores and survival
    traits = tsv(RES / "tcga_projection" / "tcga_master_trait_table.tsv", index_col=0, low_memory=False)
    panel = tsv(RES / "tcga_projection" / "emt_score_panel_scores.tsv", index_col=0)
    keep = [c for c in ["P_epithelial_like", "P_fibroblast_stromal_like", "P_pEMT_high", "pEMT_specificity", "sample_type", "hpv_status",
                        "ajcc_pathologic_stage.diagnoses", "ajcc_pathologic_t.diagnoses", "ajcc_pathologic_n.diagnoses", "tissue_or_organ_of_origin.diagnoses",
                        "age_at_index.demographic", "OS", "OS.time"] if c in traits.columns]
    s2 = traits[keep].join(panel[[c for c in panel.columns if c not in traits.columns]])
    s2 = s2.reset_index().rename(columns={"index": "sample"})
    tables.append(("S3_tcga_scores", "TCGA-HNSC samples: classifier probabilities, pEMT specificity, the EMT score panel (76GS, KS, Hallmark EMT, Puram pEMT, "
                   "Puram epithelial differentiation, MLR), HPV status and clinical variables used in the models.", s2))
    cox = []
    for lab, f in (("M1 univariable", "cox_univariable.tsv"), ("M2 adjusted (stage, age, site, HPV)", "cox_multivariable_M2.tsv"),
                   ("M3 sensitivity (M2 + pack-years)", "cox_multivariable_M3.tsv")):
        d = tsv(RES / "tcga_projection" / f); d.insert(0, "model", lab); cox.append(d)
    tables.append(("S4_cox_models", "Cox proportional-hazards models for overall survival in TCGA-HNSC primaries (pEMT specificity per unit, site reference oropharynx).",
                   pd.concat(cox)))
    tables.append(("S4b_cv_cindex", "Repeated 5-fold cross-validated concordance index of the clinical model with and without pEMT specificity.",
                   tsv(RES / "tcga_projection" / "cindex_comparison.tsv")))
    tables.append(("S4c_km_logrank", "Log-rank test across pEMT specificity tertiles (Kaplan-Meier, Figure 4b).", tsv(RES / "tcga_projection" / "km_logrank_test.tsv")))
    tables.append(("S5_emt_panel_cox", "EMT score panel: hazard ratio per SD, univariable and adjusted (stage, age, site, HPV), and cross-validated C-index.",
                   tsv(RES / "tcga_projection" / "emt_score_panel_cox.tsv").merge(tsv(RES / "tcga_projection" / "emt_score_panel_cv_cindex.tsv"), on=["score", "label"], how="left")))
    jc = RES / "tcga_projection" / "emt_score_panel_joint_cox.tsv"
    if jc.exists():
        tables.append(("S5c_emt_panel_head_to_head", "Head-to-head: pEMT specificity and each score in the same adjusted Cox model (stage, age, site, HPV). Columns give the HR per SD of each given the other, "
                       "the likelihood-ratio p for adding either to clinical covariates plus the other, and the cross-validated C-index of clinical + score with and without pEMT specificity.", tsv(jc)))
    corr = tsv(RES / "tcga_projection" / "emt_score_panel_correlations.tsv", index_col=0).reset_index().rename(columns={"index": "score"})
    tables.append(("S5b_emt_panel_correlation", "Spearman correlations among the EMT scores in TCGA-HNSC primaries.", corr))
    tables.append(("S6_clinical_associations", "Association of each score with pathological N and T stage, HPV status, site, and (legacy TCGA clinicalMatrix) grade, "
                   "lymphovascular invasion, perineural invasion, extranodal extension and positive margin (Mann-Whitney / Spearman / Kruskal-Wallis).",
                   tsv(RES / "tcga_projection" / "clinical_associations.tsv")))
    ext = RES / "tcga_projection" / "external" / "external_cox_summary.tsv"
    if ext.exists():
        tables.append(("S6b_external_cohorts_cox", "External cohorts (GEO GSE65858 Illumina HT-12, GSE41613 Affymetrix U133 Plus 2): classifier projected with within-cohort z-scoring. "
                       "HR per SD for every score, univariable and adjusted for stage, age, site and HPV16 DNA or RNA in GSE65858 and for stage and age band "
                       "in GSE41613.", tsv(ext)))
        for cid in ("GSE65858", "GSE41613"):
            f = RES / "tcga_projection" / "external" / f"{cid}_scores.tsv"
            if f.exists():
                tables.append((f"S6c_{cid}_scores", f"[{cid}] per-sample classifier probabilities, pEMT specificity, score panel and the clinical variables used.", tsv(f, index_col=0).reset_index().rename(columns={"index": "sample"})))

    # WGCNA
    tables.append(("S7_wgcna_modules", "WGCNA modules (signed, soft power 13, 8,000 most variable genes, 520 primaries): size, correlation of the module eigengene with the "
                   "classifier outputs and Hallmark EMT, overlap with the Puram pEMT signature (hypergeometric p), best-matching program, top hubs by kME.",
                   tsv(RES / "wgcna" / "module_annotation.tsv")))
    mt = tsv(RES / "wgcna" / "module_trait_correlations.tsv", index_col=0)
    mp = tsv(RES / "wgcna" / "module_trait_pvalues.tsv", index_col=0)
    long = mt.stack().rename("r").to_frame().join(mp.stack().rename("p")).reset_index().rename(columns={"level_0": "module", "level_1": "trait"})
    tables.append(("S7b_module_trait", "Module eigengene to trait correlations (Pearson r and p), primaries only.", long))
    tables.append(("S7c_module_enrichment", "Hypergeometric enrichment of each module for the Puram programs, Hallmark EMT and immune/stromal marker sets.",
                   tsv(RES / "wgcna" / "module_program_enrichment.tsv")))
    ga = tsv(RES / "wgcna" / "gene_module_assignments.tsv")
    kme = tsv(RES / "wgcna" / "gene_kME.tsv", index_col=0)
    if "gene" in ga.columns:
        ga = ga.set_index("gene")
    ga = ga.join(kme.drop(columns=[c for c in ["module"] if c in kme.columns]), how="left").reset_index()
    tables.append(("S7d_gene_modules", "Module assignment for every gene in the WGCNA input (core_member_pre_PAM = assigned before the PAM step) and kME to every module eigengene.", ga))
    tables.append(("S7e_network_seeds", "Seed gene sets for the network analysis. pEMT_axis is the M13 members with kME >= 0.7, canonical_pEMT "
                       "the Puram pEMT genes in M06, and classifier_pEMT_high the top 100 positive pEMT-high classifier coefficients.",
                   tsv(RES / "wgcna" / "network_seeds.tsv")))

    # Drug proximity (module genes as the disease set)
    for seed in ("canonical_pEMT", "pEMT_axis", "classifier_pEMT_high"):
        d = RES / "network_analysis" / seed
        for tset in ("drugbank", "stitch"):
            p = d / f"drug_proximity_{tset}_seeds.tsv"
            if p.exists():
                dp = tsv(p)
                cols = [c for c in ["drug_id", "name", "type", "status_label", "approved", "atc_level1", "atc_titles", "moa", "n_targets_in_interactome", "d_observed",
                                    "mu_random", "sigma_random", "z", "p_empirical", "p_normal", "fdr_normal", "is_hit", "n_anchor_key_proteins", "anchor_key_proteins",
                                    "z_leave_one_anchor_out", "anchor_whose_removal_hurts_most", "robust_hit", "targets_at_distance_0", "targets_at_distance_1",
                                    "targets_that_are_key_proteins", "targets_adjacent_to_key_proteins", "category_titles"] if c in dp.columns]
                dp = dp[cols].rename(columns={"n_anchor_key_proteins": "n_anchor_module_genes", "anchor_key_proteins": "anchor_module_genes",
                                              "targets_that_are_key_proteins": "targets_that_are_module_genes", "targets_adjacent_to_key_proteins": "targets_adjacent_to_module_genes"})
                tables.append((f"S8_{seed}_{tset}", f"[{seed}, {tset} targets] Guney network proximity of every drug's targets to the module genes (1,000 degree-matched draws). "
                               "A drug is a hit (is_hit) when z <= -1.96 and the normal-approximation FDR <= 0.05, and a seed-stable hit (robust_hit) when it is "
                               "still a hit after removing the single most influential anchor gene. Drug accession, name, type, approval status and ATC level 1 "
                               "from DrugBank (licensed export of 19 April 2025). DrugBank descriptive text (mechanism of action, ATC titles, categories) is "
                               "omitted because the licence does not allow redistribution.", dp))
        if (d / "centrality_pvalue_results.tsv").exists():
            kp = tsv(d / "centrality_pvalue_results.tsv", index_col=0).reset_index()
            tables.append((f"S9_{seed}_subnetwork_centrality", f"[{seed}] STRING physical subnetwork (shortest paths between module genes), node centralities and "
                           "their z against 1,000 degree-preserving rewirings. Shown for comparison and not used in the results, because this subnetwork "
                           "changed with the STRING evidence channel and was driven by hub proteins.", kp))

    # DepMap and PRISM
    for cohort in ("hnscc", "pan_squamous"):
        d = RES / "depmap_broad_prism"
        if (d / f"depmap_class_probabilities_{cohort}.tsv").exists():
            tables.append((f"S10_depmap_scores_{cohort}", f"[{cohort}] DepMap models: classifier probabilities after within-cohort z-scoring.",
                           tsv(d / f"depmap_class_probabilities_{cohort}.tsv", index_col=0).reset_index().rename(columns={"index": "depmap_id"})))
        if (d / f"prism_pemt_sensitivity_{cohort}.tsv").exists():
            tables.append((f"S11_prism_{cohort}", f"[{cohort}] PRISM secondary screen: Spearman correlation of pEMT specificity with AUC per compound "
                           "(negative = pEMT-high models more sensitive), BH-FDR, tertile AUC difference and proximity-hit flags.", tsv(d / f"prism_pemt_sensitivity_{cohort}.tsv")))
        if (d / f"prism_proximity_hit_set_test_{cohort}.tsv").exists():
            tables.append((f"S11b_prism_set_test_{cohort}", f"[{cohort}] Shift of proximity-hit compounds towards pEMT-high sensitivity, by the Mann-Whitney test "
                           "across compounds (anti-conservative, because compounds screened on the same cell lines are not independent) and by the cell-line "
                           "permutation null, which is the reported test.",
                           tsv(d / f"prism_proximity_hit_set_test_{cohort}.tsv")))
        if (d / f"crispr_pemt_dependency_{cohort}.tsv").exists():
            tables.append((f"S12_crispr_{cohort}", f"[{cohort}] CRISPR (Chronos) dependency: Spearman correlation of pEMT specificity with gene effect per gene "
                           "(negative = stronger dependency in pEMT-high models), BH-FDR, and membership of seed, key-protein and drug-target sets.",
                           tsv(d / f"crispr_pemt_dependency_{cohort}.tsv")))
        if (d / f"crispr_geneset_summary_{cohort}.tsv").exists():
            tables.append((f"S12b_crispr_sets_{cohort}", f"[{cohort}] Gene-set level CRISPR summary (Mann-Whitney of rho, set vs other genes).",
                           tsv(d / f"crispr_geneset_summary_{cohort}.tsv")))
    if (RES / "depmap_broad_prism" / "prism_moa_permutation_test.tsv").exists():
        tables.append(("S13_prism_moa_permutation", "Every mechanism-of-action class with >= 5 compounds, and the network-predicted compound sets, tested against a "
                       "permutation null that shuffles pEMT specificity across cell lines and recomputes the class statistic (5,000 permutations). Unlike the Mann-Whitney "
                       "comparison across compounds, this test does not treat compounds screened on the same lines as independent. No class "
                       "reaches FDR 0.05 in either cohort.", tsv(RES / "depmap_broad_prism" / "prism_moa_permutation_test.tsv")))

    # LINCS signature reversal
    L = RES / "network_analysis" / "lincs"
    if (L / "lincs_l1000_cp_perturbagens.tsv").exists():
        tables.append(("S14_lincs_compounds", "SigCom LINCS two-sided enrichment of the classifier signature (top 150 pEMT-high coefficients up, top 150 epithelial-like "
                       "coefficients down) against the L1000 compound library: per perturbagen, the number of signatures reversing and mimicking the query and a sign test with BH-FDR.",
                       tsv(L / "lincs_l1000_cp_perturbagens.tsv")))
    if (L / "lincs_moa_enrichment.tsv").exists():
        tables.append(("S14b_lincs_moa", "Mechanism-of-action classes among reversing L1000 signatures, against five random gene sets of the same size (specificity control).",
                       tsv(L / "lincs_moa_enrichment.tsv")))
    for key, fname, desc in (
        ("S14c_lincs_xpr", "lincs_l1000_xpr_perturbagens.tsv", "Same query against the L1000 CRISPR knockout library."),
        ("S14d_lincs_shrna", "lincs_l1000_shRNA_perturbagens.tsv", "Same query against the L1000 shRNA library."),
    ):
        if (L / fname).exists():
            tables.append((key, desc, tsv(L / fname)))

    # Independent validation, survival meta-analysis and sensitivity analyses
    T = RES / "tcga_projection"
    for key, fname, desc in (
        ("S15_published_correlations", "independent_validation_correlations.tsv",
         "Spearman correlation of five externally published HNSCC programmes (the 171-gene EGFR-induced EMT signature of Schinke 2022, "
         "three sets from Zhou 2025 with 46 EGFR invasion fDEGs, a 59-gene invasive network and 9 cetuximab-PFS predictors, and the 28-gene "
         "tumour budding signature of Ourailidis 2026) with every score in the panel, in TCGA-HNSC and both external cohorts. Gene lists taken from the "
         "source papers' supplementary files."),
        ("S15b_gene_level", "independent_validation_gene_level.tsv",
         "Per-gene Spearman correlation with pEMT specificity in TCGA-HNSC primaries, grouped into hemidesmosome/laminin-332, EGFR ligands and receptor, basal keratinocyte, "
         "classical mesenchymal EMT, and fibroblast/ECM stroma."),
        ("S15c_tcga_subtype", "independent_validation_subtype.tsv",
         "Each score by TCGA 2015 four-class expression subtype (Basal, Classical, Atypical, Mesenchymal, n = 279 matched primaries), with Kruskal-Wallis across subtypes and "
         "Basal-vs-rest and Mesenchymal-vs-rest Mann-Whitney."),
        ("S15d_joint_cox", "independent_validation_joint_cox.tsv",
         "Adjusted Cox (stage, age, site, HPV) with pEMT specificity and one published signature in the same model, per SD, with likelihood-ratio tests for adding either to the other."),
    ):
        if (T / fname).exists():
            tables.append((key, desc, tsv(T / fname)))
    if (T / "classifier_coefficient_overlap_published.tsv").exists():
        tables.append(("S15e_coefficient_overlap", "Hypergeometric overlap of the top 100 positive pEMT-high classifier coefficients with each published gene set, "
                       "against the 23,409-gene classifier universe. The classifier was trained only on single-cell pseudobulk labels, without the "
                       "EGF-stimulation, cetuximab-outcome or tumour-budding data behind those sets.",
                       tsv(T / "classifier_coefficient_overlap_published.tsv")))
    if (T / "survival_meta_analysis.tsv").exists():
        tables.append(("S16_survival_meta", "Random-effects (DerSimonian-Laird) meta-analysis of the adjusted hazard ratio per SD across TCGA-HNSC, GSE41613, "
                       "CPTAC-3 and GSE65858, for every score: pooled HR, 95% CI, Cochran's Q, I-squared and tau-squared.",
                       tsv(T / "survival_meta_analysis.tsv")))
    if (T / "survival_meta_inputs.tsv").exists():
        tables.append(("S16b_meta_inputs", "The per-cohort adjusted hazard ratios, confidence intervals, sample sizes and event counts that entered the meta-analysis.",
                       tsv(T / "survival_meta_inputs.tsv")))
    for key, fname, desc in (
        ("S17_subgroup_tcga", "subgroup_robustness_tcga.tsv",
         "Adjusted Cox hazard ratio per SD for every score within each anatomical site and HPV stratum of TCGA-HNSC. The score is standardised inside each "
         "subgroup so all rows are per SD and comparable. The survival association is not specific to a site or HPV stratum."),
        ("S17b_subgroup_external", "subgroup_robustness_external.tsv",
         "The same subgroup test inside GSE65858, GSE41613 and CPTAC-3."),
    ):
        if (T / fname).exists():
            tables.append((key, desc, tsv(T / fname)))
    for key, fname, desc in (
        ("S19_survival_sensitivity", "survival_sensitivity_pooling.tsv",
         "Pooled hazard ratio per SD under four specifications (univariable and adjusted, each fixed-effect and random-effects), with standard errors "
         "taken directly from the Cox fits, not back-calculated from confidence intervals."),
        ("S19b_sensitivity_per_cohort", "survival_sensitivity_per_cohort.tsv",
         "The per-cohort estimates behind the sensitivity analysis, univariable and adjusted."),
        ("S19c_leave_one_cohort_out", "survival_sensitivity_loo.tsv",
         "Leave-one-cohort-out random-effects pooling, testing whether either conclusion depends on a single cohort."),
        ("S19d_truncation", "survival_sensitivity_truncation.tsv",
         "Pooling after truncating follow-up in every cohort at 36 and 60 months, since median follow-up differs markedly between cohorts."),
        ("S19e_gse65858_treatment", "gse65858_treatment_strata.tsv",
         "GSE65858 stratified by treatment modality (single modality versus multimodal). GSE65858 is the only cohort with mixed treatment and the only "
         "one in which pEMT specificity is null. Stratifying does not recover an effect."),
        ("S20_gse65858_subtype", "gse65858_consensus_subtype.tsv",
         "Each score by the consensus expression clusters GSE65858 publishes for itself. These clusters share their names with the TCGA four-class calls "
         "but were derived de novo within that cohort, and the externally defined Zhou cetuximab-PFS gene set peaks in a different class here than in "
         "TCGA."),
    ):
        if (T / fname).exists():
            tables.append((key, desc, tsv(T / fname)))
    for seed in ("canonical_pEMT", "pEMT_axis", "classifier_pEMT_high"):
        f = RES / "network_analysis" / seed / "drug_proximity_drugbank_seeds_noseedtargets.tsv"
        if f.exists():
            tab = tsv(f)
            keep = [c for c in ["name", "approved", "moa", "n_targets_in_interactome", "z", "p_normal", "fdr_normal",
                                "is_hit", "robust_hit", "z_leave_one_anchor_out"] if c in tab.columns]
            tables.append((f"S21_strict_proximity_{seed}"[:31],
                           f"[{seed}] Strict proximity variant: a drug's targets that are themselves seed genes are removed before the distance is computed, "
                           "so a drug scores only if it lies in the neighbourhood of the programme and not on one of its genes.", tab[keep]))
    if (T / "external" / "CPTAC_HNSCC_cox.tsv").exists():
        tables.append(("S18_cptac_cox", "CPTAC-3 HNSCC (Huang et al. 2021, HPV-negative, surgically treated): univariable and adjusted Cox hazard ratio per SD for every score. "
                       "Survival is taken from the current GDC clinical release (40 deaths), not the frozen 2021 publication snapshot (16 deaths).",
                       tsv(T / "external" / "CPTAC_HNSCC_cox.tsv")))
    return tables


def sort_key(name: str) -> tuple[int, str]:
    """Sort key that orders sheets by table number and then letter: S1, S1b, S2, ... S21."""
    stem = name.split("_", 1)[0]
    num = int("".join(ch for ch in stem[1:] if ch.isdigit()))
    suffix = "".join(ch for ch in stem[1:] if ch.isalpha())
    return num, suffix


def main() -> None:
    tables = sorted(build(), key=lambda t: sort_key(t[0]))
    index = pd.DataFrame([{"sheet": n, "description": desc, "rows": len(df), "columns": len(df.columns)} for n, desc, df in tables])
    with pd.ExcelWriter(OUT / "Supplementary_Tables.xlsx", engine="openpyxl") as xw:
        index.to_excel(xw, sheet_name="Index", index=False)
        for name, desc, df in tables:
            df = df.copy()
            for c in df.select_dtypes("float").columns:
                df[c] = df[c].round(4)
            df.to_excel(xw, sheet_name=name[:31], index=False, startrow=1)
            ws = xw.sheets[name[:31]]
            ws.cell(row=1, column=1, value=desc).font = Font(italic=True)
            ws.freeze_panes = "A3"
            for j, col in enumerate(df.columns, start=1):
                width = min(45, max(10, int(df[col].astype(str).str.len().quantile(0.9)) + 2 if len(df) else 12, len(str(col)) + 2))
                ws.column_dimensions[get_column_letter(j)].width = width
                ws.cell(row=2, column=j).font = Font(bold=True)
                ws.cell(row=2, column=j).alignment = Alignment(wrap_text=True)
            (OUT / "tsv" / f"{name}.tsv").write_text(df.to_csv(sep="\t", index=False), encoding="utf-8")
        ws = xw.sheets["Index"]
        ws.column_dimensions["A"].width = 34; ws.column_dimensions["B"].width = 120
        for j in range(1, 5):
            ws.cell(row=1, column=j).font = Font(bold=True)
    print(f"{len(tables)} sheets written to {OUT / 'Supplementary_Tables.xlsx'}")
    print(index.to_string(index=False))


if __name__ == "__main__":
    main()
