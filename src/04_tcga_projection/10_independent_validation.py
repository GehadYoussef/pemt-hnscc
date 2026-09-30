"""Compare the pEMT axis with externally published HNSCC programmes.

All programmes were defined by other groups on their own data. The analysis has four parts:

  1. Gene programmes. Schinke 2022 (EGFR-induced EMT, 171 genes), Zhou 2025 (EGFR-driven local
     invasion, 46 functional DEGs, a 59-gene invasive network, and the 9 genes whose low expression
     predicted short PFS on cetuximab) and Ourailidis 2026 (28-gene spatial tumour budding
     signature). Each is scored as the mean per-gene z-score and correlated (Spearman) with every
     score in the EMT panel. The top 100 positive pEMT-high classifier coefficients are also tested
     for overlap with each gene set (hypergeometric test, universe = all classifier genes).
  2. Gene-level correlates. pEMT specificity is correlated with individual genes from five classes:
     hemidesmosome / laminin-332, EGFR ligands and receptor, basal keratinocyte, classical
     mesenchymal EMT, and fibroblast / ECM stroma.
  3. TCGA molecular subtype. The Basal / Classical / Atypical / Mesenchymal calls (TCGA 2015 Nature
     freeze, n = 279) classify the same tumours independently. Each score is compared across
     subtypes (Kruskal-Wallis) and between Basal or Mesenchymal and the rest (Mann-Whitney).
     Basal and Mesenchymal are the two subtypes that Klinghammer et al. linked to cetuximab
     response and resistance in PDX models.
  4. Replication. The correlations are recomputed in GSE41613, GSE65858 and CPTAC-3 HNSCC, when
     their score tables are present.

Survival is also refitted with pEMT specificity and each published signature (per SD) in one
adjusted Cox model (stage, age, site, HPV), with likelihood-ratio tests for adding either to the
other.

Inputs:  data/processed/tcga_hnsc/tcga_star_tpm_log2_for_projection.tsv
         results/tcga_projection/emt_score_panel_scores.tsv, tcga_master_trait_table.tsv
         results/tcga_projection/external/{GSE41613,GSE65858,CPTAC_HNSCC}_scores.tsv
         results/multinomial_classifier/classifier_coefficients.tsv
         the published signature files and TCGA_HNSC_4class_expression_subtype.csv in data/references/
Outputs: results/tcga_projection/
           independent_validation_correlations.tsv    every published signature x every score, all cohorts
           classifier_coefficient_overlap_published.tsv  hypergeometric overlap of the top 100 classifier
                                                      coefficients with each published gene set
           independent_validation_gene_level.tsv      per-gene Spearman with pEMT specificity, annotated by class
           independent_validation_subtype.tsv         score by TCGA molecular subtype, Kruskal-Wallis and Basal-vs-rest
           independent_validation_joint_cox.tsv       adjusted Cox with pEMT specificity and each published signature
         results/figures/Figure_5 (a gene level, b correlation heatmap, c replication, d subtype)
Usage:   python src/04_tcga_projection/10_independent_validation.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pemt import load_config, project_root  # noqa: E402
from pemt.published import load_published, LABELS, CITATIONS  # noqa: E402
from pemt.figstyle import apply_style, save_figure, PALETTE, mm  # noqa: E402
from pemt.survival import primary_tumours, prepare_covariates, design_matrix, fit_cox  # noqa: E402

cfg = load_config()
ROOT = project_root()
OUT = ROOT / cfg["paths"]["results_dir"] / "tcga_projection"
EXT = OUT / "external"
REF = ROOT / cfg["paths"]["references_dir"]
FIG = ROOT / "results" / "figures"
apply_style()
import matplotlib.pyplot as plt  # noqa: E402

OUR_SCORES = {
    "pEMT_specificity": "pEMT specificity",
    "P_pEMT_high": "P(pEMT-high)",
    "GS76": "76GS",
    "KS": "KS (Tan)",
    "hallmark_EMT": "Hallmark EMT",
    "puram_pemt": "Puram pEMT",
    "puram_epi_dif_1": "Puram epi. diff.",
    "MLR_mu": "MLR",
}
SUBTYPE_ORDER = ["Basal", "Classical", "Atypical", "Mesenchymal"]
SHORT = {"schinke2022_egfr_emt": "EGFR-induced EMT\n171 genes",
         "zhou2025_fdeg": "EGFR invasion\n46 genes",
         "zhou2025_invgrn": "Invasive network\n59 genes",
         "zhou2025_predictive": "Cetuximab PFS\n9 genes",
         "ourailidis2026_tbs": "Tumour budding\n28 genes"}

# gene classes for the gene-level correlates
GENE_CLASSES = {
    "Hemidesmosome / laminin-332": ["ITGB4", "ITGA6", "COL17A1", "LAMA3", "LAMB3", "LAMC2", "ITGA3", "PLEC"],
    "EGFR ligands and receptor": ["EGFR", "AREG", "EREG", "TGFA", "HBEGF", "NRG1", "BTC", "EPGN"],
    "Basal keratinocyte": ["KRT5", "KRT6A", "KRT14", "KRT17", "TP63", "SFN", "CDH3", "GJB2"],
    "Classical mesenchymal EMT": ["VIM", "CDH2", "ZEB1", "ZEB2", "TWIST1", "SNAI1", "SNAI2", "FN1"],
    "Fibroblast / ECM stroma": ["COL1A1", "COL1A2", "COL3A1", "FAP", "PDGFRB", "THY1", "POSTN", "DCN"],
}


def zmean(expr: pd.DataFrame, genes: list[str]) -> tuple[pd.Series, int, int]:
    g = [x for x in genes if x in expr.index]
    X = expr.loc[g]
    Z = X.sub(X.mean(axis=1), axis=0).div(X.std(axis=1).replace(0, 1), axis=0)
    return Z.mean(axis=0), len(g), len(genes)


def correlate(scores: pd.DataFrame, pub_cols: list[str], cohort: str) -> pd.DataFrame:
    rows = []
    for p in pub_cols:
        for s, lab in OUR_SCORES.items():
            if s not in scores or p not in scores:
                continue
            d = scores[[p, s]].dropna()
            if len(d) < 20:
                continue
            rho, pv = spearmanr(d[p], d[s])
            rows.append({"cohort": cohort, "published_signature": p, "published_label": LABELS[p],
                         "score": s, "score_label": lab, "n": len(d), "spearman_rho": round(float(rho), 3),
                         "p_value": float(pv)})
    return pd.DataFrame(rows)


def main() -> None:
    pub = load_published(REF)
    expr = pd.read_csv(ROOT / cfg["paths"]["processed_dir"] / "tcga_hnsc" / "tcga_star_tpm_log2_for_projection.tsv",
                       sep="\t", index_col=0)
    panel = pd.read_csv(OUT / "emt_score_panel_scores.tsv", sep="\t", index_col=0)
    traits = primary_tumours(pd.read_csv(OUT / "tcga_master_trait_table.tsv", sep="\t", index_col=0, low_memory=False))
    expr = expr[[c for c in panel.index if c in expr.columns]]
    scores = panel.loc[expr.columns].copy()

    print("Published signature coverage in TCGA-HNSC primaries")
    for key, genes in pub.items():
        scores[key], n_found, n_tot = zmean(expr, genes)
        print(f"  {key:24s} {n_found:>3d}/{n_tot:<3d} genes   {CITATIONS[key]}")
    pub_cols = list(pub)

    # 1. correlations, TCGA and the external cohorts
    corr = [correlate(scores, pub_cols, "TCGA-HNSC")]
    for cid in ("GSE41613", "GSE65858", "CPTAC_HNSCC"):
        f = EXT / f"{cid}_scores.tsv"
        if f.exists():
            corr.append(correlate(pd.read_csv(f, sep="\t", index_col=0), pub_cols, cid))
    corr = pd.concat(corr, ignore_index=True)
    corr.to_csv(OUT / "independent_validation_correlations.tsv", sep="\t", index=False)
    print("\nSpearman rho with published signatures, TCGA-HNSC primaries")
    piv = corr[corr["cohort"] == "TCGA-HNSC"].pivot(index="score_label", columns="published_label", values="spearman_rho")
    piv = piv.reindex([OUR_SCORES[s] for s in OUR_SCORES])
    print(piv.round(2).to_string())

    # 1b. overlap of the top classifier coefficients with the published gene sets
    # The classifier was trained on single-cell pseudobulk labels derived from the Puram programmes,
    # with no input from the EGFR, cetuximab or tumour budding gene sets.
    from scipy.stats import hypergeom
    coef = pd.read_csv(ROOT / cfg["paths"]["results_dir"] / "multinomial_classifier" / "classifier_coefficients.tsv",
                       sep="\t", index_col=0)
    universe = set(coef.index)
    top100 = coef[coef["pEMT_high"] > 0].sort_values("pEMT_high", ascending=False).head(100).index.tolist()
    ov_rows = []
    for key, genes in pub.items():
        gset = set(genes) & universe
        ov = sorted(set(top100) & gset)
        ov_rows.append({"published_signature": key, "published_label": LABELS[key], "citation": CITATIONS[key],
                        "n_genes_in_universe": len(gset), "overlap_with_top100_coefficients": len(ov),
                        "expected_by_chance": round(len(gset) * 100 / len(universe), 3),
                        "p_hypergeometric": float(hypergeom.sf(len(ov) - 1, len(universe), len(gset), 100)),
                        "overlapping_genes": ", ".join(ov)})
    ov_df = pd.DataFrame(ov_rows)
    ov_df.to_csv(OUT / "classifier_coefficient_overlap_published.tsv", sep="\t", index=False)
    print(f"\nOverlap of the top 100 pEMT-high classifier coefficients with each published set "
          f"(universe {len(universe)} genes)")
    print(ov_df[["published_label", "n_genes_in_universe", "overlap_with_top100_coefficients",
                 "expected_by_chance", "p_hypergeometric"]].to_string(index=False))
    for r in ov_df.itertuples():
        print(f"   {r.published_label}: {r.overlapping_genes}")

    # 2. gene-level correlates
    spec = scores["pEMT_specificity"]
    grow = []
    for cls, genes in GENE_CLASSES.items():
        for g in genes:
            if g not in expr.index:
                continue
            rho, pv = spearmanr(expr.loc[g], spec)
            grow.append({"gene": g, "gene_class": cls, "spearman_rho_vs_pEMT_specificity": round(float(rho), 3),
                         "p_value": float(pv), "in_published_signature": ";".join(k for k, v in pub.items() if g in v)})
    genes_df = pd.DataFrame(grow)
    genes_df.to_csv(OUT / "independent_validation_gene_level.tsv", sep="\t", index=False)
    print("\nGene-level correlation with pEMT specificity, median by class")
    print(genes_df.groupby("gene_class")["spearman_rho_vs_pEMT_specificity"].agg(["median", "min", "max", "size"]).round(2).to_string())

    # 3. TCGA molecular subtype
    sub_f = REF / "TCGA_HNSC_4class_expression_subtype.csv"
    sub_rows, subtypes = [], None
    if sub_f.exists():
        st = pd.read_csv(sub_f)
        # expression columns are TCGA-XX-XXXX-01A style, the subtype table is TCGA-XX-XXXX-01
        key = pd.Series({c: "-".join(c.split("-")[:3]) + "-01" for c in scores.index})
        m = key.map(st.set_index("sample_barcode")["RNA_subtype"])
        subtypes = m[m.isin(SUBTYPE_ORDER)]
        print(f"\nTCGA molecular subtype (2015 Nature freeze) matched for {len(subtypes)} of {len(scores)} primaries")
        print("  " + ", ".join(f"{k} {v}" for k, v in subtypes.value_counts().reindex(SUBTYPE_ORDER).items()))
        for s, lab in OUR_SCORES.items():
            if s not in scores:
                continue
            groups = [scores.loc[subtypes.index[subtypes == g], s].dropna().values for g in SUBTYPE_ORDER]
            H, pk = kruskal(*groups)
            basal = groups[SUBTYPE_ORDER.index("Basal")]
            rest = np.concatenate([g for i, g in enumerate(groups) if SUBTYPE_ORDER[i] != "Basal"])
            _, pb = mannwhitneyu(basal, rest, alternative="two-sided")
            mesen = groups[SUBTYPE_ORDER.index("Mesenchymal")]
            rest_m = np.concatenate([g for i, g in enumerate(groups) if SUBTYPE_ORDER[i] != "Mesenchymal"])
            _, pm = mannwhitneyu(mesen, rest_m, alternative="two-sided")
            sub_rows.append({"score": s, "score_label": lab, "highest_subtype": SUBTYPE_ORDER[int(np.argmax([np.median(g) for g in groups]))],
                             "kruskal_H": round(float(H), 1), "kruskal_p": float(pk),
                             "basal_median": round(float(np.median(basal)), 3), "basal_vs_rest_p": float(pb),
                             "mesenchymal_median": round(float(np.median(mesen)), 3), "mesenchymal_vs_rest_p": float(pm),
                             **{f"median_{g}": round(float(np.median(x)), 3) for g, x in zip(SUBTYPE_ORDER, groups)}})
        # the published signatures themselves, as a positive control on the subtype labels
        for p in pub_cols:
            groups = [scores.loc[subtypes.index[subtypes == g], p].dropna().values for g in SUBTYPE_ORDER]
            H, pk = kruskal(*groups)
            sub_rows.append({"score": p, "score_label": LABELS[p], "highest_subtype": SUBTYPE_ORDER[int(np.argmax([np.median(g) for g in groups]))],
                             "kruskal_H": round(float(H), 1), "kruskal_p": float(pk),
                             **{f"median_{g}": round(float(np.median(x)), 3) for g, x in zip(SUBTYPE_ORDER, groups)}})
        sub_df = pd.DataFrame(sub_rows)
        sub_df.to_csv(OUT / "independent_validation_subtype.tsv", sep="\t", index=False)
        print(sub_df[["score_label", "highest_subtype", "kruskal_p"] + [f"median_{g}" for g in SUBTYPE_ORDER]].to_string(index=False))

    # 4. joint Cox model with pEMT specificity and each published signature
    from scipy.stats import chi2
    score_cols = ["pEMT_specificity"] + pub_cols
    df = prepare_covariates(traits.join(scores[pub_cols]).assign(pEMT_specificity=scores["pEMT_specificity"]), score_cols)
    for c in score_cols:
        df[c] = (df[c] - df[c].mean()) / df[c].std()
    cov2 = ["stage_num", "age", "site_group", "hpv_positive"]
    jrows = []
    for p in pub_cols:
        d_both = design_matrix(df, "pEMT_specificity", cov2 + [p])
        cph_both, r = fit_cox(d_both)
        cph_pub, _ = fit_cox(d_both.drop(columns=["pEMT_specificity"]))
        cph_ref, _ = fit_cox(d_both.drop(columns=[p]))
        lr_ref = 2 * (cph_both.log_likelihood_ - cph_pub.log_likelihood_)
        lr_pub = 2 * (cph_both.log_likelihood_ - cph_ref.log_likelihood_)
        jrows.append({"published_signature": p, "published_label": LABELS[p], "citation": CITATIONS[p],
                      "n": int(r.loc["pEMT_specificity", "n"]), "events": int(r.loc["pEMT_specificity", "events"]),
                      "HR_pEMT_specificity": round(float(r.loc["pEMT_specificity", "HR"]), 3),
                      "CI_low_pEMT": round(float(r.loc["pEMT_specificity", "HR_low_95"]), 3),
                      "CI_up_pEMT": round(float(r.loc["pEMT_specificity", "HR_up_95"]), 3),
                      "p_pEMT_specificity": float(r.loc["pEMT_specificity", "p_value"]),
                      "HR_published": round(float(r.loc[p, "HR"]), 3), "p_published": float(r.loc[p, "p_value"]),
                      "LRT_p_adding_pEMT_to_clinical_plus_published": float(chi2.sf(lr_ref, 1)),
                      "LRT_p_adding_published_to_clinical_plus_pEMT": float(chi2.sf(lr_pub, 1))})
    jdf = pd.DataFrame(jrows)
    jdf.to_csv(OUT / "independent_validation_joint_cox.tsv", sep="\t", index=False)
    print("\nAdjusted Cox (stage, age, site, HPV), pEMT specificity and each published signature per SD")
    print(jdf[["published_label", "HR_pEMT_specificity", "p_pEMT_specificity", "HR_published", "p_published"]].to_string(index=False))

    figure(corr, genes_df, scores, subtypes, pub_cols)


def figure(corr, genes_df, scores, subtypes, pub_cols):
    from matplotlib.patches import Patch
    fig = plt.figure(figsize=(mm(180), mm(140)))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.86], hspace=0.42, wspace=0.70)

    # b: correlation heatmap, TCGA
    axA = fig.add_subplot(gs[0, 1])
    piv = corr[corr["cohort"] == "TCGA-HNSC"].pivot(index="score", columns="published_signature", values="spearman_rho")
    piv = piv.reindex(index=list(OUR_SCORES), columns=pub_cols)
    im = axA.imshow(piv.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axA.set_xticks(range(len(pub_cols)))
    axA.set_xticklabels([SHORT[p] for p in pub_cols], rotation=34, ha="right", fontsize=6)
    axA.set_yticks(range(len(OUR_SCORES)))
    axA.set_yticklabels([OUR_SCORES[s] for s in OUR_SCORES], fontsize=6.5)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                axA.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.6,
                         color="white" if abs(v) > 0.55 else PALETTE["black"])
    cb = fig.colorbar(im, ax=axA, fraction=0.045, pad=0.03)
    cb.set_label("Spearman \u03c1", fontsize=6); cb.ax.tick_params(labelsize=5.5)
    axA.tick_params(length=0)
    axA.set_title("Published programmes vs each score, TCGA-HNSC", fontsize=7, color=PALETTE["grey"])
    axA.text(-0.60, 1.08, "b", transform=axA.transAxes, fontweight="bold", fontsize=8)

    # d: scores by TCGA molecular subtype, z-scored across subtyped tumours
    axB = fig.add_subplot(gs[1, 1])
    if subtypes is not None and len(subtypes):
        show = [("pEMT_specificity", PALETTE["vermilion"], "pEMT specificity"),
                ("zhou2025_predictive", PALETTE["orange"], "Cetuximab PFS predictors"),
                ("puram_pemt", PALETTE["blue"], "Puram pEMT"),
                ("hallmark_EMT", PALETTE["grey"], "Hallmark EMT")]
        show = [s for s in show if s[0] in scores]
        width = 0.82 / len(show)
        for k, (s, c, _) in enumerate(show):
            v = scores.loc[subtypes.index, s]
            v = (v - v.mean()) / v.std()
            data = [v[subtypes == g].values for g in SUBTYPE_ORDER]
            pos = np.arange(len(SUBTYPE_ORDER)) + (k - (len(show) - 1) / 2) * width
            axB.boxplot(data, positions=pos, widths=width * 0.82, showfliers=False, patch_artist=True,
                        medianprops={"color": PALETTE["black"], "linewidth": 0.8},
                        boxprops={"facecolor": c, "edgecolor": PALETTE["black"], "linewidth": 0.4},
                        whiskerprops={"linewidth": 0.4}, capprops={"linewidth": 0.4})
        axB.set_xticks(np.arange(len(SUBTYPE_ORDER)))
        axB.set_xticklabels([f"{g}\n(n = {int((subtypes == g).sum())})" for g in SUBTYPE_ORDER], fontsize=6.3)
        axB.set_xlim(-0.6, len(SUBTYPE_ORDER) - 0.4)
        axB.set_ylim(-2.6, 3.4)
        axB.set_ylabel("Score (z across subtyped tumours)", fontsize=6.8)
        axB.axhline(0, color=PALETTE["lightgrey"], linewidth=0.6)
        axB.legend(handles=[Patch(facecolor=c, edgecolor=PALETTE["black"], linewidth=0.4, label=lab) for _, c, lab in show],
                   fontsize=5.6, frameon=False, loc="upper left", ncol=2, handlelength=1.0,
                   columnspacing=0.9, borderaxespad=0.2)
        axB.set_title("TCGA 2015 expression subtype", fontsize=7, color=PALETTE["grey"])
    axB.text(-0.22, 1.08, "d", transform=axB.transAxes, fontweight="bold", fontsize=8)

    # a: gene-level correlates, one row per class, one dot per gene
    axC = fig.add_subplot(gs[0, 0])
    order = list(GENE_CLASSES)
    colours = [PALETTE["vermilion"], PALETTE["orange"], PALETTE["yellow"], PALETTE["blue"], PALETTE["grey"]]
    rng = np.random.default_rng(0)
    for i, (cls, col) in enumerate(zip(order, colours)):
        d = genes_df[genes_df["gene_class"] == cls]["spearman_rho_vs_pEMT_specificity"].values
        y = len(order) - 1 - i
        axC.scatter(d, y + rng.uniform(-0.16, 0.16, len(d)), s=16, color=col, linewidths=0, zorder=3)
        axC.plot([np.median(d)] * 2, [y - 0.32, y + 0.32], color=PALETTE["black"], linewidth=1.2, zorder=4)
    axC.set_yticks(range(len(order))[::-1])
    axC.set_yticklabels([f"{c}\n({len(GENE_CLASSES[c])} genes)" for c in order], fontsize=6.2)
    axC.set_ylim(-0.7, len(order) - 0.3)
    axC.axvline(0, color=PALETTE["black"], linewidth=0.6)
    axC.set_xlim(-0.25, 0.85)
    axC.set_xlabel("Spearman \u03c1 with pEMT specificity", fontsize=6.8)
    axC.tick_params(axis="y", length=0)
    for g, cls, dx, ha in (("ITGB4", order[0], 0.025, "left"), ("VIM", order[3], -0.025, "right")):
        r = genes_df[genes_df["gene"] == g]
        if len(r):
            y = len(order) - 1 - order.index(cls)
            axC.annotate(g, (float(r["spearman_rho_vs_pEMT_specificity"].iloc[0]) + dx, y - 0.22), fontsize=5.6,
                         ha=ha, va="center", color=PALETTE["grey"])
    axC.set_title("Gene-level correlates of pEMT specificity", fontsize=7, color=PALETTE["grey"])
    axC.text(-0.42, 1.10, "a", transform=axC.transAxes, fontweight="bold", fontsize=8)

    # c: replication across cohorts
    axD = fig.add_subplot(gs[1, 0])
    cohorts = [c for c in ("TCGA-HNSC", "GSE41613", "CPTAC_HNSCC", "GSE65858") if c in set(corr["cohort"])]
    ccol = {"TCGA-HNSC": PALETTE["vermilion"], "GSE41613": PALETTE["orange"], "CPTAC_HNSCC": PALETTE["yellow"], "GSE65858": PALETTE["blue"]}
    sub = corr[corr["score"] == "pEMT_specificity"]
    width = 0.8 / max(len(cohorts), 1)
    for k, ch in enumerate(cohorts):
        d = sub[sub["cohort"] == ch].set_index("published_signature").reindex(pub_cols)
        x = np.arange(len(pub_cols)) + (k - (len(cohorts) - 1) / 2) * width
        axD.bar(x, d["spearman_rho"].values, width=width * 0.9, color=ccol[ch], linewidth=0, label=ch.replace("_HNSCC", "-3"))
    axD.set_xticks(np.arange(len(pub_cols)))
    axD.set_xticklabels([SHORT[p] for p in pub_cols], rotation=34, ha="right", fontsize=6)
    axD.axhline(0, color=PALETTE["black"], linewidth=0.6)
    axD.set_ylabel("Spearman \u03c1 with pEMT specificity", fontsize=6.8)
    axD.set_ylim(0, 1.0)
    axD.legend(fontsize=5.5, frameon=False, loc="upper left", ncol=4, handlelength=1.0, columnspacing=0.9, borderaxespad=0.2)
    axD.set_title("Replication in three external cohorts", fontsize=7, color=PALETTE["grey"])
    axD.text(-0.22, 1.10, "c", transform=axD.transAxes, fontweight="bold", fontsize=8)

    save_figure(fig, FIG, "Figure_5")


if __name__ == "__main__":
    main()
