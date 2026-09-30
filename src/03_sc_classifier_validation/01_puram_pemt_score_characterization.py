"""Characterise the raw Puram pEMT score in the GSE103322 dataset, before the classifier is applied.

All cells of the dataset are scored with scanpy.tl.score_genes on the Puram pEMT genes (after
normalisation and log1p if the matrix holds raw counts). UMAPs on the TISCH2 coordinates are
coloured by lineage, pEMT score and site, and a violin plot shows the score by lineage.
Malignant cells are compared with each other lineage of at least 5 cells by two-sided Mann-Whitney
tests with Bonferroni correction, and a Kruskal-Wallis test across lineages is printed. Patients
with malignant cells in a lymph node are LN+. Primary-tumour malignant cells of LN+ and LN- patients
are compared by one-sided Mann-Whitney tests (LN+ > LN-), at cell level and on patient medians.

Inputs:  data/processed/single_cell/HNSC_GSE103322/HNSC_GSE103322.h5ad,
         config/signatures/puram_pemt.txt
Outputs: results/sc_classifier_validation/umap_puram_all_lineages.svg, umap_puram_pemt_score.svg,
         umap_puram_site.svg, violin_pemt_by_lineage.svg, pairwise_lineage_stats.tsv,
         ln_pos_vs_neg_patient_stats.tsv
Usage:   python src/03_sc_classifier_validation/01_puram_pemt_score_characterization.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pemt import load_config, load_signatures, project_root  # noqa: E402

sc.settings.verbosity = 1

cfg = load_config()
ROOT = project_root()
OUT = ROOT / cfg["paths"]["results_dir"] / "sc_classifier_validation"
OUT.mkdir(parents=True, exist_ok=True)

H5AD_FULL = ROOT / cfg["paths"]["processed_dir"] / "single_cell" / "HNSC_GSE103322" / "HNSC_GSE103322.h5ad"
PEMT_GENES = load_signatures()["puram_pemt"]

LINEAGE_COL = "Celltype (major-lineage)"
SITE_COL = "Site"
PALETTE = {
    "Malignant": "#E41A1C", "Fibroblasts": "#FF7F00", "Myofibroblasts": "#FDBF6F",
    "CD8Tex": "#4DAF4A", "CD8T": "#33A02C", "CD4Tconv": "#1F78B4",
    "Mono/Macro": "#984EA3", "Endothelial": "#A65628", "Plasma": "#F781BF",
    "Mast": "#999999", "Myocyte": "#B2DF8A",
}


def main() -> None:
    print("Loading full GSE103322 h5ad...")
    adata = sc.read_h5ad(H5AD_FULL)
    print(f"  {adata.n_obs} cells x {adata.n_vars} genes")

    if adata.X.max() > 50:
        print("  Normalising raw counts...")
        sc.pp.normalize_total(adata, target_sum=cfg["single_cell"]["normalize_target_sum"])
        sc.pp.log1p(adata)

    present = [g for g in PEMT_GENES if g in adata.var_names]
    missing = [g for g in PEMT_GENES if g not in adata.var_names]
    print(f"\npEMT signature: {len(present)}/{len(PEMT_GENES)} genes present "
          f"(missing: {missing if missing else 'none'})")
    sc.tl.score_genes(adata, present, score_name="puram_pemt_score")

    lineage_col = LINEAGE_COL if LINEAGE_COL in adata.obs.columns else "standard_cell_type"
    adata.obsm["X_umap"] = adata.obs[["UMAP_1", "UMAP_2"]].values

    if hasattr(adata.obs[lineage_col], "cat"):
        lineages = adata.obs[lineage_col].cat.categories.tolist()
    else:
        lineages = sorted(adata.obs[lineage_col].unique().tolist())

    print("\nPlotting UMAP by lineage...")
    fig, ax = plt.subplots(figsize=(7, 6))
    for lin in lineages:
        mask = adata.obs[lineage_col] == lin
        ax.scatter(adata.obsm["X_umap"][mask, 0], adata.obsm["X_umap"][mask, 1],
                   c=PALETTE.get(lin, "#CCCCCC"), s=3, alpha=0.7, linewidths=0,
                   label=lin, rasterized=True)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title("GSE103322, cell lineages")
    ax.legend(handles=[mpatches.Patch(color=PALETTE.get(l, "#CCCCCC"), label=l) for l in lineages],
              bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "umap_puram_all_lineages.svg", dpi=600, bbox_inches="tight")
    plt.close(fig)

    print("Plotting UMAP by pEMT score...")
    scores = adata.obs["puram_pemt_score"].values
    vmin, vmax = np.percentile(scores, 2), np.percentile(scores, 98)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    sc_plot = ax.scatter(adata.obsm["X_umap"][:, 0], adata.obsm["X_umap"][:, 1],
                         c=scores, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                         s=3, alpha=0.8, linewidths=0, rasterized=True)
    plt.colorbar(sc_plot, ax=ax, shrink=0.7, label="Puram pEMT score")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title("GSE103322, Puram pEMT score (all lineages)")
    fig.tight_layout()
    fig.savefig(OUT / "umap_puram_pemt_score.svg", dpi=600, bbox_inches="tight")
    plt.close(fig)

    print("Plotting UMAP by Site...")
    site_colors = {"Primary": "#2166AC", "Lymph node": "#D6604D"}
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for site, col in site_colors.items():
        if site not in adata.obs[SITE_COL].values:
            continue
        mask = adata.obs[SITE_COL] == site
        ax.scatter(adata.obsm["X_umap"][mask, 0], adata.obsm["X_umap"][mask, 1],
                   c=col, s=3, alpha=0.7, linewidths=0, label=site, rasterized=True)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title("GSE103322, site (primary vs lymph node)")
    ax.legend(fontsize=9, frameon=False, markerscale=3)
    fig.tight_layout()
    fig.savefig(OUT / "umap_puram_site.svg", dpi=600, bbox_inches="tight")
    plt.close(fig)

    print("Plotting violin by lineage...")
    lin_medians = {l: adata.obs.loc[adata.obs[lineage_col] == l, "puram_pemt_score"].median()
                   for l in lineages}
    ordered = sorted(lineages, key=lambda l: lin_medians[l], reverse=True)
    plot_data = [adata.obs.loc[adata.obs[lineage_col] == l, "puram_pemt_score"].values for l in ordered]
    fig, ax = plt.subplots(figsize=(max(10, len(ordered) * 0.9), 5))
    parts = ax.violinplot(plot_data, positions=range(len(ordered)),
                          showmedians=True, showextrema=False)
    for body, l in zip(parts["bodies"], ordered):
        body.set_facecolor(PALETTE.get(l, "#CCCCCC")); body.set_alpha(0.75)
    parts["cmedians"].set_colors("black"); parts["cmedians"].set_linewidth(1.5)
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(ordered, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Puram pEMT score")
    ax.set_title("GSE103322, pEMT score by lineage (median descending)")
    ax.axhline(0, color="grey", linewidth=0.7, linestyle="--")
    fig.tight_layout()
    fig.savefig(OUT / "violin_pemt_by_lineage.svg", dpi=600, bbox_inches="tight")
    plt.close(fig)

    # Malignant versus each other lineage, Bonferroni-corrected
    print("\nRunning pairwise lineage stats...")
    mal_scores = adata.obs.loc[adata.obs[lineage_col] == "Malignant", "puram_pemt_score"].values
    rows = []
    for lin in [l for l in lineages if l != "Malignant"]:
        other = adata.obs.loc[adata.obs[lineage_col] == lin, "puram_pemt_score"].values
        if len(other) < 5:
            continue
        stat, p = mannwhitneyu(mal_scores, other, alternative="two-sided")
        rows.append({"comparison": f"Malignant vs {lin}",
                     "n_malignant": len(mal_scores), "n_other": len(other),
                     "median_malignant": round(float(np.median(mal_scores)), 4),
                     "median_other": round(float(np.median(other)), 4),
                     "U_statistic": round(stat, 1), "p_raw": p})
    pairwise_df = pd.DataFrame(rows)
    if not pairwise_df.empty:
        _, p_adj, _, _ = multipletests(pairwise_df["p_raw"], method="bonferroni")
        pairwise_df["p_bonferroni"] = p_adj
        pairwise_df["significant_p05"] = p_adj < 0.05
        pairwise_df["p_raw"] = pairwise_df["p_raw"].round(6)
        pairwise_df["p_bonferroni"] = pairwise_df["p_bonferroni"].round(6)
    pairwise_df.to_csv(OUT / "pairwise_lineage_stats.tsv", sep="\t", index=False)

    all_groups = [adata.obs.loc[adata.obs[lineage_col] == l, "puram_pemt_score"].values
                  for l in lineages if adata.obs[lineage_col].eq(l).sum() >= 5]
    kw_stat, kw_p = kruskal(*all_groups)
    print(f"  Kruskal-Wallis: H={kw_stat:.2f}, p={kw_p:.2e}")

    # LN+ versus LN- patients, primary-site malignant cells only
    print("Running LN+ vs LN- patient comparison...")
    mal_obs = adata.obs[adata.obs[lineage_col] == "Malignant"].copy()
    ln_patients = set(mal_obs.loc[mal_obs[SITE_COL] == "Lymph node", "Patient"].unique())
    primary = mal_obs[mal_obs[SITE_COL] == "Primary"].copy()
    primary["ln_status"] = primary["Patient"].apply(
        lambda p: "LN_pos" if p in ln_patients else "LN_neg"
    )

    pos = primary.loc[primary["ln_status"] == "LN_pos", "puram_pemt_score"].values
    neg = primary.loc[primary["ln_status"] == "LN_neg", "puram_pemt_score"].values

    patient_result: dict = {}
    if len(pos) >= 5 and len(neg) >= 5:
        _, p_cell = mannwhitneyu(pos, neg, alternative="greater")
        pos_meds = (primary.loc[primary["ln_status"] == "LN_pos"]
                    .groupby("Patient", observed=True)["puram_pemt_score"].median().dropna().values)
        neg_meds = (primary.loc[primary["ln_status"] == "LN_neg"]
                    .groupby("Patient", observed=True)["puram_pemt_score"].median().dropna().values)
        _, p_pat = mannwhitneyu(pos_meds, neg_meds, alternative="greater")
        patient_result = {
            "test": "Primary tumour pEMT: LN+ > LN- (Mann-Whitney one-sided)",
            "n_ln_pos_patients": len(pos_meds), "n_ln_neg_patients": len(neg_meds),
            "n_ln_pos_cells": len(pos), "n_ln_neg_cells": len(neg),
            "median_pemt_ln_pos_cells": round(float(np.median(pos)), 4),
            "median_pemt_ln_neg_cells": round(float(np.median(neg)), 4),
            "p_cell_level": round(p_cell, 6),
            "median_pemt_ln_pos_patient": round(float(np.median(pos_meds)), 4),
            "median_pemt_ln_neg_patient": round(float(np.median(neg_meds)), 4),
            "p_patient_level": round(p_pat, 6),
            "significant_cell_p05": p_cell < 0.05,
            "significant_patient_p05": p_pat < 0.05,
        }
        print(f"  patient-level p={p_pat:.4f}, "
              f"LN+ patient median={np.median(pos_meds):.3f}, "
              f"LN- patient median={np.median(neg_meds):.3f}")

    pd.DataFrame([patient_result]).to_csv(OUT / "ln_pos_vs_neg_patient_stats.tsv", sep="\t", index=False)
    print("Done. Outputs in", OUT)


if __name__ == "__main__":
    main()
