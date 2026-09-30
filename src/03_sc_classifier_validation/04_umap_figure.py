"""Show the per-cell classifier output on the published TISCH2 UMAP coordinates.

For every dataset with a per_cell_probabilities_<dataset>.tsv table, the malignant and
fibroblast/stromal cells are drawn on the UMAP coordinates stored in the TISCH2 object. Panels are
coloured by cell type, P(pEMT-high), P(fibroblast-stromal-like), the Puram pEMT signature (mean z of
the signature genes within the dataset), Hallmark EMT (mean z) and a KS-style
mesenchymal-minus-epithelial mean z (Tan 2014 tumour signature). In malignant cells, P(pEMT-high) is
correlated (Spearman) with each of the three scores.

Supplementary_Figure_4 shows the Puram dataset in six panels. Supplementary_Figure_5 has one row per
dataset: cell type, P(pEMT-high) and Puram pEMT signature.

Inputs:  data/processed/single_cell/<dataset>/<dataset>_malignant_fibroblast.h5ad,
         results/sc_classifier_validation/per_cell_probabilities_<dataset>.tsv,
         data/references/emt_scores/hallmark_emt_genes.txt,
         data/references/emt_scores/ks_tumor_signature.tsv, config/signatures/puram_pemt.txt
Outputs: results/sc_classifier_validation/per_cell_emt_scores_<dataset>.tsv,
         results/sc_classifier_validation/per_cell_score_correlations.tsv,
         results/figures/Supplementary_Figure_4.svg and .png,
         results/figures/Supplementary_Figure_5.svg and .png
Usage:   python src/03_sc_classifier_validation/04_umap_figure.py
"""

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pemt import load_config, load_signatures, project_root  # noqa: E402
from pemt.figstyle import apply_style, save_figure, PALETTE, CLASS_COLOURS, mm  # noqa: E402

cfg = load_config()
ROOT = project_root()
OUT = ROOT / cfg["paths"]["results_dir"] / "sc_classifier_validation"
SC = ROOT / cfg["paths"]["processed_dir"] / "single_cell"
REF = ROOT / cfg["paths"]["references_dir"] / "emt_scores"
FIG = ROOT / "results" / "figures"
apply_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

DS_LABEL = {"HNSC_GSE103322": "HNSCC GSE103322 (Puram)", "OSCC_GSE172577": "OSCC GSE172577", "LSCC_GSE150321": "Laryngeal SCC GSE150321",
            "NPC_GSE150430": "NPC GSE150430", "NPC_GSE162025": "NPC GSE162025", "THCA_GSE148673_outgroup": "Thyroid GSE148673 (outgroup)"}


def signature_genes() -> dict[str, list[str]]:
    puram = load_signatures()["puram_pemt"]
    hallmark = [l.strip() for l in (REF / "hallmark_emt_genes.txt").read_text().splitlines() if l.strip()]
    ks = pd.read_csv(REF / "ks_tumor_signature.tsv", sep="\t")
    return {"puram_pemt": list(puram), "hallmark_EMT": hallmark,
            "ks_epi": list(ks.loc[ks["class"] == "Epi", "gene"]), "ks_mes": list(ks.loc[ks["class"] == "Mes", "gene"])}


def mean_z(adata, genes: list[str]) -> np.ndarray:
    g = [x for x in genes if x in adata.var_names]
    if not g:
        return np.full(adata.n_obs, np.nan)
    X = adata[:, g].X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1
    return ((X - mu) / sd).mean(axis=1)


def load_dataset(ds: str, sigs: dict) -> pd.DataFrame | None:
    h5 = SC / ds / f"{ds}_malignant_fibroblast.h5ad"
    pc = OUT / f"per_cell_probabilities_{ds}.tsv"
    if not h5.exists() or not pc.exists():
        return None
    a = ad.read_h5ad(h5)
    prob = pd.read_csv(pc, sep="\t", index_col=0)
    common = a.obs_names.intersection(prob.index)
    a = a[common].copy()
    df = prob.loc[common, ["P_pEMT_high", "P_epithelial_like", "P_fibroblast_stromal_like", "pEMT_specificity", "standard_cell_type"]].copy()
    df["UMAP_1"], df["UMAP_2"] = a.obs["UMAP_1"].values, a.obs["UMAP_2"].values
    df["puram_pemt"] = mean_z(a, sigs["puram_pemt"])
    df["hallmark_EMT"] = mean_z(a, sigs["hallmark_EMT"])
    df["KS_mes_minus_epi"] = mean_z(a, sigs["ks_mes"]) - mean_z(a, sigs["ks_epi"])
    df.to_csv(OUT / f"per_cell_emt_scores_{ds}.tsv", sep="\t")
    return df


def scatter(ax, df, col, label, cmap="viridis", vmin=None, vmax=None):
    order = np.argsort(df[col].values)
    sc_ = ax.scatter(df["UMAP_1"].values[order], df["UMAP_2"].values[order], c=df[col].values[order], s=1.2, cmap=cmap, vmin=vmin, vmax=vmax,
                     linewidths=0, rasterized=True)
    ax.set_title(label, fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = plt.colorbar(sc_, ax=ax, fraction=0.05, pad=0.02)
    cb.ax.tick_params(labelsize=5.5)
    return sc_


def celltype_panel(ax, df, title):
    cols = df["standard_cell_type"].map({"malignant": PALETTE["orange"], "fibroblast_stromal": PALETTE["green"]}).fillna(PALETTE["grey"])
    ax.scatter(df["UMAP_1"], df["UMAP_2"], c=cols, s=1.2, linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    n_m, n_f = int((df["standard_cell_type"] == "malignant").sum()), int((df["standard_cell_type"] == "fibroblast_stromal").sum())
    ax.legend(handles=[Patch(color=PALETTE["orange"], label=f"malignant ({n_m})"), Patch(color=PALETTE["green"], label=f"fibroblast/stromal ({n_f})")],
              loc="upper left", bbox_to_anchor=(0.0, 0.0), ncol=2, fontsize=5.5, frameon=False,
              handlelength=1.0, handletextpad=0.4, columnspacing=1.0)


def main() -> None:
    sigs = signature_genes()
    rows = []
    data = {}
    for ds in DS_LABEL:
        df = load_dataset(ds, sigs)
        if df is None:
            continue
        data[ds] = df
        m = df[df["standard_cell_type"] == "malignant"]
        for col in ("puram_pemt", "hallmark_EMT", "KS_mes_minus_epi"):
            r, p = spearmanr(m["P_pEMT_high"], m[col])
            rows.append({"dataset_id": ds, "score": col, "n_malignant_cells": len(m), "spearman_vs_P_pEMT_high": r, "p": p})
        print(f"{ds}: {len(df)} cells, malignant P(pEMT-high) vs Puram {rows[-3]['spearman_vs_P_pEMT_high']:+.2f}, Hallmark {rows[-2]['spearman_vs_P_pEMT_high']:+.2f}, KS {rows[-1]['spearman_vs_P_pEMT_high']:+.2f}")
    pd.DataFrame(rows).to_csv(OUT / "per_cell_score_correlations.tsv", sep="\t", index=False)

    # Puram dataset: six panels
    if "HNSC_GSE103322" in data:
        df = data["HNSC_GSE103322"]
        fig, axes = plt.subplots(2, 3, figsize=(mm(180), mm(105)))
        axes = axes.ravel()
        celltype_panel(axes[0], df, "Cell type (TISCH2 annotation)")
        scatter(axes[1], df, "P_pEMT_high", "P(pEMT-high)", vmin=0, vmax=1)
        scatter(axes[2], df, "P_fibroblast_stromal_like", "P(fibroblast / stromal-like)", vmin=0, vmax=1)
        scatter(axes[3], df, "puram_pemt", "Puram pEMT signature (mean z)", cmap="magma")
        scatter(axes[4], df, "hallmark_EMT", "Hallmark EMT (mean z)", cmap="magma")
        scatter(axes[5], df, "KS_mes_minus_epi", "Tan signature, mes − epi (mean z)", cmap="magma")
        for ax, letter in zip(axes, "abcdef"):
            ax.text(-0.02, 1.04, letter, transform=ax.transAxes, fontweight="bold", fontsize=8, ha="right", va="bottom")
        fig.tight_layout()
        save_figure(fig, FIG, "Supplementary_Figure_4")

    # all datasets: one row each
    ds_list = list(data)
    fig, axes = plt.subplots(len(ds_list), 3, figsize=(mm(150), mm(42 * len(ds_list))))
    for i, ds in enumerate(ds_list):
        df = data[ds]
        celltype_panel(axes[i, 0], df, DS_LABEL[ds])
        scatter(axes[i, 1], df, "P_pEMT_high", "P(pEMT-high)", vmin=0, vmax=1)
        scatter(axes[i, 2], df, "puram_pemt", "Puram pEMT signature (mean z)", cmap="magma")
    fig.tight_layout()
    save_figure(fig, FIG, "Supplementary_Figure_5")


if __name__ == "__main__":
    main()
