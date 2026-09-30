"""Test whether the pEMT-high state is visible without the genes used to define it.

The training labels are quantile thresholds on the Puram pEMT and epithelial-differentiation scores,
and the same genes are classifier features, so held-out accuracy partly restates the label
definition. Here every gene of the six Puram programmes (pEMT, epithelial differentiation 1 and 2,
hypoxia, cell cycle, stress) is removed from the GSE103322 malignant cells, and four analyses are run
on the remaining genes:

  1. Unsupervised. Leiden clusters built with no labels are compared with the pEMT-high and
     epithelial-like labels by adjusted Rand index (one-sided p against 1,000 label permutations)
     and adjusted mutual information, and the per-cluster label composition is reported.
  2. Supervised, patient held out. An L2 logistic regression (C = 0.1) separates pEMT-high from
     epithelial-like cells with leave-one-patient-out folds over patients with at least 5 labelled
     cells, and the pooled AUROC is reported.
     Folds whose training or test set lacks either class are skipped.
  3. Separation in the embedding. Silhouette of the two labelled populations in the principal
     component space, against 200 label permutations.
  4. Overlap between the top 100 positive pEMT-high coefficients and the Puram programme genes.

The embedding keeps patients with at least 20 malignant cells and genes detected in at least 10 cells
that vary within every patient (ComBat needs this). It then takes 2,000 highly variable genes (Seurat
flavour), applies ComBat on patient, scales (max 10), and computes 30 principal components, a
15-neighbour graph, Leiden clusters (resolution 1.0) and a UMAP. Supplementary_Figure_7 shows the
labelled cells on the UMAP (a), each cluster coloured by its pEMT-high share (b), and the pEMT-high
share per cluster with the ARI and held-out AUROC (c).

Inputs:  data/processed/single_cell/HNSC_GSE103322/HNSC_GSE103322_labelled.h5ad,
         config/signatures/puram_*.txt,
         results/multinomial_classifier/top_positive_genes_pEMT_high.tsv
Outputs: results/label_free/cluster_composition.tsv, results/label_free/label_free_summary.tsv,
         results/figures/Supplementary_Figure_7.svg and .png
Usage:   python src/03_sc_classifier_validation/05_label_free_embedding.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SC = ROOT / "data" / "processed" / "single_cell" / "HNSC_GSE103322"
SIG = ROOT / "config" / "signatures"
CLF = ROOT / "results" / "multinomial_classifier"
OUT = ROOT / "results" / "label_free"
FIG = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))
from pemt.figstyle import apply_style, save_figure, PALETTE, CLASS_COLOURS, mm  # noqa: E402

apply_style()
import matplotlib.pyplot as plt  # noqa: E402

SEED = 20260920


def puram_genes() -> set[str]:
    genes: set[str] = set()
    for f in sorted(SIG.glob("puram_*.txt")):
        genes |= {l.strip() for l in f.read_text().splitlines() if l.strip() and not l.startswith("#")}
    return genes


def main() -> None:
    import anndata as ad
    import scanpy as sc
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, silhouette_score
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score

    a = ad.read_h5ad(SC / "HNSC_GSE103322_labelled.h5ad")
    a = a[a.obs["standard_cell_type"] == "malignant"].copy()
    drop = puram_genes()
    keep = [g for g in a.var_names if g not in drop]
    print(f"{a.n_obs} malignant cells, removing {a.n_vars - len(keep)} Puram programme genes "
          f"from {a.n_vars}, leaving {len(keep)}")
    a = a[:, keep].copy()

    lbl = a.obs["training_label"].astype(str)
    print("labelled cells:", dict(lbl.value_counts()))

    # Analysis 4: Puram genes among the top pEMT-high coefficients
    top = pd.read_csv(CLF / "top_positive_genes_pEMT_high.tsv", sep="\t").iloc[:100, 0].tolist()
    n_puram = sum(g in drop for g in top)
    print(f"\ntop 100 pEMT-high coefficients: {n_puram} are Puram programme genes, "
          f"{100 - n_puram} are not")

    # Embedding on non-Puram genes, built without the labels.
    # ComBat returns NaN for a gene that is constant within a batch, so small patients and genes
    # constant in any patient are dropped first.
    big = a.obs["Patient"].value_counts()
    a = a[a.obs["Patient"].isin(big[big >= 20].index)].copy()
    sc.pp.filter_genes(a, min_cells=10)
    X = np.asarray(a.X.todense()) if hasattr(a.X, "todense") else np.asarray(a.X)
    pat = a.obs["Patient"].astype(str).values
    varies = np.ones(a.n_vars, dtype=bool)
    for p in np.unique(pat):
        varies &= X[pat == p].std(axis=0) > 0
    a = a[:, varies].copy()
    print(f"after batch-safe filtering: {a.n_obs} cells from {a.obs['Patient'].nunique()} patients, "
          f"{a.n_vars} genes")

    sc.pp.highly_variable_genes(a, n_top_genes=2000, flavor="seurat")
    a = a[:, a.var["highly_variable"]].copy()
    sc.pp.combat(a, key="Patient")
    a.X = np.nan_to_num(np.asarray(a.X), nan=0.0, posinf=0.0, neginf=0.0)
    sc.pp.scale(a, max_value=10)
    sc.tl.pca(a, n_comps=30, svd_solver="arpack", random_state=SEED)
    sc.pp.neighbors(a, n_neighbors=15, random_state=SEED)
    sc.tl.leiden(a, resolution=1.0, key_added="leiden", random_state=SEED, flavor="igraph",
                 n_iterations=2, directed=False)
    sc.tl.umap(a, random_state=SEED)
    print(f"\n{a.obs['leiden'].nunique()} Leiden clusters on {a.n_vars} non-Puram genes")

    lab = a.obs["training_label"].astype(str)
    m = lab.isin(["pEMT_high", "epithelial_like"]).values
    y = (lab[m] == "pEMT_high").astype(int).values
    cl = a.obs["leiden"].astype(str)[m].values

    # Analysis 1: cluster-label agreement against a permutation null
    ari, ami = adjusted_rand_score(y, cl), adjusted_mutual_info_score(y, cl)
    rng = np.random.default_rng(SEED)
    null = np.array([adjusted_rand_score(rng.permutation(y), cl) for _ in range(1000)])
    p_ari = float((null >= ari).mean())
    print(f"\nunsupervised: ARI {ari:.3f} (permutation p = {p_ari:.3g}), AMI {ami:.3f}")

    comp = pd.crosstab(a.obs["leiden"], lab)
    for c in ["pEMT_high", "epithelial_like"]:
        if c not in comp:
            comp[c] = 0
    comp["pct_pEMT_high_of_labelled"] = (
        100 * comp["pEMT_high"] / (comp["pEMT_high"] + comp["epithelial_like"]).replace(0, np.nan))
    comp.to_csv(OUT / "cluster_composition.tsv", sep="\t")
    print("\nper-cluster composition of the labelled cells:")
    print(comp[["pEMT_high", "epithelial_like", "pct_pEMT_high_of_labelled"]].round(1).to_string())

    # Analysis 3: silhouette in PC space against a label-permutation null
    pcs = a.obsm["X_pca"][m]
    sil = silhouette_score(pcs, y)
    null_s = np.array([silhouette_score(pcs, rng.permutation(y)) for _ in range(200)])
    print(f"\nsilhouette of the two labelled states in non-Puram PC space: {sil:.3f} "
          f"(null mean {null_s.mean():.3f}, p = {float((null_s >= sil).mean()):.3g})")

    # Analysis 2: leave-one-patient-out classifier on non-Puram genes
    X = a.X[m]
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    groups = a.obs["Patient"].astype(str)[m].values
    ok = [g for g in np.unique(groups) if len(np.unique(y[groups == g])) > 0 and (groups == g).sum() >= 5]
    keep_m = np.isin(groups, ok)
    X, y2, groups = X[keep_m], y[keep_m], groups[keep_m]
    preds, truth = [], []
    for tr, te in LeaveOneGroupOut().split(X, y2, groups):
        if len(np.unique(y2[tr])) < 2 or len(np.unique(y2[te])) < 2:
            continue
        clf = LogisticRegression(penalty="l2", C=0.1, max_iter=2000, random_state=SEED)
        clf.fit(X[tr], y2[tr])
        preds.append(clf.predict_proba(X[te])[:, 1])
        truth.append(y2[te])
    auc = roc_auc_score(np.concatenate(truth), np.concatenate(preds))
    print(f"\nsupervised, leave-one-patient-out on non-Puram genes only: "
          f"pooled AUROC {auc:.3f} over {len(preds)} evaluable patients, n = {len(np.concatenate(truth))} cells")

    pd.DataFrame([{"n_cells": int(a.n_obs), "n_genes_removed": len(drop),
                   "n_genes_used": int(a.n_vars), "n_clusters": int(a.obs["leiden"].nunique()),
                   "ARI": ari, "ARI_permutation_p": p_ari, "AMI": ami,
                   "silhouette": sil, "silhouette_null_mean": float(null_s.mean()),
                   "loPo_AUROC_non_puram_genes": auc,
                   "top100_coefficients_that_are_puram_genes": n_puram}]).to_csv(
        OUT / "label_free_summary.tsv", sep="\t", index=False)

    # Supplementary_Figure_7
    fig, axes = plt.subplots(1, 3, figsize=(mm(180), mm(62)),
                             gridspec_kw={"width_ratios": [1, 1.15, 0.95], "wspace": 0.62})
    um = a.obsm["X_umap"]

    ax = axes[0]
    ax.scatter(um[:, 0], um[:, 1], s=2, c=PALETTE["lightgrey"], linewidths=0, rasterized=True)
    for name, col in [("epithelial_like", CLASS_COLOURS["epithelial_like"]),
                      ("pEMT_high", CLASS_COLOURS["pEMT_high"])]:
        s = (lab == name).values
        ax.scatter(um[s, 0], um[s, 1], s=4.5, c=col, linewidths=0, rasterized=True,
                   label=name.replace("_", "-").replace("-like", "-like"))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.legend(fontsize=6, loc="best", markerscale=2)
    ax.set_title("a", loc="left", fontweight="bold")

    # Cells coloured by the pEMT-high share of their cluster, with no text labels on the embedding.
    ax = axes[1]
    share = comp["pct_pEMT_high_of_labelled"]
    vals = a.obs["leiden"].map(share).astype(float).values
    unlab = np.isnan(vals)
    ax.scatter(um[unlab, 0], um[unlab, 1], s=2.5, c=PALETTE["lightgrey"], linewidths=0,
               rasterized=True)
    sm = ax.scatter(um[~unlab, 0], um[~unlab, 1], s=2.5, c=vals[~unlab], cmap="RdBu_r",
                    vmin=0, vmax=100, linewidths=0, rasterized=True)
    cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("pEMT-high (%)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.4)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title("b", loc="left", fontweight="bold")

    ax = axes[2]
    cc = comp.dropna(subset=["pct_pEMT_high_of_labelled"]).sort_values("pct_pEMT_high_of_labelled")
    yv = np.arange(len(cc))
    ax.barh(yv, cc["pct_pEMT_high_of_labelled"], color=CLASS_COLOURS["pEMT_high"], alpha=0.9, height=0.7)
    ax.axvline(100 * y.mean(), ls="--", lw=0.7, color=PALETTE["grey"])
    ax.set_yticks(yv); ax.set_yticklabels(cc.index, fontsize=6.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("pEMT-high share of labelled cells (%)")
    ax.set_ylabel("Cluster")
    ax.text(0.97, 0.05, f"ARI {ari:.2f}, p {p_ari:.3g}\nheld-out patient\nAUROC {auc:.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, linespacing=1.5)
    ax.set_title("c", loc="left", fontweight="bold")

    fig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.19)
    save_figure(fig, FIG, "Supplementary_Figure_7")
    print(f"\nwrote {OUT}/ and {FIG}/Supplementary_Figure_7.svg/.png")


if __name__ == "__main__":
    main()
