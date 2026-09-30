"""Partition the genes of the Puram pEMT programme by the cell types that express them in GSE103322.

Of the 100 genes in the Puram pEMT programme, VIM is the only classical mesenchymal marker. ZEB1,
ZEB2, SNAI1, SNAI2, TWIST1 and CDH2 are absent. The list holds laminin-332 and hemidesmosome
components (LAMA3, LAMB3, LAMC2, COL17A1, ITGA6) and a larger set of fibrillar collagens, matrix
metalloproteinases and matricellular proteins (COL1A1, MMP1, MMP2, MMP10, INHBA, TGFBI, THBS1,
PLAU, SERPINE1, TNC).

For each detected gene, log1p-normalised expression is returned to the linear scale. The table
gives the share of total expression contributed by malignant cells, CAFs (fibroblasts and
myofibroblasts), endothelial cells and immune cells, and the percentage of cells of each type that
express the gene. A gene is malignant-expressed when its malignant share exceeds its CAF share and
CAF-expressed otherwise. Within malignant cells, the log2 fold change between pEMT-high and
epithelial-like cells uses a pseudocount of 0.1. The malignant share of the hemidesmosome and
laminin-332 genes is compared with that of the other pEMT genes by a one-sided Mann-Whitney test.
The pEMT genes among the top 100 positive coefficients of the pEMT-high class of the multinomial
classifier are counted.

Figure_2 has five panels: (a) share of expression by cell type for every gene, (b) malignant share
by gene family, (c) malignant share against the pEMT-high versus epithelial-like fold change,
(d) the per-cell malignant share in GSE103322 against GSE181919, and (e) correlations of the M06 and
M13 co-expression module eigengenes with the classifier outputs and signature scores. Panel d is
drawn only when results/pemt_partition/replication_per_gene.tsv exists. That file is written by
02_partition_replication.py, which reads the gene table written here, so rerun this script after
02 to include panel d.

Inputs:  data/processed/single_cell/HNSC_GSE103322/HNSC_GSE103322.h5ad
         data/processed/single_cell/HNSC_GSE103322/HNSC_GSE103322_labelled.h5ad
         config/signatures/puram_pemt.txt
         results/multinomial_classifier/top_positive_genes_pEMT_high.tsv
         results/wgcna/module_annotation.tsv
         results/wgcna/module_trait_correlations.tsv
         results/pemt_partition/replication_per_gene.tsv (optional, panel d)
Outputs: results/pemt_partition/puram_pemt_gene_partition.tsv
         results/figures/Figure_2
Usage:   python src/09_gene_partition/01_pemt_gene_partition.py
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
OUT = ROOT / "results" / "pemt_partition"
FIG = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))
from pemt.figstyle import apply_style, save_figure, PALETTE, mm  # noqa: E402

apply_style()
import matplotlib.pyplot as plt  # noqa: E402

# Gene families, for annotation and the family comparison only. They do not enter the partition.
HEMI = {"LAMA3", "LAMB3", "LAMC2", "COL17A1", "ITGA6", "ITGB4", "ITGA3", "DST", "PLEC"}
MESEN = {"VIM", "CDH2", "ZEB1", "ZEB2", "SNAI1", "SNAI2", "TWIST1"}


def main() -> None:
    import anndata as ad
    import scipy.sparse as sp

    genes = [l.strip() for l in (SIG / "puram_pemt.txt").read_text().splitlines()
             if l.strip() and not l.startswith("#")]

    a = ad.read_h5ad(SC / "HNSC_GSE103322.h5ad")
    lab = ad.read_h5ad(SC / "HNSC_GSE103322_labelled.h5ad")
    a.obs["training_label"] = lab.obs["training_label"].reindex(a.obs_names)

    major = a.obs["Celltype (major-lineage)"].astype(str)
    grp = pd.Series("other", index=a.obs_names, dtype=object)
    grp[major.isin(["Fibroblasts", "Myofibroblasts"])] = "CAF"
    grp[major.eq("Malignant")] = "malignant"
    grp[major.eq("Endothelial")] = "endothelial"
    grp[major.isin(["CD8Tex", "CD4Tconv", "CD8T", "Plasma", "Mast", "Mono/Macro"])] = "immune"
    a.obs["group"] = grp
    print("cells:", dict(a.obs["group"].value_counts()))

    lbl = a.obs["training_label"].astype(str)
    mal_hi = (a.obs["group"] == "malignant") & lbl.eq("pEMT_high")
    mal_ep = (a.obs["group"] == "malignant") & lbl.eq("epithelial_like")

    rows = []
    for g in genes:
        if g not in a.var_names:
            continue
        col = a.X[:, a.var_names.get_loc(g)]
        v = np.asarray(col.todense()).ravel() if sp.issparse(col) else np.asarray(col).ravel()
        v = np.expm1(v)  # the matrix is log1p-normalised. Shares are taken on the linear scale.
        tot = v.sum()
        if tot <= 0:
            continue
        rec = {"gene": g, "total_expression": float(tot)}
        for grp_name in ["malignant", "CAF", "endothelial", "immune"]:
            m = (a.obs["group"] == grp_name).values
            rec[f"share_{grp_name}"] = float(v[m].sum() / tot)
            rec[f"pct_expressing_{grp_name}"] = float((v[m] > 0).mean() * 100)
        # within malignant cells, pEMT-high vs epithelial-like
        hi, ep = v[mal_hi.values], v[mal_ep.values]
        rec["mean_pEMT_high"] = float(hi.mean())
        rec["mean_epithelial_like"] = float(ep.mean())
        rec["log2FC_pEMT_high_vs_epithelial"] = float(
            np.log2((hi.mean() + 0.1) / (ep.mean() + 0.1)))
        rec["family"] = ("hemidesmosome / laminin-332" if g in HEMI else
                         "classical mesenchymal" if g in MESEN else "other pEMT gene")
        rows.append(rec)

    df = pd.DataFrame(rows)
    df["arm"] = np.where(df["share_malignant"] > df["share_CAF"], "malignant-expressed",
                         "CAF-expressed")
    df = df.sort_values("share_malignant", ascending=False)
    df.to_csv(OUT / "puram_pemt_gene_partition.tsv", sep="\t", index=False)

    n_mal = int((df["arm"] == "malignant-expressed").sum())
    print(f"\n{len(df)} of the {len(genes)} Puram pEMT genes are detected")
    print(f"  malignant-expressed (share > CAF): {n_mal}")
    print(f"  CAF-expressed:                     {len(df) - n_mal}")

    print("\nmost malignant-expressed pEMT genes:")
    print(df.head(12)[["gene", "share_malignant", "share_CAF", "family"]].round(3).to_string(index=False))
    print("\nmost CAF-expressed pEMT genes:")
    print(df.tail(12)[["gene", "share_malignant", "share_CAF", "family"]].round(3).to_string(index=False))

    hemi = df[df["family"] == "hemidesmosome / laminin-332"]
    print(f"\nhemidesmosome / laminin-332 genes in the list (n = {len(hemi)}): "
          f"median malignant share {hemi['share_malignant'].median():.2f}")
    print(hemi[["gene", "share_malignant", "share_CAF"]].round(3).to_string(index=False))
    other = df[df["family"] == "other pEMT gene"]
    print(f"remaining pEMT genes (n = {len(other)}): "
          f"median malignant share {other['share_malignant'].median():.2f}")
    from scipy.stats import mannwhitneyu
    u = mannwhitneyu(hemi["share_malignant"], other["share_malignant"], alternative="greater")
    print(f"hemidesmosome vs the rest, malignant share: Mann-Whitney one-sided p = {u.pvalue:.2g}")

    # how the two arms map onto the classifier
    top = pd.read_csv(CLF / "top_positive_genes_pEMT_high.tsv", sep="\t").iloc[:100, 0].tolist()
    df["in_classifier_top100"] = df["gene"].isin(top)
    inn = df[df["in_classifier_top100"]]
    print(f"\npEMT genes among the classifier's top 100 coefficients: {len(inn)} "
          f"({int((inn['arm'] == 'malignant-expressed').sum())} malignant-expressed, "
          f"{int((inn['arm'] == 'CAF-expressed').sum())} CAF-expressed)")
    if len(inn):
        print(inn[["gene", "share_malignant", "arm"]].round(3).to_string(index=False))
    df.to_csv(OUT / "puram_pemt_gene_partition.tsv", sep="\t", index=False)

    # ---------------------------------------------------------------- figure
    # Panels a to c split the gene list by cell type, panel d is the replication and panel e shows
    # modules M06 and M13. The full 13-module by 10-trait matrix is drawn as a supplementary figure.
    fig = plt.figure(figsize=(mm(180), mm(112)))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.0, 1.0], height_ratios=[1.0, 0.46],
                          wspace=0.62, hspace=0.62)

    # (a) the whole programme, ranked by malignant share
    ax = fig.add_subplot(gs[0, 0])
    d = df.sort_values("share_malignant")
    y = np.arange(len(d))
    ax.barh(y, d["share_malignant"] * 100, height=1.0, color=PALETTE["vermilion"], alpha=0.9)
    ax.barh(y, d["share_CAF"] * 100, left=d["share_malignant"] * 100, height=1.0,
            color=PALETTE["green"], alpha=0.9)
    rest = (1 - d["share_malignant"] - d["share_CAF"]) * 100
    ax.barh(y, rest, left=(d["share_malignant"] + d["share_CAF"]) * 100, height=1.0,
            color=PALETTE["lightgrey"], alpha=0.9)
    ax.axvline(50, color="black", lw=0.7, ls="--")
    order = list(d["gene"])
    wanted = ["COL17A1", "LAMB3", "LAMC2", "ITGA6", "MMP2", "COL1A1", "VIM", "TAGLN"]
    placed: list[int] = []
    for g in sorted((g for g in wanted if g in order), key=order.index, reverse=True):
        i = order.index(g)
        if all(abs(i - j) >= 5 for j in placed):
            ax.annotate(g, (102, i), fontsize=5.5, va="center", ha="left", annotation_clip=False)
            placed.append(i)
    ax.set_xlim(0, 100); ax.set_ylim(-1, len(d))
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share of expression (%)")
    ax.set_ylabel(f"Puram pEMT genes (n = {len(d)}), ranked")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.9) for c in
                       (PALETTE["vermilion"], PALETTE["green"], PALETTE["lightgrey"])],
              labels=["malignant", "CAF", "other"], fontsize=6, loc="upper center",
              ncol=3, columnspacing=0.8, handlelength=1.0, handletextpad=0.4,
              bbox_to_anchor=(0.5, -0.20))
    ax.set_title("a", loc="left", fontweight="bold")

    # (b) by gene family, one point per gene with the median as a rule. A strip plot is used
    # because the classical-mesenchymal family has one member (VIM).
    ax = fig.add_subplot(gs[0, 1])
    fams = ["hemidesmosome / laminin-332", "other pEMT gene", "classical mesenchymal"]
    short = {"hemidesmosome / laminin-332": "hemidesmosome",
             "other pEMT gene": "other", "classical mesenchymal": "mesenchymal"}
    keep = [f for f in fams if (df["family"] == f).sum()]
    rng = np.random.default_rng(20260920)
    for i, f in enumerate(keep, start=1):
        v = df.loc[df["family"] == f, "share_malignant"].values * 100
        ax.scatter(v, rng.normal(i, 0.07, len(v)), s=12, color=PALETTE["blue"],
                   alpha=0.8, linewidths=0, zorder=3)
        ax.plot([np.median(v)] * 2, [i - 0.26, i + 0.26], color="black", lw=1.4, zorder=4)
    ax.axvline(50, color="black", lw=0.7, ls="--")
    ax.set_yticks([])
    for i, f in enumerate(keep, start=1):
        ax.text(2, i + 0.33, f"{short[f]} (n = {int((df['family'] == f).sum())})",
                fontsize=6, va="bottom", ha="left", color=PALETTE["grey"])
    ax.set_ylim(0.45, len(keep) + 0.75)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Malignant share of expression (%)")
    ax.set_title("b", loc="left", fontweight="bold")

    # (c) within malignant cells, pEMT-high vs epithelial-like
    ax = fig.add_subplot(gs[0, 2])
    cols = {"malignant-expressed": PALETTE["vermilion"], "CAF-expressed": PALETTE["green"]}
    for arm, c in cols.items():
        s = df[df["arm"] == arm]
        ax.scatter(s["share_malignant"] * 100, s["log2FC_pEMT_high_vs_epithelial"],
                   s=13, color=c, alpha=0.8, linewidths=0, label=arm)
    ax.axhline(0, color=PALETTE["grey"], lw=0.6)
    ax.axvline(50, color="black", lw=0.7, ls="--")
    # Each label is offset from its marker and, where two would collide, pushed apart in y, with
    # a hairline back to its point.
    lab = [g for g in ["LAMC2", "COL17A1", "LAMB3", "ITGA6", "MMP10", "PLAU"]
           if (df["gene"] == g).any()]
    pts = sorted(((float(df.loc[df["gene"] == g, "share_malignant"].iloc[0]) * 100,
                   float(df.loc[df["gene"] == g, "log2FC_pEMT_high_vs_epithelial"].iloc[0]), g)
                  for g in lab), key=lambda r: -r[1])
    span = df["log2FC_pEMT_high_vs_epithelial"].max() - df["log2FC_pEMT_high_vs_epithelial"].min()
    last = None
    for gx, gy, g in pts:
        ly = gy if last is None else min(gy, last - 0.085 * span)
        ax.annotate(g, xy=(gx, gy), xytext=(gx + 3.5, ly), fontsize=5.5, va="center",
                    ha="left", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=PALETTE["grey"], lw=0.4,
                                    shrinkA=0, shrinkB=1.5))
        last = ly
    ax.set_xlim(0, 122)
    ax.set_xlabel("Malignant share of expression (%)")
    ax.set_ylabel("log$_2$ fold change,\npEMT-high vs epithelial-like")
    ax.legend(fontsize=6, loc="upper center", ncol=2, frameon=False, columnspacing=0.8,
              handletextpad=0.3, bbox_to_anchor=(0.5, -0.22))
    ax.set_title("c", loc="left", fontweight="bold")

    # (e) eigengene-trait correlations for M06 and M13
    bottom = gs[1, :].subgridspec(1, 2, width_ratios=[0.7, 2.3], wspace=0.66)
    ax = fig.add_subplot(bottom[0, 1])
    ann = pd.read_csv(ROOT / "results" / "wgcna" / "module_annotation.tsv", sep="\t").set_index("module")
    corr = pd.read_csv(ROOT / "results" / "wgcna" / "module_trait_correlations.tsv",
                       sep="\t", index_col=0)
    traits = [("P_pEMT_high", "P(pEMT-\nhigh)"), ("P_epithelial_like", "P(epithelial-\nlike)"),
              ("P_fibroblast_stromal_like", "P(fibroblast /\nstromal-like)"),
              ("pEMT_specificity", "pEMT\nspecificity"),
              ("puram_pemt", "Puram pEMT\nsignature"), ("hallmark_EMT", "Hallmark EMT")]
    mods = [("M06", "M06  fibroblast / ECM"), ("M13", "M13  basal keratinocyte")]
    M = np.array([[corr.loc[f"ME_{m}", k] for k, _ in traits] for m, _ in mods])
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(M[i, j]) > 0.55 else "black")
    ax.set_xticks(range(len(traits)))
    ax.set_xticklabels([lbl for _, lbl in traits], fontsize=6)
    ax.set_yticks(range(len(mods)))
    # Module size and Puram overlap go in the tick label, clear of the colour bar.
    ax.set_yticklabels([f"{lbl}\n{int(ann.loc[m, 'size']):,} genes, "
                        f"{int(ann.loc[m, 'puram_pemt_overlap'])} Puram pEMT"
                        for m, lbl in mods], fontsize=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.015)
    cb.set_label("Pearson r with eigengene", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    ax.set_title("e", loc="left", fontweight="bold")

    # (d) the partition on a composition-independent measure: malignant share of mean expression
    # per cell (linear scale), GSE103322 against GSE181919, from 02_partition_replication.py.
    rep_path = OUT / "replication_per_gene.tsv"
    if rep_path.exists():
        ax = fig.add_subplot(bottom[0, 0])
        rep = pd.read_csv(rep_path, sep="\t").pivot(index="gene", columns="dataset",
                                                      values="share_per_cell").dropna() * 100
        fam = rep.index.isin(HEMI)
        ax.scatter(rep.loc[~fam, "GSE103322"], rep.loc[~fam, "GSE181919"], s=8, color=PALETTE["grey"],
                   alpha=0.7, linewidths=0)
        ax.scatter(rep.loc[fam, "GSE103322"], rep.loc[fam, "GSE181919"], s=11, color=PALETTE["vermilion"],
                   linewidths=0, label="hemidesmosome")
        for g, off in (("COL1A1", (6, 7)), ("VIM", (7, -6))):
            if g in rep.index:
                ax.scatter(rep.loc[g, "GSE103322"], rep.loc[g, "GSE181919"], s=11, color=PALETTE["green"],
                           linewidths=0)
                ax.annotate(g, (rep.loc[g, "GSE103322"], rep.loc[g, "GSE181919"]), xytext=off,
                            textcoords="offset points", fontsize=5.5,
                            arrowprops=dict(arrowstyle="-", color=PALETTE["grey"], lw=0.4))
        ax.axhline(50, color="black", lw=0.6, ls="--"); ax.axvline(50, color="black", lw=0.6, ls="--")
        from scipy.stats import spearmanr as _sr
        ax.text(0.03, 0.97, f"ρ = {_sr(rep['GSE103322'], rep['GSE181919'])[0]:.2f}", transform=ax.transAxes,
                va="top", fontsize=6)
        ax.set_xlim(0, 100); ax.set_ylim(0, 100)
        ax.set_xticks([0, 50, 100]); ax.set_yticks([0, 50, 100])
        ax.set_xlabel("GSE103322, malignant %\nof per-cell mean", fontsize=6)
        ax.set_ylabel("GSE181919 tumour (20 patients)", fontsize=6)
        ax.tick_params(labelsize=5.5)
        ax.set_title("d", loc="left", fontweight="bold")

    fig.subplots_adjust(left=0.135, right=0.965, top=0.94, bottom=0.085)
    save_figure(fig, FIG, "Figure_2")
    print(f"\nwrote {OUT}/ and {FIG}/Figure_2.svg/.png")


if __name__ == "__main__":
    main()
