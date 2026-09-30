"""Benchmark the pEMT scores against the decoupling approach of Tyler and Tirosh (Nat Commun 2021, 12:2592).

Tyler and Tirosh split EMT signature genes into cancer-cell and fibroblast contributions using
single-cell data, with Puram's GSE103322 as the reference for HNSCC. Their gene set is seeded on
classical mesenchymal regulators (`initial_genes` in their `deconv.R`: SNAI1, SNAI2, TWIST1, VIM,
ZEB1 and ZEB2). The Puram pEMT programme contains only VIM.

Their decoupling principle is applied here to the same data. Their full pipeline (CCLE reference
profiles, subtype-stratified deconvolution and per-subtype gene weighting) is not reimplemented. The
EMT signature gene set is the union of Hallmark EMT, the Tan et al. tumour signature, the six seed
regulators and the 250 genes with the highest maximum Spearman correlation with any seed in
TCGA-HNSC, matching the top-250 cap of their HNSC gene filter. Each gene is assigned to the cancer
or fibroblast compartment in GSE103322 by whether its malignant share of linear-scale expression
exceeds its CAF share. Bulk tumours are scored on each compartment as the mean of per-gene
z-scores, and both scores are correlated (Spearman) with the pEMT and EMT score panel.

Supplementary_Figure_1 shows (a) the Spearman correlation of each panel score with the two
compartment scores and (b) the cancer-compartment score against pEMT specificity.

Inputs:  results/tcga_projection/emt_score_panel_scores.tsv
         data/processed/tcga_hnsc/tcga_star_tpm_log2_for_projection.tsv
         data/processed/single_cell/HNSC_GSE103322/HNSC_GSE103322.h5ad
         data/references/emt_scores/hallmark_emt_genes.txt
         data/references/emt_scores/ks_tumor_signature.tsv
         config/signatures/puram_pemt.txt
Outputs: results/tyler_tirosh/esg_compartment_assignment.tsv
         results/tyler_tirosh/tcga_scores_with_tyler_tirosh.tsv
         results/tyler_tirosh/score_correlations.tsv
         results/tyler_tirosh/shared_genes_puram_vs_tt.tsv
         results/figures/Supplementary_Figure_1
Usage:   python src/09_gene_partition/03_tyler_tirosh_benchmark.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SC = ROOT / "data" / "processed" / "single_cell" / "HNSC_GSE103322"
REF = ROOT / "data" / "references" / "emt_scores"
OUT = ROOT / "results" / "tyler_tirosh"
FIG = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))
from pemt.figstyle import apply_style, save_figure, PALETTE, mm  # noqa: E402

apply_style()
import matplotlib.pyplot as plt  # noqa: E402

# Their seed regulators, verbatim from `initial_genes` in deconv.R
SEED_GENES = ["SNAI1", "SNAI2", "TWIST1", "VIM", "ZEB1", "ZEB2"]
# Their `genes_filter_fun` for HNSC keeps the top 250 ranked genes, so the seed-correlated set is
# capped the same way. A correlation threshold pulls in most of the stroma-associated
# transcriptome, and the two compartment scores are then no longer separable.
SEED_TOP_N = 250
PANEL = ["pEMT_specificity", "P_pEMT_high", "puram_pemt", "hallmark_EMT", "KS", "GS76", "MLR_mu"]
LABELS = {"pEMT_specificity": "pEMT specificity (malignant arm)", "P_pEMT_high": "P(pEMT-high)",
          "puram_pemt": "Puram pEMT (whole list)", "hallmark_EMT": "Hallmark EMT", "KS": "KS",
          "GS76": "76GS", "MLR_mu": "MLR",
          "TT_cancer_EMT": "Tyler-Tirosh cancer compartment",
          "TT_caf_EMT": "Tyler-Tirosh fibroblast compartment"}


def zmean(expr: pd.DataFrame, genes: list[str]) -> pd.Series:
    g = [x for x in genes if x in expr.index]
    if not g:
        return pd.Series(np.nan, index=expr.columns)
    sub = expr.loc[g]
    z = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1).replace(0, np.nan), axis=0)
    return z.mean(axis=0)


def main() -> None:
    import anndata as ad
    import scipy.sparse as sp

    scores = pd.read_csv(ROOT / "results" / "tcga_projection" / "emt_score_panel_scores.tsv",
                         sep="\t", index_col=0)
    expr = pd.read_csv(ROOT / "data" / "processed" / "tcga_hnsc" /
                       "tcga_star_tpm_log2_for_projection.tsv", sep="\t", index_col=0)
    expr = expr[[c for c in expr.columns if c in scores.index]]
    print(f"TCGA: {expr.shape[0]} genes x {expr.shape[1]} tumours with scores")

    # ---------------- 1. build the EMT signature gene set their way ----------------
    hallmark = [l.strip() for l in (REF / "hallmark_emt_genes.txt").read_text().splitlines() if l.strip()]
    ks = pd.read_csv(REF / "ks_tumor_signature.tsv", sep="\t")
    ks_genes = sorted(set(ks.iloc[:, 0].astype(str)))
    present_seeds = [g for g in SEED_GENES if g in expr.index]
    # vectorised Spearman of every gene against each seed, via rank transform
    ranks = expr.rank(axis=1)
    rc = ranks.sub(ranks.mean(axis=1), axis=0)
    rc = rc.div(np.sqrt((rc ** 2).sum(axis=1)).replace(0, np.nan), axis=0)
    sc_ = rc.loc[present_seeds]
    cormat = rc.values @ sc_.values.T          # genes x seeds
    max_corr = pd.Series(np.nanmax(cormat, axis=1), index=expr.index)
    seed_correlated = sorted(max_corr.drop(index=present_seeds, errors='ignore')
                             .nlargest(SEED_TOP_N).index) + present_seeds

    esg = sorted(set(hallmark) | set(ks_genes) | set(seed_correlated))
    esg = [g for g in esg if g in expr.index]
    print(f"EMT signature genes: {len(hallmark)} Hallmark + {len(ks_genes)} Tan + "
          f"{len(seed_correlated)} seed-correlated (top {SEED_TOP_N}), {len(esg)} unique in TCGA")

    # ---------------- 2. resolve each ESG to a compartment in the single-cell data ----------------
    a = ad.read_h5ad(SC / "HNSC_GSE103322.h5ad")
    major = a.obs["Celltype (major-lineage)"].astype(str)
    grp = pd.Series("other", index=a.obs_names, dtype=object)
    grp[major.isin(["Fibroblasts", "Myofibroblasts"])] = "CAF"
    grp[major.eq("Malignant")] = "malignant"
    a.obs["group"] = grp
    mal = (a.obs["group"] == "malignant").values
    caf = (a.obs["group"] == "CAF").values

    rows = []
    for g in esg:
        if g not in a.var_names:
            continue
        col = a.X[:, a.var_names.get_loc(g)]
        v = np.asarray(col.todense()).ravel() if sp.issparse(col) else np.asarray(col).ravel()
        v = np.expm1(v)  # the matrix is log1p-normalised. Shares are taken on the linear scale.
        tot = v.sum()
        if tot <= 0:
            continue
        rows.append({"gene": g, "share_malignant": float(v[mal].sum() / tot),
                     "share_CAF": float(v[caf].sum() / tot)})
    part = pd.DataFrame(rows)
    part["compartment"] = np.where(part["share_malignant"] > part["share_CAF"], "cancer", "CAF")
    part.to_csv(OUT / "esg_compartment_assignment.tsv", sep="\t", index=False)
    cancer_esg = part.loc[part["compartment"] == "cancer", "gene"].tolist()
    caf_esg = part.loc[part["compartment"] == "CAF", "gene"].tolist()
    print(f"resolved in single cells: {len(cancer_esg)} cancer-compartment, {len(caf_esg)} fibroblast-compartment")

    # ---------------- 3. score TCGA on each compartment ----------------
    df = scores.loc[expr.columns, PANEL].copy()
    df["TT_cancer_EMT"] = zmean(expr, cancer_esg)
    df["TT_caf_EMT"] = zmean(expr, caf_esg)
    df.to_csv(OUT / "tcga_scores_with_tyler_tirosh.tsv", sep="\t")

    cols = PANEL + ["TT_cancer_EMT", "TT_caf_EMT"]
    rho = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for i in cols:
        for j in cols:
            rho.loc[i, j] = spearmanr(df[i], df[j], nan_policy="omit").statistic
    rho.round(3).to_csv(OUT / "score_correlations.tsv", sep="\t")

    print("\nSpearman with the Tyler-Tirosh cancer-compartment EMT score:")
    for c in PANEL:
        print(f"  {LABELS[c]:<34} {rho.loc['TT_cancer_EMT', c]:+.2f}")
    print(f"  {'(its own fibroblast compartment)':<34} {rho.loc['TT_cancer_EMT','TT_caf_EMT']:+.2f}")
    print("\nSpearman with the Tyler-Tirosh fibroblast-compartment score:")
    for c in PANEL:
        print(f"  {LABELS[c]:<34} {rho.loc['TT_caf_EMT', c]:+.2f}")

    # overlap of gene sets
    puram = [l.strip() for l in (ROOT / "config" / "signatures" / "puram_pemt.txt").read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    ov = sorted(set(puram) & set(esg))
    print(f"\nPuram pEMT genes inside their EMT signature gene set: {len(ov)} of {len(puram)}")
    print(f"  seed regulators present in the Puram pEMT list: "
          f"{[g for g in SEED_GENES if g in puram] or 'none'}")
    pd.DataFrame({"gene": ov}).to_csv(OUT / "shared_genes_puram_vs_tt.tsv", sep="\t", index=False)

    # ---------------- figure ----------------
    fig, axes = plt.subplots(1, 2, figsize=(mm(180), mm(72)),
                             gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.5})

    ax = axes[0]
    order = ["hallmark_EMT", "KS", "puram_pemt", "MLR_mu", "GS76", "P_pEMT_high", "pEMT_specificity"]
    y = np.arange(len(order))
    for off, tt, col, lab in [(-0.2, "TT_cancer_EMT", PALETTE["vermilion"], "cancer compartment"),
                              (0.2, "TT_caf_EMT", PALETTE["green"], "fibroblast compartment")]:
        ax.barh(y + off, [rho.loc[tt, c] for c in order], height=0.38, color=col, alpha=0.9, label=lab)
    ax.axvline(0, color=PALETTE["grey"], lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[c].replace(" (malignant arm)", "\n(malignant arm)")
                        .replace(" (whole list)", "\n(whole list)") for c in order], fontsize=6.5)
    ax.set_xlabel("Spearman with Tyler-Tirosh score")
    ax.set_xlim(-1, 1)
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
              columnspacing=1.0, handlelength=1.0, handletextpad=0.4)
    ax.set_title("a", loc="left", fontweight="bold")

    ax = axes[1]
    ax.scatter(df["TT_cancer_EMT"], df["pEMT_specificity"], s=6, color=PALETTE["blue"],
               alpha=0.6, linewidths=0, rasterized=True)
    r = rho.loc["TT_cancer_EMT", "pEMT_specificity"]
    ax.set_xlabel("Tyler-Tirosh cancer compartment")
    ax.set_ylabel("pEMT specificity (malignant arm)")
    ax.text(0.03, 0.96, f"Spearman {r:+.2f}\nn = {len(df)} tumours", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.5, linespacing=1.4)
    ax.set_title("b", loc="left", fontweight="bold")

    fig.subplots_adjust(left=0.20, right=0.985, top=0.91, bottom=0.26)
    save_figure(fig, FIG, "Supplementary_Figure_1")
    print(f"\nwrote {OUT}/ and {FIG}/Supplementary_Figure_1.svg/.png")


if __name__ == "__main__":
    main()
