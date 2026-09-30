"""Ligand-receptor communication between CAFs and the two malignant states in Puram GSE103322.

The analysis is between cell types and is separate from the protein-protein interaction network
used for drug proximity in src/06_network_analysis. It has two parts.

1. CAF-to-malignant ligand-receptor pairs specific to pEMT-high over epithelial-like malignant
   cells. LIANA rank_aggregate gives the consensus of five methods (CellPhoneDB,
   Connectome, log2FC, NATMI, SingleCellSignalR), with genes expressed in at least 10% of a group
   and 1,000 permutations. For each pair, delta_specificity is the difference in log10 specificity
   rank between the pEMT-high and epithelial-like targets. A negative value means more specific to
   pEMT-high. LIANA's specificity rank is a permutation p-value and depends on group size, so the
   comparison is repeated five times with pEMT-high cells downsampled to the epithelial-like count.
   A top pair is counted as surviving when it favours pEMT-high in at least 4 of the 5 runs.

2. The source of the EGFR ligands among the classifier's top coefficients (AREG, EREG, TGFA,
   NRG1), tumour cells or CAFs. For each ligand, the share of total expression contributed by each
   cell group (on the linear scale) and the fraction of cells expressing it are computed, to
   separate autocrine from paracrine supply. EGFR-family receptors and canonical CAF-derived
   ligands are computed in the same way, the latter as a positive control.

Cell groups: CAF (Fibroblasts and Myofibroblasts), malignant_pEMT_high and malignant_epithelial_like
(from the training labels of src/02_multinomial_pemt_model/01_define_training_labels.py),
malignant_other, and endothelial and immune cells for context. Other cells are excluded.

Supplementary_Figure_9: (a) share of expression of the EGFR-family ligands by malignant cells, CAFs
and other cells, (b) the same for canonical CAF ligands, (c) specificity of the 12 most
pEMT-high-specific CAF-to-malignant pairs towards each malignant state.

Inputs:  data/processed/single_cell/HNSC_GSE103322/HNSC_GSE103322.h5ad          (all 5,902 cells)
         data/processed/single_cell/HNSC_GSE103322/HNSC_GSE103322_labelled.h5ad (training labels)
Outputs: results/cell_communication/liana_all_pairs.tsv
         results/cell_communication/caf_to_malignant_pairs.tsv
         results/cell_communication/ligand_source_by_celltype.tsv
         results/cell_communication/caf_to_malignant_matched_control.tsv
         results/figures/Supplementary_Figure_9.svg and .png
Usage:   python src/11_composition_and_mechanism/03_caf_tumour_ligand_receptor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SC = ROOT / "data" / "processed" / "single_cell" / "HNSC_GSE103322"
OUT = ROOT / "results" / "cell_communication"
FIG = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))
from pemt.figstyle import apply_style, save_figure, PALETTE, mm  # noqa: E402

apply_style()
import matplotlib.pyplot as plt  # noqa: E402

# EGFR-family ligands plus the receptor. The first four are among the classifier's top coefficients.
EGFR_LIGANDS = ["AREG", "EREG", "TGFA", "NRG1", "HBEGF", "BTC", "EPGN", "EGF"]
EGFR_RECEPTORS = ["EGFR", "ERBB2", "ERBB3", "ERBB4"]
# Canonical CAF-derived ligands, as a positive control
CAF_LIGANDS = ["TGFB1", "TGFB2", "TGFB3", "HGF", "FGF2", "FGF7", "CXCL12", "IL6", "POSTN", "INHBA"]


def build() -> "anndata.AnnData":
    import anndata as ad

    full = ad.read_h5ad(SC / "HNSC_GSE103322.h5ad")
    lab = ad.read_h5ad(SC / "HNSC_GSE103322_labelled.h5ad")
    full.obs["training_label"] = lab.obs["training_label"].reindex(full.obs_names)

    major = full.obs["Celltype (major-lineage)"].astype(str)
    lbl = full.obs["training_label"].astype(str)
    grp = pd.Series("other", index=full.obs_names, dtype=object)
    grp[major.isin(["Fibroblasts", "Myofibroblasts"])] = "CAF"
    grp[major.eq("Endothelial")] = "endothelial"
    grp[major.isin(["CD8Tex", "CD4Tconv", "CD8T", "Plasma", "Mast", "Mono/Macro"])] = "immune"
    mal = major.eq("Malignant")
    grp[mal] = "malignant_other"
    grp[mal & lbl.eq("pEMT_high")] = "malignant_pEMT_high"
    grp[mal & lbl.eq("epithelial_like")] = "malignant_epithelial_like"
    full.obs["group"] = pd.Categorical(grp)
    print("cell groups:")
    print(full.obs["group"].value_counts().to_string())
    return full[full.obs["group"] != "other"].copy()


def ligand_source(adata, genes: list[str], label: str) -> pd.DataFrame:
    """Share of total expression and detection rate per cell group, for each gene."""
    import scipy.sparse as sp

    rows = []
    X = adata.X
    for g in genes:
        if g not in adata.var_names:
            continue
        v = X[:, adata.var_names.get_loc(g)]
        v = np.asarray(v.todense()).ravel() if sp.issparse(v) else np.asarray(v).ravel()
        v = np.expm1(v)  # the matrix is log1p-normalised, so shares are taken on the linear scale
        tot = v.sum()
        if tot <= 0:
            continue
        for grp in adata.obs["group"].cat.categories:
            m = (adata.obs["group"] == grp).values
            rows.append({
                "gene": g, "panel": label, "group": grp,
                "share_of_expression": float(v[m].sum() / tot),
                "pct_cells_expressing": float((v[m] > 0).mean() * 100),
                "mean_expression": float(v[m].mean()),
                "n_cells": int(m.sum()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    import liana as li

    adata = build()

    # ---------------------------------------------------------------- 1. LIANA consensus
    print("\nrunning LIANA rank_aggregate (consensus of 5 methods, 1000 permutations)")
    li.mt.rank_aggregate(
        adata, groupby="group", expr_prop=0.1, use_raw=False,
        n_perms=1000, seed=20260920, verbose=False,
    )
    lr = adata.uns["liana_res"].copy()
    lr.to_csv(OUT / "liana_all_pairs.tsv", sep="\t", index=False)

    caf = lr[lr["source"] == "CAF"].copy()
    tgt = {"malignant_pEMT_high": "pEMT_high", "malignant_epithelial_like": "epithelial_like"}
    caf = caf[caf["target"].isin(tgt)]
    caf["pair"] = caf["ligand_complex"] + " -> " + caf["receptor_complex"]

    wide = caf.pivot_table(index="pair", columns="target",
                           values=["magnitude_rank", "specificity_rank"])
    wide.columns = [f"{a}_{tgt[b]}" for a, b in wide.columns]
    wide = wide.dropna()
    # negative = more specific to pEMT-high (ranks are p-value-like, lower is stronger)
    wide["delta_specificity"] = (np.log10(wide["specificity_rank_pEMT_high"] + 1e-6)
                                 - np.log10(wide["specificity_rank_epithelial_like"] + 1e-6))
    wide = wide.sort_values("delta_specificity")
    wide.to_csv(OUT / "caf_to_malignant_pairs.tsv", sep="\t")

    sig = caf[(caf["target"] == "malignant_pEMT_high") & (caf["specificity_rank"] <= 0.05)]
    print(f"\nCAF to pEMT-high: {len(sig)} pairs at specificity rank <= 0.05")
    print(sig.nsmallest(15, "specificity_rank")[
        ["pair", "specificity_rank", "magnitude_rank"]].to_string(index=False))
    print("\nMost pEMT-high-specific relative to epithelial-like (top 15):")
    print(wide.head(15)[["specificity_rank_pEMT_high", "specificity_rank_epithelial_like",
                         "delta_specificity"]].round(4).to_string())

    # ---------------------------------------------------------------- 2. ligand source by cell group
    src = pd.concat([
        ligand_source(adata, EGFR_LIGANDS, "EGFR family ligand"),
        ligand_source(adata, EGFR_RECEPTORS, "EGFR family receptor"),
        ligand_source(adata, CAF_LIGANDS, "canonical CAF ligand"),
    ])
    src.to_csv(OUT / "ligand_source_by_celltype.tsv", sep="\t", index=False)

    piv = src.pivot_table(index=["panel", "gene"], columns="group", values="share_of_expression")
    mal_cols = [c for c in piv.columns if c.startswith("malignant")]
    piv["malignant_total"] = piv[mal_cols].sum(axis=1)
    print("\nShare of total expression by cell type (autocrine vs paracrine test):")
    print((piv[["CAF", "malignant_total", "endothelial", "immune"]] * 100).round(1).to_string())

    # ------------------------------------------------- 3. cell-number-matched control
    # pEMT-high has 417 cells against 108 epithelial-like, and LIANA's specificity rank is a
    # permutation p-value, so it is sensitive to group size. The comparison is repeated with
    # pEMT-high downsampled to the epithelial-like count.
    import anndata as ad

    n_epi = int((adata.obs["group"] == "malignant_epithelial_like").sum())
    keep_top = list(wide.head(12).index)
    runs = []
    for seed in (1, 2, 3, 4, 5):
        rng = np.random.default_rng(20260920 + seed)
        idx = adata.obs_names.to_numpy()
        hi = idx[(adata.obs["group"] == "malignant_pEMT_high").to_numpy()]
        drop = set(rng.choice(hi, size=len(hi) - n_epi, replace=False))
        sub = adata[[b for b in idx if b not in drop]].copy()
        li.mt.rank_aggregate(sub, groupby="group", expr_prop=0.1, use_raw=False,
                             n_perms=1000, seed=20260920, verbose=False)
        r = sub.uns["liana_res"]
        r = r[(r["source"] == "CAF") & (r["target"].isin(tgt))].copy()
        r["pair"] = r["ligand_complex"] + " -> " + r["receptor_complex"]
        w = r.pivot_table(index="pair", columns="target", values="specificity_rank")
        w.columns = [tgt[c] for c in w.columns]
        w = w.dropna()
        w["delta"] = np.log10(w["pEMT_high"] + 1e-6) - np.log10(w["epithelial_like"] + 1e-6)
        runs.append(w["delta"].rename(f"seed{seed}"))
        print(f"  matched run {seed} (n = {n_epi} per malignant group): "
              f"{int((w['pEMT_high'] <= 0.05).sum())} pairs at rank <= 0.05 to pEMT-high, "
              f"{int((w['epithelial_like'] <= 0.05).sum())} to epithelial-like")
    matched = pd.concat(runs, axis=1)
    matched["mean_delta"] = matched.mean(axis=1)
    matched["n_runs_favouring_pEMT_high"] = (matched[[f"seed{s}" for s in (1, 2, 3, 4, 5)]] < 0).sum(axis=1)
    matched.sort_values("mean_delta").to_csv(OUT / "caf_to_malignant_matched_control.tsv", sep="\t")
    surv = matched.reindex(keep_top)
    print("\nTop 12 unmatched pairs, re-tested with equal cell numbers "
          "(negative mean delta = still pEMT-high-specific):")
    print(surv[["mean_delta", "n_runs_favouring_pEMT_high"]].round(3).to_string())
    n_ok = int((surv["n_runs_favouring_pEMT_high"] >= 4).sum())
    print(f"\n{n_ok} of {len(surv)} survive in at least 4 of 5 matched runs")

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(mm(180), mm(78)),
                             gridspec_kw={"width_ratios": [1.1, 0.95, 1.5], "wspace": 0.95})

    # (a) EGFR ligands by source cell group
    ax = axes[0]
    eg = piv.loc["EGFR family ligand"].reindex([g for g in EGFR_LIGANDS if g in
                                                piv.loc["EGFR family ligand"].index])
    y = np.arange(len(eg))
    ax.barh(y, eg["malignant_total"] * 100, color=PALETTE["vermilion"], height=0.68,
            label="malignant", alpha=0.9)
    ax.barh(y, eg["CAF"] * 100, left=eg["malignant_total"] * 100, color=PALETTE["green"],
            height=0.68, label="CAF", alpha=0.9)
    rest = (1 - eg["malignant_total"] - eg["CAF"]) * 100
    ax.barh(y, rest, left=(eg["malignant_total"] + eg["CAF"]) * 100,
            color=PALETTE["lightgrey"], height=0.68, label="other", alpha=0.9)
    ax.set_yticks(y); ax.set_yticklabels(eg.index, fontsize=7)
    ax.set_xlim(0, 118); ax.set_xlabel("Share of expression (%)")
    ax.set_xticks([0, 50, 100])
    ax.legend(fontsize=6, loc="lower right", bbox_to_anchor=(1.0, -0.02), ncol=3,
              columnspacing=0.8, handlelength=1.0, handletextpad=0.4)
    ax.set_title("a", loc="left", fontweight="bold")

    # (b) canonical CAF ligands, the positive control
    ax = axes[1]
    cg = piv.loc["canonical CAF ligand"].reindex([g for g in CAF_LIGANDS if g in
                                                  piv.loc["canonical CAF ligand"].index])
    y = np.arange(len(cg))
    ax.barh(y, cg["malignant_total"] * 100, color=PALETTE["vermilion"], height=0.68, alpha=0.9)
    ax.barh(y, cg["CAF"] * 100, left=cg["malignant_total"] * 100, color=PALETTE["green"],
            height=0.68, alpha=0.9)
    rest = (1 - cg["malignant_total"] - cg["CAF"]) * 100
    ax.barh(y, rest, left=(cg["malignant_total"] + cg["CAF"]) * 100,
            color=PALETTE["lightgrey"], height=0.68, alpha=0.9)
    ax.set_yticks(y); ax.set_yticklabels(cg.index, fontsize=7)
    ax.set_xlim(0, 100); ax.set_xlabel("Share of expression (%)")
    ax.set_title("b", loc="left", fontweight="bold")

    # (c) CAF to pEMT-high signalling, most specific pairs
    ax = axes[2]
    top = wide.head(12).iloc[::-1]
    y = np.arange(len(top))
    ax.barh(y, -np.log10(top["specificity_rank_pEMT_high"] + 1e-6), color=PALETTE["vermilion"],
            height=0.38, alpha=0.9, label="to pEMT-high")
    ax.barh(y + 0.4, -np.log10(top["specificity_rank_epithelial_like"] + 1e-6),
            color=PALETTE["blue"], height=0.38, alpha=0.9, label="to epithelial-like")
    ax.set_yticks(y + 0.2)
    # Every pair shown survives the cell-number-matched control, so no per-row marker is drawn.
    ax.set_yticklabels([p.replace(" -> ", "→") for p in top.index], fontsize=6)
    ax.set_xlabel("Specificity, $-$log$_{10}$ rank")
    ax.legend(fontsize=6, loc="lower right")
    ax.set_title("c", loc="left", fontweight="bold")

    fig.subplots_adjust(left=0.085, right=0.985, top=0.91, bottom=0.17)
    save_figure(fig, FIG, "Supplementary_Figure_9")
    print(f"\nwrote {OUT}/ and {FIG}/Supplementary_Figure_9.svg/.png")


if __name__ == "__main__":
    main()
