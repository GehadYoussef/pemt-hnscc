"""Draw Figure_3, the single-cell validation of the classifier.

Panels, from the outputs of 02_apply_classifier_to_sc_datasets.py:
  a  class probabilities of every pseudobulk (at least 20 cells), by dataset and cell type
  b  P(pEMT-high) against the Puram pEMT signature, Puram malignant pseudobulks, with Spearman rho
  c  P(pEMT-high) of the Puram primary-site pseudobulks by nodal status (Puram et al. Table S1),
     with the one-sided Mann-Whitney p
  d  matched lymph node versus primary, Puram patients

Inputs:  results/sc_classifier_validation/pseudobulk_probabilities.tsv,
         puram_primary_patient_scores.tsv, puram_nodal_status_test.tsv,
         puram_ln_vs_primary_paired.tsv
Outputs: results/figures/Figure_3.svg and .png
Usage:   python src/03_sc_classifier_validation/03_validation_figure.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pemt import load_config, project_root  # noqa: E402
from pemt.figstyle import apply_style, save_figure, PALETTE, CLASS_COLOURS, mm  # noqa: E402

cfg = load_config()
ROOT = project_root()
OUT = ROOT / cfg["paths"]["results_dir"] / "sc_classifier_validation"
FIG = ROOT / "results" / "figures"
apply_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

DS_LABEL = {"HNSC_GSE103322": "HNSCC GSE103322 (Puram)", "OSCC_GSE172577": "OSCC\nGSE172577", "LSCC_GSE150321": "Laryngeal SCC\nGSE150321",
            "NPC_GSE150430": "NPC\nGSE150430", "NPC_GSE162025": "NPC\nGSE162025", "THCA_GSE148673_outgroup": "Thyroid outgroup\nGSE148673"}
CT_SHORT = {"malignant": "malig.", "fibroblast": "fibro."}
CLASSES = [("P_pEMT_high", "pEMT-high"), ("P_epithelial_like", "epithelial-like"), ("P_fibroblast_stromal_like", "fibroblast / stromal")]


def main() -> None:
    pb = pd.read_csv(OUT / "pseudobulk_probabilities.tsv", sep="\t", index_col=0)
    fig = plt.figure(figsize=(mm(180), mm(120)))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], width_ratios=[1.1, 1, 1], hspace=0.6, wspace=0.45)

    # a: stacked probabilities per pseudobulk, grouped by dataset and cell type
    ax = fig.add_subplot(gs[0, :])
    order = [d for d in DS_LABEL if d in set(pb["dataset_id"])]
    x = 0
    ticks, labels, seps, groups_x = [], [], [], []
    for ds in order:
        x_start = x
        for ct in ("malignant", "fibroblast"):
            sub = pb[(pb["dataset_id"] == ds) & (pb["standard_cell_type"] == ct)].sort_values("P_pEMT_high", ascending=False)
            if sub.empty:
                continue
            bottom = np.zeros(len(sub))
            xs = np.arange(x, x + len(sub))
            for col, lab in CLASSES:
                ax.bar(xs, sub[col].values, bottom=bottom, width=0.85, color=CLASS_COLOURS[col.replace("P_", "")], linewidth=0)
                bottom += sub[col].values
            ticks.append(xs.mean()); labels.append(f"{CT_SHORT[ct]}\nn = {len(sub)}")
            x += len(sub) + 2.5
        groups_x.append(((x_start + x - 2.5) / 2 - 0.5, DS_LABEL[ds]))
        x += 2.0
        seps.append(x - 2.25)
    for sx in seps[:-1]:
        ax.axvline(sx, color=PALETTE["lightgrey"], linewidth=0.5)
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=5.5)
    for gx, lab in groups_x:
        ax.text(gx, -0.20, lab, ha="center", va="top", fontsize=6, transform=ax.get_xaxis_transform())
    ax.set_xlim(-1, x - 1)
    ax.set_ylim(0, 1); ax.set_ylabel("Class probability")
    ax.legend(handles=[Patch(color=CLASS_COLOURS[c.replace("P_", "")], label=l) for c, l in CLASSES], loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, fontsize=6.5, frameon=False)
    ax.set_title("a", loc="left", fontweight="bold")

    # b: rank agreement in Puram malignant pseudobulks
    ax = fig.add_subplot(gs[1, 0])
    p = pb[(pb["dataset_id"] == "HNSC_GSE103322") & (pb["standard_cell_type"] == "malignant")]
    ax.scatter(p["puram_pemt_signature"], p["P_pEMT_high"], s=14, color=[PALETTE["vermilion"] if s == "Lymph node" else PALETTE["blue"] for s in p["site"]], linewidths=0)
    rho, pv = spearmanr(p["puram_pemt_signature"], p["P_pEMT_high"])
    ax.text(0.03, 0.95, f"Spearman ρ = {rho:.2f}\np = {pv:.1e}\nn = {len(p)} pseudobulks", transform=ax.transAxes, va="top", fontsize=6.5)
    ax.set_xlabel("Puram pEMT signature (mean z)")
    ax.set_ylabel("P(pEMT-high)")
    ax.set_ylim(-0.03, 1.03)
    ax.scatter([], [], s=14, color=PALETTE["blue"], label="primary"); ax.scatter([], [], s=14, color=PALETTE["vermilion"], label="lymph node")
    ax.legend(loc="lower right", fontsize=6)
    ax.set_title("b", loc="left", fontweight="bold")

    # c: nodal status
    ax = fig.add_subplot(gs[1, 1])
    prim = pd.read_csv(OUT / "puram_primary_patient_scores.tsv", sep="\t", index_col=0)
    prim = prim.dropna(subset=["N_positive"])
    groups = [prim[prim["N_positive"] == 0]["P_pEMT_high"].values, prim[prim["N_positive"] == 1]["P_pEMT_high"].values]
    rng = np.random.default_rng(0)
    for i, d in enumerate(groups, start=1):
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(d)), d, s=16, color=PALETTE["orange"], linewidths=0, zorder=2)
    ax.boxplot(groups, widths=0.45, showfliers=False, medianprops={"color": PALETTE["black"], "linewidth": 1.1}, zorder=1)
    ax.set_xticks([1, 2]); ax.set_xticklabels([f"N0\n(n = {len(groups[0])})", f"N+\n(n = {len(groups[1])})"])
    nod = pd.read_csv(OUT / "puram_nodal_status_test.tsv", sep="\t")
    pv = float(nod.loc[nod["metric"] == "P_pEMT_high", "p_one_sided"].iloc[0])
    ax.text(0.5, 1.01, f"one-sided Mann-Whitney (N+ > N0) p = {pv:.2f}", transform=ax.transAxes, ha="center", va="bottom", fontsize=6, color=PALETTE["grey"])
    ax.set_ylabel("P(pEMT-high), primary site")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("c", loc="left", fontweight="bold")

    # d: paired LN vs primary
    ax = fig.add_subplot(gs[1, 2])
    pair = pd.read_csv(OUT / "puram_ln_vs_primary_paired.tsv", sep="\t", index_col=0)
    texts = []
    for pt, r in pair.iterrows():
        ax.plot([1, 2], [r["Primary"], r["Lymph node"]], color=PALETTE["grey"], linewidth=0.8, zorder=1)
        ax.scatter([1, 2], [r["Primary"], r["Lymph node"]], s=16, color=[PALETTE["blue"], PALETTE["vermilion"]], linewidths=0, zorder=2)
        texts.append(ax.text(2.1, r["Lymph node"], pt, fontsize=5.5, va="center", color=PALETTE["grey"]))
    from adjustText import adjust_text
    adjust_text(texts, ax=ax, only_move={"text": "y", "static": "y", "explode": "y", "pull": "y"}, expand=(1.0, 1.6))
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Primary", "Lymph node"])
    ax.set_xlim(0.6, 2.7)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("P(pEMT-high)")
    ax.text(0.5, 1.01, f"{len(pair)} matched patients", transform=ax.transAxes, ha="center", va="bottom", fontsize=6, color=PALETTE["grey"])
    ax.set_title("d", loc="left", fontweight="bold")
    save_figure(fig, FIG, "Figure_3")


if __name__ == "__main__":
    main()
