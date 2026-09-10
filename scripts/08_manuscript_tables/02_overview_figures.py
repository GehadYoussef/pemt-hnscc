"""Figure 1 (pipeline schematic) and Figure 2 (datasets, training set, classifier) for the manuscript.

Added 9 Sep 2026. Reads only saved results:
  config/dataset_registry.tsv, results/filtered_malignant_fibroblast_summary.tsv,
  data/processed/pseudo_bulk/pseudobulk_metadata.tsv, results/multinomial_classifier/*.tsv
Writes Figure_1_pipeline_schematic and Figure_2_classifier (A datasets, B training set, C held-out
performance incl. the benchmark models, D top coefficients per class).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import load_config, project_root  # noqa: E402
from _lib.figstyle import apply_style, save_figure, PALETTE, CLASS_COLOURS, mm  # noqa: E402

cfg = load_config()
ROOT = project_root()
RES = ROOT / cfg["paths"]["results_dir"]
FIG = ROOT / "manuscript" / "figures"
apply_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch  # noqa: E402

CLS = [("pEMT_high", "pEMT-high"), ("epithelial_like", "epithelial-like"), ("fibroblast_stromal_like", "fibroblast / stromal")]


def box(ax, x, y, w, h, title, body, colour):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012", linewidth=0.8, edgecolor=colour, facecolor="white"))
    ax.add_patch(FancyBboxPatch((x, y + h - 0.075), w, 0.075, boxstyle="round,pad=0.008,rounding_size=0.012", linewidth=0, facecolor=colour))
    ax.text(x + w / 2, y + h - 0.037, title, ha="center", va="center", fontsize=6.2, color="white", fontweight="bold")
    ax.text(x + w / 2, y + (h - 0.075) / 2, body, ha="center", va="center", fontsize=5.8, linespacing=1.0)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, linewidth=0.7, color=PALETTE["grey"]))


def figure_1():
    """Pipeline schematic. Methods only, no results.

    One colour per stage row. Every gap between boxes is the same width and every arrow, horizontal
    or vertical, is drawn at the same length; the inter-row connectors are elbows that start at one
    named box and end on one named box. Body text is capped at what fits the box so nothing spills
    over the header band or the lower border.
    """
    PAD, CLR = 0.008, 0.006
    fig, ax = plt.subplots(figsize=(mm(180), mm(108)))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()
    GAP, H = 0.065, 0.245
    ALEN = GAP - 2 * (PAD + CLR)          # every arrow in the figure is this long
    R1, R2, R3 = 0.715, 0.395, 0.075
    X3 = [0.030, 0.365, 0.700]; W3 = 0.270
    X4 = [0.030, 0.275, 0.520, 0.765]; W4 = 0.180
    SC, BULK, PERT = PALETTE["orange"], PALETTE["blue"], PALETTE["green"]

    def hrow(xs, w, y, colour, items):
        for k, (title, body) in enumerate(items):
            box(ax, xs[k], y, w, H, title, body, colour)
        for k in range(len(items) - 1):
            arrow(ax, xs[k] + w + PAD + CLR, y + H / 2, xs[k + 1] - PAD - CLR, y + H / 2)

    hrow(X3, W3, R1, SC, [
        ("Single-cell data", "TISCH2 HNSCC and laryngeal SCC\n\nmalignant and fibroblast cells labelled\nby the Puram programmes\n\nDirichlet pseudobulk mixtures"),
        ("Classifier training", "multinomial elastic net, three classes\n\nleave-one-dataset-out cross-validation\n\nbenchmarked against ridge, random\nforest and gradient boosting"),
        ("Held-out validation", "remaining single-cell datasets\n\nclass assignment of real pseudobulks\n\nrank agreement, nodal comparisons")])

    hrow(X4, W4, R2, BULK, [
        ("Projection to bulk", "four bulk cohorts\n\nwithin-cohort z-scoring\n\npEMT specificity =\nP(pEMT-high) \u2212 max(others)"),
        ("Co-expression", "signed WGCNA, 8,000 most\nvariable genes\n\nmodule\u2013trait correlation\n\nprogramme enrichment"),
        ("Survival", "published EMT score panel\n\nadjusted Cox per cohort\n\nrandom-effects\nmeta-analysis\n\nsubgroup analyses"),
        ("Independent validation", "published gene programmes\n\nTCGA expression subtypes\n\ngene-level correlation")])

    hrow(X3, W3, R3, PERT, [
        ("Drug proximity", "Guney distance from drug targets to\nseed genes, 1,000 degree-matched draws\n\nDrugBank and STITCH targets\n\nleave-one-seed-out"),
        ("Signature reversal", "SigCom LINCS, L1000\n\ncompound, CRISPR and shRNA libraries\n\nmechanism classes against\nrandom gene sets"),
        ("Cell line screens", "PRISM drug sensitivity\n\nCRISPR gene dependency\n\ncell-line permutation null")])

    def elbow(x_from, y_from, x_to, y_to):
        """Drop, run across, then one arrow of the standard length onto the target box."""
        mid = y_to + ALEN
        ax.plot([x_from, x_from, x_to], [y_from, mid, mid], color=PALETTE["grey"], linewidth=0.7,
                solid_joinstyle="miter", zorder=1)
        arrow(ax, x_to, mid, x_to, y_to)

    # the classifier feeds the bulk projection; the co-expression modules feed the perturbation arm
    elbow(X3[1] + W3 / 2, R1 - PAD - CLR, X4[0] + W4 / 2, R2 + H + PAD + CLR)
    elbow(X4[1] + W4 / 2, R2 - PAD - CLR, X3[0] + W3 / 2, R3 + H + PAD + CLR)
    fig.tight_layout()
    save_figure(fig, FIG, "Figure_1_pipeline_schematic")


def figure_2():
    reg = pd.read_csv(ROOT / "config" / "dataset_registry.tsv", sep="\t")
    summ = pd.read_csv(RES / "filtered_malignant_fibroblast_summary.tsv", sep="\t").set_index("dataset")
    meta = pd.read_csv(ROOT / cfg["paths"]["processed_dir"] / "pseudo_bulk" / "pseudobulk_metadata.tsv", sep="\t", index_col=0)
    cv = pd.read_csv(RES / "multinomial_classifier" / "grouped_cv_performance.tsv", sep="\t")
    bench = RES / "multinomial_classifier" / "classifier_benchmark_folds.tsv"
    bench = pd.read_csv(bench, sep="\t") if bench.exists() else None
    coef = pd.read_csv(RES / "multinomial_classifier" / "classifier_coefficients.tsv", sep="\t", index_col=0)

    fig = plt.figure(figsize=(mm(180), mm(125)))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1, 1.15], height_ratios=[1, 1.15], hspace=0.75, wspace=0.5)

    # A: datasets
    ax = fig.add_subplot(gs[0, 0])
    order = list(reg["dataset_id"])
    labels = {"HNSC_GSE103322": "HNSCC GSE103322\n(Puram)", "OSCC_GSE172577": "OSCC GSE172577", "LSCC_GSE150321": "Laryngeal SCC\nGSE150321",
              "NPC_GSE150430": "NPC GSE150430", "NPC_GSE162025": "NPC GSE162025", "THCA_GSE148673_outgroup": "Thyroid GSE148673\n(outgroup)"}
    y = np.arange(len(order))[::-1]
    ax.barh(y, summ.loc[order, "malignant"], color=PALETTE["orange"], height=0.38, label="malignant")
    ax.barh(y - 0.4, summ.loc[order, "fibroblast_stromal"], color=PALETTE["green"], height=0.38, label="fibroblast / stromal")
    role = dict(zip(reg["dataset_id"], reg["role"]))
    for yi, ds in zip(y, order):
        r = role[ds].replace("_", " ")
        ax.text(max(summ.loc[ds, ["malignant", "fibroblast_stromal"]]) + 150, yi - 0.2, "training" if reg.set_index("dataset_id").loc[ds, "use_for_training"] == "yes" else r,
                va="center", fontsize=5.5, color=PALETTE["grey"])
    ax.set_yticks(y - 0.2); ax.set_yticklabels([labels[d] for d in order], fontsize=6)
    ax.set_xlabel("Cells after QC")
    ax.set_xlim(0, summ[["malignant", "fibroblast_stromal"]].values.max() * 1.6)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.28), ncol=2, fontsize=6, frameon=False)
    ax.set_title("a", loc="left", fontweight="bold")

    # B: training set composition
    ax = fig.add_subplot(gs[0, 1])
    ct = meta.groupby(["source_dataset", "label"]).size().unstack(fill_value=0)
    x = np.arange(len(ct.index))
    bottom = np.zeros(len(ct.index))
    for key, lab in CLS:
        v = ct[key].values if key in ct else np.zeros(len(ct.index))
        ax.bar(x, v, bottom=bottom, color=CLASS_COLOURS[key], width=0.6, label=lab)
        for xi, b, vv in zip(x, bottom, v):
            if vv > 0:
                ax.text(xi, b + vv / 2, str(int(vv)), ha="center", va="center", fontsize=6, color="white")
        bottom += v
    ax.set_xticks(x); ax.set_xticklabels([labels.get(d, d) for d in ct.index], fontsize=6)
    ax.set_ylabel("Pseudobulks (100 cells each)")
    ax.set_ylim(0, ct.sum(axis=1).max() * 1.08)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.1, -0.28), ncol=1, fontsize=5.5, frameon=False)
    ax.set_title("b", loc="left", fontweight="bold")

    # C: held-out performance
    ax = fig.add_subplot(gs[0, 2])
    if bench is not None:
        models_ = [m for m in ["elastic_net", "ridge", "random_forest", "lightgbm", "lightgbm_top2000"] if m in set(bench["model"])]
        names = {"elastic_net": "elastic\nnet", "ridge": "ridge", "random_forest": "random\nforest", "lightgbm": "Light-\nGBM", "lightgbm_top2000": "LightGBM\ntop 2,000"}
        for i, m in enumerate(models_):
            sub = bench[bench["model"] == m]
            for metric, marker, col in (("auroc_pEMT_high", "o", PALETTE["vermilion"]), ("accuracy", "s", PALETTE["blue"])):
                ax.scatter([i + (-0.12 if metric == "accuracy" else 0.12)] * len(sub), sub[metric], s=16, marker=marker, color=col, linewidths=0, zorder=2)
                ax.plot([i + (-0.12 if metric == "accuracy" else 0.12) - 0.1, i + (-0.12 if metric == "accuracy" else 0.12) + 0.1], [sub[metric].mean()] * 2, color=PALETTE["black"], linewidth=0.8)
        ax.set_xticks(range(len(models_))); ax.set_xticklabels([names[m] for m in models_], fontsize=5.5)
        ax.set_xlim(-0.5, len(models_) - 0.5)
        ax.scatter([], [], marker="o", color=PALETTE["vermilion"], s=16, label="AUROC, pEMT-high vs rest")
        ax.scatter([], [], marker="s", color=PALETTE["blue"], s=16, label="accuracy (argmax)")
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.28), ncol=1, fontsize=5.5, frameon=False)
        ax.set_ylim(0.25, 1.03)
        ax.set_ylabel("Held-out dataset (2 folds)")
    else:
        ax.bar([0, 1, 2], [cv["auc_pEMT_high"].mean(), cv["auc_epithelial_like"].mean(), cv["auc_fibroblast_stromal_like"].mean()], color=[CLASS_COLOURS[k] for k, _ in CLS])
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels([l for _, l in CLS], fontsize=6)
        ax.set_ylabel("Leave-one-dataset-out AUROC")
        ax.set_ylim(0.9, 1.0)
    ax.set_title("c", loc="left", fontweight="bold")

    # D: top coefficients per class
    for j, (key, lab) in enumerate(CLS):
        ax = fig.add_subplot(gs[1, j])
        top = coef[key].sort_values(ascending=False).head(15).iloc[::-1]
        ax.barh(np.arange(len(top)), top.values, color=CLASS_COLOURS[key], height=0.7)
        ax.set_yticks(np.arange(len(top))); ax.set_yticklabels(top.index, fontsize=6)
        ax.set_xlabel("Coefficient (standardised)", fontsize=6.5)
        ax.set_title(f"{lab}: {int((coef[key] != 0).sum())} non-zero genes", fontsize=6.5, color=PALETTE["grey"])
        if j == 0:
            ax.set_title("d", loc="left", fontweight="bold")
    save_figure(fig, FIG, "Figure_2_classifier")


def main() -> None:
    figure_1()
    figure_2()


if __name__ == "__main__":
    main()
