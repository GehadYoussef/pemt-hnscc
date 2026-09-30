"""Draw Figure_1, the study workflow and the classifier overview, as one figure.

One matplotlib figure, one gridspec and one run of panel letters:

  a     workflow strip, the five stages of the study
  b     single-cell datasets and their malignant and fibroblast content after QC
  c     composition of the labelled pseudobulk training set
  d     held-out performance across five learners (leave-one-dataset-out AUROC if the benchmark
        table is absent)
  e f g the 15 largest coefficients for each of the three classes

The panels are drawn from saved result files. No analysis is rerun. The per-dataset cell counts in
panel b come from Supplementary Table S7b (the TSV written by 01_build_supplementary_tables.py).
GSE181919 (Choi et al. 2023), used only to replicate the gene partition, is added with its tumour
tissue counts entered directly.

Inputs:  config/dataset_registry.tsv
         results/tables/tsv/S2b_sc_summary_by_cell_type.tsv
         data/processed/pseudo_bulk/pseudobulk_metadata.tsv
         results/multinomial_classifier/grouped_cv_performance.tsv
         results/multinomial_classifier/classifier_benchmark_folds.tsv
         results/multinomial_classifier/classifier_coefficients.tsv
Outputs: results/figures/Figure_1.svg and .png
Usage:   python src/12_figures_and_tables/03_figure1_study_overview.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))
from pemt import load_config, project_root  # noqa: E402
from pemt.figstyle import apply_style, save_figure, PALETTE, CLASS_COLOURS, mm  # noqa: E402

cfg = load_config()
ROOT = project_root()
RES = ROOT / cfg["paths"]["results_dir"]
apply_style()
import matplotlib.pyplot as plt  # noqa: E402

CLS = [("pEMT_high", "pEMT-high"), ("epithelial_like", "epithelial-like"),
       ("fibroblast_stromal_like", "fibroblast / stromal")]

STEPS = [   # (title, subtitle) for each stage of the workflow strip
    ("seven single-cell\ndatasets", "5,902 to 176,440 cells"),
    ("pEMT gene list\nresolved by cell type", "replicated in two datasets"),
    ("three-class\nclassifier", "trained on pseudobulks"),
    ("four bulk cohorts", "864 tumours, 350 deaths"),
    ("cetuximab-treated", "40 patients, 43 xenografts"),
]


def workflow_strip(ax) -> None:
    """Panel a. The five stages, with each arrow placed in the measured gap between its neighbours.

    The titles are centred on one line and their rendered bounding boxes are measured. Each arrow
    is drawn on that line, centred in the gap between two boxes. All arrows have the same length,
    the tightest gap in the row less a fixed padding on each side, kept between 1.5 and 5 axis
    units.
    """
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    Y_TITLE, Y_SUB, PAD = 70, 20, 2.2

    xs = np.linspace(11.5, 88.5, len(STEPS))
    titles = []
    for (title, sub), x in zip(STEPS, xs):
        titles.append(ax.text(x, Y_TITLE, title, fontsize=7.2, ha="center", va="center",
                              fontweight="bold", linespacing=1.5))
        ax.text(x, Y_SUB, sub, fontsize=6.2, ha="center", va="center", color=PALETTE["grey"])

    fig = ax.get_figure()
    fig.canvas.draw()                       # extents are only defined once there is a renderer
    r = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    spans = [(inv.transform((t.get_window_extent(renderer=r).x0, 0))[0],
              inv.transform((t.get_window_extent(renderer=r).x1, 0))[0]) for t in titles]

    # One arrow length for the row, set by the tightest gap.
    gaps = [(spans[i][1], spans[i + 1][0]) for i in range(len(STEPS) - 1)]
    length = max(1.5, min(min(b - a for a, b in gaps) - 2 * PAD, 5.0))
    for a, b in gaps:
        mid = (a + b) / 2
        ax.annotate("", xy=(mid + length / 2, Y_TITLE), xytext=(mid - length / 2, Y_TITLE),
                    arrowprops=dict(arrowstyle="-|>", color=PALETTE["grey"],
                                    lw=0.9, mutation_scale=8))



def classifier_panels(fig, gs) -> list:
    """Panels b to g. Returns the axes in lettering order.

    The columns differ in width, so main() places the letters at one fixed offset in figure
    coordinates.
    """
    reg = pd.read_csv(ROOT / "config" / "dataset_registry.tsv", sep="\t")
    # Per-dataset cell counts after QC, from Supplementary Table S7b (cell-level rows). Datasets
    # with no annotated fibroblasts contribute a zero.
    s1b = pd.read_csv(ROOT / "results" / "tables" / "tsv" / "S2b_sc_summary_by_cell_type.tsv",
                      sep="\t")
    s1b = s1b[s1b["level"] == "cell"]
    summ = (s1b.pivot_table(index="dataset_id", columns="standard_cell_type", values="n",
                            aggfunc="sum")
            .reindex(columns=["malignant", "fibroblast_stromal"])
            .fillna(0.0))
    summ.index.name = "dataset"
    meta = pd.read_csv(ROOT / cfg["paths"]["processed_dir"] / "pseudo_bulk" /
                       "pseudobulk_metadata.tsv", sep="\t", index_col=0)
    cv = pd.read_csv(RES / "multinomial_classifier" / "grouped_cv_performance.tsv", sep="\t")
    bench = RES / "multinomial_classifier" / "classifier_benchmark_folds.tsv"
    bench = pd.read_csv(bench, sep="\t") if bench.exists() else None
    coef = pd.read_csv(RES / "multinomial_classifier" / "classifier_coefficients.tsv",
                       sep="\t", index_col=0)

    labels = {"HNSC_GSE103322": "HNSCC GSE103322\n(Puram)", "OSCC_GSE172577": "OSCC GSE172577",
              "LSCC_GSE150321": "Laryngeal SCC\nGSE150321", "NPC_GSE150430": "NPC GSE150430",
              "NPC_GSE162025": "NPC GSE162025", "THCA_GSE148673_outgroup": "Thyroid GSE148673\n(outgroup)"}

    # b: datasets
    ax = fig.add_subplot(gs[0, 0])
    order = list(reg["dataset_id"])
    # GSE181919 (Choi et al. 2023), tumour tissue, used only to replicate the partition
    summ.loc["GSE181919"] = [5113, 1981]
    labels["GSE181919"] = "HNSCC GSE181919\n(replication)"
    order.append("GSE181919")
    y = np.arange(len(order))[::-1]
    ax.barh(y, summ.loc[order, "malignant"], color=PALETTE["orange"], height=0.38, label="malignant")
    ax.barh(y - 0.4, summ.loc[order, "fibroblast_stromal"], color=PALETTE["green"],
            height=0.38, label="fibroblast / stromal")
    role = dict(zip(reg["dataset_id"], reg["role"]))
    train = dict(zip(reg["dataset_id"], reg["use_for_training"]))
    for yi, ds in zip(y, order):
        r = "partition replication" if ds == "GSE181919" else (
            "training" if train[ds] == "yes" else role[ds].replace("_", " "))
        ax.text(max(summ.loc[ds, ["malignant", "fibroblast_stromal"]]) + 150, yi - 0.2, r,
                va="center", fontsize=5.5, color=PALETTE["grey"])
    ax.set_yticks(y - 0.2); ax.set_yticklabels([labels[d] for d in order], fontsize=6)
    ax.set_xlabel("Cells after QC")
    ax.set_xlim(0, summ[["malignant", "fibroblast_stromal"]].values.max() * 1.6)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.28), ncol=2, fontsize=6, frameon=False)
    axes = [ax]

    # c: training set composition
    ax = fig.add_subplot(gs[0, 1])
    ct = meta.groupby(["source_dataset", "label"]).size().unstack(fill_value=0)
    x = np.arange(len(ct.index))
    bottom = np.zeros(len(ct.index))
    for key, lab in CLS:
        v = ct[key].values if key in ct else np.zeros(len(ct.index))
        ax.bar(x, v, bottom=bottom, color=CLASS_COLOURS[key], width=0.6, label=lab)
        for xi, b, vv in zip(x, bottom, v):
            if vv > 0:
                ax.text(xi, b + vv / 2, str(int(vv)), ha="center", va="center",
                        fontsize=6, color="white")
        bottom += v
    ax.set_xticks(x); ax.set_xticklabels([labels.get(d, d) for d in ct.index], fontsize=6)
    ax.set_ylabel("Pseudobulks (100 cells each)")
    ax.set_ylim(0, ct.sum(axis=1).max() * 1.08)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.1, -0.28), ncol=1, fontsize=5.5, frameon=False)
    axes.append(ax)

    # d: held-out performance
    ax = fig.add_subplot(gs[0, 2])
    if bench is not None:
        models_ = [m for m in ["elastic_net", "ridge", "random_forest", "lightgbm",
                               "lightgbm_top2000"] if m in set(bench["model"])]
        names = {"elastic_net": "elastic\nnet", "ridge": "ridge", "random_forest": "random\nforest",
                 "lightgbm": "Light-\nGBM", "lightgbm_top2000": "LightGBM\ntop 2,000"}
        for i, m in enumerate(models_):
            sub = bench[bench["model"] == m]
            for metric, marker, col in (("auroc_pEMT_high", "o", PALETTE["vermilion"]),
                                        ("accuracy", "s", PALETTE["blue"])):
                off = -0.12 if metric == "accuracy" else 0.12
                ax.scatter([i + off] * len(sub), sub[metric], s=16, marker=marker,
                           color=col, linewidths=0, zorder=2)
                ax.plot([i + off - 0.1, i + off + 0.1], [sub[metric].mean()] * 2,
                        color=PALETTE["black"], linewidth=0.8)
        ax.set_xticks(range(len(models_)))
        ax.set_xticklabels([names[m] for m in models_], fontsize=5.5)
        ax.set_xlim(-0.5, len(models_) - 0.5)
        ax.scatter([], [], marker="o", color=PALETTE["vermilion"], s=16,
                   label="AUROC, pEMT-high vs rest")
        ax.scatter([], [], marker="s", color=PALETTE["blue"], s=16, label="accuracy (argmax)")
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.28), ncol=1, fontsize=5.5, frameon=False)
        ax.set_ylim(0.25, 1.03)
        ax.set_ylabel("Held-out dataset (2 folds)")
    else:
        ax.bar([0, 1, 2], [cv["auc_pEMT_high"].mean(), cv["auc_epithelial_like"].mean(),
                           cv["auc_fibroblast_stromal_like"].mean()],
               color=[CLASS_COLOURS[k] for k, _ in CLS])
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels([l for _, l in CLS], fontsize=6)
        ax.set_ylabel("Leave-one-dataset-out AUROC")
        ax.set_ylim(0.9, 1.0)
    axes.append(ax)

    # e, f, g: top coefficients per class. Each class has its own axes and its own letter.
    for j, (key, lab) in enumerate(CLS):
        ax = fig.add_subplot(gs[1, j])
        top = coef[key].sort_values(ascending=False).head(15).iloc[::-1]
        ax.barh(np.arange(len(top)), top.values, color=CLASS_COLOURS[key], height=0.7)
        ax.set_yticks(np.arange(len(top))); ax.set_yticklabels(top.index, fontsize=6)
        ax.set_xlabel("Coefficient (standardised)", fontsize=6.5)
        ax.set_title(f"{lab}: {int((coef[key] != 0).sum())} non-zero genes",
                     fontsize=6.5, color=PALETTE["grey"])
        axes.append(ax)
    return axes


def main() -> None:
    # The lower block is a 2x3 gridspec occupying 125 mm of the figure. The strip takes the 18 mm
    # above it as separate axes, so the gridspec spacing cannot squeeze it.
    strip_mm, panel_mm = 18.0, 125.0
    fig = plt.figure(figsize=(mm(180), mm(strip_mm + panel_mm)))
    frac = panel_mm / (strip_mm + panel_mm)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1, 1.15], height_ratios=[1, 1.15],
                          hspace=0.75, wspace=0.5, top=frac * 0.97, bottom=0.05,
                          left=0.085, right=0.985)
    ax_strip = fig.add_axes([0.01, frac + 0.005, 0.98, (1 - frac) - 0.02])
    workflow_strip(ax_strip)
    axes = classifier_panels(fig, gs)

    # One offset for every letter, in figure coordinates, so each sits the same distance from its
    # panel whatever the column width. The lower row clears the centred class title above it.
    DX, DY, DY_TITLED = 0.052, 0.014, 0.040
    x_letter = max(axes[0].get_position().x0 - DX, 0.004)
    for letter, ax in zip("bcdefg", axes):
        pos = ax.get_position()
        dy = DY_TITLED if letter in "efg" else DY
        fig.text(max(pos.x0 - DX, 0.004), min(pos.y1 + dy, 0.995), letter,
                 fontsize=8, fontweight="bold", ha="left", va="bottom")
    # The strip spans the full width. Its letter is aligned with the letter of the first column.
    fig.text(x_letter, 0.985, "a", fontsize=8, fontweight="bold", ha="left", va="top")
    save_figure(fig, OUT, "Figure_1")
    print(f"wrote {OUT / 'Figure_1.png'}")


if __name__ == "__main__":
    main()
