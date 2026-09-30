"""Draw Figure_4, the EMT score panel and survival, as one figure.

  a  pEMT specificity against each published EMT score in TCGA-HNSC primaries, with Spearman rho
  b  Kaplan-Meier overall survival by tertile of pEMT specificity, with log-rank p and numbers at risk
  c  Kaplan-Meier by tertile of the canonical Puram pEMT signature
  d  head-to-head adjusted Cox models, each score entered together with pEMT specificity (HR per SD)
  e  adjusted hazard ratio per SD in each of four cohorts (TCGA-HNSC, GSE41613, CPTAC-3, GSE65858)
     and the DerSimonian-Laird pooled estimate, for five scores

The panels are drawn from saved result files. No analysis is rerun.

Inputs:  results/tcga_projection/emt_score_panel_scores.tsv
         results/tcga_projection/emt_score_panel_correlations.tsv
         results/tcga_projection/emt_score_panel_joint_cox.tsv
         results/tcga_projection/tcga_master_trait_table.tsv
         results/tcga_projection/survival_meta_analysis.tsv
         results/tcga_projection/survival_meta_inputs.tsv
Outputs: results/figures/Figure_4.svg and .png
Usage:   python src/12_figures_and_tables/04_figure4_survival_and_scores.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIG = ROOT / "results" / "figures"

sys.path.insert(0, str(ROOT / "src"))
from pemt import load_config  # noqa: E402
from pemt.figstyle import apply_style, save_figure, PALETTE, mm  # noqa: E402
from pemt.survival import primary_tumours  # noqa: E402

cfg = load_config()
TC = ROOT / cfg["paths"]["results_dir"] / "tcga_projection"
apply_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402
from matplotlib.ticker import NullLocator, NullFormatter, FixedLocator  # noqa: E402

TERTILE_COLOURS = {"Low": PALETTE["blue"], "Mid": PALETTE["grey"], "High": PALETTE["vermilion"]}
SHORT = {"pEMT_specificity": "pEMT specificity", "P_pEMT_high": "P(pEMT-high)", "GS76": "76GS",
         "KS": "KS", "hallmark_EMT": "Hallmark EMT", "puram_pemt": "Puram pEMT",
         "puram_epi_dif_1": "Puram epithelial", "MLR_mu": "MLR"}


def label(ax, letter, dx=-0.12, dy=1.04):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8, fontweight="bold",
            ha="left", va="bottom")


def main() -> None:
    scores = pd.read_csv(TC / "emt_score_panel_scores.tsv", sep="\t", index_col=0)
    rho = pd.read_csv(TC / "emt_score_panel_correlations.tsv", sep="\t", index_col=0)
    joint = pd.read_csv(TC / "emt_score_panel_joint_cox.tsv", sep="\t")
    traits = primary_tumours(pd.read_csv(TC / "tcga_master_trait_table.tsv", sep="\t",
                                         index_col=0, low_memory=False))
    traits = traits.join(scores[["puram_pemt"]], how="left")
    base = traits.copy()
    base["OS_time"] = pd.to_numeric(base["OS_time"], errors="coerce")
    base["OS_event"] = pd.to_numeric(base["OS_event"], errors="coerce")
    meta = pd.read_csv(TC / "survival_meta_analysis.tsv", sep="\t").set_index("score")
    inputs = pd.read_csv(TC / "survival_meta_inputs.tsv", sep="\t")

    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test

    others = [c for c in scores.columns if c not in ("pEMT_specificity", "P_pEMT_high")]
    show = ["pEMT_specificity", "puram_pemt", "hallmark_EMT", "MLR_mu", "GS76"]
    lab = {"pEMT_specificity": "pEMT specificity", "puram_pemt": "Puram pEMT",
           "hallmark_EMT": "Hallmark EMT", "MLR_mu": "MLR-EMT", "GS76": "76GS"}
    order = ["TCGA-HNSC", "GSE41613", "CPTAC_HNSCC", "GSE65858"]
    cl = {"TCGA-HNSC": "TCGA-HNSC", "GSE41613": "GSE41613", "CPTAC_HNSCC": "CPTAC-3",
          "GSE65858": "GSE65858"}

    fig = plt.figure(figsize=(mm(180), mm(205)))
    outer = GridSpec(4, 1, height_ratios=[0.60, 1.30, 0.86, 1.02], hspace=0.78, figure=fig)

    # ---------------------------------------------------------------- a: correlations
    ga = GridSpecFromSubplotSpec(1, len(others), subplot_spec=outer[0], wspace=0.22)
    for j, c in enumerate(others):
        ax = fig.add_subplot(ga[0, j])
        ax.scatter(scores[c], scores["pEMT_specificity"], s=3, alpha=0.35,
                   color=PALETTE["grey"], linewidths=0, rasterized=True)
        lo, hi = np.nanpercentile(scores[c], [0.5, 99.5])
        pad = 0.05 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)
        ax.text(0.04, 0.96, f"ρ = {rho.loc['pEMT_specificity', c]:.2f}",
                transform=ax.transAxes, va="top", fontsize=6.5)
        ax.set_xlabel(SHORT[c], fontsize=7)
        ax.set_yticks([-1, 0, 1])
        if j == 0:
            ax.set_ylabel("pEMT specificity", fontsize=7)
            label(ax, "a", dx=-0.38, dy=1.06)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=6)

    # ---------------------------------------------------------------- b, c: Kaplan-Meier
    gk = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[1], height_ratios=[4.6, 1.0],
                                 hspace=0.06, wspace=0.30)
    for col, (score, title, letter) in enumerate(
            (("pEMT_specificity", "pEMT specificity", "b"),
             ("puram_pemt", "Canonical pEMT signature (stromal)", "c"))):
        sub = base.dropna(subset=["OS_time", "OS_event", score]).copy()
        sub = sub[sub["OS_time"] > 0]
        sub["tertile"] = pd.qcut(sub[score], q=3, labels=["Low", "Mid", "High"])
        lr = multivariate_logrank_test(sub["OS_time"], sub["tertile"], sub["OS_event"])
        ax = fig.add_subplot(gk[0, col])
        axr = fig.add_subplot(gk[1, col])
        times = np.arange(0, 11, 2)
        at_risk = {}
        for grp in ["Low", "Mid", "High"]:
            s = sub[sub["tertile"] == grp]
            tt = s["OS_time"] / 365.25
            kmf = KaplanMeierFitter()
            kmf.fit(tt, s["OS_event"], label=f"{grp} (n = {len(s)}, e = {int(s['OS_event'].sum())})")
            kmf.plot_survival_function(ax=ax, color=TERTILE_COLOURS[grp], ci_show=False,
                                       linewidth=1.2)
            at_risk[grp] = [int((tt >= x).sum()) for x in times]
        ax.set_xlim(0, 10); ax.set_ylim(0, 1.0); ax.set_xlabel("")
        ax.set_ylabel("Overall survival probability" if col == 0 else "")
        if col:
            ax.tick_params(labelleft=False)
        ax.text(0.03, 0.06, f"Log-rank p = {lr.p_value:.3f}", transform=ax.transAxes, fontsize=6.5)
        ax.legend(loc="upper right", fontsize=5.6, frameon=False)
        ax.tick_params(labelbottom=False)
        ax.set_title(title, fontsize=7, color=PALETTE["grey"])
        label(ax, letter, dx=-0.16 if col == 0 else -0.06)
        for i2, grp in enumerate(["Low", "Mid", "High"]):
            for x, n in zip(times, at_risk[grp]):
                axr.text(x, 2 - i2, str(n), ha="center", va="center", fontsize=5.6,
                         color=TERTILE_COLOURS[grp])
        axr.set_xlim(0, 10); axr.set_ylim(-0.6, 2.6)
        axr.set_yticks([2, 1, 0])
        axr.set_yticklabels(["Low", "Mid", "High"] if col == 0 else ["", "", ""], fontsize=5.8)
        axr.set_xticks(times)
        axr.set_xlabel("Years from diagnosis", fontsize=7)
        axr.set_ylabel("At risk" if col == 0 else "", fontsize=6.2)
        for sp in ("top", "right", "left"):
            axr.spines[sp].set_visible(False)
        axr.tick_params(axis="y", length=0, pad=14)

    # ---------------------------------------------------------------- d: head to head
    gd = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2], width_ratios=[1.0, 0.02], wspace=0.05)
    axC = fig.add_subplot(gd[0, 0])
    yj = np.arange(len(joint))[::-1]
    for i, (_, row) in enumerate(joint.iterrows()):
        axC.errorbar(row["HR_pEMT_specificity_given_score"], yj[i] + 0.18,
                     xerr=[[row["HR_pEMT_specificity_given_score"] - row["CI_low_pEMT"]],
                           [row["CI_up_pEMT"] - row["HR_pEMT_specificity_given_score"]]],
                     fmt="o", color=PALETTE["vermilion"], ecolor=PALETTE["vermilion"],
                     capsize=2, markersize=3.5, elinewidth=0.8)
        axC.errorbar(row["HR_score_given_pEMT_specificity"], yj[i] - 0.18,
                     xerr=[[row["HR_score_given_pEMT_specificity"] - row["CI_low_score"]],
                           [row["CI_up_score"] - row["HR_score_given_pEMT_specificity"]]],
                     fmt="s", color=PALETTE["black"], ecolor=PALETTE["black"],
                     capsize=2, markersize=3.5, elinewidth=0.8)
    axC.axvline(1.0, ls="--", color=PALETTE["grey"], linewidth=0.6)
    axC.set_yticks(yj)
    axC.set_yticklabels([SHORT[c] for c in joint["score"]], fontsize=7)
    axC.set_xlabel("HR per SD, both scores in one adjusted model (95% CI)")
    axC.scatter([], [], marker="o", color=PALETTE["vermilion"], s=14,
                label="pEMT specificity, given the score")
    axC.scatter([], [], marker="s", color=PALETTE["black"], s=14,
                label="the score, given pEMT specificity")
    axC.legend(loc="lower left", bbox_to_anchor=(0.0, 1.03), fontsize=6, frameon=False, ncol=2)
    label(axC, "d", dx=-0.16, dy=1.19)

    # ---------------------------------------------------------------- e: meta-analysis
    ge = GridSpecFromSubplotSpec(1, len(show), subplot_spec=outer[3], wspace=0.18)
    for k, sc in enumerate(show):
        axm = fig.add_subplot(ge[0, k])
        # keep one row per cohort so every panel aligns with the labels. A cohort without an
        # estimate is left blank.
        d = inputs[inputs["score"] == sc].set_index("cohort").reindex(order)
        m = meta.loc[sc]
        y = np.arange(len(d))[::-1]
        for yi, (_, r) in zip(y, d.iterrows()):
            if pd.isna(r["HR"]):
                continue
            axm.plot([r["CI_low"], r["CI_up"]], [yi, yi], color=PALETTE["grey"],
                     linewidth=0.9, zorder=2)
            axm.scatter(r["HR"], yi, s=11, color=PALETTE["grey"], marker="s", zorder=3,
                        linewidths=0)
        axm.plot([m["CI_low"], m["CI_up"]], [-1.15, -1.15], color=PALETTE["vermilion"],
                 linewidth=1.4, zorder=3)
        axm.scatter(m["HR"], -1.15, s=30, marker="D", color=PALETTE["vermilion"], zorder=4,
                    linewidths=0)
        axm.axvline(1, color=PALETTE["black"], linewidth=0.6)
        axm.set_xscale("log"); axm.set_xlim(0.45, 3.2)
        axm.xaxis.set_minor_locator(NullLocator())
        axm.xaxis.set_minor_formatter(NullFormatter())
        axm.xaxis.set_major_locator(FixedLocator([0.5, 1, 2]))
        axm.set_xticklabels(["0.5", "1", "2"], fontsize=6)
        axm.set_ylim(-2.0, len(d) - 0.4)
        axm.set_yticks(list(y) + [-1.15])
        axm.set_yticklabels(([cl.get(c, c) for c in d.index] + ["Pooled"]) if k == 0
                            else [""] * (len(d) + 1), fontsize=6)
        axm.tick_params(axis="y", length=0)
        for sp in ("top", "right", "left"):
            axm.spines[sp].set_visible(False)
        star = "*" if m["p_value"] < 0.05 else ""
        axm.set_title(f"{lab[sc]}\n{m['HR']:.2f} ({m['CI_low']:.2f}-{m['CI_up']:.2f}){star}\n"
                      f"I² = {m['I2_percent']:.0f}%", fontsize=6.0, color=PALETTE["grey"])
        if k == 0:
            label(axm, "e", dx=-0.62, dy=1.34)

    fig.text(0.5, 0.008, "e: adjusted hazard ratio per standard deviation, log scale. "
             "Diamond: DerSimonian-Laird pooled estimate across the four cohorts.",
             ha="center", fontsize=6, color=PALETTE["grey"])
    fig.subplots_adjust(left=0.115, right=0.975, top=0.965, bottom=0.048)
    save_figure(fig, FIG, "Figure_4")
    print(f"wrote {FIG / 'Figure_4.png'}")


if __name__ == "__main__":
    main()
