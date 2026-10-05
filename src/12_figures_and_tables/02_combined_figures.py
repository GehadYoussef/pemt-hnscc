"""Draw the multi-panel survival and cell-line figures as single files, with panels a, b, c.

The panels come from different analyses. They are redrawn here from the saved result tables, so
every figure stays vector.

  Supplementary_Figure_11   (a) Kaplan-Meier overall survival by tertile of pEMT specificity in
                            TCGA-HNSC primaries, with log-rank p and numbers at risk, (b) forest
                            plot of the adjusted Cox model. Covariates with a non-finite or zero
                            hazard ratio are separation artefacts and are left out.
  tcga_survival_overview    (a, b) Kaplan-Meier by tertile of pEMT specificity and of the canonical
                            Puram pEMT score, (c) adjusted hazard ratio per SD in each of four
                            cohorts and the DerSimonian-Laird pooled estimate, for five scores.
  Supplementary_Figure_15   (a) PRISM, Spearman of pEMT specificity with area under the curve per
                            compound in HNSCC models, with the canonical_pEMT DrugBank proximity
                            hits marked, (b) CRISPR, Spearman with Chronos gene effect
                            per gene, with the M13 and M06 module seed genes marked, (c) mean
                            Spearman per mechanism-of-action class against the permutation null in
                            HNSCC and pan-squamous models.

In Supplementary_Figure_15c the permutation null is drawn from its saved mean and standard
deviation (mean +/- 1.96 SD), so the interval is a normal approximation to the empirical one. The
p-values used to mark classes are the empirical permutation p-values.

Inputs:  results/tcga_projection/tcga_master_trait_table.tsv
         results/tcga_projection/cox_multivariable_M2.tsv
         results/tcga_projection/emt_score_panel_scores.tsv
         results/tcga_projection/survival_meta_analysis.tsv
         results/tcga_projection/survival_meta_inputs.tsv
         results/depmap_broad_prism/prism_pemt_sensitivity_hnscc.tsv
         results/depmap_broad_prism/crispr_pemt_dependency_hnscc.tsv
         results/depmap_broad_prism/prism_moa_permutation_test.tsv
Outputs: results/figures/Supplementary_Figure_11, tcga_survival_overview and
         Supplementary_Figure_15, each as .svg and .png
Usage:   python src/12_figures_and_tables/02_combined_figures.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pemt import load_config, project_root  # noqa: E402
from pemt.figstyle import apply_style, save_figure, PALETTE, TERTILE_COLOURS, mm  # noqa: E402
from pemt.survival import primary_tumours  # noqa: E402

cfg = load_config()
ROOT = project_root()
RES = ROOT / cfg["paths"]["results_dir"]
TC = RES / "tcga_projection"
DM = RES / "depmap_broad_prism"
FIG = ROOT / "results" / "figures"
apply_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

SCORE = "pEMT_specificity"


def label(ax, letter, dx=-0.12, dy=1.04):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontweight="bold", fontsize=8)


def figure_tcga_cox():
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test
    traits = primary_tumours(pd.read_csv(TC / "tcga_master_trait_table.tsv", sep="\t", index_col=0, low_memory=False))
    sub = traits.dropna(subset=["OS_time", "OS_event", SCORE]).copy()
    sub["OS_time"] = pd.to_numeric(sub["OS_time"], errors="coerce")
    sub["OS_event"] = pd.to_numeric(sub["OS_event"], errors="coerce")
    sub = sub[sub["OS_time"] > 0]
    sub["tertile"] = pd.qcut(sub[SCORE], q=3, labels=["Low", "Mid", "High"])
    lr = multivariate_logrank_test(sub["OS_time"], sub["tertile"], sub["OS_event"])

    fig = plt.figure(figsize=(mm(180), mm(88)))
    gs = GridSpec(2, 2, height_ratios=[5, 1.1], width_ratios=[1, 0.95], hspace=0.10, wspace=0.62, figure=fig)
    ax, axr = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    times = np.arange(0, 11, 2)
    at_risk = {}
    for grp in ["Low", "Mid", "High"]:
        s = sub[sub["tertile"] == grp]
        t = s["OS_time"] / 365.25
        kmf = KaplanMeierFitter()
        kmf.fit(t, s["OS_event"], label=f"{grp} tertile (n = {len(s)}, events = {int(s['OS_event'].sum())})")
        kmf.plot_survival_function(ax=ax, color=TERTILE_COLOURS[grp], ci_show=False, linewidth=1.2)
        at_risk[grp] = [int((t >= x).sum()) for x in times]
    ax.set_ylabel("Overall survival probability")
    ax.set_xlabel("")
    ax.set_xlim(0, 10); ax.set_ylim(0, 1.0)
    ax.text(0.03, 0.05, f"Log-rank p = {lr.p_value:.3f}", transform=ax.transAxes, fontsize=7)
    ax.legend(loc="upper right", fontsize=6)
    ax.tick_params(labelbottom=False)
    label(ax, "a", dx=-0.16)
    for i, grp in enumerate(["Low", "Mid", "High"]):
        for x, n in zip(times, at_risk[grp]):
            axr.text(x, 2 - i, str(n), ha="center", va="center", fontsize=6, color=TERTILE_COLOURS[grp])
    axr.set_xlim(0, 10); axr.set_ylim(-0.6, 2.6)
    axr.set_yticks([2, 1, 0]); axr.set_yticklabels(["Low", "Mid", "High"], fontsize=6)
    axr.set_xticks(times); axr.set_xlabel("Years from diagnosis")
    axr.set_ylabel("At risk", fontsize=6.5)
    for sp in ("top", "right", "left"):
        axr.spines[sp].set_visible(False)
    axr.tick_params(axis="y", length=0, pad=16)

    axf = fig.add_subplot(gs[:, 1])
    # The adjusted model M2 of 04_tcga_projection/06_survival_analysis.py (stage, age, site, HPV)
    cox = pd.read_csv(TC / "cox_multivariable_M2.tsv", sep="\t")
    # drop covariates the model could not estimate (a site level with too few events gives HR 0 and an
    # infinite upper bound)
    cox = cox[np.isfinite(cox["HR"]) & np.isfinite(cox["HR_up_95"]) & (cox["HR"] > 0)]
    pretty = {"pEMT_specificity": "pEMT specificity", "stage_num": "Pathological stage", "age": "Age",
              "pack_years": "Pack-years", "hpv_positive": "HPV positive",
              "site_group_oral_cavity": "Oral cavity", "site_group_oropharynx": "Oropharynx",
              "site_group_larynx_hypopharynx": "Larynx / hypopharynx", "site_group_other": "Other site",
              "site_oral_cavity": "Oral cavity", "site_larynx_hypopharynx": "Larynx / hypopharynx"}
    cox["label"] = cox["covariate"].map(lambda c: pretty.get(c, c.replace("_", " ")))
    cox = cox.iloc[::-1]
    y = np.arange(len(cox))
    for yi, r in zip(y, cox.itertuples()):
        col = PALETTE["vermilion"] if r.covariate == SCORE else PALETTE["grey"]
        axf.plot([r.HR_low_95, r.HR_up_95], [yi, yi], color=col, linewidth=1.0, zorder=2)
        axf.scatter(r.HR, yi, s=18, color=col, marker="s", zorder=3, linewidths=0)
        axf.text(1.03, (yi + 0.5) / len(cox), f"{r.HR:.2f} ({r.HR_low_95:.2f}–{r.HR_up_95:.2f})  p = {r.p_value:.3f}",
                 transform=axf.transAxes, va="center", fontsize=5.8, color=PALETTE["grey"])
    axf.axvline(1, color=PALETTE["black"], linewidth=0.6)
    axf.set_xscale("log")
    from matplotlib.ticker import NullLocator, NullFormatter, FixedLocator
    axf.xaxis.set_minor_locator(NullLocator()); axf.xaxis.set_minor_formatter(NullFormatter())
    axf.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 4]))
    axf.set_xticklabels(["0.5", "1", "2", "4"], fontsize=6)
    axf.set_yticks(y); axf.set_yticklabels(cox["label"], fontsize=6.5)
    axf.set_ylim(-0.7, len(cox) - 0.3)
    axf.tick_params(axis="y", length=0)
    for sp in ("top", "right", "left"):
        axf.spines[sp].set_visible(False)
    axf.set_xlabel("Hazard ratio (95% CI)")
    n, ev = int(cox["n"].iloc[0]), int(cox["events"].iloc[0])
    axf.set_title(f"Adjusted Cox model, {n} tumours, {ev} deaths", fontsize=7, color=PALETTE["grey"])
    label(axf, "b", dx=-0.52, dy=1.06)
    fig.tight_layout(rect=(0, 0, 0.99, 1))
    save_figure(fig, FIG, "Supplementary_Figure_11")



def figure_survival():
    """Survival overview.

    a and b apply the same Kaplan-Meier analysis to pEMT specificity and to the canonical Puram pEMT
    score in TCGA-HNSC primaries. c is the four-cohort meta-analysis of the adjusted hazard ratio
    per SD.
    """
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test
    from matplotlib.ticker import NullLocator, NullFormatter, FixedLocator
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    traits = primary_tumours(pd.read_csv(TC / "tcga_master_trait_table.tsv", sep="\t", index_col=0, low_memory=False))
    panel = pd.read_csv(TC / "emt_score_panel_scores.tsv", sep="\t", index_col=0)
    traits = traits.join(panel[["puram_pemt"]], how="left")
    base = traits.copy()
    base["OS_time"] = pd.to_numeric(base["OS_time"], errors="coerce")
    base["OS_event"] = pd.to_numeric(base["OS_event"], errors="coerce")

    meta = pd.read_csv(TC / "survival_meta_analysis.tsv", sep="\t").set_index("score")
    inputs = pd.read_csv(TC / "survival_meta_inputs.tsv", sep="\t")
    show = ["pEMT_specificity", "puram_pemt", "hallmark_EMT", "MLR_mu", "GS76"]
    lab = {"pEMT_specificity": "pEMT specificity", "puram_pemt": "Puram pEMT", "hallmark_EMT": "Hallmark EMT",
           "MLR_mu": "MLR-EMT", "GS76": "76GS"}
    order = ["TCGA-HNSC", "GSE41613", "CPTAC_HNSCC", "GSE65858"]
    cl = {"TCGA-HNSC": "TCGA-HNSC", "GSE41613": "GSE41613", "CPTAC_HNSCC": "CPTAC-3", "GSE65858": "GSE65858"}

    fig = plt.figure(figsize=(mm(180), mm(150)))
    outer = GridSpec(2, 1, height_ratios=[1.0, 1.02], hspace=0.42, figure=fig)
    top = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0], height_ratios=[4.6, 1.0], hspace=0.06, wspace=0.30)

    for col, (score, title, letter) in enumerate((("pEMT_specificity", "Tumour-intrinsic axis", "a"),
                                                  ("puram_pemt", "Canonical pEMT signature (stromal)", "b"))):
        sub = base.dropna(subset=["OS_time", "OS_event", score]).copy()
        sub = sub[sub["OS_time"] > 0]
        sub["tertile"] = pd.qcut(sub[score], q=3, labels=["Low", "Mid", "High"])
        lr = multivariate_logrank_test(sub["OS_time"], sub["tertile"], sub["OS_event"])
        ax = fig.add_subplot(top[0, col]); axr = fig.add_subplot(top[1, col])
        times = np.arange(0, 11, 2); at_risk = {}
        for grp in ["Low", "Mid", "High"]:
            s = sub[sub["tertile"] == grp]
            tt = s["OS_time"] / 365.25
            kmf = KaplanMeierFitter()
            kmf.fit(tt, s["OS_event"], label=f"{grp} (n = {len(s)}, e = {int(s['OS_event'].sum())})")
            kmf.plot_survival_function(ax=ax, color=TERTILE_COLOURS[grp], ci_show=False, linewidth=1.2)
            at_risk[grp] = [int((tt >= x).sum()) for x in times]
        ax.set_xlim(0, 10); ax.set_ylim(0, 1.0); ax.set_xlabel("")
        ax.set_ylabel("Overall survival probability" if col == 0 else "")
        if col: ax.tick_params(labelleft=False)
        ax.text(0.03, 0.06, f"Log-rank p = {lr.p_value:.3f}", transform=ax.transAxes, fontsize=6.5)
        ax.legend(loc="upper right", fontsize=5.6, frameon=False)
        ax.tick_params(labelbottom=False)
        ax.set_title(title, fontsize=7, color=PALETTE["grey"])
        label(ax, letter, dx=-0.16 if col == 0 else -0.06)
        for i2, grp in enumerate(["Low", "Mid", "High"]):
            for x, n in zip(times, at_risk[grp]):
                axr.text(x, 2 - i2, str(n), ha="center", va="center", fontsize=5.6, color=TERTILE_COLOURS[grp])
        axr.set_xlim(0, 10); axr.set_ylim(-0.6, 2.6)
        axr.set_yticks([2, 1, 0])
        axr.set_yticklabels(["Low", "Mid", "High"] if col == 0 else ["", "", ""], fontsize=5.8)
        axr.set_xticks(times); axr.set_xlabel("Years from diagnosis", fontsize=7)
        axr.set_ylabel("At risk" if col == 0 else "", fontsize=6.2)
        for sp in ("top", "right", "left"):
            axr.spines[sp].set_visible(False)
        axr.tick_params(axis="y", length=0, pad=14)

    bot = GridSpecFromSubplotSpec(1, len(show), subplot_spec=outer[1], wspace=0.18)
    for k, sc in enumerate(show):
        axm = fig.add_subplot(bot[0, k])
        d = inputs[inputs["score"] == sc].set_index("cohort").reindex(order).dropna(subset=["HR"])
        m = meta.loc[sc]
        y = np.arange(len(d))[::-1]
        for yi, (_, r) in zip(y, d.iterrows()):
            axm.plot([r["CI_low"], r["CI_up"]], [yi, yi], color=PALETTE["grey"], linewidth=0.9, zorder=2)
            axm.scatter(r["HR"], yi, s=11, color=PALETTE["grey"], marker="s", zorder=3, linewidths=0)
        axm.plot([m["CI_low"], m["CI_up"]], [-1.15, -1.15], color=PALETTE["vermilion"], linewidth=1.4, zorder=3)
        axm.scatter(m["HR"], -1.15, s=30, marker="D", color=PALETTE["vermilion"], zorder=4, linewidths=0)
        axm.axvline(1, color=PALETTE["black"], linewidth=0.6)
        axm.set_xscale("log"); axm.set_xlim(0.45, 3.2)
        axm.xaxis.set_minor_locator(NullLocator()); axm.xaxis.set_minor_formatter(NullFormatter())
        axm.xaxis.set_major_locator(FixedLocator([0.5, 1, 2]))
        axm.set_xticklabels(["0.5", "1", "2"], fontsize=6)
        axm.set_ylim(-2.0, len(d) - 0.4)
        axm.set_yticks(list(y) + [-1.15])
        axm.set_yticklabels(([cl.get(c, c) for c in d.index] + ["Pooled"]) if k == 0 else [""] * (len(d) + 1), fontsize=6)
        axm.tick_params(axis="y", length=0)
        for sp in ("top", "right", "left"):
            axm.spines[sp].set_visible(False)
        star = "*" if m["p_value"] < 0.05 else ""
        axm.set_title(f"{lab[sc]}\n{m['HR']:.2f} ({m['CI_low']:.2f}-{m['CI_up']:.2f}){star}\nI² = {m['I2_percent']:.0f}%",
                      fontsize=6.0, color=PALETTE["grey"])
        if k == 0:
            label(axm, "c", dx=-0.62, dy=1.34)
    fig.text(0.5, 0.012, "c: adjusted hazard ratio per standard deviation, log scale. Diamond: DerSimonian-Laird pooled estimate across the four cohorts.",
             ha="center", fontsize=6, color=PALETTE["grey"])
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save_figure(fig, FIG, "tcga_survival_overview")


def figure_cell_lines():
    fig = plt.figure(figsize=(mm(180), mm(150)))
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1.05], hspace=0.62, figure=fig)

    # a: PRISM
    ax = fig.add_subplot(gs[0])
    res = pd.read_csv(DM / "prism_pemt_sensitivity_hnscc.tsv", sep="\t")
    ax.scatter(res["spearman_rho"], -np.log10(res["p"]), s=5, color=PALETTE["grey"], alpha=0.5, linewidths=0, rasterized=True)
    hit = "proximity_hit_canonical_pEMT:drugbank"
    if hit in res:
        h = res[res[hit]]
        ax.scatter(h["spearman_rho"], -np.log10(h["p"]), s=12, color=PALETTE["vermilion"], linewidths=0, zorder=3)
    ax.axvline(0, color=PALETTE["lightgrey"], linewidth=0.6)
    ax.set_xlabel("Spearman ρ, pEMT specificity vs PRISM area under the curve")
    ax.set_ylabel("−log10 p")
    ax.text(0.01, 0.96, f"{len(res):,} compounds, 23 HNSCC models, 0 at FDR ≤ 0.1\nnegative ρ: more sensitive in pEMT-high models",
            transform=ax.transAxes, va="top", fontsize=6, color=PALETTE["grey"])
    label(ax, "a", dx=-0.07)

    # b: CRISPR
    ax = fig.add_subplot(gs[1])
    cr = pd.read_csv(DM / "crispr_pemt_dependency_hnscc.tsv", sep="\t")
    ax.scatter(cr["spearman_rho"], -np.log10(cr["p"]), s=3, color=PALETTE["grey"], alpha=0.4, linewidths=0, rasterized=True)
    for col, colour, lab in (("in_seed:pEMT_axis", PALETTE["vermilion"], "M13 module genes"),
                             ("in_seed:canonical_pEMT", PALETTE["blue"], "M06 module genes")):
        if col in cr:
            s = cr[cr[col]]
            ax.scatter(s["spearman_rho"], -np.log10(s["p"]), s=9, color=colour, linewidths=0, zorder=3, label=f"{lab} (n = {len(s)})")
    ax.axvline(0, color=PALETTE["lightgrey"], linewidth=0.6)
    ax.set_xlabel("Spearman ρ, pEMT specificity vs Chronos gene effect")
    ax.set_ylabel("−log10 p")
    ax.legend(loc="upper right", fontsize=6, frameon=False)
    ax.text(0.01, 0.96, f"{len(cr):,} genes, 63 HNSCC models\nnegative ρ: stronger dependency in pEMT-high models",
            transform=ax.transAxes, va="top", fontsize=6, color=PALETTE["grey"])
    label(ax, "b", dx=-0.07)

    # c: mechanism-of-action classes against the permutation null
    ax = fig.add_subplot(gs[2])
    mo = pd.read_csv(DM / "prism_moa_permutation_test.tsv", sep="\t")
    focus = ["EGFR inhibitor", "MEK inhibitor", "RAF inhibitor", "src inhibitor", "AKT inhibitor",
             "PI3K inhibitor", "mTOR inhibitor", "HDAC inhibitor"]
    width = 0.40
    from matplotlib.lines import Line2D
    for k, (cohort, marker, cl) in enumerate((("hnscc", "o", "HNSCC models"),
                                              ("pan_squamous", "^", "Pan-squamous models"))):
        tt = mo[mo["cohort"] == cohort].set_index("class")
        present = [f for f in focus if f in tt.index]
        y = np.arange(len(present))[::-1] + (0.5 - k) * width
        for yi, f in zip(y, present):
            r = tt.loc[f]
            lo, hi = r["null_mean"] - 1.96 * r["null_sd"], r["null_mean"] + 1.96 * r["null_sd"]
            ax.plot([lo, hi], [yi, yi], color=PALETTE["lightgrey"], linewidth=2.6, solid_capstyle="butt", zorder=1)
            sig = r["perm_p"] <= 0.05
            ax.scatter(r["mean_rho"], yi, s=20, marker=marker, zorder=3,
                       facecolor=PALETTE["vermilion"] if sig else "white",
                       edgecolor=PALETTE["vermilion"] if sig else PALETTE["grey"], linewidths=0.8)
    present = [f for f in focus if f in mo[mo["cohort"] == "hnscc"]["class"].values]
    ax.set_yticks(np.arange(len(present))[::-1])
    ax.set_yticklabels([f.replace(" inhibitor", "") for f in present], fontsize=6.5)
    ax.axvline(0, color=PALETTE["black"], linewidth=0.6)
    ax.set_xlabel("Mean Spearman ρ for the class, against the permutation null")
    ax.set_xlim(-0.45, 0.45)
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=PALETTE["grey"], label="HNSCC models"),
                       Line2D([], [], marker="^", linestyle="", markerfacecolor="white", markeredgecolor=PALETTE["grey"], label="Pan-squamous models"),
                       Line2D([], [], marker="o", linestyle="", markerfacecolor=PALETTE["vermilion"], markeredgecolor=PALETTE["vermilion"], label="permutation p ≤ 0.05")],
              loc="lower right", fontsize=5.8, frameon=False, handletextpad=0.4)
    ax.text(0.01, 1.02, "grey bar: central 95% of the permutation null, 5,000 permutations of the phenotype across cell lines",
            transform=ax.transAxes, fontsize=5.8, color=PALETTE["grey"])
    label(ax, "c", dx=-0.07, dy=1.08)
    fig.tight_layout()
    save_figure(fig, FIG, "Supplementary_Figure_15")


if __name__ == "__main__":
    figure_tcga_cox()
    figure_survival()
    figure_cell_lines()
    print("combined figures written")
