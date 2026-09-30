"""Pool the adjusted survival effect across the four bulk cohorts and quantify heterogeneity.

Each cohort carries an adjusted hazard ratio per SD for every score: TCGA-HNSC, GSE41613
(HPV-negative oral cavity), GSE65858 (mixed site) and CPTAC-3 (HPV-negative, surgical). log(HR)
per SD is pooled by inverse-variance weighting under a DerSimonian-Laird random-effects model, and
Cochran's Q, I-squared and tau-squared are reported. A random-effects model is used because the
cohorts differ by platform, treatment and case mix. The comparator EMT scores are pooled the same
way, in the same four cohorts.

Per-cohort estimates come from survival_sensitivity_per_cohort.tsv (written by
14_survival_sensitivity.py) when it exists, with standard errors taken from the Cox fits.
Otherwise they are read from the subgroup and external Cox tables, and the standard error is
back-calculated from the 95% CI.

Inputs:  results/tcga_projection/survival_sensitivity_per_cohort.tsv (preferred), or
         results/tcga_projection/subgroup_robustness_tcga.tsv and
         results/tcga_projection/external/external_cox_summary.tsv
Outputs: results/tcga_projection/
           survival_meta_analysis.tsv        pooled HR per SD, 95% CI, Q, I2, tau2, per score
           survival_meta_inputs.tsv          the per-cohort estimates that went in
         results/figures/panel_survival_meta (forest plot, per cohort and pooled, for five scores)
Usage:   python src/04_tcga_projection/13_survival_meta_analysis.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pemt import load_config, project_root  # noqa: E402
from pemt.figstyle import apply_style, save_figure, PALETTE, mm  # noqa: E402

cfg = load_config()
ROOT = project_root()
OUT = ROOT / cfg["paths"]["results_dir"] / "tcga_projection"
EXT = OUT / "external"
FIG = ROOT / "results" / "figures"
apply_style()
import matplotlib.pyplot as plt  # noqa: E402

SCORES = ["pEMT_specificity", "P_pEMT_high", "GS76", "KS", "hallmark_EMT", "puram_pemt", "MLR_mu"]
LABELS = {"pEMT_specificity": "pEMT specificity", "P_pEMT_high": "P(pEMT-high)", "GS76": "76GS", "KS": "KS (Tan)",
          "hallmark_EMT": "Hallmark EMT", "puram_pemt": "Puram pEMT", "MLR_mu": "MLR"}
COHORT_LABEL = {"TCGA-HNSC": "TCGA-HNSC", "GSE41613": "GSE41613", "GSE65858": "GSE65858", "CPTAC_HNSCC": "CPTAC-3"}
ORDER = ["TCGA-HNSC", "GSE41613", "CPTAC_HNSCC", "GSE65858"]


def se_from_ci(hr: float, lo: float, up: float) -> float:
    """Standard error of log(HR) from a 95% confidence interval."""
    return (np.log(up) - np.log(lo)) / (2 * 1.959964)


def dersimonian_laird(yi: np.ndarray, vi: np.ndarray) -> dict:
    wi = 1.0 / vi
    y_fixed = float((wi * yi).sum() / wi.sum())
    Q = float((wi * (yi - y_fixed) ** 2).sum())
    k = len(yi)
    C = wi.sum() - (wi ** 2).sum() / wi.sum()
    tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0
    wi_r = 1.0 / (vi + tau2)
    y_r = float((wi_r * yi).sum() / wi_r.sum())
    se_r = float(np.sqrt(1.0 / wi_r.sum()))
    I2 = max(0.0, (Q - (k - 1)) / Q * 100) if Q > 0 else 0.0
    return {"pooled_logHR": y_r, "se": se_r, "HR": float(np.exp(y_r)),
            "CI_low": float(np.exp(y_r - 1.959964 * se_r)), "CI_up": float(np.exp(y_r + 1.959964 * se_r)),
            "p_value": float(2 * norm.sf(abs(y_r / se_r))), "Q": Q, "df": k - 1,
            "p_heterogeneity": float(chi2.sf(Q, k - 1)) if k > 1 else np.nan, "I2_percent": I2, "tau2": tau2, "k_cohorts": k}


def collect() -> pd.DataFrame:
    """Per-cohort adjusted log(HR) and its variance.

    Uses the Cox-fit standard errors written by 14_survival_sensitivity.py when that file exists.
    Otherwise the standard error is back-calculated from the 95% CI, which is rounded to three
    decimal places and so adds some error to tau-squared and I-squared.
    """
    direct = OUT / "survival_sensitivity_per_cohort.tsv"
    if direct.exists():
        d = pd.read_csv(direct, sep="\t")
        d = d[(d["model"] == "adjusted") & (d["score"].isin(SCORES))].copy()
        d["cohort"] = d["cohort"].replace({"TCGA-HNSC": "TCGA-HNSC"})
        d["CI_low"] = np.exp(d["log_HR"] - 1.959964 * d["se"])
        d["CI_up"] = np.exp(d["log_HR"] + 1.959964 * d["se"])
        return d[["cohort", "score", "HR", "CI_low", "CI_up", "n", "events", "log_HR", "se", "var"]]
    rows = []
    tcga = pd.read_csv(OUT / "subgroup_robustness_tcga.tsv", sep="\t")
    t = tcga[tcga["subgroup"] == "all sites"]
    for r in t.itertuples():
        if r.score in SCORES and np.isfinite(getattr(r, "HR_per_SD", np.nan)):
            rows.append({"cohort": "TCGA-HNSC", "score": r.score, "HR": r.HR_per_SD, "CI_low": r.CI_low,
                         "CI_up": r.CI_up, "n": r.n, "events": r.events})
    ext = pd.read_csv(EXT / "external_cox_summary.tsv", sep="\t")
    for r in ext.itertuples():
        if r.score in SCORES and np.isfinite(r.HR_per_SD_adjusted):
            rows.append({"cohort": r.cohort, "score": r.score, "HR": r.HR_per_SD_adjusted,
                         "CI_low": r.CI_low_adj, "CI_up": r.CI_up_adj, "n": r.n_adj, "events": r.events_adj})
    d = pd.DataFrame(rows)
    d["log_HR"] = np.log(d["HR"])
    d["se"] = [se_from_ci(h, l, u) for h, l, u in zip(d["HR"], d["CI_low"], d["CI_up"])]
    d["var"] = d["se"] ** 2
    return d


def main() -> None:
    d = collect()
    d.to_csv(OUT / "survival_meta_inputs.tsv", sep="\t", index=False)
    print("Per-cohort adjusted HR per SD")
    piv = d.pivot(index="score", columns="cohort", values="HR").reindex(SCORES)
    print(piv.round(2).to_string())

    res = []
    for s in SCORES:
        sub = d[d["score"] == s].dropna(subset=["log_HR", "var"])
        if len(sub) < 2:
            continue
        m = dersimonian_laird(sub["log_HR"].values, sub["var"].values)
        m.update({"score": s, "label": LABELS[s], "total_n": int(sub["n"].sum()), "total_events": int(sub["events"].sum())})
        res.append(m)
    meta = pd.DataFrame(res)[["score", "label", "k_cohorts", "total_n", "total_events", "HR", "CI_low", "CI_up",
                              "p_value", "Q", "df", "p_heterogeneity", "I2_percent", "tau2"]]
    meta = meta.round({"HR": 3, "CI_low": 3, "CI_up": 3, "Q": 2, "I2_percent": 1, "tau2": 4})
    meta.to_csv(OUT / "survival_meta_analysis.tsv", sep="\t", index=False)
    print("\nRandom-effects pooled HR per SD across four cohorts")
    print(meta[["label", "k_cohorts", "total_n", "total_events", "HR", "CI_low", "CI_up", "p_value", "I2_percent", "p_heterogeneity"]].to_string(index=False))

    figure(d, meta)


def figure(d: pd.DataFrame, meta: pd.DataFrame):
    show = ["pEMT_specificity", "hallmark_EMT", "puram_pemt", "MLR_mu", "GS76"]
    fig, axes = plt.subplots(1, len(show), figsize=(mm(180), mm(72)), sharex=True)
    for ax, s in zip(axes, show):
        sub = d[d["score"] == s].set_index("cohort").reindex(ORDER).dropna(subset=["HR"])
        m = meta[meta["score"] == s].iloc[0]
        y = np.arange(len(sub))[::-1]
        for yi, (ch, r) in zip(y, sub.iterrows()):
            w = 1.0 / (r["var"] + float(m["tau2"]))
            ax.plot([r["CI_low"], r["CI_up"]], [yi, yi], color=PALETTE["grey"], linewidth=0.9, zorder=2)
            ax.scatter(r["HR"], yi, s=8 + 42 * w / (1.0 / (d[d["score"] == s]["var"].min() + float(m["tau2"]))),
                       color=PALETTE["grey"], marker="s", zorder=3, linewidths=0)
        ax.plot([m["CI_low"], m["CI_up"]], [-1.15, -1.15], color=PALETTE["vermilion"], linewidth=1.4, zorder=3)
        ax.scatter(m["HR"], -1.15, s=42, marker="D", color=PALETTE["vermilion"], zorder=4, linewidths=0)
        ax.axvline(1, color=PALETTE["black"], linewidth=0.6)
        ax.set_xscale("log")
        ax.set_xlim(0.45, 3.2)
        from matplotlib.ticker import NullLocator, NullFormatter, FixedLocator
        ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 3]))
        ax.set_xticklabels(["0.5", "1", "2", "3"], fontsize=6)
        ax.set_ylim(-2.0, len(sub) - 0.4)
        ax.set_yticks(list(y) + [-1.15])
        ax.set_yticklabels(([COHORT_LABEL.get(c, c) for c in sub.index] + ["Pooled"]) if s == show[0] else [""] * (len(sub) + 1), fontsize=6.2)
        ax.tick_params(axis="y", length=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        star = "*" if m["p_value"] < 0.05 else ""
        ax.set_title(f"{LABELS[s]}\n{m['HR']:.2f} ({m['CI_low']:.2f}-{m['CI_up']:.2f}){star}\nI² = {m['I2_percent']:.0f}%",
                     fontsize=6.4, color=PALETTE["grey"])
    fig.text(0.5, 0.035, "Adjusted hazard ratio per SD (log scale). Square area proportional to random-effects weight, "
                         "diamond = DerSimonian-Laird pooled estimate", ha="center", fontsize=6, color=PALETTE["grey"])
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    save_figure(fig, FIG, "panel_survival_meta")


if __name__ == "__main__":
    main()
