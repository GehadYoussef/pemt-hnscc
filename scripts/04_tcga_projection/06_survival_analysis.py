"""Cox proportional-hazards and Kaplan-Meier analysis of pEMT specificity in TCGA-HNSC.

Patched 9 Sep 2026. Changes versus the original:
  * anatomical site: base of tongue is oropharynx (was routed to oral cavity);
    reference category is explicitly oropharynx; the 3 'other' cases are excluded
    from adjusted models (they produced a degenerate estimate before)
  * HPV status added as a covariate (PanCanAtlas subtype, else p16 / ISH)
  * two adjusted models are reported with their own n and events:
      M2  stage + age + site + HPV                 (complete cases, most of the cohort)
      M3  M2 + pack-years                          (sensitivity; pack-years missing for ~40 %)
  * model comparison uses repeated 5-fold out-of-sample C-index, not in-sample
  * the forest plot states n and events for the model it shows

Outputs (results/tcga_projection/):
  cox_univariable.tsv, cox_multivariable_M2.tsv, cox_multivariable_M3.tsv,
  cindex_comparison.tsv, km_logrank_test.tsv, survival_cohort_summary.txt
Figures (manuscript/figures/):
  panel_km_pemt_tertile, Figure_5B_cox_forest_plot
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import load_config, project_root  # noqa: E402
from _lib.figstyle import apply_style, save_figure, PALETTE, TERTILE_COLOURS, mm  # noqa: E402
from _lib.survival import (primary_tumours, prepare_covariates, design_matrix, fit_cox,  # noqa: E402
                           cv_cindex, COVARIATE_LABELS)

cfg = load_config()
ROOT = project_root()
OUT = ROOT / cfg["paths"]["results_dir"] / "tcga_projection"
FIG = ROOT / "manuscript" / "figures"
SCORE = "pEMT_specificity"
apply_style()
import matplotlib.pyplot as plt  # noqa: E402


def km_by_tertile(df: pd.DataFrame):
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test
    sub = df.dropna(subset=["OS_time", "OS_event", SCORE]).copy()
    sub = sub[sub["OS_time"] > 0]
    sub["tertile"] = pd.qcut(sub[SCORE], q=3, labels=["Low", "Mid", "High"])
    res = multivariate_logrank_test(sub["OS_time"], sub["tertile"], sub["OS_event"])

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(mm(85), mm(82)), gridspec_kw={"height_ratios": [5, 1.1], "hspace": 0.08}, sharex=True)
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
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.0)
    ax.text(0.03, 0.05, f"Log-rank p = {res.p_value:.3f}", transform=ax.transAxes, fontsize=7)
    ax.legend(loc="upper right")
    ax.tick_params(labelbottom=False)
    for i, grp in enumerate(["Low", "Mid", "High"]):
        for x, n in zip(times, at_risk[grp]):
            axr.text(x, 2 - i, str(n), ha="center", va="center", fontsize=6, color=TERTILE_COLOURS[grp])
    axr.set_ylim(-0.6, 2.6)
    axr.set_yticks([2, 1, 0]); axr.set_yticklabels(["Low", "Mid", "High"], fontsize=6)
    axr.set_xticks(times)
    axr.set_xlabel("Years from diagnosis")
    axr.set_ylabel("At risk", fontsize=6.5)
    for sp in ("top", "right", "left"):
        axr.spines[sp].set_visible(False)
    axr.tick_params(axis="y", length=0, pad=16)
    save_figure(fig, FIG, "panel_km_pemt_tertile")

    deaths = sub.groupby("tertile", observed=True)["OS_event"].agg(["sum", "count"])
    return pd.DataFrame([{"test": "multivariate log-rank by pEMT specificity tertile", "n": len(sub),
                          "events": int(sub["OS_event"].sum()), "test_statistic": res.test_statistic,
                          "p_value": res.p_value,
                          "deaths_low": int(deaths.loc["Low", "sum"]), "deaths_mid": int(deaths.loc["Mid", "sum"]),
                          "deaths_high": int(deaths.loc["High", "sum"]),
                          "n_low": int(deaths.loc["Low", "count"]), "n_mid": int(deaths.loc["Mid", "count"]),
                          "n_high": int(deaths.loc["High", "count"])}])


def forest_plot(res: pd.DataFrame, stem: str, n: int, events: int):
    df = res.copy()
    df["label"] = df.index.map(lambda x: "pEMT specificity (per unit)" if x == SCORE else COVARIATE_LABELS.get(x, x))
    df = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(mm(120), mm(8 * len(df) + 14)))
    y = np.arange(len(df))
    for i, (_, row) in enumerate(df.iterrows()):
        colour = PALETTE["vermilion"] if row.name == SCORE else PALETTE["black"]
        ax.errorbar(row["HR"], y[i], xerr=[[row["HR"] - row["HR_low_95"]], [row["HR_up_95"] - row["HR"]]],
                    fmt="o", color=colour, ecolor=colour, capsize=2, markersize=4, elinewidth=0.9)
        ptxt = "p < 0.001" if row["p_value"] < 0.001 else f"p = {row['p_value']:.3f}"
        ax.text(1.02, (i + 0.5) / len(df), f"{row['HR']:.2f} ({row['HR_low_95']:.2f} to {row['HR_up_95']:.2f}), {ptxt}",
                transform=ax.transAxes, va="center", fontsize=7)
    ax.axvline(1.0, ls="--", color=PALETTE["grey"], linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.set_xscale("log")
    lo = max(0.2, float(df["HR_low_95"].min()) * 0.8)
    hi = min(20.0, float(df["HR_up_95"].max()) * 1.2)
    ax.set_xlim(lo, hi)
    ticks = [t for t in (0.25, 0.5, 1, 2, 4, 8) if lo <= t <= hi]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel(f"Hazard ratio for overall survival (95% CI); n = {n}, events = {events}")
    save_figure(fig, FIG, stem)


def main() -> None:
    t = pd.read_csv(OUT / "tcga_master_trait_table.tsv", sep="\t", index_col=0, low_memory=False)
    t = primary_tumours(t)
    df = prepare_covariates(t, [SCORE])
    lines = [f"Primary tumours: {len(df)}",
             f"With OS data: {int(df.dropna(subset=['OS_time', 'OS_event']).shape[0])}",
             f"HPV status available: {int(df['hpv_positive'].notna().sum())} (positive {int((df['hpv_positive'] == 1).sum())})",
             f"Site groups: {df['site_group'].value_counts(dropna=False).to_dict()}",
             f"Stage available: {int(df['stage_num'].notna().sum())}; pack-years available: {int(df['pack_years'].notna().sum())}"]

    # M1 univariable
    d1 = design_matrix(df, SCORE, [])
    _, uni = fit_cox(d1)
    uni.to_csv(OUT / "cox_univariable.tsv", sep="\t")
    lines.append(f"M1 univariable: n = {len(d1)}, events = {int(d1['OS_event'].sum())}, HR = {uni.loc[SCORE, 'HR']:.3f}, p = {uni.loc[SCORE, 'p_value']:.4f}")

    # M2 adjusted: stage + age + site + HPV
    cov2 = ["stage_num", "age", "site_group", "hpv_positive"]
    d2 = design_matrix(df, SCORE, cov2)
    _, m2 = fit_cox(d2)
    m2.to_csv(OUT / "cox_multivariable_M2.tsv", sep="\t")
    lines.append(f"M2 adjusted (stage, age, site, HPV): n = {len(d2)}, events = {int(d2['OS_event'].sum())}, "
                 f"pEMT HR = {m2.loc[SCORE, 'HR']:.3f} ({m2.loc[SCORE, 'HR_low_95']:.2f} to {m2.loc[SCORE, 'HR_up_95']:.2f}), p = {m2.loc[SCORE, 'p_value']:.4f}")

    # M3 sensitivity: + pack-years
    cov3 = cov2 + ["pack_years"]
    d3 = design_matrix(df, SCORE, cov3)
    _, m3 = fit_cox(d3)
    m3.to_csv(OUT / "cox_multivariable_M3.tsv", sep="\t")
    lines.append(f"M3 sensitivity (+ pack-years): n = {len(d3)}, events = {int(d3['OS_event'].sum())}, "
                 f"pEMT HR = {m3.loc[SCORE, 'HR']:.3f}, p = {m3.loc[SCORE, 'p_value']:.4f}")

    # out-of-sample C-index on the M2 rows
    feats_base = [c for c in d2.columns if c not in ("OS_time", "OS_event", SCORE)]
    cv = cv_cindex(d2, {"clinical (stage, age, site, HPV)": feats_base,
                        "clinical + pEMT specificity": feats_base + [SCORE],
                        "pEMT specificity alone": [SCORE]})
    cv.to_csv(OUT / "cindex_comparison.tsv", sep="\t", index=False)
    lines += [f"CV C-index {r['model']}: {r['cv_c_index_mean']:.3f} (SD {r['cv_c_index_sd']:.3f})" for _, r in cv.iterrows()]

    km = km_by_tertile(df)
    km.to_csv(OUT / "km_logrank_test.tsv", sep="\t", index=False)
    lines.append(f"KM tertiles: n = {int(km['n'][0])}, log-rank p = {km['p_value'][0]:.4f}, deaths low/mid/high = "
                 f"{int(km['deaths_low'][0])}/{int(km['deaths_mid'][0])}/{int(km['deaths_high'][0])}")

    forest_plot(m2, "Figure_5B_cox_forest_plot", len(d2), int(d2["OS_event"].sum()))
    (OUT / "survival_cohort_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
