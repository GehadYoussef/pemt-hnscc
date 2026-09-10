"""Is the survival association driven by one anatomical site or one HPV stratum?

This exists because the two external cohorts disagree: GSE41613 (HPV-negative oral cavity, surgically
treated) replicates strongly while GSE65858 (mixed site, mixed treatment) is null. The obvious
hypothesis is that the axis is prognostic only in oral cavity or only in HPV-negative disease, which
would make the discrepancy a feature rather than a failure. This script tests that hypothesis directly
in TCGA and in both external cohorts, and it does not survive: in TCGA the hazard ratio is essentially
the same in oral cavity, in larynx/hypopharynx and overall, and in GSE65858 restricting to HPV-negative
or to oral cavity does not recover an effect. The discrepancy between the two external cohorts is
therefore real and unexplained by case mix, and the manuscript says so.

Outputs (results/tcga_projection/):
  subgroup_robustness_tcga.tsv       adjusted Cox HR per SD within each site and HPV stratum, every score
  subgroup_robustness_external.tsv   the same subgroup test inside GSE65858, GSE41613 and CPTAC-3
Figures:
  Figure_S7_subgroup_robustness      forest plot of the subgroup hazard ratios, TCGA and external
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import load_config, project_root  # noqa: E402
from _lib.survival import primary_tumours, prepare_covariates, design_matrix, fit_cox  # noqa: E402
from _lib.figstyle import apply_style, save_figure, PALETTE, mm  # noqa: E402

warnings.filterwarnings("ignore")
cfg = load_config()
ROOT = project_root()
OUT = ROOT / cfg["paths"]["results_dir"] / "tcga_projection"
EXT = OUT / "external"
FIG = ROOT / "manuscript" / "figures"
apply_style()
import matplotlib.pyplot as plt  # noqa: E402

SCORES = ["pEMT_specificity", "P_pEMT_high", "GS76", "KS", "hallmark_EMT", "puram_pemt", "MLR_mu"]
MIN_N, MIN_EVENTS = 50, 15


def cox(sub: pd.DataFrame, score: str, covs: list[str], label: str, cohort: str) -> dict:
    """Adjusted Cox within one subgroup. The score is standardised INSIDE the subgroup so that every
    row of the output is a hazard ratio per standard deviation and is comparable across subgroups and
    cohorts. Standardising outside would make the external-cohort rows hazard ratios per raw unit,
    which are not comparable with anything."""
    sub = sub.copy()
    s = pd.to_numeric(sub[score], errors="coerce")
    sd = s.std()
    sub[score] = (s - s.mean()) / (sd if sd and np.isfinite(sd) and sd > 0 else 1.0)
    covs = [c for c in covs if c in sub.columns and sub[c].notna().sum() > 0.8 * len(sub) and sub[c].nunique() > 1]
    try:
        d = design_matrix(sub, score, covs)
    except Exception:
        return {"cohort": cohort, "score": score, "subgroup": label, "n": len(sub), "events": np.nan}
    if len(d) < MIN_N or d["OS_event"].sum() < MIN_EVENTS:
        return {"cohort": cohort, "score": score, "subgroup": label, "n": len(d),
                "events": int(d["OS_event"].sum()), "note": "underpowered, not fitted"}
    try:
        _, r = fit_cox(d)
    except Exception as e:
        return {"cohort": cohort, "score": score, "subgroup": label, "n": len(d), "note": f"did not converge: {type(e).__name__}"}
    return {"cohort": cohort, "score": score, "subgroup": label, "n": int(r.loc[score, "n"]), "events": int(r.loc[score, "events"]),
            "HR_per_SD": round(float(r.loc[score, "HR"]), 3), "CI_low": round(float(r.loc[score, "HR_low_95"]), 3),
            "CI_up": round(float(r.loc[score, "HR_up_95"]), 3), "p_value": float(r.loc[score, "p_value"]),
            "adjusted_for": ", ".join(covs)}


def main() -> None:
    panel = pd.read_csv(OUT / "emt_score_panel_scores.tsv", sep="\t", index_col=0)
    traits = primary_tumours(pd.read_csv(OUT / "tcga_master_trait_table.tsv", sep="\t", index_col=0, low_memory=False))
    df = prepare_covariates(traits.join(panel.drop(columns=[c for c in ("pEMT_specificity", "P_pEMT_high") if c in panel])), list(panel.columns))
    full = ["stage_num", "age", "site_group", "hpv_positive"]
    strata = [("all sites", df, full),
              ("oral cavity", df[df["site_group"] == "oral_cavity"], ["stage_num", "age", "hpv_positive"]),
              ("larynx / hypopharynx", df[df["site_group"] == "larynx_hypopharynx"], ["stage_num", "age", "hpv_positive"]),
              ("oropharynx", df[df["site_group"] == "oropharynx"], ["stage_num", "age", "hpv_positive"]),
              ("HPV-negative", df[df["hpv_positive"] == 0], ["stage_num", "age", "site_group"]),
              ("HPV-positive", df[df["hpv_positive"] == 1], ["stage_num", "age", "site_group"]),
              ("HPV-negative oral cavity", df[(df["hpv_positive"] == 0) & (df["site_group"] == "oral_cavity")], ["stage_num", "age"])]
    print("TCGA-HNSC primaries, adjusted Cox, hazard ratio per SD")
    print("  site_group:", df["site_group"].value_counts().to_dict())
    rows = [cox(sub, s, covs, label, "TCGA-HNSC") for s in SCORES for label, sub, covs in strata]
    tcga = pd.DataFrame(rows)
    tcga.to_csv(OUT / "subgroup_robustness_tcga.tsv", sep="\t", index=False)
    show = tcga[tcga["score"].isin(["pEMT_specificity", "hallmark_EMT", "MLR_mu"])]
    print(show[["score", "subgroup", "n", "events", "HR_per_SD", "CI_low", "CI_up", "p_value"]].to_string(index=False))

    rows = []
    for cid in ("GSE65858", "GSE41613", "CPTAC_HNSCC"):
        f = EXT / f"{cid}_scores.tsv"
        if not f.exists():
            continue
        d = pd.read_csv(f, sep="\t", index_col=0)
        if "site_group" not in d:
            d["site_group"] = np.nan
        subs = [("all", d, ["stage_num", "age", "site_group", "hpv_positive"])]
        if "hpv_positive" in d and d["hpv_positive"].notna().any():
            subs.append(("HPV-negative", d[d["hpv_positive"] == 0], ["stage_num", "age", "site_group"]))
        if (d["site_group"] == "oral_cavity").any():
            oc = d[d["site_group"] == "oral_cavity"]
            subs.append(("oral cavity", oc, ["stage_num", "age", "hpv_positive"]))
            if "hpv_positive" in d:
                subs.append(("HPV-negative oral cavity", oc[oc["hpv_positive"] == 0], ["stage_num", "age"]))
        rows += [cox(sub, s, covs, label, cid) for s in SCORES for label, sub, covs in subs]
    ext = pd.DataFrame(rows)
    ext.to_csv(OUT / "subgroup_robustness_external.tsv", sep="\t", index=False)
    print("\nExternal cohorts, same subgroup test")
    show = ext[ext["score"].isin(["pEMT_specificity", "hallmark_EMT", "MLR_mu"])]
    print(show[["cohort", "score", "subgroup", "n", "events", "HR_per_SD", "CI_low", "CI_up", "p_value"]].to_string(index=False))
    figure(tcga, ext)


def figure(tcga: pd.DataFrame, ext: pd.DataFrame):
    """Forest plot: is the effect confined to a site or an HPV stratum? Underpowered strata are marked."""
    from matplotlib.ticker import NullLocator, NullFormatter, FixedLocator
    show = ["pEMT_specificity", "puram_pemt", "hallmark_EMT"]
    labels = {"pEMT_specificity": "pEMT specificity (tumour-intrinsic)", "puram_pemt": "Puram pEMT (stromal)",
              "hallmark_EMT": "Hallmark EMT"}
    rows = []
    for r in tcga.itertuples():
        rows.append(("TCGA-HNSC  " + r.subgroup, r.score, r))
    for r in ext.itertuples():
        rows.append(f"{r.cohort.replace('_HNSCC', '-3')}  {r.subgroup}", ) if False else rows.append(
            (f"{r.cohort.replace('_HNSCC', '-3')}  {r.subgroup}", r.score, r))
    order = [x[0] for x in rows if x[1] == "pEMT_specificity"]
    seen, uniq = set(), []
    for o in order:
        if o not in seen:
            seen.add(o); uniq.append(o)

    fig, axes = plt.subplots(1, len(show), figsize=(mm(180), mm(96)), sharey=True)
    for ax, s in zip(axes, show):
        d = {x[0]: x[2] for x in rows if x[1] == s}
        y = np.arange(len(uniq))[::-1]
        for yi, key in zip(y, uniq):
            r = d.get(key)
            if r is None or not np.isfinite(getattr(r, "HR_per_SD", np.nan)):
                ax.text(1.0, yi, "not fitted", fontsize=5, color=PALETTE["lightgrey"], ha="center", va="center")
                continue
            small = int(getattr(r, "events", 0) or 0) < 30
            col = PALETTE["lightgrey"] if small else PALETTE["vermilion"]
            ax.plot([r.CI_low, r.CI_up], [yi, yi], color=col, linewidth=0.9, zorder=2)
            ax.scatter(r.HR_per_SD, yi, s=14, color=col, marker="s", zorder=3, linewidths=0)
        ax.axvline(1, color=PALETTE["black"], linewidth=0.6)
        ax.set_xscale("log"); ax.set_xlim(0.4, 3.6)
        ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 3]))
        ax.set_xticklabels(["0.5", "1", "2", "3"], fontsize=6)
        ax.set_ylim(-0.7, len(uniq) - 0.3)
        ax.set_yticks(y); ax.set_yticklabels(uniq, fontsize=5.6)
        ax.tick_params(axis="y", length=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.set_title(labels[s], fontsize=7, color=PALETTE["grey"])
        ax.set_xlabel("Adjusted HR per SD", fontsize=6.8)
    fig.text(0.5, 0.02, "Grey: fewer than 30 events, underpowered. Each score is standardised within the subgroup, "
                        "so every estimate is per standard deviation of that subgroup.",
             ha="center", fontsize=6, color=PALETTE["grey"])
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_figure(fig, FIG, "Figure_S7_subgroup_robustness")


if __name__ == "__main__":
    main()
