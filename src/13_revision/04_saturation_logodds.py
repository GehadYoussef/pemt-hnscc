"""Saturation of pEMT specificity at plus or minus 1, and a log-odds summary of the same projection.

pEMT specificity is a difference of two probabilities and is bounded at plus or minus 1, so tumours
accumulate near the bounds. This stage re-summarises the per-sample class probabilities already written
by 04_tcga_projection/03, 04_tcga_projection/09, 04_tcga_projection/12 and 10_cetuximab/01 on the
log-odds scale and repeats the main analyses. The classifier is not refitted and no tumour is reprojected.

Summaries (every probability floored at 1e-6, which never binds in these cohorts)
  pEMT_specificity     P(pEMT-high) - max(P(epithelial-like), P(fibroblast))         bounded in [-1, 1]
  logit_specificity    log P(pEMT-high) - log max(P(epithelial-like), P(fibroblast))  unbounded
  logit_P_pEMT_high    log(P(pEMT-high) / (1 - P(pEMT-high)))                         unbounded, monotone in P

Analyses
  1. Saturation. In the TCGA-HNSC primaries, GSE41613, GSE65858, CPTAC-3 and GSE65021: the share of
     tumours with pEMT specificity below -0.9 and above 0.9 (the outer 5% of its range at each end), the
     share of tumours in the outer 5% of the observed range of each summary, and how much spread each
     summary keeps inside the saturated tails, with histograms.
  2. GSE65021 (platinum plus cetuximab): Mann-Whitney p, rank-biserial correlation, bootstrap AUC (the
     bootstrap of 10_cetuximab/01, same seed) and Firth odds ratio per SD, unadjusted and adjusted for the
     seven covariates of 10_cetuximab/04.
  3. Spearman correlation of each summary with pEMT specificity and with the malignant core in every
     cohort.
  4. Overall survival: univariable and adjusted Cox models per SD in the four survival cohorts (the
     models of 04_tcga_projection/14 behind Tables 2 and 3), pooled by DerSimonian-Laird with the
     Hartung-Knapp interval of 13_revision/01, with leave-one-cohort-out pooling.
  5. Xenograft panels GSE84713 and GSE183881: the association and fixed-effect pooling of
     10_cetuximab/09, from the scores that stage wrote (same-patient pairs of GSE84713 averaged).
  Every pEMT specificity estimate is recomputed through the same code path and compared with the value
  in the corresponding results table (reproduction_check.tsv).

Inputs:  results/tcga_projection/tcga_master_trait_table.tsv, emt_score_panel_scores.tsv and external/
         results/cetuximab_cohort/GSE65021_scores.tsv, GSE65021_pfs_association.tsv, GSE65021_firth_single.tsv
         results/arm_scores/<cohort>_arm_scores.tsv, results/pdx_cetuximab/*.tsv
         results/survival_meta_hk/survival_meta_hk.tsv (13_revision/01)
Outputs: results/specificity_saturation/*.tsv, summary.txt, Figure_specificity_saturation (.svg, .png)
Usage:   python src/13_revision/04_saturation_logodds.py
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm, spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from pemt import firth as _firth  # noqa: E402
from pemt.figstyle import apply_style, save_figure, PALETTE, mm  # noqa: E402
from pemt.survival import primary_tumours  # noqa: E402

RES = ROOT / "results"
TP = RES / "tcga_projection"
OUT = RES / "specificity_saturation"

EPS = 1e-6
TAIL = 0.9
P_COLS = ["P_epithelial_like", "P_fibroblast_stromal_like", "P_pEMT_high"]
SUMMARIES = ["pEMT_specificity", "logit_specificity", "logit_P_pEMT_high"]
LABELS = {"pEMT_specificity": "pEMT specificity", "logit_specificity": "Logit specificity",
          "logit_P_pEMT_high": "logit P(pEMT-high)", "P_pEMT_high": "P(pEMT-high)",
          "malignant_arm_core": "Malignant core"}
COHORTS = ["TCGA-HNSC", "GSE41613", "GSE65858", "CPTAC_HNSCC", "GSE65021"]
COHORT_LABELS = {"TCGA-HNSC": "TCGA-HNSC", "GSE41613": "GSE41613", "GSE65858": "GSE65858",
                 "CPTAC_HNSCC": "CPTAC-3", "GSE65021": "GSE65021"}
LOG = []


def say(line: str = "") -> None:
    print(line)
    LOG.append(line)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, [sys.argv[0]]
    spec.loader.exec_module(mod)
    sys.argv = argv
    return mod


def add_logits(df: pd.DataFrame) -> pd.DataFrame:
    """The two log-odds summaries from the stored class probabilities."""
    p = df[P_COLS].clip(lower=EPS, upper=1 - EPS)
    other = p[["P_epithelial_like", "P_fibroblast_stromal_like"]].max(axis=1)
    df = df.copy()
    df["logit_specificity"] = np.log(p["P_pEMT_high"]) - np.log(other)
    df["logit_P_pEMT_high"] = np.log(p["P_pEMT_high"] / (1 - p["P_pEMT_high"]))
    # the stored pEMT specificity must equal the definition applied to the stored probabilities
    recomputed = df["P_pEMT_high"] - df[["P_epithelial_like", "P_fibroblast_stromal_like"]].max(axis=1)
    assert np.allclose(recomputed, df["pEMT_specificity"], atol=1e-9), "stored specificity disagrees"
    return df


def score_tables() -> dict[str, pd.DataFrame]:
    """Per-sample probabilities, summaries and the malignant core for the five patient cohorts."""
    arms = RES / "arm_scores"
    traits = primary_tumours(pd.read_csv(TP / "tcga_master_trait_table.tsv", sep="\t", index_col=0,
                                         low_memory=False))
    panel = pd.read_csv(TP / "emt_score_panel_scores.tsv", sep="\t", index_col=0)
    out = {"TCGA-HNSC": traits.loc[panel.index.intersection(traits.index), P_COLS + ["pEMT_specificity"]]}
    for cid in ("GSE41613", "GSE65858", "CPTAC_HNSCC"):
        out[cid] = pd.read_csv(TP / "external" / f"{cid}_scores.tsv", sep="\t", index_col=0)
    out["GSE65021"] = pd.read_csv(RES / "cetuximab_cohort" / "GSE65021_scores.tsv", sep="\t", index_col=0)
    for cid in COHORTS:
        a = pd.read_csv(arms / f"{cid}_arm_scores.tsv", sep="\t", index_col=0)[["malignant_arm_core"]]
        out[cid] = add_logits(out[cid]).join(a, how="left")
    return out


# 1. saturation
def saturation(tabs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for cid in COHORTS:
        d = tabs[cid]
        spec = d["pEMT_specificity"]
        low, high = spec < -TAIL, spec > TAIL
        for s in SUMMARIES:
            x = d[s]
            lo, hi = x.min(), x.max()
            w = 0.05 * (hi - lo)
            rows.append({
                "cohort": COHORT_LABELS[cid], "summary": LABELS[s], "n": len(x),
                "min": lo, "max": hi, "median": x.median(), "sd": x.std(),
                # fixed tails of pEMT specificity (outer 5% of its theoretical range at each end)
                "share_spec_below_-0.9": float(low.mean()), "share_spec_above_0.9": float(high.mean()),
                # outer 5% of the observed range of this summary, at each end
                "share_lowest_5pct_of_range": float((x <= lo + w).mean()),
                "share_highest_5pct_of_range": float((x >= hi - w).mean()),
                # spread kept inside the saturated tails, relative to the whole cohort
                "sd_ratio_in_low_tail": float(x[low].std() / x.std()) if low.sum() > 2 else np.nan,
                "sd_ratio_in_high_tail": float(x[high].std() / x.std()) if high.sum() > 2 else np.nan,
                "range_share_of_low_tail": float((x[low].max() - x[low].min()) / (hi - lo)) if low.sum() > 1 else np.nan,
                "rho_with_malignant_core_in_low_tail": spearmanr(x[low], d.loc[low, "malignant_arm_core"],
                                                                 nan_policy="omit")[0] if low.sum() > 4 else np.nan,
            })
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "saturation_tails.tsv", sep="\t", index=False, float_format="%.4g")
    say("Saturation, share of tumours in each tail:")
    show = res[["cohort", "summary", "n", "share_spec_below_-0.9", "share_spec_above_0.9",
                "share_lowest_5pct_of_range", "share_highest_5pct_of_range", "sd_ratio_in_low_tail",
                "range_share_of_low_tail", "rho_with_malignant_core_in_low_tail"]]
    say(show.round(3).to_string(index=False))
    return res


def figure(tabs: dict[str, pd.DataFrame]) -> None:
    """Rows: pEMT specificity, logit specificity, logit P(pEMT-high). Columns: cohorts. Bars are stacked
    by the pEMT specificity tail each tumour falls in, so the logit rows show where the saturated
    tumours go once the bound is removed."""
    apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    cols = {"low": PALETTE["blue"], "mid": PALETTE["lightgrey"], "high": PALETTE["vermilion"]}
    fig, axes = plt.subplots(3, len(COHORTS), figsize=(mm(180), mm(118)), squeeze=False)
    xlabs = {"pEMT_specificity": "pEMT specificity", "logit_specificity": "Logit specificity",
             "logit_P_pEMT_high": "logit P(pEMT-high)"}
    lims = {s: (min(tabs[c][s].min() for c in COHORTS), max(tabs[c][s].max() for c in COHORTS))
            for s in SUMMARIES}
    for j, cid in enumerate(COHORTS):
        d = tabs[cid]
        spec = d["pEMT_specificity"]
        grp = np.where(spec < -TAIL, "low", np.where(spec > TAIL, "high", "mid"))
        for i, s in enumerate(SUMMARIES):
            ax = axes[i, j]
            if s == "pEMT_specificity":
                bins = np.linspace(-1, 1, 41)
                ax.axvspan(-1, -TAIL, color=PALETTE["lightgrey"], alpha=0.35, linewidth=0, zorder=0)
                ax.axvspan(TAIL, 1, color=PALETTE["lightgrey"], alpha=0.35, linewidth=0, zorder=0)
                ax.set_xlim(-1.05, 1.05)
                ax.set_xticks([-1, 0, 1])
            else:
                lo, hi = lims[s]
                bins = np.linspace(np.floor(lo), np.ceil(hi), 41)
                ax.set_xlim(np.floor(lo), np.ceil(hi))
            ax.hist([d.loc[grp == g, s] for g in ("low", "mid", "high")], bins=bins, stacked=True,
                    color=[cols[g] for g in ("low", "mid", "high")], edgecolor="white", linewidth=0.25,
                    zorder=2)
            if s == "pEMT_specificity":
                ax.text(0.5, 0.97, f"below -0.9: {100 * (spec < -TAIL).mean():.0f}%\nabove 0.9: {100 * (spec > TAIL).mean():.0f}%",
                        transform=ax.transAxes, ha="center", va="top", fontsize=6, linespacing=1.3)
                ax.set_title(f"{COHORT_LABELS[cid]}, n = {len(d)}", fontsize=7, color=PALETTE["grey"],
                             loc="center", pad=8)
            else:
                rho = spearmanr(d[s], spec)[0]
                ax.text(0.03, 0.97, f"rho = {rho:.2f}", transform=ax.transAxes, ha="left", va="top",
                        fontsize=6)
            ax.set_xlabel(xlabs[s], fontsize=7)
            if j == 0:
                ax.set_ylabel("Tumours")
                ax.text(-0.42, 1.12, "abc"[i], transform=ax.transAxes, fontweight="bold", fontsize=8,
                        ha="left", va="bottom")
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
            ax.tick_params(labelsize=6)
    fig.legend(handles=[Patch(color=cols["low"], label="pEMT specificity below -0.9"),
                        Patch(color=cols["mid"], label="-0.9 to 0.9"),
                        Patch(color=cols["high"], label="above 0.9")],
               loc="lower center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.95, bottom=0.12, hspace=0.75, wspace=0.38)
    save_figure(fig, OUT, "Figure_specificity_saturation")


# 2. GSE65021
def cetuximab(d: pd.DataFrame, cx, f04) -> pd.DataFrame:
    rows = []
    y = d["long_pfs"].to_numpy(int)
    for s in SUMMARIES + ["P_pEMT_high"]:
        a, b = d.loc[y == 1, s], d.loc[y == 0, s]
        u = mannwhitneyu(a, b, alternative="two-sided")
        auc, lo, hi = cx.bootstrap_auc(y, d[s].to_numpy())
        rec = {"summary": LABELS[s], "median_long": a.median(), "median_short": b.median(),
               "rank_biserial": 2 * u.statistic / (len(a) * len(b)) - 1, "p_two_sided": u.pvalue,
               "AUC": auc, "AUC_low_95": lo, "AUC_up_95": hi}
        for model, covs in (("unadjusted", []), ("adjusted", f04.COVS)):
            r, n = f04.fit(d, [s], covs)
            rec.update({f"OR_{model}": r[s][0], f"OR_low_{model}": r[s][1], f"OR_up_{model}": r[s][2],
                        f"p_{model}": r[s][3], f"n_{model}": n})
        rows.append(rec)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "gse65021_association.tsv", sep="\t", index=False, float_format="%.4g")
    say("\nGSE65021, long against short PFS on platinum plus cetuximab (Firth OR per SD):")
    say(res[["summary", "median_long", "median_short", "p_two_sided", "AUC", "AUC_low_95", "AUC_up_95",
             "OR_unadjusted", "OR_low_unadjusted", "OR_up_unadjusted", "p_unadjusted", "OR_adjusted",
             "p_adjusted"]].round(3).to_string(index=False))
    return res


# 3. correlations
def correlations(tabs: dict[str, pd.DataFrame], pdx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for cid, d in list(tabs.items()) + list(pdx.items()):
        for s in SUMMARIES:
            ok = d["malignant_arm_core"].notna()
            rows.append({"cohort": COHORT_LABELS.get(cid, cid), "summary": LABELS[s], "n": int(ok.sum()),
                         "rho_with_pEMT_specificity": spearmanr(d[s], d["pEMT_specificity"])[0],
                         "rho_with_malignant_core": spearmanr(d.loc[ok, s], d.loc[ok, "malignant_arm_core"])[0],
                         "rho_with_P_pEMT_high": spearmanr(d[s], d["P_pEMT_high"])[0]})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "spearman_by_cohort.tsv", sep="\t", index=False, float_format="%.4g")
    say("\nSpearman correlations by cohort:")
    say(res.pivot(index="cohort", columns="summary", values=["rho_with_pEMT_specificity",
                                                             "rho_with_malignant_core"]).round(3).to_string())
    return res


# 4. survival
def survival(tabs: dict[str, pd.DataFrame], s14, hk) -> tuple[pd.DataFrame, pd.DataFrame]:
    coh = s14.cohorts()
    rows = []
    for cid in ("TCGA-HNSC", "GSE41613", "GSE65858", "CPTAC_HNSCC"):
        df, covs = coh[cid]
        df = df.drop(columns=[c for c in SUMMARIES[1:] if c in df]).join(tabs[cid][SUMMARIES[1:]], how="left")
        if cid == "TCGA-HNSC":
            common = df.index.intersection(tabs[cid].index)
            assert np.allclose(df.loc[common, "pEMT_specificity"], tabs[cid].loc[common, "pEMT_specificity"])
        for s in SUMMARIES:
            for model in ("univariable", "adjusted"):
                r = s14.cox_direct(df, s, covs if model == "adjusted" else [])
                if r:
                    rows.append({"summary": LABELS[s], "score": s, "model": model, "cohort": COHORT_LABELS[cid],
                                 "adjusted_for": ", ".join(covs) if model == "adjusted" else "", **r})
    per = pd.DataFrame(rows)
    per.to_csv(OUT / "survival_per_cohort.tsv", sep="\t", index=False, float_format="%.5g")
    pooled = []
    for (s, model), g in per.groupby(["score", "model"], sort=False):
        for left_out in ["none"] + list(g["cohort"]):
            h = g if left_out == "none" else g[g["cohort"] != left_out]
            pooled.append({"summary": LABELS[s], "model": model, "cohort_left_out": left_out,
                           "n_total": int(h["n"].sum()), "events_total": int(h["events"].sum()),
                           **hk.pool(h["log_HR"].to_numpy(), h["se"].to_numpy())})
    pooled = pd.DataFrame(pooled)
    pooled.to_csv(OUT / "survival_pooled_dl_hk.tsv", sep="\t", index=False, float_format="%.4g")
    say("\nOverall survival, HR per SD by cohort:")
    say(per.pivot_table(index=["model", "summary"], columns="cohort", values="HR").round(3).to_string())
    say("\nPooled HR per SD, DerSimonian-Laird with Hartung-Knapp interval (all four cohorts):")
    say(pooled[pooled["cohort_left_out"] == "none"][["summary", "model", "n_total", "events_total", "HR",
                                                     "DL_low", "DL_up", "DL_p", "HK_low", "HK_up", "HK_p",
                                                     "I2_percent"]].round(3).to_string(index=False))
    loo = pooled[(pooled["model"] == "adjusted") & (pooled["cohort_left_out"] != "none")]
    say("\nAdjusted, leave one cohort out:")
    say(loo.pivot(index="cohort_left_out", columns="summary", values="HR").round(3).to_string())
    return per, pooled


# 5. xenografts
def pdx_tables() -> dict[str, pd.DataFrame]:
    d = RES / "pdx_cetuximab"
    s84 = add_logits(pd.read_csv(d / "GSE84713_scores.tsv", sep="\t", index_col=0))
    num = SUMMARIES + ["P_pEMT_high", "malignant_arm_core"]
    per_patient = s84.groupby("patient").agg({**{c: "mean" for c in num}, "response": "max"})
    s183 = add_logits(pd.read_csv(d / "GSE183881_scores.tsv", sep="\t", index_col=0))
    return {"GSE84713": per_patient, "GSE183881": s183}


def xenografts(pdx: dict[str, pd.DataFrame], cx) -> tuple[pd.DataFrame, pd.DataFrame]:
    def z(v):
        return (v - v.mean()) / v.std()

    rows = []
    for cid, df in pdx.items():
        y = df["response"].to_numpy(int)
        for s in SUMMARIES:
            a, b = df.loc[y == 1, s], df.loc[y == 0, s]
            auc, lo, hi = cx.bootstrap_auc(y, df[s].to_numpy())
            f = _firth.firth(y, np.column_stack([np.ones(len(df)), z(df[s])]), ["c", s])[s]
            rows.append({"cohort": cid, "summary": LABELS[s], "n_responders": int(y.sum()),
                         "n_non_responders": int((1 - y).sum()), "AUC": auc, "AUC_low": lo, "AUC_up": hi,
                         "p_two_sided": mannwhitneyu(a, b).pvalue, "OR_firth": f[0], "OR_low": f[1],
                         "OR_up": f[2], "p_firth": f[3]})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "pdx_association.tsv", sep="\t", index=False, float_format="%.4g")
    # fixed-effect pooling of the log Firth OR, standard error from the interval, as in 10_cetuximab/09
    pooled = []
    for s, d in res.groupby("summary", sort=False):
        b = np.log(d["OR_firth"])
        se = (np.log(d["OR_up"]) - np.log(d["OR_low"])) / (2 * 1.96)
        w = 1 / se ** 2
        m = float((w * b).sum() / w.sum())
        sd = float(np.sqrt(1 / w.sum()))
        pooled.append({"summary": s, "pooled_OR": np.exp(m), "low": np.exp(m - 1.96 * sd),
                       "up": np.exp(m + 1.96 * sd), "p": 2 * norm.sf(abs(m / sd)),
                       "n_models": int((d["n_responders"] + d["n_non_responders"]).sum())})
    pooled = pd.DataFrame(pooled)
    pooled.to_csv(OUT / "pdx_pooled.tsv", sep="\t", index=False, float_format="%.4g")
    say("\nXenograft panels:")
    say(res.round(3).to_string(index=False))
    say(pooled.round(3).to_string(index=False))
    return res, pooled


# reproduction of the pEMT specificity estimates in the results tables
def reproduction(cet: pd.DataFrame, pooled: pd.DataFrame, pdx_pooled: pd.DataFrame) -> None:
    hk = pd.read_csv(RES / "survival_meta_hk" / "survival_meta_hk.tsv", sep="\t")
    hk = hk[(hk["score"] == "pEMT_specificity") & (hk["cohort_left_out"] == "none")].set_index("model")
    ass = pd.read_csv(RES / "cetuximab_cohort" / "GSE65021_pfs_association.tsv", sep="\t")
    ass = ass.set_index("score").loc["pEMT_specificity"]
    fs = pd.read_csv(RES / "cetuximab_cohort" / "GSE65021_firth_single.tsv", sep="\t")
    fs = fs[fs["score"] == "pEMT specificity"].set_index("model")
    p09 = pd.read_csv(RES / "pdx_cetuximab" / "pdx_cetuximab_pooled.tsv", sep="\t")
    p09 = p09.set_index("score").loc["pEMT_specificity"]
    c = cet.set_index("summary").loc["pEMT specificity"]
    pv = pooled[(pooled["summary"] == "pEMT specificity") & (pooled["cohort_left_out"] == "none")].set_index("model")
    q = pdx_pooled.set_index("summary").loc["pEMT specificity"]
    rows = [("GSE65021 AUC", ass["AUC"], c["AUC"]), ("GSE65021 AUC lower", ass["AUC_low_95"], c["AUC_low_95"]),
            ("GSE65021 AUC upper", ass["AUC_up_95"], c["AUC_up_95"]),
            ("GSE65021 Mann-Whitney p", ass["p_two_sided"], c["p_two_sided"]),
            ("GSE65021 Firth OR unadjusted", fs.loc["unadjusted", "OR"], c["OR_unadjusted"]),
            ("GSE65021 Firth OR adjusted", fs.loc["adjusted", "OR"], c["OR_adjusted"]),
            ("Pooled adjusted HR", hk.loc["adjusted", "HR"], pv.loc["adjusted", "HR"]),
            ("Pooled adjusted HR, HK lower", hk.loc["adjusted", "HK_low"], pv.loc["adjusted", "HK_low"]),
            ("Pooled adjusted HR, HK upper", hk.loc["adjusted", "HK_up"], pv.loc["adjusted", "HK_up"]),
            ("Pooled univariable HR", hk.loc["univariable", "HR"], pv.loc["univariable", "HR"]),
            ("PDX pooled Firth OR", p09["pooled_OR"], q["pooled_OR"])]
    rep = pd.DataFrame(rows, columns=["estimate", "results_table", "this_script"])
    rep["abs_difference"] = (rep["results_table"] - rep["this_script"]).abs()
    rep.to_csv(OUT / "reproduction_check.tsv", sep="\t", index=False, float_format="%.5g")
    say("\nReproduction of the pEMT specificity estimates in the results tables:")
    say(rep.round(4).to_string(index=False))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cx = _load(SRC / "10_cetuximab" / "01_cetuximab_cohort.py", "cx")
    f04 = _load(SRC / "10_cetuximab" / "04_cetuximab_firth.py", "f04")
    hk = _load(SRC / "13_revision" / "01_hartung_knapp_meta.py", "hk")
    s14 = _load(SRC / "04_tcga_projection" / "14_survival_sensitivity.py", "s14")

    tabs = score_tables()
    say(f"Smallest class probability in any cohort: "
        f"{min(float(t[P_COLS].min().min()) for t in tabs.values()):.2e} (floor {EPS:g} never binds)")
    pd.concat({COHORT_LABELS[k]: v[P_COLS + SUMMARIES + ["malignant_arm_core"]] for k, v in tabs.items()},
              names=["cohort", "sample"]).to_csv(OUT / "per_sample_summaries.tsv", sep="\t", float_format="%.6g")

    saturation(tabs)
    figure(tabs)
    cet = cetuximab(tabs["GSE65021"], cx, f04)
    pdx = pdx_tables()
    correlations(tabs, pdx)
    _, pooled = survival(tabs, s14, hk)
    _, pdx_pooled = xenografts(pdx, cx)
    reproduction(cet, pooled, pdx_pooled)
    (OUT / "summary.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
