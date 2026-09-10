"""Stress tests for the survival conclusion before it is stated in the manuscript.

The claim under test is that the stromal (Puram) programme is a consistent prognostic marker across
four cohorts while the tumour-intrinsic axis is not. That is a strong negative statement about our own
score and it should not rest on one modelling choice. This script attacks it from six directions:

  1. Standard errors taken directly from the Cox fit rather than back-calculated from a rounded
     confidence interval, in case the meta-analysis inherited rounding error.
  2. Univariable as well as adjusted pooling, because the four cohorts do not record the same
     covariates (GSE41613 has only stage and age) and the adjustment sets could drive the difference.
  3. Fixed-effect alongside random-effects pooling, since with four studies the DerSimonian-Laird
     tau-squared is itself poorly estimated.
  4. Leave-one-cohort-out pooling, to see whether either conclusion depends on a single cohort.
  5. Per-cohort significance, because a small pooled p-value with consistent direction is a different
     claim from replication, and the manuscript should not blur them.
  6. Treatment stratification in GSE65858, which is the one cohort with mixed treatment and the one
     that is null. It records treatment as mono or multi modality. If the axis is prognostic only in
     one stratum that is an interpretable finding rather than a failure to replicate.

Also tests follow-up truncation, since CPTAC-3 has a median follow-up of about two years and TCGA far
longer, so a late-acting effect would be attenuated in CPTAC by administrative censoring alone.

Outputs (results/tcga_projection/):
  survival_sensitivity_pooling.tsv     fixed vs random, univariable vs adjusted, per score
  survival_sensitivity_loo.tsv         leave-one-cohort-out pooled estimates
  survival_sensitivity_truncation.tsv  pooling after truncating every cohort at 36 and 60 months
  gse65858_treatment_strata.tsv        GSE65858 by treatment modality
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import load_config, project_root  # noqa: E402
from _lib.survival import primary_tumours, prepare_covariates, design_matrix, fit_cox  # noqa: E402

warnings.filterwarnings("ignore")
cfg = load_config()
ROOT = project_root()
OUT = ROOT / cfg["paths"]["results_dir"] / "tcga_projection"
EXT = OUT / "external"
SCORES = ["pEMT_specificity", "GS76", "KS", "hallmark_EMT", "puram_pemt", "MLR_mu"]
LABELS = {"pEMT_specificity": "pEMT specificity", "GS76": "76GS", "KS": "KS (Tan)",
          "hallmark_EMT": "Hallmark EMT", "puram_pemt": "Puram pEMT", "MLR_mu": "MLR"}


def pool(yi, vi, model="random"):
    yi, vi = np.asarray(yi, float), np.asarray(vi, float)
    wi = 1.0 / vi
    y_f = float((wi * yi).sum() / wi.sum())
    Q = float((wi * (yi - y_f) ** 2).sum())
    k = len(yi)
    C = wi.sum() - (wi ** 2).sum() / wi.sum()
    tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0
    if model == "fixed":
        tau2 = 0.0
    w = 1.0 / (vi + tau2)
    y = float((w * yi).sum() / w.sum())
    se = float(np.sqrt(1.0 / w.sum()))
    return {"HR": float(np.exp(y)), "CI_low": float(np.exp(y - 1.96 * se)), "CI_up": float(np.exp(y + 1.96 * se)),
            "p_value": float(2 * norm.sf(abs(y / se))), "I2_percent": max(0.0, (Q - (k - 1)) / Q * 100) if Q > 0 else 0.0,
            "tau2": tau2, "Q": Q, "p_heterogeneity": float(chi2.sf(Q, k - 1)) if k > 1 else np.nan, "k": k}


def cox_direct(df, score, covs):
    """Refit and take coef and its standard error straight from the model, not from the CI."""
    covs = [c for c in covs if c in df.columns and df[c].notna().sum() > 0.8 * len(df) and df[c].nunique() > 1]
    d = df.copy()
    s = pd.to_numeric(d[score], errors="coerce")
    d[score] = (s - s.mean()) / (s.std() if s.std() and s.std() > 0 else 1.0)
    try:
        X = design_matrix(d, score, covs)
        if len(X) < 40 or X["OS_event"].sum() < 10:
            return None
        cph, _ = fit_cox(X)
    except Exception:
        return None
    se = float(cph.standard_errors_[score])
    coef = float(cph.params_[score])
    return {"log_HR": coef, "se": se, "var": se ** 2, "n": len(X), "events": int(X["OS_event"].sum()),
            "HR": float(np.exp(coef)), "p": float(cph.summary.loc[score, "p"])}


def cohorts():
    panel = pd.read_csv(OUT / "emt_score_panel_scores.tsv", sep="\t", index_col=0)
    traits = primary_tumours(pd.read_csv(OUT / "tcga_master_trait_table.tsv", sep="\t", index_col=0, low_memory=False))
    tcga = prepare_covariates(traits.join(panel.drop(columns=[c for c in ("pEMT_specificity", "P_pEMT_high") if c in panel])),
                              list(panel.columns))
    out = {"TCGA-HNSC": (tcga, ["stage_num", "age", "site_group", "hpv_positive"])}
    for cid, covs in (("GSE41613", ["stage_num", "age"]),
                      ("GSE65858", ["stage_num", "age", "site_oral_cavity", "site_larynx_hypopharynx", "hpv_positive"]),
                      ("CPTAC_HNSCC", ["stage_num", "age", "site_oral_cavity", "site_larynx_hypopharynx"])):
        f = EXT / f"{cid}_scores.tsv"
        if f.exists():
            out[cid] = (pd.read_csv(f, sep="\t", index_col=0), covs)
    return out


def main() -> None:
    coh = cohorts()
    print("cohorts:", {k: len(v[0]) for k, v in coh.items()})

    # ---- 1-3. univariable and adjusted, fixed and random, SEs straight from the fit ---------------
    rows, per_cohort = [], []
    for score in SCORES:
        for adj_label, use_covs in (("univariable", False), ("adjusted", True)):
            est = {}
            for cid, (df, covs) in coh.items():
                r = cox_direct(df, score, covs if use_covs else [])
                if r:
                    est[cid] = r
                    per_cohort.append({"score": score, "model": adj_label, "cohort": cid, **r})
            if len(est) < 2:
                continue
            yi = [e["log_HR"] for e in est.values()]
            vi = [e["var"] for e in est.values()]
            for m in ("fixed", "random"):
                rows.append({"score": score, "label": LABELS[score], "model": adj_label, "pooling": m,
                             "n_total": sum(e["n"] for e in est.values()), "events_total": sum(e["events"] for e in est.values()),
                             **pool(yi, vi, m)})
    res = pd.DataFrame(rows).round({"HR": 3, "CI_low": 3, "CI_up": 3, "I2_percent": 1, "tau2": 4, "Q": 2})
    res.to_csv(OUT / "survival_sensitivity_pooling.tsv", sep="\t", index=False)
    pc = pd.DataFrame(per_cohort).round({"HR": 3, "log_HR": 4, "se": 4, "p": 4})
    pc.to_csv(OUT / "survival_sensitivity_per_cohort.tsv", sep="\t", index=False)

    print("\n=== Pooled HR per SD, all four combinations ===")
    for score in SCORES:
        s = res[res["score"] == score]
        line = f"{LABELS[score]:18s}"
        for adj in ("univariable", "adjusted"):
            for m in ("fixed", "random"):
                r = s[(s["model"] == adj) & (s["pooling"] == m)]
                if len(r):
                    r = r.iloc[0]
                    line += f" | {adj[:3]}-{m[:3]} {r['HR']:.2f} ({r['CI_low']:.2f}-{r['CI_up']:.2f}) p={r['p_value']:.3f} I2={r['I2_percent']:.0f}%"
        print(line)

    print("\n=== Per-cohort adjusted, how many cohorts reach p < 0.05 and in which direction ===")
    a = pc[pc["model"] == "adjusted"]
    for score in SCORES:
        s = a[a["score"] == score]
        sig = s[s["p"] < 0.05]
        print(f"  {LABELS[score]:18s} HRs " + ", ".join(f"{r.cohort.replace('_HNSCC','-3')} {r.HR:.2f}{'*' if r.p<0.05 else ''}" for r in s.itertuples())
              + f"   [{len(sig)}/{len(s)} significant, {(s['HR']>1).sum()}/{len(s)} above 1]")

    # ---- 4. leave-one-cohort-out ------------------------------------------------------------------
    loo = []
    for score in SCORES:
        s = a[a["score"] == score]
        for drop in list(s["cohort"]) + [None]:
            sub = s[s["cohort"] != drop] if drop else s
            if len(sub) < 2:
                continue
            loo.append({"score": score, "label": LABELS[score], "dropped": drop or "none (all four)",
                        **pool(sub["log_HR"].values, sub["var"].values, "random")})
    loo = pd.DataFrame(loo).round({"HR": 3, "CI_low": 3, "CI_up": 3, "I2_percent": 1, "tau2": 4, "Q": 2})
    loo.to_csv(OUT / "survival_sensitivity_loo.tsv", sep="\t", index=False)
    print("\n=== Leave-one-cohort-out (adjusted, random effects) ===")
    for score in ("pEMT_specificity", "puram_pemt"):
        print(f"  {LABELS[score]}")
        for r in loo[loo["score"] == score].itertuples():
            print(f"     drop {str(r.dropped).replace('_HNSCC','-3'):18s} HR {r.HR:.2f} ({r.CI_low:.2f}-{r.CI_up:.2f}) p={r.p_value:.4f} I2={r.I2_percent:.0f}%")

    # ---- 6. GSE65858 by treatment modality --------------------------------------------------------
    g = coh.get("GSE65858")
    trows = []
    if g is not None and "treatment" in g[0].columns:
        df, covs = g
        print("\n=== GSE65858 by treatment modality ===")
        print("   treatment:", df["treatment"].value_counts(dropna=False).to_dict())
        for score in SCORES:
            for lab, sub in [("all", df)] + [(f"treatment {t}", df[df["treatment"] == t]) for t in sorted(df["treatment"].dropna().unique())]:
                r = cox_direct(sub, score, covs)
                if r:
                    trows.append({"score": score, "label": LABELS[score], "stratum": lab, **r})
        t = pd.DataFrame(trows).round({"HR": 3, "p": 4})
        t.to_csv(OUT / "gse65858_treatment_strata.tsv", sep="\t", index=False)
        for score in SCORES:
            s = t[t["score"] == score]
            print(f"   {LABELS[score]:18s} " + ", ".join(f"{r.stratum} HR {r.HR:.2f} (n={r.n}, e={r.events}) p={r.p:.3f}" for r in s.itertuples()))
    else:
        print("\nGSE65858 treatment field not present in the scores table; rerun 09_external_cohorts.py")

    # ---- truncation sensitivity -------------------------------------------------------------------
    trunc = []
    for months in (36, 60):
        for score in SCORES:
            est = {}
            for cid, (df, covs) in coh.items():
                d = df.copy()
                # TCGA/GEO times are in days for TCGA trait table, months elsewhere; normalise by scale
                scale = 30.44 if d["OS_time"].max() > 500 else 1.0
                cut = months * scale
                d["OS_event"] = np.where(d["OS_time"] > cut, 0, d["OS_event"])
                d["OS_time"] = np.minimum(d["OS_time"], cut)
                r = cox_direct(d, score, covs)
                if r:
                    est[cid] = r
            if len(est) >= 2:
                trunc.append({"truncated_at_months": months, "score": score, "label": LABELS[score],
                              "events_total": sum(e["events"] for e in est.values()),
                              **pool([e["log_HR"] for e in est.values()], [e["var"] for e in est.values()], "random")})
    tr = pd.DataFrame(trunc).round({"HR": 3, "CI_low": 3, "CI_up": 3, "I2_percent": 1, "tau2": 4, "Q": 2})
    tr.to_csv(OUT / "survival_sensitivity_truncation.tsv", sep="\t", index=False)
    print("\n=== Follow-up truncation (adjusted, random effects) ===")
    for r in tr.itertuples():
        if r.score in ("pEMT_specificity", "puram_pemt"):
            print(f"   {r.label:18s} at {r.truncated_at_months} mo: HR {r.HR:.2f} ({r.CI_low:.2f}-{r.CI_up:.2f}) p={r.p_value:.4f} I2={r.I2_percent:.0f}%  events={r.events_total}")


if __name__ == "__main__":
    main()
