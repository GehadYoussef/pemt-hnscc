"""Test whether the cetuximab-benefit signal in GSE65021 is specific to the tumour-intrinsic axis.

Reads the per-patient scores written by 01_cetuximab_cohort.py. The classifier is not re-projected.
All analyses here were specified post hoc, after the first results of 01_cetuximab_cohort.py.

  1. Correlation structure. Spearman correlations between all scores in this cohort, printed next
     to the TCGA correlations with pEMT specificity. GSE65021 is recurrent or metastatic disease,
     so the correlations can differ from TCGA.
  2. Head-to-head logistic models. pEMT specificity is entered together with each other score,
     unadjusted and adjusted for the clinical covariates, to test whether the axis adds to the
     epithelial scores.
  3. Primary specimens only. Mann-Whitney tests and AUCs on the 31 primary-tumour specimens, with
     the 9 recurrence or metastasis specimens removed.
  4. Scores oriented away from the epithelial state against the epithelial scores. Two of the three
     MLR-EMT predictor genes, CLDN7 and CDH1, are epithelial, and MLR-EMT correlates -0.76 with
     76GS in TCGA. MLR-EMT and KS are each entered alongside 76GS and alongside Puram epithelial
     differentiation. In these cohorts the high end of MLR-EMT is the hybrid state.
  5. A combined marker. The partner is Puram epithelial differentiation, which comes from the same
     single-cell source as the axis and defines the classifier's epithelial-like class. It was not
     chosen on its performance here. The primary combination is an equal-weight sum of the two
     standardised scores, with no fitted weights. A fitted two-term logistic model is reported only
     under leave-one-out cross-validation. 76GS and MLR-EMT (sign reversed) are sensitivity
     partners. AUCs and the gain over the axis alone carry 2,000-sample bootstrap intervals.

Inputs:  results/cetuximab_cohort/GSE65021_scores.tsv
Outputs: results/cetuximab_cohort/GSE65021_score_correlations.tsv
         results/cetuximab_cohort/GSE65021_headtohead.tsv
         results/cetuximab_cohort/GSE65021_primaries_only.tsv
         results/cetuximab_cohort/GSE65021_mlr_vs_epithelial.tsv
         results/cetuximab_cohort/GSE65021_combined_score.tsv
Usage:   python src/10_cetuximab/03_cetuximab_headtohead.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "cetuximab_cohort"

SCORES = ["pEMT_specificity", "P_pEMT_high", "puram_epi_dif_1", "GS76", "puram_pemt",
          "hallmark_EMT", "KS", "MLR_mu", "schinke2022_egfr_emt", "zhou2025_fdeg",
          "zhou2025_invgrn", "ourailidis2026_tbs", "zhou2025_predictive"]
LABELS = {"pEMT_specificity": "pEMT specificity", "P_pEMT_high": "P(pEMT-high)",
          "puram_epi_dif_1": "Puram epithelial", "GS76": "76GS", "puram_pemt": "Puram pEMT",
          "hallmark_EMT": "Hallmark EMT", "KS": "KS", "MLR_mu": "MLR",
          "schinke2022_egfr_emt": "Schinke EGFR-EMT", "zhou2025_fdeg": "Zhou fDEG",
          "zhou2025_invgrn": "Zhou invGRN", "ourailidis2026_tbs": "Tumour budding",
          "zhou2025_predictive": "Zhou 9-gene (derived here)"}


def z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def logit_or(df: pd.DataFrame, terms: list[str], covariates: list[str] | None = None):
    """Odds ratio of long PFS per SD for each term, all terms entered together."""
    covariates = covariates or []
    sub = df.dropna(subset=["long_pfs"] + terms + covariates).copy()
    for t in terms:
        sub[t] = z(sub[t])
    cov = [c for c in covariates if sub[c].nunique() > 1]
    X = sm.add_constant(sub[terms + cov].astype(float), has_constant="add")
    try:
        res = sm.Logit(sub["long_pfs"].astype(float), X).fit(disp=0, maxiter=500)
        return {t: (float(np.exp(res.params[t])), *np.exp(res.conf_int().loc[t]).tolist(),
                    float(res.pvalues[t])) for t in terms}, len(sub)
    except Exception as exc:
        return {t: (np.nan, np.nan, np.nan, np.nan) for t in terms}, len(sub)


def main() -> None:
    df = pd.read_csv(OUT / "GSE65021_scores.tsv", sep="\t", index_col=0)
    print(f"{len(df)} patients, {int(df['long_pfs'].sum())} long PFS, "
          f"{int((1 - df['long_pfs']).sum())} short PFS, "
          f"{int((1 - df['specimen_recurrence']).sum())} primary specimens")

    # 1. correlation structure within this cohort
    rho = pd.DataFrame(index=SCORES, columns=SCORES, dtype=float)
    for a in SCORES:
        for b in SCORES:
            rho.loc[a, b] = spearmanr(df[a], df[b]).statistic
    rho.index = [LABELS[s] for s in SCORES]
    rho.columns = [LABELS[s] for s in SCORES]
    rho.round(3).to_csv(OUT / "GSE65021_score_correlations.tsv", sep="\t")
    print("\nSpearman correlations with pEMT specificity in GSE65021 "
          "(TCGA value in brackets where available):")
    tcga = {"GS76": 0.00, "KS": 0.09, "hallmark_EMT": 0.19, "MLR_mu": 0.26, "puram_pemt": 0.51,
            "zhou2025_predictive": 0.75, "ourailidis2026_tbs": 0.71, "zhou2025_fdeg": 0.64,
            "zhou2025_invgrn": 0.61, "schinke2022_egfr_emt": 0.42}
    for s in SCORES:
        if s == "pEMT_specificity":
            continue
        t = f" [TCGA {tcga[s]:+.2f}]" if s in tcga else ""
        print(f"  {LABELS[s]:<28} {spearmanr(df['pEMT_specificity'], df[s]).statistic:+.2f}{t}")

    # 2. head-to-head: the axis entered with each other score
    covs = ["age", "stage_num", "grade", "prior_rt", "site_oral_cavity", "site_oropharynx",
            "specimen_recurrence"]
    rows = []
    for other in [s for s in SCORES if s not in ("pEMT_specificity", "P_pEMT_high")]:
        for adj, covset in (("unadjusted", []), ("adjusted", covs)):
            ors, n = logit_or(df, ["pEMT_specificity", other], covset)
            a, o = ors["pEMT_specificity"], ors[other]
            rows.append({"other_score": LABELS[other], "model": adj, "n": n,
                         "rho_with_axis": spearmanr(df["pEMT_specificity"], df[other]).statistic,
                         "OR_axis": a[0], "axis_low": a[1], "axis_up": a[2], "p_axis": a[3],
                         "OR_other": o[0], "other_low": o[1], "other_up": o[2], "p_other": o[3]})
    h2h = pd.DataFrame(rows)
    h2h.to_csv(OUT / "GSE65021_headtohead.tsv", sep="\t", index=False)
    print("\nHead-to-head logistic models (OR of long PFS per SD, both terms entered together):")
    print(h2h[h2h["model"] == "unadjusted"][
        ["other_score", "rho_with_axis", "OR_axis", "p_axis", "OR_other", "p_other"]
    ].round(3).to_string(index=False))

    # 3. primaries only
    prim = df[df["specimen_recurrence"] == 0]
    rows = []
    for s in SCORES:
        a = prim.loc[prim["long_pfs"] == 1, s]
        b = prim.loc[prim["long_pfs"] == 0, s]
        u = mannwhitneyu(a, b, alternative="greater")
        rows.append({"score": LABELS[s], "n_long": len(a), "n_short": len(b),
                     "rank_biserial": 2 * mannwhitneyu(a, b).statistic / (len(a) * len(b)) - 1,
                     "AUC": roc_auc_score(prim["long_pfs"], prim[s]),
                     "p_one_sided": u.pvalue})
    pr = pd.DataFrame(rows).sort_values("p_one_sided")
    pr.to_csv(OUT / "GSE65021_primaries_only.tsv", sep="\t", index=False)
    print(f"\nPrimary-tumour specimens only (n = {len(prim)}):")
    print(pr.round(3).to_string(index=False))

    # 4. the scores oriented away from the epithelial state against the epithelial scores
    rows = []
    for mes in ("MLR_mu", "KS"):
        for epi in ("GS76", "puram_epi_dif_1"):
            for adj, covset in (("unadjusted", []), ("adjusted", covs)):
                ors, n = logit_or(df, [mes, epi], covset)
                m, e = ors[mes], ors[epi]
                rows.append({"away_from_epithelial_score": LABELS[mes], "epithelial_score": LABELS[epi],
                             "model": adj, "n": n, "rho": spearmanr(df[mes], df[epi]).statistic,
                             "OR_away": m[0], "away_low": m[1], "away_up": m[2], "p_away": m[3],
                             "OR_epi": e[0], "epi_low": e[1], "epi_up": e[2], "p_epi": e[3]})
    me = pd.DataFrame(rows)
    me.to_csv(OUT / "GSE65021_mlr_vs_epithelial.tsv", sep="\t", index=False)
    print("\nScores oriented away from the epithelial state against epithelial scores "
          "(OR of long PFS per SD):")
    print(me[["away_from_epithelial_score", "epithelial_score", "model", "rho", "OR_away", "p_away",
              "OR_epi", "p_epi"]].round(3).to_string(index=False))

    # 5. combined marker: axis + a pre-specified epithelial partner
    combined_marker(df)


PARTNERS = [("puram_epi_dif_1", +1, "primary"), ("GS76", +1, "sensitivity"),
            ("MLR_mu", -1, "sensitivity")]
N_BOOT = 2000
SEED = 20260921


def loo_auc(y: np.ndarray, X: np.ndarray) -> float:
    """AUC of held-out predictions from an unpenalised logistic model refitted with each patient left out."""
    pred = np.empty(len(y))
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        res = sm.Logit(y[keep], sm.add_constant(X[keep], has_constant="add")).fit(disp=0, maxiter=500)
        pred[i] = res.predict(sm.add_constant(X[i:i + 1], has_constant="add"))[0]
    return roc_auc_score(y, pred)


def combined_marker(df: pd.DataFrame) -> None:
    rng = np.random.default_rng(SEED)
    y = df["long_pfs"].to_numpy(int)
    axis = z(df["pEMT_specificity"]).to_numpy()
    prim = (df["specimen_recurrence"] == 0).to_numpy()
    boots = [rng.choice(len(y), len(y), replace=True) for _ in range(N_BOOT)]
    boots = [b for b in boots if 0 < y[b].sum() < len(b)]

    def ci(stat):
        vals = [stat(b) for b in boots]
        return np.percentile(vals, [2.5, 97.5])

    rows = []
    a_lo, a_hi = ci(lambda b: roc_auc_score(y[b], axis[b]))
    rows.append({"model": "axis alone", "partner": "", "role": "reference",
                 "AUC": roc_auc_score(y, axis), "AUC_low": a_lo, "AUC_up": a_hi,
                 "gain_over_axis": 0.0, "gain_low": np.nan, "gain_up": np.nan,
                 "LOO_AUC_fitted": loo_auc(y, axis[:, None]),
                 "AUC_primaries_only": roc_auc_score(y[prim], axis[prim])})
    for col, sign, role in PARTNERS:
        part = sign * z(df[col]).to_numpy()
        comb = axis + part
        lo, hi = ci(lambda b: roc_auc_score(y[b], comb[b]))
        g_lo, g_hi = ci(lambda b: roc_auc_score(y[b], comb[b]) - roc_auc_score(y[b], axis[b]))
        rows.append({"model": "axis + partner, equal weights", "partner": LABELS[col]
                     + (" (sign reversed)" if sign < 0 else ""), "role": role,
                     "AUC": roc_auc_score(y, comb), "AUC_low": lo, "AUC_up": hi,
                     "gain_over_axis": roc_auc_score(y, comb) - roc_auc_score(y, axis),
                     "gain_low": g_lo, "gain_up": g_hi,
                     "LOO_AUC_fitted": loo_auc(y, np.column_stack([axis, part])),
                     "AUC_primaries_only": roc_auc_score(y[prim], comb[prim])})
    cm = pd.DataFrame(rows)
    cm.to_csv(OUT / "GSE65021_combined_score.tsv", sep="\t", index=False)
    print(f"\nCombined marker ({len(boots)} bootstrap resamples, partner pre-specified as "
          "Puram epithelial differentiation):")
    print(cm.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
