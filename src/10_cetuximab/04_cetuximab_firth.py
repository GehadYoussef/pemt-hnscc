"""Refit every logistic model reported for GSE65021 with Firth's penalty.

With 14 long-PFS patients, the ordinary maximum-likelihood models of 01_cetuximab_cohort.py and
03_cetuximab_headtohead.py give odds ratios of 18 to 60 per SD whenever a strong epithelial score is
in the model, or when seven clinical covariates are added. These values come from quasi-separation.
The same models are refitted here with Firth's penalty (pemt.firth), with profile
penalised-likelihood intervals and penalised likelihood-ratio p-values. Scores and covariates are
standardised before fitting.

Models
  1. each score alone, unadjusted and adjusted for age, stage, grade, prior radiotherapy, two site
     indicators and specimen type
  2. the axis with each other score (head-to-head), unadjusted
  3. each score oriented away from the epithelial state (MLR-EMT, KS) with each epithelial score
     (76GS, Puram epithelial), unadjusted

Inputs:  results/cetuximab_cohort/GSE65021_scores.tsv
Outputs: results/cetuximab_cohort/GSE65021_firth_single.tsv
         results/cetuximab_cohort/GSE65021_firth_headtohead.tsv
         results/cetuximab_cohort/GSE65021_firth_vs_epithelial.tsv
Usage:   python src/10_cetuximab/04_cetuximab_firth.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from pemt import firth as _firth  # noqa: E402

OUT = HERE.parents[1] / "results" / "cetuximab_cohort"
COVS = ["age", "stage_num", "grade", "prior_rt", "site_oral_cavity", "site_oropharynx", "specimen_recurrence"]
SCORES = ["pEMT_specificity", "P_pEMT_high", "puram_epi_dif_1", "GS76", "puram_pemt", "hallmark_EMT",
          "KS", "MLR_mu", "schinke2022_egfr_emt", "zhou2025_fdeg", "zhou2025_invgrn", "ourailidis2026_tbs",
          "zhou2025_predictive"]
LABELS = {"pEMT_specificity": "pEMT specificity", "P_pEMT_high": "P(pEMT-high)",
          "puram_epi_dif_1": "Puram epithelial", "GS76": "76GS", "puram_pemt": "Puram pEMT",
          "hallmark_EMT": "Hallmark EMT", "KS": "KS", "MLR_mu": "MLR-EMT",
          "schinke2022_egfr_emt": "Schinke EGFR-EMT", "zhou2025_fdeg": "Zhou fDEG",
          "zhou2025_invgrn": "Zhou invGRN", "ourailidis2026_tbs": "Tumour budding",
          "zhou2025_predictive": "Zhou 9-gene (derived here)"}


def fit(df: pd.DataFrame, terms: list[str], covs: list[str]) -> tuple[dict, int]:
    sub = df.dropna(subset=["long_pfs"] + terms + covs).copy()
    covs = [c for c in covs if sub[c].nunique() > 1]
    # Scores per SD. Covariates are also centred and scaled, which leaves the score ORs unchanged
    # and keeps the Newton iterations well conditioned (age would otherwise enter in years).
    for t in terms + covs:
        sub[t] = (sub[t] - sub[t].mean()) / sub[t].std()
    X = np.column_stack([np.ones(len(sub))] + [sub[c].astype(float) for c in terms + covs])
    return _firth.firth(sub["long_pfs"].to_numpy(float), X, ["const"] + terms + covs), len(sub)


def main() -> None:
    df = pd.read_csv(OUT / "GSE65021_scores.tsv", sep="\t", index_col=0)

    rows = []
    for s in SCORES:
        for model, covs in (("unadjusted", []), ("adjusted", COVS)):
            r, n = fit(df, [s], covs)
            rows.append({"score": LABELS[s], "model": model, "n": n, "OR": r[s][0], "OR_low": r[s][1],
                         "OR_up": r[s][2], "p": r[s][3]})
    single = pd.DataFrame(rows)
    single.to_csv(OUT / "GSE65021_firth_single.tsv", sep="\t", index=False)
    print("Firth OR of long PFS per SD, each score alone:")
    print(single.pivot(index="score", columns="model", values=["OR", "p"]).round(3).to_string())

    rows = []
    for other in [s for s in SCORES if s not in ("pEMT_specificity", "P_pEMT_high")]:
        r, n = fit(df, ["pEMT_specificity", other], [])
        a, o = r["pEMT_specificity"], r[other]
        rows.append({"other_score": LABELS[other], "n": n,
                     "rho_with_axis": spearmanr(df["pEMT_specificity"], df[other])[0],
                     "OR_axis": a[0], "axis_low": a[1], "axis_up": a[2], "p_axis": a[3],
                     "OR_other": o[0], "other_low": o[1], "other_up": o[2], "p_other": o[3]})
    h2h = pd.DataFrame(rows)
    h2h.to_csv(OUT / "GSE65021_firth_headtohead.tsv", sep="\t", index=False)
    print("\nFirth head-to-head with the axis (unadjusted):")
    print(h2h[["other_score", "rho_with_axis", "OR_axis", "p_axis", "OR_other", "p_other"]].round(3).to_string(index=False))

    rows = []
    for away in ("MLR_mu", "KS"):
        for epi in ("GS76", "puram_epi_dif_1"):
            r, n = fit(df, [away, epi], [])
            rows.append({"away_from_epithelial_score": LABELS[away], "epithelial_score": LABELS[epi],
                         "rho": spearmanr(df[away], df[epi])[0],
                         "OR_away": r[away][0], "away_low": r[away][1], "away_up": r[away][2], "p_away": r[away][3],
                         "OR_epi": r[epi][0], "epi_low": r[epi][1], "epi_up": r[epi][2], "p_epi": r[epi][3]})
    me = pd.DataFrame(rows)
    me.to_csv(OUT / "GSE65021_firth_vs_epithelial.tsv", sep="\t", index=False)
    print("\nFirth, scores oriented away from the epithelial state with each epithelial score:")
    print(me.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
