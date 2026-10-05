"""Hartung-Knapp random-effects pooling of the survival hazard ratios.

The DerSimonian-Laird interval can be too narrow when four cohorts are pooled and heterogeneity is
present. This stage repools the per-cohort log hazard ratios and their standard errors, taken from the
Cox fits, with the DerSimonian-Laird between-cohort variance, and gives the Hartung-Knapp interval and
p-value (t distribution with k - 1 degrees of freedom) beside the DerSimonian-Laird ones. The pooling is
repeated with each cohort left out. No model is refitted.

Per-cohort estimates: the arm scores, pEMT specificity, the Puram pEMT signature, Hallmark EMT and
MLR-EMT, univariable and adjusted, from 10_cetuximab/05_direct_arm_scores.py, and the adjusted
estimates of 76GS and KS from 04_tcga_projection/13_survival_meta_analysis.py.

Inputs:  results/arm_scores/survival_per_cohort.tsv
         results/tcga_projection/survival_meta_inputs.tsv
Outputs: results/survival_meta_hk/survival_meta_hk.tsv
Usage:   python src/13_revision/01_hartung_knapp_meta.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = RES / "survival_meta_hk"

LABELS = {"pEMT_specificity": "pEMT specificity", "puram_pemt": "Puram pEMT", "hallmark_EMT": "Hallmark EMT",
          "KS": "KS (Tan)", "MLR_mu": "MLR-EMT", "GS76": "76GS", "malignant_arm_core": "Malignant core",
          "stromal_arm_core": "Stromal core", "malignant_arm": "Malignant arm, total share",
          "stromal_arm": "Stromal arm, total share", "malignant_consensus": "Malignant arm, consensus",
          "stromal_consensus": "Stromal arm, consensus"}


def pool(y: np.ndarray, se: np.ndarray) -> dict:
    """DerSimonian-Laird random-effects pooling with the Hartung-Knapp interval beside it."""
    v = se ** 2
    k = len(y)
    w = 1 / v
    mu_f = (w * y).sum() / w.sum()
    q = (w * (y - mu_f) ** 2).sum()
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (v + tau2)
    mu = (ws * y).sum() / ws.sum()
    se_dl = np.sqrt(1 / ws.sum())
    se_hk = np.sqrt((ws * (y - mu) ** 2).sum() / (k - 1) / ws.sum())
    t = stats.t.ppf(0.975, k - 1)
    return {"k": k, "HR": np.exp(mu),
            "DL_low": np.exp(mu - 1.96 * se_dl), "DL_up": np.exp(mu + 1.96 * se_dl),
            "DL_p": 2 * stats.norm.sf(abs(mu / se_dl)),
            "HK_low": np.exp(mu - t * se_hk), "HK_up": np.exp(mu + t * se_hk),
            "HK_p": 2 * stats.t.sf(abs(mu / se_hk), k - 1),
            "I2_percent": 100 * max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0, "tau2": tau2}


def main() -> None:
    a = pd.read_csv(RES / "arm_scores" / "survival_per_cohort.tsv", sep="\t")
    b = pd.read_csv(RES / "tcga_projection" / "survival_meta_inputs.tsv", sep="\t")
    b["model"] = "adjusted"
    cols = ["score", "model", "cohort", "log_HR", "se"]
    d = pd.concat([a[cols], b[cols]]).drop_duplicates(["score", "model", "cohort"])
    rows = []
    for (score, model), g in d.groupby(["score", "model"], sort=False):
        for left_out in ["none"] + list(g["cohort"]):
            h = g if left_out == "none" else g[g["cohort"] != left_out]
            r = pool(h["log_HR"].to_numpy(), h["se"].to_numpy())
            rows.append({"score": score, "label": LABELS.get(score, score), "model": model,
                         "cohort_left_out": left_out, **r})
    res = pd.DataFrame(rows)
    res = res[res["score"].isin(LABELS)]
    order = list(LABELS)
    res["o"] = res["score"].map(order.index)
    res = res.sort_values(["o", "model", "cohort_left_out"], key=lambda c: c if c.name != "cohort_left_out"
                          else c.map(lambda x: "" if x == "none" else x)).drop(columns="o")
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "survival_meta_hk.tsv", sep="\t", index=False, float_format="%.4g")
    print(res[res["cohort_left_out"] == "none"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
