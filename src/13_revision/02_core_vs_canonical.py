"""GSE65021: each malignant-arm measure and the canonical Puram pEMT signature in one Firth model.

The malignant core and pEMT specificity are each entered together with the canonical Puram pEMT
signature in a Firth-penalised logistic regression of long progression-free survival. All scores are
standardised within the cohort, as in every other GSE65021 model. No score is refitted.

Inputs:  results/cetuximab_cohort/GSE65021_scores.tsv, results/arm_scores/GSE65021_arm_scores.tsv
Outputs: results/cetuximab_cohort/GSE65021_core_vs_canonical.tsv
Usage:   python src/13_revision/02_core_vs_canonical.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pemt.firth import firth  # noqa: E402

RES = ROOT / "results"
OUT = RES / "cetuximab_cohort"


def z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)


def main() -> None:
    sc = pd.read_csv(OUT / "GSE65021_scores.tsv", sep="\t", index_col=0)
    arm = pd.read_csv(RES / "arm_scores" / "GSE65021_arm_scores.tsv", sep="\t", index_col=0)
    d = sc[["long_pfs", "puram_pemt", "pEMT_specificity"]].join(arm[["malignant_arm_core"]]).dropna()
    y = d["long_pfs"].to_numpy()
    rows = []
    for primary in ["malignant_arm_core", "pEMT_specificity"]:
        X = np.column_stack([np.ones(len(d)), z(d[primary]), z(d["puram_pemt"])])
        res = firth(y, X, ["const", primary, "puram_pemt"])
        rho = spearmanr(d[primary], d["puram_pemt"])[0]
        for term in (primary, "puram_pemt"):
            o, lo, up, p = res[term]
            rows.append({"model": f"{primary} + puram_pemt", "term": term, "n": len(d),
                         "rho_between_terms": round(float(rho), 3), "OR_firth": o, "OR_low": lo,
                         "OR_up": up, "p": p})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "GSE65021_core_vs_canonical.tsv", sep="\t", index=False, float_format="%.4g")
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
