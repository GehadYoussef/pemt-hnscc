"""Test whether the cetuximab association of the malignant arm in GSE65021 is a tumour-purity effect.

The composition simulation (src/11_composition_and_mechanism/01_composition_simulation.py) shows
that the malignant core and pEMT specificity rise with the pEMT state of the malignant cells and
also fall as fibroblasts and immune cells dilute the tumour, so in bulk they partly track purity.
Non-malignant content is therefore added as a covariate to the Firth model of long PFS.

Non-malignant content is measured two ways: the stromal core (fibroblast genes, from
05_direct_arm_scores.py) and an immune score, the mean per-gene z-score of 18 pan-leukocyte,
lymphocyte and myeloid markers fixed in IMMUNE below. Both are standardised within the cohort. The
malignant core and pEMT specificity are each fitted alone, with the stromal core, with the immune
score and with both, and the immune score is also fitted alone.

Inputs:  data/raw/external_cohorts/GSE65021_series_matrix.txt.gz
         data/raw/external_cohorts/GPL10558.annot.gz
         results/cetuximab_cohort/GSE65021_scores.tsv
         results/arm_scores/GSE65021_arm_scores.tsv
Outputs: results/cetuximab_cohort/GSE65021_purity_adjustment.tsv
Usage:   python src/10_cetuximab/07_purity_adjustment.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
from pemt.firth import firth  # noqa: E402

IMMUNE = ["PTPRC", "CD2", "CD3D", "CD3E", "CD247", "CD8A", "CD19", "MS4A1", "CD79A", "CD68", "CD14",
          "CD163", "LYZ", "FCGR3A", "ITGAM", "CSF1R", "CD74", "HLA-DRA"]


def main() -> None:
    spec = importlib.util.spec_from_file_location("cx", ROOT / "src" / "10_cetuximab" / "01_cetuximab_cohort.py")
    cx = importlib.util.module_from_spec(spec)
    sys.argv = [sys.argv[0]]
    spec.loader.exec_module(cx)
    expr, _ = cx.read_series_matrix(ROOT / "data" / "raw" / "external_cohorts" / "GSE65021_series_matrix.txt.gz")
    e = cx.collapse(expr, cx.read_annotation(ROOT / "data" / "raw" / "external_cohorts" / "GPL10558.annot.gz"))
    g = [x for x in IMMUNE if x in e.index]
    z = e.loc[g].sub(e.loc[g].mean(axis=1), axis=0).div(e.loc[g].std(axis=1), axis=0)
    immune = z.mean(axis=0)

    sc = pd.read_csv(ROOT / "results/cetuximab_cohort/GSE65021_scores.tsv", sep="\t", index_col=0)
    arms = pd.read_csv(ROOT / "results/arm_scores/GSE65021_arm_scores.tsv", sep="\t", index_col=0)
    d = sc[["long_pfs", "pEMT_specificity"]].join(arms[["malignant_arm_core", "stromal_arm_core"]])
    d["immune_score"] = immune.loc[d.index]
    std = lambda s: (s - s.mean()) / s.std()  # noqa: E731
    for c in ["pEMT_specificity", "malignant_arm_core", "stromal_arm_core", "immune_score"]:
        d[c] = std(d[c])
    y = d["long_pfs"].values.astype(float)

    rows = []
    for score in ["malignant_arm_core", "pEMT_specificity"]:
        for label, covs in [("alone", []), ("+ stromal core", ["stromal_arm_core"]), ("+ immune score", ["immune_score"]),
                            ("+ stromal core + immune score", ["stromal_arm_core", "immune_score"])]:
            names = [score] + covs
            X = np.column_stack([np.ones(len(d))] + [d[c].values for c in names])
            res = firth(y, X, ["const"] + names)
            or_, lo, hi, p = res[score]
            rows.append({"score": score, "model": label, "OR_per_SD": or_, "CI_low": lo, "CI_up": hi, "p": p,
                         "rho_with_stromal_core": spearmanr(d[score], d["stromal_arm_core"])[0],
                         "rho_with_immune_score": spearmanr(d[score], d["immune_score"])[0]})
    out = pd.DataFrame(rows)
    imm_alone = firth(y, np.column_stack([np.ones(len(d)), d["immune_score"].values]), ["const", "immune_score"])["immune_score"]
    print(f"immune markers present {len(g)}/{len(IMMUNE)}, immune score alone OR {imm_alone[0]:.2f} "
          f"({imm_alone[1]:.2f} to {imm_alone[2]:.2f}), p {imm_alone[3]:.3f}")
    out.to_csv(ROOT / "results/cetuximab_cohort/GSE65021_purity_adjustment.tsv", sep="\t", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
