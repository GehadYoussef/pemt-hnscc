"""GSE65021: the Basal centroid score and the malignant core scored one patient at a time.

The Basal centroid score correlates each patient with the TCGA 2015 Basal centroid after median-centring
every centroid gene across the cohort, and the malignant core is the mean z-score of its 12 genes across
the cohort, so both need a cohort. This stage scores each patient against four references and reports
the AUC for long progression-free survival under each:

  within cohort       all 40 patients (the procedure of 03_basal_centroid.py and 10_cetuximab/01)
  leave one out       the other 39 patients only
  external reference  GSE65858, an external cohort on the same Illumina array family (gene medians for
                      the centroid, gene means and SDs for the core)
  no reference        the patient's own log2 values: Pearson correlation of uncentred expression with the
                      Basal centroid, and the mean log2 expression of the 12 core genes

Inputs:  data/raw/external_cohorts/{GSE65021,GSE65858}_series_matrix.txt.gz, GPL10558.annot.gz
         data/raw/tcga2015_subtype_centroids/classification_centroid_728genes.txt (from 03_basal_centroid.py)
         results/cetuximab_cohort/GSE65021_scores.tsv, results/arm_scores/arm_gene_sets.tsv
Outputs: results/single_patient/single_patient_auc.tsv, single_patient_scores.tsv
Usage:   python src/13_revision/08_single_patient_scoring.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "external_cohorts"
CDIR = ROOT / "data" / "raw" / "tcga2015_subtype_centroids"
OUT = ROOT / "results" / "single_patient"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    cx = _load(ROOT / "src" / "10_cetuximab" / "01_cetuximab_cohort.py", "cx")
    x, _ = cx.read_series_matrix(RAW / "GSE65021_series_matrix.txt.gz")
    e = cx.collapse(x, cx.read_annotation(RAW / "GPL10558.annot.gz"))
    sc = pd.read_csv(ROOT / "results/cetuximab_cohort/GSE65021_scores.tsv", sep="\t", index_col=0)
    e = e[sc.index]
    y = sc["long_pfs"].to_numpy(int)
    xr, _ = cx.read_series_matrix(RAW / "GSE65858_series_matrix.txt.gz")
    ref = cx.collapse(xr, cx.read_annotation(RAW / "GPL10558.annot.gz"))

    cen = pd.read_csv(CDIR / "classification_centroid_728genes.txt", sep="\t", index_col=0)["Basal"]
    core = pd.read_csv(ROOT / "results/arm_scores/arm_gene_sets.tsv", sep="\t") \
        .set_index("score").loc["malignant_arm_core", "genes"].split(", ")

    shared = [g for g in cen.index if g in e.index and g in ref.index]
    cgenes = [g for g in core if g in e.index and g in ref.index]
    E = e.loc[shared]
    C = e.loc[cgenes]

    basal, corez = {}, {}
    basal["within cohort"] = E.sub(E.median(axis=1), axis=0).corrwith(cen[shared])
    basal["leave one out"] = pd.Series({s: np.corrcoef(E[s] - E.drop(columns=s).median(axis=1), cen[shared])[0, 1]
                                        for s in E.columns})
    basal["external reference (GSE65858)"] = E.sub(ref.loc[shared].median(axis=1), axis=0).corrwith(cen[shared])
    basal["no reference"] = E.corrwith(cen[shared])

    zc = lambda X, mu, sd: X.sub(mu, axis=0).div(sd.replace(0, 1), axis=0).mean(axis=0)  # noqa: E731
    corez["within cohort"] = zc(C, C.mean(axis=1), C.std(axis=1))
    corez["leave one out"] = pd.Series({s: zc(C[[s]], C.drop(columns=s).mean(axis=1),
                                                C.drop(columns=s).std(axis=1))[s] for s in C.columns})
    corez["external reference (GSE65858)"] = zc(C, ref.loc[cgenes].mean(axis=1), ref.loc[cgenes].std(axis=1))
    corez["no reference"] = C.mean(axis=0)

    rows, scores = [], pd.DataFrame(index=e.columns)
    for name, d in (("Basal centroid score", basal), ("malignant core", corez)):
        base = d["within cohort"]
        for scheme, v in d.items():
            v = v.reindex(e.columns)
            auc, lo, hi = cx.bootstrap_auc(y, v.to_numpy())
            rows.append({"score": name, "scheme": scheme, "genes": len(shared) if name.startswith("Basal") else len(cgenes),
                         "AUC": auc, "AUC_low": lo, "AUC_up": hi,
                         "rho_with_within_cohort": spearmanr(v, base)[0]})
            scores[f"{name} | {scheme}"] = v
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "single_patient_auc.tsv", sep="\t", index=False, float_format="%.4g")
    scores.to_csv(OUT / "single_patient_scores.tsv", sep="\t", float_format="%.5g")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
