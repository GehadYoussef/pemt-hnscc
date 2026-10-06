"""Basal centroid score in the two cetuximab-treated xenograft panels, against the malignant core.

In GSE65021 the malignant core and the Basal centroid score correlate at 0.80 and cannot be separated
(03_basal_centroid.py). The xenograft panels are the only other cetuximab-treated data, so this stage
asks whether the Basal score tracks response there as the malignant core does.

Expression is loaded exactly as in 10_cetuximab/09_pdx_cetuximab.py (GSE84713: GCRMA log2, GPL570,
highest-mean probe, the two same-patient model pairs averaged per patient; GSE183881: RSEM isoform TPM
summed to genes, log2(x + 1)). The Basal score is the Pearson correlation with the TCGA 2015 Basal
centroid after median-centring each centroid gene within the panel, as in 03_basal_centroid.py. The
malignant core and the other scores are read from results/pdx_cetuximab.

For each panel: AUC with 2,000-resample bootstrap interval, two-sided Mann-Whitney p, Firth odds ratio
per SD, the Spearman correlation of each score with the Basal score, a joint Firth model of the
malignant core and the Basal score, and the paired bootstrap difference in AUC (core minus Basal). The
two panels are pooled by inverse-variance weighting of the log Firth odds ratios, as in
10_cetuximab/09. The depositors' own subtype labels for GSE84713 are tabulated against response.

Inputs:  data/raw/pdx/ (as for 10_cetuximab/09), data/raw/external_cohorts/GPL570.annot.gz
         data/raw/tcga2015_subtype_centroids/classification_centroid_728genes.txt (from 03_basal_centroid.py)
         results/pdx_cetuximab/{GSE84713,GSE183881}_scores.tsv
Outputs: results/pdx_basal/*.tsv
Usage:   python src/13_revision/09_pdx_basal_centroid.py
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pemt.firth import firth  # noqa: E402

D = ROOT / "data" / "raw" / "pdx"
PDX = ROOT / "results" / "pdx_cetuximab"
CDIR = ROOT / "data" / "raw" / "tcga2015_subtype_centroids"
OUT = ROOT / "results" / "pdx_basal"
SCORES = ["basal_score", "malignant_arm_core", "pEMT_specificity", "stromal_arm_core", "zhou2025_predictive"]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def basal_score(e: pd.DataFrame) -> pd.Series:
    cen = pd.read_csv(CDIR / "classification_centroid_728genes.txt", sep="\t", index_col=0)["Basal"]
    shared = cen.index.intersection(e.index)
    x = e.loc[shared]
    return x.sub(x.median(axis=1), axis=0).corrwith(cen[shared]).rename("basal_score")


def gse84713() -> pd.DataFrame:
    ext = _load(ROOT / "src/04_tcga_projection/09_external_cohorts.py", "ext84")
    expr, _ = ext.read_series_matrix(D / "GSE84713_series_matrix.txt.gz")
    e = ext.collapse(expr, ext.read_annotation(ROOT / "data/raw/external_cohorts/GPL570.annot.gz"))
    sc = pd.read_csv(PDX / "GSE84713_scores.tsv", sep="\t", index_col=0)
    s = sc.join(basal_score(e[sc.index]))
    num = [c for c in s.columns if c not in ("model", "patient", "response", "molecular subtype")]
    per = s.groupby("patient").agg({**{c: "mean" for c in num}, "response": "max", "molecular subtype": "first"})
    tab = pd.crosstab(per["molecular subtype"], per["response"]).rename(columns={0: "non_responder", 1: "responder"})
    tab.to_csv(OUT / "GSE84713_published_subtype.tsv", sep="\t")
    return per


def gse183881() -> pd.DataFrame:
    s09 = _load(ROOT / "src/10_cetuximab/09_pdx_cetuximab.py", "s09")
    meta = s09.characteristics(D / "GSE183881_series_matrix.txt.gz")
    meta = meta[meta["cetuximab response"].isin(["Sensitive", "Resistant"])]
    tx = pd.read_csv(D / "GSE183881_expression-human-clsfy.summary.isoforms.TPM.tab.gz", sep="\t", index_col=0)
    tx.columns = [re.search(r"(JGHL\d+)", c).group(1) for c in tx.columns]
    m = pd.read_csv(D / "gencode.v38.metadata.HGNC.gz", sep="\t", header=None, names=["tx", "sym", "hgnc"])
    m["tx"] = m["tx"].str.split(".").str[0]
    sym = m.drop_duplicates("tx").set_index("tx")["sym"]
    tx = tx[tx.index.isin(sym.index)]
    gene = tx.groupby(sym.reindex(tx.index).values).sum()
    e = np.log2(gene[meta["title"].tolist()] + 1)
    e.columns = meta.index
    sc = pd.read_csv(PDX / "GSE183881_scores.tsv", sep="\t", index_col=0)
    return sc.join(basal_score(e))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cx = _load(ROOT / "src" / "10_cetuximab" / "01_cetuximab_cohort.py", "cx")
    z = lambda v: (v - v.mean()) / v.std()  # noqa: E731
    assoc, joint, paired = [], [], []
    for cohort, d in (("GSE84713", gse84713()), ("GSE183881", gse183881())):
        d.to_csv(OUT / f"{cohort}_scores_with_basal.tsv", sep="\t")
        y = d["response"].to_numpy(int)
        for sc in SCORES:
            auc, lo, hi = cx.bootstrap_auc(y, d[sc].to_numpy())
            f = firth(y, np.column_stack([np.ones(len(d)), z(d[sc])]), ["c", sc])[sc]
            assoc.append({"cohort": cohort, "score": sc, "n_responders": int(y.sum()), "n_non_responders": int((1 - y).sum()),
                          "AUC": auc, "AUC_low": lo, "AUC_up": hi,
                          "p_two_sided": mannwhitneyu(d.loc[y == 1, sc], d.loc[y == 0, sc]).pvalue,
                          "OR_firth": f[0], "OR_low": f[1], "OR_up": f[2], "p_firth": f[3],
                          "rho_with_basal_score": spearmanr(d[sc], d["basal_score"])[0]})
        X = np.column_stack([np.ones(len(d)), z(d["malignant_arm_core"]), z(d["basal_score"])])
        jf = firth(y, X, ["c", "malignant_arm_core", "basal_score"])
        for term in ("malignant_arm_core", "basal_score"):
            o, lo, hi, p = jf[term]
            joint.append({"cohort": cohort, "term": term, "OR_firth": o, "OR_low": lo, "OR_up": hi, "p": p,
                          "rho_between_terms": spearmanr(d["malignant_arm_core"], d["basal_score"])[0]})
        rng = np.random.default_rng(20260920)
        a, b = d["malignant_arm_core"].to_numpy(), d["basal_score"].to_numpy()
        diffs = []
        for _ in range(2000):
            i = rng.integers(0, len(y), len(y))
            if y[i].min() == y[i].max():
                continue
            diffs.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
        paired.append({"cohort": cohort, "AUC_core": roc_auc_score(y, a), "AUC_basal": roc_auc_score(y, b),
                       "difference": roc_auc_score(y, a) - roc_auc_score(y, b),
                       "diff_low": np.percentile(diffs, 2.5), "diff_up": np.percentile(diffs, 97.5)})
    assoc = pd.DataFrame(assoc)
    rows = []
    for sc, g in assoc.groupby("score", sort=False):
        bb = np.log(g["OR_firth"])
        se = (np.log(g["OR_up"]) - np.log(g["OR_low"])) / (2 * 1.96)
        w = 1 / se ** 2
        m, s = float((w * bb).sum() / w.sum()), float(np.sqrt(1 / w.sum()))
        rows.append({"score": sc, "pooled_OR": np.exp(m), "low": np.exp(m - 1.96 * s), "up": np.exp(m + 1.96 * s),
                     "p": 2 * norm.sf(abs(m / s))})
    assoc.to_csv(OUT / "pdx_basal_association.tsv", sep="\t", index=False, float_format="%.4g")
    pd.DataFrame(joint).to_csv(OUT / "pdx_basal_joint.tsv", sep="\t", index=False, float_format="%.4g")
    pd.DataFrame(paired).to_csv(OUT / "pdx_basal_paired_auc.tsv", sep="\t", index=False, float_format="%.4g")
    pd.DataFrame(rows).to_csv(OUT / "pdx_basal_pooled.tsv", sep="\t", index=False, float_format="%.4g")
    print(assoc[["cohort", "score", "AUC", "p_two_sided", "OR_firth", "p_firth"]].round(3).to_string(index=False))
    print(pd.DataFrame(joint).round(3).to_string(index=False))
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
