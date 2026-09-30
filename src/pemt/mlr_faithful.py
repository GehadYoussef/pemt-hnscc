"""MLR-EMT score (George et al. 2017), following the method published by the Jolly group.

Reference: Cancer-Systems-Biology-Lab/EMT_Scoring_RNASeq (MLR3_Code/MLR3_automated.m, MLR3.m,
EMT_score_func.R, counts_to_TPM.R), and the George et al. supplement ("MLR Model" sheet). The
reference files (gene list and RelevantData.mat with the NCI-60 data) are read from
data/raw/mlr_reference/, and the GPL570 annotation from data/raw/external_cohorts/GPL570.annot.gz.

The published method, as reproduced here:
  1. RNA-seq only: log2(TPM + 1) is mapped to a microarray-like scale, MA = 0.57 + 0.37 * log2TPM
     (rnaToMA in counts_to_TPM.R). Microarray data enter on their log2 scale unchanged.
  2. Normalisation: one offset per cohort, d = mean(normaliser genes over all samples of the cohort) -
     mean(the same genes over the 59 NCI-60 training lines), subtracted from every value. No per-gene
     or per-sample rescaling.
  3. Predictors: X1 = CLDN7, X2 = VIM / CDH1 (a ratio of the log-scale values).
  4. Ordinal logit with the published coefficients (alpha1, alpha2, beta1, beta2) =
     (-7.8714, 0.0413, 1.3571, -1.9566), as MATLAB mnrval(..., 'model', 'ordinal'):
     P(E) = s(a1 + b.x), P(E or H) = s(a2 + b.x), P(M) = 1 - P(E or H).
  5. Score: P(H) if P(E) > P(M), else 2 - P(H), on 0 (epithelial) to 2 (mesenchymal) with 1 = hybrid.

Two departures from the reference code, both towards the published method:
  * The R wrapper locates NCI-60 rows by their row number in a GPL570 annotation table, but
    DataNCI60 is stored in a different probe order (checked: 0 of the first 1,000 rows coincide), so
    the as-coded NCI-60 mean averages unrelated probes. Here each normaliser gene is located in
    DataNCI60 by probe ID (its first GPL570 probe), which is what the method describes.
  * The MATLAB code takes rows 6-25 of whatever genes were found, so a missing gene shifts which genes
    are treated as normalisers, and the NCI-60 rows can then refer to different genes from the cohort
    rows. Here the normalisers are the first 20 genes of the published normaliser list present in both
    the cohort and NCI-60, and the same genes are averaged on both sides.

nci60_norm_as_coded() returns the as-coded NCI-60 mean for comparison with the first departure.

Usage:   from pemt.mlr_faithful import mlr_faithful
"""

from __future__ import annotations

import gzip
import io
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# src/pemt/ is two levels below the repository root.
_ROOT = Path(__file__).resolve().parents[2]
REF = _ROOT / "data" / "raw" / "mlr_reference"
GPL570_ANNOT = _ROOT / "data" / "raw" / "external_cohorts" / "GPL570.annot.gz"
A1, A2, B1, B2 = -7.87139071184979, 0.0412975537779503, 1.35705632225475, -1.95661140686887
PREDICTORS = ["CLDN7", "VIM", "CDH1", "GRHL2", "OVOL1"]
N_NORMALISERS = 20
NCI60_DROP_COLUMN = 33          # MATLAB NCI_data(:, 34) = [] (1-based), as in MLR3_automated.m


def gene_list() -> list[str]:
    return [g.strip() for g in (REF / "genes_for_EMT_score.txt").read_text().splitlines() if g.strip()]


@lru_cache(maxsize=1)
def _gpl570_first_probe() -> dict[str, str]:
    with gzip.open(GPL570_ANNOT, "rt", errors="replace") as fh:
        lines = fh.readlines()
    s = next(i for i, l in enumerate(lines) if l.startswith("ID\t"))
    e = next((i for i, l in enumerate(lines) if l.startswith("!platform_table_end")), len(lines))
    ann = pd.read_csv(io.StringIO("".join(lines[s:e])), sep="\t", dtype=str)
    first: dict[str, str] = {}
    for pid, sym in zip(ann["ID"], ann["Gene symbol"]):
        if isinstance(sym, str) and sym not in first:
            first[sym] = pid
    rownum = {sym: int(np.where(ann["ID"].values == pid)[0][0]) for sym, pid in first.items()
              if sym in set(gene_list())}
    _gpl570_first_probe.rownum = rownum          # type: ignore[attr-defined]
    return first


@lru_cache(maxsize=1)
def _nci60() -> tuple[np.ndarray, np.ndarray]:
    import scipy.io as sio
    m = sio.loadmat(REF / "RelevantData.mat")
    labels = np.array([str(x[0]) if len(x) else "" for x in m["LabelsNCI60"].ravel()])
    data = np.delete(m["DataNCI60"], NCI60_DROP_COLUMN, axis=1)
    return labels, data


def nci60_gene_rows(genes: list[str]) -> dict[str, np.ndarray]:
    """NCI-60 expression (59 lines) for each gene, found by its first GPL570 probe ID."""
    first = _gpl570_first_probe()
    labels, data = _nci60()
    pos = {l: i for i, l in enumerate(labels) if "(" not in l}
    out = {}
    for g in genes:
        pid = first.get(g)
        if pid in pos:
            row = data[pos[pid]]
            if np.isfinite(row).all():
                out[g] = row
    return out


def normalisers_for(cohort_genes: set[str]) -> list[str]:
    """First 20 published normaliser genes present in both the cohort and NCI-60."""
    nci = nci60_gene_rows(gene_list()[len(PREDICTORS):])
    use = [g for g in gene_list()[len(PREDICTORS):] if g in cohort_genes and g in nci]
    return use[:N_NORMALISERS]


def nci60_norm_as_coded() -> float:
    """The NCI-60 normaliser mean exactly as MLR3_automated.m computes it, row numbers and all."""
    _gpl570_first_probe()
    rownum = _gpl570_first_probe.rownum          # type: ignore[attr-defined]
    _, data = _nci60()
    idx = [rownum[g] for g in gene_list() if g in rownum]
    return float(np.nanmean(data[idx][5:25]))


def ordinal_probs(x1: np.ndarray, x2: np.ndarray) -> pd.DataFrame:
    s = lambda v: 1.0 / (1.0 + np.exp(-v))  # noqa: E731
    eta = B1 * x1 + B2 * x2
    pe, pem = s(A1 + eta), s(A2 + eta)
    return pd.DataFrame({"P_E": pe, "P_H": pem - pe, "P_M": 1.0 - pem})


def score_from_probs(p: pd.DataFrame) -> np.ndarray:
    return np.where(p["P_E"] > p["P_M"], p["P_H"], 2.0 - p["P_H"])


def mlr_faithful(expr: pd.DataFrame, rnaseq: bool) -> pd.DataFrame:
    """Score every sample of a cohort.

    expr is genes x samples on a log2 scale (log2(TPM+1) for RNA-seq). Returns one row per sample
    with P_E, P_H, P_M, the MLR score (column MLR_faithful) and the most probable state (E, H or M).
    The offset d and the normaliser genes are stored in the frame's attrs."""
    e = expr.copy()
    if rnaseq:
        e = 0.57 + 0.37 * e
    missing = [g for g in ("CLDN7", "VIM", "CDH1") if g not in e.index]
    if missing:
        raise ValueError(f"MLR predictors missing: {missing}")
    norm = normalisers_for(set(e.index))
    nci = nci60_gene_rows(norm)
    d = float(e.loc[norm].values.mean() - np.mean([nci[g].mean() for g in norm]))
    n = e - d
    p = ordinal_probs(n.loc["CLDN7"].values, (n.loc["VIM"] / n.loc["CDH1"]).values)
    p.index = e.columns
    p["MLR_faithful"] = score_from_probs(p)
    p["state"] = p[["P_E", "P_H", "P_M"]].idxmax(axis=1).str[2]
    p.attrs.update({"offset_d": d, "normalisers": norm, "n_normalisers": len(norm)})
    return p


def validate_on_nci60() -> pd.DataFrame:
    """Apply the model to the training lines themselves (no normalisation needed: d = 0)."""
    rows = nci60_gene_rows(["CLDN7", "VIM", "CDH1"])
    p = ordinal_probs(rows["CLDN7"], rows["VIM"] / rows["CDH1"])
    p["MLR_faithful"] = score_from_probs(p)
    p["state"] = p[["P_E", "P_H", "P_M"]].idxmax(axis=1).str[2]
    return p
