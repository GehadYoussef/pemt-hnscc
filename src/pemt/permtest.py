"""Permutation tests over cell lines for compound-set and gene-set effects.

Compounds screened in the same cell lines give Spearman rho values that are not independent, so a
Mann-Whitney test of one compound class against all other compounds gives p-values that are too
small. The exchangeable unit here is the cell line. The phenotype is permuted across lines and the
whole set statistic is recomputed, which keeps the correlation structure between compounds intact.

Statistic. AUC (or gene effect) is ranked within each compound across the lines where it was measured.
The phenotype is ranked across all matched lines. rho_j is the correlation of those two rank vectors
over the lines where compound j has data. A set statistic is mean(rho_j) over the set. Observed and
permuted statistics use identical code, so the test is exact under the label-permutation null.

Usage:   from pemt.permtest import permutation_setup, set_statistic
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def rho_matrix(values: pd.DataFrame, phen_ranks: np.ndarray) -> np.ndarray:
    """Spearman rho of every column of `values` against every phenotype rank vector.

    values      lines x features, NaN where the feature was not measured in that line
    phen_ranks  lines x k phenotype rank vectors (k = 1 observed, or many permuted)
    returns     features x k
    """
    v = values.values.astype(float)
    mask = (~np.isnan(v)).astype(float)
    r = np.full(v.shape, np.nan)
    for j in range(v.shape[1]):
        obs = ~np.isnan(v[:, j])
        r[obs, j] = rankdata(v[obs, j])
    r = np.nan_to_num(r)
    n = mask.sum(axis=0)
    mu = (r * mask).sum(axis=0) / n
    rc = (r - mu) * mask
    sd = np.sqrt((rc ** 2).sum(axis=0) / n)
    sd[sd == 0] = np.nan
    rc = rc / sd

    p = phen_ranks
    pmu = (mask.T @ p) / n[:, None]
    pvar = (mask.T @ (p ** 2)) / n[:, None] - pmu ** 2
    psd = np.sqrt(np.clip(pvar, 0, None))
    psd[psd == 0] = np.nan
    cov = (rc.T @ p) / n[:, None]      # rc is centred within each feature, so the phenotype mean drops out
    return cov / psd


def permutation_setup(values: pd.DataFrame, phenotype: np.ndarray, n_perm: int, seed: int):
    """Observed rho per feature and the null rho matrix, sharing one permutation draw.

    Returns (observed rho vector, features x n_perm null matrix, {feature name: row index}).
    """
    obs = rho_matrix(values, rankdata(phenotype)[:, None])[:, 0]
    rng = np.random.default_rng(seed)
    perm = np.column_stack([rng.permutation(rankdata(phenotype)) for _ in range(n_perm)])
    null = rho_matrix(values, perm)
    return obs, null, {c: i for i, c in enumerate(values.columns)}


def set_statistic(obs: np.ndarray, null: np.ndarray, rows: list[int]) -> dict:
    """Mean rho over a set of features, with its permutation null, z and two-sided empirical p."""
    stat = float(np.nanmean(obs[rows]))
    nulls = np.nanmean(null[rows, :], axis=0)
    nm, ns = float(nulls.mean()), float(nulls.std(ddof=1))
    n_perm = null.shape[1]
    return {"n_features": len(rows), "mean_rho": round(stat, 4), "null_mean": round(nm, 4),
            "null_sd": round(ns, 4), "z": round((stat - nm) / ns if ns > 0 else 0.0, 2),
            "perm_p": round(float((np.sum(np.abs(nulls - nm) >= abs(stat - nm)) + 1) / (n_perm + 1)), 4),
            "_nulls": nulls}
