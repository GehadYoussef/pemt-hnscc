"""Firth penalised logistic regression with profile penalised-likelihood inference.

With 14 events in the GSE65021 cetuximab cohort, ordinary maximum likelihood gives odds ratios of 20
to 60 per standard deviation in the adjusted and head-to-head models, a sign of quasi-separation.
Firth's correction (Jeffreys-prior penalty, 0.5 log|I(beta)|) removes the first-order bias and stays
finite under separation. Inference follows logistf (Heinze and Schemper 2002). The p-value is the
penalised likelihood-ratio test of beta_j = 0 and the interval is the profile penalised-likelihood
interval. Both behave better than Wald intervals at this sample size.

The fit is implemented here with Newton-Raphson and step halving. `selftest()` checks it against
direct numerical maximisation of the same penalised log-likelihood.

Usage:   from pemt.firth import firth
         python src/pemt/firth.py   (runs the self-test)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.stats import chi2


def _pll(beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    eta = X @ beta
    p = 1.0 / (1.0 + np.exp(-eta))
    ll = float(np.sum(y * eta - np.logaddexp(0.0, eta)))
    w = p * (1 - p)
    sign, logdet = np.linalg.slogdet(X.T @ (X * w[:, None]))
    return ll + 0.5 * logdet if sign > 0 else -np.inf


def _fit(X: np.ndarray, y: np.ndarray, fixed: dict[int, float] | None = None,
         max_iter: int = 500, tol: float = 1e-9, init: np.ndarray | None = None) -> np.ndarray:
    """Newton-Raphson on the Firth-modified score, with some coefficients held fixed (profiling)."""
    fixed = fixed or {}
    k = X.shape[1]
    free = np.array([j for j in range(k) if j not in fixed])
    beta = np.zeros(k) if init is None else np.array(init, float)
    for j, v in fixed.items():
        beta[j] = v
    cur = _pll(beta, X, y)
    for _ in range(max_iter):
        p = 1.0 / (1.0 + np.exp(-(X @ beta)))
        w = p * (1 - p)
        info = X.T @ (X * w[:, None])
        inv = np.linalg.pinv(info)
        h = np.einsum("ij,jk,ik->i", X * np.sqrt(w)[:, None], inv, X * np.sqrt(w)[:, None])
        score = X.T @ (y - p + h * (0.5 - p))
        step = np.linalg.solve(info[np.ix_(free, free)], score[free])
        # step halving keeps the penalised likelihood non-decreasing
        t = 1.0
        while True:
            cand = beta.copy()
            cand[free] += t * np.clip(step, -5, 5)
            new = _pll(cand, X, y)
            if new >= cur - 1e-12 or t < 1e-6:
                break
            t /= 2
        beta, delta, cur = cand, abs(new - cur), new
        if np.max(np.abs(t * step)) < tol and delta < tol:
            break
    return beta


def firth(y, X, names: list[str], alpha: float = 0.05) -> dict[str, tuple[float, float, float, float]]:
    """Fit a Firth logistic model. X must include the intercept column, named "const" in `names`.

    Returns {name: (OR, CI low, CI up, p)} for every column except the intercept, with profile
    penalised-likelihood intervals and penalised likelihood-ratio p-values."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    beta = _fit(X, y)
    beta = _fit(X, y, init=beta)   # polish
    full = _pll(beta, X, y)
    crit = chi2.ppf(1 - alpha, 1)
    out = {}
    for j, name in enumerate(names):
        if name == "const":
            continue

        def dev(b, j=j):
            return 2 * (full - _pll(_fit(X, y, {j: b}, init=beta), X, y))

        p = float(chi2.sf(max(dev(0.0), 0.0), 1))
        bounds = []
        for direction in (-1, 1):
            step, b = 0.5, beta[j]
            far = b + direction * step
            while dev(far) < crit and abs(far - b) < 50:
                step *= 2
                far = b + direction * step
            bounds.append(brentq(lambda v: dev(v) - crit, *sorted((b, far))) if dev(far) >= crit
                          else direction * np.inf)
        out[name] = (float(np.exp(beta[j])), float(np.exp(bounds[0])), float(np.exp(bounds[1])), p)
    return out


def selftest(seed: int = 1) -> float:
    """Maximum absolute difference between the Newton fit and a direct numerical maximisation."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(40), rng.normal(size=(40, 3))])
    y = (rng.random(40) < 1 / (1 + np.exp(-(X @ np.array([-0.6, 1.5, -0.8, 0.0]))))).astype(float)
    a = _fit(X, y)
    b = minimize(lambda v: -_pll(v, X, y), np.zeros(4), method="BFGS", options={"gtol": 1e-10}).x
    # a separated design: ML diverges, the Firth fit must stay finite
    Xs = np.column_stack([np.ones(20), np.r_[np.arange(10) - 10.0, np.arange(10) + 1.0]])
    ys = np.r_[np.zeros(10), np.ones(10)]
    assert np.isfinite(_fit(Xs, ys)).all()
    return float(np.max(np.abs(a - b)))


if __name__ == "__main__":
    print(f"Newton vs direct maximisation, max |diff| = {selftest():.2e}")
