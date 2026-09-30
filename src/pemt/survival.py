"""Shared survival helpers for the TCGA-HNSC analyses.

Covariate preparation, complete-case Cox designs, Cox fits and repeated k-fold
cross-validated concordance.

Site grouping: 'base of tongue' is oropharynx and is tested before the generic
'tongue' rule. The reference category is set explicitly to oropharynx. The
residual 'other' group (n = 3) is excluded from adjusted models.

Usage:   from pemt.survival import primary_tumours, prepare_covariates, design_matrix, fit_cox
"""

import numpy as np
import pandas as pd

STAGE_MAP = {"Stage I": 1, "Stage II": 2, "Stage III": 3, "Stage IVA": 4, "Stage IVB": 4, "Stage IVC": 4}
SITE_ORDER = ["oropharynx", "oral_cavity", "larynx_hypopharynx"]   # first = reference
SITE_LABEL = {"oropharynx": "Oropharynx", "oral_cavity": "Oral cavity", "larynx_hypopharynx": "Larynx / hypopharynx"}


def site_group(s) -> str | None:
    if pd.isna(s):
        return None
    s = str(s).lower()
    if any(x in s for x in ("base of tongue", "tonsil", "oropharynx")):
        return "oropharynx"
    if any(x in s for x in ("tongue", "floor of mouth", "mouth", "cheek", "gum", "lip", "palate", "retromolar", "mandible")):
        return "oral_cavity"
    if any(x in s for x in ("larynx", "supraglottis", "glottis", "hypopharynx")):
        return "larynx_hypopharynx"
    return "other"


def aliquot_type(sample_id: str) -> str:
    parts = sample_id.split("-")
    return parts[3][:2] if len(parts) >= 4 else "??"


def primary_tumours(traits: pd.DataFrame) -> pd.DataFrame:
    return traits[[aliquot_type(s) == "01" for s in traits.index]].copy()


def prepare_covariates(t: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(index=t.index)
    df["OS_time"] = pd.to_numeric(t["OS_time"], errors="coerce")
    df["OS_event"] = pd.to_numeric(t["OS_event"], errors="coerce")
    for c in score_cols:
        df[c] = pd.to_numeric(t[c], errors="coerce")
    df["stage_num"] = t["ajcc_pathologic_stage.diagnoses"].map(STAGE_MAP)
    df["age"] = pd.to_numeric(t["age_at_index.demographic"], errors="coerce")
    df["pack_years"] = pd.to_numeric(t["pack_years_smoked.exposures"], errors="coerce")
    df["site_group"] = t["tissue_or_organ_of_origin.diagnoses"].apply(site_group)
    df["hpv_positive"] = t["hpv_status"].map({"HPV+": 1, "HPV-": 0}) if "hpv_status" in t.columns else np.nan
    return df


def design_matrix(df: pd.DataFrame, score: str, covariates: list[str]) -> pd.DataFrame:
    """Complete-case design with explicit site reference (oropharynx)."""
    cols = ["OS_time", "OS_event", score] + [c for c in covariates if c != "site_group"]
    sub = df.dropna(subset=cols + (["site_group"] if "site_group" in covariates else [])).copy()
    sub = sub[sub["OS_time"] > 0]
    if "site_group" in covariates:
        sub = sub[sub["site_group"].isin(SITE_ORDER)]
        for s in SITE_ORDER[1:]:
            sub[f"site_{s}"] = (sub["site_group"] == s).astype(int)
        cols += [f"site_{s}" for s in SITE_ORDER[1:]]
    return sub[cols]


def fit_cox(design: pd.DataFrame, penalizer: float = 0.0):
    from lifelines import CoxPHFitter
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(design, duration_col="OS_time", event_col="OS_event")
    res = cph.summary[["coef", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].copy()
    res.columns = ["coef", "HR", "HR_low_95", "HR_up_95", "p_value"]
    res["n"] = len(design)
    res["events"] = int(design["OS_event"].sum())
    res["c_index"] = cph.concordance_index_
    return cph, res


def cv_cindex(design: pd.DataFrame, feature_sets: dict[str, list[str]], n_splits: int = 5, n_repeats: int = 20, seed: int = 42) -> pd.DataFrame:
    """Repeated k-fold out-of-sample concordance for several feature sets on the same rows."""
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    from sklearn.model_selection import RepeatedKFold
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    rows = []
    for name, feats in feature_sets.items():
        cis = []
        for tr, te in rkf.split(design):
            d_tr, d_te = design.iloc[tr], design.iloc[te]
            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(d_tr[["OS_time", "OS_event"] + feats], duration_col="OS_time", event_col="OS_event")
            risk = cph.predict_partial_hazard(d_te[feats])
            cis.append(concordance_index(d_te["OS_time"], -risk, d_te["OS_event"]))
        rows.append({"model": name, "n": len(design), "cv_c_index_mean": float(np.mean(cis)),
                     "cv_c_index_sd": float(np.std(cis)), "n_folds": len(cis)})
    return pd.DataFrame(rows)


COVARIATE_LABELS = {
    "stage_num": "Pathological stage (per stage)",
    "age": "Age at diagnosis (per year)",
    "pack_years": "Smoking (per pack-year)",
    "hpv_positive": "HPV positive (vs negative)",
    "site_oral_cavity": "Oral cavity (vs oropharynx)",
    "site_larynx_hypopharynx": "Larynx / hypopharynx (vs oropharynx)",
}
