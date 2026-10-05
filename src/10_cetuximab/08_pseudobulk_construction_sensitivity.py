"""Compare summed (sum-then-log) and mean-of-log construction of the training pseudobulks.

02_multinomial_pemt_model/02_create_pseudobulk_mixtures.py builds each training pseudobulk the
bulk-like way, summing the linear counts per 10,000 of the sampled cells, renormalising to 10,000 and
taking log1p, and the main classifier is trained on these. The alternative tested here is the mean of
the sampled cells' log-normalised profiles.

  1. Re-draws the training mixtures with the same seed and the same sampling calls as
     02_create_pseudobulk_mixtures.py, so every mixture contains exactly the same cells, and
     aggregates them both ways. Both versions are checked against the matrices saved by stage
     02-02 (pseudobulk_expression.tsv, summed; pseudobulk_expression_meanlog.tsv, mean of log) to
     confirm that the draws are identical.
  2. Grouped (leave-one-dataset-out) CV for both constructions, evaluated two ways: as in
     03_train_multinomial_classifier.py (training-fold StandardScaler), and matched to deployment
     (held-out fold z-scored within itself, then only the fitted logistic layer applied, as bulk
     cohorts are scored).
  3. Refits the classifier on the mean-of-log pseudobulks (the earlier construction) and projects
     it into the TCGA-HNSC primaries and GSE65021 with within-cohort z-scoring. Reports the overlap
     of the top 100 pEMT-high coefficients with the main (summed) classifier, the Spearman
     correlation of the resulting pEMT specificity with the main classifier's, and its AUC for long
     PFS in GSE65021 with a 2,000-sample bootstrap interval, beside the main classifier's AUC.

The run takes about 10 minutes, most of it in the SAGA fits.

Inputs:  config/config.yaml, config/dataset_registry.tsv
         data/processed/single_cell/<dataset>/<dataset>_labelled.h5ad (training datasets)
         data/processed/pseudo_bulk/pseudobulk_expression.tsv (summed, as trained)
         data/processed/pseudo_bulk/pseudobulk_expression_meanlog.tsv
         results/multinomial_classifier/tcga_depmap_ready_classifier.pkl
         results/multinomial_classifier/classifier_genes.txt
         data/processed/tcga_hnsc/tcga_star_tpm_log2_for_projection.tsv
         results/tcga_projection/emt_score_panel_scores.tsv
         data/raw/external_cohorts/GSE65021_series_matrix.txt.gz
         data/raw/external_cohorts/GPL10558.annot.gz
         results/cetuximab_cohort/GSE65021_scores.tsv
Outputs: results/pseudobulk_sensitivity/cv_by_construction.tsv
         results/pseudobulk_sensitivity/summed_construction_summary.tsv
Usage:   python src/10_cetuximab/08_pseudobulk_construction_sensitivity.py
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pemt import load_config, load_registry, nan_guard  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
cfg = load_config()
PB, CLF = cfg["pseudobulk"], cfg["classifier"]
SC_DIR = ROOT / cfg["paths"]["processed_dir"] / "single_cell"
PB_SAVED = ROOT / cfg["paths"]["processed_dir"] / "pseudo_bulk" / "pseudobulk_expression.tsv"
PB_SAVED_ML = ROOT / cfg["paths"]["processed_dir"] / "pseudo_bulk" / "pseudobulk_expression_meanlog.tsv"
CLF_DIR = ROOT / cfg["paths"]["results_dir"] / "multinomial_classifier"
OUT = ROOT / "results" / "pseudobulk_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)
CLASSES = ["pEMT_high", "epithelial_like", "fibroblast_stromal_like"]


def build_both() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Repeat the sampling of 02_create_pseudobulk_mixtures.py call for call.

    Returns (mean-of-log, summed, metadata).
    """
    import scanpy as sc

    rng = np.random.default_rng(cfg["random_seed"])
    reg = load_registry()
    mean_log, summed, labels, sources = [], [], [], []
    for ds in reg.loc[reg["use_for_training"] == "yes", "dataset_id"].tolist():
        path = SC_DIR / ds / f"{ds}_labelled.h5ad"
        if not path.exists():
            continue
        adata = sc.read_h5ad(path)
        adata = adata[adata.obs["training_label"].isin(CLASSES)].copy()
        if adata.n_obs < PB["min_labelled_cells"]:
            continue
        X = adata.to_df()
        lin = np.expm1(X.values)
        y = adata.obs["training_label"].astype(str).values
        idx = {c: np.where(y == c)[0] for c in CLASSES}
        if any(len(idx[c]) < PB["min_cells_per_class"] for c in CLASSES):
            continue
        for _ in range(PB["n_mixtures_per_dataset"]):
            props = rng.dirichlet([1, 1, 1])
            dom = int(np.argmax(props))
            if props[dom] < PB["min_class_fraction_for_label"]:
                continue
            rows = []
            for c, p in zip(CLASSES, props):
                n = max(1, int(round(p * PB["cells_per_mixture"])))
                rows.append(rng.choice(idx[c], size=n, replace=True))
            rows = np.concatenate(rows)
            mean_log.append(pd.Series(X.values[rows].mean(axis=0), index=X.columns))
            s = lin[rows].sum(axis=0)
            summed.append(pd.Series(np.log1p(s / s.sum() * 1e4), index=X.columns))
            labels.append(CLASSES[dom])
            sources.append(ds)
    meta = pd.DataFrame({"label": labels, "source_dataset": sources},
                        index=[f"PB_{i:06d}" for i in range(len(labels))])
    ml, sm = pd.DataFrame(mean_log), pd.DataFrame(summed)
    ml.index = sm.index = meta.index
    return ml, sm, meta


def model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    return Pipeline([("scaler", StandardScaler()), ("nan_guard", FunctionTransformer(nan_guard)),
                     ("clf", LogisticRegression(solver=CLF["solver"], penalty="elasticnet", l1_ratio=CLF["l1_ratio"],
                                                C=CLF["C"], tol=CLF["tol"], max_iter=CLF["max_iter"]))])


def variance_filter(X: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    Xg = X.copy()
    Xg["__g__"] = groups.values
    mv = Xg.groupby("__g__")[X.columns].var().min(axis=0)
    return X.loc[:, (X.var(axis=0) > CLF["variance_floor"]) & (mv > 0)]


def zscore(X: pd.DataFrame) -> np.ndarray:
    return np.nan_to_num(((X - X.mean()) / X.std().replace(0, 1.0)).values)


def cv(X: pd.DataFrame, meta: pd.DataFrame, name: str) -> list[dict]:
    from sklearn.model_selection import GroupKFold
    y, g = meta["label"], meta["source_dataset"]
    rows = []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=g.nunique()).split(X, y, g), 1):
        m = model().fit(X.iloc[tr], y.iloc[tr])
        cls = list(m.named_steps["clf"].classes_)
        for mode, probs in [("training scaler (as in 03_train_multinomial_classifier.py)", m.predict_proba(X.iloc[te])),
                            ("deployment-matched (within-fold z-score)",
                             m.named_steps["clf"].predict_proba(zscore(X.iloc[te])))]:
            r = {"construction": name, "fold": fold, "held_out": g.iloc[te].iloc[0], "evaluation": mode,
                 "accuracy": accuracy_score(y.iloc[te], np.array(cls)[probs.argmax(1)])}
            for c in cls:
                r[f"auc_{c}"] = roc_auc_score((y.iloc[te] == c).astype(int), probs[:, cls.index(c)])
            rows.append(r)
        print(f"  {name} fold {fold} done")
    return rows


def project(expr_gxs: pd.DataFrame, clf, genes: list[str], ref_samples=None) -> pd.Series:
    X = expr_gxs.T
    Xf = pd.DataFrame(0.0, index=X.index, columns=genes)
    shared = [g for g in genes if g in X.columns]
    Xf[shared] = X[shared]
    ref = Xf.loc[ref_samples] if ref_samples is not None else Xf
    Z = np.nan_to_num(((Xf - ref.mean()) / ref.std().replace(0, 1.0)).values)
    p = pd.DataFrame(clf.predict_proba(Z), index=X.index, columns=list(clf.classes_))
    return p["pEMT_high"] - p[["epithelial_like", "fibroblast_stromal_like"]].max(axis=1)


def main() -> None:
    mean_log, summed, meta = build_both()
    for path, rebuilt, what in [(PB_SAVED, summed, "summed training matrix"),
                                (PB_SAVED_ML, mean_log, "mean-of-log matrix")]:
        saved = pd.read_csv(path, sep="\t", index_col=0)
        assert list(saved.columns) == list(rebuilt.columns), f"gene order differs from the saved {what}"
        diff = float(np.nanmax(np.abs(saved.values - rebuilt.values)))
        print(f"{len(meta)} mixtures rebuilt, max |difference| from the saved {what} = {diff:.2e}")
        assert diff < 1e-4, f"re-drawn mixtures do not match the saved {what}"

    rows = []
    for name, X in [("sum then log (as trained, primary)", summed),
                    ("mean of log (earlier construction)", mean_log)]:
        rows += cv(variance_filter(X, meta["source_dataset"]), meta, name)
    cvres = pd.DataFrame(rows)
    cvres.to_csv(OUT / "cv_by_construction.tsv", sep="\t", index=False)
    print(cvres.groupby(["construction", "evaluation"])[["accuracy", "auc_pEMT_high"]].mean().round(3))

    Xs = variance_filter(mean_log, meta["source_dataset"])
    m = model().fit(Xs, meta["label"])
    clf, genes = m.named_steps["clf"], list(Xs.columns)
    coef = pd.Series(clf.coef_[list(clf.classes_).index("pEMT_high")], index=genes)
    old = __import__("joblib").load(CLF_DIR / "tcga_depmap_ready_classifier.pkl").named_steps["clf"]
    old_genes = pd.read_csv(CLF_DIR / "classifier_genes.txt", header=None)[0].tolist()
    old_coef = pd.Series(old.coef_[list(old.classes_).index("pEMT_high")], index=old_genes)
    top_new, top_old = set(coef.nlargest(100).index), set(old_coef.nlargest(100).index)

    out = {"n_mixtures": len(meta), "top100_pEMT_high_overlap_with_main": len(top_new & top_old),
           "nonzero_genes_meanlog_classifier": int((np.abs(clf.coef_).sum(0) > 0).sum()),
           "nonzero_genes_main_classifier": int((np.abs(old.coef_).sum(0) > 0).sum())}

    tcga = pd.read_csv(ROOT / cfg["paths"]["processed_dir"] / "tcga_hnsc" / "tcga_star_tpm_log2_for_projection.tsv",
                       sep="\t", index_col=0)
    prim = [s for s in tcga.columns if s[13:15] == "01"]
    new_t = project(tcga[prim], clf, genes)
    pub_t = pd.read_csv(ROOT / "results" / "tcga_projection" / "emt_score_panel_scores.tsv", sep="\t", index_col=0)
    common = new_t.index.intersection(pub_t.index)
    out["TCGA_n"] = len(common)
    out["TCGA_rho_with_main_pEMT_specificity"] = spearmanr(new_t[common], pub_t.loc[common, "pEMT_specificity"])[0]

    spec = importlib.util.spec_from_file_location("cx", ROOT / "src" / "10_cetuximab" / "01_cetuximab_cohort.py")
    cx = importlib.util.module_from_spec(spec)
    sys.argv = [sys.argv[0]]
    spec.loader.exec_module(cx)
    expr, gmeta = cx.read_series_matrix(ROOT / "data" / "raw" / "external_cohorts" / "GSE65021_series_matrix.txt.gz")
    e = cx.collapse(expr, cx.read_annotation(ROOT / "data" / "raw" / "external_cohorts" / "GPL10558.annot.gz"))
    new_g = project(e, clf, genes)
    pub_g = pd.read_csv(ROOT / "results" / "cetuximab_cohort" / "GSE65021_scores.tsv", sep="\t", index_col=0)
    y = pub_g.loc[new_g.index, "long_pfs"].values
    out["GSE65021_rho_with_main_pEMT_specificity"] = spearmanr(new_g, pub_g.loc[new_g.index, "pEMT_specificity"])[0]
    out["GSE65021_AUC_main_summed"] = roc_auc_score(y, pub_g.loc[new_g.index, "pEMT_specificity"])
    out["GSE65021_AUC_meanlog_construction"] = roc_auc_score(y, new_g.values)
    rng = np.random.default_rng(20260925)
    draws = (rng.integers(0, len(y), len(y)) for _ in range(2000))
    lo, hi = np.percentile([roc_auc_score(y[i], new_g.values[i]) for i in draws if len(set(y[i])) == 2],
                           [2.5, 97.5])
    out["GSE65021_AUC_meanlog_low"], out["GSE65021_AUC_meanlog_up"] = lo, hi
    summary = pd.Series(out)
    summary.to_csv(OUT / "summed_construction_summary.tsv", sep="\t", header=["value"])
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
