"""Benchmark alternative classifiers on the same pseudobulk training set.

The main model is the multinomial elastic net of 03_train_multinomial_classifier.py, a sparse linear
model whose coefficients are applied to z-scored TCGA and DepMap profiles and read as gene weights.
It is compared here with a dense linear model and with tree ensembles. All models use the same
pseudobulks, the same variance filter and the same leave-one-dataset-out folds (GroupKFold over
source datasets):

  elastic_net        the main configuration (saga, l1_ratio, C from config)
  ridge              L2 multinomial logistic regression (same C)
  random_forest      500 trees, sqrt(features)
  lightgbm           gradient-boosted trees (300 rounds, learning rate 0.05, 31 leaves), if installed
  lightgbm_top2000   LightGBM on the 2,000 most variable genes of the training fold

Metrics per fold: accuracy, macro one-vs-rest AUROC, per-class AUROC and log loss, plus the Spearman
correlation between the elastic-net P(pEMT-high) and each other model's P(pEMT-high) on the held-out
fold (the TCGA and DepMap analyses use ranks). Supplementary_Figure_6 shows accuracy, AUROC for
pEMT-high, log loss and this Spearman correlation (panels a to d).

Inputs:  data/processed/pseudo_bulk/pseudobulk_expression.tsv,
         data/processed/pseudo_bulk/pseudobulk_metadata.tsv
Outputs: results/multinomial_classifier/classifier_benchmark_folds.tsv,
         results/multinomial_classifier/classifier_benchmark_summary.tsv,
         results/figures/Supplementary_Figure_6.svg and .png
Usage:   python src/02_multinomial_pemt_model/05_benchmark_classifiers.py
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pemt import load_config, nan_guard, project_root  # noqa: E402
from pemt.figstyle import apply_style, save_figure, PALETTE, mm  # noqa: E402

cfg = load_config()
ROOT = project_root()
CLF = cfg["classifier"]
PB_DIR = ROOT / cfg["paths"]["processed_dir"] / "pseudo_bulk"
OUT_DIR = ROOT / cfg["paths"]["results_dir"] / "multinomial_classifier"
FIG = ROOT / "results" / "figures"
apply_style()
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")


def models(n_features: int) -> dict:
    out = {
        "elastic_net": Pipeline([("scaler", StandardScaler()), ("nan_guard", FunctionTransformer(nan_guard)),
                                 ("clf", LogisticRegression(solver=CLF["solver"], penalty="elasticnet", l1_ratio=CLF["l1_ratio"], C=CLF["C"],
                                                            tol=CLF["tol"], max_iter=CLF["max_iter"]))]),
        "ridge": Pipeline([("scaler", StandardScaler()), ("nan_guard", FunctionTransformer(nan_guard)),
                           ("clf", LogisticRegression(solver="lbfgs", penalty="l2", C=CLF["C"], max_iter=2000))]),
        "random_forest": RandomForestClassifier(n_estimators=500, max_features="sqrt", n_jobs=-1, random_state=0),
    }
    try:
        import lightgbm as lgb
        out["lightgbm"] = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                                             colsample_bytree=0.3, min_child_samples=10, random_state=0, verbose=-1, n_jobs=-1)
        out["lightgbm_top2000"] = "top2000"
    except ImportError:
        print("lightgbm not installed, skipping (pip install lightgbm)")
    return out


def main() -> None:
    X = pd.read_csv(PB_DIR / "pseudobulk_expression.tsv", sep="\t", index_col=0)
    meta = pd.read_csv(PB_DIR / "pseudobulk_metadata.tsv", sep="\t", index_col=0).loc[X.index]
    groups = meta["source_dataset"]
    Xg = X.copy(); Xg["__group__"] = groups.values
    min_var_per_group = Xg.groupby("__group__")[X.columns].var().min(axis=0)
    X = X.loc[:, (X.var(axis=0) > CLF["variance_floor"]) & (min_var_per_group > 0)]
    y = meta["label"]
    classes = sorted(y.unique())
    print(f"{X.shape[0]} pseudobulks x {X.shape[1]} genes, datasets: {groups.value_counts().to_dict()}")
    cv = GroupKFold(n_splits=min(CLF["cv_n_splits"], groups.nunique()))
    folds = list(cv.split(X, y, groups))
    rows, ref_probs = [], {}
    for name, model in models(X.shape[1]).items():
        for fold, (tr, te) in enumerate(folds, 1):
            t0 = time.time()
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            if isinstance(model, str) and model == "top2000":
                import lightgbm as lgb
                top = Xtr.var(axis=0).sort_values(ascending=False).index[:2000]
                Xtr, Xte = Xtr[top], Xte[top]
                mdl = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                                         colsample_bytree=0.5, min_child_samples=10, random_state=0, verbose=-1, n_jobs=-1)
            else:
                mdl = model
            mdl.fit(Xtr, y.iloc[tr])
            probs = mdl.predict_proba(Xte)
            cls = list(mdl.classes_)
            P = pd.DataFrame(probs, index=Xte.index, columns=cls)[classes]
            pred = P.idxmax(axis=1)
            row = {"model": name, "fold": fold, "held_out_dataset": groups.iloc[te].iloc[0], "n_test": len(te), "fit_seconds": time.time() - t0,
                   "accuracy": accuracy_score(y.iloc[te], pred), "log_loss": log_loss(y.iloc[te], P.values, labels=classes),
                   "auroc_macro_ovr": roc_auc_score(y.iloc[te], P.values, multi_class="ovr", labels=classes, average="macro")}
            for c in classes:
                row[f"auroc_{c}"] = roc_auc_score((y.iloc[te] == c).astype(int), P[c])
            if name == "elastic_net":
                ref_probs[fold] = P["pEMT_high"]
            else:
                row["spearman_P_pEMT_high_vs_elastic_net"] = spearmanr(P["pEMT_high"], ref_probs[fold].loc[P.index])[0]
            rows.append(row)
            print(f"  {name} fold {fold} ({row['held_out_dataset']}): acc {row['accuracy']:.3f}, macro AUROC {row['auroc_macro_ovr']:.3f}, "
                  f"AUROC pEMT {row['auroc_pEMT_high']:.3f}, {row['fit_seconds']:.0f} s")
    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "classifier_benchmark_folds.tsv", sep="\t", index=False)
    num = [c for c in res.columns if c not in ("model", "fold", "held_out_dataset")]
    summ = res.groupby("model")[num].mean().reindex([m for m in ["elastic_net", "ridge", "random_forest", "lightgbm", "lightgbm_top2000"] if m in set(res["model"])])
    summ.to_csv(OUT_DIR / "classifier_benchmark_summary.tsv", sep="\t")
    print(summ.round(3).to_string())

    fig, axes = plt.subplots(1, 4, figsize=(mm(180), mm(58)))
    order = list(summ.index)
    for ax, metric, lab in zip(axes, ["accuracy", "auroc_pEMT_high", "log_loss", "spearman_P_pEMT_high_vs_elastic_net"],
                               ["Accuracy (held-out dataset)", "AUROC, pEMT-high vs rest", "Log loss (lower is better)", "Spearman ρ of P(pEMT-high)\nwith the elastic net"]):
        for i, m in enumerate(order):
            sub = res[res["model"] == m]
            ax.scatter([i] * len(sub), sub[metric], s=18, color=PALETTE["vermilion"] if m == "elastic_net" else PALETTE["blue"], zorder=2, linewidths=0)
            ax.plot([i - 0.25, i + 0.25], [summ.loc[m, metric]] * 2, color=PALETTE["black"], linewidth=1)
        short = {"elastic_net": "elastic\nnet", "ridge": "ridge", "random_forest": "random\nforest",
                 "lightgbm": "LGBM", "lightgbm_top2000": "LGBM\ntop2k"}
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([short.get(m, m.replace("_", "\n")) for m in order], fontsize=6)
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_ylabel(lab)
    axes[0].set_ylim(0.3, 1.0); axes[1].set_ylim(0.9, 1.005); axes[3].set_ylim(0.5, 1.0)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=5.5)
    fig.tight_layout()
    for ax, letter in zip(axes, "abcd"):
        ax.text(-0.30, 1.06, letter, transform=ax.transAxes, fontweight="bold", fontsize=8)
    fig.tight_layout()
    save_figure(fig, FIG, "Supplementary_Figure_6")


if __name__ == "__main__":
    main()
