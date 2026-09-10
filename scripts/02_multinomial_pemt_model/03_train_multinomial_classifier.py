"""Train multinomial elastic-net logistic regression on pseudobulks.

Performs grouped CV (groups = source dataset) for performance reporting,
then refits on the full data and writes the final model and gene list.
All hyperparameters come from config.classifier.
"""

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import load_config, project_root, nan_guard  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")  # numeric warnings only; UserWarnings stay visible
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import FunctionTransformer, StandardScaler  # noqa: E402

cfg = load_config()
ROOT = project_root()
CLF = cfg["classifier"]
PB_DIR = ROOT / cfg["paths"]["processed_dir"] / "pseudo_bulk"
OUT_DIR = ROOT / cfg["paths"]["results_dir"] / "multinomial_classifier"


def main() -> None:
    X = pd.read_csv(PB_DIR / "pseudobulk_expression.tsv", sep="\t", index_col=0)
    meta = pd.read_csv(PB_DIR / "pseudobulk_metadata.tsv", sep="\t", index_col=0)
    groups = meta["source_dataset"]

    # Variance filter: drop genes that are constant globally OR constant within
    # any source dataset (so grouped CV folds never get an all-zero column).
    Xg = X.copy()
    Xg["__group__"] = groups.values
    min_var_per_group = Xg.groupby("__group__")[X.columns].var().min(axis=0)
    X = X.loc[:, (X.var(axis=0) > CLF["variance_floor"]) & (min_var_per_group > 0)]
    print(f"Genes after variance filter: {X.shape[1]}")
    y = meta["label"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("nan_guard", FunctionTransformer(nan_guard)),
        # penalty is set explicitly (patched 9 Sep 2026): before scikit-learn 1.8
        # the default penalty is L2 and l1_ratio is silently ignored, which would
        # turn this into a ridge model on older installations.
        ("clf", LogisticRegression(
            solver=CLF["solver"],
            penalty="elasticnet",
            l1_ratio=CLF["l1_ratio"],
            C=CLF["C"],
            tol=CLF["tol"],
            max_iter=CLF["max_iter"],
        )),
    ])

    rows = []
    if groups.nunique() >= 2:
        cv = GroupKFold(n_splits=min(CLF["cv_n_splits"], groups.nunique()))
        for fold, (tr, te) in enumerate(cv.split(X, y, groups), 1):
            model.fit(X.iloc[tr], y.iloc[tr])
            pred = model.predict(X.iloc[te])
            probs = model.predict_proba(X.iloc[te])
            classes = list(model.named_steps["clf"].classes_)
            row = {"fold": fold, "accuracy": accuracy_score(y.iloc[te], pred)}
            for cls in classes:
                row[f"auc_{cls}"] = roc_auc_score(
                    (y.iloc[te] == cls).astype(int), probs[:, classes.index(cls)]
                )
            rows.append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / "grouped_cv_performance.tsv", sep="\t", index=False)

    model.fit(X, y)
    joblib.dump(model, OUT_DIR / "tcga_depmap_ready_classifier.pkl")
    pd.Series(X.columns).to_csv(OUT_DIR / "classifier_genes.txt", index=False, header=False)
    print("Saved classifier and gene list to", OUT_DIR)


if __name__ == "__main__":
    main()
