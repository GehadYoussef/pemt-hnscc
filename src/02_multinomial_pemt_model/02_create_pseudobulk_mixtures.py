"""Build pseudobulk mixtures of labelled cells for classifier training.

For each training dataset (use_for_training = yes in config/dataset_registry.tsv), class proportions
are drawn n_mixtures_per_dataset (1,000) times from a flat Dirichlet(1, 1, 1). A draw is kept only when
its dominant class has at least min_class_fraction_for_label (0.60) of the mixture, and the mixture is
labelled by that class. Each kept mixture samples about cells_per_mixture (100) labelled cells with
replacement, in the drawn proportions (at least one cell per class).

Bulk expression is the logarithm of summed expression, so each pseudobulk sums the sampled cells'
linear counts per 10,000 (expm1 of the log1p matrices), renormalises the sum to 10,000 and takes
log1p (sum-then-log). The mean of the same cells' log1p profiles is computed from the same draws and
written as pseudobulk_expression_meanlog.tsv, the alternative construction tested by
10_cetuximab/08_pseudobulk_construction_sensitivity.py. Both matrices therefore contain the same cells.

A dataset is skipped if it has fewer than min_labelled_cells (50) labelled cells or fewer than
min_cells_per_class (10) in any class. All sizes come from the pseudobulk section of
config/config.yaml, and the random seed from random_seed.

Inputs:  data/processed/single_cell/<dataset>/<dataset>_labelled.h5ad, config/dataset_registry.tsv
Outputs: data/processed/pseudo_bulk/pseudobulk_expression.tsv (sum-then-log, used for training),
         data/processed/pseudo_bulk/pseudobulk_expression_meanlog.tsv (mean of log1p, same draws),
         data/processed/pseudo_bulk/pseudobulk_metadata.tsv
Usage:   python src/02_multinomial_pemt_model/02_create_pseudobulk_mixtures.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pemt import load_config, load_registry, project_root  # noqa: E402

cfg = load_config()
ROOT = project_root()
PB = cfg["pseudobulk"]
SC_DIR = ROOT / cfg["paths"]["processed_dir"] / "single_cell"
OUT_DIR = ROOT / cfg["paths"]["processed_dir"] / "pseudo_bulk"

CLASSES = ["pEMT_high", "epithelial_like", "fibroblast_stromal_like"]


def main() -> None:
    rng = np.random.default_rng(cfg["random_seed"])
    registry = load_registry()
    training_datasets = registry.loc[registry["use_for_training"] == "yes", "dataset_id"].tolist()

    summed, mean_log, labels, sources = [], [], [], []

    for ds in training_datasets:
        in_path = SC_DIR / ds / f"{ds}_labelled.h5ad"
        if not in_path.exists():
            continue
        adata = sc.read_h5ad(in_path)
        adata = adata[adata.obs["training_label"].isin(CLASSES)].copy()
        if adata.n_obs < PB["min_labelled_cells"]:
            print(ds, "too few labelled cells")
            continue

        X = adata.to_df()
        logX = X.values
        lin = np.expm1(logX)                 # counts per 10,000 (the matrices are log1p of CP10k)
        y = adata.obs["training_label"].astype(str).values
        idx = {c: np.where(y == c)[0] for c in CLASSES}

        if any(len(idx[c]) < PB["min_cells_per_class"] for c in CLASSES):
            print(ds, "not enough cells in all classes", {c: len(idx[c]) for c in CLASSES})
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
            s = lin[rows].sum(axis=0)
            summed.append(pd.Series(np.log1p(s / s.sum() * 1e4), index=X.columns))
            mean_log.append(pd.Series(logX[rows].mean(axis=0), index=X.columns))
            labels.append(CLASSES[dom])
            sources.append(ds)

    meta = pd.DataFrame({
        "pseudo_bulk_id": [f"PB_{i:06d}" for i in range(len(labels))],
        "label": labels,
        "source_dataset": sources,
    }).set_index("pseudo_bulk_id")
    pseudo = pd.DataFrame(summed)
    pseudo.index = meta.index
    pseudo_ml = pd.DataFrame(mean_log)
    pseudo_ml.index = meta.index

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pseudo.to_csv(OUT_DIR / "pseudobulk_expression.tsv", sep="\t")
    pseudo_ml.to_csv(OUT_DIR / "pseudobulk_expression_meanlog.tsv", sep="\t")
    meta.to_csv(OUT_DIR / "pseudobulk_metadata.tsv", sep="\t")
    print("Pseudo-bulk mixtures (sum-then-log, training matrix):", pseudo.shape)
    print("Mean-of-log matrix from the same draws written for the sensitivity analysis")


if __name__ == "__main__":
    main()
