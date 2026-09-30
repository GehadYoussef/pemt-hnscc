"""Build pseudobulk mixtures of labelled cells for classifier training.

For each training dataset (use_for_training = yes in config/dataset_registry.tsv), class proportions
are drawn n_mixtures_per_dataset (1,000) times from a flat Dirichlet(1, 1, 1). A draw is kept only when
its dominant class has at least min_class_fraction_for_label (0.60) of the mixture, and the mixture is
labelled by that class. Each kept mixture samples about cells_per_mixture (100) labelled cells with
replacement, in the drawn proportions (at least one cell per class), and averages their log1p
expression. A dataset is skipped if it has fewer than min_labelled_cells (50) labelled cells or fewer
than min_cells_per_class (10) in any class. All sizes come from the pseudobulk section of
config/config.yaml, and the random seed from random_seed.

Inputs:  data/processed/single_cell/<dataset>/<dataset>_labelled.h5ad, config/dataset_registry.tsv
Outputs: data/processed/pseudo_bulk/pseudobulk_expression.tsv,
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

    profiles, labels, sources = [], [], []

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
            sampled = []
            for c, p in zip(CLASSES, props):
                n = max(1, int(round(p * PB["cells_per_mixture"])))
                sampled.append(X.iloc[rng.choice(idx[c], size=n, replace=True)])
            profiles.append(pd.concat(sampled).mean(axis=0))
            labels.append(CLASSES[dom])
            sources.append(ds)

    pseudo = pd.DataFrame(profiles)
    meta = pd.DataFrame({
        "pseudo_bulk_id": [f"PB_{i:06d}" for i in range(len(labels))],
        "label": labels,
        "source_dataset": sources,
    }).set_index("pseudo_bulk_id")
    pseudo.index = meta.index

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pseudo.to_csv(OUT_DIR / "pseudobulk_expression.tsv", sep="\t")
    meta.to_csv(OUT_DIR / "pseudobulk_metadata.tsv", sep="\t")
    print("Pseudo-bulk mixtures:", pseudo.shape)


if __name__ == "__main__":
    main()
