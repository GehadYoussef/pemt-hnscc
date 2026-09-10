"""Build pseudo-bulk mixtures for classifier training.

Per training dataset, sample n Dirichlet-weighted mixtures of labelled cells.
A mixture is kept and labelled by its dominant class only when that class has
at least min_class_fraction_for_label of the mixture. All sizing parameters
come from config.pseudobulk.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import load_config, load_registry, project_root  # noqa: E402

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
