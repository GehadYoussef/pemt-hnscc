"""Normalise, log1p and subset each dataset to malignant and fibroblast_stromal cells.

Cells with fewer than min_genes_per_cell (200) detected genes and genes detected in fewer than
min_cells_per_gene (10) cells are removed. Counts are normalised to normalize_target_sum (10,000) per
cell and log1p-transformed. All thresholds come from the single_cell section of config/config.yaml.

Inputs:  data/processed/single_cell/<dataset>/<dataset>.h5ad
Outputs: data/processed/single_cell/<dataset>/<dataset>_malignant_fibroblast.h5ad
Usage:   python src/01_single_cell_qc/03_filter_malignant_and_fibroblast_cells.py
"""

import sys
from pathlib import Path

import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pemt import load_config, load_registry, project_root  # noqa: E402

cfg = load_config()
ROOT = project_root()
SC = cfg["single_cell"]


def main() -> None:
    registry = load_registry()
    sc_root = ROOT / cfg["paths"]["processed_dir"] / "single_cell"

    for ds in registry["dataset_id"]:
        in_path = sc_root / ds / f"{ds}.h5ad"
        if not in_path.exists():
            continue

        adata = sc.read_h5ad(in_path)
        sc.pp.filter_cells(adata, min_genes=SC["min_genes_per_cell"])
        sc.pp.filter_genes(adata, min_cells=SC["min_cells_per_gene"])
        sc.pp.normalize_total(adata, target_sum=SC["normalize_target_sum"])
        sc.pp.log1p(adata)

        keep = adata.obs["standard_cell_type"].isin(["malignant", "fibroblast_stromal"])
        subset = adata[keep].copy()

        out = sc_root / ds / f"{ds}_malignant_fibroblast.h5ad"
        subset.write(out)
        print(ds, subset.obs["standard_cell_type"].value_counts().to_dict())


if __name__ == "__main__":
    main()
