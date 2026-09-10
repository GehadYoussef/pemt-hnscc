"""Assign training labels to single cells.

Uses quantile thresholds on Puram pEMT and epithelial scores within malignant
cells, with hypoxia and cell-cycle exclusion. Fibroblast_stromal cells are
labelled directly. All thresholds come from config.training_labels.
"""

import sys
from pathlib import Path

import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import load_config, load_registry, project_root  # noqa: E402

cfg = load_config()
ROOT = project_root()
TL = cfg["training_labels"]
SC_DIR = ROOT / cfg["paths"]["processed_dir"] / "single_cell"


def main() -> None:
    registry = load_registry()
    for ds in registry["dataset_id"]:
        in_path = SC_DIR / ds / f"{ds}_scored.h5ad"
        if not in_path.exists():
            continue

        adata = sc.read_h5ad(in_path)
        adata.obs["training_label"] = "exclude"

        mal = adata.obs["standard_cell_type"] == "malignant"
        fib = adata.obs["standard_cell_type"] == "fibroblast_stromal"

        # Use Puram epi_dif_1 (the canonical epithelial-differentiation NMF
        # program) for consistency with using Puram throughout the exclusion
        # logic. Falls back to the small canonical 'epithelial' marker score
        # if puram_epi_dif_1_score has not been populated.
        if "puram_pemt_score" not in adata.obs:
            print(ds, "missing puram_pemt_score")
            continue
        if "puram_epi_dif_1_score" in adata.obs:
            epi_col = "puram_epi_dif_1_score"
        elif "epithelial_score" in adata.obs:
            epi_col = "epithelial_score"
        else:
            print(ds, "missing epithelial score (neither puram_epi_dif_1 nor epithelial)")
            continue

        pemt = adata.obs["puram_pemt_score"]
        epi = adata.obs[epi_col]

        if mal.sum() > 10:
            pemt_hi = pemt[mal].quantile(TL["pemt_high_quantile"])
            pemt_lo = pemt[mal].quantile(TL["pemt_low_quantile"])
            epi_hi = epi[mal].quantile(TL["epithelial_high_quantile"])

            mask_pemt = mal & (pemt >= pemt_hi)
            # Use Puram's own NMF programs for exclusion. These are by
            # construction more orthogonal to puram_pemt than generic markers,
            # because they were derived as separate factors from the same data.
            if "puram_hypoxia_score" in adata.obs:
                mask_pemt &= adata.obs["puram_hypoxia_score"] < adata.obs.loc[mal, "puram_hypoxia_score"].quantile(TL["hypoxia_exclusion_quantile"])
            if "puram_cell_cycle_score" in adata.obs:
                mask_pemt &= adata.obs["puram_cell_cycle_score"] < adata.obs.loc[mal, "puram_cell_cycle_score"].quantile(TL["cell_cycle_exclusion_quantile"])

            mask_epi = mal & (pemt <= pemt_lo) & (epi >= epi_hi)
            adata.obs.loc[mask_pemt, "training_label"] = "pEMT_high"
            adata.obs.loc[mask_epi, "training_label"] = "epithelial_like"

        adata.obs.loc[fib, "training_label"] = "fibroblast_stromal_like"

        out_path = SC_DIR / ds / f"{ds}_labelled.h5ad"
        adata.write(out_path)
        print(ds, adata.obs["training_label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
