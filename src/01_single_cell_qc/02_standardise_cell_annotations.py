"""Standardise cell-type annotations across TISCH2 datasets.

Each cell gets a standard_cell_type (malignant, fibroblast_stromal, endothelial,
nonmalignant_epithelial, immune, proliferating_T_or_ambiguous, other_stromal_unspecified,
missing_annotation or other). Fine-lineage labels (Celltype major and minor lineage) override the
broad malignancy label, so keratinocytes, endothelial cells, ductal cells and proliferating T cells
that some TISCH datasets place under broad "Stromal cells" are not counted as fibroblast_stromal.

Inputs:  data/processed/single_cell/<dataset>/<dataset>.h5ad
Outputs: the same .h5ad files, updated in place with obs["standard_cell_type"]
Usage:   python src/01_single_cell_qc/02_standardise_cell_annotations.py
"""

import sys
from pathlib import Path

import pandas as pd
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pemt import load_registry, project_root, load_config  # noqa: E402

cfg = load_config()
ROOT = project_root()

FINE_IDENTITY_COLS = [
    "Celltype (major-lineage)",
    "Celltype (minor-lineage)",
    "Celltype",
    "Celltype (original)",
]
BROAD_IDENTITY_COLS = ["Celltype (malignancy)"]

IMMUNE_TERMS = (
    "immune", "t cell", "b cell", "macrophage", "monocyte", "myeloid",
    "dendritic", "plasma", "mast", "neutrophil", "mono/macro", "mono.macro",
    "cd4", "cd8", "tconv", "tex", "tcm", "tem", "treg",
)
# Short lineage codes are matched as whole tokens only. A substring test on "b",
# "dc", "nk", "m1" or "m2" would relabel any cell whose annotation contains
# those letters.
IMMUNE_TOKENS = {"nk", "dc", "pdc", "cdc1", "cdc2", "m1", "m2", "b", "tprolif"}


def _has_immune_token(text: str) -> bool:
    import re as _re
    return any(tok in IMMUNE_TOKENS for tok in _re.split(r"[^a-z0-9]+", text))


def _norm(x) -> str:
    return str(x).strip().lower() if pd.notna(x) else ""


def _joined(row, cols) -> str:
    return " ".join(_norm(row[c]) for c in cols if c in row.index and _norm(row[c]))


def standard_category(row) -> str:
    fine = _joined(row, FINE_IDENTITY_COLS)
    broad = _joined(row, BROAD_IDENTITY_COLS)

    if any(x in fine for x in ("fibroblast", "myofibroblast", "caf")):
        return "fibroblast_stromal"
    if "endothelial" in fine:
        return "endothelial"
    if any(x in fine for x in ("keratinocyte", "ductal", "gland", "epithelial")):
        return "nonmalignant_epithelial"
    if any(x in fine for x in ("tprolif", "proliferating t")):
        return "proliferating_T_or_ambiguous"
    if any(x in fine for x in IMMUNE_TERMS) or _has_immune_token(fine) or "immune cells" in broad:
        return "immune"
    if any(x in fine for x in ("malignant", "cancer cell", "carcinoma")) or "malignant cells" in broad:
        return "malignant"
    if "stromal cells" in broad:
        return "other_stromal_unspecified"
    if not (fine or broad):
        return "missing_annotation"
    return "other"


def main() -> None:
    registry = load_registry()
    sc_dir = ROOT / cfg["paths"]["processed_dir"] / "single_cell"

    for ds in registry["dataset_id"]:
        f = sc_dir / ds / f"{ds}.h5ad"
        if not f.exists():
            continue
        adata = sc.read_h5ad(f)
        adata.obs["standard_cell_type"] = adata.obs.apply(standard_category, axis=1)
        adata.write(f)
        print(ds, adata.obs["standard_cell_type"].value_counts().to_dict())


if __name__ == "__main__":
    main()
