"""Standardise cell-type annotations across TISCH2 datasets.

Fine-lineage labels (Celltype major/minor) override broad malignancy labels.
This is what protects fibroblast_stromal from absorbing keratinocytes,
endothelial cells, ductal cells, and proliferating T cells that some TISCH
datasets bucket under broad "Stromal cells".
"""

import sys
from pathlib import Path

import pandas as pd
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import load_registry, project_root, load_config  # noqa: E402

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
# Short lineage codes are matched as whole tokens only (patched 9 Sep 2026: the
# previous substring test on "b", "dc", "nk", "m1", "m2" would have relabelled
# any cell whose annotation merely contained those letters).
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
