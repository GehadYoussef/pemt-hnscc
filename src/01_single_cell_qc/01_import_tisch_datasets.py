"""Import TISCH2 single-cell datasets to AnnData (.h5ad).

For each dataset in config/dataset_registry.tsv, the expression file in data/raw/single_cell/<dataset>/
is read (a 10x .h5 file, or a delimited genes x cells table), and the TISCH2 cell metadata table
(CellMetainfo_table.tsv) is attached to the cells. Datasets without metadata or an expression file are
skipped.

Inputs:  data/raw/single_cell/<dataset>/ (expression file and CellMetainfo_table.tsv),
         config/dataset_registry.tsv
Outputs: data/processed/single_cell/<dataset>/<dataset>.h5ad
Usage:   python src/01_single_cell_qc/01_import_tisch_datasets.py
"""

import sys
from pathlib import Path

import pandas as pd
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pemt import load_config, load_registry, project_root  # noqa: E402

cfg = load_config()
ROOT = project_root()
META_NAME = cfg["single_cell"]["metadata_file_name"]


def find_expression_file(folder: Path) -> Path | None:
    candidates = []
    for pat in ("*expression*", "*matrix*", "*.mtx", "*.h5", "*.tsv", "*.csv"):
        candidates.extend(folder.glob(pat))
    candidates = [p for p in candidates if "CellMetainfo" not in p.name]
    return candidates[0] if candidates else None


def main() -> None:
    registry = load_registry()
    raw_dir = ROOT / "data" / "raw" / "single_cell"
    out_root = ROOT / cfg["paths"]["processed_dir"] / "single_cell"

    for ds in registry["dataset_id"]:
        in_dir = raw_dir / ds
        out_dir = out_root / ds
        out_dir.mkdir(parents=True, exist_ok=True)

        meta_file = in_dir / META_NAME
        expr_file = find_expression_file(in_dir)

        if not meta_file.exists():
            print(f"{ds}: missing {META_NAME}, skipping")
            continue
        if expr_file is None:
            print(f"{ds}: metadata found, expression missing, skipping")
            continue

        meta = pd.read_csv(meta_file, sep="\t")
        if expr_file.suffix.lower() == ".h5":
            adata = sc.read_10x_h5(expr_file, gex_only=False)
            adata.var_names_make_unique()
        else:
            expr = pd.read_csv(expr_file, sep=None, engine="python", index_col=0)
            adata = sc.AnnData(expr.T)

        cell_col = next(
            (c for c in ("Cell", "cell", meta.columns[0]) if c in meta.columns),
            meta.columns[0],
        )
        meta = meta.set_index(cell_col)
        adata.obs = meta.reindex(adata.obs_names)
        adata.obs["dataset_id"] = ds
        out_path = out_dir / f"{ds}.h5ad"
        adata.write(out_path)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
