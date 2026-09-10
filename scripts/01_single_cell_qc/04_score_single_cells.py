"""Score filtered single-cell datasets with pEMT and lineage signatures.

Default scope is the Puram anchor (HNSC_GSE103322) so the canonical reference
is developed and inspected before broader generalisation. All thresholds and
the signature dictionary come from config and signatures.yaml.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import load_config, load_registry, load_signatures, project_root  # noqa: E402

cfg = load_config()
ROOT = project_root()
SC_DIR = ROOT / cfg["paths"]["processed_dir"] / "single_cell"
RESULTS_DIR = ROOT / cfg["paths"]["results_dir"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=["puram_only", "training", "validation", "outgroup", "all"],
        default="puram_only",
    )
    parser.add_argument("--dataset", action="append", default=None)
    parser.add_argument("--output-suffix", default="scored")
    parser.add_argument(
        "--min-present-genes",
        type=int,
        default=cfg["single_cell"]["min_present_signature_genes"],
    )
    return parser.parse_args()


def _yes_no(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["yes", "true", "1"])


def select_datasets(registry: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    if args.dataset:
        requested = list(dict.fromkeys(args.dataset))
        missing = sorted(set(requested) - set(registry["dataset_id"]))
        if missing:
            raise ValueError(f"Unknown dataset_id: {missing}")
        return requested

    if args.scope == "puram_only":
        mask = (
            (registry["dataset_id"] == "HNSC_GSE103322")
            | (registry["short_name"].astype(str).str.contains("Puram", case=False, na=False))
            | (registry["role"].astype(str) == "core_discovery")
        )
        out = registry.loc[mask, "dataset_id"].tolist()
        if not out:
            raise ValueError("No Puram/core-discovery dataset found in registry.")
        return list(dict.fromkeys(out))

    col = {
        "training": "use_for_training",
        "validation": "use_for_validation",
        "outgroup": "use_as_outgroup",
    }.get(args.scope)
    if col:
        return registry.loc[_yes_no(registry[col]), "dataset_id"].tolist()
    return registry["dataset_id"].tolist()


def summarise_scores(adata, dataset_id: str, signature_names) -> list[dict]:
    rows = []
    if "standard_cell_type" in adata.obs.columns:
        groups = [(str(k), v.index) for k, v in adata.obs.groupby("standard_cell_type", observed=True)]
    else:
        groups = [("all_cells", adata.obs.index)]

    for group_name, cell_index in groups:
        sub = adata.obs.loc[cell_index]
        row = {"dataset": dataset_id, "standard_cell_type": group_name, "n_cells": int(sub.shape[0])}
        for sig in signature_names:
            col = f"{sig}_score"
            row[f"{col}_mean"] = float(sub[col].mean()) if col in sub.columns else pd.NA
            row[f"{col}_median"] = float(sub[col].median()) if col in sub.columns else pd.NA
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    sigs = load_signatures()
    registry = load_registry()
    selected = select_datasets(registry, args)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_rows, coverage_rows, dist_rows = [], [], []

    for ds in selected:
        in_path = SC_DIR / ds / f"{ds}_malignant_fibroblast.h5ad"
        if not in_path.exists():
            run_rows.append({"dataset": ds, "scope": args.scope, "status": "missing_input",
                             "input_file": str(in_path), "output_file": pd.NA,
                             "n_cells": pd.NA, "n_genes": pd.NA})
            print("Missing filtered input", ds, in_path)
            continue

        adata = sc.read_h5ad(in_path)
        scored = []
        for name, genes in sigs.items():
            present = [g for g in genes if g in adata.var_names]
            missing = [g for g in genes if g not in adata.var_names]
            coverage_rows.append({
                "dataset": ds, "signature": name,
                "present_genes": len(present), "total_signature_genes": len(genes),
                "missing_genes": len(missing),
                "present_gene_symbols": ";".join(present),
                "missing_gene_symbols": ";".join(missing),
                "scored": len(present) >= args.min_present_genes,
            })
            if len(present) >= args.min_present_genes:
                sc.tl.score_genes(adata, present, score_name=f"{name}_score")
                scored.append(name)

        out_path = SC_DIR / ds / f"{ds}_{args.output_suffix}.h5ad"
        adata.write(out_path)
        dist_rows.extend(summarise_scores(adata, ds, sigs.keys()))
        run_rows.append({"dataset": ds, "scope": args.scope, "status": "scored",
                         "input_file": str(in_path), "output_file": str(out_path),
                         "n_cells": int(adata.n_obs), "n_genes": int(adata.n_vars),
                         "scored_signatures": ";".join(scored)})
        print("Scored", ds, "->", out_path)

    pd.DataFrame(run_rows).to_csv(RESULTS_DIR / "single_cell_scoring_run_summary.tsv", sep="\t", index=False)
    pd.DataFrame(coverage_rows).to_csv(RESULTS_DIR / "single_cell_signature_gene_coverage.tsv", sep="\t", index=False)
    pd.DataFrame(dist_rows).to_csv(RESULTS_DIR / "single_cell_score_distribution_by_cell_type.tsv", sep="\t", index=False)
    print("Finished scoring scope:", args.scope)
    print("Datasets requested:", ", ".join(selected))


if __name__ == "__main__":
    main()
