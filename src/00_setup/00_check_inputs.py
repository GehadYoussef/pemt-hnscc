"""Check that all expected raw input files are present.

Core files are read from the tcga, depmap, broad_prism and networks sections of config/config.yaml,
so inputs are added or removed in the YAML. Each dataset in config/dataset_registry.tsv must also
have its cell metadata table (CellMetainfo_table.tsv) under data/raw/single_cell/<dataset>/. Missing
files are printed, and the exit status is 1 if any core file is absent.

Inputs:  config/config.yaml, config/dataset_registry.tsv, the data/raw/ files named in the config
Outputs: none (report printed to the console)
Usage:   python src/00_setup/00_check_inputs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pemt import load_config, load_registry, project_root  # noqa: E402

cfg = load_config()
ROOT = project_root()


def required_files() -> list[str]:
    files: list[str] = [
        cfg["tcga"]["star_counts_file"],
        cfg["tcga"]["star_tpm_file"],
        cfg["tcga"]["clinical_file"],
        cfg["tcga"]["survival_file"],
        cfg["depmap"]["expression_file"],
        cfg["depmap"]["model_file"],
        cfg["broad_prism"]["secondary_cell_line_file"],
        cfg["broad_prism"]["secondary_curve_file"],
        cfg["broad_prism"]["secondary_lfc_file"],
        cfg["broad_prism"]["secondary_treatment_file"],
        cfg["broad_prism"]["corsello_file"],
    ]
    files.extend(v for k, v in cfg["networks"].items() if k.endswith("_file"))
    return files


def main() -> int:
    print("Core file check")
    missing = [f for f in required_files() if not (ROOT / f).exists()]
    if missing:
        print("Missing:")
        for f in missing:
            print("  -", f)
    else:
        print("All core files present.")

    print("\nSingle-cell metadata check")
    sc_dir = ROOT / "data" / "raw" / "single_cell"
    meta_name = cfg["single_cell"]["metadata_file_name"]
    registry = load_registry()
    for ds in registry["dataset_id"]:
        meta = sc_dir / ds / meta_name
        print(ds, "OK" if meta.exists() else f"MISSING {meta_name}")

    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
