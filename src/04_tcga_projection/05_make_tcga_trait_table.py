"""Join projection scores with clinical and survival metadata for WGCNA traits.

Inputs:  results/tcga_projection/tcga_pemt_specificity_scores.tsv
         data/processed/tcga_hnsc/tcga_clinical_survival_merged.tsv
Outputs: results/tcga_projection/tcga_master_trait_table.tsv
Usage:   python src/04_tcga_projection/05_make_tcga_trait_table.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pemt import load_config, project_root  # noqa: E402

cfg = load_config()
ROOT = project_root()
PROC = ROOT / cfg["paths"]["processed_dir"] / "tcga_hnsc"
DIR = ROOT / cfg["paths"]["results_dir"] / "tcga_projection"


def main() -> None:
    scores = pd.read_csv(DIR / "tcga_pemt_specificity_scores.tsv", sep="\t", index_col=0)
    meta = pd.read_csv(PROC / "tcga_clinical_survival_merged.tsv", sep="\t", index_col=0)
    traits = scores.join(meta, how="left")
    DIR.mkdir(parents=True, exist_ok=True)
    traits.to_csv(DIR / "tcga_master_trait_table.tsv", sep="\t")
    print("Trait table:", traits.shape)


if __name__ == "__main__":
    main()
