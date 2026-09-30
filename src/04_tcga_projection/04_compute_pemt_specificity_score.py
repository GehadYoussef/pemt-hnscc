"""Derive pEMT specificity and contrast scores from class probabilities.

pEMT_specificity is P_pEMT_high minus the larger of P_epithelial_like and
P_fibroblast_stromal_like. pEMT_vs_stroma and pEMT_vs_epithelial are
P_pEMT_high minus each of those two probabilities.

Inputs:  results/tcga_projection/tcga_class_probabilities.tsv
Outputs: results/tcga_projection/tcga_pemt_specificity_scores.tsv
Usage:   python src/04_tcga_projection/04_compute_pemt_specificity_score.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pemt import load_config, project_root  # noqa: E402

cfg = load_config()
ROOT = project_root()
DIR = ROOT / cfg["paths"]["results_dir"] / "tcga_projection"


def main() -> None:
    p = pd.read_csv(DIR / "tcga_class_probabilities.tsv", sep="\t", index_col=0)
    for col in ("P_pEMT_high", "P_epithelial_like", "P_fibroblast_stromal_like"):
        if col not in p.columns:
            raise ValueError(f"Missing {col}. Check classifier class names.")

    p["pEMT_specificity"] = (
        p["P_pEMT_high"] - p[["P_epithelial_like", "P_fibroblast_stromal_like"]].max(axis=1)
    )
    p["pEMT_vs_stroma"] = p["P_pEMT_high"] - p["P_fibroblast_stromal_like"]
    p["pEMT_vs_epithelial"] = p["P_pEMT_high"] - p["P_epithelial_like"]
    p.to_csv(DIR / "tcga_pemt_specificity_scores.tsv", sep="\t")
    print(f"Wrote specificity scores for {p.shape[0]} samples")


if __name__ == "__main__":
    main()
