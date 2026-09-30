"""Export classifier coefficients and the top positive genes of each class.

The coefficient matrix of the fitted multinomial model is written as genes x classes, and for each
class the 200 genes with the largest positive coefficients are written to their own table.

Inputs:  results/multinomial_classifier/tcga_depmap_ready_classifier.pkl,
         results/multinomial_classifier/classifier_genes.txt
Outputs: results/multinomial_classifier/classifier_coefficients.tsv,
         results/multinomial_classifier/top_positive_genes_<class>.tsv
Usage:   python src/02_multinomial_pemt_model/04_export_classifier_coefficients.py
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pemt import load_config, project_root  # noqa: E402

cfg = load_config()
ROOT = project_root()
CLF_DIR = ROOT / cfg["paths"]["results_dir"] / "multinomial_classifier"

TOP_N_PER_CLASS = 200


def main() -> None:
    model = joblib.load(CLF_DIR / "tcga_depmap_ready_classifier.pkl")
    genes = pd.read_csv(CLF_DIR / "classifier_genes.txt", header=None)[0].tolist()
    clf = model.named_steps["clf"]

    coef = pd.DataFrame(clf.coef_, index=clf.classes_, columns=genes).T
    coef.to_csv(CLF_DIR / "classifier_coefficients.tsv", sep="\t")

    for cls in clf.classes_:
        coef[cls].sort_values(ascending=False).head(TOP_N_PER_CLASS).to_csv(
            CLF_DIR / f"top_positive_genes_{cls}.tsv", sep="\t"
        )


if __name__ == "__main__":
    main()
