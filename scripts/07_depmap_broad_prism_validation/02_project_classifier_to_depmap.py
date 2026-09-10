"""Project the classifier onto DepMap models, within lineage cohorts.

Patched 9 Sep 2026. The original z-scored every gene across all DepMap lineages
(leukaemias, neural tumours and so on), so a squamous line was scored relative
to a pan-cancer background. Here each cohort (hnscc, pan_squamous) is z-scored
on its own models and scored separately; scores are cohort-relative rankings,
as in the TCGA projection.

Outputs (results/depmap_broad_prism/): depmap_class_probabilities_<cohort>.tsv
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import load_config, project_root  # noqa: E402

cfg = load_config()
ROOT = project_root()
CLF_DIR = ROOT / cfg["paths"]["results_dir"] / "multinomial_classifier"
PROC = ROOT / cfg["paths"]["processed_dir"] / "depmap"
OUT = ROOT / cfg["paths"]["results_dir"] / "depmap_broad_prism"
COHORTS = ["hnscc", "pan_squamous"]


def main() -> None:
    model = joblib.load(CLF_DIR / "tcga_depmap_ready_classifier.pkl")
    genes = pd.read_csv(CLF_DIR / "classifier_genes.txt", header=None)[0].tolist()
    clf = model.named_steps["clf"]
    nz = [genes[i] for i in np.where((clf.coef_ != 0).any(axis=0))[0]]
    expr = pd.read_csv(PROC / "depmap_expression_log2tpm.tsv", sep="\t", index_col=0, usecols=lambda c: c == "ModelID" or c in set(nz) or c == "Unnamed: 0")
    coh = pd.read_csv(PROC / "depmap_cohorts.tsv", sep="\t", index_col=0)
    OUT.mkdir(parents=True, exist_ok=True)
    col = {g: i for i, g in enumerate(genes)}
    for cohort in COHORTS:
        ids = coh.index[coh[f"cohort_{cohort}"]].intersection(expr.index)
        X = expr.loc[ids]
        shared = [g for g in nz if g in X.columns]
        Z = (X[shared] - X[shared].mean()) / X[shared].std().replace(0, 1.0)
        full = np.zeros((len(ids), len(genes)), dtype=np.float32)
        full[:, [col[g] for g in shared]] = np.nan_to_num(Z.values)
        probs = clf.predict_proba(full)
        p = pd.DataFrame(probs, index=ids, columns=[f"P_{c}" for c in clf.classes_])
        p["pEMT_specificity"] = p["P_pEMT_high"] - p[["P_epithelial_like", "P_fibroblast_stromal_like"]].max(axis=1)
        p["pEMT_vs_epithelial"] = p["P_pEMT_high"] - p["P_epithelial_like"]
        p = p.join(coh[["cell_line_name", "lineage", "oncotree_code", "model_type", "is_organoid", "primary_or_metastasis"]])
        p.to_csv(OUT / f"depmap_class_probabilities_{cohort}.tsv", sep="\t")
        print(f"{cohort}: {len(ids)} models ({int(p['is_organoid'].sum())} organoids), {len(shared)} informative genes; "
              f"top pEMT: {', '.join(p.sort_values('pEMT_specificity', ascending=False)['cell_line_name'].head(6))}")


if __name__ == "__main__":
    main()
