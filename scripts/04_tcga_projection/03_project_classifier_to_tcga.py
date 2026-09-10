"""Project the trained classifier onto TCGA-HNSC log2(TPM+1).

Patched 9 Sep 2026. What this produces and what it does not:

The training pseudobulks (mean log1p of normalised single-cell counts) and the
bulk target (log2 TPM) are on different scales, so the train-time StandardScaler
cannot be applied. Each gene is instead z-scored on the bulk cohort's own
distribution and the logistic coefficients are applied to those z-scores.

Consequence, stated plainly: an average tumour sits at z = 0 on every gene and
therefore receives roughly the training class prior. The resulting P values
are COHORT-RELATIVE rankings passed through a softmax, not calibrated state
probabilities. Rank-based uses (Cox, tertiles, correlations, WGCNA traits) are
valid; absolute statements ("35 % of tumours are pEMT-high", "normal tissue
has P = 0.007") are not, and the manuscript must not make them.

Change versus the original: the per-gene mean and SD are computed on PRIMARY
tumours only and then applied to every sample, so primary-tumour scores no
longer depend on how many normals or metastases are in the cohort.
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
OUT_DIR = ROOT / cfg["paths"]["results_dir"] / "tcga_projection"


def aliquot_type(sample_id: str) -> str:
    parts = sample_id.split("-")
    return parts[3][:2] if len(parts) >= 4 else "??"


def project(expr_samples_x_genes: pd.DataFrame, genes: list, clf, ref_index) -> pd.DataFrame:
    """z-score on the reference samples' per-gene mean/SD, apply coefficients."""
    shared = [g for g in genes if g in expr_samples_x_genes.columns]
    X_full = pd.DataFrame(0.0, index=expr_samples_x_genes.index, columns=genes)
    X_full[shared] = expr_samples_x_genes[shared]
    ref = X_full.loc[ref_index]
    means = ref.mean(axis=0)
    stds = ref.std(axis=0).replace(0, 1.0)
    Z = ((X_full - means) / stds).fillna(0.0)
    Z = np.nan_to_num(Z.values, nan=0.0, posinf=0.0, neginf=0.0)
    probs = clf.predict_proba(Z)
    out = pd.DataFrame(probs, index=X_full.index, columns=[f"P_{c}" for c in clf.classes_])
    return out, len(shared)


def main() -> None:
    model = joblib.load(CLF_DIR / "tcga_depmap_ready_classifier.pkl")
    genes = pd.read_csv(CLF_DIR / "classifier_genes.txt", header=None)[0].tolist()
    clf = model.named_steps["clf"]

    expr = pd.read_csv(ROOT / cfg["paths"]["processed_dir"] / "tcga_hnsc" / "tcga_star_tpm_log2_for_projection.tsv",
                       sep="\t", index_col=0)
    X = expr.T
    primaries = [s for s in X.index if aliquot_type(s) == "01"]
    p, n_shared = project(X, genes, clf, primaries)
    p["sample_type"] = [{"01": "primary", "11": "normal", "06": "metastatic"}.get(aliquot_type(s), "other") for s in p.index]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p.to_csv(OUT_DIR / "tcga_class_probabilities.tsv", sep="\t")
    note = (
        f"Projected {p.shape[0]} samples using {n_shared}/{len(genes)} classifier genes.\n"
        f"Per-gene z-scores use mean/SD of the {len(primaries)} primary tumours.\n"
        "Scores are cohort-relative (see script docstring); do not report them as absolute probabilities.\n"
        f"Mean P among primaries: {p.loc[primaries, [c for c in p.columns if c.startswith('P_')]].mean().round(3).to_dict()}\n"
    )
    (OUT_DIR / "projection_note.txt").write_text(note)
    print(note)


if __name__ == "__main__":
    main()
