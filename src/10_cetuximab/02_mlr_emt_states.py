"""Compute the published MLR-EMT score and EMT state in every bulk cohort.

The published MLR-EMT (George et al. 2017) is computed with pemt.mlr_faithful, which follows the
reference code of its authors. Expression is loaded as in the other scripts of the pipeline. For
comparison, the earlier simplified version (score_mlr_approx in
src/04_tcga_projection/07_emt_score_panel.py) is computed on the same data and stored as MLR_mu.
The simplified version departs from the published method in three ways. X2 is log2VIM - log2CDH1
instead of log2VIM / log2CDH1. Each input is z-scored within the cohort and rescaled to a fixed mean
and SD instead of receiving one NCI-60-anchored offset. The score is P(H) + 2 P(M) instead of P(H)
if P(E) > P(M), else 2 - P(H).

Cohorts, scales and handling:
  TCGA-HNSC primaries  log2(TPM+1), RNA-seq          rnaToMA conversion applied
  CPTAC-3              log2(TPM+1), RNA-seq          rnaToMA conversion applied
  GSE41613             Affymetrix U133 Plus 2, log2  used as is
  GSE65858             Illumina HT-12 v4, log2       used as is
  GSE65021             Illumina DASL HT-12 v4, log2  used as is

The published model is first applied to the NCI-60 training lines as a check. For each cohort, the
summary gives the normalisation offset, the counts of epithelial (E), hybrid (H) and mesenchymal (M)
states, and the Spearman correlation of the published score with the simplified version, pEMT
specificity, 76GS, KS and the Puram pEMT and epithelial scores. CPTAC-3 is skipped when its
expression matrix is missing.

Inputs:  data/processed/tcga_hnsc/tcga_star_tpm_log2_for_projection.tsv
         results/tcga_projection/emt_score_panel_scores.tsv
         data/raw/external_cohorts/{GSE41613,GSE65858,GSE65021}_series_matrix.txt.gz
         data/raw/external_cohorts/{GPL570,GPL10558}.annot.gz
         results/tcga_projection/external/{GSE41613,GSE65858,CPTAC_HNSCC}_scores.tsv
         data/processed/cptac_rnaseq/cptac_hnscc_log2tpm1_symbol_TUMOR.tsv
         results/cetuximab_cohort/GSE65021_scores.tsv
         data/raw/mlr_reference/ (read by pemt.mlr_faithful)
Outputs: results/mlr_faithful/<cohort>_mlr_faithful.tsv
         results/mlr_faithful/summary.tsv
         results/mlr_faithful/nci60_validation.tsv
Usage:   python src/10_cetuximab/02_mlr_emt_states.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results" / "mlr_faithful"
sys.path.insert(0, str(HERE.parent))
from pemt import mlr_faithful as M  # noqa: E402


def _load(path: Path, name: str):
    sys.path.insert(0, str(path.parent))
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cohorts():
    """Yield (name, expression genes x samples, is_rnaseq, reference score table, expression for MLR_mu)."""
    tp = ROOT / "results" / "tcga_projection"
    ref = pd.read_csv(tp / "emt_score_panel_scores.tsv", sep="\t", index_col=0)
    e = pd.read_csv(ROOT / "data/processed/tcga_hnsc/tcga_star_tpm_log2_for_projection.tsv", sep="\t", index_col=0)
    sub = e[ref.index.intersection(e.columns)]
    yield "TCGA", sub, True, ref, sub

    ext = _load(ROOT / "src/04_tcga_projection/09_external_cohorts.py", "ext")
    raw = ROOT / "data" / "raw" / "external_cohorts"
    for gse, annot in (("GSE41613", "GPL570.annot.gz"), ("GSE65858", "GPL10558.annot.gz")):
        expr, _ = ext.read_series_matrix(raw / f"{gse}_series_matrix.txt.gz")
        e = ext.collapse(expr, ext.read_annotation(raw / annot))
        ref = pd.read_csv(tp / "external" / f"{gse}_scores.tsv", sep="\t", index_col=0)
        yield gse, e[ref.index.intersection(e.columns)], False, ref, e

    # CPTAC-3 matrix written by src/04_tcga_projection/00_prepare_cptac_rnaseq.py.
    f = ROOT / "data" / "processed" / "cptac_rnaseq" / "cptac_hnscc_log2tpm1_symbol_TUMOR.tsv"
    if not f.exists():
        print("  CPTAC skipped: run src/04_tcga_projection/00_prepare_cptac_rnaseq.py and 12_cptac_validation.py first")
    else:
        e = pd.read_csv(f, sep="	", index_col=0)
        ref = pd.read_csv(tp / "external" / "CPTAC_HNSCC_scores.tsv", sep="\t", index_col=0)
        yield "CPTAC", e[ref.index.intersection(e.columns)], True, ref, e[ref.index.intersection(e.columns)]

    cx = _load(ROOT / "src" / "10_cetuximab" / "01_cetuximab_cohort.py", "cx")
    expr, _ = cx.read_series_matrix(cx.DATA / "GSE65021_series_matrix.txt.gz")
    e = cx.collapse(expr, cx.read_annotation(cx.DATA / "GPL10558.annot.gz"))
    # Reference scores come from 01_cetuximab_cohort.py. The simplified MLR-EMT is recomputed in main().
    ref = pd.read_csv(ROOT / "results" / "cetuximab_cohort" / "GSE65021_scores.tsv", sep="\t", index_col=0)
    yield "GSE65021", e[ref.index.intersection(e.columns)], False, ref, e[ref.index.intersection(e.columns)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v = M.validate_on_nci60()
    v.to_csv(OUT / "nci60_validation.tsv", sep="\t", index=False)
    print("NCI-60 training lines with the published model:",
          v["state"].value_counts().reindex(list("EHM"), fill_value=0).to_dict())
    print(f"NCI-60 normaliser mean by probe ID vs as coded: "
          f"{np.mean([r.mean() for r in M.nci60_gene_rows(M.normalisers_for(set(M.gene_list()))).values()]):.3f}"
          f" vs {M.nci60_norm_as_coded():.3f}")

    panel = _load(ROOT / "src/04_tcga_projection/07_emt_score_panel.py", "panel")
    rows = []
    for name, e, rnaseq, ref, e_all in cohorts():
        f = M.mlr_faithful(e, rnaseq)
        meta = dict(f.attrs)
        f = f.join(panel.score_mlr_approx(e_all).rename("MLR_mu"))
        f = f.join(ref[[c for c in ("pEMT_specificity", "GS76", "KS", "puram_pemt",
                                     "puram_epi_dif_1", "hallmark_EMT") if c in ref]])
        f.to_csv(OUT / f"{name}_mlr_faithful.tsv", sep="\t")
        cnt = f["state"].value_counts().reindex(list("EHM"), fill_value=0)
        rho = lambda c: spearmanr(f["MLR_faithful"], f[c], nan_policy="omit")[0] if c in f else np.nan  # noqa: E731
        rows.append({"cohort": name, "n": len(f), "rnaseq": rnaseq, "offset_d": meta["offset_d"],
                     "n_normalisers": meta["n_normalisers"],
                     "E": int(cnt["E"]), "H": int(cnt["H"]), "M": int(cnt["M"]),
                     "median_score": float(f["MLR_faithful"].median()),
                     "rho_vs_approximation": rho("MLR_mu"), "rho_vs_axis": rho("pEMT_specificity"),
                     "rho_vs_76GS": rho("GS76"), "rho_vs_KS": rho("KS"),
                     "rho_vs_puram_pemt": rho("puram_pemt"), "rho_vs_puram_epi": rho("puram_epi_dif_1")})
        print(f"  {name}: n={len(f)}, offset {meta['offset_d']:+.3f}, states {cnt.to_dict()}")
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "summary.tsv", sep="\t", index=False)
    print("\n" + s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
