"""Score the RNA-seq cohorts with MLR-EMT with and without the mapping to the microarray scale.

The reference code maps log2(TPM + 1) to a microarray-like scale (MA = 0.57 + 0.37 * log2TPM,
rnaToMA) before the NCI-60 offset is applied, and with that mapping every TCGA and CPTAC-3 tumour is
called hybrid. Entering log2(TPM + 1) directly is the other convention in use. TCGA-HNSC and
CPTAC-3 are scored both ways. The output gives the offset, the state calls, the Spearman correlation
of the score and of P(hybrid) with the mapped version, and the correlation of P(hybrid) with pEMT
specificity. Microarray cohorts are unaffected.

Inputs:  results/mlr_faithful/{TCGA,CPTAC}_mlr_faithful.tsv (sample set and pEMT specificity,
         from 02_mlr_emt_states.py)
         data/processed/tcga_hnsc/tcga_star_tpm_log2_for_projection.tsv
         data/processed/cptac_rnaseq/cptac_hnscc_log2tpm1_symbol_TUMOR.tsv
         data/raw/mlr_reference/ (read by pemt.mlr_faithful)
Outputs: results/mlr_faithful/rnaseq_convention.tsv
Usage:   python src/10_cetuximab/11_mlr_rnaseq_convention.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from pemt import mlr_faithful as M  # noqa: E402

ROOT = HERE.parents[1]
RES = ROOT / "results" / "mlr_faithful"


def cohorts():
    ref = pd.read_csv(RES / "TCGA_mlr_faithful.tsv", sep="\t", index_col=0)
    e = pd.read_csv(ROOT / "data/processed/tcga_hnsc/tcga_star_tpm_log2_for_projection.tsv",
                    sep="\t", index_col=0)
    yield "TCGA", e[ref.index], ref
    ref = pd.read_csv(RES / "CPTAC_mlr_faithful.tsv", sep="\t", index_col=0)
    e = pd.read_csv(ROOT / "data/processed/cptac_rnaseq/cptac_hnscc_log2tpm1_symbol_TUMOR.tsv",
                    sep="\t", index_col=0)
    yield "CPTAC-3", e[ref.index], ref


def main() -> None:
    rows = []
    for name, e, ref in cohorts():
        mapped, direct = M.mlr_faithful(e, True), M.mlr_faithful(e, False)
        for conv, f in (("rnaToMA (reference code)", mapped), ("log2(TPM + 1) direct", direct)):
            calls = f["state"].value_counts()
            row = {"cohort": name, "convention": conv, "n": len(f),
                   "offset_d": round(f.attrs["offset_d"], 3),
                   "n_epithelial": int(calls.get("E", 0)), "n_hybrid": int(calls.get("H", 0)),
                   "n_mesenchymal": int(calls.get("M", 0)),
                   "rho_score_vs_mapped": spearmanr(f["MLR_faithful"], mapped["MLR_faithful"])[0],
                   "rho_P_H_vs_mapped": spearmanr(f["P_H"], mapped["P_H"])[0]}
            if "pEMT_specificity" in ref.columns:
                row["rho_P_H_vs_pEMT_specificity"] = spearmanr(
                    f["P_H"], ref.loc[f.index, "pEMT_specificity"], nan_policy="omit")[0]
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "rnaseq_convention.tsv", sep="\t", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
