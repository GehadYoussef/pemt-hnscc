"""Prepare the TCGA-HNSC WGCNA input matrix.

Patched 9 Sep 2026. Reads the corrected primary-tumour log2(TPM+1) matrix
(stage 04-01), keeps the top-N most variable genes (config.wgcna.top_variable_genes),
and records how many Puram program genes survive the variance cut. No genes are
forced in: with the double-log bug removed, 76 of 100 pEMT genes are inside the
top 8,000 by variance on their own.

Outputs (results/wgcna/):
  datExpr_for_wgcna.pkl          samples x genes (float32)
  wgcna_input_signature_coverage.tsv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import load_config, load_signatures, project_root  # noqa: E402

cfg = load_config()
ROOT = project_root()
EXPR_PATH = ROOT / cfg["paths"]["processed_dir"] / "tcga_hnsc" / "tcga_log2tpm_primary_for_wgcna.tsv"
OUT_DIR = ROOT / cfg["paths"]["results_dir"] / "wgcna"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    expr = pd.read_csv(EXPR_PATH, sep="\t", index_col=0)
    print(f"Input: {expr.shape[0]} genes x {expr.shape[1]} primary tumours (log2 TPM+1)")
    datExpr = expr.T
    gene_var = datExpr.var(axis=0).replace([np.inf, -np.inf], np.nan).dropna()
    gene_var = gene_var[gene_var > 0]
    top_n = min(int(cfg["wgcna"]["top_variable_genes"]), gene_var.size)
    top_genes = gene_var.sort_values(ascending=False).head(top_n).index
    datExpr = datExpr[top_genes].astype(np.float32)
    print(f"Kept {datExpr.shape[0]} samples x {datExpr.shape[1]} top-variable genes")

    rows = []
    for name, genes in load_signatures().items():
        present = [g for g in genes if g in gene_var.index]
        kept = [g for g in genes if g in top_genes]
        rows.append({"signature": name, "n_genes": len(genes), "in_expressed_matrix": len(present), "in_wgcna_input": len(kept)})
    cov = pd.DataFrame(rows)
    cov.to_csv(OUT_DIR / "wgcna_input_signature_coverage.tsv", sep="\t", index=False)
    print(cov.to_string(index=False))
    datExpr.to_pickle(OUT_DIR / "datExpr_for_wgcna.pkl")


if __name__ == "__main__":
    main()
