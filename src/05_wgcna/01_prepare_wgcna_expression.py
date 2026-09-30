"""Prepare the TCGA-HNSC expression matrix used as WGCNA input.

Reads the primary-tumour log2(TPM+1) matrix written by
src/04_tcga_projection/01_preprocess_tcga_expression.py. Genes with zero or
non-finite variance are dropped, and the top-N most variable genes are kept
(config wgcna.top_variable_genes, 8,000). No genes are forced in. The number
of genes from each signature that survive the variance cut is recorded. With
8,000 genes, 76 of the 100 Puram pEMT genes are in the WGCNA input.

Inputs:  data/processed/tcga_hnsc/tcga_log2tpm_primary_for_wgcna.tsv,
         config/signatures.yaml, config/signatures/*.txt
Outputs: results/wgcna/datExpr_for_wgcna.pkl (samples x genes, float32),
         results/wgcna/wgcna_input_signature_coverage.tsv
Usage:   python src/05_wgcna/01_prepare_wgcna_expression.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pemt import load_config, load_signatures, project_root  # noqa: E402

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
