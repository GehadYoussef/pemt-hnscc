"""Prepare GSE181919 (Choi et al., Nat Commun 2023) for the gene partition replication.

The series deposits a dense UMI matrix (genes x 54,239 cells, 128 MB gzipped) and per-cell metadata
with author annotation (patient, sample, tissue type, and cell type including Malignant.cells and
Fibroblasts). Loading the matrix whole would need about 9 GB, so it is streamed. Every line
contributes to the per-cell library sizes, and only the genes needed downstream are kept.

Kept: the 100 Puram pEMT genes, plus the EGFR-family ligands and receptors and the fibroblast ligands
of the ligand-source analysis (src/11_composition_and_mechanism/03_caf_tumour_ligand_receptor.py), as
counts per 10,000 (linear) for every cell.

Inputs:  data/raw/GSE181919/GSE181919_UMI_counts.txt.gz,
         data/raw/GSE181919/GSE181919_Barcode_metadata.txt.gz, config/signatures/puram_pemt.txt
Outputs: data/processed/GSE181919/GSE181919_subset.h5ad
Usage:   python src/01_single_cell_qc/05_prepare_gse181919.py
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
D = ROOT / "data" / "raw" / "GSE181919"
PROC = ROOT / "data" / "processed" / "GSE181919"
SIG = ROOT / "config" / "signatures" / "puram_pemt.txt"
LIGANDS = ["EGF", "TGFA", "AREG", "EREG", "HBEGF", "BTC", "EPGN", "NRG1", "NRG2", "EGFR", "ERBB2",
           "ERBB3", "ERBB4", "FGF7", "FGF2", "HGF", "CXCL12", "FGF10", "IGF1", "PDGFA", "PDGFB"]


def main() -> None:
    import anndata as ad

    wanted = {l.strip() for l in SIG.read_text().splitlines() if l.strip() and not l.startswith("#")}
    wanted |= set(LIGANDS)
    with gzip.open(D / "GSE181919_UMI_counts.txt.gz", "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        cells = [h.replace(".", "-") for h in header]
        libsize = np.zeros(len(cells))
        keep, rows = [], []
        for i, line in enumerate(fh):
            gene, rest = line.split("\t", 1)
            v = np.fromstring(rest, sep="\t")
            if len(v) != len(cells):
                raise ValueError(f"row {gene}: {len(v)} values for {len(cells)} cells")
            libsize += v
            if gene in wanted:
                keep.append(gene)
                rows.append(v.astype(np.float32))
            if i % 2000 == 0:
                print(f"  {i} genes read", flush=True)
    meta = pd.read_csv(D / "GSE181919_Barcode_metadata.txt.gz", sep="\t", index_col=0)
    missing = set(cells) - set(meta.index)
    if missing:
        raise ValueError(f"{len(missing)} matrix barcodes have no metadata, e.g. {sorted(missing)[:3]}")
    X = (np.vstack(rows).T / libsize[:, None] * 1e4).astype(np.float32)   # CP10K, linear
    a = ad.AnnData(X=X, obs=meta.loc[cells].copy(), var=pd.DataFrame(index=keep))
    a.obs["library_size"] = libsize
    PROC.mkdir(parents=True, exist_ok=True)
    a.write_h5ad(PROC / "GSE181919_subset.h5ad")
    print(f"wrote {a.shape[0]} cells x {a.shape[1]} genes, not found: {sorted(wanted - set(keep))}")


if __name__ == "__main__":
    main()
