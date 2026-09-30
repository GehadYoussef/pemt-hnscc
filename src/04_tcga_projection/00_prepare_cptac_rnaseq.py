"""Build the CPTAC-3 HNSCC RNA-seq matrix from GDC STAR gene quantifications.

The GDC STAR - Counts gene quantifications (GENCODE v36) for the CPTAC-3 HNSCC
primary tumours are downloaded from the GDC API in batches, with retries. TPM
values are read for Ensembl genes, Ensembl identifiers are collapsed to gene
symbols by highest mean expression, and values are transformed to
log2(TPM + 1).

Some patients have two or three primary-tumour aliquots. Every aliquot is
downloaded, and one aliquot per patient is kept: the first by sample
identifier.

Inputs:  data/references/cptac_gdc_star_manifest.tsv (GDC file list for the CPTAC-3 cases)
Outputs: data/raw/cptac_rnaseq/star/<file_id>.tsv (downloaded GDC files)
         data/processed/cptac_rnaseq/cptac_hnscc_log2tpm1_symbol_TUMOR.tsv
Usage:   python src/04_tcga_projection/00_prepare_cptac_rnaseq.py
"""

from __future__ import annotations

import io
import time
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data" / "raw" / "cptac_rnaseq"
PROC = ROOT / "data" / "processed" / "cptac_rnaseq"
STAR = D / "star"
BATCH = 10


def download(man: pd.DataFrame) -> None:
    STAR.mkdir(parents=True, exist_ok=True)
    todo = [f for f in man["file_id"] if not (STAR / f"{f}.tsv").exists()]
    for i in range(0, len(todo), BATCH):
        ids = todo[i:i + BATCH]
        for attempt in range(5):             # the GDC data endpoint drops connections under load
            try:
                r = requests.post("https://api.gdc.cancer.gov/data", json={"ids": ids}, timeout=600)
                r.raise_for_status()
                break
            except requests.exceptions.RequestException as exc:
                print(f"  retry {attempt + 1} after {exc.__class__.__name__}", flush=True)
                time.sleep(15 * (attempt + 1))
        else:
            raise RuntimeError(f"GDC download failed for batch starting {ids[0]}")
        if len(ids) == 1:
            (STAR / f"{ids[0]}.tsv").write_bytes(r.content)
        else:
            with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tar:
                for m in tar.getmembers():
                    if m.isfile() and m.name.endswith(".tsv"):
                        fid = m.name.split("/")[0]
                        (STAR / f"{fid}.tsv").write_bytes(tar.extractfile(m).read())
        print(f"  downloaded {min(i + BATCH, len(todo))}/{len(todo)}", flush=True)


def read_tpm(fid: str) -> pd.Series:
    t = pd.read_csv(STAR / f"{fid}.tsv", sep="\t", comment="#")
    t = t[t["gene_id"].str.startswith("ENSG")]
    return t.set_index("gene_id")["tpm_unstranded"], t.set_index("gene_id")["gene_name"]


def main() -> None:
    man = pd.read_csv(ROOT / "data" / "references" / "cptac_gdc_star_manifest.tsv", sep="\t")
    man = man[man["sample_type"] == "Primary Tumor"].sort_values(["case", "sample"])
    download(man)
    tpm, names = {}, None
    for fid in man["file_id"]:
        s, n = read_tpm(fid)
        tpm[fid] = s
        names = n if names is None else names
    X = pd.DataFrame(tpm)
    # collapse Ensembl IDs to symbols, keeping the highest-mean row
    sym = names.reindex(X.index)
    X["_sym"], X["_mean"] = sym.values, X.mean(axis=1).values
    X = X.sort_values("_mean", ascending=False).drop_duplicates("_sym").set_index("_sym").drop(columns="_mean")
    X = np.log2(X + 1)
    first = man.groupby("case")["file_id"].first()
    Xf = X[first.values]
    Xf.columns = first.index
    PROC.mkdir(parents=True, exist_ok=True)
    Xf.to_csv(PROC / "cptac_hnscc_log2tpm1_symbol_TUMOR.tsv", sep="\t")
    print(f"{X.shape[0]} genes, {Xf.shape[1]} patients (first primary-tumour aliquot)")


if __name__ == "__main__":
    main()
