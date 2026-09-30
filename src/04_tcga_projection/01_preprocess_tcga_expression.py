"""Preprocess TCGA-HNSC expression for classifier projection and WGCNA.

The UCSC Xena GDC STAR TPM file is already on the log2(TPM + 1) scale. Each
matrix is checked for scale and is only transformed if its maximum value
suggests linear values. Ensembl identifiers are mapped to gene symbols with a
cached mapping table, keeping the first row per symbol.

The projection matrix keeps every sample (primary tumours, normals and
metastases). The WGCNA matrix is log2(TPM + 1), restricted to primary tumours
(TCGA aliquot type 01) and to genes with log2(TPM + 1) > 1 in at least 20 % of primary tumours
(thresholds from config.yaml).

Inputs:  data/raw/tcga_hnsc/expression/TCGA-HNSC.star_tpm.tsv
         data/references/ensembl_to_symbol_mapping.tsv
Outputs: data/processed/tcga_hnsc/tcga_star_tpm_log2_for_projection.tsv (genes x all 566 samples)
         data/processed/tcga_hnsc/tcga_log2tpm_primary_for_wgcna.tsv (expressed genes x 520 primaries)
         data/processed/tcga_hnsc/tcga_preprocess_report.txt
Usage:   python src/04_tcga_projection/01_preprocess_tcga_expression.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pemt import load_config, project_root  # noqa: E402

cfg = load_config()
ROOT = project_root()
TCGA = cfg["tcga"]
OUT_DIR = ROOT / cfg["paths"]["processed_dir"] / "tcga_hnsc"
REF_DIR = ROOT / cfg["paths"]["references_dir"]


def aliquot_type(sample_id: str) -> str:
    parts = sample_id.split("-")
    return parts[3][:2] if len(parts) >= 4 else "??"


def ensure_log2(df: pd.DataFrame, name: str, report: list) -> pd.DataFrame:
    """Apply log2(x + 1) only if the maximum value suggests a linear scale."""
    mx = float(np.nanmax(df.values))
    if mx > TCGA["log2_tpm_max_check"]:
        report.append(f"{name}: max value {mx:.1f} > {TCGA['log2_tpm_max_check']}, treated as linear and transformed with log2(x+1)")
        return np.log2(df + 1)
    report.append(f"{name}: max value {mx:.2f}, already on the log2 scale and used as is")
    return df


def main() -> None:
    report = []
    tpm = pd.read_csv(ROOT / TCGA["star_tpm_file"], sep="\t", index_col=0)
    report.append(f"star_tpm raw: {tpm.shape[0]} Ensembl IDs x {tpm.shape[1]} samples")

    mapping_cache = REF_DIR / "ensembl_to_symbol_mapping.tsv"
    if not mapping_cache.exists():
        raise SystemExit(f"Missing {mapping_cache}. It is provided in data/references/.")
    mapping_df = pd.read_csv(mapping_cache, sep="\t", index_col=0)

    tpm.index = [e.split(".")[0] for e in tpm.index]
    tpm = tpm.join(mapping_df, how="inner").set_index("symbol")
    tpm = tpm[~tpm.index.duplicated(keep="first")]
    report.append(f"after symbol mapping: {tpm.shape[0]} genes")

    tpm_log = ensure_log2(tpm, "star_tpm", report)

    primaries = [c for c in tpm_log.columns if aliquot_type(c) == "01"]
    normals = [c for c in tpm_log.columns if aliquot_type(c) == "11"]
    mets = [c for c in tpm_log.columns if aliquot_type(c) == "06"]
    report.append(f"samples: {len(primaries)} primary, {len(normals)} normal, {len(mets)} metastatic")

    prim = tpm_log[primaries]
    keep = (prim > TCGA["counts_min_log2_value"]).mean(axis=1) >= TCGA["counts_min_sample_frac"]
    wgcna = prim.loc[keep]
    report.append(f"WGCNA matrix: {wgcna.shape[0]} genes expressed (log2(TPM+1) > "
                  f"{TCGA['counts_min_log2_value']} in >= {TCGA['counts_min_sample_frac']:.0%} of primaries) x {wgcna.shape[1]} primaries")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tpm_log.to_csv(OUT_DIR / "tcga_star_tpm_log2_for_projection.tsv", sep="\t")
    wgcna.to_csv(OUT_DIR / "tcga_log2tpm_primary_for_wgcna.tsv", sep="\t")
    (OUT_DIR / "tcga_preprocess_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
