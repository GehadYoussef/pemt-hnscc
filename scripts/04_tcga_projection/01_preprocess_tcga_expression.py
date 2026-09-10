"""Preprocess TCGA-HNSC expression for classifier projection and WGCNA.

Patched 9 Sep 2026. Changes versus the original:
  1. The UCSC Xena GDC files are ALREADY log-transformed: star_tpm is
     log2(TPM + 1) and star_counts is log2(count + 1). The original applied a
     second log2 to the counts file, so WGCNA ran on log2(log2(count+1)+1).
     Both files are now checked for scale and never re-logged.
  2. WGCNA input is log2(TPM + 1), not un-normalised counts, so co-expression
     does not track sequencing depth.
  3. WGCNA input is restricted to primary tumours (TCGA aliquot type 01).
     Normals and metastases are kept only in the projection matrix.
  4. Expression filter: log2(TPM+1) > 1 in at least 20 % of primary tumours.

Outputs (data/processed/tcga_hnsc/):
  tcga_star_tpm_log2_for_projection.tsv     genes x all 566 samples, log2(TPM+1)
  tcga_log2tpm_primary_for_wgcna.tsv        genes x 520 primaries, expression-filtered
  tcga_preprocess_report.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import load_config, project_root  # noqa: E402

cfg = load_config()
ROOT = project_root()
TCGA = cfg["tcga"]
OUT_DIR = ROOT / cfg["paths"]["processed_dir"] / "tcga_hnsc"
REF_DIR = ROOT / cfg["paths"]["references_dir"]


def aliquot_type(sample_id: str) -> str:
    parts = sample_id.split("-")
    return parts[3][:2] if len(parts) >= 4 else "??"


def ensure_log2(df: pd.DataFrame, name: str, report: list) -> pd.DataFrame:
    """Xena distributes log2-transformed matrices. Only log if values look linear."""
    mx = float(np.nanmax(df.values))
    if mx > TCGA["log2_tpm_max_check"]:
        report.append(f"{name}: max value {mx:.1f} > {TCGA['log2_tpm_max_check']} -> treated as linear, applied log2(x+1)")
        return np.log2(df + 1)
    report.append(f"{name}: max value {mx:.2f} -> already log2 scale, used as is")
    return df


def main() -> None:
    report = []
    tpm = pd.read_csv(ROOT / TCGA["star_tpm_file"], sep="\t", index_col=0)
    report.append(f"star_tpm raw: {tpm.shape[0]} Ensembl IDs x {tpm.shape[1]} samples")

    mapping_cache = REF_DIR / "ensembl_to_symbol_mapping.tsv"
    if not mapping_cache.exists():
        raise SystemExit(f"Missing {mapping_cache}. Run the original stage 04-01 once online to build it.")
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
    (OUT_DIR / "tcga_preprocess_report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
