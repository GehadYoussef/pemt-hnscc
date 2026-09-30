"""Merge TCGA-HNSC clinical, survival and HPV tables on sample ID.

The GDC clinical table has no HPV field, so HPV status is taken from two
sources:
  * PanCanAtlas molecular subtype (HNSC_HPV+ / HNSC_HPV-) for 487 patients,
    exported from cBioPortal study hnsc_tcga_pan_can_atlas_2018 (attribute
    SUBTYPE).
  * Legacy TCGA p16 and ISH testing from the Xena clinicalMatrix, when present.
The final column hpv_status is PanCanAtlas where available, else p16, else ISH.

Inputs:  data/raw/tcga_hnsc/clinical/TCGA-HNSC.clinical.tsv
         data/raw/tcga_hnsc/clinical/TCGA-HNSC.survival.tsv
         data/raw/tcga_hnsc/clinical/TCGA-HNSC.hpv_status_pancanatlas_cbioportal.tsv
         data/raw/tcga_hnsc/clinical/TCGA.HNSC.sampleMap_HNSC_clinicalMatrix (optional)
Outputs: data/processed/tcga_hnsc/tcga_clinical_survival_merged.tsv
Usage:   python src/04_tcga_projection/02_prepare_tcga_metadata.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pemt import load_config, project_root  # noqa: E402

cfg = load_config()
ROOT = project_root()
TCGA = cfg["tcga"]
CLIN_DIR = ROOT / "data" / "raw" / "tcga_hnsc" / "clinical"
OUT_DIR = ROOT / cfg["paths"]["processed_dir"] / "tcga_hnsc"


def main() -> None:
    clinical = pd.read_csv(ROOT / TCGA["clinical_file"], sep="\t")
    survival = pd.read_csv(ROOT / TCGA["survival_file"], sep="\t")

    c_col = "sample" if "sample" in clinical.columns else clinical.columns[0]
    s_col = "sample" if "sample" in survival.columns else survival.columns[0]
    clinical = clinical.set_index(c_col)
    survival = survival.set_index(s_col)
    survival = survival.rename(columns={"OS.time": "OS_time", "OS": "OS_event", "_PATIENT": "patient_id"})

    meta = clinical.join(survival, how="left", rsuffix="_survival")
    meta["patient_barcode"] = ["-".join(s.split("-")[:3]) for s in meta.index]

    hpv_pca = CLIN_DIR / "TCGA-HNSC.hpv_status_pancanatlas_cbioportal.tsv"
    if hpv_pca.exists():
        h = pd.read_csv(hpv_pca, sep="\t")
        meta = meta.merge(h, left_on="patient_barcode", right_on="patient_id", how="left").set_index(meta.index)
        meta = meta.drop(columns=["patient_id_y"], errors="ignore")
        print(f"PanCanAtlas HPV status joined: {meta['hpv_status_pancanatlas'].notna().sum()} samples")
    else:
        meta["hpv_status_pancanatlas"] = pd.NA
        print("PanCanAtlas HPV file not found, skipping")

    legacy = CLIN_DIR / "TCGA.HNSC.sampleMap_HNSC_clinicalMatrix"
    if legacy.exists():
        lg = pd.read_csv(legacy, sep="\t", index_col=0, low_memory=False)
        lg.index = [s[:15] for s in lg.index]
        for col in ("hpv_status_by_p16_testing", "hpv_status_by_ish_testing"):
            if col in lg.columns:
                meta[col] = meta.index.map(lambda s: lg[col].get(s[:15], pd.NA))
        print("Legacy p16/ISH HPV columns joined")

    def resolve(row):
        v = row.get("hpv_status_pancanatlas")
        if isinstance(v, str) and v in ("HPV+", "HPV-"):
            return v
        for col in ("hpv_status_by_p16_testing", "hpv_status_by_ish_testing"):
            v = row.get(col)
            if isinstance(v, str) and v.lower() == "positive":
                return "HPV+"
            if isinstance(v, str) and v.lower() == "negative":
                return "HPV-"
        return pd.NA
    meta["hpv_status"] = meta.apply(resolve, axis=1)
    print("hpv_status:", meta["hpv_status"].value_counts(dropna=False).to_dict())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta.to_csv(OUT_DIR / "tcga_clinical_survival_merged.tsv", sep="\t")
    print("Saved merged TCGA metadata:", meta.shape)


if __name__ == "__main__":
    main()
