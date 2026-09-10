"""Preprocess DepMap expression and define the cell-model cohorts.

Patched 9 Sep 2026. Adds cohort definitions so that projection and drug/CRISPR
tests are done within lineage instead of across all of DepMap:
  hnscc         OncotreeLineage == "Head and Neck" (primary cohort)
  pan_squamous  hnscc plus OncotreeCode in ESCC, LUSC, CESC, CSCC (sensitivity)
Organoid models (ModelType == "Organoid", HCMI and others) are kept and flagged.

Outputs (data/processed/depmap/):
  depmap_expression_log2tpm.tsv (models x genes), depmap_model_metadata.tsv,
  depmap_cohorts.tsv (ModelID, cohort flags, lineage, model type)
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import load_config, project_root  # noqa: E402

cfg = load_config()
ROOT = project_root()
DM = cfg["depmap"]
OUT = ROOT / cfg["paths"]["processed_dir"] / "depmap"
META_COLS = ("SequencingID", "ModelConditionID", "IsDefaultEntryForMC", "IsDefaultEntryForModel")
SQUAMOUS_CODES = {"ESCC", "LUSC", "CESC", "CSCC"}


def main() -> None:
    raw = pd.read_csv(ROOT / DM["expression_file"], index_col=0)
    if "ModelID" in raw.columns:
        if "IsDefaultEntryForModel" in raw.columns:
            raw = raw[raw["IsDefaultEntryForModel"].astype(str).str.lower().isin(["true", "yes", "1"])]
        raw = raw.set_index("ModelID").drop(columns=[c for c in META_COLS if c in raw.columns])
    raw.columns = [c.split(" (")[0] if " (" in c else c for c in raw.columns]
    raw = raw.loc[:, ~pd.Index(raw.columns).duplicated()]
    raw = raw[~raw.index.duplicated(keep="first")]

    model = pd.read_csv(ROOT / DM["model_file"]).set_index("ModelID")
    coh = pd.DataFrame(index=raw.index)
    coh["lineage"] = model["OncotreeLineage"].reindex(coh.index)
    coh["primary_disease"] = model["OncotreePrimaryDisease"].reindex(coh.index)
    coh["oncotree_code"] = model["OncotreeCode"].reindex(coh.index)
    coh["model_type"] = model["ModelType"].reindex(coh.index)
    coh["cell_line_name"] = model["CellLineName"].reindex(coh.index)
    coh["primary_or_metastasis"] = model["PrimaryOrMetastasis"].reindex(coh.index)
    coh["is_organoid"] = coh["model_type"].astype(str).str.lower().eq("organoid")
    coh["cohort_hnscc"] = coh["lineage"].eq("Head and Neck")
    coh["cohort_pan_squamous"] = coh["cohort_hnscc"] | coh["oncotree_code"].isin(SQUAMOUS_CODES)

    OUT.mkdir(parents=True, exist_ok=True)
    raw.to_csv(OUT / "depmap_expression_log2tpm.tsv", sep="\t")
    model.to_csv(OUT / "depmap_model_metadata.tsv", sep="\t")
    coh.to_csv(OUT / "depmap_cohorts.tsv", sep="\t")
    print(f"DepMap expression: {raw.shape[0]} models x {raw.shape[1]} genes")
    print(f"cohort hnscc: {int(coh['cohort_hnscc'].sum())} (organoids {int((coh['cohort_hnscc'] & coh['is_organoid']).sum())}); "
          f"pan_squamous: {int(coh['cohort_pan_squamous'].sum())} (organoids {int((coh['cohort_pan_squamous'] & coh['is_organoid']).sum())})")


if __name__ == "__main__":
    main()
