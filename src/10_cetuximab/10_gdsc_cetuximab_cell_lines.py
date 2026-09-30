"""Test the malignant arm against cetuximab sensitivity of head and neck cancer cell lines (GDSC1).

GDSC1 (Sanger, drug 1114) is the only public drug-response resource with cetuximab for head and
neck models: 35 HNSC cell lines, with fitted dose-response AUC and ln IC50. PRISM (Broad) screens
small molecules only, and the 8 head and neck HCMI organoid models on the GDC carry no drug
response. Expression is DepMap log2(TPM + 1), default entry per model, matched through the Sanger
model identifier.

Cell lines lack stroma, so only the malignant arm is tested. Most HNSC lines have a cetuximab AUC
above 0.85, which limits the power of this test.

The primary panel is the 35 HNSC lines. A sensitivity panel adds oesophageal squamous (OncotreeCode
ESCC), lung squamous (LUSC) and cervical squamous (CESC) lines. For each score, standardised within
the panel, the output gives the Spearman correlation with cetuximab AUC and ln IC50 (lower = more
sensitive, so a negative correlation means the score is higher in sensitive lines) and the AUC for
separating the most sensitive quartile of lines by cetuximab AUC.

Inputs:  data/raw/gdsc/GDSC1_fitted_dose_response_27Oct23.xlsx
         data/raw/depmap/model_metadata/Model.csv
         data/raw/depmap/expression/OmicsExpressionProteinCodingGenesTPMLogp1.csv
         results/multinomial_classifier/ (classifier, via 09_external_cohorts.py)
         results/arm_scores/arm_gene_sets.tsv
         config/signatures/
         data/references/ (published signature gene lists)
Outputs: results/gdsc_cetuximab/gdsc_cetuximab_association.tsv
         results/gdsc_cetuximab/gdsc_cetuximab_scores.tsv
Usage:   python src/10_cetuximab/10_gdsc_cetuximab_cell_lines.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEP = ROOT / "data" / "raw" / "depmap"
OUT = ROOT / "results" / "gdsc_cetuximab"
OUT.mkdir(parents=True, exist_ok=True)
SCORES = ["pEMT_specificity", "malignant_arm_core", "malignant_arm", "malignant_consensus", "puram_pemt",
          "puram_epi_dif_1", "GS76", "zhou2025_predictive"]


def _load(path: Path, name: str):
    sys.path.insert(0, str(path.parent))
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    g = pd.read_excel(ROOT / "data" / "raw" / "gdsc" / "GDSC1_fitted_dose_response_27Oct23.xlsx")
    g = g[g["DRUG_ID"] == 1114].reset_index(drop=True)          # cetuximab
    model = pd.read_csv(DEP / "model_metadata" / "Model.csv", low_memory=False)
    model = model.dropna(subset=["SangerModelID"]).set_index("SangerModelID")
    model = model[~model.index.duplicated(keep="first")]
    g = g[g["SANGER_MODEL_ID"].isin(model.index)].copy()
    g["ModelID"] = model.loc[g["SANGER_MODEL_ID"], "ModelID"].values
    g["OncotreeCode"] = model.loc[g["SANGER_MODEL_ID"], "OncotreeCode"].values
    g = g.drop_duplicates("ModelID")

    expr = pd.read_csv(DEP / "expression" / "OmicsExpressionProteinCodingGenesTPMLogp1.csv", index_col=0,
                       low_memory=False)
    expr = expr[expr["IsDefaultEntryForModel"].astype(str).str.lower().isin(["yes", "true"])]
    expr = expr.set_index("ModelID").drop(columns=["ModelConditionID", "IsDefaultEntryForMC", "IsDefaultEntryForModel"],
                                          errors="ignore")
    expr.columns = [c.split(" (")[0] for c in expr.columns]
    expr = expr.loc[:, ~expr.columns.duplicated()]
    expr = expr[~expr.index.duplicated(keep="first")].apply(pd.to_numeric, errors="coerce")

    ext = _load(ROOT / "src/04_tcga_projection/09_external_cohorts.py", "ext")
    panel = _load(ROOT / "src/04_tcga_projection/07_emt_score_panel.py", "panel")
    from pemt import load_signatures
    from pemt.published import load_published

    sigs = load_signatures()
    zhou = load_published(ROOT / "data" / "references")["zhou2025_predictive"]
    arms = pd.read_csv(ROOT / "results/arm_scores/arm_gene_sets.tsv", sep="\t").set_index("score")["genes"].str.split(", ")

    panels = {"HNSC": g[g["TCGA_DESC"] == "HNSC"],
              "squamous (HNSC, ESCC, LUSC, CESC)": g[(g["TCGA_DESC"].isin(["HNSC", "LUSC", "CESC"]))
                                                     | ((g["TCGA_DESC"] == "ESCA") & (g["OncotreeCode"] == "ESCC"))]}
    rows, all_scores = [], []
    for name, d in panels.items():
        d = d[d["ModelID"].isin(expr.index)]
        e = expr.loc[d["ModelID"]].T                      # genes x lines
        s, *_ = ext.project(e)
        for k in ("malignant_arm_core", "malignant_arm", "malignant_consensus"):
            s[k], _, _ = panel.zmean(e, arms[k])
        s["puram_pemt"], _, _ = panel.zmean(e, sigs["puram_pemt"])
        s["puram_epi_dif_1"], _, _ = panel.zmean(e, sigs["puram_epi_dif_1"])
        s["GS76"], _ = panel.score_76gs(e)
        s["zhou2025_predictive"], _, _ = panel.zmean(e, zhou)
        s = s.join(d.set_index("ModelID")[["CELL_LINE_NAME", "TCGA_DESC", "AUC", "LN_IC50"]])
        s["panel"] = name
        all_scores.append(s)
        sens = (s["AUC"] <= s["AUC"].quantile(0.25)).astype(int)
        for sc in SCORES:
            r_auc = spearmanr(s[sc], s["AUC"]); r_ic = spearmanr(s[sc], s["LN_IC50"])
            rows.append({"panel": name, "n_lines": len(s), "score": sc,
                         "rho_with_AUC": r_auc[0], "p_AUC": r_auc[1],
                         "rho_with_lnIC50": r_ic[0], "p_lnIC50": r_ic[1],
                         "AUC_most_sensitive_quartile": roc_auc_score(sens, s[sc])})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "gdsc_cetuximab_association.tsv", sep="\t", index=False)
    pd.concat(all_scores).to_csv(OUT / "gdsc_cetuximab_scores.tsv", sep="\t")
    hn = pd.concat(all_scores).query("panel == 'HNSC'")
    print(f"HNSC lines: {len(hn)}, cetuximab AUC median {hn['AUC'].median():.2f} "
          f"(range {hn['AUC'].min():.2f} to {hn['AUC'].max():.2f}), lines with AUC < 0.8: {(hn['AUC'] < 0.8).sum()}")
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
