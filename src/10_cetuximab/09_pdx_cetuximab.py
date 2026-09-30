"""Test the pEMT arms against cetuximab response in two head and neck cancer PDX panels.

No second public patient cohort with pre-treatment expression and cetuximab outcome is available.
Two PDX panels are public with per-model response labels in their GEO series matrices:

  GSE84713  Klinghammer et al., Int J Cancer 2017. 28 PDX (26 patients) on Affymetrix HG-U133 Plus 2
            (GCRMA log2, deposited). Untreated third-passage tumours. Responder or non-responder by
            relative tumour volume after 3 weeks of cetuximab. 19 responders, 9 non-responders.
  GSE183881 Bouhaddou et al., JCI Insight 2021. RNA-seq of untreated PDX (human reads separated from
            mouse by Xenome, RSEM isoform TPM summed to genes via GENCODE v38 HGNC metadata).
            Sensitive (>= 50% reduction in endpoint volume relative to vehicle) or resistant. 17 of
            65 models tested (12 sensitive, 5 resistant).

In a PDX the stroma is murine, so the human expression readout comes from the malignant compartment
and the panels test the malignant arm. The stromal arm is reported but cannot be interpreted,
because the human fibroblasts that express it are absent.

Scores, all standardised within each panel as in every other cohort: pEMT specificity (trained
classifier), the malignant and stromal cores and total-share arms, the malignant consensus arm, the
Puram pEMT signature, Puram epithelial differentiation, 76GS, and the Zhou nine-gene predictor. The
Zhou predictor was derived on GSE65021, and Zhou and colleagues also examined GSE84713, so it is not
independent of that panel. Each score is tested by two-sided Mann-Whitney p, bootstrap AUC and Firth
odds ratio per SD. In GSE84713 the two pairs of models from one patient (11269A/B, 11437A/B) are
averaged for the primary analysis, and all 28 models form a sensitivity analysis. The primary
analyses of the two panels are pooled by fixed-effect inverse-variance weighting of the log Firth
odds ratios. strengthen() adds a continuous endpoint, paired AUC comparisons and the exploratory
combined marker.

Inputs:  data/raw/pdx/GSE84713_series_matrix.txt.gz
         data/raw/pdx/GSE183881_series_matrix.txt.gz
         data/raw/pdx/GSE183881_expression-human-clsfy.summary.isoforms.TPM.tab.gz
         data/raw/pdx/gencode.v38.metadata.HGNC.gz
         data/raw/external_cohorts/GPL570.annot.gz
         results/multinomial_classifier/ (classifier, via 09_external_cohorts.py)
         results/arm_scores/arm_gene_sets.tsv
         config/signatures/
         data/references/ (published signature gene lists)
         data/references/klinghammer2020_table1_cetuximab_RTV.tsv
Outputs: results/pdx_cetuximab/GSE84713_scores.tsv
         results/pdx_cetuximab/GSE183881_scores.tsv
         results/pdx_cetuximab/pdx_cetuximab_association.tsv
         results/pdx_cetuximab/pdx_cetuximab_pooled.tsv
         results/pdx_cetuximab/GSE84713_continuous_RTV.tsv
         results/pdx_cetuximab/GSE84713_RTV_overlap.tsv
         results/pdx_cetuximab/pdx_paired_auc.tsv
         results/pdx_cetuximab/pdx_combined_marker.tsv
Usage:   python src/10_cetuximab/09_pdx_cetuximab.py
"""

from __future__ import annotations

import gzip
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
D = ROOT / "data" / "raw" / "pdx"
OUT = ROOT / "results" / "pdx_cetuximab"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE.parent))
from pemt import firth as _firth  # noqa: E402

SCORES = ["pEMT_specificity", "malignant_arm_core", "malignant_arm", "malignant_consensus", "stromal_arm_core",
          "stromal_arm", "puram_pemt", "puram_epi_dif_1", "GS76", "zhou2025_predictive"]


def _load(path: Path, name: str):
    sys.path.insert(0, str(path.parent))
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def characteristics(path: Path) -> pd.DataFrame:
    rows, desc = {}, []
    with gzip.open(path, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("!Sample_geo_accession"):
                ids = [x.strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_title"):
                rows["title"] = [x.strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_description"):
                desc.append([x.strip('"') for x in line.rstrip("\n").split("\t")[1:]])
            elif line.startswith("!Sample_characteristics_ch1"):
                vals = [x.strip('"') for x in line.rstrip("\n").split("\t")[1:]]
                key = vals[0].split(":", 1)[0].strip()
                rows[key] = [v.split(":", 1)[1].strip() if ":" in v else v for v in vals]
            elif line.startswith("!series_matrix_table_begin"):
                break
    m = pd.DataFrame(rows, index=ids)
    for i, d in enumerate(desc):
        m[f"description_{i}"] = d
    return m


def score(e: pd.DataFrame) -> pd.DataFrame:
    ext = _load(ROOT / "src/04_tcga_projection/09_external_cohorts.py", "ext")
    panel = _load(ROOT / "src/04_tcga_projection/07_emt_score_panel.py", "panel")
    from pemt import load_signatures
    from pemt.published import load_published

    sigs = load_signatures()
    arms = pd.read_csv(ROOT / "results/arm_scores/arm_gene_sets.tsv", sep="\t").set_index("score")["genes"].str.split(", ")
    s, *_ = ext.project(e)
    for k in ("malignant_arm_core", "malignant_arm", "malignant_consensus", "stromal_arm_core", "stromal_arm"):
        s[k], _, _ = panel.zmean(e, arms[k])
    s["puram_pemt"], _, _ = panel.zmean(e, sigs["puram_pemt"])
    s["puram_epi_dif_1"], _, _ = panel.zmean(e, sigs["puram_epi_dif_1"])
    s["GS76"], _ = panel.score_76gs(e)
    s["zhou2025_predictive"], _, _ = panel.zmean(e, load_published(ROOT / "data" / "references")["zhou2025_predictive"])
    return s


def association(df: pd.DataFrame, cohort: str, label: str) -> pd.DataFrame:
    cx = _load(ROOT / "src" / "10_cetuximab" / "01_cetuximab_cohort.py", "cx")
    y = df["response"].to_numpy(int)
    z = lambda v: (v - v.mean()) / v.std()  # noqa: E731
    rows = []
    for sc in SCORES:
        a, b = df.loc[y == 1, sc], df.loc[y == 0, sc]
        auc, lo, hi = cx.bootstrap_auc(y, df[sc].to_numpy())
        f = _firth.firth(y, np.column_stack([np.ones(len(df)), z(df[sc])]), ["c", sc])[sc]
        rows.append({"cohort": cohort, "analysis": label, "score": sc, "n_responders": int(y.sum()),
                     "n_non_responders": int((1 - y).sum()), "AUC": auc, "AUC_low": lo, "AUC_up": hi,
                     "p_two_sided": mannwhitneyu(a, b).pvalue, "OR_firth": f[0], "OR_low": f[1], "OR_up": f[2],
                     "p_firth": f[3], "rho_pEMT_specificity": spearmanr(df[sc], df["pEMT_specificity"])[0]})
    return pd.DataFrame(rows)


def gse84713() -> list[pd.DataFrame]:
    ext = _load(ROOT / "src/04_tcga_projection/09_external_cohorts.py", "ext84")
    expr, _ = ext.read_series_matrix(D / "GSE84713_series_matrix.txt.gz")
    e = ext.collapse(expr, ext.read_annotation(ROOT / "data/raw/external_cohorts/GPL570.annot.gz"))
    meta = characteristics(D / "GSE84713_series_matrix.txt.gz")
    meta["model"] = meta["description_1"]
    meta["patient"] = meta["model"].str.replace(r"[AB]$", "", regex=True)
    meta["response"] = (meta["response to cetuximab"] == "responder").astype(int)
    # score on all 28 models, then average the two same-patient pairs for the primary analysis
    s = score(e[meta.index]).join(meta[["model", "patient", "response", "molecular subtype"]])
    s.to_csv(OUT / "GSE84713_scores.tsv", sep="\t")
    per_patient = s.groupby("patient").agg({**{c: "mean" for c in SCORES}, "response": "max",
                                            "molecular subtype": "first"})
    discordant = s.groupby("patient")["response"].nunique().gt(1)
    print(f"GSE84713: {len(s)} models, {per_patient.shape[0]} patients, "
          f"responders {int(per_patient.response.sum())}/{len(per_patient)}, "
          f"same-patient pairs with discordant labels: {int(discordant.sum())}")
    return [association(per_patient, "GSE84713", "per patient (primary)"),
            association(s, "GSE84713", "all models")]


def gse183881() -> list[pd.DataFrame]:
    meta = characteristics(D / "GSE183881_series_matrix.txt.gz")
    meta = meta[meta["cetuximab response"].isin(["Sensitive", "Resistant"])]
    tx = pd.read_csv(D / "GSE183881_expression-human-clsfy.summary.isoforms.TPM.tab.gz", sep="\t", index_col=0)
    tx.columns = [re.search(r"(JGHL\d+)", c).group(1) for c in tx.columns]
    m = pd.read_csv(D / "gencode.v38.metadata.HGNC.gz", sep="\t", header=None, names=["tx", "sym", "hgnc"])
    m["tx"] = m["tx"].str.split(".").str[0]
    sym = m.drop_duplicates("tx").set_index("tx")["sym"]
    tx = tx[tx.index.isin(sym.index)]
    gene = tx.groupby(sym.reindex(tx.index).values).sum()        # isoform TPM summed to gene TPM
    e = np.log2(gene[meta["title"].tolist()] + 1)
    e.columns = meta.index
    s = score(e).join(meta[["title", "cetuximab response"]])
    s["response"] = (s["cetuximab response"] == "Sensitive").astype(int)
    s.to_csv(OUT / "GSE183881_scores.tsv", sep="\t")
    print(f"GSE183881: {len(s)} tested models, sensitive {int(s.response.sum())}, genes {e.shape[0]}")
    return [association(s, "GSE183881", "tested models")]


def main() -> None:
    res = pd.concat(gse84713() + gse183881(), ignore_index=True)
    res.to_csv(OUT / "pdx_cetuximab_association.tsv", sep="\t", index=False)
    show = ["cohort", "analysis", "score", "n_responders", "n_non_responders", "AUC", "AUC_low", "AUC_up",
            "p_two_sided", "OR_firth", "p_firth", "rho_pEMT_specificity"]
    print(res[show].round(3).to_string(index=False))

    # fixed-effect pooling of the two panels (primary analyses), inverse variance on log Firth OR
    from scipy.stats import norm
    prim = res[res["analysis"].isin(["per patient (primary)", "tested models"])]
    rows = []
    for sc, d in prim.groupby("score", sort=False):
        b = np.log(d["OR_firth"]); se = (np.log(d["OR_up"]) - np.log(d["OR_low"])) / (2 * 1.96)
        w = 1 / se ** 2; m = float((w * b).sum() / w.sum()); s = float(np.sqrt(1 / w.sum()))
        rows.append({"score": sc, "pooled_OR": np.exp(m), "low": np.exp(m - 1.96 * s), "up": np.exp(m + 1.96 * s),
                     "p": 2 * norm.sf(abs(m / s)), "n_models": int((d["n_responders"] + d["n_non_responders"]).sum())})
    pooled = pd.DataFrame(rows)
    pooled.to_csv(OUT / "pdx_cetuximab_pooled.tsv", sep="\t", index=False)
    print("\nFixed-effect pooling of the two panels:\n" + pooled.round(3).to_string(index=False))
    strengthen()


def strengthen() -> None:
    """Run three further PDX analyses.

    1. Continuous endpoint. Klinghammer et al. 2020 (Oncotarget 11:3688, Table 1) report cetuximab
       relative tumour volume (RTV, lower is better) for 33 PDX, 15 of which are GSE84713 models
       (10980 = 10980B, 11437 = mean of 11437A/B). Spearman correlation of each score with RTV, and
       agreement of that experiment's RECIST-like category with the deposited GEO label.
    2. Paired AUC comparison, as in the patient cohort: malignant core and pEMT specificity against 76GS,
       Puram epithelial differentiation and the Puram pEMT signature (2,000 paired bootstraps).
    3. The exploratory combination from the patient cohort (pEMT specificity + Puram epithelial
       differentiation, equal weights of standardised scores), evaluated without refitting.
    """
    rng = np.random.default_rng(20260924)
    s84 = pd.read_csv(OUT / "GSE84713_scores.tsv", sep="\t", index_col=0)
    s84["model_key"] = s84["model"].astype(str).str.replace(r"[AB]$", "", regex=True)
    per = s84.groupby("model_key").agg({**{c: "mean" for c in SCORES}, "response": "max"})
    rtv = pd.read_csv(ROOT / "data" / "references" / "klinghammer2020_table1_cetuximab_RTV.tsv", sep="\t", comment="#", dtype={"model": str})
    rtv = rtv.set_index("model")
    m = per.join(rtv, how="inner")
    rows = [{"score": sc, "n_models": len(m), "spearman_with_RTV": spearmanr(m[sc], m["cetuximab_RTV"])[0],
             "p": spearmanr(m[sc], m["cetuximab_RTV"])[1]} for sc in SCORES]
    cont = pd.DataFrame(rows)
    cont.to_csv(OUT / "GSE84713_continuous_RTV.tsv", sep="\t", index=False)
    m["responder_2020"] = m["cetuximab_category"].isin(["CR", "PR", "SD"]).astype(int)
    agree = float((m["responder_2020"] == m["response"]).mean())
    m[["cetuximab_RTV", "cetuximab_category", "response"]].to_csv(OUT / "GSE84713_RTV_overlap.tsv", sep="\t")
    print(f"\nGSE84713 x Klinghammer 2020 RTV: {len(m)} models, GEO label agrees with the 2020 category "
          f"(CR/PR/SD vs PD) in {agree:.0%}\n" + cont.round(3).to_string(index=False))

    s183 = pd.read_csv(OUT / "GSE183881_scores.tsv", sep="\t", index_col=0)
    panels = {"GSE84713": per.reset_index(), "GSE183881": s183}
    z = lambda v: (v - v.mean()) / v.std()  # noqa: E731
    prow, crow = [], []
    for name, d in panels.items():
        y = d["response"].to_numpy(int)
        boots = [b for b in (rng.integers(0, len(y), len(y)) for _ in range(2000)) if 0 < y[b].sum() < len(b)]
        for prim in ("malignant_arm_core", "pEMT_specificity"):
            for other in ("GS76", "puram_epi_dif_1", "puram_pemt"):
                a, o = d[prim].to_numpy(), d[other].to_numpy()
                diffs = np.array([roc_auc_score(y[b], a[b]) - roc_auc_score(y[b], o[b]) for b in boots])
                prow.append({"panel": name, "primary": prim, "comparator": other,
                             "difference": roc_auc_score(y, a) - roc_auc_score(y, o),
                             "low": np.percentile(diffs, 2.5), "up": np.percentile(diffs, 97.5)})
        comb = (z(d["pEMT_specificity"]) + z(d["puram_epi_dif_1"])).to_numpy()
        vals = [roc_auc_score(y[b], comb[b]) for b in boots]
        crow.append({"panel": name, "AUC_combined": roc_auc_score(y, comb),
                     "low": np.percentile(vals, 2.5), "up": np.percentile(vals, 97.5),
                     "AUC_pEMT_specificity": roc_auc_score(y, d["pEMT_specificity"]),
                     "AUC_malignant_core": roc_auc_score(y, d["malignant_arm_core"])})
    paired = pd.DataFrame(prow)
    paired.to_csv(OUT / "pdx_paired_auc.tsv", sep="\t", index=False)
    combined = pd.DataFrame(crow)
    combined.to_csv(OUT / "pdx_combined_marker.tsv", sep="\t", index=False)
    print("\nPaired AUC differences in the PDX panels:\n" + paired.round(3).to_string(index=False))
    print("\nPatient-cohort combination evaluated in the PDX panels:\n" + combined.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
