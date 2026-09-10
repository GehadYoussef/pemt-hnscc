"""Third external cohort: CPTAC-3 HNSCC (Huang et al., Cancer Cell 2021).

Added after the first two external cohorts disagreed. GSE41613 (HPV-negative oral cavity, surgically
treated, Affymetrix) replicated the adjusted survival effect at HR 1.79 per SD while GSE65858
(mixed-site, mixed-treatment, Illumina) was null at HR 0.99, and 11_subgroup_robustness.py ruled out
site and HPV status as the explanation. CPTAC-3 is the informative tie-breaker for two reasons: it is
RNA-seq quantified by the same GDC STAR pipeline as our TCGA training data, so platform is no longer a
candidate explanation, and like GSE41613 it is an HPV-negative surgical cohort.

Cohort (Huang C et al., Cancer Cell 2021;39:361-379.e16, PMID 33417831): 110 primary tumours with
RNA-seq, 107 with overall survival, 41 deaths, median follow-up 27.2 months. HPV-negative by design
(HPV_inference NO for all cases). Site mix is roughly half larynx and half oral cavity, so it is not a
site-matched replicate of GSE41613; that is deliberate, since site was already excluded as a modifier.

Survival note: the LinkedOmics survival columns are the frozen 2021 publication snapshot with only 16
deaths and are NOT used. OS_days/OS_event come from the current GDC clinical release (41 deaths),
corroborated by cBioPortal's ohnca_cptac_gdc study (39 deceased).

Procedure identical to the other cohorts: classifier genes z-scored within the cohort, elastic-net
coefficients applied, pEMT specificity = P(pEMT-high) - max(other two); the EMT score panel and the
five published programmes computed on the same samples; Cox HR per SD, univariable and adjusted.

Outputs (results/tcga_projection/external/):
  CPTAC_HNSCC_scores.tsv, CPTAC_HNSCC_cox.tsv, and CPTAC rows appended to external_cox_summary.tsv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import load_config, load_signatures, project_root  # noqa: E402
from _lib.published import load_published  # noqa: E402
from _lib.survival import fit_cox  # noqa: E402

cfg = load_config()
ROOT = project_root()
CPTAC = ROOT / "data" / "raw" / "external_cohorts" / "cptac"
OUT = ROOT / cfg["paths"]["results_dir"] / "tcga_projection" / "external"
REF = ROOT / cfg["paths"]["references_dir"]
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module  # noqa: E402
_panel = import_module("07_emt_score_panel")
_ext = import_module("09_external_cohorts")

SCORE_LABELS = _ext.SCORE_LABELS


def load_expression() -> pd.DataFrame:
    """log2(TPM+1), genes x tumours, one tumour per patient, columns are CPTAC case ids."""
    f = CPTAC / "gdc" / "cptac_hnscc_gdc_log2tpm1_symbol_TUMOR.tsv"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found; run the CPTAC acquisition step first")
    e = pd.read_csv(f, sep="\t", index_col=0)
    e = e[~e.index.duplicated(keep="first")]
    return e


def load_clinical() -> pd.DataFrame:
    c = pd.read_csv(CPTAC / "cptac_hnscc_clinical.tsv", sep="\t")
    c = c.set_index("case_id")
    out = pd.DataFrame(index=c.index)
    out["OS_time"] = pd.to_numeric(c["OS_months"], errors="coerce")
    out["OS_event"] = pd.to_numeric(c["OS_event"], errors="coerce")
    age_col = "LO_age" if "LO_age" in c.columns else ("age_at_index" if "age_at_index" in c.columns else None)
    out["age"] = pd.to_numeric(c[age_col], errors="coerce") if age_col else np.nan
    stage_map = {"Stage I": 1, "Stage II": 2, "Stage III": 3, "Stage IV": 4, "Stage IVA": 4, "Stage IVB": 4, "Stage IVC": 4}
    st = c["ajcc_pathologic_stage"].astype(str).str.strip() if "ajcc_pathologic_stage" in c.columns else pd.Series(index=c.index, dtype=str)
    out["stage_num"] = st.map(stage_map)
    site = c["LO_tumor_site_curated"].astype(str).str.lower() if "LO_tumor_site_curated" in c.columns else pd.Series("", index=c.index)
    out["site_group"] = np.select(
        [site.str.contains("oral|lip|tongue|mouth", na=False), site.str.contains("laryn|hypophar", na=False), site.str.contains("orophar", na=False)],
        ["oral_cavity", "larynx_hypopharynx", "oropharynx"], default="other")
    out["site_oral_cavity"] = (out["site_group"] == "oral_cavity").astype(int)
    out["site_larynx_hypopharynx"] = (out["site_group"] == "larynx_hypopharynx").astype(int)
    out["hpv_status"] = "HPV-"
    out["primary"] = True
    return out


def main() -> None:
    e = load_expression()
    clin = load_clinical()
    probs, n_shared, n_nz_shared, n_nz = _ext.project(e)
    scores = probs.copy()
    sigs = load_signatures()
    hm = [l.strip() for l in (REF / "emt_scores" / "hallmark_emt_genes.txt").read_text().splitlines() if l.strip()]
    scores["GS76"], _ = _panel.score_76gs(e)
    scores["KS"], _, _ = _panel.score_ks(e)
    scores["hallmark_EMT"], _, _ = _panel.zmean(e, hm)
    scores["puram_pemt"], _, _ = _panel.zmean(e, sigs["puram_pemt"])
    scores["puram_epi_dif_1"], _, _ = _panel.zmean(e, sigs["puram_epi_dif_1"])
    scores["MLR_mu"] = _panel.score_mlr(e)
    for key, genes in load_published(REF).items():
        scores[key], nf, nt = _panel.zmean(e, genes)
        print(f"  {key}: {nf}/{nt} genes present")

    df = clin.join(scores, how="inner")
    df.to_csv(OUT / "CPTAC_HNSCC_scores.tsv", sep="\t")
    usable = df.dropna(subset=["OS_time", "OS_event"])
    print(f"\n[CPTAC-3 HNSCC] {len(df)} tumours projected, {len(usable)} with OS, {int(usable['OS_event'].sum())} deaths, "
          f"median follow-up {usable['OS_time'].median():.1f} months")
    print(f"  classifier genes present {n_shared}, non-zero-coefficient genes present {n_nz_shared}/{n_nz}")
    print("  site:", df["site_group"].value_counts().to_dict())

    covs = ["stage_num", "age", "site_oral_cavity", "site_larynx_hypopharynx"]
    rows = []
    for sc in SCORE_LABELS:
        if sc not in df.columns:
            continue
        u, n_u, e_u = _ext.cox_per_sd(df, sc, [])
        a, n_a, e_a = _ext.cox_per_sd(df, sc, covs)
        rows.append({"cohort": "CPTAC_HNSCC", "score": sc, "label": SCORE_LABELS[sc],
                     "HR_per_SD_univariable": u["HR"], "CI_low_uni": u["HR_low_95"], "CI_up_uni": u["HR_up_95"],
                     "p_univariable": u["p_value"], "n_uni": n_u, "events_uni": e_u,
                     "HR_per_SD_adjusted": a["HR"], "CI_low_adj": a["HR_low_95"], "CI_up_adj": a["HR_up_95"],
                     "p_adjusted": a["p_value"], "n_adj": n_a, "events_adj": e_a,
                     "adjusted_for": ", ".join(covs), "classifier_genes_present": n_shared,
                     "nonzero_coefficient_genes_present": f"{n_nz_shared}/{n_nz}"})
    cox = pd.DataFrame(rows)
    cox.to_csv(OUT / "CPTAC_HNSCC_cox.tsv", sep="\t", index=False)
    print(cox[["label", "HR_per_SD_univariable", "p_univariable", "HR_per_SD_adjusted", "p_adjusted", "n_adj"]].round(3).to_string(index=False))

    summ = OUT / "external_cox_summary.tsv"
    if summ.exists():
        prev = pd.read_csv(summ, sep="\t")
        prev = prev[prev["cohort"] != "CPTAC_HNSCC"]
        pd.concat([prev, cox], ignore_index=True).to_csv(summ, sep="\t", index=False)
        print(f"appended CPTAC rows to {summ}")


if __name__ == "__main__":
    main()
