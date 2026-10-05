"""Basal-subtype centroid score in GSE65021 and TCGA-HNSC, as a comparator for the malignant arm.

The malignant arm marks the TCGA Basal subtype (Figure 5d), so this stage asks whether the cetuximab
association in GSE65021 is the Basal subtype itself. The four HNSCC expression subtypes are called in
GSE65021 with the published centroid predictor, and the Basal call and a continuous Basal score are
tested against long progression-free survival.

Centroids
  Walter et al. 2013 (PLoS One 8:e56823) define the four subtypes, but their supplement does not
  deposit the centroid values (Table S1 is the samr list of differentially expressed genes). The
  centroid table that the TCGA 2015 HNSCC study used for its published calls is deposited on the GDC
  publication page of that study (https://gdc.cancer.gov/about-data/publications/hnsc_2014):
  classification_centroid_728genes.txt, 728 genes by Basal, Mesenchymal, Atypical and Classical.
  727 of the 728 genes are in Walter Table S1. The same page deposits the RNA-seq matrix that TCGA
  classified (transformed_rnaseq_to_classify.txt, 279 primaries), used here to check the method. Both
  files are downloaded from the GDC on first use into data/raw/tcga2015_subtype_centroids/.

Method (the centroid predictor of Walter et al. and TCGA)
  Expression on a log2 scale, each centroid gene median-centred across the samples of the cohort, and
  the Pearson correlation of each sample with each centroid over the shared genes. The call is the
  centroid with the highest correlation, and the continuous Basal score is the correlation with the
  Basal centroid. On the TCGA classification matrix (log2(x + 1)) this reproduces all 279 published
  calls.

GSE65021 expression is loaded as in 10_cetuximab/01_cetuximab_cohort.py (log2, GPL10558 annotation,
highest-mean probe per gene). Analyses in GSE65021: subtype distribution by outcome, Fisher's test of
the Basal call, AUC with a 2,000-resample bootstrap interval, two-sided Mann-Whitney p and Firth odds
ratio per SD of the Basal score, Spearman correlation with the malignant core and pEMT specificity,
joint Firth models of each primary measure with the Basal score, and the paired bootstrap difference
in AUC against the Basal score (as in 10_cetuximab/06_cetuximab_sensitivity_analyses.py). In the
TCGA-HNSC primaries (the projection matrix, log2 TPM): the same Basal score, its agreement with the
published calls, and its Spearman correlation with pEMT specificity and the malignant core.

Inputs:  data/raw/tcga2015_subtype_centroids/ (downloaded), data/raw/external_cohorts/GSE65021 files
         data/references/TCGA_HNSC_4class_expression_subtype.csv, zhou2025_cetuximab_predictive_9genes.txt
         data/processed/tcga_hnsc/tcga_star_tpm_log2_for_projection.tsv
         results/cetuximab_cohort/GSE65021_scores.tsv, results/arm_scores/{GSE65021,TCGA-HNSC}_arm_scores.tsv
         results/pemt_partition/puram_pemt_gene_partition.tsv, results/tcga_projection/emt_score_panel_scores.tsv
Outputs: results/basal_centroid/*.tsv
Usage:   python src/13_revision/03_basal_centroid.py
"""

from __future__ import annotations

import gzip
import importlib.util
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, kruskal, mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pemt.firth import firth  # noqa: E402

CDIR = ROOT / "data" / "raw" / "tcga2015_subtype_centroids"
REF = ROOT / "data" / "references"
RES = ROOT / "results"
OUT = RES / "basal_centroid"

GDC = "https://api.gdc.cancer.gov/data/"
FILES = {"classification_centroid_728genes.txt": "a6d6c614-90dd-4870-9daf-47ade3fad506",
         "transformed_rnaseq_to_classify.txt.gz": "1d4dc8c2-50e9-49b6-881f-346840034755"}
SUBTYPES = ["Basal", "Mesenchymal", "Atypical", "Classical"]
N_BOOT, SEED = 2000, 20260924   # the paired bootstrap seed of 10_cetuximab/06


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fetch() -> None:
    """Download the GDC files if absent. The matrix is stored gzipped."""
    CDIR.mkdir(parents=True, exist_ok=True)
    for fn, uuid in FILES.items():
        dest = CDIR / fn
        if dest.exists():
            continue
        raw = urllib.request.urlopen(GDC + uuid).read()
        dest.write_bytes(gzip.compress(raw) if fn.endswith(".gz") else raw)


def centroids() -> pd.DataFrame:
    return pd.read_csv(CDIR / "classification_centroid_728genes.txt", sep="\t", index_col=0)[SUBTYPES]


def classify(e_log2: pd.DataFrame, cen: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Correlation of each sample (columns of a genes by samples log2 matrix) with each centroid."""
    shared = cen.index.intersection(e_log2.index)
    x = e_log2.loc[shared]
    x = x.sub(x.median(axis=1), axis=0)
    r = pd.DataFrame({k: x.corrwith(cen.loc[shared, k]) for k in SUBTYPES})
    r.columns = [f"r_{k}" for k in SUBTYPES]
    r["call"] = r.idxmax(axis=1).str[2:]
    r["basal_score"] = r["r_Basal"]
    return r, len(shared)


def z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()   # ddof = 1, as in 10_cetuximab/05_direct_arm_scores.py


def published_calls() -> pd.Series:
    st = pd.read_csv(REF / "TCGA_HNSC_4class_expression_subtype.csv")
    return st.set_index("sample_barcode")["RNA_subtype"]


def tcga_method_check(cen: pd.DataFrame, pub: pd.Series):
    t = pd.read_csv(CDIR / "transformed_rnaseq_to_classify.txt.gz", sep="\t", index_col=0)
    t.columns = [c.replace(".", "-")[:15] for c in t.columns]
    r, n = classify(np.log2(t + 1), cen)
    agree = (r["call"] == pub.reindex(r.index)).mean()
    r.to_csv(OUT / "TCGA_own_matrix_centroid_calls.tsv", sep="\t", float_format="%.5g")
    return {"check": "TCGA classification matrix (GDC), log2(x+1)", "n_samples": len(r), "genes_used": n,
            "agreement_with_published_calls": agree}, r


def gse65021(cen: pd.DataFrame) -> pd.DataFrame:
    cx = _load(ROOT / "src" / "10_cetuximab" / "01_cetuximab_cohort.py", "cx")
    x, _ = cx.read_series_matrix(cx.DATA / "GSE65021_series_matrix.txt.gz")
    e = cx.collapse(x, cx.read_annotation(cx.DATA / "GPL10558.annot.gz"))
    sc = pd.read_csv(RES / "cetuximab_cohort" / "GSE65021_scores.tsv", sep="\t", index_col=0)
    arm = pd.read_csv(RES / "arm_scores" / "GSE65021_arm_scores.tsv", sep="\t", index_col=0)
    r, n = classify(e[sc.index], cen)
    print(f"GSE65021: {n} of {len(cen)} centroid genes present")
    # Sensitivity: drop the centroid genes that are among the 100 pEMT genes or the Zhou nine. DSG2,
    # the one malignant-core gene in the centroid, is among the pEMT genes.
    pemt = set(pd.read_csv(RES / "pemt_partition" / "puram_pemt_gene_partition.tsv", sep="\t")["gene"])
    zhou = set((REF / "zhou2025_cetuximab_predictive_9genes.txt").read_text().split())
    drop = (pemt | zhou) & set(cen.index)
    r2, n2 = classify(e[sc.index], cen.drop(index=list(drop)))
    r["basal_score_no_pemt_genes"] = r2["basal_score"]
    print(f"  sensitivity score without {len(drop)} pEMT or Zhou genes uses {n2} genes")
    d = sc[["long_pfs", "pEMT_specificity", "puram_pemt", "GS76"]].join(
        arm[["malignant_arm_core", "stromal_arm_core"]]).join(r)
    d.attrs["genes_used"] = n
    d.attrs["bootstrap_auc"] = cx.bootstrap_auc
    return d


def analyse(d: pd.DataFrame) -> None:
    y = d["long_pfs"].to_numpy(int)
    boot_auc = d.attrs["bootstrap_auc"]

    # 1. subtype distribution by outcome, and Fisher's test of the Basal call against long PFS
    tab = pd.crosstab(d["call"], d["long_pfs"]).reindex(SUBTYPES, fill_value=0)
    tab.columns = ["short_pfs", "long_pfs"]
    tab["total"] = tab.sum(axis=1)
    basal = (d["call"] == "Basal").astype(int)
    two = pd.crosstab(basal, d["long_pfs"]).reindex(index=[1, 0], columns=[1, 0], fill_value=0)
    odds, p_f = fisher_exact(two.values)
    tab.attrs = {}
    tab.to_csv(OUT / "GSE65021_subtype_distribution.tsv", sep="\t")
    fisher = pd.DataFrame([{"basal_long": two.loc[1, 1], "basal_short": two.loc[1, 0],
                            "nonbasal_long": two.loc[0, 1], "nonbasal_short": two.loc[0, 0],
                            "odds_ratio": odds, "fisher_p_two_sided": p_f}])
    fisher.to_csv(OUT / "GSE65021_basal_call_fisher.tsv", sep="\t", index=False, float_format="%.4g")
    print("\nSubtype calls by outcome:\n" + tab.to_string())
    print("Basal call against long PFS:\n" + fisher.round(4).to_string(index=False))

    # 2. single-score association for each centroid correlation, the core and pEMT specificity
    rows = []
    for sc in ["basal_score", "basal_score_no_pemt_genes", "r_Mesenchymal", "r_Atypical", "r_Classical",
               "malignant_arm_core", "pEMT_specificity"]:
        a, b = d.loc[y == 1, sc], d.loc[y == 0, sc]
        auc, lo, hi = boot_auc(y, d[sc].to_numpy())
        f = firth(y, np.column_stack([np.ones(len(d)), z(d[sc])]), ["const", sc])[sc]
        rows.append({"score": sc, "n": len(d), "median_long": a.median(), "median_short": b.median(),
                     "AUC": auc, "AUC_low": lo, "AUC_up": hi, "p_mannwhitney_two_sided": mannwhitneyu(a, b).pvalue,
                     "OR_firth_per_SD": f[0], "OR_low": f[1], "OR_up": f[2], "p_firth": f[3],
                     "rho_malignant_core": spearmanr(d[sc], d["malignant_arm_core"])[0],
                     "rho_pEMT_specificity": spearmanr(d[sc], d["pEMT_specificity"])[0],
                     "rho_stromal_core": spearmanr(d[sc], d["stromal_arm_core"])[0],
                     "rho_76GS": spearmanr(d[sc], d["GS76"])[0]})
    assoc = pd.DataFrame(rows)
    assoc.to_csv(OUT / "GSE65021_centroid_score_association.tsv", sep="\t", index=False, float_format="%.4g")
    print("\nAssociation with long PFS:\n" + assoc.round(3).to_string(index=False))

    # 3. joint Firth models: each primary measure with the Basal score
    jrows = []
    for primary in ["malignant_arm_core", "pEMT_specificity"]:
        X = np.column_stack([np.ones(len(d)), z(d[primary]), z(d["basal_score"])])
        res = firth(y, X, ["const", primary, "basal_score"])
        rho = spearmanr(d[primary], d["basal_score"])[0]
        for term in (primary, "basal_score"):
            o, lo, up, p = res[term]
            jrows.append({"model": f"{primary} + basal_score", "term": term, "n": len(d),
                          "rho_between_terms": rho, "OR_firth": o, "OR_low": lo, "OR_up": up, "p": p})
    joint = pd.DataFrame(jrows)
    joint.to_csv(OUT / "GSE65021_joint_firth_with_basal.tsv", sep="\t", index=False, float_format="%.4g")
    print("\nJoint Firth models:\n" + joint.round(4).to_string(index=False))

    # 4. paired bootstrap AUC difference against the Basal score
    rng = np.random.default_rng(SEED)
    boots = [b for b in (rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)) if 0 < y[b].sum() < len(b)]
    prow = []
    for primary in ["malignant_arm_core", "pEMT_specificity"]:
        a, o = d[primary].to_numpy(), d["basal_score"].to_numpy()
        diff = np.array([roc_auc_score(y[b], a[b]) - roc_auc_score(y[b], o[b]) for b in boots])
        prow.append({"primary": primary, "comparator": "basal_score", "AUC_primary": roc_auc_score(y, a),
                     "AUC_comparator": roc_auc_score(y, o), "difference": roc_auc_score(y, a) - roc_auc_score(y, o),
                     "diff_low": np.percentile(diff, 2.5), "diff_up": np.percentile(diff, 97.5),
                     "p_bootstrap_two_sided": min(1.0, 2 * min(np.mean(diff <= 0), np.mean(diff >= 0))),
                     "n_boot": len(boots)})
    paired = pd.DataFrame(prow)
    paired.to_csv(OUT / "GSE65021_paired_auc_vs_basal.tsv", sep="\t", index=False, float_format="%.4g")
    print("\nPaired AUC difference against the Basal score:\n" + paired.round(3).to_string(index=False))


def tcga(cen: pd.DataFrame, pub: pd.Series, own: pd.DataFrame) -> pd.DataFrame:
    ref = pd.read_csv(RES / "tcga_projection" / "emt_score_panel_scores.tsv", sep="\t", index_col=0)
    e = pd.read_csv(ROOT / "data" / "processed" / "tcga_hnsc" / "tcga_star_tpm_log2_for_projection.tsv",
                    sep="\t", index_col=0)
    e = e[ref.index.intersection(e.columns)]
    e = e[~e.index.duplicated()]
    arm = pd.read_csv(RES / "arm_scores" / "TCGA-HNSC_arm_scores.tsv", sep="\t", index_col=0)
    r, n = classify(e, cen)
    d = r.join(ref[["pEMT_specificity", "puram_pemt", "GS76"]]).join(arm[["malignant_arm_core", "stromal_arm_core"]])
    d["published_call"] = pd.Series(d.index.str[:15], index=d.index).map(pub)
    d["basal_score_tcga_matrix"] = pd.Series(d.index.str[:15], index=d.index).map(own["basal_score"])
    d.to_csv(OUT / "TCGA_HNSC_centroid_calls.tsv", sep="\t", float_format="%.5g")
    m = d.dropna(subset=["published_call"])
    agree = (m["call"] == m["published_call"]).mean()
    # centring over the 279 classified primaries only, as TCGA did
    r279, _ = classify(e[m.index], cen)
    agree279 = (r279["call"] == m["published_call"]).mean()
    kw = kruskal(*[m.loc[m["published_call"] == g, "basal_score"] for g in SUBTYPES]).pvalue
    rows = [{"cohort": "TCGA-HNSC primaries", "subset": "all", "n": len(d), "genes_used": n,
             "rho_pEMT_specificity": spearmanr(d["basal_score"], d["pEMT_specificity"])[0],
             "rho_malignant_core": spearmanr(d["basal_score"], d["malignant_arm_core"])[0],
             "rho_stromal_core": spearmanr(d["basal_score"], d["stromal_arm_core"])[0],
             "rho_76GS": spearmanr(d["basal_score"], d["GS76"])[0]},
            {"cohort": "TCGA-HNSC primaries", "subset": "with published call", "n": len(m), "genes_used": n,
             "rho_pEMT_specificity": spearmanr(m["basal_score"], m["pEMT_specificity"])[0],
             "rho_malignant_core": spearmanr(m["basal_score"], m["malignant_arm_core"])[0],
             "rho_stromal_core": spearmanr(m["basal_score"], m["stromal_arm_core"])[0],
             "rho_76GS": spearmanr(m["basal_score"], m["GS76"])[0],
             "call_agreement_with_published": agree,
             "call_agreement_centred_on_279": agree279,
             "rho_basal_score_vs_tcga_matrix": spearmanr(m["basal_score"], m["basal_score_tcga_matrix"])[0],
             "kruskal_p_basal_score_by_published_call": kw,
             **{f"median_basal_score_{g}": m.loc[m["published_call"] == g, "basal_score"].median() for g in SUBTYPES}}]
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "TCGA_HNSC_basal_score_summary.tsv", sep="\t", index=False, float_format="%.4g")
    conf = pd.crosstab(m["published_call"], m["call"]).reindex(index=SUBTYPES, columns=SUBTYPES, fill_value=0)
    conf.to_csv(OUT / "TCGA_HNSC_call_confusion.tsv", sep="\t")
    print(f"\nTCGA-HNSC: {n} centroid genes present, {len(d)} primaries, {len(m)} with published calls")
    print(f"call distribution (all primaries): {d['call'].value_counts().reindex(SUBTYPES).to_dict()}")
    print("published calls (rows) against these calls (columns):\n" + conf.to_string())
    print(out.round(3).T.to_string())
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fetch()
    cen = centroids()
    pub = published_calls()
    chk, own = tcga_method_check(cen, pub)
    print(f"Method check: {chk}")
    d = gse65021(cen)
    d.drop(columns=["puram_pemt"]).to_csv(OUT / "GSE65021_centroid_calls.tsv", sep="\t", float_format="%.5g")
    analyse(d)
    t = tcga(cen, pub, own)
    pd.DataFrame([chk, {"check": "GSE65021 genes used", "n_samples": len(d), "genes_used": d.attrs["genes_used"]},
                  {"check": "TCGA projection matrix (log2 TPM) against published calls",
                   "n_samples": int(t.loc[1, "n"]), "genes_used": int(t.loc[1, "genes_used"]),
                   "agreement_with_published_calls": t.loc[1, "call_agreement_with_published"]}]) \
        .to_csv(OUT / "method_checks.tsv", sep="\t", index=False, float_format="%.4g")


if __name__ == "__main__":
    main()
