"""Two published partial-EMT gene sets as external controls: mammary EMT clusters and the HNSCC spatial pEMT programme.

The mammary partial EMT signature used by Sahoo et al. 2024, in whose breast cancer data the basal
subtype tracks partial more than full EMT, and the spatial partial EMT programme of Simkin et al. 2026
are scored beside the malignant and stromal cores.

Sources
  Knutsen E, et al. Front Oncol 2023, 13:1249895. The mammary EMT signature, clustered in TCGA BRCA into
  EMT_down, EMT_partial and EMT_up. The lists are taken as tabulated by Sahoo S, et al. iScience 2024,
  27:110116, Table S1 (mmc2.xlsx, https://ars.els-cdn.com/content/image/1-s2.0-S2589004224013415-mmc2.xlsx),
  committed as data/references/sahoo2024_mammary_emt_clusters.xlsx. One Ensembl-only entry is dropped.
  Simkin D, et al. Nat Genet 2026, 58:2240-2253. Spatial meta-programmes of 26 HNSCC
  (Final_Metaprograms_Extended.csv, https://ndownloader.figshare.com/files/50992821), committed as
  data/references/simkin2026_spatial_metaprograms.csv: the malignant pEMT programme (47 genes) and the
  Fibroblast and Epithelial programmes.

Analyses
  1. GSE103322 single cells: for each gene of each set, the share of total expression (linear scale)
     contributed by malignant cells and by fibroblasts, as in 09_gene_partition/01. Per set: genes
     detected, median malignant share, and the number of genes with a larger malignant than fibroblast
     share.
  2. TCGA-HNSC primaries (log2 TPM, the projection matrix): each set scored as the mean per-gene z-score
     (the zmean of 04_tcga_projection/07). Spearman correlation with the Basal centroid score of
     03_basal_centroid.py, the malignant and stromal cores and pEMT specificity. The Basal correlations
     of each partial set and its full EMT or fibroblast counterpart are compared with the Steiger test
     for dependent correlations sharing one variable.
  3. GSE65021 (platinum plus cetuximab, 14 long and 26 short PFS): the same scores, AUC for long PFS
     with the bootstrap interval of 10_cetuximab/01, two-sided Mann-Whitney p, and the Spearman
     correlation with the Basal score and the two cores. Any set with p < 0.05 is entered with the Basal
     score in a joint Firth model.

Inputs:  data/references/{sahoo2024_mammary_emt_clusters.xlsx,simkin2026_spatial_metaprograms.csv}
         data/processed/single_cell/HNSC_GSE103322/HNSC_GSE103322.h5ad
         data/processed/tcga_hnsc/tcga_star_tpm_log2_for_projection.tsv
         data/raw/external_cohorts/ (GSE65021, as for 10_cetuximab/01)
         results/basal_centroid/{TCGA_HNSC,GSE65021}_centroid_calls.tsv (from 03_basal_centroid.py)
         results/arm_scores/, results/cetuximab_cohort/GSE65021_scores.tsv,
         results/tcga_projection/emt_score_panel_scores.tsv
Outputs: results/external_pemt_controls/*.tsv
Usage:   python src/13_revision/10_external_pemt_controls.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm, spearmanr

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from pemt import firth as _firth  # noqa: E402

REF = ROOT / "data" / "references"
RES = ROOT / "results"
OUT = RES / "external_pemt_controls"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gene_sets() -> dict[str, list[str]]:
    t = pd.read_excel(REF / "sahoo2024_mammary_emt_clusters.xlsx")
    sets = {f"Knutsen_{c}": [g for g in t[c].dropna().astype(str).str.strip() if g and not g.startswith("ENSG")]
            for c in ("EMT_down", "EMT_partial", "EMT_up")}
    m = pd.read_csv(REF / "simkin2026_spatial_metaprograms.csv")
    for c, name in (("pEMT", "Simkin_pEMT"), ("Fibroblast", "Simkin_Fibroblast"), ("Epithelial", "Simkin_Epithelial")):
        sets[name] = m[c].dropna().astype(str).str.strip().tolist()
    return sets


def single_cell(sets: dict[str, list[str]]) -> pd.DataFrame:
    import anndata as ad
    import scipy.sparse as sp
    a = ad.read_h5ad(ROOT / "data/processed/single_cell/HNSC_GSE103322/HNSC_GSE103322.h5ad")
    major = a.obs["Celltype (major-lineage)"].astype(str)
    mal = major.eq("Malignant").to_numpy()
    caf = major.isin(["Fibroblasts", "Myofibroblasts"]).to_numpy()
    genes = sorted({g for v in sets.values() for g in v if g in a.var_names})
    X = a[:, genes].X
    X = np.expm1(X.toarray() if sp.issparse(X) else np.asarray(X))
    tot = X.sum(axis=0)
    per = pd.DataFrame({"gene": genes, "share_malignant": X[mal].sum(axis=0) / np.where(tot > 0, tot, np.nan),
                        "share_fibroblast": X[caf].sum(axis=0) / np.where(tot > 0, tot, np.nan)}).dropna()
    per.to_csv(OUT / "GSE103322_gene_shares.tsv", sep="\t", index=False, float_format="%.4g")
    per = per.set_index("gene")
    rows = []
    for k, v in sets.items():
        p = per.reindex([g for g in v if g in per.index])
        rows.append({"set": k, "genes_in_set": len(v), "genes_detected": len(p),
                     "median_malignant_share": p["share_malignant"].median(),
                     "median_fibroblast_share": p["share_fibroblast"].median(),
                     "n_malignant_predominant": int((p["share_malignant"] > p["share_fibroblast"]).sum())})
    return pd.DataFrame(rows)


def steiger(r12: float, r13: float, r23: float, n: int) -> float:
    """Two-sided p for r12 vs r13 (variable 1 shared), Steiger 1980 z with pooled r."""
    z12, z13 = np.arctanh(r12), np.arctanh(r13)
    rm = (r12 + r13) / 2
    f = (1 - r23) / (2 * (1 - rm ** 2))
    h = (1 - f * rm ** 2) / (1 - rm ** 2)
    c = (r23 * (1 - 2 * rm ** 2) - 0.5 * rm ** 2 * (1 - 2 * rm ** 2 - r23 ** 2)) / (1 - rm ** 2) ** 2
    z = (z12 - z13) * np.sqrt((n - 3) / (2 * (1 - c)))
    return float(2 * norm.sf(abs(z)))


def tcga(sets, zmean) -> pd.DataFrame:
    e = pd.read_csv(ROOT / "data/processed/tcga_hnsc/tcga_star_tpm_log2_for_projection.tsv", sep="\t", index_col=0)
    b = pd.read_csv(RES / "basal_centroid/TCGA_HNSC_centroid_calls.tsv", sep="\t", index_col=0)
    arms = pd.read_csv(RES / "arm_scores/TCGA-HNSC_arm_scores.tsv", sep="\t", index_col=0)
    spec = pd.read_csv(RES / "tcga_projection/emt_score_panel_scores.tsv", sep="\t", index_col=0)
    e = e[b.index.intersection(e.columns)]
    e = e[~e.index.duplicated()]
    s = pd.DataFrame({k: zmean(e, v)[0] for k, v in sets.items()})
    d = s.join(b[["basal_score"]]).join(arms[["malignant_arm_core", "stromal_arm_core"]]).join(spec[["pEMT_specificity"]])
    d.to_csv(OUT / "TCGA-HNSC_scores.tsv", sep="\t", float_format="%.5g")
    rows = []
    for k in sets:
        r = {"set": k, "n": len(d), "genes_scored": zmean(e, sets[k])[1]}
        for c in ("basal_score", "malignant_arm_core", "stromal_arm_core", "pEMT_specificity"):
            r[f"rho_{c}"], r[f"p_{c}"] = spearmanr(d[k], d[c])
        rows.append(r)
    out = pd.DataFrame(rows)
    rb = lambda x, y: spearmanr(d[x], d[y])[0]  # noqa: E731
    out.attrs["steiger"] = pd.DataFrame([{
        "comparison": f"rho(basal, {a}) vs rho(basal, {c})", "rho_a": rb("basal_score", a), "rho_b": rb("basal_score", c),
        "rho_between_sets": rb(a, c), "n": len(d),
        "p_steiger": steiger(rb("basal_score", a), rb("basal_score", c), rb(a, c), len(d))}
        for a, c in (("Knutsen_EMT_partial", "Knutsen_EMT_up"), ("Simkin_pEMT", "Simkin_Fibroblast"),
                     ("malignant_arm_core", "stromal_arm_core"))])
    return out


def gse65021(sets, zmean) -> pd.DataFrame:
    cx = _load(SRC / "10_cetuximab" / "01_cetuximab_cohort.py", "cx")
    x, _ = cx.read_series_matrix(cx.DATA / "GSE65021_series_matrix.txt.gz")
    e = cx.collapse(x, cx.read_annotation(cx.DATA / "GPL10558.annot.gz"))
    sc = pd.read_csv(RES / "cetuximab_cohort/GSE65021_scores.tsv", sep="\t", index_col=0)
    arms = pd.read_csv(RES / "arm_scores/GSE65021_arm_scores.tsv", sep="\t", index_col=0)
    b = pd.read_csv(RES / "basal_centroid/GSE65021_centroid_calls.tsv", sep="\t", index_col=0)
    e = e[sc.index]
    s = pd.DataFrame({k: zmean(e, v)[0] for k, v in sets.items()})
    d = s.join(sc[["long_pfs", "pEMT_specificity"]]).join(arms[["malignant_arm_core", "stromal_arm_core"]]) \
        .join(b[["basal_score"]])
    d.to_csv(OUT / "GSE65021_scores.tsv", sep="\t", float_format="%.5g")
    y = d["long_pfs"].to_numpy(int)
    z = lambda v: (v - v.mean()) / v.std()  # noqa: E731
    rows = []
    for k in list(sets) + ["malignant_arm_core", "stromal_arm_core", "basal_score"]:
        auc, lo, hi = cx.bootstrap_auc(y, d[k].to_numpy())
        r = {"score": k, "genes_scored": zmean(e, sets[k])[1] if k in sets else np.nan, "AUC": auc,
             "AUC_low": lo, "AUC_up": hi, "p_two_sided": mannwhitneyu(d.loc[y == 1, k], d.loc[y == 0, k]).pvalue,
             "rho_basal_score": spearmanr(d[k], d["basal_score"])[0],
             "rho_malignant_core": spearmanr(d[k], d["malignant_arm_core"])[0],
             "rho_stromal_core": spearmanr(d[k], d["stromal_arm_core"])[0]}
        if k in sets and r["p_two_sided"] < 0.05:
            f = _firth.firth(y, np.column_stack([np.ones(len(d)), z(d[k]), z(d["basal_score"])]), ["c", k, "basal"])
            r.update({"joint_OR_set": f[k][0], "joint_p_set": f[k][3], "joint_OR_basal": f["basal"][0],
                      "joint_p_basal": f["basal"][3]})
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    zmean = _load(SRC / "04_tcga_projection" / "07_emt_score_panel.py", "panel").zmean
    sets = gene_sets()
    pd.DataFrame([{"set": k, "n_genes": len(v), "genes": ", ".join(v)} for k, v in sets.items()]) \
        .to_csv(OUT / "gene_sets.tsv", sep="\t", index=False)
    core = set(pd.read_csv(RES / "arm_scores/arm_gene_sets.tsv", sep="\t", index_col=0)
               .loc["malignant_arm_core", "genes"].replace(" ", "").split(",")) \
        if (RES / "arm_scores/arm_gene_sets.tsv").exists() else set()
    for k, v in sets.items():
        print(f"{k}: {len(v)} genes, {len(core & set(v))} shared with the malignant core")
    sc = single_cell(sets)
    sc.to_csv(OUT / "GSE103322_set_partition.tsv", sep="\t", index=False, float_format="%.4g")
    print("\nGSE103322 cell of origin:\n" + sc.round(3).to_string(index=False))
    t = tcga(sets, zmean)
    t.to_csv(OUT / "tcga_correlations.tsv", sep="\t", index=False, float_format="%.4g")
    t.attrs["steiger"].to_csv(OUT / "tcga_basal_steiger.tsv", sep="\t", index=False, float_format="%.4g")
    print("\nTCGA-HNSC Spearman:\n" + t[[c for c in t.columns if not c.startswith("p_")]].round(3).to_string(index=False))
    print(t.attrs["steiger"].round(4).to_string(index=False))
    g = gse65021(sets, zmean)
    g.to_csv(OUT / "gse65021_pfs.tsv", sep="\t", index=False, float_format="%.4g")
    print("\nGSE65021 long vs short PFS:\n" + g.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
