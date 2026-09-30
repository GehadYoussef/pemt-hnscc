"""Replicate the malignant/stromal partition of the pEMT genes on a composition-independent measure.

Share of total expression depends on how many malignant cells and fibroblasts a dataset contains.
The composition-independent measure is the mean expression per cell of each type. Every share is
computed on the linear scale (counts per 10,000). The TISCH2 matrices for GSE103322 and GSE150321
are natural-log log1p of CP10K, as verified by the per-cell sums, and are returned to CP10K with
expm1. GSE150321 has two patients. GSE181919 (Choi et al., Nat Commun 2023, 23 patients,
author-annotated malignant cells and fibroblasts) is restricted to tumour tissue (primary tumour
and lymph-node metastasis) and is also analysed per patient.

For each Puram pEMT gene and dataset:
  share_total      malignant share of the summed malignant + fibroblast expression
  share_per_cell   mean per malignant cell / (mean per malignant cell + mean per fibroblast)
  arm_per_cell     malignant if share_per_cell > 0.5
In GSE181919, share_per_cell is also computed per patient (patients with >= 20 malignant cells and
>= 20 fibroblasts in tumour tissue) and summarised as the median across patients and the fraction of
patients on the malignant side. The per-dataset summary gives the arm counts, the Spearman
correlation of share_per_cell with the discovery share of total expression, and a one-sided
Mann-Whitney test of the hemidesmosome and laminin-332 genes against the rest. Datasets are
compared pairwise by Spearman correlation of share_per_cell and Cohen's kappa of the arm calls.

Inputs:  results/pemt_partition/puram_pemt_gene_partition.tsv
         data/processed/single_cell/HNSC_GSE103322/HNSC_GSE103322.h5ad
         data/processed/single_cell/LSCC_GSE150321/LSCC_GSE150321.h5ad
         data/processed/GSE181919/GSE181919_subset.h5ad
Outputs: results/pemt_partition/replication_per_gene.tsv
         results/pemt_partition/replication_GSE181919_per_patient.tsv
         results/pemt_partition/replication_GSE181919_patient_summary.tsv
         results/pemt_partition/replication_summary.tsv
         results/pemt_partition/replication_cross_dataset.tsv
Usage:   python src/09_gene_partition/02_partition_replication.py
         Run after 09_gene_partition/01_pemt_gene_partition.py and
         01_single_cell_qc/05_prepare_gse181919.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SC = ROOT / "data" / "processed" / "single_cell"
OUT = ROOT / "results" / "pemt_partition"
HEMI = {"LAMA3", "LAMB3", "LAMC2", "COL17A1", "ITGA6", "ITGB4", "ITGA3", "DST", "PLEC"}
MIN_CELLS = 20


def kappa(a: pd.Series, b: pd.Series) -> float:
    po = float((a == b).mean())
    pe = sum(float((a == k).mean()) * float((b == k).mean()) for k in set(a) | set(b))
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def load(name: str, genes: list[str]):
    """Return (linear CP10K cells x genes DataFrame, malignant mask, fibroblast mask, patient ids)."""
    import anndata as ad
    import scipy.sparse as sp

    if name == "GSE181919":
        a = ad.read_h5ad(ROOT / "data" / "processed" / "GSE181919" / "GSE181919_subset.h5ad")
        a = a[a.obs["tissue.type"].isin(["CA", "LN"])].copy()
        X = a.X                                           # already linear CP10K
        ct = a.obs["cell.type"].astype(str)
        mal, caf = ct.eq("Malignant.cells").values, ct.eq("Fibroblasts").values
        patient = a.obs["patient.id"].astype(str).values
    else:
        f = {"GSE103322": SC / "HNSC_GSE103322" / "HNSC_GSE103322.h5ad",
             "GSE150321": SC / "LSCC_GSE150321" / "LSCC_GSE150321.h5ad"}[name]
        a = ad.read_h5ad(f)
        a = a[:, [g for g in genes if g in a.var_names]].copy()
        X = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
        X = np.expm1(X)                                   # back to linear CP10K
        major = a.obs["Celltype (major-lineage)"].astype(str)
        mal = major.eq("Malignant").values
        caf = major.isin(["Fibroblasts", "Myofibroblasts"]).values
        patient = a.obs["Patient"].astype(str).values if "Patient" in a.obs else np.array(["all"] * a.n_obs)
    return pd.DataFrame(np.asarray(X), columns=list(a.var_names)), mal, caf, patient


def shares(X: pd.DataFrame, mal: np.ndarray, caf: np.ndarray, genes: list[str]) -> pd.DataFrame:
    g = [x for x in genes if x in X.columns]
    M, C = X.loc[mal, g], X.loc[caf, g]
    d = pd.DataFrame(index=g)
    d["share_total"] = M.sum() / (M.sum() + C.sum())
    d["share_per_cell"] = M.mean() / (M.mean() + C.mean())
    d["pct_malignant"] = (M > 0).mean() * 100
    d["pct_fibroblast"] = (C > 0).mean() * 100
    return d[(M.sum() + C.sum()) > 0]


def main() -> None:
    ref = pd.read_csv(OUT / "puram_pemt_gene_partition.tsv", sep="\t").set_index("gene")
    genes = ref.index.tolist()
    per, patient_rows = [], []
    for name in ("GSE103322", "GSE150321", "GSE181919"):
        X, mal, caf, pat = load(name, genes)
        d = shares(X, mal, caf, genes)
        d["dataset"] = name
        d["n_malignant"], d["n_fibroblast"] = int(mal.sum()), int(caf.sum())
        n_pat = len(set(pat[mal | caf]))
        print(f"{name}: {int(mal.sum())} malignant cells, {int(caf.sum())} fibroblasts, "
              f"{n_pat} patients, {len(d)} pEMT genes detected")
        per.append(d.reset_index(names="gene"))
        if name == "GSE181919":
            for p in sorted(set(pat)):
                m, c = mal & (pat == p), caf & (pat == p)
                if m.sum() >= MIN_CELLS and c.sum() >= MIN_CELLS:
                    s = shares(X, m, c, genes)["share_per_cell"]
                    patient_rows.append(s.rename(p))
    long = pd.concat(per, ignore_index=True)
    long["arm_per_cell"] = np.where(long["share_per_cell"] > 0.5, "malignant-expressed", "CAF-expressed")
    long["family"] = np.where(long["gene"].isin(HEMI), "hemidesmosome / laminin-332", "other")
    long.to_csv(OUT / "replication_per_gene.tsv", sep="\t", index=False)

    pp = pd.concat(patient_rows, axis=1)
    pp.to_csv(OUT / "replication_GSE181919_per_patient.tsv", sep="\t")
    pat_summary = pd.DataFrame({"n_patients": pp.notna().sum(axis=1),
                                "median_share_per_cell": pp.median(axis=1),
                                "frac_patients_malignant_side": (pp > 0.5).sum(axis=1) / pp.notna().sum(axis=1)})
    pat_summary.to_csv(OUT / "replication_GSE181919_patient_summary.tsv", sep="\t")
    print(f"\nGSE181919 per-patient analysis: {pp.shape[1]} patients with >= {MIN_CELLS} malignant cells "
          f"and >= {MIN_CELLS} fibroblasts")

    rows = []
    for name, d in long.groupby("dataset"):
        m = d.merge(ref[["arm", "share_malignant"]].reset_index(), on="gene")
        hemi = m[m["family"].str.startswith("hemi")]["share_per_cell"]
        rest = m[~m["family"].str.startswith("hemi")]["share_per_cell"]
        rows.append({
            "dataset": name, "genes": len(m),
            "malignant_by_total_share": int((m["share_total"] > 0.5).sum()),
            "malignant_by_per_cell": int((m["arm_per_cell"] == "malignant-expressed").sum()),
            "stromal_by_per_cell": int((m["arm_per_cell"] == "CAF-expressed").sum()),
            "rho_per_cell_vs_discovery_total_share": spearmanr(m["share_per_cell"], m["share_malignant"])[0],
            "hemidesmosome_median_share_per_cell": float(hemi.median()),
            "other_median_share_per_cell": float(rest.median()),
            "hemidesmosome_vs_other_p_one_sided": mannwhitneyu(hemi, rest, alternative="greater").pvalue})
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "replication_summary.tsv", sep="\t", index=False)

    w = long.pivot(index="gene", columns="dataset", values="share_per_cell").dropna()
    cross = {"genes_in_all": len(w)}
    for a_, b_ in (("GSE103322", "GSE150321"), ("GSE103322", "GSE181919"), ("GSE150321", "GSE181919")):
        cross[f"rho_{a_}_vs_{b_}"] = spearmanr(w[a_], w[b_])[0]
        cross[f"kappa_{a_}_vs_{b_}"] = kappa(w[a_] > 0.5, w[b_] > 0.5)
    cross["malignant_side_in_all_three"] = int((w > 0.5).all(axis=1).sum())
    cross["stromal_side_in_all_three"] = int((w < 0.5).all(axis=1).sum())
    pd.DataFrame([cross]).to_csv(OUT / "replication_cross_dataset.tsv", sep="\t", index=False)
    print("\n" + summ.round(3).T.to_string())
    print("\n" + pd.Series(cross).round(3).to_string())

    show = ["COL17A1", "LAMB3", "LAMC2", "LAMA3", "ITGA6", "ITGB6", "COL1A1", "TAGLN", "COL5A2", "THBS1",
            "MMP2", "VIM"]
    tab = w.reindex(show).join(pat_summary[["median_share_per_cell", "frac_patients_malignant_side"]]
                               .add_prefix("GSE181919_patients_"))
    print("\nlandmark genes, per-cell malignant share (linear):\n" + tab.round(3).to_string())


if __name__ == "__main__":
    main()
