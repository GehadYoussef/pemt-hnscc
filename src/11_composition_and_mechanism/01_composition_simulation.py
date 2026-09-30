"""Test whether each score follows the malignant cell state or the cellular composition, by mixture simulation.

Synthetic tumours are built from single cells of GSE181919 (Choi et al. 2023), a dataset the
classifier was not trained on. Tumour and lymph-node tissue cells are used with the author cell-type
annotation. Malignant cells are split by their own Puram pEMT score (per-cell mean z-score of log1p
counts per 10,000 over the pEMT genes, z-scored across malignant cells) into a pEMT-high pool (top
quartile) and a pEMT-low pool (bottom quartile).

Each synthetic tumour has 300 cells, drawn with replacement, in a full factorial design over
  q  share of the malignant cells drawn from the pEMT-high pool   0, 0.25, 0.5, 0.75, 1
  f  fibroblast fraction of the tumour                              0.05, 0.15, 0.30, 0.45, 0.60
  u  immune fraction (T, B/plasma, macrophage, dendritic cells)     0.05, 0.20, 0.35
The malignant fraction is 1 - f - u. There are 6 replicates per design cell, 450 tumours in all.
Each tumour is the mean of its cells' linear counts per 10,000 (the bulk-like sum, renormalised),
then log1p. The 450 tumours form one cohort and are scored as a bulk cohort is scored. pEMT
specificity comes from the trained classifier with within-cohort z-scoring (classifier genes absent
from the matrix are set to zero), and each gene-set score is a mean per-gene z-score. The scores are
pEMT specificity, the malignant and stromal arm cores, the total-share malignant arm, the Puram pEMT
signature, Hallmark EMT and Puram epithelial differentiation.

Each standardised score is regressed by OLS on standardised q, f and u. The partial R^2 of a factor
is (SSR of the model without it - SSR of the full model) / SSR of the model without it. The first run
streams the 128 MB count matrix, keeps only the genes needed for scoring (library sizes use all
genes) and caches the result.

Inputs:  data/raw/GSE181919/GSE181919_UMI_counts.txt.gz
         data/raw/GSE181919/GSE181919_Barcode_metadata.txt.gz
         results/multinomial_classifier/tcga_depmap_ready_classifier.pkl
         results/multinomial_classifier/classifier_genes.txt
         results/arm_scores/arm_gene_sets.tsv
         data/references/emt_scores/hallmark_emt_genes.txt
         data/references/emt_scores/gs76_genes.tsv
         config/signatures.yaml and config/signatures/*.txt (through pemt.load_signatures)
Outputs: data/processed/GSE181919/GSE181919_composition_subset.h5ad (cache)
         results/composition_simulation/composition_regression.tsv
         results/composition_simulation/simulated_tumours.tsv
Usage:   python src/11_composition_and_mechanism/01_composition_simulation.py
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pemt import load_config, load_signatures  # noqa: E402

cfg = load_config()
CLF_DIR = ROOT / cfg["paths"]["results_dir"] / "multinomial_classifier"
REF = ROOT / cfg["paths"]["references_dir"] / "emt_scores"
D = ROOT / "data" / "raw" / "GSE181919"
CACHE = ROOT / "data" / "processed" / "GSE181919" / "GSE181919_composition_subset.h5ad"
OUT = ROOT / "results" / "composition_simulation"
OUT.mkdir(parents=True, exist_ok=True)
IMMUNE = {"T.cells", "B_Plasma.cells", "Macrophages", "Dendritic.cells"}
SEED = 20260925


def classifier():
    clf = joblib.load(CLF_DIR / "tcga_depmap_ready_classifier.pkl").named_steps["clf"]
    genes = pd.read_csv(CLF_DIR / "classifier_genes.txt", header=None)[0].tolist()
    return clf, genes


def wanted_genes() -> set[str]:
    clf, genes = classifier()
    nz = np.abs(clf.coef_).sum(axis=0) > 0
    w = {g for g, k in zip(genes, nz) if k}
    for s in load_signatures().values():
        w |= set(s)
    w |= {l.strip() for l in (REF / "hallmark_emt_genes.txt").read_text().splitlines() if l.strip()}
    w |= set(pd.read_csv(REF / "gs76_genes.tsv", sep="\t").iloc[:, 0].astype(str))
    arms = pd.read_csv(ROOT / "results/arm_scores/arm_gene_sets.tsv", sep="\t")["genes"].str.split(", ")
    for a in arms:
        w |= set(a)
    return w


def load_cells():
    import anndata as ad
    if CACHE.exists():
        return ad.read_h5ad(CACHE)
    wanted = wanted_genes()
    with gzip.open(D / "GSE181919_UMI_counts.txt.gz", "rt") as fh:
        cells = [h.replace(".", "-") for h in fh.readline().rstrip("\n").split("\t")]
        lib = np.zeros(len(cells))
        keep, rows = [], []
        for i, line in enumerate(fh):
            gene, rest = line.split("\t", 1)
            v = np.fromstring(rest, sep="\t")
            lib += v
            if gene in wanted:
                keep.append(gene)
                rows.append(v.astype(np.float32))
            if i % 5000 == 0:
                print(f"  {i} genes streamed", flush=True)
    meta = pd.read_csv(D / "GSE181919_Barcode_metadata.txt.gz", sep="\t", index_col=0).loc[cells]
    X = (np.vstack(rows).T / lib[:, None] * 1e4).astype(np.float32)
    a = ad.AnnData(X=X, obs=meta, var=pd.DataFrame(index=keep))
    a = a[a.obs["tissue.type"].isin(["CA", "LN"])].copy()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    a.write_h5ad(CACHE)
    return a


def zmean(logexpr: pd.DataFrame, genes) -> pd.Series:
    g = [x for x in genes if x in logexpr.columns]
    z = (logexpr[g] - logexpr[g].mean()) / logexpr[g].std().replace(0, 1.0)
    return z.mean(axis=1)


def main() -> None:
    a = load_cells()
    ct = a.obs["cell.type"].astype(str)
    lin = pd.DataFrame(a.X, index=a.obs_names, columns=a.var_names)
    sigs = load_signatures()

    mal = lin[ct == "Malignant.cells"]
    s = zmean(np.log1p(mal), sigs["puram_pemt"])
    hi = mal.index[s >= s.quantile(0.75)]
    lo = mal.index[s <= s.quantile(0.25)]
    fib = lin.index[ct == "Fibroblasts"]
    imm = lin.index[ct.isin(IMMUNE)]
    print(f"pools: pEMT-high {len(hi)}, pEMT-low {len(lo)}, fibroblasts {len(fib)}, immune {len(imm)}")

    rng = np.random.default_rng(SEED)
    N = 300
    design, profiles = [], []
    for q in (0, 0.25, 0.5, 0.75, 1):
        for f in (0.05, 0.15, 0.30, 0.45, 0.60):
            for u in (0.05, 0.20, 0.35):
                n_f, n_u = round(f * N), round(u * N)
                n_m = N - n_f - n_u
                n_hi = round(q * n_m)
                for _ in range(6):
                    pick = np.concatenate([rng.choice(hi, n_hi), rng.choice(lo, n_m - n_hi),
                                           rng.choice(fib, n_f), rng.choice(imm, n_u)])
                    profiles.append(lin.loc[pick].values.mean(axis=0))
                    design.append({"q_pEMT_high_share": q, "f_fibroblast": f, "u_immune": u})
    bulk = np.log1p(pd.DataFrame(profiles, columns=lin.columns))
    design = pd.DataFrame(design)

    clf, genes = classifier()
    Xf = pd.DataFrame(0.0, index=bulk.index, columns=genes)
    shared = [g for g in genes if g in bulk.columns]
    Xf[shared] = bulk[shared]
    Z = np.nan_to_num(((Xf - Xf.mean()) / Xf.std().replace(0, 1.0)).values)
    p = pd.DataFrame(clf.predict_proba(Z), columns=list(clf.classes_))
    arms = pd.read_csv(ROOT / "results/arm_scores/arm_gene_sets.tsv", sep="\t").set_index("score")["genes"].str.split(", ")
    hm = [l.strip() for l in (REF / "hallmark_emt_genes.txt").read_text().splitlines() if l.strip()]
    scores = pd.DataFrame({
        "pEMT specificity": p["pEMT_high"] - p[["epithelial_like", "fibroblast_stromal_like"]].max(axis=1),
        "Malignant core": zmean(bulk, arms["malignant_arm_core"]),
        "Stromal core": zmean(bulk, arms["stromal_arm_core"]),
        "Malignant arm, total share": zmean(bulk, arms["malignant_arm"]),
        "Puram pEMT (canonical)": zmean(bulk, sigs["puram_pemt"]),
        "Hallmark EMT": zmean(bulk, hm),
        "Puram epithelial differentiation": zmean(bulk, sigs["puram_epi_dif_1"]),
    })
    print(f"classifier genes present {len(shared)}/{len(genes)}")

    import statsmodels.api as sm
    Xd = (design - design.mean()) / design.std()
    rows = []
    for name, v in scores.items():
        yv = (v - v.mean()) / v.std()
        full = sm.OLS(yv.values, sm.add_constant(Xd.values)).fit()
        r = {"score": name, "R2": full.rsquared}
        for j, col in enumerate(Xd.columns):
            red = sm.OLS(yv.values, sm.add_constant(Xd.drop(columns=col).values)).fit()
            r[f"beta_{col}"] = full.params[j + 1]
            r[f"partialR2_{col}"] = (red.ssr - full.ssr) / red.ssr
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "composition_regression.tsv", sep="\t", index=False)
    pd.concat([design, scores], axis=1).to_csv(OUT / "simulated_tumours.tsv", sep="\t", index=False)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
