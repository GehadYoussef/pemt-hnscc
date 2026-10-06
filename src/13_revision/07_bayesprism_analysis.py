"""Analyse the BayesPrism deconvolution of TCGA-HNSC and GSE65021 written by 06_bayesprism_deconvolution.R.

The R stage writes per-sample cell-type fractions and the malignant- and fibroblast-specific expression
(Z) for TCGA-HNSC (520 primaries, RNA-seq counts) and GSE65021 (40 patients, Illumina DASL microarray,
linear intensities), with GSE181919 as the single-cell reference, to results/bayesprism/.

Deconvolved scores. BayesPrism's malignant-cell expression Z (expected counts attributed to malignant
cells, per sample and gene) is scaled to counts per million of the malignant total in each sample,
log2(x + 1) transformed, standardised per gene within cohort, and averaged over a gene set (mean per-gene
z-score, the rule used for every signature in the project). Gene sets: the 12-gene malignant core and
the 100 canonical Puram pEMT genes (results/arm_scores/arm_gene_sets.tsv, config/signatures/puram_pemt.txt).
The 14-gene stromal core is scored the same way on the fibroblast-specific Z.

Analyses
  1. Spearman correlations: fibroblast fraction with the bulk stromal core (and the classifier's
     fibroblast probability and ESTIMATE stromal score), malignant fraction with purity proxies (ESTIMATE
     purity and scores, an 18-marker immune score as in 10_cetuximab/07_purity_adjustment.py, the stromal
     core).
  2. Spearman correlations of the deconvolved malignant core and Puram pEMT scores with the bulk
     malignant core, pEMT specificity and the bulk Puram score.
  3. GSE65021: AUC for long PFS (2,000-resample bootstrap interval, the function and seed of
     10_cetuximab/01_cetuximab_cohort.py), two-sided Mann-Whitney p and rank-biserial correlation, for the
     deconvolved malignant pEMT, the deconvolved malignant core and the fibroblast fraction (malignant
     fraction and bulk comparators alongside), in all 40 patients and in the 31 primary-tumour specimens.
  4. TCGA: deconvolved scores by the 2015 expression subtype (Basal, Classical, Atypical, Mesenchymal),
     Kruskal-Wallis and Basal against the rest, as in 04_tcga_projection/10_independent_validation.py.

Inputs:  results/bayesprism/ (from 06_bayesprism_deconvolution.R), results/arm_scores/arm_gene_sets.tsv
         results/arm_scores/{TCGA-HNSC,GSE65021}_arm_scores.tsv, results/cetuximab_cohort/GSE65021_scores.tsv
         results/tcga_projection/emt_score_panel_scores.tsv, tcga_pemt_specificity_scores.tsv
         data/processed/tcga_hnsc/tcga_star_tpm_log2_for_projection.tsv, data/raw/external_cohorts/GSE65021 files
         data/references/TCGA_HNSC_4class_expression_subtype.csv, config/signatures/puram_pemt.txt
Outputs: results/bayesprism/<cohort>_deconvolved_scores.tsv, correlations.tsv, gse65021_pfs_auc.tsv,
         tcga_subtype.tsv, fraction_summary.tsv, deconvolved_gene_coverage.tsv, analysis_log.txt
Usage:   python src/13_revision/07_bayesprism_analysis.py
"""

from __future__ import annotations

import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
RES = ROOT / "results"
TP = RES / "tcga_projection"
REF = ROOT / "data" / "references"
BP = RES / "bayesprism"

ALIAS = {"CXCR7": "ACKR3", "LEPREL1": "P3H2", "PRKCDBP": "CAVIN3", "DFNA5": "GSDME"}
IMMUNE = ["PTPRC", "CD2", "CD3D", "CD3E", "CD247", "CD8A", "CD19", "MS4A1", "CD79A", "CD68", "CD14",
          "CD163", "LYZ", "FCGR3A", "ITGAM", "CSF1R", "CD74", "HLA-DRA"]
SUBTYPE_ORDER = ["Basal", "Classical", "Atypical", "Mesenchymal"]


def fractions_file(cohort: str) -> Path:
    """Final (updated-reference) fractions when present, otherwise the initial Gibbs fractions.

    The second Gibbs step refines the fractions only. Z, the cell-type-specific expression used for
    every deconvolved score, comes from the first step, so a run stopped during the second step still
    gives every deconvolved score.
    """
    f = BP / f"{cohort}_fractions.tsv"
    return f if f.exists() else BP / f"{cohort}_fractions_initial.tsv"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, [sys.argv[0]]
    spec.loader.exec_module(mod)
    sys.argv = argv
    return mod


def gene_sets() -> dict[str, list[str]]:
    s = pd.read_csv(RES / "arm_scores" / "arm_gene_sets.tsv", sep="\t").set_index("score")["genes"]
    split = lambda k: [g.strip() for g in s[k].split(",")]  # noqa: E731
    puram = [l.strip() for l in (ROOT / "config" / "signatures" / "puram_pemt.txt").read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    return {"malignant_core": split("malignant_arm_core"),
            "puram_pemt": [ALIAS.get(g, g) for g in puram],
            "stromal_core": split("stromal_arm_core")}


def zmean(expr: pd.DataFrame, genes: list[str]) -> tuple[pd.Series, int]:
    """Mean per-gene z-score, with expr as genes by samples (the rule of 04_tcga_projection/07)."""
    g = [x for x in genes if x in expr.index]
    X = expr.loc[g]
    Z = X.sub(X.mean(axis=1), axis=0).div(X.std(axis=1).replace(0, 1), axis=0)
    return Z.mean(axis=0), len(g)


def deconvolved_expression(cohort: str, cell: str) -> pd.DataFrame:
    """BayesPrism Z (samples by genes) as log2 CPM of the cell type's own total, genes by samples."""
    z = pd.read_csv(BP / f"{cohort}_Z_{cell}.tsv.gz", sep="\t", index_col=0)
    cpm = z.div(z.sum(axis=1), axis=0) * 1e6
    return np.log2(cpm + 1).T


def bulk_tables(cohort: str) -> pd.DataFrame:
    arms = pd.read_csv(RES / "arm_scores" / f"{cohort}_arm_scores.tsv", sep="\t", index_col=0)
    arms = arms[["malignant_arm_core", "stromal_arm_core"]].add_prefix("bulk_")
    if cohort == "TCGA-HNSC":
        sc = pd.read_csv(TP / "emt_score_panel_scores.tsv", sep="\t", index_col=0)[["pEMT_specificity", "puram_pemt"]]
        pr = pd.read_csv(TP / "tcga_pemt_specificity_scores.tsv", sep="\t", index_col=0)[["P_fibroblast_stromal_like"]]
        expr = pd.read_csv(ROOT / "data" / "processed" / "tcga_hnsc" / "tcga_star_tpm_log2_for_projection.tsv",
                           sep="\t", index_col=0)
        d = sc.join(pr, how="left")
    else:
        sc = pd.read_csv(RES / "cetuximab_cohort" / "GSE65021_scores.tsv", sep="\t", index_col=0)
        d = sc[["long_pfs", "specimen_recurrence", "pEMT_specificity", "puram_pemt", "P_fibroblast_stromal_like"]]
        cx = _load(SRC / "10_cetuximab" / "01_cetuximab_cohort.py", "cx")
        x, _ = cx.read_series_matrix(cx.DATA / "GSE65021_series_matrix.txt.gz")
        expr = cx.collapse(x, cx.read_annotation(cx.DATA / "GPL10558.annot.gz"))
    d = d.rename(columns={"pEMT_specificity": "bulk_pEMT_specificity", "puram_pemt": "bulk_puram_pemt",
                          "P_fibroblast_stromal_like": "bulk_P_fibroblast"})
    d["bulk_immune_score"], _ = zmean(expr[d.index.intersection(expr.columns)], IMMUNE)
    est = pd.read_csv(BP / f"{cohort}_estimate.tsv", sep="\t", index_col=0)
    return d.join(arms, how="left").join(est, how="left")


def cohort_table(cohort: str, sets: dict[str, list[str]], cover: list) -> pd.DataFrame:
    frac = pd.read_csv(fractions_file(cohort), sep="\t", index_col=0).add_prefix("frac_")
    zm = deconvolved_expression(cohort, "malignant")
    zf = deconvolved_expression(cohort, "fibroblast")
    d = frac.copy()
    for k, src, lab in (("malignant_core", zm, "dc_mal_core"), ("puram_pemt", zm, "dc_mal_puram_pemt"),
                        ("stromal_core", zf, "dc_fib_stromal_core")):
        d[lab], n = zmean(src, sets[k])
        cover.append({"cohort": cohort, "score": lab, "genes_present": n, "genes_in_set": len(sets[k]),
                      "missing": ", ".join(g for g in sets[k] if g not in src.index)})
    return d.join(bulk_tables(cohort), how="left")


def corr_rows(d: pd.DataFrame, cohort: str, pairs: list[tuple[str, str]]) -> list[dict]:
    rows = []
    for x, y in pairs:
        if x not in d or y not in d:
            continue
        s = d[[x, y]].dropna()
        rho, p = spearmanr(s[x], s[y])
        rows.append({"cohort": cohort, "x": x, "y": y, "n": len(s), "spearman_rho": rho, "p_value": p})
    return rows


def main() -> None:
    cohorts = [c for c in ("TCGA-HNSC", "GSE65021") if fractions_file(c).exists()]
    if not cohorts:
        print(f"No BayesPrism fractions in {BP}. Run src/13_revision/06_bayesprism_deconvolution.R first.")
        return
    sets = gene_sets()
    cover, tables, corr = [], {}, []
    for cohort in cohorts:
        d = cohort_table(cohort, sets, cover)
        tables[cohort] = d
        d.to_csv(BP / f"{cohort}_deconvolved_scores.tsv", sep="\t")
        pairs = [("frac_Fibroblast", "bulk_stromal_arm_core"), ("frac_Fibroblast", "bulk_P_fibroblast"),
                 ("frac_Fibroblast", "stromal_estimate"),
                 ("frac_Malignant", "purity_estimate_affy"), ("frac_Malignant", "estimate_score"),
                 ("frac_Malignant", "stromal_estimate"), ("frac_Malignant", "immune_estimate"),
                 ("frac_Malignant", "bulk_immune_score"), ("frac_Malignant", "bulk_stromal_arm_core"),
                 ("frac_Malignant", "bulk_malignant_arm_core"), ("frac_Malignant", "bulk_pEMT_specificity"),
                 ("dc_fib_stromal_core", "bulk_stromal_arm_core")]
        for dc in ("dc_mal_core", "dc_mal_puram_pemt"):
            pairs += [(dc, y) for y in ("bulk_malignant_arm_core", "bulk_pEMT_specificity", "bulk_puram_pemt",
                                        "bulk_stromal_arm_core", "frac_Malignant", "frac_Fibroblast")]
        pairs.append(("dc_mal_core", "dc_mal_puram_pemt"))
        corr += corr_rows(d, cohort, pairs)
    corr = pd.DataFrame(corr)
    corr.to_csv(BP / "correlations.tsv", sep="\t", index=False)
    pd.DataFrame(cover).to_csv(BP / "deconvolved_gene_coverage.tsv", sep="\t", index=False)

    summ = []
    for cohort, d in tables.items():
        if not (BP / f"{cohort}_fractions_cv.tsv").exists():
            continue
        cv = pd.read_csv(BP / f"{cohort}_fractions_cv.tsv", sep="\t", index_col=0)
        for c in [c for c in d if c.startswith("frac_")]:
            t = c[5:]
            summ.append({"cohort": cohort, "cell_type": t, "mean": d[c].mean(), "median": d[c].median(),
                         "q25": d[c].quantile(.25), "q75": d[c].quantile(.75),
                         "median_posterior_cv": cv[t].median() if t in cv else np.nan})
    summ = pd.DataFrame(summ)
    summ.to_csv(BP / "fraction_summary.tsv", sep="\t", index=False)

    if "GSE65021" not in tables:
        return
    cx = _load(SRC / "10_cetuximab" / "01_cetuximab_cohort.py", "cx_auc")
    g = tables["GSE65021"]
    auc_rows = []
    scores = ["dc_mal_puram_pemt", "dc_mal_core", "frac_Fibroblast", "frac_Malignant", "dc_fib_stromal_core",
              "bulk_malignant_arm_core", "bulk_pEMT_specificity", "bulk_puram_pemt", "bulk_stromal_arm_core"]
    for subset, sub in (("all 40", g), ("primary specimens", g[g["specimen_recurrence"] == 0])):
        y = sub["long_pfs"].to_numpy(int)
        for sc in scores:
            a, b = sub.loc[sub.long_pfs == 1, sc], sub.loc[sub.long_pfs == 0, sc]
            auc, lo, hi = cx.bootstrap_auc(y, sub[sc].to_numpy(float))
            mw = mannwhitneyu(a, b, alternative="two-sided")
            auc_rows.append({"subset": subset, "score": sc, "n_long": len(a), "n_short": len(b), "AUC": auc,
                             "AUC_low": lo, "AUC_up": hi, "rank_biserial": 2 * mw.statistic / (len(a) * len(b)) - 1,
                             "p_two_sided": mw.pvalue,
                             "pre_specified": sc in ("dc_mal_puram_pemt", "dc_mal_core", "frac_Fibroblast")})
    auc = pd.DataFrame(auc_rows)
    auc.to_csv(BP / "gse65021_pfs_auc.tsv", sep="\t", index=False)

    if "TCGA-HNSC" not in tables:
        print(pd.DataFrame(cover).to_string(), summ.round(3).to_string(), corr.round(3).to_string(),
              auc.round(3).to_string(), sep="\n\n")
        return
    t = tables["TCGA-HNSC"]
    st = pd.read_csv(REF / "TCGA_HNSC_4class_expression_subtype.csv")
    key = pd.Series({c: "-".join(c.split("-")[:3]) + "-01" for c in t.index})
    sub = key.map(st.set_index("sample_barcode")["RNA_subtype"])
    sub = sub[sub.isin(SUBTYPE_ORDER)]
    srows = []
    for s in ["dc_mal_puram_pemt", "dc_mal_core", "dc_fib_stromal_core", "frac_Malignant", "frac_Fibroblast",
              "bulk_puram_pemt", "bulk_malignant_arm_core", "bulk_pEMT_specificity"]:
        groups = [t.loc[sub.index[sub == k], s].dropna().to_numpy() for k in SUBTYPE_ORDER]
        H, pk = kruskal(*groups)
        basal = groups[0]
        rest = np.concatenate(groups[1:])
        mw = mannwhitneyu(basal, rest, alternative="two-sided")
        meds = [float(np.median(x)) for x in groups]
        srows.append({"score": s, "highest_subtype": SUBTYPE_ORDER[int(np.argmax(meds))],
                      "kruskal_H": H, "kruskal_p": pk, "basal_vs_rest_p": mw.pvalue,
                      "basal_vs_rest_rank_biserial": 2 * mw.statistic / (len(basal) * len(rest)) - 1,
                      **{f"n_{k}": len(x) for k, x in zip(SUBTYPE_ORDER, groups)},
                      **{f"median_{k}": m for k, m in zip(SUBTYPE_ORDER, meds)}})
    subt = pd.DataFrame(srows)
    subt.to_csv(BP / "tcga_subtype.tsv", sep="\t", index=False)

    buf = io.StringIO()
    with redirect_stdout(buf):
        pd.set_option("display.width", 200)
        print("Deconvolved gene coverage\n", pd.DataFrame(cover).to_string(index=False))
        print("\nCell-type fractions\n", summ.round(3).to_string(index=False))
        print("\nSpearman correlations\n", corr.round(4).to_string(index=False))
        print("\nGSE65021 long PFS\n", auc.round(3).to_string(index=False))
        print(f"\nTCGA subtype (matched {len(sub)} of {len(t)})\n", subt.round(4).to_string(index=False))
    print(buf.getvalue())
    (BP / "analysis_log.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
