"""Test the tumour-intrinsic pEMT axis against cetuximab benefit in a treated cohort (GSE65021).

GSE65021 (Bossi et al., Clin Cancer Res 2016, 22:3961-70, PMID 26920888) has 40 patients with
recurrent or metastatic HNSCC given first-line platinum plus cetuximab, selected as two extreme
groups: 14 with progression-free survival (PFS) beyond 12 months (LONG) and 26 with PFS under 5.6
months (SHORT). The groups were balanced by design on performance status, weight loss, prior
radiotherapy, grade, primary site and residual disease at the primary site. The platform is the
Illumina Whole-Genome DASL HumanHT-12 v4 BeadChip (GPL14951), 29,377 ILMN probes on a linear
scale. GPL14951 shares the ILMN_ probe namespace with GPL10558. 27,548 of the 29,377 probes (93.8%)
map through the GPL10558 annotation, giving 19,436 unique gene symbols, so the GPL14951 SOFT file
is not needed. Data are log2-transformed and collapsed to the highest-mean probe per gene, as in
src/04_tcga_projection/09_external_cohorts.py.

The classifier is applied with within-cohort z-scoring. It was trained on single-cell pseudobulks
from GSE103322 and GSE150321 and is independent of this cohort. The comparison panel adds 76GS, KS,
Hallmark EMT, the Puram pEMT and epithelial scores, the published MLR-EMT on the microarray scale
and five published HNSCC signatures. The direction, higher in the LONG group, was specified before
the analysis from the link between pEMT specificity and the TCGA Basal subtype, the cetuximab
response of basal patient-derived xenografts (Klinghammer et al.) and the Zhou 2025 nine-gene
predictor of LONG PFS. One- and two-sided p-values are both reported. Zhou et al. derived their
predictor on this cohort, and six of the nine genes (COL17A1, ITGA3, ITGB4, LAMA3, LAMC2, MT2A) are
among the classifier's top 100 positive pEMT-high coefficients. The zhou2025_predictive score is
therefore reported as a positive control and excluded from the FDR correction.

GEO holds only the LONG/SHORT groups, with no survival times. Each score is tested with Mann-Whitney
U (rank-biserial correlation as effect size), ROC AUC with a 2,000-sample bootstrap interval, and
logistic regression per SD, unadjusted and adjusted for the covariates GEO records (age, stage,
grade, prior radiotherapy, primary site, and whether the specimen came from the primary or the
recurrence). Benjamini-Hochberg FDR is applied across the independent scores. With 14 against 26
patients the cohort is powered only for a large effect.

Figure_6 shows (a) pEMT specificity by PFS group, (b) its ROC curve with the Zhou nine-gene set as
a positive control, (c) the AUC of the direct malignant-arm and stromal-arm scores and (d) the
rank-biserial correlation of every independent score. Panel c reads the tables written by
05_direct_arm_scores.py, so the run order is this script, then 02 to 05, then this script again
with --figure-only. Supplementary_Figure_8 shows the ROC curves of the exploratory combined marker
(z-scored pEMT specificity plus z-scored Puram epithelial score), with the bootstrap interval from
03_cetuximab_headtohead.py when that table exists.

Inputs:  data/raw/external_cohorts/GSE65021_series_matrix.txt.gz
         data/raw/external_cohorts/GPL10558.annot.gz
         results/multinomial_classifier/tcga_depmap_ready_classifier.pkl
         results/multinomial_classifier/classifier_genes.txt
         data/references/emt_scores/ (Hallmark EMT, 76GS, KS gene lists)
         data/references/ (published signature gene lists, read by pemt.published)
         config/signatures/
         results/arm_scores/gse65021_arm_association.tsv (Figure_6c)
         results/arm_scores/arm_gene_sets.tsv (Figure_6c)
         results/cetuximab_cohort/GSE65021_combined_score.tsv (optional, Supplementary_Figure_8)
Outputs: results/cetuximab_cohort/GSE65021_scores.tsv
         results/cetuximab_cohort/GSE65021_pfs_association.tsv
         results/figures/Figure_6
         results/figures/Supplementary_Figure_8
Usage:   python src/10_cetuximab/01_cetuximab_cohort.py [--figure-only]
"""

from __future__ import annotations

import argparse
import gzip
import io
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

sys.path.insert(0, str(ROOT / "src"))
from pemt import load_config, load_signatures  # noqa: E402
from pemt.published import load_published, LABELS as PUB_LABELS  # noqa: E402
from pemt.figstyle import apply_style, save_figure, PALETTE, mm  # noqa: E402

cfg = load_config()
CLF_DIR = ROOT / cfg["paths"]["results_dir"] / "multinomial_classifier"
REF_DIR = ROOT / cfg["paths"]["references_dir"]
EMT_REF = REF_DIR / "emt_scores"
DATA = ROOT / "data" / "raw" / "external_cohorts"
OUT = ROOT / "results" / "cetuximab_cohort"
FIG = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

apply_style()
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(ROOT / "src" / "04_tcga_projection"))
from importlib import import_module  # noqa: E402

_panel = import_module("07_emt_score_panel")  # zmean, score_76gs, score_ks, score_mlr

SITE_LABELS = {"0": "oral cavity", "1": "oropharynx", "2": "hypopharynx", "3": "larynx"}

SCORE_LABELS = {
    "pEMT_specificity": "pEMT specificity",
    "P_pEMT_high": "P(pEMT-high)",
    "GS76": "76GS",
    "KS": "KS",
    "hallmark_EMT": "Hallmark EMT",
    "puram_pemt": "Puram pEMT",
    "puram_epi_dif_1": "Puram epithelial",
    "MLR_mu": "MLR-EMT",
}
# Reported separately from the panel above: derived on these same 40 patients (see module docstring).
NON_INDEPENDENT = {"zhou2025_predictive"}


def read_series_matrix(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expression (probes x samples) and per-sample characteristics from a GEO series matrix."""
    chars: list[list[str]] = []
    ids: list[str] | None = None
    rows: list[list[str]] = []
    with gzip.open(path, "rt", errors="replace") as fh:
        in_table = False
        for line in fh:
            if line.startswith("!Sample_geo_accession"):
                ids = [x.strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                chars.append([x.strip('"') for x in line.rstrip("\n").split("\t")[1:]])
            elif line.startswith("!series_matrix_table_begin"):
                in_table = True
            elif line.startswith("!series_matrix_table_end"):
                break
            elif in_table:
                rows.append(line.rstrip("\n").split("\t"))
    header = [x.strip('"') for x in rows[0]]
    expr = pd.DataFrame(
        [r[1:] for r in rows[1:]],
        index=[r[0].strip('"') for r in rows[1:]],
        columns=header[1:],
    ).apply(pd.to_numeric, errors="coerce")
    meta = pd.DataFrame(index=ids)
    for row in chars:
        for sid, v in zip(ids, row):
            if ":" in v:
                k, val = v.split(":", 1)
                meta.loc[sid, k.strip().lower()] = val.strip()
    return expr, meta


def read_annotation(path: Path) -> pd.Series:
    """Map ILMN probes to gene symbols from a GEO .annot.gz, dropping multi-mapping probes."""
    with gzip.open(path, "rt", errors="replace") as fh:
        lines = fh.readlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("ID\t"))
    end = next((i for i, l in enumerate(lines) if l.startswith("!platform_table_end")), len(lines))
    tab = pd.read_csv(io.StringIO("".join(lines[start:end])), sep="\t", dtype=str)
    sym = tab.set_index("ID")["Gene symbol"].dropna()
    return sym[~sym.str.contains("///")]


def collapse(expr: pd.DataFrame, sym: pd.Series) -> pd.DataFrame:
    """log2(x + 1) if the data are linear, then the highest-mean probe per gene.

    Same rule as src/04_tcga_projection/09_external_cohorts.py.
    """
    if expr.max().max() > 100:
        expr = np.log2(expr.clip(lower=0) + 1)
    e = expr.loc[expr.index.intersection(sym.index)].copy()
    e["gene"] = sym.loc[e.index].values
    e["_mean"] = e.drop(columns="gene").mean(axis=1)
    e = e.sort_values("_mean", ascending=False).drop_duplicates("gene").set_index("gene").drop(columns="_mean")
    return e


def project(expr_genes_x_samples: pd.DataFrame) -> tuple[pd.DataFrame, int, int, int]:
    """Apply the classifier with within-cohort z-scoring, as for every other bulk cohort.

    Classifier genes missing from the cohort are set to zero. Returns the class probabilities with
    pEMT specificity, the number of classifier genes present, the number of non-zero-coefficient
    genes present and the total number of non-zero-coefficient genes.
    """
    model = joblib.load(CLF_DIR / "tcga_depmap_ready_classifier.pkl")
    genes = pd.read_csv(CLF_DIR / "classifier_genes.txt", header=None)[0].tolist()
    clf = model.named_steps["clf"]
    X = expr_genes_x_samples.T
    shared = [g for g in genes if g in X.columns]
    X_full = pd.DataFrame(0.0, index=X.index, columns=genes)
    X_full[shared] = X[shared]
    Z = ((X_full - X_full.mean(axis=0)) / X_full.std(axis=0).replace(0, 1.0)).fillna(0.0)
    probs = clf.predict_proba(np.nan_to_num(Z.values))
    out = pd.DataFrame(probs, index=X.index, columns=[f"P_{c}" for c in clf.classes_])
    out["pEMT_specificity"] = out["P_pEMT_high"] - out[["P_epithelial_like", "P_fibroblast_stromal_like"]].max(axis=1)
    nonzero = np.abs(clf.coef_).sum(axis=0) > 0
    n_nz_shared = int(sum(1 for g, nz in zip(genes, nonzero) if nz and g in X.columns))
    return out, len(shared), n_nz_shared, int(nonzero.sum())


def clinical(meta: pd.DataFrame) -> pd.DataFrame:
    """Code the GEO characteristics as analysis covariates, following !Series_overall_design."""
    d = pd.DataFrame(index=meta.index)
    d["long_pfs"] = (meta["pfs"].str.upper() == "LONG").astype(int)
    d["age"] = pd.to_numeric(meta["age"], errors="coerce")
    d["female"] = pd.to_numeric(meta["gender"], errors="coerce")
    d["stage_num"] = pd.to_numeric(meta["stage"], errors="coerce")
    d["grade"] = pd.to_numeric(meta["grading"], errors="coerce")
    d["prior_rt"] = pd.to_numeric(meta["rt"], errors="coerce")
    site_code = meta["site of primary"].astype(str).str.strip()
    d["site"] = site_code.map(SITE_LABELS).fillna("unknown")
    d["site_oral_cavity"] = (site_code == "0").astype(int)
    d["site_oropharynx"] = (site_code == "1").astype(int)
    spec = meta["primary vs rec/met"].astype(str).str.lower()
    d["specimen_recurrence"] = spec.str.contains("rec|met").astype(int)
    return d


def bootstrap_auc(y: np.ndarray, x: np.ndarray, n: int = 2000, seed: int = 20260920) -> tuple[float, float, float]:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    obs = roc_auc_score(y, x)
    draws = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        draws.append(roc_auc_score(y[idx], x[idx]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(obs), float(lo), float(hi)


def logistic_per_sd(df: pd.DataFrame, score: str, covariates: list[str]) -> tuple[float, float, float, float, int]:
    """Odds ratio of LONG PFS per SD of the score, by ordinary logistic regression.

    No Firth correction is applied.
    """
    import statsmodels.api as sm

    cols = ["long_pfs", score] + covariates
    sub = df.dropna(subset=cols).copy()
    sub[score] = (sub[score] - sub[score].mean()) / sub[score].std()
    covariates = [c for c in covariates if sub[c].nunique() > 1]
    X = sm.add_constant(sub[[score] + covariates].astype(float), has_constant="add")
    try:
        res = sm.Logit(sub["long_pfs"].astype(float), X).fit(disp=0, maxiter=200)
        or_ = float(np.exp(res.params[score]))
        lo, hi = np.exp(res.conf_int().loc[score])
        return or_, float(lo), float(hi), float(res.pvalues[score]), len(sub)
    except Exception as exc:  # perfect separation or non-convergence in n = 40
        print(f"    logistic failed for {score} ({exc.__class__.__name__}), reporting NaN")
        return np.nan, np.nan, np.nan, np.nan, len(sub)


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(pv)
    adj = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    q[ok] = out
    return q


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure-only", action="store_true",
                    help="redraw Figure 6 from the saved tables without recomputing")
    args = ap.parse_args()
    if args.figure_only:
        df = pd.read_csv(OUT / "GSE65021_scores.tsv", sep="	", index_col=0)
        res = pd.read_csv(OUT / "GSE65021_pfs_association.tsv", sep="	")
        make_figure(df, res, int(df["long_pfs"].sum()), int((1 - df["long_pfs"]).sum()))
        print(f"redrew {FIG}/Figure_6.svg/.png")
        return

    sigs = load_signatures()
    hm = [l.strip() for l in (EMT_REF / "hallmark_emt_genes.txt").read_text().splitlines() if l.strip()]

    expr, meta = read_series_matrix(DATA / "GSE65021_series_matrix.txt.gz")
    sym = read_annotation(DATA / "GPL10558.annot.gz")
    e = collapse(expr, sym)
    print(f"GSE65021: {expr.shape[0]} probes x {expr.shape[1]} samples, {e.shape[0]} genes "
          f"({100 * len(expr.index.intersection(sym.index)) / expr.shape[0]:.1f}% of probes annotated via GPL10558)")

    probs, n_shared, n_nz_shared, n_nz = project(e)
    scores = probs.copy()
    scores["GS76"], _ = _panel.score_76gs(e)
    scores["KS"], _, _ = _panel.score_ks(e)
    scores["hallmark_EMT"], _, _ = _panel.zmean(e, hm)
    scores["puram_pemt"], _, _ = _panel.zmean(e, sigs["puram_pemt"])
    scores["puram_epi_dif_1"], _, _ = _panel.zmean(e, sigs["puram_epi_dif_1"])
    # Published MLR-EMT (George et al. 2017, reference code of its authors) on the microarray
    # scale, stored as MLR_mu.
    scores["MLR_mu"] = _panel.score_mlr(e, rnaseq=False)
    published = load_published(REF_DIR)
    for key, genes in published.items():
        scores[key], n_found, n_tot = _panel.zmean(e, genes)
        print(f"  {key}: {n_found}/{n_tot} genes present")

    df = clinical(meta).join(scores)
    df.to_csv(OUT / "GSE65021_scores.tsv", sep="\t")
    n_long, n_short = int(df["long_pfs"].sum()), int((1 - df["long_pfs"]).sum())
    print(f"\n{len(df)} patients: {n_long} long PFS, {n_short} short PFS, "
          f"classifier genes present {n_shared}, non-zero-coefficient genes {n_nz_shared}/{n_nz}")

    covs = ["age", "stage_num", "grade", "prior_rt", "site_oral_cavity", "site_oropharynx", "specimen_recurrence"]
    tested = list(SCORE_LABELS) + list(published)
    rows = []
    for sc in tested:
        a = df.loc[df["long_pfs"] == 1, sc].dropna()
        b = df.loc[df["long_pfs"] == 0, sc].dropna()
        u_two = mannwhitneyu(a, b, alternative="two-sided")
        u_one = mannwhitneyu(a, b, alternative="greater")  # pre-specified: higher in LONG
        rbc = 2 * u_two.statistic / (len(a) * len(b)) - 1
        auc, auc_lo, auc_hi = bootstrap_auc(df["long_pfs"].values, df[sc].values)
        or_u, lo_u, hi_u, p_u, n_u = logistic_per_sd(df, sc, [])
        or_a, lo_a, hi_a, p_a, n_a = logistic_per_sd(df, sc, covs)
        rows.append({
            "score": sc,
            "label": SCORE_LABELS.get(sc, PUB_LABELS.get(sc, sc)),
            "independent_of_cohort": sc not in NON_INDEPENDENT,
            "median_long": a.median(), "median_short": b.median(),
            "n_long": len(a), "n_short": len(b),
            "rank_biserial": rbc,
            "p_two_sided": u_two.pvalue, "p_one_sided_higher_in_long": u_one.pvalue,
            "AUC": auc, "AUC_low_95": auc_lo, "AUC_up_95": auc_hi,
            "OR_per_SD_univariable": or_u, "OR_low_uni": lo_u, "OR_up_uni": hi_u, "p_uni": p_u, "n_uni": n_u,
            "OR_per_SD_adjusted": or_a, "OR_low_adj": lo_a, "OR_up_adj": hi_a, "p_adj": p_a, "n_adj": n_a,
            "adjusted_for": ", ".join(covs),
        })
    res = pd.DataFrame(rows)
    # FDR over the independent scores only, leaving out the Zhou positive control.
    ind = res["independent_of_cohort"]
    res.loc[ind, "q_two_sided"] = bh(res.loc[ind, "p_two_sided"].values)
    res.loc[ind, "q_one_sided"] = bh(res.loc[ind, "p_one_sided_higher_in_long"].values)
    res = res.sort_values("p_one_sided_higher_in_long")
    res.to_csv(OUT / "GSE65021_pfs_association.tsv", sep="\t", index=False)

    show = ["label", "median_long", "median_short", "rank_biserial", "AUC", "p_one_sided_higher_in_long",
            "q_one_sided", "OR_per_SD_adjusted", "p_adj"]
    print("\nLONG vs SHORT PFS on cetuximab (one-sided test is the pre-specified direction):")
    print(res[show].round(3).to_string(index=False))
    print("\nNOTE: zhou2025_predictive was derived on these 40 patients. It is a positive control and is excluded "
          "from the FDR correction.")

    make_figure(df, res, n_long, n_short)
    print(f"\nwrote {OUT}/GSE65021_scores.tsv, {OUT}/GSE65021_pfs_association.tsv, "
          f"{FIG}/Figure_6.svg/.png")


# Short forms for the panel d axis labels. The full citations are in the figure legend.
SHORT_LABELS = {
    "pEMT specificity": "pEMT specificity",
    "P(pEMT-high)": "P(pEMT-high)",
    "Puram epithelial": "Puram epithelial",
    "76GS": "76GS",
    "Puram pEMT": "Puram pEMT",
    "Hallmark EMT": "Hallmark EMT",
    "KS": "KS",
    "MLR-EMT": "MLR-EMT",
    "EGFR-induced EMT, 171 genes (Schinke 2022)": "EGFR-induced EMT",
    "EGFR invasion fDEGs, 46 genes (Zhou 2025)": "EGFR invasion fDEGs",
    "Invasive gene network, 59 genes (Zhou 2025)": "Invasive gene network",
    "Tumour budding signature, 28 genes (Ourailidis 2026)": "Tumour budding",
    "Cetuximab PFS predictors, 9 genes (Zhou 2025)": "Cetuximab PFS predictors",
}


def make_figure(df: pd.DataFrame, res: pd.DataFrame, n_long: int, n_short: int) -> None:
    """Draw Figure_6 on a 2 x 3 grid, then Supplementary_Figure_8.

      a  the classifier axis by outcome group
      b  its ROC curve, with the Zhou nine-gene set (derived in this cohort) as a positive control
      c  the direct arm scores of the single-cell partition, from 05_direct_arm_scores.py: AUC
         with bootstrap interval for every malignant-arm and stromal-arm definition
      d  rank-biserial correlation for every independently derived score, across the bottom row
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    fig = plt.figure(figsize=(mm(180), mm(138)))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.15], wspace=0.55, hspace=0.62)
    r0 = res[res["score"] == "pEMT_specificity"].iloc[0]

    # (a) the axis by outcome group
    ax = fig.add_subplot(gs[0, 0])
    groups = [df.loc[df["long_pfs"] == 0, "pEMT_specificity"], df.loc[df["long_pfs"] == 1, "pEMT_specificity"]]
    cols = [PALETTE["blue"], PALETTE["vermilion"]]
    bp = ax.boxplot(groups, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops={"color": "black", "linewidth": 1.1})
    for patch, col in zip(bp["boxes"], cols):
        patch.set_facecolor(col); patch.set_alpha(0.30); patch.set_edgecolor(col)
    rng = np.random.default_rng(20260920)
    for i, (g, col) in enumerate(zip(groups, cols), start=1):
        ax.scatter(rng.normal(i, 0.06, len(g)), g, s=10, color=col, alpha=0.85, linewidths=0, zorder=3)
    ax.set_xlim(0.4, 2.6)
    ax.set_ylim(-1.15, 1.45)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"Short PFS\nn = {n_short}", f"Long PFS\nn = {n_long}"])
    ax.set_ylabel("pEMT specificity")
    ax.text(0.5, 0.99, f"two-sided p = {r0['p_two_sided']:.3f}\n"
                       f"rank-biserial = {r0['rank_biserial']:+.2f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.5, linespacing=1.4)
    ax.set_title("a", loc="left", fontweight="bold")

    # (b) discrimination
    ax = fig.add_subplot(gs[0, 1])
    fpr, tpr, _ = roc_curve(df["long_pfs"], df["pEMT_specificity"])
    ax.plot(fpr, tpr, color=PALETTE["vermilion"], linewidth=1.5, zorder=3)
    rz = res[res["score"] == "zhou2025_predictive"].iloc[0]
    fpr2, tpr2, _ = roc_curve(df["long_pfs"], df["zhou2025_predictive"])
    ax.plot(fpr2, tpr2, color=PALETTE["grey"], linewidth=1.1, linestyle="--", zorder=2)
    ax.plot([0, 1], [0, 1], color=PALETTE["lightgrey"], linewidth=0.7, linestyle=":", zorder=1)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.text(0.97, 0.18, f"pEMT specificity {r0['AUC']:.2f}\n"
            f"({r0['AUC_low_95']:.2f} to {r0['AUC_up_95']:.2f})",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.3,
            color=PALETTE["vermilion"], linespacing=1.4)
    ax.text(0.97, 0.02, f"Zhou 9-gene {rz['AUC']:.2f}\n(derived in this cohort)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.3,
            color=PALETTE["grey"], linespacing=1.4)
    ax.set_title("b", loc="left", fontweight="bold")

    # (c) the two arms scored directly, from 05_direct_arm_scores.py
    ax = fig.add_subplot(gs[0, 2])
    arm_path = ROOT / "results" / "arm_scores" / "gse65021_arm_association.tsv"
    if arm_path.exists():
        arms = pd.read_csv(arm_path, sep="\t").set_index("score")
        # Cores first (primary, specified before the cohort was examined). Consensus sets last,
        # marked with a dagger because they were defined post hoc, after the first results.
        n = pd.read_csv(ROOT / "results" / "arm_scores" / "arm_gene_sets.tsv", sep="	")             .set_index("score")["n_genes"]
        rows = [("malignant_arm_core", f"core ({n['malignant_arm_core']})"),
                ("malignant_arm", f"total share ({n['malignant_arm']})"),
                ("malignant_consensus", f"consensus ({n['malignant_consensus']}) †"),
                ("stromal_arm_core", f"core ({n['stromal_arm_core']})"),
                ("stromal_arm", f"total share ({n['stromal_arm']})"),
                ("stromal_consensus", f"consensus ({n['stromal_consensus']}) †")]
        y = np.arange(len(rows))[::-1]
        for yi, (k, lab) in zip(y, rows):
            r = arms.loc[k]
            col = PALETTE["vermilion"] if k.startswith("malignant") else PALETTE["green"]
            ax.plot([r["AUC_low"], r["AUC_up"]], [yi, yi], color=col, linewidth=1.1)
            ax.scatter(r["AUC"], yi, s=18, color=col, zorder=3, linewidths=0)
        ax.axvline(0.5, color=PALETTE["grey"], linewidth=0.7, linestyle=":")
        ax.set_yticks(y)
        ax.set_yticklabels([lab for _, lab in rows], fontsize=6.3)
        for tick, (k, _) in zip(ax.get_yticklabels(), rows):
            tick.set_color(PALETTE["vermilion"] if k.startswith("malignant") else PALETTE["green"])
        ax.set_xlim(0.15, 1.0)
        ax.set_xticks([0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("AUC, long vs short PFS (95% CI)")
        ax.text(0.17, 5.42, "malignant arm", fontsize=6.3, color=PALETTE["vermilion"], va="bottom")
        ax.text(0.17, 2.42, "stromal arm", fontsize=6.3, color=PALETTE["green"], va="bottom")
        ax.set_ylim(-0.6, 6.0)
    ax.set_title("c", loc="left", fontweight="bold")

    # (d) every independently derived score, by direction of effect
    ax = fig.add_subplot(gs[1, :])
    plot_res = res[res["independent_of_cohort"]].sort_values("rank_biserial")
    y = np.arange(len(plot_res))
    bar_cols = [PALETTE["vermilion"] if s in ("pEMT_specificity", "P_pEMT_high") else PALETTE["black"]
                for s in plot_res["score"]]
    ax.barh(y, plot_res["rank_biserial"], color=bar_cols, alpha=0.9, height=0.66)
    ax.axvline(0, color=PALETTE["grey"], linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT_LABELS.get(l, l) for l in plot_res["label"]], fontsize=6.5)
    ax.set_ylim(-0.7, len(plot_res) + 0.45)
    ax.set_xlim(-0.62, 0.72)
    ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
    ax.set_xlabel("Rank-biserial correlation with long PFS")
    ax.text(0.45, len(plot_res) - 0.05, "higher in long PFS", ha="center", va="center", fontsize=6.3,
            color=PALETTE["grey"])
    ax.text(-0.42, len(plot_res) - 0.05, "higher in short PFS", ha="center", va="center", fontsize=6.3,
            color=PALETTE["grey"])
    ax.set_title("d", loc="left", fontweight="bold")

    fig.subplots_adjust(left=0.10, right=0.985, top=0.95, bottom=0.09)
    save_figure(fig, FIG, "Figure_6")
    combined_marker_figure(df, res)


def combined_marker_figure(df: pd.DataFrame, res: pd.DataFrame) -> None:
    """Draw Supplementary_Figure_8, the ROC curves of the exploratory combined marker and its parts.

    The combined marker is the sum of z-scored pEMT specificity and the z-scored Puram epithelial
    score. Its bootstrap interval is read from GSE65021_combined_score.tsv when that file exists.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    fig, ax = plt.subplots(figsize=(mm(80), mm(72)))
    r0 = res[res["score"] == "pEMT_specificity"].iloc[0]
    zs = lambda s: (s - s.mean()) / s.std()  # noqa: E731
    yy = df["long_pfs"].to_numpy(int)
    axis_z, epi_z = zs(df["pEMT_specificity"]), zs(df["puram_epi_dif_1"])
    comb = axis_z + epi_z
    comb_path = OUT / "GSE65021_combined_score.tsv"
    comb_ci = ""
    if comb_path.exists():
        cm = pd.read_csv(comb_path, sep="\t")
        row = cm[(cm["role"] == "primary")].iloc[0]
        comb_ci = f"\n({row['AUC_low']:.2f} to {row['AUC_up']:.2f})"
    re_ = res[res["score"] == "puram_epi_dif_1"].iloc[0]
    curves = [(comb, PALETTE["black"], 1.7, f"Combined {roc_auc_score(yy, comb):.2f}{comb_ci}"),
              (epi_z, PALETTE["blue"], 1.1, f"Puram epithelial {re_['AUC']:.2f}"),
              (axis_z, PALETTE["vermilion"], 1.1, f"pEMT specificity {r0['AUC']:.2f}")]
    for vals, col, lw, _ in curves:
        f_, t_, _ = roc_curve(yy, vals)
        ax.plot(f_, t_, color=col, linewidth=lw, zorder=3)
    ax.plot([0, 1], [0, 1], color=PALETTE["lightgrey"], linewidth=0.7, linestyle=":", zorder=1)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ypos = 0.02
    for _, col, _, lab in reversed(curves):
        ax.text(0.97, ypos, lab, transform=ax.transAxes, ha="right", va="bottom", fontsize=6.3,
                color=col, linespacing=1.4)
        ypos += 0.10 if "\n" not in lab else 0.17
    ax.text(0.03, 0.97, "exploratory", transform=ax.transAxes, ha="left", va="top", fontsize=6.3,
            color=PALETTE["grey"], style="italic")

    fig.subplots_adjust(left=0.2, right=0.96, top=0.95, bottom=0.17)
    save_figure(fig, FIG, "Supplementary_Figure_8")



if __name__ == "__main__":
    main()
