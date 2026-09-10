"""Compare pEMT specificity with published EMT scores on TCGA-HNSC primaries.

Added 9 Sep 2026 (replaces 07_mlr_emt_comparison.py, kept in _originals_pre_9Sep2026).

Scores computed on log2(TPM+1), primary tumours only:
  76GS        Byers et al. 2013: sum over the 76 genes of expression weighted by
              each gene's Pearson correlation with CDH1 across the cohort, centred.
              Higher = more epithelial. (csb-iisc/EMT_score_MicroArray implementation)
  KS          Tan et al. 2014 tumour signature (144 epithelial / 170 mesenchymal genes):
              per-sample two-sample Kolmogorov-Smirnov statistic between the
              mesenchymal and epithelial gene expression distributions, signed
              as in the csb-iisc implementation. Higher = more mesenchymal.
  Hallmark EMT  MSigDB HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION (200 genes):
              mean per-gene z-score. Higher = more mesenchymal.
  Puram pEMT  mean per-gene z-score of the Puram et al. 2017 pEMT program (100 genes).
  Puram epithelial differentiation 1  same, for the epi_dif_1 program.
  MLR         George et al. 2017 ordinal model (three predictors, twenty normalisers), cross-platform
              recalibrated as in the original script (documented approximation;
              the published 20-gene normaliser step is not reproduced).
              mu in [0, 2]; higher = more mesenchymal.

For every score: Spearman correlation with pEMT specificity and with each other,
univariable Cox HR per SD, and adjusted Cox (stage, age, site, HPV) HR per SD on
the same rows, plus repeated 5-fold CV C-index alone.

Head-to-head (added after review): pEMT specificity and each other score in the same
adjusted Cox model, likelihood-ratio tests for adding either to the other, and the
cross-validated C-index of clinical + score with and without pEMT specificity.

Outputs (results/tcga_projection/):
  emt_score_panel_scores.tsv, emt_score_panel_correlations.tsv,
  emt_score_panel_cox.tsv, emt_score_panel_cv_cindex.tsv, emt_score_panel_joint_cox.tsv
Figures: Figure_5_emt_score_panel (A scatter vs each score, B adjusted HR per SD, C joint models)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import load_config, load_signatures, project_root  # noqa: E402
from _lib.figstyle import apply_style, save_figure, PALETTE, mm  # noqa: E402
from _lib.survival import primary_tumours, prepare_covariates, design_matrix, fit_cox, cv_cindex  # noqa: E402

cfg = load_config()
ROOT = project_root()
OUT = ROOT / cfg["paths"]["results_dir"] / "tcga_projection"
FIG = ROOT / "manuscript" / "figures"
REF = ROOT / cfg["paths"]["references_dir"] / "emt_scores"
apply_style()
import matplotlib.pyplot as plt  # noqa: E402

ALPHA1, ALPHA2, BETA1, BETA2 = -7.87, 0.0413, 1.36, -1.96
SCORE_LABELS = {
    "pEMT_specificity": "pEMT specificity (this study)",
    "P_pEMT_high": "P(pEMT-high) (this study)",
    "GS76": "76GS (Byers 2013; epithelial-high)",
    "KS": "KS (Tan 2014)",
    "hallmark_EMT": "Hallmark EMT (MSigDB)",
    "puram_pemt": "Puram pEMT signature",
    "puram_epi_dif_1": "Puram epithelial differentiation",
    "MLR_mu": "MLR (George 2017)",
}


def zmean(expr: pd.DataFrame, genes: list[str]) -> pd.Series:
    g = [x for x in genes if x in expr.index]
    X = expr.loc[g]
    Z = X.sub(X.mean(axis=1), axis=0).div(X.std(axis=1).replace(0, 1), axis=0)
    return Z.mean(axis=0), len(g), len(genes)


def score_76gs(expr: pd.DataFrame) -> tuple[pd.Series, int]:
    tab = pd.read_csv(REF / "gs76_genes.tsv", sep="\t")
    genes = []
    for old, new in zip(tab["symbol_byers2013"], tab["symbol_current"]):
        if new in expr.index:
            genes.append(new)
        elif old in expr.index:
            genes.append(old)
    genes = list(dict.fromkeys(genes))
    X = expr.loc[genes]
    cdh1 = expr.loc["CDH1"]
    w = X.apply(lambda r: np.corrcoef(r.values, cdh1.values)[0, 1], axis=1)
    score = (X.mul(w, axis=0)).sum(axis=0)
    return score - score.mean(), len(genes)


def score_ks(expr: pd.DataFrame) -> tuple[pd.Series, int, int]:
    sig = pd.read_csv(REF / "ks_tumor_signature.tsv", sep="\t")
    epi = [g for g in sig.loc[sig["class"] == "Epi", "gene"] if g in expr.index]
    mes = [g for g in sig.loc[sig["class"] == "Mes", "gene"] if g in expr.index]
    E, M = expr.loc[epi], expr.loc[mes]
    out = {}
    for s in expr.columns:
        e, m = E[s].values, M[s].values
        two = ks_2samp(m, e)
        grt = ks_2samp(m, e, alternative="greater")   # ecdf(Mes) > ecdf(Epi): mesenchymal genes LOWER
        less = ks_2samp(e, m, alternative="greater")  # ecdf(Epi) > ecdf(Mes): epithelial genes LOWER -> mesenchymal
        if grt.pvalue < 0.05:
            out[s] = -grt.statistic
        elif less.pvalue < 0.05:
            out[s] = less.statistic
        else:
            mx = max(grt.statistic, less.statistic)
            out[s] = mx if less.statistic == mx else -mx
    return pd.Series(out), len(epi), len(mes)


def score_mlr(expr: pd.DataFrame) -> pd.Series:
    x1 = expr.loc["CLDN7"].values.astype(float)
    x2 = (expr.loc["VIM"] - expr.loc["CDH1"]).values.astype(float)
    x1 = (x1 - x1.mean()) / (x1.std() + 1e-9) * 2.0 + 8.0
    x2 = (x2 - x2.mean()) / (x2.std() + 1e-9) * 2.0
    eta = BETA1 * x1 + BETA2 * x2
    sig = lambda v: 1.0 / (1.0 + np.exp(-v))  # noqa: E731
    p_le_e, p_le_em = sig(ALPHA1 + eta), sig(ALPHA2 + eta)
    mu = 1 * np.clip(p_le_em - p_le_e, 0, 1) + 2 * (1 - p_le_em)
    return pd.Series(mu, index=expr.columns)


def main() -> None:
    expr = pd.read_csv(ROOT / cfg["paths"]["processed_dir"] / "tcga_hnsc" / "tcga_star_tpm_log2_for_projection.tsv",
                       sep="\t", index_col=0)
    traits = primary_tumours(pd.read_csv(OUT / "tcga_master_trait_table.tsv", sep="\t", index_col=0, low_memory=False))
    expr = expr[traits.index]
    sigs = load_signatures()
    cov = []

    scores = pd.DataFrame(index=traits.index)
    scores["pEMT_specificity"] = traits["pEMT_specificity"]
    scores["P_pEMT_high"] = traits["P_pEMT_high"]
    scores["GS76"], n76 = score_76gs(expr); cov.append(f"76GS: {n76}/76 genes")
    scores["KS"], ne, nm = score_ks(expr); cov.append(f"KS tumour signature: {ne} epithelial, {nm} mesenchymal genes")
    hm = [l.strip() for l in (REF / "hallmark_emt_genes.txt").read_text().splitlines() if l.strip()]
    scores["hallmark_EMT"], a, b = zmean(expr, hm); cov.append(f"Hallmark EMT: {a}/{b} genes")
    scores["puram_pemt"], a, b = zmean(expr, sigs["puram_pemt"]); cov.append(f"Puram pEMT: {a}/{b} genes")
    scores["puram_epi_dif_1"], a, b = zmean(expr, sigs["puram_epi_dif_1"]); cov.append(f"Puram epi_dif_1: {a}/{b} genes")
    scores["MLR_mu"] = score_mlr(expr)
    scores.to_csv(OUT / "emt_score_panel_scores.tsv", sep="\t")
    (OUT / "emt_score_panel_gene_coverage.txt").write_text("\n".join(cov) + "\n")

    cols = list(SCORE_LABELS)
    rho = pd.DataFrame(index=cols, columns=cols, dtype=float)
    pval = rho.copy()
    for a in cols:
        for b in cols:
            r, p = spearmanr(scores[a], scores[b], nan_policy="omit")
            rho.loc[a, b], pval.loc[a, b] = r, p
    rho.to_csv(OUT / "emt_score_panel_correlations.tsv", sep="\t")
    pval.to_csv(OUT / "emt_score_panel_correlation_pvalues.tsv", sep="\t")

    # Cox per SD, same rows for every score
    df = prepare_covariates(traits.join(scores.drop(columns=["pEMT_specificity", "P_pEMT_high"])), cols)
    for c in cols:
        df[c] = (df[c] - df[c].mean()) / df[c].std()
    cox_rows, cv_rows = [], []
    cov2 = ["stage_num", "age", "site_group", "hpv_positive"]
    for c in cols:
        d1 = design_matrix(df, c, [])
        _, r1 = fit_cox(d1)
        d2 = design_matrix(df, c, cov2)
        _, r2 = fit_cox(d2)
        cox_rows.append({"score": c, "label": SCORE_LABELS[c],
                         "HR_per_SD_univariable": r1.loc[c, "HR"], "CI_low_uni": r1.loc[c, "HR_low_95"], "CI_up_uni": r1.loc[c, "HR_up_95"],
                         "p_univariable": r1.loc[c, "p_value"], "n_uni": r1.loc[c, "n"], "events_uni": r1.loc[c, "events"],
                         "HR_per_SD_adjusted": r2.loc[c, "HR"], "CI_low_adj": r2.loc[c, "HR_low_95"], "CI_up_adj": r2.loc[c, "HR_up_95"],
                         "p_adjusted": r2.loc[c, "p_value"], "n_adj": r2.loc[c, "n"], "events_adj": r2.loc[c, "events"]})
        cv = cv_cindex(d1, {c: [c]}, n_repeats=10)
        cv_rows.append({"score": c, "label": SCORE_LABELS[c], "cv_c_index": cv["cv_c_index_mean"][0], "cv_c_index_sd": cv["cv_c_index_sd"][0]})
    cox = pd.DataFrame(cox_rows)
    cox.to_csv(OUT / "emt_score_panel_cox.tsv", sep="\t", index=False)
    pd.DataFrame(cv_rows).to_csv(OUT / "emt_score_panel_cv_cindex.tsv", sep="\t", index=False)
    print(cox[["label", "HR_per_SD_univariable", "p_univariable", "HR_per_SD_adjusted", "p_adjusted", "n_adj"]].round(3).to_string(index=False))

    # Head-to-head: pEMT specificity and each other score in the SAME adjusted model.
    # Reports both coefficients, the likelihood-ratio test for adding pEMT specificity to
    # clinical + score (and for adding the score to clinical + pEMT specificity), and the
    # cross-validated C-index of clinical + score vs clinical + score + pEMT specificity.
    from scipy.stats import chi2
    joint_rows = []
    ref = "pEMT_specificity"
    for c in [x for x in cols if x not in (ref, "P_pEMT_high")]:
        d_both = design_matrix(df, ref, cov2 + [c])
        cph_both, r_both = fit_cox(d_both)
        d_score = d_both.drop(columns=[ref]); cph_score, _ = fit_cox(d_score)
        d_ref = d_both.drop(columns=[c]); cph_ref, _ = fit_cox(d_ref)
        lr_add_ref = 2 * (cph_both.log_likelihood_ - cph_score.log_likelihood_)
        lr_add_score = 2 * (cph_both.log_likelihood_ - cph_ref.log_likelihood_)
        clin = [x for x in d_both.columns if x not in ("OS_time", "OS_event", ref, c)]
        cv = cv_cindex(d_both, {"clinical": clin, "clinical + score": clin + [c], "clinical + pEMT specificity": clin + [ref], "clinical + both": clin + [c, ref]}, n_repeats=10)
        cvd = dict(zip(cv["model"], cv["cv_c_index_mean"]))
        joint_rows.append({"score": c, "label": SCORE_LABELS[c], "n": r_both.loc[ref, "n"], "events": r_both.loc[ref, "events"],
                           "HR_pEMT_specificity_given_score": r_both.loc[ref, "HR"], "CI_low_pEMT": r_both.loc[ref, "HR_low_95"], "CI_up_pEMT": r_both.loc[ref, "HR_up_95"], "p_pEMT_specificity_given_score": r_both.loc[ref, "p_value"],
                           "HR_score_given_pEMT_specificity": r_both.loc[c, "HR"], "CI_low_score": r_both.loc[c, "HR_low_95"], "CI_up_score": r_both.loc[c, "HR_up_95"], "p_score_given_pEMT_specificity": r_both.loc[c, "p_value"],
                           "LRT_p_adding_pEMT_to_clinical_plus_score": chi2.sf(lr_add_ref, 1), "LRT_p_adding_score_to_clinical_plus_pEMT": chi2.sf(lr_add_score, 1),
                           "cv_cindex_clinical": cvd["clinical"], "cv_cindex_clinical_plus_score": cvd["clinical + score"],
                           "cv_cindex_clinical_plus_pEMT": cvd["clinical + pEMT specificity"], "cv_cindex_clinical_plus_both": cvd["clinical + both"]})
    joint = pd.DataFrame(joint_rows)
    joint.to_csv(OUT / "emt_score_panel_joint_cox.tsv", sep="\t", index=False)
    print(joint[["label", "HR_pEMT_specificity_given_score", "p_pEMT_specificity_given_score", "HR_score_given_pEMT_specificity", "p_score_given_pEMT_specificity",
                 "LRT_p_adding_pEMT_to_clinical_plus_score", "cv_cindex_clinical_plus_score", "cv_cindex_clinical_plus_both"]].round(3).to_string(index=False))

    # Figure 6: A pEMT specificity against each score (scatter, Spearman rho); B adjusted HR per SD for every score;
    # C head-to-head joint models: HR of pEMT specificity given the score, and of the score given pEMT specificity
    SHORT = {"pEMT_specificity": "pEMT specificity", "P_pEMT_high": "P(pEMT-high)", "GS76": "76GS", "KS": "KS", "hallmark_EMT": "Hallmark EMT",
             "puram_pemt": "Puram pEMT", "puram_epi_dif_1": "Puram epithelial", "MLR_mu": "MLR"}
    others = [c for c in cols if c not in ("pEMT_specificity", "P_pEMT_high")]
    fig = plt.figure(figsize=(mm(180), mm(120)))
    gs = fig.add_gridspec(2, 13, height_ratios=[1, 1.1], hspace=0.55, wspace=1.2)
    for j, c in enumerate(others):
        ax = fig.add_subplot(gs[0, 2 * j:2 * j + 2])
        ax.scatter(scores[c], scores["pEMT_specificity"], s=3, alpha=0.35, color=PALETTE["grey"], linewidths=0, rasterized=True)
        lo, hi = np.nanpercentile(scores[c], [0.5, 99.5]); pad = 0.05 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)
        r = rho.loc["pEMT_specificity", c]
        ax.text(0.04, 0.96, f"ρ = {r:.2f}", transform=ax.transAxes, va="top", fontsize=6.5)
        ax.set_xlabel(SHORT[c], fontsize=7)
        ax.set_yticks([-1, 0, 1])
        if j == 0:
            ax.set_ylabel("pEMT specificity", fontsize=7)
            ax.set_title("a", loc="left", fontweight="bold")
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=6)
    # B
    axB = fig.add_subplot(gs[1, 0:6])
    y = np.arange(len(cols))[::-1]
    for i, (_, row) in enumerate(cox.iterrows()):
        colour = PALETTE["vermilion"] if row["score"] in ("pEMT_specificity", "P_pEMT_high") else PALETTE["black"]
        axB.errorbar(row["HR_per_SD_adjusted"], y[i], xerr=[[row["HR_per_SD_adjusted"] - row["CI_low_adj"]], [row["CI_up_adj"] - row["HR_per_SD_adjusted"]]],
                     fmt="o", color=colour, ecolor=colour, capsize=2, markersize=3.5, elinewidth=0.8)
    axB.set_xlim(0.8, 1.42)
    axB.axvline(1.0, ls="--", color=PALETTE["grey"], linewidth=0.6)
    axB.set_yticks(y); axB.set_yticklabels([SHORT[c] for c in cols], fontsize=7)
    axB.set_xlabel("Adjusted HR per SD (95% CI)")
    axB.set_title("b", loc="left", fontweight="bold")
    # C head-to-head
    axC = fig.add_subplot(gs[1, 7:13])
    yj = np.arange(len(joint))[::-1]
    for i, (_, row) in enumerate(joint.iterrows()):
        axC.errorbar(row["HR_pEMT_specificity_given_score"], yj[i] + 0.18,
                     xerr=[[row["HR_pEMT_specificity_given_score"] - row["CI_low_pEMT"]], [row["CI_up_pEMT"] - row["HR_pEMT_specificity_given_score"]]],
                     fmt="o", color=PALETTE["vermilion"], ecolor=PALETTE["vermilion"], capsize=2, markersize=3.5, elinewidth=0.8)
        axC.errorbar(row["HR_score_given_pEMT_specificity"], yj[i] - 0.18,
                     xerr=[[row["HR_score_given_pEMT_specificity"] - row["CI_low_score"]], [row["CI_up_score"] - row["HR_score_given_pEMT_specificity"]]],
                     fmt="s", color=PALETTE["black"], ecolor=PALETTE["black"], capsize=2, markersize=3.5, elinewidth=0.8)
    axC.axvline(1.0, ls="--", color=PALETTE["grey"], linewidth=0.6)
    axC.set_yticks(yj); axC.set_yticklabels([SHORT[c] for c in joint["score"]], fontsize=7)
    axC.set_xlabel("HR per SD, both scores in one adjusted model (95% CI)")
    axC.scatter([], [], marker="o", color=PALETTE["vermilion"], s=14, label="pEMT specificity, given the score")
    axC.scatter([], [], marker="s", color=PALETTE["black"], s=14, label="the score, given pEMT specificity")
    axC.legend(loc="upper left", bbox_to_anchor=(0.0, -0.3), fontsize=6, frameon=False, ncol=1)
    axC.set_title("c", loc="left", fontweight="bold")
    save_figure(fig, FIG, "Figure_5_emt_score_panel")


if __name__ == "__main__":
    main()
