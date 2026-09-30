"""Association of pEMT specificity and the EMT score panel with TCGA-HNSC clinicopathology.

Repeats the TCGA analysis of Puram et al. 2017 (their Fig. 6), which related
the bulk pEMT signature to nodal metastasis, grade, lymphovascular invasion
and extracapsular extension, for every score in the panel on the 520
TCGA-HNSC primaries:

  * pathological N stage: N+ vs N0 (Mann-Whitney two-sided, rank-biserial
    effect size), and N0 / N1 / N2 / N3 trend (Spearman)
  * pathological T stage: T3-T4 vs T1-T2 (Mann-Whitney)
  * HPV status (Mann-Whitney) and anatomical site (Kruskal-Wallis)
  * histological grade (Spearman trend), lymphovascular invasion, perineural
    invasion, extracapsular spread and margin status (Mann-Whitney), when the
    legacy TCGA clinicalMatrix is present

Inputs:  results/tcga_projection/tcga_master_trait_table.tsv
         results/tcga_projection/emt_score_panel_scores.tsv
         data/raw/tcga_hnsc/clinical/TCGA.HNSC.sampleMap_HNSC_clinicalMatrix (optional)
Outputs: results/tcga_projection/clinical_associations.tsv
         results/figures/Supplementary_Figure_18 (pEMT specificity by N stage, T stage, HPV,
         site, and the legacy clinicalMatrix variables when present)
Usage:   python src/04_tcga_projection/08_clinical_associations.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pemt import load_config, project_root  # noqa: E402
from pemt.figstyle import apply_style, save_figure, PALETTE, mm  # noqa: E402
from pemt.survival import primary_tumours, site_group, SITE_LABEL  # noqa: E402

cfg = load_config()
ROOT = project_root()
OUT = ROOT / cfg["paths"]["results_dir"] / "tcga_projection"
FIG = ROOT / "results" / "figures"
LEGACY = ROOT / "data" / "raw" / "tcga_hnsc" / "clinical" / "TCGA.HNSC.sampleMap_HNSC_clinicalMatrix"
apply_style()
import matplotlib.pyplot as plt  # noqa: E402

SCORES = {"pEMT_specificity": "pEMT specificity", "P_pEMT_high": "P(pEMT-high)", "puram_pemt": "Puram pEMT signature",
          "hallmark_EMT": "Hallmark EMT", "KS": "KS score", "GS76": "76GS", "MLR_mu": "MLR"}


def n_group(v):
    if pd.isna(v) or v in ("NX",):
        return np.nan
    return "N0" if v == "N0" else "N+"


def n_ordinal(v):
    if pd.isna(v) or v == "NX":
        return np.nan
    return {"N0": 0, "N1": 1, "N2": 2, "N2a": 2, "N2b": 2, "N2c": 2, "N3": 3}.get(v, np.nan)


def t_group(v):
    if pd.isna(v) or v in ("TX", "T0"):
        return np.nan
    return "T1-T2" if v in ("T1", "T2") else "T3-T4"


def rank_biserial(a, b):
    u, _ = mannwhitneyu(a, b)
    return 2 * u / (len(a) * len(b)) - 1


def main() -> None:
    traits = primary_tumours(pd.read_csv(OUT / "tcga_master_trait_table.tsv", sep="\t", index_col=0, low_memory=False))
    panel = pd.read_csv(OUT / "emt_score_panel_scores.tsv", sep="\t", index_col=0)
    df = traits.join(panel.drop(columns=[c for c in panel.columns if c in traits.columns]))
    df["N_group"] = df["ajcc_pathologic_n.diagnoses"].map(n_group)
    df["N_ordinal"] = df["ajcc_pathologic_n.diagnoses"].map(n_ordinal)
    df["T_group"] = df["ajcc_pathologic_t.diagnoses"].map(t_group)
    df["site"] = df["tissue_or_organ_of_origin.diagnoses"].map(site_group)
    df["hpv"] = df["hpv_status"]

    extra = {}
    if LEGACY.exists():
        lg = pd.read_csv(LEGACY, sep="\t", index_col=0, low_memory=False)
        lg.index = [s[:15] for s in lg.index]
        ece = lg["presence_of_pathological_nodal_extracapsular_spread"].map(
            {"No Extranodal Extension": "NO", "Microscopic Extension": "YES", "Gross Extension": "YES"}) if "presence_of_pathological_nodal_extracapsular_spread" in lg.columns else None
        if ece is not None:
            lg["extracapsular_spread_pathologic"] = ece
        if "margin_status" in lg.columns:
            lg["positive_margin"] = lg["margin_status"].map({"Positive": "YES", "Negative": "NO", "Close": "NO"})
        for col, label in (("neoplasm_histologic_grade", "grade"), ("lymphovascular_invasion_present", "LVI"),
                           ("perineural_invasion_present", "PNI"), ("extracapsular_spread_pathologic", "ECE"), ("positive_margin", "margin")):
            if col in lg.columns:
                df[label] = df.index.map(lambda s: lg[col].get(s[:15], np.nan))
                extra[label] = col

    rows = []
    for sc, lab in SCORES.items():
        x = pd.to_numeric(df[sc], errors="coerce")
        # N0 vs N+
        a, b = x[df["N_group"] == "N+"].dropna(), x[df["N_group"] == "N0"].dropna()
        u, p = mannwhitneyu(a, b)
        rows.append({"score": sc, "label": lab, "variable": "pathological N stage", "comparison": "N+ vs N0", "n_a": len(a), "n_b": len(b),
                     "median_a": a.median(), "median_b": b.median(), "effect_rank_biserial": rank_biserial(a, b), "p": p})
        ok = df["N_ordinal"].notna() & x.notna()
        r, p = spearmanr(df.loc[ok, "N_ordinal"], x[ok])
        rows.append({"score": sc, "label": lab, "variable": "pathological N stage", "comparison": "trend N0<N1<N2<N3 (Spearman)", "n_a": int(ok.sum()), "n_b": np.nan,
                     "median_a": np.nan, "median_b": np.nan, "effect_rank_biserial": r, "p": p})
        a, b = x[df["T_group"] == "T3-T4"].dropna(), x[df["T_group"] == "T1-T2"].dropna()
        u, p = mannwhitneyu(a, b)
        rows.append({"score": sc, "label": lab, "variable": "pathological T stage", "comparison": "T3-T4 vs T1-T2", "n_a": len(a), "n_b": len(b),
                     "median_a": a.median(), "median_b": b.median(), "effect_rank_biserial": rank_biserial(a, b), "p": p})
        a, b = x[df["hpv"] == "HPV+"].dropna(), x[df["hpv"] == "HPV-"].dropna()
        u, p = mannwhitneyu(a, b)
        rows.append({"score": sc, "label": lab, "variable": "HPV status", "comparison": "HPV+ vs HPV-", "n_a": len(a), "n_b": len(b),
                     "median_a": a.median(), "median_b": b.median(), "effect_rank_biserial": rank_biserial(a, b), "p": p})
        groups = [x[df["site"] == s].dropna() for s in SITE_LABEL]
        h, p = kruskal(*groups)
        rows.append({"score": sc, "label": lab, "variable": "anatomical site", "comparison": "Kruskal-Wallis across sites", "n_a": int(sum(len(g) for g in groups)), "n_b": np.nan,
                     "median_a": np.nan, "median_b": np.nan, "effect_rank_biserial": np.nan, "p": p})
        for label in extra:
            v = df[label].astype(str).str.upper()
            if label == "grade":
                ok = v.isin(["G1", "G2", "G3", "G4"]) & x.notna()
                if ok.sum() > 20:
                    r, p = spearmanr(v[ok].map({"G1": 1, "G2": 2, "G3": 3, "G4": 4}), x[ok])
                    rows.append({"score": sc, "label": lab, "variable": "histological grade", "comparison": "trend G1<G2<G3<G4 (Spearman)", "n_a": int(ok.sum()), "n_b": np.nan,
                                 "median_a": np.nan, "median_b": np.nan, "effect_rank_biserial": r, "p": p})
            else:
                a, b = x[v == "YES"].dropna(), x[v == "NO"].dropna()
                if len(a) >= 10 and len(b) >= 10:
                    u, p = mannwhitneyu(a, b)
                    rows.append({"score": sc, "label": lab, "variable": label, "comparison": "present vs absent", "n_a": len(a), "n_b": len(b),
                                 "median_a": a.median(), "median_b": b.median(), "effect_rank_biserial": rank_biserial(a, b), "p": p})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "clinical_associations.tsv", sep="\t", index=False)
    show = res[res["score"].isin(["pEMT_specificity", "puram_pemt", "hallmark_EMT"])]
    print(show[["label", "variable", "comparison", "n_a", "n_b", "median_a", "median_b", "effect_rank_biserial", "p"]].round(3).to_string(index=False))

    # Figure: pEMT specificity by N stage, T stage, HPV, site (+ grade, LVI, PNI, ECE when the legacy clinicalMatrix is present)
    x = df["pEMT_specificity"]
    panels = [("Pathological N stage", "N_group", ["N0", "N+"], "pathological N stage", "N+ vs N0", "Mann-Whitney"),
              ("Pathological T stage", "T_group", ["T1-T2", "T3-T4"], "pathological T stage", "T3-T4 vs T1-T2", "Mann-Whitney"),
              ("HPV status", "hpv", ["HPV-", "HPV+"], "HPV status", "HPV+ vs HPV-", "Mann-Whitney"),
              ("Anatomical site", "site", list(SITE_LABEL), "anatomical site", "Kruskal-Wallis across sites", "Kruskal-Wallis")]
    if "grade" in extra:
        df["grade_group"] = df["grade"].astype(str).str.upper().where(df["grade"].astype(str).str.upper().isin(["G1", "G2", "G3", "G4"]))
        df.loc[df["grade_group"] == "G4", "grade_group"] = "G3"
        panels.append(("Histological grade", "grade_group", ["G1", "G2", "G3"], "histological grade", "trend G1<G2<G3<G4 (Spearman)", "Spearman trend"))
    for label, title in (("LVI", "Lymphovascular invasion"), ("PNI", "Perineural invasion"), ("ECE", "Extranodal extension")):
        if label in extra:
            df[f"{label}_group"] = df[label].astype(str).str.upper().where(df[label].astype(str).str.upper().isin(["YES", "NO"]))
            panels.append((title, f"{label}_group", ["NO", "YES"], label, "present vs absent", "Mann-Whitney"))
    ncol = 4
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(mm(180), mm(58 * nrow)), gridspec_kw={"width_ratios": [1, 1, 1, 1.4]})
    axes = np.array(axes).reshape(nrow, ncol)
    rng = np.random.default_rng(0)
    for k, (title, col, levels, var, comp, test) in enumerate(panels):
        ax = axes[k // ncol, k % ncol]
        data = [x[df[col] == lv].dropna().values for lv in levels]
        for i, d in enumerate(data, start=1):
            ax.scatter(i + rng.uniform(-0.22, 0.22, len(d)), d, s=3, alpha=0.3, color=PALETTE["orange"], linewidths=0, rasterized=True, zorder=1)
        ax.boxplot(data, widths=0.5, showfliers=False, medianprops={"color": PALETTE["black"], "linewidth": 1.1},
                   boxprops={"linewidth": 0.8}, whiskerprops={"linewidth": 0.8}, capprops={"linewidth": 0.8}, zorder=2)
        ax.set_xticks(range(1, len(levels) + 1))
        short = {"oropharynx": "Oropharynx", "oral_cavity": "Oral cavity", "larynx_hypopharynx": "Larynx/\nhypopharynx", "NO": "absent", "YES": "present"}
        labels = [f"{short.get(lv, SITE_LABEL.get(lv, lv))}\n(n = {len(d)})" for lv, d in zip(levels, data)]
        ax.set_xticklabels(labels, fontsize=6.5)
        ax.set_title(title, fontsize=8, pad=10)
        sub = res[(res["score"] == "pEMT_specificity") & (res["variable"] == var) & (res["comparison"] == comp)]
        p = float(sub.iloc[0]["p"]) if len(sub) else np.nan
        ax.text(0.5, 1.01, f"{test} " + ("p < 0.001" if p < 0.001 else f"p = {p:.2f}"), transform=ax.transAxes, va="bottom", ha="center", fontsize=6.5, color=PALETTE["grey"])
        ax.set_ylim(-1.08, 1.08)
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        if k % ncol == 0:
            ax.set_ylabel("pEMT specificity")
    for k in range(len(panels), nrow * ncol):
        axes[k // ncol, k % ncol].set_axis_off()
    fig.tight_layout()
    save_figure(fig, FIG, "Supplementary_Figure_18")


if __name__ == "__main__":
    main()
