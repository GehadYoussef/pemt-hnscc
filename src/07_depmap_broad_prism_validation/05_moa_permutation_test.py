"""Test mechanism-of-action class effects in PRISM against a permutation null over cell lines.

The per-class Mann-Whitney test in 03_prism_drug_sensitivity.py treats the
compounds' Spearman rho values as independent. Every rho is computed on the
same 23 (HNSCC) or 63 (pan-squamous) cell lines and compounds with related
targets are strongly correlated, so that test is anti-conservative.

Here the phenotype labels are permuted across cell lines
(broad_prism.moa_permutations, 5,000, seed broad_prism.moa_permutation_seed)
and the class statistic is recomputed for each permutation. The
compound-compound correlation structure is left intact, so the null absorbs
the redundancy between compounds.

Statistic. AUC is ranked within each compound across the lines that compound
was screened in. pEMT specificity is ranked across all matched lines. For
compound j, rho_j is the correlation of those two rank vectors over the lines
where j has data. The class statistic is mean(rho_j) over the class. Observed
and permuted statistics are computed with identical machinery (pemt.permtest),
so the test is exact under the label-permutation null regardless of the
ranking convention. Each MoA class with at least 5 compounds gets a z-score, a
two-sided empirical p with add-one correction, and Benjamini-Hochberg FDR
across classes within the cohort. The proximity-hit compound sets are tested
against the same null and reported without FDR.

Inputs:  data/raw/prism_broad/secondary_screen/secondary-screen-dose-response-curve-parameters.csv
         results/depmap_broad_prism/depmap_class_probabilities_<cohort>.tsv,
         prism_pemt_sensitivity_<cohort>.tsv
Outputs: results/depmap_broad_prism/prism_moa_permutation_test.tsv (one row per
         MoA class or proximity set per cohort)
         results/figures/Supplementary_Figure_14 (.svg, .png): observed class
         effect against the central 95% of its permutation null for the
         plotted classes, both cohorts. A combined version of this panel is
         panel c of Supplementary_Figure_15, drawn by
         src/12_figures_and_tables/02_combined_figures.py.
Usage:   python src/07_depmap_broad_prism_validation/05_moa_permutation_test.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pemt import load_config, project_root  # noqa: E402
from pemt.figstyle import apply_style, save_figure, PALETTE, mm  # noqa: E402
from pemt.permtest import permutation_setup, set_statistic  # noqa: E402

cfg = load_config()
ROOT = project_root()
PR = cfg["broad_prism"]
OUT = ROOT / cfg["paths"]["results_dir"] / "depmap_broad_prism"
FIG = ROOT / "results" / "figures"
MIN_N = int(PR["min_cells_per_drug"])
MIN_CLASS = 5
N_PERM = int(PR.get("moa_permutations", 5000))
SEED = int(PR.get("moa_permutation_seed", 20260909))
COHORTS = ["hnscc", "pan_squamous"]
# classes plotted in the figure. Every class with >= MIN_CLASS compounds is tested.
FOCUS = ["EGFR inhibitor", "MEK inhibitor", "RAF inhibitor", "src inhibitor", "AKT inhibitor",
         "PI3K inhibitor", "mTOR inhibitor", "HDAC inhibitor", "tubulin polymerization inhibitor",
         "topoisomerase inhibitor"]
apply_style()
import matplotlib.pyplot as plt  # noqa: E402


def moa_terms(res: pd.DataFrame) -> dict[str, set[str]]:
    terms: dict[str, set[str]] = {}
    for comp, m in zip(res["compound"], res["moa"].astype(str)):
        for t in [x.strip() for x in m.split(",") if x.strip() and x.strip().lower() != "nan"]:
            terms.setdefault(t, set()).add(comp)
    return terms


def main() -> None:
    curve = pd.read_csv(ROOT / PR["secondary_curve_file"],
                        usecols=["depmap_id", "auc", "name", "moa", "passed_str_profiling"])
    curve = curve[curve["passed_str_profiling"].astype(str).str.lower().isin(["true", "1"])]
    curve = curve.dropna(subset=["auc", "name", "depmap_id"])
    agg = curve.groupby(["name", "depmap_id"], as_index=False).agg(auc=("auc", "median"), moa=("moa", "first"))

    rows = []
    figdata = {}
    for cohort in COHORTS:
        sc = pd.read_csv(OUT / f"depmap_class_probabilities_{cohort}.tsv", sep="\t", index_col=0)
        m = agg[agg["depmap_id"].isin(sc.index)]
        auc = m.pivot(index="depmap_id", columns="name", values="auc")
        auc = auc.loc[:, auc.notna().sum(axis=0) >= MIN_N]
        phen = sc.loc[auc.index, "pEMT_specificity"].values

        res = pd.read_csv(OUT / f"prism_pemt_sensitivity_{cohort}.tsv", sep="\t")
        res = res[res["compound"].isin(auc.columns)]
        terms = {t: c & set(auc.columns) for t, c in moa_terms(res).items()}
        terms = {t: c for t, c in terms.items() if len(c) >= MIN_CLASS}

        obs, null, idx = permutation_setup(auc, phen, N_PERM, SEED)

        print(f"[{cohort}] {auc.shape[0]} lines, {auc.shape[1]} compounds, {len(terms)} MoA classes with >= {MIN_CLASS} compounds, {N_PERM} permutations")
        crow = []
        for term, comps in sorted(terms.items()):
            s = set_statistic(obs, null, [idx[c] for c in comps])
            nulls = s.pop("_nulls")
            crow.append({"cohort": cohort, "class": term, "n_cmpd": s.pop("n_features"), **s})
            if term in FOCUS:
                figdata[(cohort, term)] = (s["mean_rho"], nulls)
        # proximity-hit compound sets share targets and cell lines like the MoA classes, so they get the same null
        hit_rows = []
        for col in [c for c in res.columns if c.startswith("proximity_hit_")]:
            comps = set(res.loc[res[col].astype(bool), "compound"]) & set(auc.columns)
            if len(comps) < 3:
                continue
            s = set_statistic(obs, null, [idx[c] for c in comps])
            s.pop("_nulls")
            hit_rows.append({"cohort": cohort, "class": "proximity set: " + col.replace("proximity_hit_", ""),
                             "n_cmpd": s.pop("n_features"), **s})
        for r in hit_rows:
            print(f"   {r['class']:44s} n={r['n_cmpd']:>3d}  mean rho {r['mean_rho']:+.3f}  z {r['z']:+.2f}  p {r['perm_p']:.4f}")

        c = pd.DataFrame(crow)
        c["perm_fdr"] = multipletests(c["perm_p"], method="fdr_bh")[1].round(4)
        c = pd.concat([c, pd.DataFrame(hit_rows)], ignore_index=True)
        rows.append(c)
        top = c.sort_values("perm_p").head(6)
        for r in top.itertuples():
            print(f"   {r._2:34s} n={r.n_cmpd:>3d}  mean rho {r.mean_rho:+.3f}  z {r.z:+.2f}  p {r.perm_p:.4f}  FDR {r.perm_fdr:.3f}")

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT / "prism_moa_permutation_test.tsv", sep="\t", index=False)
    print(f"wrote {OUT / 'prism_moa_permutation_test.tsv'} ({len(out)} class x cohort rows)")

    # figure: observed class effect against its own permutation null
    fig, axes = plt.subplots(1, 2, figsize=(mm(180), mm(78)), sharex=True)
    titles = {"hnscc": "HNSCC models", "pan_squamous": "Pan-squamous models"}
    for ax, cohort, letter in zip(axes, COHORTS, "ab"):
        present = [t for t in FOCUS if (cohort, t) in figdata]
        y = np.arange(len(present))[::-1]
        tab = out[out["cohort"] == cohort].set_index("class")
        for yi, t in zip(y, present):
            stat, nulls = figdata[(cohort, t)]
            lo, hi = np.percentile(nulls, [2.5, 97.5])
            ax.plot([lo, hi], [yi, yi], color=PALETTE["lightgrey"], linewidth=4, solid_capstyle="butt", zorder=1)
            sig = tab.loc[t, "perm_p"] <= 0.05
            ax.scatter(stat, yi, s=26, color=PALETTE["vermilion"] if sig else PALETTE["grey"], zorder=3, linewidths=0)
        ax.axvline(0, color=PALETTE["black"], linewidth=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{t.replace(' inhibitor', '')} (n = {int(tab.loc[t, 'n_cmpd'])})" for t in present], fontsize=6.5)
        ax.set_xlabel("Mean Spearman ρ, pEMT specificity vs AUC")
        ax.set_title(titles[cohort], fontsize=7, color=PALETTE["grey"])
        ax.text(-0.44, 1.06, letter, transform=ax.transAxes, fontweight="bold", fontsize=8)
        for yi, t in zip(y, present):
            ax.text(0.985, (yi + 0.5) / len(present), f"p = {tab.loc[t, 'perm_p']:.2f}", transform=ax.transAxes,
                    ha="right", va="center", fontsize=5.8, color=PALETTE["grey"])
    axes[0].set_xlim(-0.45, 0.60)
    for ax in axes:
        ax.set_xticks([-0.4, -0.2, 0.0, 0.2, 0.4])
    fig.text(0.5, 0.005, "grey bar: 95% of the permutation null (phenotype labels shuffled across cell lines, "
                         f"{N_PERM:,} permutations). Negative ρ: more sensitive in pEMT-high models.",
             ha="center", fontsize=6, color=PALETTE["grey"])
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    save_figure(fig, FIG, "Supplementary_Figure_14")


if __name__ == "__main__":
    main()
