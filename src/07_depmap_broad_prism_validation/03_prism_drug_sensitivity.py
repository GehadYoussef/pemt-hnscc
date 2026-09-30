"""Test PRISM secondary-screen drug sensitivity against pEMT specificity within each cohort.

The curve table has several rows per compound and model (screens,
replicates). Rows failing the STR profiling QC are dropped, and AUC is
aggregated as the median per (compound, model). For every compound with at
least broad_prism.min_cells_per_drug (5) matched models, the output gives
the Spearman correlation between pEMT specificity and AUC (negative rho means
pEMT-high models are more sensitive), Benjamini-Hochberg FDR, and the median
AUC difference between the top and bottom pEMT tertiles. Compounds are
cross-referenced by name with the proximity hits of every seed set and target
definition (all hits, and the robust_hit subset).

Two set-level summaries follow. For each MoA class with at least 5 compounds,
a Mann-Whitney test compares its rho values with those of all other
compounds. This test treats compounds as independent and is anti-conservative,
so its p-values are descriptive only. The reportable class test is the
cell-line permutation null in 05_moa_permutation_test.py. For each set of
proximity-hit compounds, the same Mann-Whitney comparison is reported for
reference next to the cell-line permutation test from pemt.permtest
(broad_prism.moa_permutations, 5,000, seed broad_prism.moa_permutation_seed).

Inputs:  data/raw/prism_broad/secondary_screen/secondary-screen-dose-response-curve-parameters.csv
         results/depmap_broad_prism/depmap_class_probabilities_<cohort>.tsv
         results/network_analysis/<seed_name>/drug_proximity_*_seeds_significant.tsv
Outputs: results/depmap_broad_prism/prism_pemt_sensitivity_<cohort>.tsv,
         prism_matched_models_<cohort>.tsv, prism_moa_class_enrichment_<cohort>.tsv,
         prism_proximity_hit_set_test_<cohort>.tsv
         results/figures/panel_prism_sensitivity (.svg, .png), hnscc cohort:
         a, volcano of rho against nominal p with canonical-module DrugBank
         proximity hits highlighted
         b, rho of proximity-hit compounds against other compounds, with
         permutation p-values
Usage:   python src/07_depmap_broad_prism_validation/03_prism_drug_sensitivity.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pemt import load_config, project_root  # noqa: E402
from pemt.figstyle import apply_style, save_figure, PALETTE, mm  # noqa: E402
from pemt.permtest import permutation_setup, set_statistic  # noqa: E402

cfg = load_config()
ROOT = project_root()
PR = cfg["broad_prism"]
OUT = ROOT / cfg["paths"]["results_dir"] / "depmap_broad_prism"
NET = ROOT / cfg["paths"]["results_dir"] / "network_analysis"
FIG = ROOT / "results" / "figures"
MIN_N = int(PR["min_cells_per_drug"])
COHORTS = ["hnscc", "pan_squamous"]
N_PERM = int(PR.get("moa_permutations", 5000))
PERM_SEED = int(PR.get("moa_permutation_seed", 20260909))
apply_style()
import matplotlib.pyplot as plt  # noqa: E402


def proximity_hit_names() -> dict[str, set[str]]:
    hits = {}
    for d in NET.glob("*/drug_proximity_*_seeds_significant.tsv"):
        t = pd.read_csv(d, sep="\t")
        key = f"{d.parent.name}:{d.stem.replace('drug_proximity_', '').replace('_seeds_significant', '')}"
        hits[key] = set(t["name"].dropna().str.lower())
        if "robust_hit" in t:
            hits[key + ":robust"] = set(t.loc[t["robust_hit"], "name"].dropna().str.lower())
    return hits


def main() -> None:
    curve = pd.read_csv(ROOT / PR["secondary_curve_file"], usecols=["depmap_id", "screen_id", "auc", "name", "moa", "target", "phase", "passed_str_profiling", "broad_id"])
    n_all = len(curve)
    curve = curve[curve["passed_str_profiling"].astype(str).str.lower().isin(["true", "1"])]
    curve = curve.dropna(subset=["auc", "name", "depmap_id"])
    agg = curve.groupby(["name", "depmap_id"], as_index=False).agg(auc=("auc", "median"), n_curves=("auc", "size"),
                                                                   moa=("moa", "first"), target=("target", "first"), phase=("phase", "first"))
    print(f"PRISM secondary: {n_all:,} curve rows, {len(curve):,} passing QC, {len(agg):,} compound x model pairs")
    hits = proximity_hit_names()

    for cohort in COHORTS:
        sc = pd.read_csv(OUT / f"depmap_class_probabilities_{cohort}.tsv", sep="\t", index_col=0)
        m = agg.merge(sc[["pEMT_specificity", "P_pEMT_high", "cell_line_name", "is_organoid"]], left_on="depmap_id", right_index=True)
        models = m["depmap_id"].nunique()
        m[["depmap_id", "cell_line_name", "pEMT_specificity"]].drop_duplicates().to_csv(OUT / f"prism_matched_models_{cohort}.tsv", sep="\t", index=False)
        rows = []
        for name, g in m.groupby("name"):
            if len(g) < MIN_N:
                continue
            rho, p = spearmanr(g["pEMT_specificity"], g["auc"])
            q = g["pEMT_specificity"].quantile([1 / 3, 2 / 3])
            hi, lo = g[g["pEMT_specificity"] >= q.iloc[1]]["auc"], g[g["pEMT_specificity"] <= q.iloc[0]]["auc"]
            rows.append({"compound": name, "n_models": len(g), "spearman_rho": rho, "p": p,
                         "auc_median_pemt_high_tertile": hi.median(), "auc_median_pemt_low_tertile": lo.median(),
                         "delta_auc_high_minus_low": hi.median() - lo.median(),
                         "moa": g["moa"].iloc[0], "target": g["target"].iloc[0], "phase": g["phase"].iloc[0],
                         **{f"proximity_hit_{k}": name.lower() in v for k, v in hits.items()}})
        res = pd.DataFrame(rows)
        res["fdr"] = multipletests(res["p"], method="fdr_bh")[1]
        res = res.sort_values("spearman_rho")
        res.to_csv(OUT / f"prism_pemt_sensitivity_{cohort}.tsv", sep="\t", index=False)
        sig = res[res["fdr"] <= 0.1]
        print(f"[{cohort}] {models} models with PRISM data, {len(res)} compounds tested (>= {MIN_N} models), "
              f"FDR <= 0.1: {len(sig)} ({int((sig['spearman_rho'] < 0).sum())} more sensitive in pEMT-high), "
              f"top sensitising: {', '.join(f'{r.compound} ({r.spearman_rho:+.2f})' for r in res.head(6).itertuples())}")

        # mechanism-of-action class enrichment: for every MoA term with >= min_class compounds, test whether the compounds of
        # that class are shifted towards sensitivity (negative rho) or resistance in pEMT-high models. Mann-Whitney vs all
        # other compounds, BH-FDR.
        min_class = 5
        terms = {}
        for comp, moa_str in zip(res["compound"], res["moa"].astype(str)):
            for t in [x.strip() for x in moa_str.split(",") if x.strip() and x.strip().lower() != "nan"]:
                terms.setdefault(t, set()).add(comp)
        moa_rows = []
        for t, comps in terms.items():
            if len(comps) < min_class:
                continue
            inclass = res["compound"].isin(comps)
            a, b = res.loc[inclass, "spearman_rho"], res.loc[~inclass, "spearman_rho"]
            u, pv = mannwhitneyu(a, b, alternative="two-sided")
            moa_rows.append({"cohort": cohort, "moa_class": t, "n_compounds": int(inclass.sum()), "median_rho_class": a.median(), "median_rho_other": b.median(),
                             "fraction_negative_rho": float((a < 0).mean()), "direction": "sensitive in pEMT-high" if a.median() < b.median() else "resistant in pEMT-high",
                             "mannwhitney_p": pv, "top_compounds": ", ".join(res.loc[inclass].sort_values("spearman_rho")["compound"].head(5))})
        moa = pd.DataFrame(moa_rows)
        if len(moa):
            moa["fdr"] = multipletests(moa["mannwhitney_p"], method="fdr_bh")[1]
            moa = moa.sort_values("mannwhitney_p")
            moa.to_csv(OUT / f"prism_moa_class_enrichment_{cohort}.tsv", sep="\t", index=False)
            print(f"  MoA classes tested: {len(moa)}, FDR <= 0.1: {int((moa['fdr'] <= 0.1).sum())}")
            print("  NOTE: this Mann-Whitney test treats compounds as independent and is anti-conservative. "
                  "The reportable test is the cell-line permutation null in 05_moa_permutation_test.py. "
                  "These p-values are descriptive only.")
            print(moa.head(10)[["moa_class", "n_compounds", "median_rho_class", "direction", "mannwhitney_p", "fdr"]].round(4).to_string(index=False))

        # set-level test: shift of proximity-hit compounds towards sensitivity in pEMT-high models.
        # The Mann-Whitney version is for reference only. Hit compounds share targets and cell lines,
        # so it is anti-conservative. The reportable p-value is the cell-line permutation null (pemt.permtest).
        auc_mat = m.pivot(index="depmap_id", columns="name", values="auc")
        auc_mat = auc_mat.loc[:, auc_mat.notna().sum(axis=0) >= MIN_N]
        phen = sc.loc[auc_mat.index, "pEMT_specificity"].values
        obs, null, idx = permutation_setup(auc_mat, phen, N_PERM, PERM_SEED)
        set_rows = []
        for k in hits:
            col = f"proximity_hit_{k}"
            a, b = res.loc[res[col], "spearman_rho"], res.loc[~res[col], "spearman_rho"]
            if len(a) >= 3:
                u, p = mannwhitneyu(a, b, alternative="two-sided")
                comps = set(res.loc[res[col], "compound"]) & set(auc_mat.columns)
                st = set_statistic(obs, null, [idx[c] for c in comps]) if len(comps) >= 3 else {}
                st.pop("_nulls", None)
                set_rows.append({"cohort": cohort, "hit_set": k, "n_hit_compounds_in_prism": len(a), "n_other": len(b),
                                 "median_rho_hits": a.median(), "median_rho_other": b.median(),
                                 "mannwhitney_p_anticonservative": p,
                                 "permutation_mean_rho": st.get("mean_rho"), "permutation_z": st.get("z"),
                                 "permutation_p": st.get("perm_p")})
        if set_rows:
            pd.DataFrame(set_rows).to_csv(OUT / f"prism_proximity_hit_set_test_{cohort}.tsv", sep="\t", index=False)
            for r in set_rows:
                print(f"  set test {r['hit_set']}: {r['n_hit_compounds_in_prism']} hit compounds, median rho "
                      f"{r['median_rho_hits']:+.2f} vs {r['median_rho_other']:+.2f}, permutation p = {r['permutation_p']} "
                      f"(Mann-Whitney {r['mannwhitney_p_anticonservative']:.2g}, anti-conservative)")

        if cohort == "hnscc":
            from adjustText import adjust_text
            fig, axes = plt.subplots(1, 2, figsize=(mm(180), mm(75)), gridspec_kw={"width_ratios": [1.3, 1]})
            ax = axes[0]
            hit_col = "proximity_hit_canonical_pEMT:drugbank" if "proximity_hit_canonical_pEMT:drugbank" in res else None
            ax.scatter(res["spearman_rho"], -np.log10(res["p"]), s=6, color=PALETTE["grey"], alpha=0.5, linewidths=0, rasterized=True)
            if hit_col:
                h = res[res[hit_col]]
                ax.scatter(h["spearman_rho"], -np.log10(h["p"]), s=12, color=PALETTE["vermilion"], linewidths=0, zorder=3)
            lab = pd.concat([res.nsmallest(7, "p")[res.nsmallest(7, "p")["spearman_rho"] < 0], res[res["spearman_rho"] > 0].nsmallest(4, "p")])
            if hit_col:
                lab = pd.concat([lab, res[res[hit_col]].nsmallest(4, "p")]).drop_duplicates("compound")
            ax.set_ylim(-0.55, float(-np.log10(res["p"].min())) * 1.4)
            texts = [ax.text(r.spearman_rho, -np.log10(r.p), r.compound, fontsize=5.5) for r in lab.itertuples()]
            adjust_text(texts, ax=ax, arrowprops={"arrowstyle": "-", "color": PALETTE["grey"], "lw": 0.3},
                        expand=(1.35, 1.9), force_text=(0.6, 1.2), only_move={"text": "xy", "static": "xy"})
            ax.axvline(0, color=PALETTE["lightgrey"], linewidth=0.6)
            ax.set_xlabel(f"Spearman ρ, pEMT specificity vs PRISM AUC ({models} HNSCC models)")
            ax.set_ylabel("−log10 p (nominal)")
            ax.text(0.02, 0.97, f"{len(res)} compounds, {int((res['fdr'] <= 0.1).sum())} at FDR ≤ 0.1\nnegative ρ: more sensitive in pEMT-high models",
                    transform=ax.transAxes, va="top", fontsize=6.5, color=PALETTE["grey"])
            if hit_col:
                ax.text(0.02, 0.85, "orange: canonical-module proximity hits", transform=ax.transAxes, va="top", fontsize=6.5, color=PALETTE["vermilion"])
            ax.set_title("a", loc="left", fontweight="bold")
            ax2 = axes[1]
            sets = [("Other\ncompounds", ~(res.filter(like="proximity_hit_").any(axis=1)), PALETTE["grey"]),
                    ("M06 proximity\nhits", res.get("proximity_hit_canonical_pEMT:drugbank", pd.Series(False, index=res.index)), PALETTE["vermilion"]),
                    ("M13 proximity\nhits", res.get("proximity_hit_pEMT_axis:drugbank", pd.Series(False, index=res.index)), PALETTE["blue"]),
                    ("Classifier-seed\nproximity hits", res.get("proximity_hit_classifier_pEMT_high:drugbank", pd.Series(False, index=res.index)), PALETTE["orange"])]
            sets = [t for t in sets if t[1].sum() >= 3]
            groups = [res.loc[mask, "spearman_rho"].values for _, mask, _ in sets]
            rng = np.random.default_rng(0)
            for i, (d, (_, _, c)) in enumerate(zip(groups, sets), start=1):
                ax2.scatter(i + rng.uniform(-0.2, 0.2, len(d)), d, s=4 if i == 1 else 9, alpha=0.4 if i == 1 else 0.9, color=c, linewidths=0, rasterized=True, zorder=1)
            ax2.boxplot(groups, widths=0.45, showfliers=False, medianprops={"color": PALETTE["black"], "linewidth": 1.1}, zorder=2)
            ax2.set_xticks(range(1, len(groups) + 1)); ax2.set_xticklabels([f"{n}\n(n = {len(d)})" for (n, _, _), d in zip(sets, groups)], fontsize=6.5)
            ax2.axhline(0, color=PALETTE["lightgrey"], linewidth=0.6)
            ax2.set_ylabel("Spearman ρ, pEMT specificity vs AUC")
            plab = {"canonical_pEMT:drugbank": "M06", "pEMT_axis:drugbank": "M13", "classifier_pEMT_high:drugbank": "classifier-seed"}
            ptxt = ["permutation test over cell lines:"] + [
                f"{plab[r['hit_set']]} hits vs null: p = {r['permutation_p']:.2f}" for r in set_rows
                if r["hit_set"] in plab and r["permutation_p"] is not None]
            ax2.text(0.02, 0.98, "\n".join(ptxt), transform=ax2.transAxes, va="top", fontsize=6, color=PALETTE["grey"])
            ax2.set_title("b", loc="left", fontweight="bold")
            fig.tight_layout()
            save_figure(fig, FIG, "panel_prism_sensitivity")


if __name__ == "__main__":
    main()
