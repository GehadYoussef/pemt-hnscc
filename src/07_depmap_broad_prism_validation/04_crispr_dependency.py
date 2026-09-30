"""Test CRISPR (Chronos) gene dependency against pEMT specificity within each cohort.

Uses the DepMap Chronos gene effect (more negative means more essential).
Genes measured in fewer than 90% of a cohort's models are dropped. For each
cohort the script reports:
  * per gene, the Spearman correlation between pEMT specificity and gene
    effect across models (at least 8 models, negative rho means more essential
    in pEMT-high models), with Benjamini-Hochberg FDR over all genes
  * mean gene effect in the top and bottom pEMT tertiles
  * a gene-set summary for the WGCNA seed sets and for the targets of
    significant proximity drugs (targets that are, or are adjacent to,
    disease-set genes). Each set with at least 5 tested genes is compared
    with all other genes by a Mann-Whitney test on rho.

Inputs:  data/raw/depmap/crispr/CRISPRGeneEffect.csv
         results/depmap_broad_prism/depmap_class_probabilities_<cohort>.tsv
         results/wgcna/network_seeds.tsv
         results/network_analysis/<seed_name>/drug_proximity_*_seeds_significant.tsv
Outputs: results/depmap_broad_prism/crispr_pemt_dependency_<cohort>.tsv,
         crispr_geneset_summary_<cohort>.tsv
         results/figures/panel_crispr_dependency (.svg, .png), hnscc cohort:
         a, volcano of rho against nominal p with the pEMT-axis and canonical
         pEMT seed genes highlighted
         b, rho of M13 genes, M06 genes and M06 proximity-drug targets against
         all other genes, with Mann-Whitney p-values
Usage:   python src/07_depmap_broad_prism_validation/04_crispr_dependency.py
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

cfg = load_config()
ROOT = project_root()
OUT = ROOT / cfg["paths"]["results_dir"] / "depmap_broad_prism"
NET = ROOT / cfg["paths"]["results_dir"] / "network_analysis"
WG = ROOT / cfg["paths"]["results_dir"] / "wgcna"
FIG = ROOT / "results" / "figures"
CRISPR = ROOT / "data" / "raw" / "depmap" / "crispr" / "CRISPRGeneEffect.csv"
COHORTS = ["hnscc", "pan_squamous"]
apply_style()
import matplotlib.pyplot as plt  # noqa: E402


def gene_sets() -> dict[str, set[str]]:
    sets = {}
    seeds = pd.read_csv(WG / "network_seeds.tsv", sep="\t")
    for s, g in seeds.groupby("seed_name"):
        sets[f"seed:{s}"] = set(g["gene"])
    for d in NET.iterdir():
        for f in d.glob("drug_proximity_*_seeds_significant.tsv") if d.is_dir() else []:
            t = pd.read_csv(f, sep="\t")
            targets = set()
            for col in ("targets_that_are_key_proteins", "targets_adjacent_to_key_proteins"):
                for v in t[col].dropna():
                    targets |= set(str(v).split(";"))
            sets[f"proximal_drug_targets:{d.name}:{f.stem.replace('drug_proximity_', '').replace('_seeds_significant', '')}"] = targets - {""}
    return sets


def main() -> None:
    ce = pd.read_csv(CRISPR, index_col=0)
    ce.columns = [c.split(" (")[0] for c in ce.columns]
    ce = ce.loc[:, ~pd.Index(ce.columns).duplicated()]
    print(f"CRISPR gene effect: {ce.shape[0]} models x {ce.shape[1]} genes")
    sets = gene_sets()
    for cohort in COHORTS:
        sc = pd.read_csv(OUT / f"depmap_class_probabilities_{cohort}.tsv", sep="\t", index_col=0)
        ids = sc.index.intersection(ce.index)
        X = ce.loc[ids]
        y = sc.loc[ids, "pEMT_specificity"]
        keep = X.notna().mean() >= 0.9
        X = X.loc[:, keep]
        rho = np.full(X.shape[1], np.nan); pv = np.full(X.shape[1], np.nan)
        yv = y.values
        for j, g in enumerate(X.columns):
            x = X[g].values
            ok = np.isfinite(x)
            if ok.sum() >= 8:
                rho[j], pv[j] = spearmanr(yv[ok], x[ok])
        q = y.quantile([1 / 3, 2 / 3])
        hi, lo = y >= q.iloc[1], y <= q.iloc[0]
        res = pd.DataFrame({"gene": X.columns, "n_models": X.notna().sum().values, "spearman_rho": rho, "p": pv,
                            "mean_effect_pemt_high_tertile": X[hi].mean().values, "mean_effect_pemt_low_tertile": X[lo].mean().values,
                            "mean_effect_all": X.mean().values})
        res["delta_effect_high_minus_low"] = res["mean_effect_pemt_high_tertile"] - res["mean_effect_pemt_low_tertile"]
        res = res.dropna(subset=["p"])
        res["fdr"] = multipletests(res["p"], method="fdr_bh")[1]
        for name, s in sets.items():
            res[f"in_{name}"] = res["gene"].isin(s)
        res = res.sort_values("spearman_rho")
        res.to_csv(OUT / f"crispr_pemt_dependency_{cohort}.tsv", sep="\t", index=False)

        summ = []
        for name, s in sets.items():
            inset = res[res["gene"].isin(s)]
            rest = res[~res["gene"].isin(s)]
            if len(inset) >= 5:
                u, p = mannwhitneyu(inset["spearman_rho"], rest["spearman_rho"])
                summ.append({"gene_set": name, "n_genes_tested": len(inset), "median_rho_set": inset["spearman_rho"].median(),
                             "median_rho_other": rest["spearman_rho"].median(), "p_mannwhitney": p,
                             "n_fdr10_selective_in_pemt_high": int(((inset["fdr"] <= 0.1) & (inset["spearman_rho"] < 0)).sum()),
                             "top_selective": ", ".join(inset.head(8)["gene"])})
        summ = pd.DataFrame(summ)
        summ.to_csv(OUT / f"crispr_geneset_summary_{cohort}.tsv", sep="\t", index=False)
        sig = res[(res["fdr"] <= 0.1) & (res["spearman_rho"] < 0)]
        print(f"[{cohort}] {len(ids)} models with CRISPR, {len(res)} genes, pEMT-high-selective dependencies at FDR <= 0.1: {len(sig)}, "
              f"top: {', '.join(res.head(10)['gene'])}")
        if len(summ):
            print(summ[["gene_set", "n_genes_tested", "median_rho_set", "median_rho_other", "p_mannwhitney", "n_fdr10_selective_in_pemt_high"]].round(3).to_string(index=False))

        if cohort == "hnscc":
            from adjustText import adjust_text
            fig, axes = plt.subplots(1, 2, figsize=(mm(180), mm(75)), gridspec_kw={"width_ratios": [1.3, 1]})
            ax = axes[0]
            ax.scatter(res["spearman_rho"], -np.log10(res["p"]), s=4, color=PALETTE["grey"], alpha=0.4, linewidths=0, rasterized=True)
            lab = res.nsmallest(8, "p")
            for name, colour, short in (("seed:pEMT_axis", PALETTE["vermilion"], "pEMT-axis seed genes"), ("seed:canonical_pEMT", PALETTE["blue"], "canonical pEMT seed genes")):
                sub = res[res["gene"].isin(sets.get(name, set()))]
                ax.scatter(sub["spearman_rho"], -np.log10(sub["p"]), s=10, color=colour, linewidths=0, zorder=3, label=f"{short} (n = {len(sub)})")
                lab = pd.concat([lab, sub[sub["p"] < 0.05]])
            lab = lab.drop_duplicates("gene")
            texts = [ax.text(r.spearman_rho, -np.log10(r.p), r.gene, fontsize=5.5) for r in lab.itertuples()]
            adjust_text(texts, ax=ax, arrowprops={"arrowstyle": "-", "color": PALETTE["grey"], "lw": 0.3}, expand=(1.2, 1.6))
            ax.axvline(0, color=PALETTE["lightgrey"], linewidth=0.6)
            ax.set_xlabel(f"Spearman ρ, pEMT specificity vs Chronos gene effect ({len(ids)} HNSCC models)")
            ax.set_ylabel("−log10 p (nominal)")
            ax.text(0.02, 0.97, f"{len(res):,} genes\n{len(sig)} pEMT-high-selective at FDR ≤ 0.1\nnegative ρ: stronger dependency\nin pEMT-high models",
                    transform=ax.transAxes, va="top", fontsize=6.5, color=PALETTE["grey"])
            ax.set_ylim(-0.1, float(-np.log10(res["p"].min())) * 1.45)
            ax.legend(loc="upper right", fontsize=6)
            ax.set_title("a", loc="left", fontweight="bold")

            ax2 = axes[1]
            panels = [("All other\ngenes", None, PALETTE["grey"]), ("M13 module\ngenes", "seed:pEMT_axis", PALETTE["vermilion"]),
                      ("M06 module\ngenes", "seed:canonical_pEMT", PALETTE["blue"]),
                      ("M06 drug\ntargets", "proximal_drug_targets:canonical_pEMT:drugbank", PALETTE["orange"])]
            in_any = res["gene"].isin(set().union(*[sets.get(k, set()) for _, k, _ in panels if k]))
            groups, labels, cols = [], [], []
            for name, key, c in panels:
                d = res.loc[~in_any, "spearman_rho"].values if key is None else res.loc[res["gene"].isin(sets.get(key, set())), "spearman_rho"].values
                if len(d) == 0:
                    continue
                groups.append(d); cols.append(c)
                ptxt = ""
                if key is not None:
                    row = summ[summ["gene_set"] == key] if len(summ) else pd.DataFrame()
                    if len(row):
                        ptxt = f"\np = {float(row.iloc[0]['p_mannwhitney']):.2f}"
                labels.append(f"{name}\n(n = {len(d)}){ptxt}")
            rng = np.random.default_rng(0)
            for i, (d, c) in enumerate(zip(groups, cols), start=1):
                ax2.scatter(i + rng.uniform(-0.2, 0.2, len(d)), d, s=2 if i == 1 else 8, alpha=0.25 if i == 1 else 0.9, color=c, linewidths=0, rasterized=True, zorder=1)
            ax2.boxplot(groups, widths=0.45, showfliers=False, medianprops={"color": PALETTE["black"], "linewidth": 1.1}, zorder=2)
            ax2.set_xticks(range(1, len(groups) + 1)); ax2.set_xticklabels(labels, fontsize=6)
            ax2.axhline(0, color=PALETTE["lightgrey"], linewidth=0.6)
            ax2.set_ylabel("Spearman ρ, pEMT specificity vs gene effect")
            ax2.text(0.02, 0.98, "Mann-Whitney vs all other genes\nM13 pEMT-axis module, M06 canonical pEMT module",
                     transform=ax2.transAxes, va="top", fontsize=6, color=PALETTE["grey"])
            ax2.set_title("b", loc="left", fontweight="bold")
            fig.tight_layout()
            save_figure(fig, FIG, "panel_crispr_dependency")


if __name__ == "__main__":
    main()
