"""Relate the pEMT scores to protein abundance and EGFR to MAPK phosphorylation in CPTAC-3 HNSCC.

CPTAC-3 HNSCC (Huang et al. Cancer Cell 2021, LinkedOmics) has proteome and phosphoproteome data
on the tumours that also carry the transcriptomic scores. Only tumours with scores, proteome and
phosphoproteome are used.

Protein level. Every score is correlated (Spearman) with the abundance of each protein in five
protein sets fixed in advance: hemidesmosome and laminin-332, EGFR family receptor, fibrillar
collagen and stroma, classical mesenchymal, and basal keratin. q values are Benjamini-Hochberg.

Phosphosite level. pEMT specificity is correlated with every phosphosite quantified in at least
30 tumours. Sites measured on the same tumours are not independent, so the EGFR to MAPK cascade
is tested as a set. The statistic is the mean site correlation, and the null permutes pEMT
specificity across tumours and recomputes the statistic 2,000 times (two-sided p). Every site is
also tested after regressing out its parent protein measured in the same tumour (linear fit,
Spearman of the residual). Sites on ITGB4, the hemidesmosome integrin of the malignant arm, are
reported separately.

Figure_7: (a) median correlation with protein abundance per protein set, for pEMT specificity and
the canonical Puram pEMT score, (b) the distribution of site correlations with the cascade sites
and their permutation p, (c) raw against protein-adjusted site correlations.

Inputs:  results/tcga_projection/external/CPTAC_HNSCC_scores.tsv
         data/raw/cptac_proteome/HS_CPTAC_HNSCC_Proteomics_TMT_Gene_level_Tumor.cct
         data/raw/cptac_proteome/HS_CPTAC_HNSCC_Phosphoproteomics_TMT_site_level_Tumor.cct
Outputs: results/cptac_proteome/protein_correlations.tsv
         results/cptac_proteome/phosphosite_correlations.tsv
         results/cptac_proteome/pathway_permutation_test.tsv
         results/figures/Figure_7.svg and .png
Usage:   python src/11_composition_and_mechanism/02_cptac_proteome_phospho.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "data" / "raw" / "cptac_proteome"
OUT = ROOT / "results" / "cptac_proteome"
FIG = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))
from pemt.figstyle import apply_style, save_figure, PALETTE, mm  # noqa: E402

apply_style()
import matplotlib.pyplot as plt  # noqa: E402

SEED = 20260920
MIN_N = 30            # a site must be quantified in at least this many tumours
N_PERM = 2000

# Protein sets, with membership fixed in advance.
SETS = {
    "hemidesmosome / laminin-332": ["ITGB4", "ITGA3", "ITGA6", "COL17A1", "LAMA3", "LAMB3",
                                    "LAMC2", "DST", "PLEC"],
    "EGFR family receptor": ["EGFR", "ERBB2", "ERBB3"],
    "fibrillar collagen / stroma": ["COL1A1", "COL1A2", "COL3A1", "COL5A1", "COL5A2", "COL6A1",
                                    "COL6A2", "COL6A3", "FN1", "THBS1", "THBS2", "MMP2", "TAGLN"],
    "classical mesenchymal": ["VIM", "CDH2", "ZEB1", "SNAI2"],
    "basal keratin": ["KRT5", "KRT6A", "KRT14", "KRT17", "SFN"],
}
# EGFR to MAPK cascade, for the phosphosite pathway test
MAPK_PATHWAY = ["EGFR", "ERBB2", "ERBB3", "SHC1", "GRB2", "SOS1", "HRAS", "KRAS", "NRAS",
                "RAF1", "BRAF", "ARAF", "MAP2K1", "MAP2K2", "MAPK1", "MAPK3", "RPS6KA1",
                "RPS6KA3", "ELK1", "JUN", "FOS", "EGFR", "PTPN11", "GAB1", "CBL"]
# Scores correlated with protein abundance. MLR_mu is the MLR-EMT score.
SCORES = ["pEMT_specificity", "P_pEMT_high", "puram_pemt", "GS76", "hallmark_EMT", "KS", "MLR_mu"]


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sc = pd.read_csv(ROOT / "results" / "tcga_projection" / "external" / "CPTAC_HNSCC_scores.tsv",
                     sep="\t", index_col=0)
    prot = pd.read_csv(DATA / "HS_CPTAC_HNSCC_Proteomics_TMT_Gene_level_Tumor.cct",
                       sep="\t", index_col=0)
    ph = pd.read_csv(DATA / "HS_CPTAC_HNSCC_Phosphoproteomics_TMT_site_level_Tumor.cct", sep="\t")
    ph = ph.set_index("ID.ID")
    ph_gene = ph["gene"]
    ph = ph.drop(columns=[c for c in ("combine", "gene") if c in ph.columns])
    shared = sorted(set(sc.index) & set(prot.columns) & set(ph.columns))
    print(f"scores {len(sc)} cases, proteome {prot.shape[1]}, phospho {ph.shape[1]}, "
          f"{len(shared)} with all three")
    return sc.loc[shared], prot[shared], ph[shared].assign(gene=ph_gene).set_index(
        pd.Index(ph.index, name="site"))


def bh(p):
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    pv, n = p[ok], ok.sum()
    o = np.argsort(pv)
    adj = np.minimum.accumulate((pv[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n); out[o] = np.clip(adj, 0, 1); q[ok] = out
    return q


def corr(x: pd.Series, y: pd.Series, min_n: int = MIN_N):
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < min_n:
        return np.nan, np.nan, len(d)
    r = spearmanr(d.iloc[:, 0], d.iloc[:, 1])
    return float(r.statistic), float(r.pvalue), len(d)


def main() -> None:
    sc, prot, ph = load()
    ph_gene = ph.pop("gene")

    # ------------------------------------------------------- 1. protein level
    rows = []
    for setname, genes in SETS.items():
        for g in genes:
            if g not in prot.index:
                continue
            for s in SCORES:
                r, p, n = corr(sc[s], prot.loc[g])
                rows.append({"set": setname, "gene": g, "score": s, "rho": r, "p": p, "n": n})
    pr = pd.DataFrame(rows).dropna(subset=["rho"])
    pr["q"] = bh(pr["p"].values)
    pr.to_csv(OUT / "protein_correlations.tsv", sep="\t", index=False)

    piv = pr.pivot_table(index="set", columns="score", values="rho", aggfunc="median")
    print("\nmedian Spearman with protein abundance, by protein set:")
    print(piv[["pEMT_specificity", "puram_pemt", "GS76", "hallmark_EMT"]].round(2).to_string())

    # ------------------------------------------------------- 2. phosphosite level
    axis = sc["pEMT_specificity"]
    rows = []
    for site in ph.index:
        g = ph_gene.loc[site]
        v = ph.loc[site]
        r, p, n = corr(axis, v)
        if np.isnan(r):
            continue
        # protein-adjusted: residual of the site on its own parent protein
        r_adj = np.nan
        if g in prot.index:
            d = pd.concat([axis, v.rename("site"), prot.loc[g].rename("prot")], axis=1).dropna()
            if len(d) >= MIN_N and d["prot"].std() > 0:
                b = np.polyfit(d["prot"], d["site"], 1)
                resid = d["site"] - np.polyval(b, d["prot"])
                r_adj = float(spearmanr(d[axis.name], resid).statistic)
        rows.append({"site": site, "gene": g, "rho": r, "p": p, "n": n, "rho_protein_adjusted": r_adj})
    ps = pd.DataFrame(rows)
    ps["q"] = bh(ps["p"].values)
    ps["in_MAPK_pathway"] = ps["gene"].isin(MAPK_PATHWAY)
    ps.sort_values("rho").to_csv(OUT / "phosphosite_correlations.tsv", sep="\t", index=False)
    print(f"\n{len(ps)} phosphosites quantified in at least {MIN_N} tumours, "
          f"{int((ps['q'] <= 0.05).sum())} correlate with the axis at FDR 0.05")

    # pathway-level statistic against a permutation null over samples
    rng = np.random.default_rng(SEED)
    sub = ps[ps["in_MAPK_pathway"]]
    print(f"\nEGFR/MAPK cascade: {len(sub)} sites on {sub['gene'].nunique()} proteins")
    if len(sub):
        obs = float(sub["rho"].mean())
        idx = list(sub["site"])
        mat = ph.loc[idx]
        null = np.empty(N_PERM)
        a = axis.values
        for i in range(N_PERM):
            perm = pd.Series(rng.permutation(a), index=axis.index)
            null[i] = np.nanmean([corr(perm, mat.loc[s])[0] for s in idx])
        z = (obs - np.nanmean(null)) / np.nanstd(null)
        p_perm = float((np.abs(null) >= abs(obs)).mean())
        print(f"  mean rho {obs:+.3f}, permutation null mean {np.nanmean(null):+.3f}, "
              f"z {z:+.2f}, two-sided p = {p_perm:.4f}")
        pd.DataFrame([{"set": "EGFR/MAPK cascade", "n_sites": len(sub), "mean_rho": obs,
                       "null_mean": float(np.nanmean(null)), "null_sd": float(np.nanstd(null)),
                       "z": z, "p_permutation": p_perm, "n_permutations": N_PERM}]).to_csv(
            OUT / "pathway_permutation_test.tsv", sep="\t", index=False)
        print("\n  individual cascade sites (top by |rho|):")
        print(sub.reindex(sub["rho"].abs().sort_values(ascending=False).index)
              .head(12)[["site", "rho", "rho_protein_adjusted", "q", "n"]].round(3).to_string(index=False))

    itg = ps[ps["gene"] == "ITGB4"]
    print(f"\nITGB4 phosphosites: {len(itg)}")
    if len(itg):
        print(itg[["site", "rho", "rho_protein_adjusted", "q", "n"]].round(3).to_string(index=False))

    print("\nmost axis-correlated phosphosites overall:")
    print(ps.nlargest(12, "rho")[["site", "gene", "rho", "rho_protein_adjusted", "q"]]
          .round(3).to_string(index=False))

    # ------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(mm(180), mm(72)),
                             gridspec_kw={"width_ratios": [1.25, 1.0, 1.0], "wspace": 0.62})

    ax = axes[0]
    order = ["hemidesmosome / laminin-332", "EGFR family receptor", "basal keratin",
             "classical mesenchymal", "fibrillar collagen / stroma"]
    order = [o for o in order if o in set(pr["set"])]
    y = np.arange(len(order))
    for off, s, col, lab in [(-0.2, "pEMT_specificity", PALETTE["vermilion"], "pEMT specificity"),
                             (0.2, "puram_pemt", PALETTE["green"], "canonical pEMT score")]:
        vals = [pr[(pr["set"] == o) & (pr["score"] == s)]["rho"].median() for o in order]
        ax.barh(y + off, vals, height=0.38, color=col, alpha=0.9, label=lab)
    ax.axvline(0, color=PALETTE["grey"], lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([o.replace(" / ", "/\n") for o in order], fontsize=6.5)
    ax.set_xlabel("Median Spearman with protein")
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2,
              columnspacing=1.0, handlelength=1.0, handletextpad=0.4)
    ax.set_title("a", loc="left", fontweight="bold")

    ax = axes[1]
    bg = ps[~ps["in_MAPK_pathway"]]["rho"].dropna()
    fg = ps[ps["in_MAPK_pathway"]]["rho"].dropna()
    # Cascade sites are drawn as a rug below the histogram so that they do not hide it.
    n, _, _ = ax.hist(bg, bins=40, color=PALETTE["lightgrey"], density=True,
                      label=f"other sites (n = {len(bg):,})")
    top = n.max()
    if len(fg):
        ax.scatter(fg, np.full(len(fg), -0.10 * top), s=7, marker="|",
                   color=PALETTE["vermilion"], linewidths=0.8,
                   label=f"EGFR/MAPK cascade (n = {len(fg)})")
        ax.axvline(fg.mean(), color=PALETTE["vermilion"], lw=1.4)
        ax.annotate(f"cascade mean {fg.mean():+.3f}\npermutation p = {p_perm:.2f}",
                    (fg.mean(), top * 0.58), fontsize=6, ha="left" if fg.mean() < 0.1 else "right",
                    xytext=(4 if fg.mean() < 0.1 else -4, 0), textcoords="offset points",
                    color=PALETTE["vermilion"], linespacing=1.4)
    ax.axvline(0, color="black", lw=0.6, ls="--")
    ax.set_ylim(-0.18 * top, top * 1.12)
    ax.set_xlabel("Spearman with pEMT specificity")
    ax.set_ylabel("Density")
    ax.legend(fontsize=5.5, loc="upper left", frameon=True, framealpha=1, edgecolor="none")
    ax.set_title("b", loc="left", fontweight="bold")

    ax = axes[2]
    d = ps.dropna(subset=["rho_protein_adjusted"])
    ax.scatter(d["rho"], d["rho_protein_adjusted"], s=3, color=PALETTE["lightgrey"],
               linewidths=0, rasterized=True)
    dm = d[d["in_MAPK_pathway"]]
    ax.scatter(dm["rho"], dm["rho_protein_adjusted"], s=14, color=PALETTE["vermilion"],
               linewidths=0, label="EGFR/MAPK cascade")
    di = d[d["gene"] == "ITGB4"]
    ax.scatter(di["rho"], di["rho_protein_adjusted"], s=18, color=PALETTE["blue"],
               linewidths=0, label="ITGB4")
    ax.axhline(0, color=PALETTE["grey"], lw=0.6); ax.axvline(0, color=PALETTE["grey"], lw=0.6)
    ax.set_xlabel("Spearman, raw site")
    ax.set_ylabel("Spearman, parent protein\nregressed out")
    ax.legend(fontsize=6, loc="upper left")
    ax.set_title("c", loc="left", fontweight="bold")

    fig.subplots_adjust(left=0.115, right=0.985, top=0.91, bottom=0.26)
    save_figure(fig, FIG, "Figure_7")
    print(f"\nwrote {OUT}/ and {FIG}/Figure_7.svg/.png")


if __name__ == "__main__":
    main()
