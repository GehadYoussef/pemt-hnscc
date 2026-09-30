"""Correlate WGCNA modules with traits, annotate them by program enrichment and rank pEMT candidates.

Only primary tumours are used. Normal samples sit at the epithelial extreme
and would drive the epithelial-vs-pEMT axis. Each module eigengene is
correlated (Pearson) with the classifier traits and with the EMT score panel
(76GS, KS, Hallmark EMT, Puram signature scores, MLR) from
src/04_tcga_projection/07_emt_score_panel.py. These p-values are descriptive,
because the classifier traits come from the same expression matrix.

Every module is tested for enrichment of each signature (the six Puram
programs, the lineage marker sets and Hallmark EMT) by a one-sided
hypergeometric test within the WGCNA gene universe. FDR is p x n / rank
across all module-program pairs, capped at 1. Modules whose
eigengene correlates positively with P(pEMT-high) are ranked by Puram-pEMT
enrichment p, then by that correlation. pEMT_specificity is not used for
ranking, because its epithelial-loss term favours epithelial modules.

Inputs:  results/wgcna/gene_module_assignments.tsv, gene_kME.tsv,
         module_eigengenes.tsv
         results/tcga_projection/tcga_master_trait_table.tsv,
         emt_score_panel_scores.tsv
         data/references/emt_scores/hallmark_emt_genes.txt
         config/signatures.yaml, config/signatures/*.txt
Outputs: results/wgcna/module_trait_correlations.tsv, module_trait_pvalues.tsv,
         module_program_enrichment.tsv, module_annotation.tsv,
         module_pemt_ranking.tsv
         results/figures/Supplementary_Figure_3 (.svg, .png): module x trait
         correlation heatmap, rows labelled by the most enriched program
         (hypergeometric p < 1e-3) and module size
Usage:   python src/05_wgcna/03_module_trait_correlations.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pemt import load_config, load_signatures, project_root  # noqa: E402
from pemt.figstyle import apply_style, save_figure, PALETTE, mm  # noqa: E402
from pemt.survival import primary_tumours  # noqa: E402

cfg = load_config()
ROOT = project_root()
WG = ROOT / cfg["paths"]["results_dir"] / "wgcna"
TCGA = ROOT / cfg["paths"]["results_dir"] / "tcga_projection"
REF = ROOT / cfg["paths"]["references_dir"] / "emt_scores"
FIG = ROOT / "results" / "figures"
apply_style()
import matplotlib.pyplot as plt  # noqa: E402

TRAIT_LABELS = {
    "P_pEMT_high": "P(pEMT-high)", "P_epithelial_like": "P(epithelial-like)",
    "P_fibroblast_stromal_like": "P(fibroblast / stromal-like)", "pEMT_specificity": "pEMT specificity",
    "puram_pemt": "Puram pEMT signature", "puram_epi_dif_1": "Puram epithelial differentiation",
    "hallmark_EMT": "Hallmark EMT", "KS": "KS score", "GS76": "76GS", "MLR_mu": "MLR",
}


def main() -> None:
    assign = pd.read_csv(WG / "gene_module_assignments.tsv", sep="\t")
    kme = pd.read_csv(WG / "gene_kME.tsv", sep="\t", index_col=0)
    ME = pd.read_csv(WG / "module_eigengenes.tsv", sep="\t", index_col=0)
    traits = primary_tumours(pd.read_csv(TCGA / "tcga_master_trait_table.tsv", sep="\t", index_col=0, low_memory=False))
    panel = pd.read_csv(TCGA / "emt_score_panel_scores.tsv", sep="\t", index_col=0)
    traits = traits.join(panel.drop(columns=[c for c in panel.columns if c in traits.columns]))
    common = ME.index.intersection(traits.index)
    ME, traits = ME.loc[common], traits.loc[common]
    trait_cols = [c for c in TRAIT_LABELS if c in traits.columns]

    cor = pd.DataFrame(index=ME.columns, columns=trait_cols, dtype=float)
    pv = cor.copy()
    for me in ME.columns:
        for t in trait_cols:
            x = pd.to_numeric(traits[t], errors="coerce")
            ok = x.notna().values
            r, p = pearsonr(ME[me].values[ok], x.values[ok])
            cor.loc[me, t], pv.loc[me, t] = r, p
    cor.to_csv(WG / "module_trait_correlations.tsv", sep="\t")
    pv.to_csv(WG / "module_trait_pvalues.tsv", sep="\t")

    # program enrichment
    sigs = load_signatures()
    hm = [l.strip() for l in (REF / "hallmark_emt_genes.txt").read_text().splitlines() if l.strip()]
    sigs["hallmark_EMT"] = hm
    universe = set(assign["gene"])
    rows = []
    for me in ME.columns:
        mod = me.replace("ME_", "")
        genes = set(assign.loc[assign["module"] == mod, "gene"])
        for name, sg in sigs.items():
            s = set(sg) & universe
            k = len(genes & s)
            p = hypergeom.sf(k - 1, len(universe), len(s), len(genes)) if k else 1.0
            rows.append({"module": mod, "program": name, "module_size": len(genes), "program_genes_in_universe": len(s),
                         "overlap": k, "expected": len(genes) * len(s) / len(universe), "p_hypergeom": p})
    enr = pd.DataFrame(rows)
    enr["fdr"] = np.minimum(1.0, enr["p_hypergeom"] * len(enr) / enr["p_hypergeom"].rank())
    enr.to_csv(WG / "module_program_enrichment.tsv", sep="\t", index=False)

    # annotation table: size, best program, top hubs by kME, key correlations
    ann = []
    for me in ME.columns:
        mod = me.replace("ME_", "")
        sub = kme[kme["module"] == mod].sort_values(f"k{me}", ascending=False)
        top = ", ".join(sub.index[:8])
        e = enr[(enr["module"] == mod) & (enr["overlap"] > 0)].sort_values("p_hypergeom")
        best = f"{e.iloc[0]['program']} ({int(e.iloc[0]['overlap'])} genes, p = {e.iloc[0]['p_hypergeom']:.1e})" if len(e) else "none"
        pemt = enr[(enr["module"] == mod) & (enr["program"] == "puram_pemt")].iloc[0]
        ann.append({"module": mod, "size": int((assign["module"] == mod).sum()),
                    "r_P_pEMT_high": round(cor.loc[me, "P_pEMT_high"], 3),
                    "r_P_epithelial_like": round(cor.loc[me, "P_epithelial_like"], 3),
                    "r_P_fibroblast_stromal_like": round(cor.loc[me, "P_fibroblast_stromal_like"], 3),
                    "r_pEMT_specificity": round(cor.loc[me, "pEMT_specificity"], 3),
                    "r_hallmark_EMT": round(cor.loc[me, "hallmark_EMT"], 3) if "hallmark_EMT" in cor.columns else np.nan,
                    "puram_pemt_overlap": int(pemt["overlap"]), "puram_pemt_p": pemt["p_hypergeom"],
                    "best_program": best, "top_hubs_by_kME": top})
    ann = pd.DataFrame(ann)
    ann.to_csv(WG / "module_annotation.tsv", sep="\t", index=False)

    rank = ann[ann["r_P_pEMT_high"] > 0].sort_values(["puram_pemt_p", "r_P_pEMT_high"], ascending=[True, False])
    rank.to_csv(WG / "module_pemt_ranking.tsv", sep="\t", index=False)
    print(ann[["module", "size", "r_P_pEMT_high", "r_P_fibroblast_stromal_like", "r_P_epithelial_like", "puram_pemt_overlap", "puram_pemt_p", "best_program"]].to_string(index=False))
    if len(rank):
        b = rank.iloc[0]
        print(f"\npEMT module candidate: {b['module']} (size {b['size']}, {b['puram_pemt_overlap']} Puram pEMT genes, "
              f"p = {b['puram_pemt_p']:.1e}, r[P(pEMT-high)] = {b['r_P_pEMT_high']:+.2f}, r[stromal] = {b['r_P_fibroblast_stromal_like']:+.2f})")

    # Supplementary_Figure_3: module x trait heatmap with module annotation
    order = ann.sort_values("r_P_pEMT_high", ascending=False)["module"].tolist()
    M = cor.loc[[f"ME_{m}" for m in order], trait_cols].values.astype(float)
    fig, ax = plt.subplots(figsize=(mm(150), mm(6 * len(order) + 30)))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(trait_cols))); ax.set_xticklabels([TRAIT_LABELS[t] for t in trait_cols], rotation=45, ha="right")
    PROGRAM_SHORT = {"puram_epi_dif_2": "basal keratinocyte", "puram_epi_dif_1": "epithelial differentiation", "hallmark_EMT": "fibroblast / ECM",
                     "puram_cell_cycle": "cell cycle", "immune": "immune", "puram_hypoxia": "hypoxia", "puram_stress": "stress", "stromal": "stromal", "endothelial": "endothelial"}
    def short_annotation(m):
        e = enr[(enr["module"] == m) & (enr["overlap"] > 0)].sort_values("p_hypergeom")
        if len(e) and e.iloc[0]["p_hypergeom"] < 1e-3:
            return PROGRAM_SHORT.get(e.iloc[0]["program"], e.iloc[0]["program"])
        return ""
    ylab = [f"{m}  {short_annotation(m)}" for m in order]
    ax.set_yticks(range(len(order))); ax.set_yticklabels(ylab)
    for i, m in enumerate(order):
        ax.text(len(trait_cols) - 0.4, i, f"{int(ann.set_index('module').loc[m, 'size']):,} genes", va="center", ha="left", fontsize=5.5, color=PALETTE["grey"], clip_on=False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.5, color="white" if abs(M[i, j]) > 0.55 else "black")
    ax.spines["top"].set_visible(True); ax.spines["right"].set_visible(True)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.12); cb.set_label("Pearson correlation with module eigengene")
    save_figure(fig, FIG, "Supplementary_Figure_3")


if __name__ == "__main__":
    main()
