"""Draw the standard WGCNA module-trait correlation heatmap.

Rows are module eigengenes, labelled with module size. Columns are sample
traits. Cell colour is the Pearson correlation r. Cell text is r, with the
p-value in brackets on the line below. Modules are ordered by decreasing
absolute correlation with the primary trait (config wgcna.primary_trait).

Inputs:  results/wgcna/module_trait_correlations.tsv, module_trait_pvalues.tsv,
         gene_module_assignments.tsv
Outputs: results/figures/module_trait_heatmap.svg (600 dpi)
Usage:   python src/05_wgcna/05_plot_module_trait_heatmap.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pemt import load_config, project_root  # noqa: E402

cfg = load_config()
ROOT = project_root()
WGCNA_DIR = ROOT / cfg["paths"]["results_dir"] / "wgcna"
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    cors = pd.read_csv(WGCNA_DIR / "module_trait_correlations.tsv", sep="\t", index_col=0)
    pvals = pd.read_csv(WGCNA_DIR / "module_trait_pvalues.tsv", sep="\t", index_col=0)
    assignments = pd.read_csv(WGCNA_DIR / "gene_module_assignments.tsv", sep="\t")

    primary = cfg["wgcna"]["primary_trait"]
    order = cors[primary].abs().sort_values(ascending=False).index
    cors = cors.loc[order]
    pvals = pvals.loc[order]

    sizes = assignments["module"].value_counts().to_dict()
    row_labels = [f"{me.replace('ME_', '')} (n={sizes.get(me.replace('ME_', ''), 0)})"
                  for me in cors.index]

    nice_col = {
        "pEMT_specificity": "pEMT_specificity",
        "P_pEMT_high": "P(pEMT)",
        "P_epithelial_like": "P(epithelial)",
        "P_fibroblast_stromal_like": "P(fibroblast/stromal)",
        "pEMT_vs_stroma": "pEMT − stroma",
        "pEMT_vs_epithelial": "pEMT − epithelial",
    }
    col_labels = [nice_col.get(c, c) for c in cors.columns]

    n_rows, n_cols = cors.shape
    fig_w = max(7.0, 1.6 * n_cols + 1.6)
    fig_h = max(4.0, 0.55 * n_rows + 1.2)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=600)

    cmap = mpl.cm.RdBu_r
    norm = mpl.colors.Normalize(vmin=-1.0, vmax=1.0)
    mat = cors.values
    img = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")

    for i in range(n_rows):
        for j in range(n_cols):
            r = mat[i, j]
            p = pvals.values[i, j]
            txt = f"{r:.2f}\n({p:.0e})"
            colour = "white" if abs(r) > 0.4 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    color=colour, fontsize=8)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xlabel("Trait", fontsize=11)
    ax.set_ylabel("Module (size)", fontsize=11)
    ax.set_title("WGCNA module–trait correlations (TCGA-HNSC)",
                 fontsize=12)

    cbar = fig.colorbar(img, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Pearson r", fontsize=10)

    plt.tight_layout()
    out_path = FIG_DIR / "module_trait_heatmap.svg"
    fig.savefig(out_path, format="svg", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
