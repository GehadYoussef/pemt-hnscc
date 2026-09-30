"""Draw the drug-proximity figures for the three seed sets.

Drug proximity is computed from the seed genes themselves (Guney et al. 2016),
with DrugBank curated targets unless stated.

  Supplementary_Figure_16  a  canonical pEMT module (M06): the approved drugs
                              closest to the seed genes, ranked by proximity z.
                              Colour shows hit status, point size the number of
                              anchor seed genes, and the line the z after
                              leaving out the most influential anchor.
                           b  pEMT-axis module (M13): the same panel
                           c  classifier pEMT-high coefficient genes: the same panel
                           d  the seed gene each approved classifier-set hit
                              depends on, from the leave-one-anchor-out scan
  Supplementary_Figure_17  agreement of proximity z between STITCH and DrugBank
                           targets for the three seed sets (Spearman rho, with
                           approved hits under both definitions labelled)

With --with-subnetwork, the STRING physical subnetwork of each seed set that
has key_proteins.tsv is also drawn (descriptive only). Seed genes, key
proteins and seed genes that are key proteins are coloured separately. The
file stems are panel_subnetwork_pEMT_axis, panel_subnetwork_canonical_pEMT and
panel_network_<seed_name>_subnetwork for other seed sets. key_protein_figure()
draws key-protein centrality z-scores and is not called by main().

Inputs:  results/network_analysis/<seed_name>/drug_proximity_drugbank_seeds.tsv,
         drug_proximity_stitch_seeds.tsv
         with --with-subnetwork: subnetwork_edges.tsv, key_proteins.tsv,
         seed_genes_in_string.txt
Outputs: results/figures/Supplementary_Figure_16, Supplementary_Figure_17
         (.svg, .png), and the subnetwork panels with --with-subnetwork
Usage:   python src/06_network_analysis/04_network_figures.py [--with-subnetwork]
"""

import sys
from pathlib import Path

import random

import igraph as ig
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pemt import load_config, project_root  # noqa: E402
from pemt.figstyle import apply_style, save_figure, PALETTE, mm  # noqa: E402

cfg = load_config()
ROOT = project_root()
OUT = ROOT / cfg["paths"]["results_dir"] / "network_analysis"
FIG = ROOT / "results" / "figures"
Z_THR = float(cfg["network_analysis"]["proximity_z_threshold"])
apply_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SEED_TITLE = {"pEMT_axis": "pEMT-axis module (M13 hubs)", "canonical_pEMT": "canonical pEMT genes (M06)"}
# file stems for the subnetwork panels. Other seed sets fall back to panel_network_<seed_name>_<kind>.
FIG_STEM = {"pEMT_axis": {"subnetwork": "panel_subnetwork_pEMT_axis"}, "canonical_pEMT": {"subnetwork": "panel_subnetwork_canonical_pEMT"}}


def stem(seed_dir: Path, kind: str) -> str:
    return FIG_STEM.get(seed_dir.name, {}).get(kind, f"panel_network_{seed_dir.name}_{kind}")


def _scaled(g_sub: ig.Graph) -> np.ndarray:
    lay = g_sub.layout_kamada_kawai(maxiter=50 * g_sub.vcount()) if g_sub.vcount() <= 400 else g_sub.layout_fruchterman_reingold(niter=800)
    xy = np.array(lay.coords, dtype=float)
    if len(xy) == 1:
        return np.zeros((1, 2))
    xy -= xy.min(axis=0)
    span = xy.max(axis=0)
    span[span == 0] = 1
    return xy / span.max()


def packed_layout(g: ig.Graph) -> np.ndarray:
    """Lay out the giant component in the unit square and stack smaller components in a column on the left."""
    comps = sorted(g.connected_components(), key=len, reverse=True)
    xy = np.zeros((g.vcount(), 2))
    giant = comps[0]
    xy[giant] = _scaled(g.subgraph(giant))
    y = 0.72
    for comp in comps[1:]:
        sub = _scaled(g.subgraph(comp))
        h = 0.06 + 0.04 * np.sqrt(len(comp))
        sub = sub * h
        xy[comp] = sub + np.array([-0.32, y - h])
        y -= h + 0.05
        if y < 0:
            y = 0.72
    return xy


def subnetwork_figure(seed_dir: Path):
    edges = pd.read_csv(seed_dir / "subnetwork_edges.tsv", sep="\t")
    key = pd.read_csv(seed_dir / "key_proteins.tsv", sep="\t", index_col=0)
    seeds = {l.strip() for l in (seed_dir / "seed_genes_in_string.txt").read_text().splitlines() if l.strip()}
    keep = set(key.index) | seeds
    sub = edges[edges["gene_a"].isin(keep) & edges["gene_b"].isin(keep)]
    g = ig.Graph.TupleList(sub.itertuples(index=False, name=None), directed=False)
    g.simplify()
    isolated = sorted(keep - set(g.vs["name"]))
    if g.vcount() == 0:
        return
    random.seed(20260909); ig.set_random_number_generator(random)
    xy = packed_layout(g)
    names = g.vs["name"]
    deg = np.array(g.degree())
    fig, ax = plt.subplots(figsize=(mm(130), mm(120)))
    for e in g.es:
        a, b = e.tuple
        ax.plot([xy[a, 0], xy[b, 0]], [xy[a, 1], xy[b, 1]], color=PALETTE["lightgrey"], linewidth=0.25, alpha=0.8, zorder=1)
    colours = [PALETTE["vermilion"] if n in seeds and n in key.index else PALETTE["orange"] if n in seeds else PALETTE["blue"] for n in names]
    ax.scatter(xy[:, 0], xy[:, 1], s=6 + 2.5 * np.sqrt(deg), c=colours, linewidths=0.3, edgecolors="white", zorder=2)
    # label: seed genes that are key proteins, the 20 highest-betweenness key proteins, and (small networks) all seeds
    n_label = 60 if g.vcount() <= 150 else 30
    top = set(key.sort_values("betweenness", ascending=False).index[:20]) | (seeds & set(key.index))
    if g.vcount() <= 150:
        top |= seeds
    top = sorted(top, key=lambda n: -key["betweenness"].get(n, 0))[:n_label]
    texts = [ax.text(xy[i, 0], xy[i, 1], n, fontsize=5, zorder=3) for i, n in enumerate(names) if n in top]
    from adjustText import adjust_text
    adjust_text(texts, x=xy[:, 0], y=xy[:, 1], ax=ax, expand=(1.3, 1.6), force_text=(0.4, 0.8), force_static=(0.3, 0.5),
                arrowprops={"arrowstyle": "-", "color": PALETTE["grey"], "lw": 0.3}, min_arrow_len=4)
    ax.set_axis_off()
    handles = [Line2D([], [], marker="o", linestyle="", color=PALETTE["orange"], label="Seed gene"),
               Line2D([], [], marker="o", linestyle="", color=PALETTE["vermilion"], label="Seed gene and key protein"),
               Line2D([], [], marker="o", linestyle="", color=PALETTE["blue"], label="Key protein (connector)")]
    ax.legend(handles=handles, loc="upper left", fontsize=6)
    ax.text(0.99, 0.01, f"{g.vcount()} nodes, {g.ecount()} edges" + (f", {len(isolated)} without edges not shown" if isolated else ""),
            transform=ax.transAxes, ha="right", fontsize=6, color=PALETTE["grey"])
    save_figure(fig, FIG, stem(seed_dir, "subnetwork"))


def key_protein_figure(seed_dir: Path):
    """Plot key-protein centrality z-scores against the degree-preserving null.

    Rows are sorted by betweenness z, which is the key-protein criterion. Eigenvector and PageRank
    are shown for context only. Their per-node null variance is near zero once degree is fixed, so
    their z-scores are inflated.
    """
    key = pd.read_csv(seed_dir / "key_proteins.tsv", sep="\t", index_col=0)
    if key.empty or "betweenness_z" not in key.columns:
        return
    metrics = [("betweenness", "Betweenness"), ("eigenvector", "Eigenvector"), ("pagerank", "PageRank")]
    n_show = 40
    k = key.sort_values("betweenness_z", ascending=False).head(n_show).iloc[::-1]
    fig, ax = plt.subplots(figsize=(mm(95), mm(4.2 * len(k) + 20)))
    y = np.arange(len(k))
    zmax = 8.0
    cmap = plt.get_cmap("RdBu_r")
    for j, (m, _) in enumerate(metrics):
        z = k[f"{m}_z"].values
        col = [cmap(0.5 + 0.5 * np.clip(v / zmax, -1, 1)) for v in z]
        if m == "betweenness":
            ax.scatter(np.full(len(k), j), y, s=60, c=col, edgecolors=PALETTE["black"], linewidths=0.9, zorder=2)
            for yi, v in zip(y, z):
                ax.text(j + 0.32, yi, f"{v:.1f}", va="center", fontsize=5, color=PALETTE["grey"])
        else:
            ax.scatter(np.full(len(k), j + 0.5), y, s=60, c=col, edgecolors=PALETTE["lightgrey"], linewidths=0.4, zorder=2)
    ax.set_xlim(-0.5, 3.6)
    ax.set_xticks([0, 1.5, 2.5]); ax.set_xticklabels([lab for _, lab in metrics], fontsize=7)
    ax.set_yticks(y); ax.set_yticklabels([f"{g}*" if s_ else g for g, s_ in zip(k.index, k["is_seed"])], fontsize=6)
    for i, (_, r) in enumerate(k.iterrows()):
        ax.text(2.85, i, f"degree {int(r['degree'])}", va="center", fontsize=5.5, color=PALETTE["grey"])
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-zmax, zmax))
    cb = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.02, aspect=30, extend="both")
    cb.set_label("z-score vs degree-preserving null (1,000 rewirings)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    more = f", {len(key) - len(k)} further key proteins not shown" if len(key) > len(k) else ""
    ax.text(0, 1.01, f"key protein = betweenness FDR ≤ 0.05 and empirical p ≤ 0.01 (n = {len(key)}{more})\n"
                     "* seed gene. Eigenvector and PageRank are shown for context and not used for the call.",
            transform=ax.transAxes, fontsize=6, color=PALETTE["grey"], va="bottom")
    save_figure(fig, FIG, stem(seed_dir, "key"))


ATC_NAMES = {"A": "Alimentary/metabolism", "B": "Blood", "C": "Cardiovascular", "D": "Dermatological", "G": "Genito-urinary/sex hormones",
             "H": "Systemic hormones", "J": "Anti-infective", "L": "Antineoplastic/immunomodulating", "M": "Musculoskeletal",
             "N": "Nervous system", "P": "Antiparasitic", "R": "Respiratory", "S": "Sensory organs", "V": "Various"}


def module_proximity_panel(ax, seed_dir: Path, title: str, n_show: int = 20):
    """Plot approved drugs ranked by proximity z to the seed genes (DrugBank curated targets)."""
    f = seed_dir / "drug_proximity_drugbank_seeds.tsv"
    if not f.exists():
        return None
    db = pd.read_csv(f, sep="\t")
    appr = db[(db["approved"] == 1) & db["is_drug"]].sort_values("z").head(n_show)
    y = np.arange(len(appr))[::-1]
    for yi, (_, r) in zip(y, appr.iterrows()):
        robust = bool(r.get("robust_hit", False)); hit = bool(r["is_hit"])
        colour = PALETTE["vermilion"] if robust else PALETTE["orange"] if hit else PALETTE["lightgrey"]
        ax.plot([r["z"], 0], [yi, yi], color=PALETTE["lightgrey"], linewidth=0.8, zorder=1)
        ax.scatter(r["z"], yi, s=16 + 2 * min(int(r["n_anchor_key_proteins"]), 20), color=colour, zorder=2, linewidths=0)
        if hit:
            ax.plot([r["z_leave_one_anchor_out"], r["z"]], [yi, yi], color=colour, linewidth=1.4, zorder=1, alpha=0.6)
    ax.set_yticks(y); ax.set_yticklabels(appr["name"], fontsize=6.5)
    ax.axvline(Z_THR, ls="--", color=PALETTE["grey"], linewidth=0.6)
    ax.axvline(0, color=PALETTE["lightgrey"], linewidth=0.6)
    ax.set_xlim(min(-8, float(appr["z"].min()) - 0.5), 1.5)
    ax.set_xlabel("Proximity z-score, drug targets to seed genes")
    n_hit = int((db["is_hit"] & (db["approved"] == 1)).sum()); n_rob = int((db.get("robust_hit", False) & (db["approved"] == 1)).sum()) if "robust_hit" in db else 0
    ax.set_title(f"{title}: {n_hit} approved hit{'s' if n_hit != 1 else ''}, {n_rob} seed-stable", fontsize=7, color=PALETTE["grey"])
    return appr


def anchor_dependence_panel(ax, seed_dir: Path, n_show: int = 10):
    """Plot the seed gene each approved hit depends on.

    For every approved proximity hit, the leave-one-anchor-out scan names the seed gene whose removal
    raises z the most. Bars count hits per seed gene, split by whether the hit survives that removal
    (robust_hit).
    """
    f = seed_dir / "drug_proximity_drugbank_seeds.tsv"
    if not f.exists():
        return
    db = pd.read_csv(f, sep="\t")
    hits = db[(db["approved"] == 1) & db["is_drug"] & db["is_hit"]]
    if hits.empty or "anchor_whose_removal_hurts_most" not in hits.columns:
        ax.axis("off")
        return
    counts = hits["anchor_whose_removal_hurts_most"].fillna("none").value_counts().head(n_show)
    y = np.arange(len(counts))[::-1]
    lost = np.array([int(((hits["anchor_whose_removal_hurts_most"] == g) & (~hits["robust_hit"].astype(bool))).sum()) for g in counts.index])
    robust = counts.values - lost
    ax.barh(y, lost, color=PALETTE["orange"], height=0.68, linewidth=0)
    ax.barh(y, robust, left=lost, color=PALETTE["vermilion"], height=0.68, linewidth=0)
    ax.set_yticks(y)
    ax.set_yticklabels(counts.index, fontsize=6.5)
    ax.set_xlabel("Approved proximity hits")
    ax.set_xlim(0, max(counts.values) * 1.18)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("Seed gene each hit depends on", fontsize=7, color=PALETTE["grey"])


def modules_figure():
    panels = [("canonical_pEMT", "Canonical pEMT module (M06)"),
              ("pEMT_axis", "pEMT-axis module (M13)"),
              ("classifier_pEMT_high", "Classifier pEMT-high genes")]
    fig, axes = plt.subplots(2, 2, figsize=(mm(180), mm(165)))
    axes = axes.ravel()
    for ax, (name, title), letter in zip(axes, panels, "abc"):
        module_proximity_panel(ax, OUT / name, title, n_show=14)
        ax.text(-0.52, 1.05, letter, transform=ax.transAxes, fontweight="bold", fontsize=8)
    anchor_dependence_panel(axes[3], OUT / "classifier_pEMT_high")
    axes[3].text(-0.30, 1.05, "d", transform=axes[3].transAxes, fontweight="bold", fontsize=8)
    handles = [Line2D([], [], marker="o", linestyle="", color=PALETTE["vermilion"], label="hit, seed-stable (survives removing any one seed gene)"),
               Line2D([], [], marker="o", linestyle="", color=PALETTE["orange"], label="hit, lost when one seed gene is removed"),
               Line2D([], [], marker="o", linestyle="", color=PALETTE["lightgrey"], label="not a hit"),
               Line2D([], [], color=PALETTE["vermilion"], linewidth=1.4, alpha=0.6, label="z after removing the most influential seed")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=6, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    save_figure(fig, FIG, "Supplementary_Figure_16")


def agreement_figure():
    panels = [("canonical_pEMT", "Canonical pEMT module (M06)"), ("pEMT_axis", "pEMT-axis module (M13)"),
              ("classifier_pEMT_high", "Classifier pEMT-high genes")]
    fig, axes = plt.subplots(1, 3, figsize=(mm(180), mm(68)))
    for ax, (name, title), letter in zip(axes, panels, "abc"):
        fdb, fst = OUT / name / "drug_proximity_drugbank_seeds.tsv", OUT / name / "drug_proximity_stitch_seeds.tsv"
        if not (fdb.exists() and fst.exists()):
            continue
        db, st = pd.read_csv(fdb, sep="\t"), pd.read_csv(fst, sep="\t")
        m = db[["drug_id", "z", "name", "approved", "is_hit"]].merge(st[["drug_id", "z", "is_hit"]], on="drug_id", suffixes=("_drugbank", "_stitch"))
        ax.scatter(m["z_stitch"], m["z_drugbank"], s=5, color=PALETTE["grey"], alpha=0.45, linewidths=0, rasterized=True)
        both = m[m["is_hit_drugbank"] & m["is_hit_stitch"] & (m["approved"] == 1)]
        ax.scatter(both["z_stitch"], both["z_drugbank"], s=10, color=PALETTE["vermilion"], linewidths=0)
        texts = [ax.text(r.z_stitch, r.z_drugbank, str(r.name), fontsize=5.5) for r in both.sort_values("z_drugbank").head(8).itertuples()]
        if texts:
            from adjustText import adjust_text
            adjust_text(texts, ax=ax, arrowprops={"arrowstyle": "-", "color": PALETTE["grey"], "lw": 0.3}, expand=(1.3, 1.6))
        from scipy.stats import spearmanr
        rho, _ = spearmanr(m["z_stitch"], m["z_drugbank"])
        ax.axvline(Z_THR, ls="--", color=PALETTE["grey"], linewidth=0.6); ax.axhline(Z_THR, ls="--", color=PALETTE["grey"], linewidth=0.6)
        ax.set_xlabel("z, STITCH association targets"); ax.set_ylabel("z, DrugBank curated targets")
        ax.text(0.03, 0.97, f"ρ = {rho:.2f}, {len(m)} drugs\n{len(both)} approved hits with both", transform=ax.transAxes, va="top", fontsize=6.5)
        ax.set_title(title, fontsize=7, color=PALETTE["grey"])
        ax.text(-0.18, 1.06, letter, transform=ax.transAxes, fontweight="bold", fontsize=8)
    fig.tight_layout()
    save_figure(fig, FIG, "Supplementary_Figure_17")


def main() -> None:
    modules_figure()
    agreement_figure()
    for seed_dir in sorted(p for p in OUT.iterdir() if p.is_dir()):
        if (seed_dir / "key_proteins.tsv").exists() and "--with-subnetwork" in sys.argv:
            subnetwork_figure(seed_dir)
    print("figures written")


if __name__ == "__main__":
    main()
