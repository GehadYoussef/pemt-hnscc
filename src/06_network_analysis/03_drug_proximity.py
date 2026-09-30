"""Compute network proximity of drug targets to each seed set (Guney et al. 2016).

The disease set is chosen by network_analysis.disease_set (seeds in
config.yaml) or by --disease seeds|key_proteins. With seeds, it is the seed
genes of each set in results/wgcna/network_seeds.tsv that are present in the
interactome, which is the design of Guney et al. With key_proteins, it is the
betweenness-selected key proteins of the STRING shortest-path subnetwork. That
set is hub-driven and depends on the STRING evidence channel (3 of 48 and 15
of 116 key proteins are shared between all-evidence and physical-evidence
runs), and is kept for comparison. Its outputs carry no suffix, and the
seed-based outputs carry "_seeds". Column names use "key_proteins" for the
disease set in both modes.

Proximity is the closest measure: the mean, over a drug's targets in the
interactome, of the shortest-path distance to the nearest disease-set gene.
The null uses network_analysis.permutation_n (1,000) draws of degree-matched
random disease sets and target sets. Nodes are binned by degree into bins of
at least network_analysis.proximity_min_bin_size (100) nodes, and the seed is
network_analysis.proximity_seed. Distances from each random disease set are
computed once and reused for every drug, and target draws are vectorised.
Each drug's target genes are sorted before the random targets are drawn, so
results do not depend on Python's hash seed. Two target definitions are run:
  stitch     STITCH-DrugBank association links, many per drug
             (for example, estradiol has 87 targets)
  drugbank   DrugBank curated human target bonds (bond_type = target) from the
             local DrugBank export (estradiol has 7)

The output reports z, the empirical p from the random draws (floor 1/1,001,
so it cannot support FDR control over thousands of drugs), a
normal-approximation p from z (the Guney et al. criterion) with
Benjamini-Hochberg FDR, and the number of targets at distance 0, 1 and 2. A
hit is z <= network_analysis.proximity_z_threshold (-1.96) and
normal-approximation FDR <= 0.05. Every drug is annotated from the DrugBank
export (name, type, status, ATC codes, categories, mechanism of action), and
entries without a DrugBank name are flagged by is_drug = False.

Anchors are the disease-set genes that are a drug's targets or direct
neighbours of its targets. For every drug, the observed distance is recomputed with each anchor removed from the
disease set, against the same null. The worst case is reported as
z_leave_one_anchor_out, with the anchor that causes it. The column robust_hit
is is_hit and z_leave_one_anchor_out <= the z threshold.

With --exclude-seed-targets, a drug's targets that are themselves seed genes
are removed before the distance is computed, and drugs whose targets are all
seed genes are dropped. These outputs carry a "_noseedtargets" suffix and feed Supplementary Table S49.

src/11_composition_and_mechanism/04_malignant_arm_proximity.py imports this
module and runs it on the malignant-arm seed set.

Inputs:  results/wgcna/network_seeds.tsv (seeds mode)
         results/network_analysis/<seed_name>/key_proteins.txt (key_proteins mode)
         data/references/human_interactome_gene_symbols.edgelist
         data/references/drug_targets_gene_symbols.csv (STITCH targets)
         data/raw/drug_targets/drugbank/drugbank_drug_target_flat_export.csv
         data/raw/drug_targets/drugbank/drugbank_drug_summary_export.csv
Outputs: results/network_analysis/<seed_name>/
         drug_proximity_<stitch|drugbank>[_seeds][_noseedtargets].tsv and the
         matching _significant.tsv (hits only)
Usage:   python src/06_network_analysis/03_drug_proximity.py
         [--disease seeds|key_proteins] [--exclude-seed-targets] [--only <seed_name>]
"""

import sys
import time
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pemt import load_config, project_root  # noqa: E402

cfg = load_config()
ROOT = project_root()
NA = cfg["network_analysis"]
NET = cfg["networks"]
OUT = ROOT / cfg["paths"]["results_dir"] / "network_analysis"
REF = ROOT / cfg["paths"]["references_dir"]
DB_DIR = ROOT / "data" / "raw" / "drug_targets" / "drugbank"
EXCLUDE_SEED_TARGETS = False
N_RANDOM = int(NA["permutation_n"])
MIN_BIN = int(NA["proximity_min_bin_size"])
SEED = int(NA["proximity_seed"])
Z_THR = float(NA["proximity_z_threshold"])


def load_interactome() -> ig.Graph:
    cache = REF / "human_interactome_gene_symbols.edgelist"
    edges = set()
    for line in open(cache):
        parts = line.strip().split(",")
        if len(parts) >= 2 and parts[0] != parts[1]:
            a, b = parts[0].strip(), parts[1].strip()
            edges.add((a, b) if a < b else (b, a))
    g = ig.Graph.TupleList(sorted(edges), directed=False)
    g.simplify()
    return g


def load_targets_stitch() -> dict[str, set[str]]:
    df = pd.read_csv(REF / "drug_targets_gene_symbols.csv")
    return df.groupby("DrugID")["Gene"].apply(set).to_dict()


def load_targets_drugbank() -> dict[str, set[str]]:
    df = pd.read_csv(DB_DIR / "drugbank_drug_target_flat_export.csv")
    df = df[(df["bond_type"] == "target") & (df["organism"] == "Humans") & df["gene_name"].notna()]
    return df.groupby("drugbank_id")["gene_name"].apply(set).to_dict()


def load_annotation() -> pd.DataFrame:
    ann = pd.read_csv(DB_DIR / "drugbank_drug_summary_export.csv")
    ann["atc_level1"] = ann["atc_codes"].fillna("").map(lambda s: ";".join(sorted({c.strip()[0] for c in s.split(",") if c.strip()})))
    return ann.set_index("drugbank_id")[["name", "type", "status_label", "approved", "withdrawn", "investigational",
                                          "atc_codes", "atc_level1", "atc_titles", "category_titles", "n_targets", "moa"]]


def degree_bins(g: ig.Graph, min_size: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return per-node bin id and the node arrays of each bin (Guney binning)."""
    deg = np.asarray(g.degree())
    order = np.argsort(deg, kind="stable")
    bins, cur, cur_deg = [], [], None
    values = sorted(set(deg))
    by_deg = {d: np.where(deg == d)[0] for d in values}
    i = 0
    while i < len(values):
        members = list(by_deg[values[i]])
        while len(members) < min_size and i + 1 < len(values):
            i += 1
            members += list(by_deg[values[i]])
        i += 1
        if bins and len(members) < min_size:
            bins[-1] = np.concatenate([bins[-1], np.array(members)])
        else:
            bins.append(np.array(members))
    node_bin = np.zeros(g.vcount(), dtype=int)
    for b, members in enumerate(bins):
        node_bin[members] = b
    return node_bin, bins


def sample_matched(rng, nodes: np.ndarray, node_bin: np.ndarray, bins: list[np.ndarray], n_draws: int) -> np.ndarray:
    """n_draws x len(nodes) matrix of degree-matched random node ids (no repeats within a draw)."""
    out = np.empty((n_draws, len(nodes)), dtype=int)
    for j, v in enumerate(nodes):
        cand = bins[node_bin[v]]
        out[:, j] = rng.choice(cand, size=n_draws, replace=True)
    # resolve within-draw duplicates by resampling those positions
    for i in range(n_draws):
        row = out[i]
        seen, dup = set(), []
        for j, v in enumerate(row):
            if v in seen:
                dup.append(j)
            seen.add(v)
        for j in dup:
            cand = bins[node_bin[nodes[j]]]
            for _ in range(50):
                v = rng.choice(cand)
                if v not in seen:
                    row[j] = v; seen.add(v); break
    return out


def min_dist_to_set(g: ig.Graph, sources: np.ndarray) -> np.ndarray:
    D = np.asarray(g.distances(source=list(sources)), dtype=float)   # |sources| x N
    return D.min(axis=0)


def main() -> None:
    # disease set from network_analysis.disease_set, overridden by --disease:
    #   seeds         the seed genes themselves (the module), the design of Guney et al.
    #   key_proteins  the key proteins of the shortest-path subnetwork
    disease_mode = str(NA.get("disease_set", "seeds"))
    global EXCLUDE_SEED_TARGETS
    EXCLUDE_SEED_TARGETS = "--exclude-seed-targets" in sys.argv
    if "--disease" in sys.argv:
        disease_mode = sys.argv[sys.argv.index("--disease") + 1]
    assert disease_mode in ("seeds", "key_proteins"), disease_mode
    suffix = "_seeds" if disease_mode == "seeds" else ""
    rng = np.random.default_rng(SEED)
    g = load_interactome()
    names = g.vs["name"]
    idx = {n: i for i, n in enumerate(names)}
    print(f"Interactome: {g.vcount():,} nodes, {g.ecount():,} edges")
    node_bin, bins = degree_bins(g, MIN_BIN)
    ann = load_annotation()
    target_sets = {"stitch": load_targets_stitch(), "drugbank": load_targets_drugbank()}

    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    for seed_dir in sorted(p for p in OUT.iterdir() if p.is_dir()):
        if only and seed_dir.name != only:
            continue
        if disease_mode == "seeds":
            all_seeds = pd.read_csv(ROOT / cfg["paths"]["results_dir"] / "wgcna" / "network_seeds.tsv", sep="\t")
            key = list(all_seeds.loc[all_seeds["seed_name"] == seed_dir.name, "gene"])
        else:
            key = [l.strip() for l in (seed_dir / "key_proteins.txt").read_text().splitlines() if l.strip()]
        disease = np.array([idx[k] for k in key if k in idx])
        print(f"\n[{seed_dir.name}] disease set = {disease_mode}: {len(key)}, in interactome {len(disease)}")
        if len(disease) < 3:
            print("  too few key proteins in the interactome, skipping")
            continue
        d_obs_all = min_dist_to_set(g, disease)                   # N
        key_set = set(disease.tolist())
        neigh = [set(g.neighbors(v)) for v in range(g.vcount())]
        D_key = np.asarray(g.distances(source=list(disease)), dtype=float)   # |key| x N, for leave-one-out
        key_pos = {int(v): i for i, v in enumerate(disease)}
        t0 = time.time()
        rand_disease = sample_matched(rng, disease, node_bin, bins, N_RANDOM)
        D_rand = np.vstack([min_dist_to_set(g, rand_disease[i]) for i in range(N_RANDOM)])   # N_RANDOM x N
        D_rand[~np.isfinite(D_rand)] = np.nan
        print(f"  random disease-set distances done in {(time.time() - t0) / 60:.1f} min")

        for tname, tsets in target_sets.items():
            rows = []
            for drug, genes in tsets.items():
                t = np.array([idx[x] for x in sorted(genes) if x in idx])    # sorted so the draws do not depend on the hash seed
                if EXCLUDE_SEED_TARGETS:
                    # Strict variant. Targets that are seed genes sit at distance 0 by definition and are
                    # removed. Drugs left with no targets are skipped below.
                    t =np.array([v for v in t if int(v) not in key_set], dtype=int)
                if len(t) == 0:
                    continue
                d_t = d_obs_all[t]
                d_t = d_t[np.isfinite(d_t)]
                if len(d_t) == 0:
                    continue
                d_obs = float(d_t.mean())
                rt = sample_matched(rng, t, node_bin, bins, N_RANDOM)
                d_rand = np.nanmean(D_rand[np.arange(N_RANDOM)[:, None], rt], axis=1)
                d_rand = d_rand[np.isfinite(d_rand)]
                mu, sd = float(d_rand.mean()), float(d_rand.std())
                z = (d_obs - mu) / sd if sd > 0 else 0.0
                p_emp = (1 + int((d_rand <= d_obs).sum())) / (len(d_rand) + 1)
                dist_prof = {int(k): int((d_obs_all[t] == k).sum()) for k in (0, 1, 2)}
                at0 = [names[i] for i in t if d_obs_all[i] == 0]
                at1 = [names[i] for i in t if d_obs_all[i] == 1]
                anchors = set()
                for i in t:
                    if i in key_set:
                        anchors.add(i)
                    anchors |= (neigh[i] & key_set)
                # leave-one-anchor-out: remove each anchor from the disease set, recompute the mean distance with the same null
                z_loo, worst = z, ""
                for a in anchors:
                    keep = [key_pos[v] for v in disease if v != a]
                    if not keep:
                        continue
                    d_t2 = D_key[np.ix_(keep, t)].min(axis=0)
                    d_t2 = d_t2[np.isfinite(d_t2)]
                    if len(d_t2) == 0:
                        continue
                    z2 = (float(d_t2.mean()) - mu) / sd if sd > 0 else 0.0
                    if z2 > z_loo:
                        z_loo, worst = z2, names[a]
                anchors = sorted(names[i] for i in anchors)
                rows.append({"drug_id": drug, "n_targets_in_interactome": len(t), "n_key_proteins": len(disease),
                             "d_observed": d_obs, "mu_random": mu, "sigma_random": sd, "z": z, "p_empirical": p_emp,
                             "targets_at_distance_0": dist_prof[0], "targets_at_distance_1": dist_prof[1], "targets_at_distance_2": dist_prof[2],
                             "targets_that_are_key_proteins": ";".join(at0), "targets_adjacent_to_key_proteins": ";".join(at1),
                             "n_anchor_key_proteins": len(anchors), "anchor_key_proteins": ";".join(anchors),
                             "z_leave_one_anchor_out": z_loo, "anchor_whose_removal_hurts_most": worst})
            res = pd.DataFrame(rows)
            res["fdr_empirical"] = multipletests(res["p_empirical"], method="fdr_bh")[1]
            res["p_normal"] = norm.cdf(res["z"])
            res["fdr_normal"] = multipletests(res["p_normal"], method="fdr_bh")[1]
            res["significant_z"] = res["z"] <= Z_THR
            res["is_hit"] = res["significant_z"] & (res["fdr_normal"] <= 0.05)
            res["robust_hit"] = res["is_hit"] & (res["z_leave_one_anchor_out"] <= Z_THR)
            res = res.join(ann, on="drug_id")
            res["is_drug"] = res["name"].notna()
            res = res.sort_values("z")
            tag = "_noseedtargets" if EXCLUDE_SEED_TARGETS else ""
            res.to_csv(seed_dir / f"drug_proximity_{tname}{suffix}{tag}.tsv", sep="\t", index=False)
            sig = res[res["is_hit"]]
            sig.to_csv(seed_dir / f"drug_proximity_{tname}{suffix}{tag}_significant.tsv", sep="\t", index=False)
            appr = sig[sig["approved"] == 1] if "approved" in sig else sig
            print(f"  [{tname}] {len(res)} drugs, z <= {Z_THR}: {int(res['significant_z'].sum())}, hits (z & normal FDR <= 0.05): {len(sig)}, "
                  f"approved among those: {len(appr)}, seed-stable (still hits after removing any one anchor): {int(sig['robust_hit'].sum())} ({int(appr['robust_hit'].sum())} approved), top: "
                  + ", ".join(f"{r['name']} ({r['z']:.1f})" for _, r in sig.head(8).iterrows() if isinstance(r["name"], str)))


if __name__ == "__main__":
    main()
