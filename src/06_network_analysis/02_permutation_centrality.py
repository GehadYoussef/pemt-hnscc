"""Identify key proteins by a per-node, degree-preserving permutation test.

The subnetwork is rewired network_analysis.permutation_n times (1,000) by
degree-preserving edge swaps (igraph rewire, 10 x |E| swaps per replicate,
seed network_analysis.permutation_seed). A node's observed betweenness,
closeness, eigenvector centrality and PageRank are compared with its own
values across the rewired graphs. Degree is preserved by construction and is
not tested.

Two statistics are reported per node and metric. The first is the empirical
p = (1 + #null >= observed) / (N + 1), with a Benjamini-Hochberg FDR. Its
floor of 1/(N + 1) is too coarse for FDR control on its own with N = 1,000.
The second is a z-score against the node's own null, with a
one-sided normal-approximation p and Benjamini-Hochberg FDR across nodes.
Betweenness is scored on the log1p scale because its null is right-skewed.
The other metrics are scored on the raw scale.

A key protein has normal-approximation FDR <= network_analysis.key_protein_fdr
(0.05) and empirical p <= 0.01 for at least one centrality listed in
network_analysis.key_metrics (betweenness only in config.yaml). Eigenvector
centrality and PageRank are reported but not used for the call. Both are
dominated by degree, which rewiring holds fixed, so their per-node null
variance is close to zero and their z-scores are inflated (|z| > 20 is
common).

Inputs:  results/network_analysis/<seed_name>/subnetwork_edges.tsv,
         centrality_results.tsv
Outputs: results/network_analysis/<seed_name>/centrality_pvalue_results.tsv,
         key_proteins.tsv, key_proteins.txt, permutation_settings.txt
Usage:   python src/06_network_analysis/02_permutation_centrality.py [--select-only]
         --select-only re-applies the key-protein criterion to an existing
         centrality_pvalue_results.tsv without repeating the permutations.
"""

import random
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
OUT = ROOT / cfg["paths"]["results_dir"] / "network_analysis"
N_PERM = int(NA["permutation_n"])
SEED = int(NA.get("permutation_seed", 20260909))
FDR_THR = float(NA.get("key_protein_fdr", 0.05))
METRICS = ["betweenness", "closeness", "eigenvector", "pagerank"]
KEY_METRICS = list(NA.get("key_metrics", ["betweenness"]))


def centralities(g: ig.Graph) -> dict[str, np.ndarray]:
    return {
        "betweenness": np.asarray(g.betweenness(), dtype=float),
        "closeness": np.nan_to_num(np.asarray(g.closeness(), dtype=float)),
        "eigenvector": np.asarray(g.eigenvector_centrality(scale=True), dtype=float),
        "pagerank": np.asarray(g.pagerank(), dtype=float),
    }


def select_key(res: pd.DataFrame) -> pd.DataFrame:
    sig = {k: (res[f"{k}_fdr_normal"] <= FDR_THR) & (res[f"{k}_p"] <= 0.01) for k in KEY_METRICS}
    res = res.copy()
    res["is_key_protein"] = np.column_stack([sig[k].values for k in KEY_METRICS]).any(axis=1)
    res["key_by"] = ["; ".join(k for k in KEY_METRICS if sig[k].iloc[i]) for i in range(len(res))]
    res["max_z"] = res[[f"{k}_z" for k in KEY_METRICS]].max(axis=1)
    return res.sort_values(["is_key_protein", "max_z"], ascending=[False, False])


def write_key(seed_dir: Path, res: pd.DataFrame, n: int, m: int) -> pd.DataFrame:
    res.to_csv(seed_dir / "centrality_pvalue_results.tsv", sep="\t")
    key = res[res["is_key_protein"]]
    key.to_csv(seed_dir / "key_proteins.tsv", sep="\t")
    (seed_dir / "key_proteins.txt").write_text("\n".join(key.index) + "\n", encoding="utf-8")
    (seed_dir / "permutation_settings.txt").write_text(
        f"nodes\t{n}\nedges\t{m}\npermutations\t{N_PERM}\nswaps_per_permutation\t{10 * m}\nseed\t{SEED}\n"
        f"fdr_threshold\t{FDR_THR}\nkey_criterion\tnormal-approximation FDR <= threshold and empirical p <= 0.01\nkey_metrics\t{','.join(KEY_METRICS)}\nkey_proteins\t{len(key)}\n"
        f"key_proteins_that_are_seed_genes\t{int(key['is_seed'].sum())}\n", encoding="utf-8")
    print(f"[{seed_dir.name}] key proteins: {len(key)} of {n} ({int(key['is_seed'].sum())} are seed genes), "
          f"top by betweenness z: {', '.join(key.sort_values('betweenness_z', ascending=False).index[:15])}")
    return key


def main() -> None:
    if "--select-only" in sys.argv:
        for seed_dir in sorted(p for p in OUT.iterdir() if p.is_dir()):
            f = seed_dir / "centrality_pvalue_results.tsv"
            if not f.exists():
                continue
            res = pd.read_csv(f, sep="\t", index_col=0)
            edges = pd.read_csv(seed_dir / "subnetwork_edges.tsv", sep="\t")
            write_key(seed_dir, select_key(res), len(res), len(edges))
        return
    random.seed(SEED)
    ig.set_random_number_generator(random)
    for seed_dir in sorted(p for p in OUT.iterdir() if p.is_dir()):
        edges = pd.read_csv(seed_dir / "subnetwork_edges.tsv", sep="\t")
        g = ig.Graph.TupleList(edges.itertuples(index=False, name=None), directed=False)
        g.simplify()
        names = g.vs["name"]
        n, m = g.vcount(), g.ecount()
        obs = centralities(g)
        counts = {k: np.zeros(n) for k in METRICS}
        sums = {k: np.zeros(n) for k in METRICS}
        sumsq = {k: np.zeros(n) for k in METRICS}
        t0 = time.time()
        for i in range(N_PERM):
            r = g.copy()
            r.rewire(n=10 * m, allowed_edge_types="simple")
            null = centralities(r)
            for k in METRICS:
                counts[k] += (null[k] >= obs[k])
                v = np.log1p(null[k]) if k == "betweenness" else null[k]
                sums[k] += v
                sumsq[k] += v ** 2
            if (i + 1) % 100 == 0:
                print(f"[{seed_dir.name}] {i + 1}/{N_PERM} permutations, {(time.time() - t0) / 60:.1f} min")
        res = pd.DataFrame({"gene": names, "degree": g.degree()})
        for k in METRICS:
            res[k] = obs[k]
            res[f"{k}_p"] = (1 + counts[k]) / (N_PERM + 1)
            res[f"{k}_fdr"] = multipletests(res[f"{k}_p"], method="fdr_bh")[1]
            mu = sums[k] / N_PERM
            sd = np.sqrt(np.maximum(sumsq[k] / N_PERM - mu ** 2, 0))
            o = np.log1p(obs[k]) if k == "betweenness" else obs[k]
            res[f"{k}_null_mean"] = mu
            res[f"{k}_null_sd"] = sd
            z = np.where(sd > 0, (o - mu) / np.where(sd > 0, sd, 1), 0.0)
            res[f"{k}_z"] = z
            res[f"{k}_p_normal"] = norm.sf(z)
            res[f"{k}_fdr_normal"] = multipletests(res[f"{k}_p_normal"], method="fdr_bh")[1]
        cent = pd.read_csv(seed_dir / "centrality_results.tsv", sep="\t", index_col=0)
        res["is_seed"] = res["gene"].map(cent["is_seed"]).fillna(False).astype(bool)
        write_key(seed_dir, select_key(res.set_index("gene")), n, m)


if __name__ == "__main__":
    main()
