"""Build STRING shortest-path subnetworks for each seed set (stage 05 output).

Patched 9 Sep 2026. Same construction as the original (union of all shortest
paths between every pair of seed genes on STRING v11 (evidence per networks.string_mode,
default physical edges with experimental or database support, score >= 400), in gene
space), now run for every seed set in results/wgcna/network_seeds.tsv and
implemented with igraph. Nodes are flagged as seed genes or connectors.

Centralities on each subnetwork: degree, betweenness, closeness, eigenvector,
PageRank, plus membrane-to-nucleus subset betweenness (COMPARTMENT annotations).

Outputs (results/network_analysis/<seed_name>/):
  subnetwork_edges.tsv, subnetwork_nodes.tsv, centrality_results.tsv,
  membrane_genes.txt, nucleus_genes.txt, seed_genes_in_string.txt
"""

import sys
from pathlib import Path

import igraph as ig
import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import load_config, project_root  # noqa: E402

cfg = load_config()
ROOT = project_root()
NET = cfg["networks"]
OUT = ROOT / cfg["paths"]["results_dir"] / "network_analysis"
WG = ROOT / cfg["paths"]["results_dir"] / "wgcna"


def load_string_gene_graph() -> ig.Graph:
    """STRING v11.0 in gene-symbol space.

    networks.string_mode selects the evidence:
      all                    the all-channel links file (score >= 400), as in the original pipeline
      physical               STRING physical subnetwork, combined score >= string_score_threshold
      physical_experimental  physical subnetwork, score >= threshold, and at least one of the
                             experiments / experiments_transferred / database / database_transferred
                             channels > 0 (drops edges supported by text-mining alone)
    The derived gene edge list is cached in data/references/string_<mode>_gene_edges.tsv.
    """
    import gzip
    mode = NET.get("string_mode", "all")
    thr = int(NET.get("string_score_threshold", 400))
    p2g = {}
    for line in open(ROOT / NET["protein_to_gene_file"]):
        parts = line.strip().split(",")
        if len(parts) == 2:
            p2g[parts[0].strip()] = parts[1].strip()
    cache = ROOT / cfg["paths"]["references_dir"] / f"string_{mode}_{thr}_gene_edges.tsv"
    edges = set()
    if cache.exists():
        for line in open(cache):
            a, b = line.rstrip("\n").split("\t")[:2]
            edges.add((a, b))
    elif mode == "all":
        for line in open(ROOT / NET["string_links_file"]):
            parts = line.strip().split(",")
            if len(parts) != 2:
                continue
            a, b = p2g.get(parts[0].strip()), p2g.get(parts[1].strip())
            if a and b and a != b:
                edges.add((a, b) if a < b else (b, a))
    else:
        n_rows = n_kept = 0
        with gzip.open(ROOT / NET["string_physical_full_file"], "rt") as fh:
            header = fh.readline().split()
            col = {h: i for i, h in enumerate(header)}
            for line in fh:
                parts = line.split()
                if int(parts[col["combined_score"]]) < thr:
                    continue
                n_rows += 1
                if mode == "physical_experimental":
                    ev = max(int(parts[col[c]]) for c in ("experiments", "experiments_transferred", "database", "database_transferred"))
                    if ev == 0:
                        continue
                a, b = p2g.get(parts[0].split(".", 1)[1]), p2g.get(parts[1].split(".", 1)[1])
                if a and b and a != b:
                    edges.add((a, b) if a < b else (b, a)); n_kept += 1
        print(f"STRING {mode}: {n_rows:,} protein pairs at score >= {thr}, {n_kept:,} mapped to gene symbols with the required evidence")
        cache.write_text("".join(f"{a}\t{b}\n" for a, b in sorted(edges)))
    g = ig.Graph.TupleList(sorted(edges), directed=False)
    g.simplify()
    return g


def shortest_path_subnetwork(g: ig.Graph, seeds: list[str]) -> set[tuple[str, str]]:
    idx = {v["name"]: v.index for v in g.vs}
    sid = [idx[s] for s in seeds]
    edges = set()
    for s in sid:
        for path in g.get_all_shortest_paths(s, to=sid):
            for a, b in zip(path[:-1], path[1:]):
                na, nb = g.vs[a]["name"], g.vs[b]["name"]
                edges.add((na, nb) if na < nb else (nb, na))
    return edges


def main() -> None:
    seeds = pd.read_csv(WG / "network_seeds.tsv", sep="\t")
    string_g = load_string_gene_graph()
    names = set(string_g.vs["name"])
    print(f"STRING gene space: {string_g.vcount():,} nodes, {string_g.ecount():,} edges")
    mem = set(pd.read_csv(ROOT / NET["compartment_membrane_file"], sep="\t", header=None).iloc[:, 0])
    nuc = set(pd.read_csv(ROOT / NET["compartment_nucleus_file"], sep="\t", header=None).iloc[:, 0])

    for seed_name, grp in seeds.groupby("seed_name"):
        out = OUT / seed_name
        out.mkdir(parents=True, exist_ok=True)
        genes = [g for g in grp["gene"] if g in names]
        missing = sorted(set(grp["gene"]) - set(genes))
        print(f"\n[{seed_name}] {len(genes)}/{len(grp)} seed genes in STRING; missing: {missing[:15]}{' ...' if len(missing) > 15 else ''}")
        (out / "seed_genes_in_string.txt").write_text("\n".join(genes) + "\n")

        edges = shortest_path_subnetwork(string_g, genes)
        sub = ig.Graph.TupleList(sorted(edges), directed=False)
        sub.simplify()
        print(f"[{seed_name}] subnetwork: {sub.vcount():,} nodes, {sub.ecount():,} edges")
        pd.DataFrame(sorted(edges), columns=["gene_a", "gene_b"]).to_csv(out / "subnetwork_edges.tsv", sep="\t", index=False)

        node_names = sub.vs["name"]
        mem_nodes = sorted(set(node_names) & mem)
        nuc_nodes = sorted(set(node_names) & nuc)
        (out / "membrane_genes.txt").write_text("\n".join(mem_nodes) + "\n")
        (out / "nucleus_genes.txt").write_text("\n".join(nuc_nodes) + "\n")

        cent = pd.DataFrame({
            "gene": node_names,
            "is_seed": [n in set(genes) for n in node_names],
            "degree": sub.degree(),
            "betweenness": sub.betweenness(),
            "closeness": sub.closeness(),
            "eigenvector": sub.eigenvector_centrality(scale=True),
            "pagerank": sub.pagerank(),
        }).set_index("gene")
        if mem_nodes and nuc_nodes:
            nxg = nx.Graph(list(edges))
            bsub = nx.betweenness_centrality_subset(nxg, sources=mem_nodes, targets=nuc_nodes)
            cent["betweenness_membrane_to_nucleus"] = [bsub.get(n, 0.0) for n in node_names]
        cent["is_membrane"] = [n in mem for n in node_names]
        cent["is_nucleus"] = [n in nuc for n in node_names]
        cent.to_csv(out / "centrality_results.tsv", sep="\t")
        pd.DataFrame({"gene": node_names, "is_seed": cent["is_seed"].values}).to_csv(out / "subnetwork_nodes.tsv", sep="\t", index=False)
        print(f"[{seed_name}] seeds in subnetwork {int(cent['is_seed'].sum())}, connectors {int((~cent['is_seed']).sum())}, "
              f"membrane {len(mem_nodes)}, nucleus {len(nuc_nodes)}")


if __name__ == "__main__":
    main()
