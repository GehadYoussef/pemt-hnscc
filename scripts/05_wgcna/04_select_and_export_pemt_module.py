"""Select and export the module(s) that seed the network analysis.

Patched 9 Sep 2026. Two seed sets are exported and both go through stage 06:

  pEMT_axis      the module ranked first in module_pemt_ranking.tsv (strongest
                 Puram-pEMT enrichment among modules whose eigengene correlates
                 positively with P(pEMT-high)); genes with module membership
                 kME >= wgcna.seed_min_kME (default 0.5) so PAM-assigned genes
                 with weak membership do not dilute the seed
  canonical_pEMT the Puram pEMT signature genes that fall in the module with the
                 largest pEMT overlap (in TCGA this is the stromal module), i.e.
                 the canonical program as it co-expresses in bulk

config.wgcna.primary_module can override the automatic choice (e.g. "M13").

Outputs (results/wgcna/):
  module_<M>_genes.txt for every module, module_<M>_hubs_kME.tsv for the seeds,
  network_seeds.tsv (seed_name, module, gene)
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import load_config, load_signatures, project_root  # noqa: E402

cfg = load_config()
ROOT = project_root()
WG = ROOT / cfg["paths"]["results_dir"] / "wgcna"
MIN_KME = float(cfg["wgcna"].get("seed_min_kME", 0.5))


def main() -> None:
    assign = pd.read_csv(WG / "gene_module_assignments.tsv", sep="\t")
    kme = pd.read_csv(WG / "gene_kME.tsv", sep="\t", index_col=0)
    rank = pd.read_csv(WG / "module_pemt_ranking.tsv", sep="\t")
    ann = pd.read_csv(WG / "module_annotation.tsv", sep="\t").set_index("module")
    pemt = set(load_signatures()["puram_pemt"])

    for m in sorted(assign["module"].unique()):
        if m == "grey":
            continue
        (WG / f"module_{m}_genes.txt").write_text("\n".join(assign.loc[assign["module"] == m, "gene"]) + "\n")

    override = str(cfg["wgcna"].get("primary_module", "auto"))
    primary = override if override.lower() != "auto" else rank.iloc[0]["module"]
    canonical = ann["puram_pemt_overlap"].idxmax()

    seeds = []
    sub = kme[kme["module"] == primary].sort_values(f"kME_{primary}", ascending=False)
    sub[[f"kME_{primary}"]].to_csv(WG / f"module_{primary}_hubs_kME.tsv", sep="\t")
    strong = sub.index[sub[f"kME_{primary}"] >= MIN_KME].tolist()
    seeds += [{"seed_name": "pEMT_axis", "module": primary, "gene": g, "kME": round(float(sub.loc[g, f"kME_{primary}"]), 3)} for g in strong]

    csub = kme[kme["module"] == canonical]
    cgenes = sorted(set(csub.index) & pemt)
    seeds += [{"seed_name": "canonical_pEMT", "module": canonical, "gene": g, "kME": round(float(csub.loc[g, f"kME_{canonical}"]), 3)} for g in cgenes]
    csub.sort_values(f"kME_{canonical}", ascending=False)[[f"kME_{canonical}"]].to_csv(WG / f"module_{canonical}_hubs_kME.tsv", sep="\t")

    # third seed set (added after review): the classifier's own pEMT-high axis, the 100 largest positive
    # coefficients (contains EGFR, its ligands AREG/EREG/NRG1, the alpha6beta4/alpha3beta1 integrins and laminin-332)
    coef = pd.read_csv(ROOT / cfg["paths"]["results_dir"] / "multinomial_classifier" / "classifier_coefficients.tsv", sep="\t", index_col=0)
    top = coef[coef["pEMT_high"] > 0].sort_values("pEMT_high", ascending=False).head(100)
    seeds += [{"seed_name": "classifier_pEMT_high", "module": "classifier", "gene": g, "kME": round(float(v), 4)} for g, v in top["pEMT_high"].items()]

    seeds_df = pd.DataFrame(seeds)
    seeds_df.to_csv(WG / "network_seeds.tsv", sep="\t", index=False)
    print(f"classifier_pEMT_high seed: {len(top)} genes; top: {', '.join(top.index[:12])}")
    print(f"pEMT_axis seed: module {primary}, {len(strong)} genes with kME >= {MIN_KME} (module size {len(sub)}); "
          f"top: {', '.join(strong[:12])}")
    print(f"canonical_pEMT seed: {len(cgenes)} Puram pEMT genes in module {canonical} (module size {len(csub)})")


if __name__ == "__main__":
    main()
