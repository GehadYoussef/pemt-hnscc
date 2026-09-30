"""Network proximity of drug targets to the malignant arm of pEMT, independent of the classifier.

The classifier seed set (top 100 positive pEMT-high coefficients) contains EGFR and three of its
ligands, and a drug whose single target is a seed sits at distance zero. The seed set used here
does not depend on the classifier. It holds the Puram pEMT genes that the single-cell partition
(src/09_gene_partition/01_pemt_gene_partition.py) assigns to the malignant arm, none of which is an
EGFR-family gene.

The proximity code of src/06_network_analysis/03_drug_proximity.py is imported without change,
pointed at a working root that holds only this seed set, and run twice with the same random seed,
so both variants share one null:
  standard                Guney et al. proximity with the seed genes as the disease set
  --exclude-seed-targets  a drug's targets that are themselves seeds are removed first
The summary reports, for DrugBank and STITCH targets and for each variant, the drugs tested, the
hits, the approved hits, the hits that stay hits after leaving out the most influential anchor
gene, and the ten leading approved hits with their z.

Inputs:  results/pemt_partition/puram_pemt_gene_partition.tsv
         data/references/human_interactome_gene_symbols.edgelist
         data/references/drug_targets_gene_symbols.csv (STITCH targets)
         data/raw/drug_targets/drugbank/drugbank_drug_target_flat_export.csv
         data/raw/drug_targets/drugbank/drugbank_drug_summary_export.csv
Outputs: results/network_proximity_malignant_arm/results/wgcna/network_seeds.tsv (seed list read by the proximity code)
         results/network_proximity_malignant_arm/malignant_arm/drug_proximity_{drugbank,stitch}_seeds[_noseedtargets][_significant].tsv
         results/network_proximity_malignant_arm/summary.tsv
Usage:   python src/11_composition_and_mechanism/04_malignant_arm_proximity.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEED_NAME = "malignant_arm"
WORK = ROOT / "results" / "network_proximity_malignant_arm"


def load_proximity_module():
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location(
        "drug_proximity", ROOT / "src" / "06_network_analysis" / "03_drug_proximity.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    part = pd.read_csv(ROOT / "results" / "pemt_partition" / "puram_pemt_gene_partition.tsv", sep="\t")
    genes = part.loc[part["arm"] == "malignant-expressed", "gene"].tolist()

    # working root laid out the way the proximity module expects: <root>/<results_dir>/wgcna/network_seeds.tsv
    # for the seed list and <OUT>/<seed_name>/ for the outputs
    prox = load_proximity_module()
    results_dir = prox.cfg["paths"]["results_dir"]
    (WORK / results_dir / "wgcna").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"seed_name": SEED_NAME, "module": SEED_NAME, "gene": genes, "kME": 1.0}).to_csv(
        WORK / results_dir / "wgcna" / "network_seeds.tsv", sep="\t", index=False)
    out = WORK
    (out / SEED_NAME).mkdir(parents=True, exist_ok=True)
    prox.ROOT = WORK
    prox.OUT = out

    print(f"{SEED_NAME} seed set: {len(genes)} genes")
    for extra in ([], ["--exclude-seed-targets"]):
        sys.argv = ["04_malignant_arm_proximity.py", "--disease", "seeds", "--only", SEED_NAME] + extra
        prox.main()

    rows = []
    for tname in ("drugbank", "stitch"):
        for tag, label in (("", "standard"), ("_noseedtargets", "seed targets excluded")):
            f = out / SEED_NAME / f"drug_proximity_{tname}_seeds{tag}.tsv"
            res = pd.read_csv(f, sep="\t")
            hits = res[res["is_hit"]]
            appr = hits[hits["approved"] == 1] if "approved" in hits else hits
            rows.append({"targets": tname, "variant": label, "drugs_tested": len(res),
                         "hits": len(hits), "approved_hits": len(appr),
                         "robust_hits": int(hits["robust_hit"].sum()),
                         "robust_approved_hits": int(appr["robust_hit"].sum()),
                         "top_approved": "; ".join(f"{r['name']} ({r['z']:.1f})"
                                                   for _, r in appr.head(10).iterrows())})
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "summary.tsv", sep="\t", index=False)
    print("\n" + summ.to_string(index=False))


if __name__ == "__main__":
    main()
