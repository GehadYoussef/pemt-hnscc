"""Test LINCS mechanism-of-action enrichment against 50 random gene sets of the query's size.

src/06_network_analysis/05_lincs_signature_reversal.py compares the mechanism-of-action shares of
the reversing L1000 compound signatures of the real query with those of 5 random gene sets of the
same size. Here 50 random sets are drawn in the same way (same 4,000-gene universe, same seed, so
the first five sets are the same five). For each mechanism class the output gives the real share,
the mean and SD of the random shares, the enrichment (real share over mean random share, with the
denominator floored at 1e-4), and an empirical p-value: the fraction of random sets whose share is
at least the real share, with add-one correction. q values are Benjamini-Hochberg.

The per-perturbagen sign test is also recomputed as a two-sided binomial test, next to the
one-sided test of 05_lincs_signature_reversal.py. Reversing perturbagens are those with at least 5
signatures, FDR <= 0.05 and more than half of their signatures reversing.

Random-set queries go to the public SigCom LINCS API. Their shares are cached after each set, so an
interrupted run resumes where it stopped.

Inputs:  results/network_analysis/lincs/lincs_l1000_cp_signatures.tsv
         results/network_analysis/lincs/lincs_query_signature.tsv
         results/network_analysis/lincs/lincs_l1000_cp_perturbagens.tsv
         results/multinomial_classifier/classifier_genes.txt
         data/raw/prism_broad/secondary_screen/secondary-screen-dose-response-curve-parameters.csv (mechanism-of-action annotation)
         SigCom LINCS API
Outputs: results/lincs_controls/random_set_shares.tsv (cache)
         results/lincs_controls/lincs_moa_enrichment_50sets.tsv
         results/lincs_controls/lincs_l1000_cp_perturbagens_two_sided.tsv
Usage:   python src/11_composition_and_mechanism/05_lincs_random_controls.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results" / "lincs_controls"
OUT.mkdir(parents=True, exist_ok=True)
N_SETS = 50

spec = importlib.util.spec_from_file_location("lincs", ROOT / "src/06_network_analysis/05_lincs_signature_reversal.py")
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)


def main() -> None:
    real = pd.read_csv(L.OUT / "lincs_l1000_cp_signatures.tsv", sep="\t")
    q = pd.read_csv(L.OUT / "lincs_query_signature.tsv", sep="\t")
    n_up, n_dn = int(((q.direction == "up") & q.mapped_to_lincs).sum()), int(((q.direction == "down") & q.mapped_to_lincs).sum())
    mmap = L.moa_map()
    real_share = L.moa_share(real, mmap)

    genes = pd.read_csv(L.CLF_DIR / "classifier_genes.txt", header=None)[0]
    universe = list(L.map_entities(list(genes.sample(4000, random_state=0))).values())
    rng = np.random.default_rng(L.CONTROL_SEED)
    shares = []
    cache = OUT / "random_set_shares.tsv"
    done = pd.read_csv(cache, sep="\t", index_col=0) if cache.exists() else pd.DataFrame()
    for k in range(N_SETS):
        pick = rng.choice(len(universe), size=n_up + n_dn, replace=False)   # drawn every time to keep the stream
        if str(k) in done.columns:
            shares.append(done[str(k)])
            continue
        c = L.run_query([universe[i] for i in pick[:n_up]], [universe[i] for i in pick[n_up:]], "l1000_cp", L.LIMIT)
        if c.empty:
            print(f"  set {k + 1}: no results")
            continue
        c["moa"] = c["perturbagen"].astype(str).str.lower().map(mmap).fillna("")
        s = L.moa_share(c, mmap).rename(str(k))
        shares.append(s)
        pd.concat(shares, axis=1).fillna(0.0).to_csv(cache, sep="\t")
        print(f"  random set {k + 1}/{N_SETS} done", flush=True)

    ctrl = pd.concat(shares, axis=1).fillna(0.0)
    rows = []
    for cls in sorted(set(real_share.index) | set(ctrl.index)):
        rs = float(real_share.get(cls, 0.0))
        cs = ctrl.reindex([cls]).fillna(0.0).values.ravel()
        if rs == 0 and cs.max() == 0:
            continue
        rows.append({"moa_class": cls, "share_real": rs, "share_random_mean": cs.mean(), "share_random_sd": cs.std(),
                     "enrichment": rs / max(cs.mean(), 1e-4), "n_random_sets": len(cs),
                     "empirical_p": (1 + (cs >= rs).sum()) / (1 + len(cs))})
    en = pd.DataFrame(rows).sort_values("share_real", ascending=False)
    en["empirical_q"] = multipletests(en["empirical_p"], method="fdr_bh")[1]
    en.to_csv(OUT / "lincs_moa_enrichment_50sets.tsv", sep="\t", index=False)
    print(en.head(15).round(4).to_string(index=False))

    ag = pd.read_csv(L.OUT / "lincs_l1000_cp_perturbagens.tsv", sep="\t")
    ag["sign_test_p_two_sided"] = [binomtest(int(r), int(n), 0.5).pvalue for r, n in zip(ag.n_reversing, ag.n_signatures)]
    ag["fdr_two_sided"] = multipletests(ag["sign_test_p_two_sided"], method="fdr_bh")[1]
    ag.to_csv(OUT / "lincs_l1000_cp_perturbagens_two_sided.tsv", sep="\t", index=False)
    hit = lambda col: ag[(ag.n_signatures >= 5) & (ag[col] <= 0.05) & (ag.fraction_reversing > 0.5)].perturbagen.tolist()
    print("reversing at FDR <= 0.05, one-sided:", hit("fdr"))
    print("reversing at FDR <= 0.05, two-sided:", hit("fdr_two_sided"))


if __name__ == "__main__":
    main()
