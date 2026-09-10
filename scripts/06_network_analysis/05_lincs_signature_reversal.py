"""LINCS L1000 signature reversal for the pEMT axis (SigCom LINCS API).

Added 9 Sep 2026, after the drug-proximity arm found no targetable neighbourhood for the
tumour-intrinsic axis. Proximity asks a topological question (are a drug's protein targets close
to the module in the interactome?); this asks a transcriptional one (does any perturbagen push
cells from the pEMT-high state towards the epithelial-like state?), which does not require the
module genes to be druggable.

Query signature: the classifier's own axis, not a WGCNA module. Up = the genes with the largest
positive pEMT-high coefficients, down = the genes with the largest positive epithelial-like
coefficients (the contrast the classifier itself learned). Both sets are mapped to SigCom LINCS
gene entities and submitted to the two-sided rank enrichment endpoint against

  l1000_cp        ~1.4 M chemical perturbagen signatures (compound x cell line x dose x time)
  l1000_xpr       CRISPR knockout signatures
  l1000_shRNA     shRNA knockdown signatures

"Reversers" are perturbagens whose signature is anti-correlated with the query (they move cells
away from pEMT-high); "mimickers" reinforce it. Per perturbagen the individual signatures are
aggregated (median z-sum, count of signatures, count in head and neck / squamous lines) and a
one-sided sign test is reported against the null that reverser and mimicker signatures are equally
likely for that perturbagen, with Benjamini-Hochberg FDR across perturbagens.

Specificity control: the same query is repeated with `n_random_controls` random gene sets of the
same size drawn from the classifier's gene space. Mechanism-of-action classes (annotated from the
PRISM secondary-screen compound table) are compared between the real query's reversing signatures
and the random-set reversing signatures, so that transcriptionally promiscuous perturbagens
(bortezomib, MG-132, HDAC inhibitors, topoisomerase poisons: the classic CMap artefact) can be
told apart from a class that is specific to this axis.

Outputs (results/network_analysis/lincs/):
  lincs_<library>_signatures.tsv     per-signature hits with metadata
  lincs_<library>_perturbagens.tsv   per-perturbagen aggregate, sorted by reversal strength
  lincs_query_signature.tsv          the up/down gene lists actually submitted
  lincs_moa_enrichment.tsv           MoA class share of reversing signatures, real vs random sets
Figure: Figure_9_lincs_reversal (top reversing compounds and their mechanism classes)
"""

import json
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import load_config, project_root  # noqa: E402
from _lib.figstyle import apply_style, save_figure, PALETTE, mm  # noqa: E402

cfg = load_config()
ROOT = project_root()
CLF_DIR = ROOT / cfg["paths"]["results_dir"] / "multinomial_classifier"
OUT = ROOT / cfg["paths"]["results_dir"] / "network_analysis" / "lincs"
FIG = ROOT / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
apply_style()
import matplotlib.pyplot as plt  # noqa: E402

META = "https://maayanlab.cloud/sigcom-lincs/metadata-api"
DATA = "https://data-api.sigcom-lincs.k8s.maayanlab.cloud/v1/api/v1"
LIBRARIES = {"l1000_cp": "chemical perturbagens", "l1000_xpr": "CRISPR knockout", "l1000_shRNA": "shRNA knockdown"}
LC = cfg.get("lincs", {}) or {}
N_GENES = int(LC.get("n_signature_genes", 150))
LIMIT = int(LC.get("result_limit", 2000))
N_CONTROL = int(LC.get("n_random_controls", 5))
CONTROL_SEED = int(LC.get("control_seed", 20260909))


def post(url: str, body: dict, timeout: int = 900) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def query_signature() -> tuple[list[str], list[str]]:
    coef = pd.read_csv(CLF_DIR / "classifier_coefficients.tsv", sep="\t", index_col=0)
    up = list(coef[coef["pEMT_high"] > 0].sort_values("pEMT_high", ascending=False).index[:N_GENES])
    dn = list(coef[coef["epithelial_like"] > 0].sort_values("epithelial_like", ascending=False).index[:N_GENES])
    return up, dn


def map_entities(symbols: list[str]) -> dict[str, str]:
    ids = {}
    for i in range(0, len(symbols), 200):
        chunk = symbols[i:i + 200]
        r = post(f"{META}/entities/find", {"filter": {"where": {"meta.symbol": {"inq": chunk}}, "fields": ["id", "meta.symbol"], "limit": 1000}})
        for e in r:
            ids[e["meta"]["symbol"]] = e["id"]
    return ids


def signature_metadata(uuids: list[str]) -> dict[str, dict]:
    meta = {}
    for i in range(0, len(uuids), 100):
        chunk = uuids[i:i + 100]
        r = post(f"{META}/signatures/find", {"filter": {"where": {"id": {"inq": chunk}}, "limit": 200}})
        for s in r:
            meta[s["id"]] = s.get("meta", {})
    return meta


def perturbagen_name(m: dict) -> str:
    for k in ("pert_name", "perturbation", "pert_id", "cmap_name", "gene_symbol"):
        v = m.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, dict) and v.get("name"):
            return str(v["name"]).strip()
    return "unknown"


def run_query(up: list[str], dn: list[str], lib: str, limit: int) -> pd.DataFrame:
    r = post(f"{DATA}/enrich/ranktwosided", {"up_entities": up, "down_entities": dn, "limit": limit, "database": lib})
    res = r.get("results", [])
    if not res:
        return pd.DataFrame()
    meta = signature_metadata([x["uuid"] for x in res])
    rows = []
    for x in res:
        m = meta.get(x["uuid"], {})
        rows.append({"uuid": x["uuid"], "type": x["type"], "z_sum": x.get("z-sum"), "z_up": x.get("z-up"), "z_down": x.get("z-down"),
                     "p_up": x.get("p-up"), "p_down": x.get("p-down"), "fdr_up": x.get("fdr-up"), "fdr_down": x.get("fdr-down"),
                     "perturbagen": perturbagen_name(m), "cell_line": m.get("cell_line") or m.get("cell_id") or "",
                     "dose": m.get("pert_dose", ""), "time": m.get("pert_time", "")})
    return pd.DataFrame(rows)


def moa_map() -> dict[str, str]:
    """Compound name -> mechanism of action, from the PRISM secondary-screen annotation (LINCS metadata has none)."""
    f = ROOT / cfg["broad_prism"]["secondary_curve_file"]
    if not f.exists():
        return {}
    t = pd.read_csv(f, usecols=["name", "moa"]).dropna().drop_duplicates("name")
    return {str(n).lower(): str(m) for n, m in zip(t["name"], t["moa"])}


def moa_share(sig: pd.DataFrame, mmap: dict[str, str]) -> pd.Series:
    """Share of reversing signatures attributable to each MoA term."""
    rev = sig[sig["type"] == "reversers"]
    counts = defaultdict(int)
    for name in rev["perturbagen"].astype(str):
        for t in [x.strip() for x in mmap.get(name.lower(), "").split(",") if x.strip()]:
            counts[t] += 1
    return pd.Series(counts, dtype=float) / max(len(rev), 1)


def main() -> None:
    if "--figure-only" in sys.argv:
        figure()
        return
    up_sym, dn_sym = query_signature()
    ids = map_entities(up_sym + dn_sym)
    up = [ids[g] for g in up_sym if g in ids]
    dn = [ids[g] for g in dn_sym if g in ids]
    pd.DataFrame({"direction": ["up"] * len(up_sym) + ["down"] * len(dn_sym), "gene": up_sym + dn_sym,
                  "mapped_to_lincs": [g in ids for g in up_sym + dn_sym]}).to_csv(OUT / "lincs_query_signature.tsv", sep="\t", index=False)
    print(f"Query signature: {len(up)}/{len(up_sym)} up (pEMT-high coefficients) and {len(dn)}/{len(dn_sym)} down (epithelial-like coefficients) mapped to LINCS genes")

    mmap = moa_map()
    for lib, label in LIBRARIES.items():
        t0 = time.time()
        try:
            sig = run_query(up, dn, lib, LIMIT)
        except Exception as exc:  # noqa: BLE001
            print(f"[{lib}] query failed: {exc}")
            continue
        print(f"[{lib}] {label}: {len(sig)} signatures returned in {time.time() - t0:.0f} s")
        if sig.empty:
            continue
        sig["moa"] = sig["perturbagen"].astype(str).str.lower().map(mmap).fillna("")
        sig.to_csv(OUT / f"lincs_{lib}_signatures.tsv", sep="\t", index=False)

        agg = []
        for name, g in sig.groupby("perturbagen"):
            if name == "unknown":
                continue
            rev = g[g["type"] == "reversers"]
            mim = g[g["type"] == "mimickers"]
            n_rev, n_mim = len(rev), len(mim)
            if n_rev + n_mim < 3:
                continue
            bt = binomtest(n_rev, n_rev + n_mim, 0.5, alternative="greater")
            agg.append({"perturbagen": name, "n_signatures": n_rev + n_mim, "n_reversing": n_rev, "n_mimicking": n_mim,
                        "fraction_reversing": n_rev / (n_rev + n_mim), "median_z_sum_reversing": rev["z_sum"].median() if n_rev else np.nan,
                        "best_z_sum": g["z_sum"].abs().max(), "cell_lines": ", ".join(sorted(set(g["cell_line"].dropna().astype(str)))[:6]),
                        "moa": mmap.get(str(name).lower(), ""), "sign_test_p": bt.pvalue})
        if not agg:
            continue
        ag = pd.DataFrame(agg)
        ag["fdr"] = multipletests(ag["sign_test_p"], method="fdr_bh")[1]
        ag = ag.sort_values(["fraction_reversing", "n_reversing"], ascending=False)
        ag.to_csv(OUT / f"lincs_{lib}_perturbagens.tsv", sep="\t", index=False)
        top = ag[(ag["n_signatures"] >= 5) & (ag["fdr"] <= 0.05)]
        print(f"  {len(ag)} perturbagens with >= 3 signatures; {len(top)} reversing at FDR <= 0.05")
        print("  top reversers: " + ", ".join(f"{r.perturbagen} ({r.n_reversing}/{r.n_signatures})" for r in ag.head(12).itertuples()))

    # Specificity control: random gene sets of the same size, MoA class shares compared
    real = pd.read_csv(OUT / "lincs_l1000_cp_signatures.tsv", sep="\t") if (OUT / "lincs_l1000_cp_signatures.tsv").exists() else pd.DataFrame()
    if not real.empty:
        universe = list(map_entities(list(pd.read_csv(CLF_DIR / "classifier_genes.txt", header=None)[0].sample(4000, random_state=0))).values())
        rng = np.random.default_rng(CONTROL_SEED)
        real_share = moa_share(real, mmap)
        ctrl_shares = []
        for k in range(N_CONTROL):
            pick = rng.choice(len(universe), size=len(up) + len(dn), replace=False)
            u = [universe[i] for i in pick[:len(up)]]
            d = [universe[i] for i in pick[len(up):]]
            c = run_query(u, d, "l1000_cp", LIMIT)
            if not c.empty:
                c["moa"] = c["perturbagen"].astype(str).str.lower().map(mmap).fillna("")
                ctrl_shares.append(moa_share(c, mmap))
            print(f"  random control {k + 1}/{N_CONTROL} done")
        ctrl = pd.concat(ctrl_shares, axis=1).fillna(0.0) if ctrl_shares else pd.DataFrame()
        classes = sorted(set(real_share.index) | set(ctrl.index))
        rows = []
        for cls in classes:
            rs = float(real_share.get(cls, 0.0))
            cs = ctrl.reindex([cls]).fillna(0.0).values.ravel() if len(ctrl) else np.zeros(1)
            if rs == 0 and cs.max() == 0:
                continue
            rows.append({"moa_class": cls, "share_real": rs, "share_random_mean": float(cs.mean()), "share_random_sd": float(cs.std()),
                         "enrichment": rs / max(float(cs.mean()), 1e-4), "n_random_sets": len(cs)})
        en = pd.DataFrame(rows).sort_values("share_real", ascending=False)
        en.to_csv(OUT / "lincs_moa_enrichment.tsv", sep="\t", index=False)
        print("\nMoA classes among the reversing signatures, real query vs random gene sets:")
        print(en.head(12).round(4).to_string(index=False))

    figure()


def figure() -> None:
    """Redraw Figure 11 from the cached LINCS result tables, without re-querying the API."""
    # Figure
    f = OUT / "lincs_l1000_cp_perturbagens.tsv"
    if f.exists():
        ag = pd.read_csv(f, sep="\t")
        top = ag[ag["n_reversing"] >= 5].sort_values("n_reversing", ascending=False).head(22).iloc[::-1]
        fig, axes = plt.subplots(1, 2, figsize=(mm(180), mm(95)), gridspec_kw={"width_ratios": [1.15, 1.25]})
        ax = axes[0]
        y = np.arange(len(top))
        ax.barh(y, top["n_reversing"], color=[PALETTE["vermilion"] if q <= 0.05 else PALETTE["orange"] for q in top["fdr"]], height=0.7)
        ax.set_yticks(y); ax.set_yticklabels(top["perturbagen"], fontsize=6.5)
        ax.set_xlabel("L1000 signatures reversing the pEMT axis")
        ax.set_xlim(0, float(top["n_reversing"].max()) * 1.28)
        for yi, (_, r) in zip(y, top.iterrows()):
            mim = int(r["n_mimicking"])
            ax.text(r["n_reversing"] + 0.5, yi, f"{int(r['n_reversing'])} vs {mim} mimicking", va="center", fontsize=5.5, color=PALETTE["grey"])
        ax.scatter([], [], marker="s", color=PALETTE["vermilion"], s=16, label="sign test FDR ≤ 0.05")
        ax.scatter([], [], marker="s", color=PALETTE["orange"], s=16, label="FDR > 0.05")
        ax.legend(loc="lower right", fontsize=6, frameon=False)
        ax.set_title("a", loc="left", fontweight="bold")
        ax2 = axes[1]
        enf = OUT / "lincs_moa_enrichment.tsv"
        if enf.exists():
            en = pd.read_csv(enf, sep="\t")
            en = en[(en["share_real"] >= 0.005) | (en["share_random_mean"] >= 0.005)].sort_values("share_real", ascending=False).head(14).iloc[::-1]
            y2 = np.arange(len(en))
            ax2.barh(y2 + 0.2, en["share_real"] * 100, height=0.38, color=PALETTE["vermilion"], label="pEMT axis")
            ax2.barh(y2 - 0.2, en["share_random_mean"] * 100, height=0.38, color=PALETTE["grey"], label=f"random gene sets (mean of {N_CONTROL})")
            ax2.set_yticks(y2); ax2.set_yticklabels([c.replace("tyrosine kinase receptor inhibitor", "inhibitor").replace(" inhibitor", "") for c in en["moa_class"]], fontsize=6.5)
            ax2.set_xlabel("Share of reversing signatures (%), by drug class")
            ax2.legend(fontsize=6, frameon=False, loc="lower right")
        ax2.set_title("b", loc="left", fontweight="bold")
        fig.tight_layout()
        save_figure(fig, FIG, "Figure_9_lincs_reversal")


if __name__ == "__main__":
    main()
