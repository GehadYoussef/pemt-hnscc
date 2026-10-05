"""Comparison with published programmes: the Gavish et al. 2023 meta-programmes and Tyler and Tirosh 2021.

09_gene_partition/03_tyler_tirosh_benchmark.py implements the Tyler and Tirosh decoupling principle on
these data. This stage uses what the two papers published, unchanged.

Sources
  Gavish A, et al. Hallmarks of transcriptional intratumour heterogeneity across a thousand tumours.
  Nature 2023, 618:598-606 (PMID 37258682). Supplementary Table 2, sheet "Cancer MPs": 41 cancer-cell
  meta-programmes (MPs) of 50 genes each, downloaded on first use from
  https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-023-06130-4/MediaObjects/41586_2023_6130_MOESM6_ESM.xlsx
  into data/raw/gavish2023/. The same lists are in MP_list.RDS.gz at https://github.com/tiroshlab/3ca
  (ITH_hallmarks/MPs_distribution).

  Tyler M, Tirosh I. Decoupling epithelial-mesenchymal transitions from stromal profiles by integrative
  expression analysis. Nat Commun 2021, 12:2592 (PMID 33972543). Code availability names
  https://github.com/m20ty/decoupling_emt_stroma (Zenodo 10.5281/zenodo.4553528). Neither per-sample
  TCGA-HNSC scores nor the deconvolved HNSC cancer-EMT (pEMT) and CAF gene lists are published in a
  usable form. The repository and the Zenodo release hold R code only, and the deconvolution output
  (deconv_data_all.rds, written by deconv_all.R) and the input marker table (emt_markers.csv) are not
  deposited. The Supplementary Information has Table S1 (single-cell datasets) and figures, in which the
  HNSC gene orderings appear only as partial heatmap labels. The Source Data file (MOESM3, a zip holding
  one csv) contains simulated bulk profiles built from single cells, not TCGA scores, and the archive
  served by the publisher is truncated. Nothing is reconstructed from figures. The Tyler and Tirosh
  comparison therefore uses the compartment scores of 09_gene_partition/03 and the availability record
  written to tyler_tirosh_availability.tsv.

Analyses
  1. TCGA-HNSC (520 primaries, log2(TPM + 1)): each MP scored as a mean per-gene z-score (the zmean of
     04_tcga_projection/07, as for every published signature). Spearman correlation with pEMT
     specificity, P(pEMT-high), the 12-gene malignant core, the 14-gene stromal core, the canonical
     Puram pEMT signature, Hallmark EMT and the two Tyler and Tirosh compartments.
  2. GSE65021 (platinum plus cetuximab, 14 long and 26 short PFS): the same MP scores, AUC for long PFS
     with the bootstrap interval of 10_cetuximab/01 (2,000 resamples, seed 20260920), rank-biserial
     correlation and two-sided Mann-Whitney p, Benjamini-Hochberg adjusted over the 41 MPs. EMT-II is
     also scored without the 7 genes it shares with the malignant core (MP13_minus_malignant_core). The
     malignant core, stromal core, pEMT specificity and the Puram signature are tested the same way for
     reference. EMT-II and each MP with BH q < 0.05 are then entered with the malignant core in a Firth
     model.
  3. Overlap of each MP with the malignant core, the stromal core, the classifier's top 100 pEMT-high
     coefficients and the Puram pEMT list: one-sided hypergeometric test over the classifier genes, with
     MP genes mapped to current symbols as below.

Gene symbols. The MP lists use the symbols of the original studies (for example CTGF, CYR61, C15ORF48).
Each MP gene is matched as written, then with ORF in lower case (C15ORF48 becomes C15orf48), then through
PREVIOUS_SYMBOL, a table of HGNC previous symbols and their current approved symbols. Unmatched genes are
dropped and counted in mp_gene_coverage.tsv.

Inputs:  data/raw/gavish2023/ (downloaded), data/raw/external_cohorts/GSE65021 files
         data/processed/tcga_hnsc/tcga_star_tpm_log2_for_projection.tsv, config/signatures/puram_pemt.txt
         results/tcga_projection/emt_score_panel_scores.tsv
         results/multinomial_classifier/classifier_genes.txt, top_positive_genes_pEMT_high.tsv
         results/arm_scores/{TCGA-HNSC_arm_scores,GSE65021_arm_scores,arm_gene_sets}.tsv
         results/tyler_tirosh/tcga_scores_with_tyler_tirosh.tsv, results/cetuximab_cohort/GSE65021_scores.tsv
Outputs: results/published_programmes/*.tsv
Usage:   python src/13_revision/05_gavish_meta_programmes.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, mannwhitneyu, spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from pemt import firth as _firth  # noqa: E402

RES = ROOT / "results"
OUT = RES / "published_programmes"
RAW = ROOT / "data" / "raw" / "gavish2023"

GAVISH_URL = ("https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2F"
              "s41586-023-06130-4/MediaObjects/41586_2023_6130_MOESM6_ESM.xlsx")
GAVISH_FILE = RAW / "Gavish2023_SupplementaryTable2_MOESM6.xlsx"
GAVISH_SHEET = "Cancer MPs"

# MPs named in Gavish et al. as EMT programmes, plus the epithelial senescence MP and the glioma
# mesenchymal MP, which are the other candidates for a partial-EMT-like state.
EMT_RELATED = ["MP12", "MP13", "MP14", "MP15", "MP16", "MP19"]

# HGNC previous symbol and current approved symbol, for MP genes whose list symbol is absent from the
# TCGA matrix. Applied only when the symbol as written is not found.
PREVIOUS_SYMBOL = {
    "CTGF": "CCN2", "CYR61": "CCN1", "NOV": "CCN3", "PRKCDBP": "CAVIN3", "SDPR": "CAVIN2",
    "SEPP1": "SELENOP", "C10ORF10": "DEPP1", "KIAA0101": "PCLAF", "H2AFZ": "H2AZ1", "H2AFV": "H2AZ2",
    "H3F3A": "H3-3A", "HIST1H4C": "H4C3", "HIST1H1E": "H1-4", "HIST1H1C": "H1-2", "HIST1H1B": "H1-5",
    "HIST2H2AC": "H2AC20", "HIST3H2A": "H2AC25", "GARS": "GARS1", "SARS": "SARS1", "CARS": "CARS1",
    "YARS": "YARS1", "WARS": "WARS1", "EPRS": "EPRS1", "QARS": "QARS1", "HN1": "JPT1",
    "LINC00152": "CYTOR", "FAM45A": "DENND10", "GUCY1B3": "GUCY1B1", "FYB": "FYB1", "SEPT4": "SEPTIN4",
    "SEPT8": "SEPTIN8", "ATP5G1": "ATP5MC1", "ATP5G3": "ATP5MC3", "ATP5I": "ATP5ME", "ATP5E": "ATP5F1E",
    "ATP5J2": "ATP5MF", "PLA2G16": "PLAAT3", "TMEM206": "PACC1", "LPPR1": "PLPPR1", "FAM19A2": "TAFA2",
    "HRSP12": "RIDA", "TMEM27": "CLTRN", "NRD1": "NRDC", "FAM208B": "TASOR2", "C6ORF48": "SNHG32",
    "METTL7B": "TMT1B", "HMP19": "NSG2", "GRAMD3": "GRAMD2B", "C15ORF48": "NMES1",
}

TCGA_REF = ["pEMT_specificity", "P_pEMT_high", "malignant_arm_core", "stromal_arm_core", "puram_pemt",
            "hallmark_EMT", "TT_cancer_EMT", "TT_caf_EMT"]
GSE_REF = ["malignant_arm_core", "stromal_arm_core", "pEMT_specificity", "puram_pemt", "hallmark_EMT"]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


# sources
def gavish_mps() -> dict[str, list[str]]:
    RAW.mkdir(parents=True, exist_ok=True)
    if not GAVISH_FILE.exists():
        req = urllib.request.Request(GAVISH_URL, headers={"User-Agent": "Mozilla/5.0"})
        GAVISH_FILE.write_bytes(urllib.request.urlopen(req, timeout=120).read())
    sha = hashlib.sha256(GAVISH_FILE.read_bytes()).hexdigest()
    x = pd.read_excel(GAVISH_FILE, sheet_name=GAVISH_SHEET)
    mps = {}
    for col in x.columns:
        name = " ".join(str(col).split())
        mps[name] = [str(g).strip() for g in x[col].dropna()]
    pd.DataFrame([{"file": GAVISH_FILE.name, "url": GAVISH_URL, "sheet": GAVISH_SHEET, "sha256": sha,
                   "n_programmes": len(mps)}]).to_csv(OUT / "gavish_source.tsv", sep="\t", index=False)
    return mps


def mp_id(name: str) -> str:
    return name.split()[0]


def map_symbols(genes: list[str], universe: set[str]) -> tuple[list[str], list[str]]:
    found, missing = [], []
    for g in genes:
        for cand in (g, g.replace("ORF", "orf"), PREVIOUS_SYMBOL.get(g)):
            if cand and cand in universe:
                found.append(cand)
                break
        else:
            missing.append(g)
    return list(dict.fromkeys(found)), missing


def tyler_tirosh_availability() -> None:
    rows = [
        {"item": "per-sample TCGA-HNSC cancer-EMT (pEMT) and CAF scores", "published": "no",
         "where_checked": "Nat Commun article and Supplementary Information, Source Data, "
                          "github.com/m20ty/decoupling_emt_stroma, Zenodo 10.5281/zenodo.4553528",
         "note": "the deconvolution output deconv_data_all.rds is written by deconv_all.R but not deposited"},
        {"item": "deconvolved HNSC cancer-EMT and CAF gene lists", "published": "no (figure labels only)",
         "where_checked": "Supplementary Information PDF (Table S1 only, Fig. 3 and Fig. S10 heatmaps)",
         "note": "heatmap labels are a partial annotation set and were not transcribed"},
        {"item": "input EMT marker table (emt_markers.csv)", "published": "no",
         "where_checked": "github.com/m20ty/decoupling_emt_stroma (code reads ../../emt_markers.csv)",
         "note": "seed genes SNAI1, SNAI2, TWIST1, VIM, ZEB1, ZEB2 are in deconv.R and are used by "
                 "09_gene_partition/03_tyler_tirosh_benchmark.py"},
        {"item": "Source Data (41467_2021_22800_MOESM3_ESM.zip)", "published": "yes, but not usable",
         "where_checked": "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2F"
                          "s41467-021-22800-1/MediaObjects/41467_2021_22800_MOESM3_ESM.zip",
         "note": "simulated bulk profiles from single cells (1,000 per cancer type), not TCGA scores. "
                 "The archive is truncated on the server (522,082,304 bytes, csv compressed length 671,282,178)"},
    ]
    pd.DataFrame(rows).to_csv(OUT / "tyler_tirosh_availability.tsv", sep="\t", index=False)


# scoring
def malignant_core() -> list[str]:
    sets = pd.read_csv(RES / "arm_scores" / "arm_gene_sets.tsv", sep="\t", index_col=0)
    return [g.strip() for g in sets.loc["malignant_arm_core", "genes"].split(",")]


def score_all(expr: pd.DataFrame, mps: dict[str, list[str]], zmean, cohort: str, cover: list) -> pd.DataFrame:
    idx = set(expr.index)
    s = pd.DataFrame(index=expr.columns)
    for name, genes in mps.items():
        found, missing = map_symbols(genes, idx)
        s[mp_id(name)], n, _ = zmean(expr, found)
        cover.append({"cohort": cohort, "MP": mp_id(name), "name": name, "genes_in_list": len(genes),
                      "genes_scored": n, "missing": ", ".join(missing)})
    # sensitivity: EMT-II shares 7 genes with the malignant core, so it is also scored without them
    core = set(malignant_core())
    mp13 = next(v for k, v in mps.items() if mp_id(k) == "MP13")
    found, _ = map_symbols(mp13, idx)
    s["MP13_minus_malignant_core"], n, _ = zmean(expr, [g for g in found if g not in core])
    cover.append({"cohort": cohort, "MP": "MP13_minus_malignant_core", "name": "MP13 EMT-II without core genes",
                  "genes_in_list": len(mp13), "genes_scored": n, "missing": ""})
    return s


def tcga(mps, zmean, cover) -> pd.DataFrame:
    ref = pd.read_csv(RES / "tcga_projection" / "emt_score_panel_scores.tsv", sep="\t", index_col=0)
    e = pd.read_csv(ROOT / "data" / "processed" / "tcga_hnsc" / "tcga_star_tpm_log2_for_projection.tsv",
                    sep="\t", index_col=0)
    e = e[ref.index.intersection(e.columns)]
    print(f"TCGA-HNSC: {e.shape[0]} genes x {e.shape[1]} primaries")
    s = score_all(e, mps, zmean, "TCGA-HNSC", cover)
    arms = pd.read_csv(RES / "arm_scores" / "TCGA-HNSC_arm_scores.tsv", sep="\t", index_col=0)
    tt = pd.read_csv(RES / "tyler_tirosh" / "tcga_scores_with_tyler_tirosh.tsv", sep="\t", index_col=0)
    d = s.join(ref[["pEMT_specificity", "P_pEMT_high", "puram_pemt", "hallmark_EMT"]]) \
        .join(arms[["malignant_arm_core", "stromal_arm_core"]]).join(tt[["TT_cancer_EMT", "TT_caf_EMT"]])
    d.to_csv(OUT / "TCGA-HNSC_mp_scores.tsv", sep="\t")

    names = {mp_id(k): k for k in mps}
    rows = []
    for m in s.columns:
        r = {"MP": m, "name": names.get(m, m), "emt_related": m in EMT_RELATED}
        for c in TCGA_REF:
            rho, p = spearmanr(d[m], d[c], nan_policy="omit")
            r[f"rho_{c}"], r[f"p_{c}"] = rho, p
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "tcga_mp_correlations.tsv", sep="\t", index=False)
    show = out[out.emt_related][["name"] + [f"rho_{c}" for c in TCGA_REF]]
    show.columns = ["MP"] + TCGA_REF
    print("\nTCGA-HNSC Spearman, EMT-related MPs:\n" + show.round(2).to_string(index=False))
    for c in ("pEMT_specificity", "malignant_arm_core"):
        top = out.nlargest(3, f"rho_{c}")
        print(f"  MPs most correlated with {c}: " +
              ", ".join(f"{n} {v:+.2f}" for n, v in zip(top["name"], top[f"rho_{c}"])))
    return d


def gse65021(mps, zmean, cover) -> pd.DataFrame:
    cx = _load(SRC / "10_cetuximab" / "01_cetuximab_cohort.py", "cx")
    x, _ = cx.read_series_matrix(cx.DATA / "GSE65021_series_matrix.txt.gz")
    e = cx.collapse(x, cx.read_annotation(cx.DATA / "GPL10558.annot.gz"))
    ref = pd.read_csv(RES / "cetuximab_cohort" / "GSE65021_scores.tsv", sep="\t", index_col=0)
    arms = pd.read_csv(RES / "arm_scores" / "GSE65021_arm_scores.tsv", sep="\t", index_col=0)
    e = e[ref.index.intersection(e.columns)]
    s = score_all(e, mps, zmean, "GSE65021", cover)
    d = s.join(ref[["long_pfs", "pEMT_specificity", "puram_pemt", "hallmark_EMT"]]) \
        .join(arms[["malignant_arm_core", "stromal_arm_core"]])
    d.to_csv(OUT / "GSE65021_mp_scores.tsv", sep="\t")
    y = d["long_pfs"].to_numpy(int)
    print(f"\nGSE65021: {e.shape[0]} genes x {e.shape[1]} patients ({y.sum()} long, {len(y) - y.sum()} short PFS)")

    names = {mp_id(k): k for k in mps}
    rows = []
    for sc in list(s.columns) + GSE_REF:
        a, b = d.loc[y == 1, sc], d.loc[y == 0, sc]
        auc, lo, hi = cx.bootstrap_auc(y, d[sc].values)
        u = mannwhitneyu(a, b, alternative="two-sided")
        kind = "MP" if sc in names else ("sensitivity" if sc.startswith("MP") else "reference")
        rows.append({"score": sc, "name": names.get(sc, sc), "kind": kind,
                     "emt_related": sc in EMT_RELATED, "AUC": auc, "AUC_low": lo, "AUC_up": hi,
                     "rank_biserial": 2 * u.statistic / (len(a) * len(b)) - 1, "p_two_sided": u.pvalue,
                     "rho_pEMT_specificity": spearmanr(d[sc], d["pEMT_specificity"])[0],
                     "rho_malignant_core": spearmanr(d[sc], d["malignant_arm_core"])[0],
                     "rho_stromal_core": spearmanr(d[sc], d["stromal_arm_core"])[0]})
    out = pd.DataFrame(rows)
    mp = out["kind"] == "MP"
    out.loc[mp, "q_BH_41MPs"] = bh(out.loc[mp, "p_two_sided"])
    emt = out["emt_related"]
    out.loc[emt, "q_BH_EMT_related"] = bh(out.loc[emt, "p_two_sided"])
    out.to_csv(OUT / "gse65021_mp_pfs_auc.tsv", sep="\t", index=False)
    cols = ["name", "AUC", "AUC_low", "AUC_up", "p_two_sided", "q_BH_41MPs", "rho_malignant_core"]
    print("\nGSE65021, long against short PFS (EMT-related MPs and references):")
    print(out[emt | ~mp][cols].round(3).to_string(index=False))
    print("MPs with two-sided p < 0.05:")
    print(out[mp & (out.p_two_sided < 0.05)][cols].round(3).to_string(index=False) or "  none")

    # does an MP that separates the groups carry information beyond the malignant core?
    def z(v):
        return (v - v.mean()) / v.std()

    jrows = []
    for m in ["MP13", "MP13_minus_malignant_core"] + out.loc[mp & (out.q_BH_41MPs < 0.05), "score"].tolist():
        pair = (m, "malignant_arm_core")
        X = np.column_stack([np.ones(len(d))] + [z(d[c]) for c in pair])
        f = _firth.firth(y, X, ["const", *pair])
        jrows.append({"terms": " + ".join(pair), "rho": spearmanr(d[pair[0]], d[pair[1]])[0],
                      "OR_MP": f[m][0], "OR_MP_low": f[m][1], "OR_MP_up": f[m][2], "p_MP": f[m][3],
                      "OR_core": f[pair[1]][0], "OR_core_low": f[pair[1]][1], "OR_core_up": f[pair[1]][2],
                      "p_core": f[pair[1]][3]})
    j = pd.DataFrame(jrows)
    j.to_csv(OUT / "gse65021_mp_joint_firth.tsv", sep="\t", index=False)
    print("\nJoint Firth models with the malignant core (OR per SD):\n" + j.round(3).to_string(index=False))
    return d


# overlaps
def overlaps(mps) -> None:
    clf = RES / "multinomial_classifier"
    universe = set(pd.read_csv(clf / "classifier_genes.txt", header=None)[0].astype(str))
    sets = pd.read_csv(RES / "arm_scores" / "arm_gene_sets.tsv", sep="\t", index_col=0)

    def split(s):
        return [g.strip() for g in s.split(",")]

    puram = [line.strip() for line in (ROOT / "config" / "signatures" / "puram_pemt.txt").read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    refs = {"malignant_core": split(sets.loc["malignant_arm_core", "genes"]),
            "stromal_core": split(sets.loc["stromal_arm_core", "genes"]),
            "classifier_top100_pEMT_high":
                pd.read_csv(clf / "top_positive_genes_pEMT_high.tsv", sep="\t").iloc[:100, 0].astype(str).tolist(),
            "puram_pemt": puram}
    refs = {k: [g for g in v if g in universe] for k, v in refs.items()}
    N = len(universe)
    rows = []
    for name, genes in mps.items():
        g, _ = map_symbols(genes, universe)
        for rk, rg in refs.items():
            k = sorted(set(g) & set(rg))
            rows.append({"MP": mp_id(name), "name": name, "emt_related": mp_id(name) in EMT_RELATED,
                         "reference": rk, "universe": N, "MP_genes_in_universe": len(g),
                         "reference_genes_in_universe": len(rg), "overlap": len(k),
                         "expected": len(g) * len(rg) / N,
                         "p_hypergeom": hypergeom.sf(len(k) - 1, N, len(rg), len(g)),
                         "overlap_genes": ", ".join(k)})
    out = pd.DataFrame(rows)
    for rk in refs:
        m = out.reference == rk
        out.loc[m, "q_BH_41MPs"] = bh(out.loc[m, "p_hypergeom"])
    out.to_csv(OUT / "mp_gene_overlap.tsv", sep="\t", index=False)
    show = out[out.emt_related].pivot(index="name", columns="reference", values="overlap")
    print(f"\nOverlap with the gene sets of this study (universe {N} classifier genes), EMT-related MPs:\n{show}")
    sig = out[(out.q_BH_41MPs < 0.05)][["name", "reference", "overlap", "p_hypergeom", "overlap_genes"]]
    print("\nAll MP and reference overlaps with BH q < 0.05:\n" + sig.to_string(index=False))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = _load(SRC / "04_tcga_projection" / "07_emt_score_panel.py", "panel")
    mps = gavish_mps()
    pd.DataFrame([{"MP": mp_id(k), "name": k, "emt_related": mp_id(k) in EMT_RELATED, "n_genes": len(v),
                   "genes": ", ".join(v)} for k, v in mps.items()]) \
        .to_csv(OUT / "gavish_mp_gene_lists.tsv", sep="\t", index=False)
    print(f"Gavish et al. 2023: {len(mps)} cancer MPs. EMT-related: "
          + ", ".join(k for k in mps if mp_id(k) in EMT_RELATED))
    tyler_tirosh_availability()
    cover: list = []
    tcga(mps, panel.zmean, cover)
    gse65021(mps, panel.zmean, cover)
    pd.DataFrame(cover).to_csv(OUT / "mp_gene_coverage.tsv", sep="\t", index=False)
    overlaps(mps)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
