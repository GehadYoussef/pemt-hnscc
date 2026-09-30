"""Score the two arms of the pEMT partition directly in every bulk cohort.

Elsewhere the malignant arm is measured by the classifier axis (pEMT specificity) and the stromal
arm by the full 100-gene Puram signature. Here each arm is scored from the partition's own gene
sets in results/pemt_partition/puram_pemt_gene_partition.tsv (GSE103322):
  malignant_arm        genes whose malignant share exceeds their fibroblast share
  stromal_arm          genes whose fibroblast share is the larger
  malignant_arm_core   genes with a malignant share of at least 80%
  stromal_arm_core     genes with a fibroblast share of at least 60%
The core sets are the primary definitions and were specified before GSE65021 was examined. The
consensus sets (malignant_consensus, stromal_consensus) were defined post hoc, after the first
results. They hold the genes on the same side of parity by mean expression per cell in all of
GSE103322, GSE150321 and GSE181919, from 02_partition_replication.py. Sensitivity sets drop the
nine Zhou et al. genes, which were derived on GSE65021, from the malignant arm and its core, and
vary the malignant-share threshold (0.6, 0.7, 0.9). Each set is scored as a mean per-gene z-score
within cohort, as for every published signature.

Analyses
  1. TCGA-HNSC: Spearman correlation of each arm score with pEMT specificity, the Puram signature,
     Hallmark EMT, 76GS, MLR-EMT, Puram epithelial differentiation and the eigengenes of modules
     M06 (fibroblast/ECM) and M13 (basal).
  2. Overall survival: univariable and adjusted Cox per SD in TCGA-HNSC, GSE41613 and GSE65858, and
     CPTAC-3 when its RNA-seq matrix exists, pooled by DerSimonian-Laird random effects and by
     fixed effect. Covariates and Cox models come from
     src/04_tcga_projection/14_survival_sensitivity.py.
  3. TCGA sensitivity: the same adjusted models after excluding the patients whose legacy clinical
     record marks systemic ('targeted molecular') therapy, which in HNSCC would include cetuximab,
     and in the patients recorded as not receiving it. The harmonised files carry no drug names, so
     this flag is the closest available proxy for cetuximab exposure.
  4. GSE65021 (platinum plus cetuximab): for each arm score, AUC with bootstrap interval,
     rank-biserial correlation, two-sided Mann-Whitney p and Firth odds ratio per SD. Joint Firth
     models enter the two arms together, and each malignant arm together with an epithelial score.

Inputs:  results/pemt_partition/puram_pemt_gene_partition.tsv
         results/pemt_partition/replication_per_gene.tsv
         data/references/zhou2025_cetuximab_predictive_9genes.txt
         data/processed/tcga_hnsc/tcga_star_tpm_log2_for_projection.tsv
         results/tcga_projection/emt_score_panel_scores.tsv
         results/tcga_projection/external/{GSE41613,GSE65858,CPTAC_HNSCC}_scores.tsv
         data/raw/external_cohorts/{GSE41613,GSE65858,GSE65021}_series_matrix.txt.gz
         data/raw/external_cohorts/{GPL570,GPL10558}.annot.gz
         data/processed/cptac_rnaseq/cptac_hnscc_log2tpm1_symbol_TUMOR.tsv (optional)
         results/tcga_projection/tcga_master_trait_table.tsv
         results/wgcna/module_eigengenes.tsv
         data/raw/tcga_hnsc/clinical/TCGA.HNSC.sampleMap_HNSC_clinicalMatrix
         results/cetuximab_cohort/GSE65021_scores.tsv
Outputs: results/arm_scores/<cohort>_arm_scores.tsv
         results/arm_scores/gene_coverage.tsv
         results/arm_scores/arm_gene_sets.tsv
         results/arm_scores/tcga_arm_correlations.tsv
         results/arm_scores/survival_per_cohort.tsv
         results/arm_scores/survival_pooled.tsv
         results/arm_scores/tcga_systemic_therapy_sensitivity.tsv
         results/arm_scores/gse65021_arm_association.tsv
         results/arm_scores/gse65021_arm_joint_models.tsv
Usage:   python src/10_cetuximab/05_direct_arm_scores.py
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results" / "arm_scores"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "04_tcga_projection"))
from pemt import firth as _firth  # noqa: E402

ARMS = ["malignant_arm", "stromal_arm", "malignant_arm_core", "stromal_arm_core", "malignant_consensus",
        "stromal_consensus"]
COMPARATORS = ["pEMT_specificity", "puram_pemt"]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def consensus() -> dict[str, list[str]]:
    r = pd.read_csv(ROOT / "results/pemt_partition/replication_per_gene.tsv", sep="\t")
    w = r.pivot(index="gene", columns="dataset", values="share_per_cell").dropna()
    return {"malignant_consensus": w.index[(w > 0.5).all(axis=1)].tolist(),
            "stromal_consensus": w.index[(w < 0.5).all(axis=1)].tolist()}


def gene_sets() -> dict[str, list[str]]:
    d = pd.read_csv(ROOT / "results/pemt_partition/puram_pemt_gene_partition.tsv", sep="\t")
    zhou = set((ROOT / "data/references/zhou2025_cetuximab_predictive_9genes.txt").read_text().split())
    mal = d.loc[d["arm"] == "malignant-expressed", "gene"].tolist()
    core = d.loc[d["share_malignant"] >= 0.8, "gene"].tolist()
    return {"malignant_arm": d.loc[d["arm"] == "malignant-expressed", "gene"].tolist(),
            "stromal_arm": d.loc[d["arm"] == "CAF-expressed", "gene"].tolist(),
            "malignant_arm_core": d.loc[d["share_malignant"] >= 0.8, "gene"].tolist(),
            "stromal_arm_core": d.loc[d["share_CAF"] >= 0.6, "gene"].tolist(),
            # sensitivity: Zhou et al. derived nine genes on GSE65021 itself, so drop them
            "malignant_arm_noZhou": [g for g in mal if g not in zhou],
            "malignant_arm_core_noZhou": [g for g in core if g not in zhou],
            # consensus arms (post hoc): same side of parity by mean expression per cell, on the
            # linear scale, in all of GSE103322, GSE150321 and GSE181919 (02_partition_replication.py)
            **consensus(),
            # sensitivity: other malignant-share thresholds besides the 80% core
            **{f"malignant_share_ge_{t:.1f}": d.loc[d["share_malignant"] >= t, "gene"].tolist()
               for t in (0.6, 0.7, 0.9)}}


def expression():
    """Yield (cohort, genes x samples, score panel module). CPTAC-3 only when its matrix exists."""
    panel = _load(ROOT / "src/04_tcga_projection/07_emt_score_panel.py", "panel")
    tp = ROOT / "results" / "tcga_projection"
    ref = pd.read_csv(tp / "emt_score_panel_scores.tsv", sep="\t", index_col=0)
    e = pd.read_csv(ROOT / "data/processed/tcga_hnsc/tcga_star_tpm_log2_for_projection.tsv", sep="\t", index_col=0)
    yield "TCGA-HNSC", e[ref.index.intersection(e.columns)], panel
    ext = _load(ROOT / "src/04_tcga_projection/09_external_cohorts.py", "ext")
    raw = ROOT / "data" / "raw" / "external_cohorts"
    for gse, annot in (("GSE41613", "GPL570.annot.gz"), ("GSE65858", "GPL10558.annot.gz")):
        x, _ = ext.read_series_matrix(raw / f"{gse}_series_matrix.txt.gz")
        e = ext.collapse(x, ext.read_annotation(raw / annot))
        ref = pd.read_csv(tp / "external" / f"{gse}_scores.tsv", sep="\t", index_col=0)
        yield gse, e[ref.index.intersection(e.columns)], panel
    cp = ROOT_CPTAC = ROOT / "data" / "processed" / "cptac_rnaseq" / "cptac_hnscc_log2tpm1_symbol_TUMOR.tsv"
    if cp.exists():   # written by src/04_tcga_projection/00_prepare_cptac_rnaseq.py
        ref = pd.read_csv(tp / "external" / "CPTAC_HNSCC_scores.tsv", sep="	", index_col=0)
        e = pd.read_csv(cp, sep="	", index_col=0)
        yield "CPTAC_HNSCC", e[ref.index.intersection(e.columns)], panel
    cx = _load(ROOT / "src" / "10_cetuximab" / "01_cetuximab_cohort.py", "cx")
    x, _ = cx.read_series_matrix(cx.DATA / "GSE65021_series_matrix.txt.gz")
    yield "GSE65021", cx.collapse(x, cx.read_annotation(cx.DATA / "GPL10558.annot.gz")), panel


def main() -> None:
    sets = gene_sets()
    cover, scores = [], {}
    for cohort, e, panel in expression():
        s = pd.DataFrame(index=e.columns)
        for k, genes in sets.items():
            s[k], found, tot = panel.zmean(e, genes)
            cover.append({"cohort": cohort, "score": k, "genes_present": found, "genes_in_set": tot})
        scores[cohort] = s
        s.to_csv(OUT / f"{cohort}_arm_scores.tsv", sep="\t")
    pd.DataFrame(cover).to_csv(OUT / "gene_coverage.tsv", sep="\t", index=False)
    pd.DataFrame([{"score": k, "n_genes": len(v), "genes": ", ".join(v)} for k, v in sets.items()]) \
        .to_csv(OUT / "arm_gene_sets.tsv", sep="\t", index=False)
    print({k: len(v) for k, v in sets.items()})

    tcga_correlations(scores["TCGA-HNSC"])
    survival(scores)
    cetuximab(scores["GSE65021"])


def tcga_correlations(arms: pd.DataFrame) -> None:
    tp = ROOT / "results" / "tcga_projection"
    ref = pd.read_csv(tp / "emt_score_panel_scores.tsv", sep="\t", index_col=0)
    me = pd.read_csv(ROOT / "results/wgcna/module_eigengenes.tsv", sep="\t", index_col=0)[["ME_M06", "ME_M13"]]
    d = arms.join(ref).join(me)
    cols = ["pEMT_specificity", "puram_pemt", "hallmark_EMT", "GS76", "MLR_mu", "puram_epi_dif_1",
            "ME_M06", "ME_M13"] + ARMS
    rows = []
    for a in ARMS:
        rows.append({"score": a, **{c: spearmanr(d[a], d[c], nan_policy="omit")[0] for c in cols}})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "tcga_arm_correlations.tsv", sep="\t", index=False)
    print("\nTCGA-HNSC Spearman correlations:\n" + out.round(2).to_string(index=False))


def survival(scores: dict[str, pd.DataFrame]) -> None:
    s14 = _load(ROOT / "src/04_tcga_projection/14_survival_sensitivity.py", "s14")
    coh = s14.cohorts()
    use = {"TCGA-HNSC": "TCGA-HNSC", "GSE41613": "GSE41613", "GSE65858": "GSE65858"}
    if "CPTAC_HNSCC" in scores:
        use["CPTAC_HNSCC"] = "CPTAC_HNSCC"
    rows, pooled = [], []
    for sc in ARMS + COMPARATORS + ["hallmark_EMT", "MLR_mu"]:
        for adj in ("univariable", "adjusted"):
            est = {}
            for cid in use:
                df, covs = coh[cid]
                d = df.join(scores[cid][ARMS], how="left") if sc in ARMS else df
                r = s14.cox_direct(d, sc, covs if adj == "adjusted" else [])
                if r:
                    est[cid] = r
                    rows.append({"score": sc, "model": adj, "cohort": cid, **r})
            for m in ("random", "fixed"):
                pooled.append({"score": sc, "model": adj, "pooling": m,
                               **s14.pool([v["log_HR"] for v in est.values()], [v["var"] for v in est.values()], m)})
    pd.DataFrame(rows).to_csv(OUT / "survival_per_cohort.tsv", sep="\t", index=False)
    p = pd.DataFrame(pooled)
    p.to_csv(OUT / "survival_pooled.tsv", sep="\t", index=False)
    show = p[(p["model"] == "adjusted") & (p["pooling"] == "random")]
    print(f"\nAdjusted HR per SD, random effects over {', '.join(use)}:")
    print(show[["score", "HR", "CI_low", "CI_up", "p_value", "I2_percent"]].round(3).to_string(index=False))
    per = pd.DataFrame(rows)
    print(per[per["model"] == "adjusted"].pivot(index="score", columns="cohort", values="HR").round(2))

    # TCGA without patients recorded as receiving systemic ('targeted molecular') therapy
    clin = pd.read_csv(ROOT / "data/raw/tcga_hnsc/clinical/TCGA.HNSC.sampleMap_HNSC_clinicalMatrix", sep="\t")
    tmt = clin.set_index("sampleID")["targeted_molecular_therapy"]
    df, covs = coh["TCGA-HNSC"]
    df = df.join(scores["TCGA-HNSC"][ARMS], how="left")
    flag = pd.Series(df.index.str[:15], index=df.index).map(tmt)
    srows = []
    for label, sub in (("all", df), ("excluding targeted_molecular_therapy = YES", df[flag != "YES"]),
                       ("targeted_molecular_therapy = NO only", df[flag == "NO"])):
        for sc in ARMS + COMPARATORS:
            r = s14.cox_direct(sub, sc, covs)
            if r:
                srows.append({"subset": label, "score": sc, **r})
    sens = pd.DataFrame(srows)
    sens.to_csv(OUT / "tcga_systemic_therapy_sensitivity.tsv", sep="\t", index=False)
    print(f"\nTCGA targeted_molecular_therapy among analysed primaries: {flag.value_counts(dropna=False).to_dict()}")
    print(sens[["subset", "score", "n", "events", "HR", "p"]].round(3).to_string(index=False))


def cetuximab(arms: pd.DataFrame) -> None:
    cx = _load(ROOT / "src" / "10_cetuximab" / "01_cetuximab_cohort.py", "cx2")
    df = pd.read_csv(ROOT / "results/cetuximab_cohort/GSE65021_scores.tsv", sep="\t", index_col=0).join(arms)
    y = df["long_pfs"].to_numpy(int)
    z = lambda s: (s - s.mean()) / s.std()  # noqa: E731
    rows = []
    extra = ["malignant_arm_noZhou", "malignant_arm_core_noZhou", "malignant_share_ge_0.6",
             "malignant_share_ge_0.7", "malignant_share_ge_0.9"]
    for sc in ARMS + extra + COMPARATORS:
        a, b = df.loc[df.long_pfs == 1, sc], df.loc[df.long_pfs == 0, sc]
        auc, lo, hi = cx.bootstrap_auc(y, df[sc].values)
        X = np.column_stack([np.ones(len(df)), z(df[sc])])
        f = _firth.firth(y, X, ["const", sc])[sc]
        rows.append({"score": sc, "AUC": auc, "AUC_low": lo, "AUC_up": hi,
                     "rank_biserial": 2 * mannwhitneyu(a, b).statistic / (len(a) * len(b)) - 1,
                     "p_two_sided": mannwhitneyu(a, b).pvalue,
                     "OR_firth": f[0], "OR_low": f[1], "OR_up": f[2], "p_firth": f[3],
                     "rho_axis": spearmanr(df[sc], df["pEMT_specificity"])[0],
                     "rho_puram_pemt": spearmanr(df[sc], df["puram_pemt"])[0]})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "gse65021_arm_association.tsv", sep="\t", index=False)
    print("\nGSE65021, long vs short PFS on platinum plus cetuximab:\n" + out.round(3).to_string(index=False))

    jrows = []
    for pair in (("malignant_arm", "stromal_arm"), ("malignant_arm_core", "stromal_arm_core"),
                 ("pEMT_specificity", "stromal_arm"),
                 # malignant arm against epithelial differentiation scores
                 ("malignant_arm", "puram_epi_dif_1"), ("malignant_arm_core", "puram_epi_dif_1"),
                 ("malignant_arm_core", "GS76")):
        X = np.column_stack([np.ones(len(df))] + [z(df[p]) for p in pair])
        f = _firth.firth(y, X, ["const", *pair])
        jrows.append({"terms": " + ".join(pair), "rho": spearmanr(df[pair[0]], df[pair[1]])[0],
                      **{f"OR_{k}": f[p][0] for k, p in zip("12", pair)},
                      **{f"p_{k}": f[p][3] for k, p in zip("12", pair)}})
    j = pd.DataFrame(jrows)
    j.to_csv(OUT / "gse65021_arm_joint_models.tsv", sep="\t", index=False)
    print("\nJoint Firth models:\n" + j.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
