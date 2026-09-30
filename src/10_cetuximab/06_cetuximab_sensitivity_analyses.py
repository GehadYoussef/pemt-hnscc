"""Run the comparison, multiplicity, transportability and MLR-EMT state analyses for GSE65021.

These analyses were added post hoc, after the primary results of 01_cetuximab_cohort.py and
05_direct_arm_scores.py.

1. Paired comparison. Paired bootstrap (2,000 resamples of patients) of the AUC difference between
   each primary measure (the classifier axis, pEMT specificity, and the malignant core) and 76GS,
   Puram epithelial differentiation, the Puram pEMT signature and the Zhou nine-gene set.
   The bootstrap p-value is twice the smaller tail proportion of the differences.
2. Multiplicity. Benjamini-Hochberg over one family: every independently derived score from
   01_cetuximab_cohort.py plus the four arm scores specified before the cohort was examined
   (malignant and stromal cores, malignant and stromal total-share arms). Two-sided Mann-Whitney p.
3. Transportability. Every score is standardised within the cohort, so a patient's value depends on
   the other 39 patients, who were selected as outcome extremes.
   a. Leave-one-out standardisation: each patient is scored with gene means and SDs from the other
      39 patients only (classifier axis and malignant core).
   b. External reference: each patient is scored alone against gene means and SDs from GSE65858
      (Illumina HT-12 v4, the whole-genome version of the same array family), with no information
      from GSE65021.
   Reported: AUC under each scheme and Spearman agreement with the within-cohort score.
4. MLR-EMT read categorically. MLR-EMT is an ordinal three-state model with partial EMT as its
   middle (hybrid) state, so the states are analysed alongside the 0-2 score. For GSE65021 and TCGA:
   state counts, and for each state probability and the score, AUC (GSE65021), Firth odds ratio per
   SD, and Spearman correlation with pEMT specificity, the malignant core, 76GS and the Puram pEMT
   signature. In GSE65021, hybrid and mesenchymal state counts are compared between outcome groups
   by Fisher's exact test.

Inputs:  results/cetuximab_cohort/GSE65021_scores.tsv
         results/cetuximab_cohort/GSE65021_pfs_association.tsv
         results/arm_scores/GSE65021_arm_scores.tsv
         results/arm_scores/arm_gene_sets.tsv
         data/raw/external_cohorts/{GSE65021,GSE65858}_series_matrix.txt.gz
         data/raw/external_cohorts/GPL10558.annot.gz
         results/multinomial_classifier/tcga_depmap_ready_classifier.pkl
         results/multinomial_classifier/classifier_genes.txt
         results/mlr_faithful/{GSE65021,TCGA}_mlr_faithful.tsv
Outputs: results/cetuximab_cohort/GSE65021_sensitivity_paired_auc.tsv
         results/cetuximab_cohort/GSE65021_sensitivity_fdr_family.tsv
         results/cetuximab_cohort/GSE65021_sensitivity_transportability.tsv
         results/cetuximab_cohort/GSE65021_sensitivity_transportability_scores.tsv
         results/cetuximab_cohort/GSE65021_sensitivity_mlr_states.tsv
         results/cetuximab_cohort/GSE65021_sensitivity_mlr_state_counts.tsv
Usage:   python src/10_cetuximab/06_cetuximab_sensitivity_analyses.py
         Run after 01_cetuximab_cohort.py, 02_mlr_emt_states.py and 05_direct_arm_scores.py.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CX = ROOT / "results" / "cetuximab_cohort"
N_BOOT, SEED = 2000, 20260924


def _load(path: Path, name: str):
    sys.path.insert(0, str(path.parent))
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def scores() -> pd.DataFrame:
    df = pd.read_csv(CX / "GSE65021_scores.tsv", sep="\t", index_col=0)
    arms = pd.read_csv(ROOT / "results" / "arm_scores" / "GSE65021_arm_scores.tsv", sep="\t", index_col=0)
    return df.join(arms)


def paired(df: pd.DataFrame) -> pd.DataFrame:
    y = df["long_pfs"].to_numpy(int)
    rng = np.random.default_rng(SEED)
    boots = [b for b in (rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)) if 0 < y[b].sum() < len(b)]
    rows = []
    for primary in ("pEMT_specificity", "malignant_arm_core"):
        for other in ("GS76", "puram_epi_dif_1", "puram_pemt", "zhou2025_predictive"):
            a, o = df[primary].to_numpy(), df[other].to_numpy()
            d = [roc_auc_score(y[b], a[b]) - roc_auc_score(y[b], o[b]) for b in boots]
            obs = roc_auc_score(y, a) - roc_auc_score(y, o)
            rows.append({"primary": primary, "comparator": other,
                         "AUC_primary": roc_auc_score(y, a), "AUC_comparator": roc_auc_score(y, o),
                         "difference": obs, "diff_low": np.percentile(d, 2.5), "diff_up": np.percentile(d, 97.5),
                         "p_bootstrap_two_sided": min(1.0, 2 * min(np.mean(np.array(d) <= 0), np.mean(np.array(d) >= 0)))})
    out = pd.DataFrame(rows)
    out.to_csv(CX / "GSE65021_sensitivity_paired_auc.tsv", sep="\t", index=False)
    print("Paired AUC differences:\n" + out.round(3).to_string(index=False))
    return out


def multiplicity(df: pd.DataFrame) -> pd.DataFrame:
    s10 = _load(ROOT / "src" / "10_cetuximab" / "01_cetuximab_cohort.py", "s10")
    assoc = pd.read_csv(CX / "GSE65021_pfs_association.tsv", sep="\t")
    family = assoc.loc[assoc["independent_of_cohort"], ["score", "label"]].values.tolist()
    family += [["malignant_arm_core", "Malignant core (12 genes)"], ["stromal_arm_core", "Stromal core"],
               ["malignant_arm", "Malignant arm, total share"], ["stromal_arm", "Stromal arm, total share"]]
    rows = []
    for sc, lab in family:
        a, b = df.loc[df.long_pfs == 1, sc], df.loc[df.long_pfs == 0, sc]
        rows.append({"score": sc, "label": lab, "AUC": roc_auc_score(df["long_pfs"], df[sc]),
                     "p_two_sided": mannwhitneyu(a, b).pvalue})
    out = pd.DataFrame(rows)
    out["q_BH"] = s10.bh(out["p_two_sided"].to_numpy())
    out = out.sort_values("p_two_sided")
    out.to_csv(CX / "GSE65021_sensitivity_fdr_family.tsv", sep="\t", index=False)
    print(f"\nFDR over {len(out)} scores:\n" + out.round(4).to_string(index=False))
    return out


def transport(df: pd.DataFrame) -> pd.DataFrame:
    import joblib

    s10 = _load(ROOT / "src" / "10_cetuximab" / "01_cetuximab_cohort.py", "s10b")
    expr, _ = s10.read_series_matrix(s10.DATA / "GSE65021_series_matrix.txt.gz")
    e = s10.collapse(expr, s10.read_annotation(s10.DATA / "GPL10558.annot.gz"))[df.index]
    ext = _load(ROOT / "src/04_tcga_projection/09_external_cohorts.py", "ext")
    raw = ROOT / "data" / "raw" / "external_cohorts"
    x, _ = ext.read_series_matrix(raw / "GSE65858_series_matrix.txt.gz")
    ref = ext.collapse(x, ext.read_annotation(raw / "GPL10558.annot.gz"))

    clf_dir = ROOT / "results" / "multinomial_classifier"
    model = joblib.load(clf_dir / "tcga_depmap_ready_classifier.pkl").named_steps["clf"]
    cgenes = pd.read_csv(clf_dir / "classifier_genes.txt", header=None)[0].tolist()
    core = pd.read_csv(ROOT / "results" / "arm_scores" / "arm_gene_sets.tsv", sep="\t") \
        .set_index("score").loc["malignant_arm_core", "genes"].split(", ")

    def axis_from_z(Z: pd.DataFrame) -> np.ndarray:
        Zf = pd.DataFrame(0.0, index=Z.index, columns=cgenes)
        shared = [g for g in cgenes if g in Z.columns]
        Zf[shared] = Z[shared]
        p = model.predict_proba(np.nan_to_num(Zf.values))
        cls = list(model.classes_)
        hi = p[:, cls.index("pEMT_high")]
        return hi - np.maximum(p[:, cls.index("epithelial_like")], p[:, cls.index("fibroblast_stromal_like")])

    X = e.T                                                     # patients x genes
    genes = [g for g in set(cgenes) | set(core) if g in X.columns]
    X = X[genes]
    within = (X - X.mean()) / X.std().replace(0, 1)
    loo = pd.DataFrame(index=X.index, columns=genes, dtype=float)
    for i in X.index:
        rest = X.drop(index=i)
        loo.loc[i] = (X.loc[i] - rest.mean()) / rest.std().replace(0, 1)
    R = ref.T.reindex(columns=genes)
    external = (X - R.mean()) / R.std().replace(0, 1)

    y = df["long_pfs"].to_numpy(int)
    rows, out_scores = [], pd.DataFrame(index=X.index)
    for scheme, Z in (("within cohort", within), ("leave one out", loo), ("external reference (GSE65858)", external)):
        Z = Z.fillna(0.0)
        ax = axis_from_z(Z)
        cs = Z[[g for g in core if g in Z.columns]].mean(axis=1).to_numpy()
        out_scores[f"axis_{scheme}"] = ax
        out_scores[f"core_{scheme}"] = cs
        for name, v, ref_col in (("pEMT specificity", ax, "pEMT_specificity"), ("malignant core", cs, "malignant_arm_core")):
            rows.append({"score": name, "scheme": scheme, "AUC": roc_auc_score(y, v),
                         "rho_with_published_score": spearmanr(v, df[ref_col])[0]})
    out = pd.DataFrame(rows)
    out.to_csv(CX / "GSE65021_sensitivity_transportability.tsv", sep="\t", index=False)
    out_scores.to_csv(CX / "GSE65021_sensitivity_transportability_scores.tsv", sep="\t")
    print("\nTransportability:\n" + out.round(3).to_string(index=False))
    return out


def mlr_states(df: pd.DataFrame) -> pd.DataFrame:
    sys.path.insert(0, str(HERE.parent))
    from pemt import firth as _firth

    fa = ROOT / "results" / "mlr_faithful"
    g = pd.read_csv(fa / "GSE65021_mlr_faithful.tsv", sep=chr(9), index_col=0)[["P_E", "P_H", "P_M", "MLR_faithful", "state"]]
    d = df.join(g)
    y = d["long_pfs"].to_numpy(int)
    z = lambda s: (s - s.mean()) / s.std()  # noqa: E731
    rows = []
    for c, lab in (("P_H", "P(hybrid)"), ("P_E", "P(epithelial)"), ("P_M", "P(mesenchymal)"), ("MLR_faithful", "MLR-EMT score")):
        f = _firth.firth(y, np.column_stack([np.ones(len(d)), z(d[c])]), ["const", c])[c]
        rows.append({"cohort": "GSE65021", "quantity": lab, "min": d[c].min(), "median": d[c].median(), "max": d[c].max(),
                     "AUC_long_pfs": roc_auc_score(y, d[c]),
                     "p_two_sided": mannwhitneyu(d.loc[y == 1, c], d.loc[y == 0, c]).pvalue,
                     "OR_firth": f[0], "OR_low": f[1], "OR_up": f[2], "p_firth": f[3],
                     "rho_pEMT_specificity": spearmanr(d[c], d["pEMT_specificity"])[0],
                     "rho_malignant_core": spearmanr(d[c], d["malignant_arm_core"])[0],
                     "rho_76GS": spearmanr(d[c], d["GS76"])[0],
                     "rho_puram_pemt": spearmanr(d[c], d["puram_pemt"])[0]})
    tc = pd.read_csv(fa / "TCGA_mlr_faithful.tsv", sep=chr(9), index_col=0)
    for c, lab in (("P_H", "P(hybrid)"), ("MLR_faithful", "MLR-EMT score")):
        rows.append({"cohort": "TCGA-HNSC", "quantity": lab, "min": tc[c].min(), "median": tc[c].median(), "max": tc[c].max(),
                     "rho_pEMT_specificity": spearmanr(tc[c], tc["pEMT_specificity"])[0],
                     "rho_76GS": spearmanr(tc[c], tc["GS76"])[0],
                     "rho_puram_pemt": spearmanr(tc[c], tc["puram_pemt"])[0]})
    out = pd.DataFrame(rows)
    states = pd.crosstab(d["state"], d["long_pfs"]).reindex(list("EHM"), fill_value=0)
    odds, p = fisher_exact(states.loc[["H", "M"]].values)
    out.attrs["states"] = states
    out.to_csv(CX / "GSE65021_sensitivity_mlr_states.tsv", sep=chr(9), index=False)
    states.assign(fisher_H_vs_M_p=p).to_csv(CX / "GSE65021_sensitivity_mlr_state_counts.tsv", sep=chr(9))
    nl = chr(10)
    print(nl + "MLR-EMT read categorically:" + nl + out.round(3).to_string(index=False))
    print(f"GSE65021 states by outcome (columns long_pfs 0/1):{nl}{states}{nl}Fisher H vs M p = {p:.3f}, "
          f"TCGA states {tc['state'].value_counts().to_dict()}")
    return out


if __name__ == "__main__":
    d = scores()
    paired(d)
    multiplicity(d)
    transport(d)
    mlr_states(d)
