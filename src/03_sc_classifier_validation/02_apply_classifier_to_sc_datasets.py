"""Apply the trained classifier to the single-cell datasets and run the sanity checks.

The StandardScaler inside the saved pipeline was fitted on pseudobulk means and is not applied here.
As in the bulk projection, each gene is z-scored on the cohort being scored, classifier genes absent
from the cohort are set to 0 after scaling, and the logistic coefficients are applied. Two levels are
scored:

  cell level        every malignant and fibroblast cell, z-scored within dataset
                    (descriptive, since single cells are far sparser than the
                    100-cell pseudobulks the model was trained on)
  pseudobulk level  mean log1p expression of malignant cells per patient x site
                    and of fibroblasts per patient (groups of at least
                    min_cells_per_pseudobulk cells, 20 by default), z-scored across
                    pseudobulks within dataset. This is the scale the model was
                    trained on and the level at which the checks are decided.

Checks (pseudobulk level unless stated):
  1. fibroblast pseudobulks: fibroblast_stromal_like is the top class in at least 50%
  2. malignant pseudobulks: fibroblast_stromal_like is the top class in at most 30%
  3. rank agreement: Spearman correlation of P(pEMT_high) with the Puram pEMT signature
     (mean z) within each dataset, for malignant pseudobulks (n >= 4) and malignant cells.
     Absolute P levels are not comparable between datasets under within-cohort
     standardisation, so the outgroup table is written for reference only.
  4. Puram nodal status (Puram et al. Table S1, pathological N stage): primary-site
     malignant pseudobulk P(pEMT_high), pEMT specificity and Puram pEMT signature,
     N+ versus N0 patients (one-sided Mann-Whitney, at least 3 patients per group)
  5. Puram lymph node versus primary malignant pseudobulks, paired within patient
     where both exist (two-sided Wilcoxon, descriptive, since metastatic cells may
     have re-epithelialised)

Inputs:  data/processed/single_cell/<dataset>/<dataset>_malignant_fibroblast.h5ad,
         results/multinomial_classifier/tcga_depmap_ready_classifier.pkl,
         results/multinomial_classifier/classifier_genes.txt,
         config/patient_metadata/puram_tableS1_patients.tsv, config/signatures/puram_pemt.txt
Outputs: results/sc_classifier_validation/per_cell_probabilities_<dataset>.tsv,
         pseudobulk_probabilities.tsv, summary_by_cell_type.tsv,
         rank_agreement_with_puram_signature.tsv, outgroup_comparison.tsv,
         puram_primary_patient_scores.tsv, puram_nodal_status_test.tsv,
         puram_ln_vs_primary_paired.tsv, classifier_validation_report.md
Usage:   python src/03_sc_classifier_validation/02_apply_classifier_to_sc_datasets.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import mannwhitneyu, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pemt import load_config, load_registry, load_signatures, project_root  # noqa: E402

cfg = load_config()
ROOT = project_root()
SC_DIR = ROOT / cfg["paths"]["processed_dir"] / "single_cell"
CLF_DIR = ROOT / cfg["paths"]["results_dir"] / "multinomial_classifier"
OUT = ROOT / cfg["paths"]["results_dir"] / "sc_classifier_validation"
OUT.mkdir(parents=True, exist_ok=True)
MIN_SHARED = cfg["classifier"]["min_classifier_genes_in_target"]
MIN_CELLS = int(cfg["sc_classifier_validation"].get("min_cells_per_pseudobulk", 20))
HNSC_ROLES = {"core_discovery", "primary_validation", "training_support"}


def zscore_project(X: pd.DataFrame, genes: list[str], clf) -> pd.DataFrame:
    """Z-score each gene within X, set absent classifier genes to 0 and apply the coefficients.

    X need only hold the genes with a non-zero coefficient, since the others contribute z = 0.
    pEMT_specificity is P(pEMT_high) minus the larger of the other two class probabilities."""
    shared = [g for g in genes if g in X.columns]
    Xs = X[shared].astype(np.float32)
    Z = (Xs - Xs.mean(axis=0)) / Xs.std(axis=0).replace(0, 1.0)
    full = np.zeros((X.shape[0], len(genes)), dtype=np.float32)
    col = {g: i for i, g in enumerate(genes)}
    full[:, [col[g] for g in shared]] = np.nan_to_num(Z.values)
    probs = clf.predict_proba(full)
    p = pd.DataFrame(probs, index=X.index, columns=[f"P_{c}" for c in clf.classes_])
    p["pEMT_specificity"] = p["P_pEMT_high"] - p[["P_epithelial_like", "P_fibroblast_stromal_like"]].max(axis=1)
    p["predicted_class"] = [clf.classes_[i] for i in np.argmax(probs, axis=1)]
    p["n_classifier_genes_shared"] = len(shared)
    return p


def signature_score(X: pd.DataFrame, genes: list[str]) -> pd.Series:
    g = [x for x in genes if x in X.columns]
    S = X[g]
    Z = (S - S.mean(axis=0)) / S.std(axis=0).replace(0, 1.0)
    return Z.mean(axis=1)


def main() -> None:
    model = joblib.load(CLF_DIR / "tcga_depmap_ready_classifier.pkl")
    genes = pd.read_csv(CLF_DIR / "classifier_genes.txt", header=None)[0].tolist()
    clf = model.named_steps["clf"]
    registry = load_registry()
    pemt_sig = load_signatures()["puram_pemt"]
    nz = np.where((clf.coef_ != 0).any(axis=0))[0]
    use_genes = sorted(set(genes[i] for i in nz) | set(pemt_sig))   # genes that can move a prediction, plus the signature

    summaries, pb_all = [], []
    for _, reg in registry.iterrows():
        ds, role = reg["dataset_id"], reg["role"]
        path = SC_DIR / ds / f"{ds}_malignant_fibroblast.h5ad"
        if not path.exists():
            print(f"SKIP {ds}: file missing"); continue
        adata = sc.read_h5ad(path)
        present = [g for g in use_genes if g in adata.var_names]
        X = adata[:, present].to_df().astype(np.float32)
        shared = [g for g in genes if g in adata.var_names]
        if len(shared) < MIN_SHARED:
            print(f"SKIP {ds}: only {len(shared)} classifier genes"); continue
        obs = adata.obs
        ctype = obs["standard_cell_type"].astype(str)
        patient = obs["Patient"].astype(str) if "Patient" in obs else pd.Series("all", index=obs.index)
        site = obs["Site"].astype(str) if "Site" in obs else pd.Series("Primary", index=obs.index)

        # cell level
        pc = zscore_project(X, genes, clf)
        pc["standard_cell_type"] = ctype.values
        pc["patient"] = patient.values
        pc["site"] = site.values
        pc["dataset_id"] = ds
        pc.to_csv(OUT / f"per_cell_probabilities_{ds}.tsv", sep="\t")
        for ct, grp in pc.groupby("standard_cell_type"):
            summaries.append({"dataset_id": ds, "role": role, "level": "cell", "standard_cell_type": ct, "n": len(grp),
                              "mean_P_pEMT_high": grp["P_pEMT_high"].mean(), "mean_P_epithelial_like": grp["P_epithelial_like"].mean(),
                              "mean_P_fibroblast_stromal_like": grp["P_fibroblast_stromal_like"].mean(),
                              "pct_top_class_pEMT_high": (grp["predicted_class"] == "pEMT_high").mean() * 100,
                              "pct_top_class_epithelial_like": (grp["predicted_class"] == "epithelial_like").mean() * 100,
                              "pct_top_class_fibroblast_stromal_like": (grp["predicted_class"] == "fibroblast_stromal_like").mean() * 100})

        # pseudobulk level: malignant per patient x site, fibroblast per patient
        keys = pd.DataFrame({"ctype": ctype.values, "patient": patient.values, "site": site.values}, index=obs.index)
        keys["group"] = np.where(keys["ctype"] == "malignant",
                                 "malignant|" + keys["patient"] + "|" + keys["site"],
                                 "fibroblast|" + keys["patient"] + "|all")
        counts = keys["group"].value_counts()
        keep = counts[counts >= MIN_CELLS].index
        rows = {g: X.loc[keys.index[keys["group"] == g]].mean(axis=0) for g in keep}
        if len(rows) < 3:
            print(f"{ds}: fewer than 3 pseudobulks with >= {MIN_CELLS} cells, pseudobulk level skipped"); continue
        PB = pd.DataFrame(rows).T
        ppb = zscore_project(PB, genes, clf)
        ppb["puram_pemt_signature"] = signature_score(PB, pemt_sig)
        ppb["dataset_id"] = ds; ppb["role"] = role
        ppb["standard_cell_type"] = [g.split("|")[0] for g in ppb.index]
        ppb["patient"] = [g.split("|")[1] for g in ppb.index]
        ppb["site"] = [g.split("|")[2] for g in ppb.index]
        ppb["n_cells"] = [int(counts[g]) for g in ppb.index]
        pb_all.append(ppb)
        for ct, grp in ppb.groupby("standard_cell_type"):
            summaries.append({"dataset_id": ds, "role": role, "level": "pseudobulk", "standard_cell_type": ct, "n": len(grp),
                              "mean_P_pEMT_high": grp["P_pEMT_high"].mean(), "mean_P_epithelial_like": grp["P_epithelial_like"].mean(),
                              "mean_P_fibroblast_stromal_like": grp["P_fibroblast_stromal_like"].mean(),
                              "pct_top_class_pEMT_high": (grp["predicted_class"] == "pEMT_high").mean() * 100,
                              "pct_top_class_epithelial_like": (grp["predicted_class"] == "epithelial_like").mean() * 100,
                              "pct_top_class_fibroblast_stromal_like": (grp["predicted_class"] == "fibroblast_stromal_like").mean() * 100})
        print(f"{ds}: {adata.n_obs} cells, {len(shared)}/{len(genes)} genes, {len(ppb)} pseudobulks")

    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "summary_by_cell_type.tsv", sep="\t", index=False)
    pb = pd.concat(pb_all)
    pb.to_csv(OUT / "pseudobulk_probabilities.tsv", sep="\t")

    lines = ["# Single-cell classifier validation report (pseudobulk level, cohort z-scoring)", ""]
    pbs = summary[summary["level"] == "pseudobulk"]
    # check 1
    fib = pbs[pbs["standard_cell_type"] == "fibroblast"]
    for _, r in fib.iterrows():
        ok = r["pct_top_class_fibroblast_stromal_like"] >= 50
        lines.append(f"- {'PASS' if ok else 'FAIL'} fibroblast specificity, {r['dataset_id']}: {r['pct_top_class_fibroblast_stromal_like']:.0f}% of {int(r['n'])} fibroblast pseudobulks have the stromal class on top")
    # check 2
    mal = pbs[pbs["standard_cell_type"] == "malignant"]
    for _, r in mal.iterrows():
        ok = r["pct_top_class_fibroblast_stromal_like"] <= 30
        lines.append(f"- {'PASS' if ok else 'FAIL'} malignant not stromal, {r['dataset_id']}: {r['pct_top_class_fibroblast_stromal_like']:.0f}% stromal-top, "
                     f"{r['pct_top_class_pEMT_high']:.0f}% pEMT-top, {r['pct_top_class_epithelial_like']:.0f}% epithelial-top ({int(r['n'])} pseudobulks)")
    # check 3 (scale-free): within-cohort standardisation makes absolute P levels
    # incomparable between datasets, so the check is whether the classifier's
    # ranking agrees with the Puram pEMT signature inside each dataset
    from scipy.stats import spearmanr
    agree = []
    for ds, grp in pb[pb["standard_cell_type"] == "malignant"].groupby("dataset_id"):
        if len(grp) >= 4:
            r, pv = spearmanr(grp["P_pEMT_high"], grp["puram_pemt_signature"])
            agree.append({"dataset_id": ds, "role": grp["role"].iloc[0], "level": "pseudobulk", "n": len(grp), "spearman_P_pEMT_high_vs_puram_signature": r, "p": pv})
    for ds in registry["dataset_id"]:
        f = OUT / f"per_cell_probabilities_{ds}.tsv"
        if f.exists():
            cells = pd.read_csv(f, sep="\t", index_col=0)
            cells = cells[cells["standard_cell_type"] == "malignant"]
            adata = sc.read_h5ad(SC_DIR / ds / f"{ds}_malignant_fibroblast.h5ad")
            present = [g for g in pemt_sig if g in adata.var_names]
            sig = signature_score(adata[cells.index, present].to_df().astype(np.float32), present)
            r, pv = spearmanr(cells["P_pEMT_high"], sig.loc[cells.index])
            agree.append({"dataset_id": ds, "role": registry.set_index("dataset_id").loc[ds, "role"], "level": "cell", "n": len(cells), "spearman_P_pEMT_high_vs_puram_signature": r, "p": pv})
    agree = pd.DataFrame(agree)
    agree.to_csv(OUT / "rank_agreement_with_puram_signature.tsv", sep="\t", index=False)
    for _, r in agree.iterrows():
        lines.append(f"- rank agreement, {r['dataset_id']} ({r['level']}, n = {int(r['n'])}): Spearman P(pEMT-high) vs Puram pEMT signature = {r['spearman_P_pEMT_high_vs_puram_signature']:+.2f}")
    outg = mal[["dataset_id", "role", "n", "mean_P_pEMT_high", "pct_top_class_pEMT_high", "pct_top_class_fibroblast_stromal_like"]].sort_values("mean_P_pEMT_high", ascending=False)
    outg.to_csv(OUT / "outgroup_comparison.tsv", sep="\t", index=False)
    lines.append("- outgroup: absolute P levels are not comparable across datasets under within-cohort standardisation, see rank agreement above")

    # check 4: Puram nodal status
    meta = pd.read_csv(ROOT / "config" / "patient_metadata" / "puram_tableS1_patients.tsv", sep="\t")
    id_map = {t: r["designation"] for _, r in meta.iterrows() for t in str(r["tisch_patient_ids"]).split(";")}
    puram = pb[(pb["dataset_id"] == "HNSC_GSE103322") & (pb["standard_cell_type"] == "malignant")].copy()
    puram["designation"] = puram["patient"].map(id_map)
    prim = puram[puram["site"] == "Primary"].groupby("designation").agg(
        P_pEMT_high=("P_pEMT_high", "mean"), pEMT_specificity=("pEMT_specificity", "mean"),
        puram_pemt_signature=("puram_pemt_signature", "mean"), n_cells=("n_cells", "sum")).join(meta.set_index("designation")[["N_stage", "N_positive", "grade", "LVI", "ECE"]])
    prim.to_csv(OUT / "puram_primary_patient_scores.tsv", sep="\t")
    rows = []
    for metric in ("P_pEMT_high", "pEMT_specificity", "puram_pemt_signature"):
        pos = prim.loc[prim["N_positive"] == 1, metric].dropna(); neg = prim.loc[prim["N_positive"] == 0, metric].dropna()
        if len(pos) >= 3 and len(neg) >= 3:
            u, p = mannwhitneyu(pos, neg, alternative="greater")
            rows.append({"metric": metric, "n_N_positive": len(pos), "n_N0": len(neg), "median_N_positive": pos.median(), "median_N0": neg.median(), "U": u, "p_one_sided": p})
    nodal = pd.DataFrame(rows)
    nodal.to_csv(OUT / "puram_nodal_status_test.tsv", sep="\t", index=False)
    for _, r in nodal.iterrows():
        lines.append(f"- nodal status (Puram et al. Table S1), {r['metric']}: N+ median {r['median_N_positive']:.3f} (n = {int(r['n_N_positive'])}) vs N0 {r['median_N0']:.3f} (n = {int(r['n_N0'])}), one-sided p = {r['p_one_sided']:.3f}")

    # check 5: LN vs primary, paired within patient
    piv = puram.pivot_table(index="designation", columns="site", values="P_pEMT_high", aggfunc="mean")
    if {"Primary", "Lymph node"} <= set(piv.columns):
        both = piv.dropna(subset=["Primary", "Lymph node"])
        both.to_csv(OUT / "puram_ln_vs_primary_paired.tsv", sep="\t")
        if len(both) >= 3:
            w, p = wilcoxon(both["Lymph node"], both["Primary"])
            lines.append(f"- lymph node vs primary (paired, {len(both)} patients): median P(pEMT-high) LN {both['Lymph node'].median():.3f} vs primary {both['Primary'].median():.3f}, Wilcoxon two-sided p = {p:.3f} (descriptive)")

    (OUT / "classifier_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
