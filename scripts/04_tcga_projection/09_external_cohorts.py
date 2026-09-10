"""External validation of the adjusted survival effect in two public bulk HNSCC cohorts.

Added 9 Sep 2026. Cohorts (GEO series matrices in data/raw/external_cohorts/):
  GSE65858  Wichmann et al. 2015, 270 HNSCC (oral cavity, oropharynx, hypopharynx, larynx),
            Illumina HT-12 v4 (GPL10558); overall and progression-free survival, UICC stage,
            T and N category, site, age, smoking, HPV16 DNA/RNA status
  GSE41613  Lohavanichbutr et al. 2013, 97 HPV-negative oral cavity SCC, Affymetrix U133 Plus 2
            (GPL570); overall survival (vital status, follow-up months), stage I/II vs III/IV, age band

Procedure, identical to the TCGA projection: probes are collapsed to gene symbols (highest mean
probe per gene), values are log2 if not already, the classifier genes are z-scored within the
cohort (missing genes set to 0) and the elastic-net coefficients applied; pEMT specificity is
P(pEMT-high) - max(P(epithelial-like), P(fibroblast/stromal-like)). The EMT score panel is
computed on the same samples. Cox models: univariable and adjusted for the covariates each cohort
has (GSE65858: stage, age, site, HPV16 DNA/RNA; GSE41613: stage, age band), HR per SD for every
score; Kaplan-Meier by tertile of pEMT specificity.

Outputs (results/tcga_projection/external/): <cohort>_scores.tsv, <cohort>_cox.tsv,
  external_cox_summary.tsv
Figure: Figure_S6_external_cohorts (KM by tertile and adjusted HR per SD per score, per cohort)
"""

import gzip
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib import load_config, load_signatures, project_root  # noqa: E402
from _lib.published import load_published  # noqa: E402
from _lib.figstyle import apply_style, save_figure, PALETTE, TERTILE_COLOURS, mm  # noqa: E402
from _lib.survival import fit_cox  # noqa: E402

cfg = load_config()
ROOT = project_root()
RAW = ROOT / "data" / "raw" / "external_cohorts"
CLF_DIR = ROOT / cfg["paths"]["results_dir"] / "multinomial_classifier"
OUT = ROOT / cfg["paths"]["results_dir"] / "tcga_projection" / "external"
FIG = ROOT / "manuscript" / "figures"
REF = ROOT / cfg["paths"]["references_dir"] / "emt_scores"
OUT.mkdir(parents=True, exist_ok=True)
apply_style()
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module  # noqa: E402
_panel = import_module("07_emt_score_panel")  # score functions (zmean, score_76gs, score_ks, score_mlr)

COHORTS = {
    "GSE65858": {"matrix": "GSE65858_series_matrix.txt.gz", "annot": "GPL10558.annot.gz", "label": "GSE65858, mixed-site HNSCC"},
    "GSE41613": {"matrix": "GSE41613_series_matrix.txt.gz", "annot": "GPL570.annot.gz", "label": "GSE41613, HPV-negative OSCC"},
}
SCORE_LABELS = {"pEMT_specificity": "pEMT specificity", "P_pEMT_high": "P(pEMT-high)", "GS76": "76GS", "KS": "KS", "hallmark_EMT": "Hallmark EMT",
                "puram_pemt": "Puram pEMT", "puram_epi_dif_1": "Puram epithelial", "MLR_mu": "MLR"}


def read_series_matrix(path: Path):
    chars, ids, rows = [], None, []
    with gzip.open(path, "rt") as fh:
        in_table = False
        for line in fh:
            if line.startswith("!Sample_geo_accession"):
                ids = [x.strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                chars.append([x.strip('"') for x in line.rstrip("\n").split("\t")[1:]])
            elif line.startswith("!series_matrix_table_begin"):
                in_table = True
            elif line.startswith("!series_matrix_table_end"):
                break
            elif in_table:
                rows.append(line.rstrip("\n").split("\t"))
    header = [x.strip('"') for x in rows[0]]
    expr = pd.DataFrame([r[1:] for r in rows[1:]], index=[r[0].strip('"') for r in rows[1:]], columns=header[1:]).apply(pd.to_numeric, errors="coerce")
    meta = pd.DataFrame(index=ids)
    for row in chars:
        for sid, v in zip(ids, row):
            if ":" in v:
                k, val = v.split(":", 1)
                meta.loc[sid, k.strip().lower()] = val.strip()
    return expr, meta


def read_annotation(path: Path) -> pd.Series:
    with gzip.open(path, "rt", errors="replace") as fh:
        lines = fh.readlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("ID\t"))
    end = next((i for i, l in enumerate(lines) if l.startswith("!platform_table_end")), len(lines))
    tab = pd.read_csv(pd.io.common.StringIO("".join(lines[start:end])), sep="\t", dtype=str)
    sym = tab.set_index("ID")["Gene symbol"].dropna()
    sym = sym[~sym.str.contains("///")]
    return sym


def collapse(expr: pd.DataFrame, sym: pd.Series) -> pd.DataFrame:
    if expr.max().max() > 100:
        expr = np.log2(expr.clip(lower=0) + 1)
    e = expr.loc[expr.index.intersection(sym.index)].copy()
    e["gene"] = sym.loc[e.index].values
    e["_mean"] = e.drop(columns="gene").mean(axis=1)
    e = e.sort_values("_mean", ascending=False).drop_duplicates("gene").set_index("gene").drop(columns="_mean")
    return e


def project(expr_genes_x_samples: pd.DataFrame):
    model = joblib.load(CLF_DIR / "tcga_depmap_ready_classifier.pkl")
    genes = pd.read_csv(CLF_DIR / "classifier_genes.txt", header=None)[0].tolist()
    clf = model.named_steps["clf"]
    X = expr_genes_x_samples.T
    shared = [g for g in genes if g in X.columns]
    X_full = pd.DataFrame(0.0, index=X.index, columns=genes)
    X_full[shared] = X[shared]
    Z = ((X_full - X_full.mean(axis=0)) / X_full.std(axis=0).replace(0, 1.0)).fillna(0.0)
    probs = clf.predict_proba(np.nan_to_num(Z.values))
    out = pd.DataFrame(probs, index=X.index, columns=[f"P_{c}" for c in clf.classes_])
    out["pEMT_specificity"] = out["P_pEMT_high"] - out[["P_epithelial_like", "P_fibroblast_stromal_like"]].max(axis=1)
    nonzero = (np.abs(clf.coef_).sum(axis=0) > 0)
    return out, len(shared), int(sum(1 for g, nz in zip(genes, nonzero) if nz and g in X.columns)), int(nonzero.sum())


def clinical_GSE65858(meta: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame(index=meta.index)
    d["OS_time"] = pd.to_numeric(meta["os"], errors="coerce")
    d["OS_event"] = (meta["os_event"].str.upper() == "TRUE").astype(int)
    d["age"] = pd.to_numeric(meta["age"], errors="coerce")
    stage = meta["uicc_stage"].str.upper().str.replace("A", "").str.replace("B", "").str.replace("C", "")
    d["stage_num"] = stage.map({"I": 1, "II": 2, "III": 3, "IV": 4})
    site = meta["tumor_site"].str.lower()
    oral = site.str.contains("oral|cavum|oris")
    lar = site.str.contains("laryn|hypophar")
    d["site_oral_cavity"] = oral.astype(int)
    d["site_larynx_hypopharynx"] = lar.astype(int)
    d["site_group"] = np.select([oral, lar, site.str.contains("oropharynx")], ["oral_cavity", "larynx_hypopharynx", "oropharynx"], default="unknown")
    d.loc[site.isin(["na", "nan"]) | site.isna(), ["site_oral_cavity", "site_larynx_hypopharynx"]] = np.nan
    d["hpv_positive"] = (meta["hpv16_dna_rna"] == "DNA+RNA+").astype(int)
    d["hpv_status"] = np.where(meta["hpv16_dna_rna"] == "DNA+RNA+", "HPV+", np.where(meta["hpv16_dna_rna"].isin(["DNA-RNA-", "DNA+RNA-", "NA"]) | meta["hpv_dna"].eq("Negative"), "HPV-", "unknown"))
    d["n_positive"] = (meta["n_category"].astype(str).str.strip() != "0").astype(int)
    d.loc[meta["n_category"].isna(), "n_positive"] = np.nan
    d["primary"] = meta["tumor_type"].str.lower().eq("primary")
    # treatment modality (mono = single modality, multi = multimodal). GSE65858 is the only cohort here
    # with mixed treatment, and it is the only one where the tumour-intrinsic axis is null, so this is
    # the variable the survival sensitivity analysis needs.
    if "treatment" in meta.columns:
        d["treatment"] = meta["treatment"].astype(str).str.strip().str.lower().replace({"nan": np.nan, "na": np.nan})
    # the cohort carries its own consensus expression-subtype calls, an independent subtype label set
    if "consensus_cluster" in meta.columns:
        cc = meta["consensus_cluster"].astype(str)
        d["consensus_cluster"] = cc
        d["consensus_subtype"] = cc.str.extract(r"^(Basal|Classical|Atypical|Mesenchymal|Inflamed|Immune)", expand=False)
    return d


def clinical_GSE41613(meta: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame(index=meta.index)
    d["OS_time"] = pd.to_numeric(meta["fu time"], errors="coerce") * 30.44
    d["OS_event"] = meta["vital"].str.lower().str.startswith("dead").astype(int)
    d["stage_num"] = meta["tumor stage"].map({"I/II": 1.5, "III/IV": 3.5})
    d["age"] = meta["age"].map({"40-49": 45, "50-59": 55, "60-88": 70, "<40": 35, "30-39": 35}).astype(float)
    d["hpv_status"] = "HPV-"
    d["primary"] = True
    return d


def cox_per_sd(df: pd.DataFrame, score: str, covariates: list[str]):
    cols = ["OS_time", "OS_event", score] + covariates
    sub = df.dropna(subset=cols).copy()
    sub = sub[sub["OS_time"] > 0]
    sub[score] = (sub[score] - sub[score].mean()) / sub[score].std()
    covariates = [c for c in covariates if sub[c].nunique() > 1]
    _, r = fit_cox(sub[["OS_time", "OS_event", score] + covariates])
    return r.loc[score], len(sub), int(sub["OS_event"].sum())


def main() -> None:
    sigs = load_signatures()
    hm = [l.strip() for l in (REF / "hallmark_emt_genes.txt").read_text().splitlines() if l.strip()]
    summary, km_data = [], {}
    for cid, spec in COHORTS.items():
        expr, meta = read_series_matrix(RAW / spec["matrix"])
        sym = read_annotation(RAW / spec["annot"])
        e = collapse(expr, sym)
        probs, n_shared, n_nz_shared, n_nz = project(e)
        scores = probs.copy()
        scores["GS76"], _ = _panel.score_76gs(e)
        scores["KS"], _, _ = _panel.score_ks(e)
        scores["hallmark_EMT"], _, _ = _panel.zmean(e, hm)
        scores["puram_pemt"], _, _ = _panel.zmean(e, sigs["puram_pemt"])
        scores["puram_epi_dif_1"], _, _ = _panel.zmean(e, sigs["puram_epi_dif_1"])
        scores["MLR_mu"] = _panel.score_mlr(e)
        for key, genes in load_published(ROOT / cfg["paths"]["references_dir"]).items():
            scores[key], n_found, n_tot = _panel.zmean(e, genes)
            print(f"  [{cid}] {key}: {n_found}/{n_tot} genes present")
        clin = clinical_GSE65858(meta) if cid == "GSE65858" else clinical_GSE41613(meta)
        df = clin.join(scores)
        df = df[df["primary"]]
        df.to_csv(OUT / f"{cid}_scores.tsv", sep="\t")
        covs = ["stage_num", "age", "site_oral_cavity", "site_larynx_hypopharynx", "hpv_positive"] if cid == "GSE65858" else ["stage_num", "age"]
        rows = []
        for sc in SCORE_LABELS:
            u, n_u, e_u = cox_per_sd(df, sc, [])
            a, n_a, e_a = cox_per_sd(df, sc, covs)
            rows.append({"cohort": cid, "score": sc, "label": SCORE_LABELS[sc], "HR_per_SD_univariable": u["HR"], "CI_low_uni": u["HR_low_95"], "CI_up_uni": u["HR_up_95"],
                         "p_univariable": u["p_value"], "n_uni": n_u, "events_uni": e_u, "HR_per_SD_adjusted": a["HR"], "CI_low_adj": a["HR_low_95"], "CI_up_adj": a["HR_up_95"],
                         "p_adjusted": a["p_value"], "n_adj": n_a, "events_adj": e_a, "adjusted_for": ", ".join(covs),
                         "classifier_genes_present": n_shared, "nonzero_coefficient_genes_present": f"{n_nz_shared}/{n_nz}"})
        cox = pd.DataFrame(rows)
        cox.to_csv(OUT / f"{cid}_cox.tsv", sep="\t", index=False)
        summary.append(cox)
        km_data[cid] = df
        print(f"\n[{cid}] {len(df)} primaries, {int(df['OS_event'].sum())} deaths; classifier genes present {n_shared}, non-zero-coefficient genes present {n_nz_shared}/{n_nz}")
        print(cox[["label", "HR_per_SD_univariable", "p_univariable", "HR_per_SD_adjusted", "p_adjusted", "n_adj"]].round(3).to_string(index=False))
        if cid == "GSE65858":
            from scipy.stats import mannwhitneyu
            a, b = df.loc[df["n_positive"] == 1, "pEMT_specificity"].dropna(), df.loc[df["n_positive"] == 0, "pEMT_specificity"].dropna()
            print(f"  N+ vs N0 pEMT specificity: medians {a.median():.2f} vs {b.median():.2f}, n {len(a)}/{len(b)}, Mann-Whitney p = {mannwhitneyu(a, b).pvalue:.3f}")
            a, b = df.loc[df["hpv_status"] == "HPV+", "pEMT_specificity"], df.loc[df["hpv_status"] == "HPV-", "pEMT_specificity"]
            print(f"  HPV+ vs HPV- pEMT specificity: medians {a.median():.2f} vs {b.median():.2f}, n {len(a)}/{len(b)}, p = {mannwhitneyu(a, b).pvalue:.2g}")
    pd.concat(summary).to_csv(OUT / "external_cox_summary.tsv", sep="\t", index=False)

    # Figure: per cohort, KM by tertile (left) and adjusted HR per SD per score (right)
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test
    fig, axes = plt.subplots(2, 2, figsize=(mm(180), mm(130)), gridspec_kw={"width_ratios": [1.1, 1]})
    for i, (cid, spec) in enumerate(COHORTS.items()):
        df = km_data[cid].dropna(subset=["OS_time", "OS_event", "pEMT_specificity"])
        df = df[df["OS_time"] > 0]
        df["tertile"] = pd.qcut(df["pEMT_specificity"], 3, labels=["Low", "Mid", "High"])
        ax = axes[i, 0]
        for grp in ["Low", "Mid", "High"]:
            s = df[df["tertile"] == grp]
            KaplanMeierFitter().fit(s["OS_time"] / 365.25, s["OS_event"], label=f"{grp} (n = {len(s)}, events = {int(s['OS_event'].sum())})").plot_survival_function(
                ax=ax, color=TERTILE_COLOURS[grp], ci_show=False, linewidth=1.2)
        lr = multivariate_logrank_test(df["OS_time"], df["tertile"], df["OS_event"])
        ax.set_xlim(0, min(10, float(df["OS_time"].max() / 365.25)))
        ax.set_ylim(0, 1)
        ax.set_xlabel("Years"); ax.set_ylabel("Overall survival probability")
        ax.text(0.03, 0.05, f"log-rank p = {lr.p_value:.2f}", transform=ax.transAxes, fontsize=7)
        ax.legend(loc="upper right", fontsize=6, frameon=False)
        ax.set_title(f"{spec['label']}, n = {len(df)}", fontsize=7, color=PALETTE["grey"], loc="right")
        ax.set_title("ac"[i], loc="left", fontweight="bold")
        ax2 = axes[i, 1]
        cox = summary[i]
        y = np.arange(len(cox))[::-1]
        for j, (_, r) in enumerate(cox.iterrows()):
            col = PALETTE["vermilion"] if r["score"] in ("pEMT_specificity", "P_pEMT_high") else PALETTE["black"]
            ax2.errorbar(r["HR_per_SD_adjusted"], y[j], xerr=[[r["HR_per_SD_adjusted"] - r["CI_low_adj"]], [r["CI_up_adj"] - r["HR_per_SD_adjusted"]]],
                         fmt="o", color=col, ecolor=col, capsize=2, markersize=3.5, elinewidth=0.8)
        ax2.axvline(1, ls="--", color=PALETTE["grey"], linewidth=0.6)
        ax2.set_yticks(y); ax2.set_yticklabels(cox["label"], fontsize=7)
        ax2.set_xlabel(f"Adjusted HR per SD (95% CI), n = {int(cox['n_adj'].iloc[0])}")
        ax2.set_xscale("log")
        ticks = [0.5, 0.7, 1, 1.4, 2]
        ax2.set_xticks(ticks); ax2.set_xticklabels([f"{t:g}" for t in ticks]); ax2.xaxis.set_minor_locator(plt.NullLocator())
        ax2.set_title("bd"[i], loc="left", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, FIG, "Figure_S6_external_cohorts")


if __name__ == "__main__":
    main()
