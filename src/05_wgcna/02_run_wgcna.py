"""Build the WGCNA co-expression network and detect modules.

Python implementation of the WGCNA steps: Pearson correlation,
signed adjacency ((1 + cor) / 2)^power, topological overlap (TOM), average
linkage on 1 - TOM, the hybrid dynamic tree cut from the dynamicTreeCut
package (cutreeHybrid), module eigengenes, and merging of modules whose
eigengenes correlate at or above 1 - wgcna.merge_cut_height (0.25). The
network is signed (config wgcna.TOM_type), so anti-correlated genes, such as
co-up pEMT genes and epithelial genes, are not fused into one module.

With wgcna.soft_power set to auto, the soft-thresholding power is the smallest
power from 1 to 20 whose scale-free topology fit (signed R^2) reaches
wgcna.scale_free_r2 (0.80 in config.yaml), as in WGCNA pickSoftThreshold.
If no power reaches it, the power with the highest fit is used. The fit table
and a diagnostic figure are written.

The PAM stage of the tree cut is reimplemented here with
pamRespectsDendro = FALSE semantics. An unassigned gene joins its nearest
module (mean TOM dissimilarity) when that distance is below the module
diameter or below cutHeight (the lowest dendrogram height plus 99% of the
height range). Module membership (kME, the correlation of each gene with each
module eigengene) is written for every gene.

Inputs:  results/wgcna/datExpr_for_wgcna.pkl
Outputs: results/wgcna/scale_free_fit.tsv, module_eigengenes.tsv,
         gene_module_assignments.tsv, gene_kME.tsv, wgcna_dendrogram.npz,
         wgcna_diagnostics.txt
         results/figures/Supplementary_Figure_2 (.svg, .png): scale-free fit
         and mean connectivity against soft-thresholding power
Usage:   python src/05_wgcna/02_run_wgcna.py [--figure-only]
         --figure-only redraws Supplementary_Figure_2 from scale_free_fit.tsv
         without refitting the network.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pemt import load_config, project_root  # noqa: E402
from pemt.figstyle import apply_style, save_figure, PALETTE, mm  # noqa: E402

cfg = load_config()
W = cfg["wgcna"]
ROOT = project_root()
WGCNA_DIR = ROOT / cfg["paths"]["results_dir"] / "wgcna"
FIG = ROOT / "results" / "figures"
apply_style()
import matplotlib.pyplot as plt  # noqa: E402


def soft_threshold(corr, power, signed):
    return ((1.0 + corr) / 2.0) ** power if signed else np.abs(corr) ** power


def scale_free_fit(k: np.ndarray, n_breaks: int = 10) -> tuple[float, float]:
    """WGCNA scaleFreeFitIndex: signed R^2 of log10 p(k) vs log10 k, and slope."""
    k = k[k > 0]
    bins = np.linspace(k.min(), k.max(), n_breaks + 1)
    idx = np.clip(np.digitize(k, bins[1:-1]), 0, n_breaks - 1)
    dk = np.array([k[idx == b].mean() if (idx == b).any() else np.nan for b in range(n_breaks)])
    pk = np.array([(idx == b).mean() for b in range(n_breaks)])
    ok = np.isfinite(dk) & (pk > 0)
    x, y = np.log10(dk[ok]), np.log10(pk[ok])
    if len(x) < 3:
        return np.nan, np.nan
    slope, intercept = np.polyfit(x, y, 1)
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    return -np.sign(slope) * r2, slope


def pick_soft_power(corr, signed, target_r2, powers=range(1, 21)):
    rows = []
    for p in powers:
        A = soft_threshold(corr, p, signed)
        np.fill_diagonal(A, 0.0)
        k = A.sum(axis=1)
        r2, slope = scale_free_fit(k)
        rows.append({"power": p, "signed_R2": r2, "slope": slope, "mean_k": k.mean(), "median_k": np.median(k), "max_k": k.max()})
        del A
    fit = pd.DataFrame(rows)
    ok = fit[fit["signed_R2"] >= target_r2]
    chosen = int(ok["power"].iloc[0]) if len(ok) else int(fit.loc[fit["signed_R2"].idxmax(), "power"])
    return chosen, fit


def topological_overlap(A):
    np.fill_diagonal(A, 0.0)
    L = A @ A
    k = A.sum(axis=1)
    K = np.minimum.outer(k, k)
    TOM = (L + A) / (K + 1.0 - A)
    np.fill_diagonal(TOM, 1.0)
    np.clip(TOM, 0.0, 1.0, out=TOM)
    return TOM


def first_pc_eigengene(X):
    Xs = X - X.mean(axis=0, keepdims=True)
    sd = Xs.std(axis=0, keepdims=True); sd[sd == 0] = 1.0
    Xs = Xs / sd
    U, S, _ = np.linalg.svd(Xs, full_matrices=False)
    pc1 = U[:, 0] * S[0]
    if np.corrcoef(pc1, Xs.mean(axis=1))[0, 1] < 0:
        pc1 = -pc1
    return pc1


def eigengenes(datExpr, labels):
    modules = sorted(set(labels) - {0})
    ME = np.zeros((datExpr.shape[0], len(modules)))
    for j, m in enumerate(modules):
        ME[:, j] = first_pc_eigengene(datExpr.values[:, labels == m])
    return modules, ME


def merge_close_modules(datExpr, labels, cut_height, max_iter=50):
    labels = labels.copy()
    for _ in range(max_iter):
        modules, ME = eigengenes(datExpr, labels)
        if len(modules) < 2:
            break
        C = np.corrcoef(ME, rowvar=False)
        np.fill_diagonal(C, -np.inf)
        i, j = np.unravel_index(np.argmax(C), C.shape)
        if C[i, j] < 1.0 - cut_height:
            break
        labels[labels == modules[j]] = modules[i]
    surviving = sorted(set(labels) - {0})
    remap = {m: k + 1 for k, m in enumerate(surviving)}; remap[0] = 0
    labels = np.array([remap[m] for m in labels])
    modules, ME = eigengenes(datExpr, labels)
    ME_df = pd.DataFrame(ME, index=datExpr.index, columns=[f"ME_M{m:02d}" for m in modules])
    return labels, ME_df


def scale_free_figure(fit: pd.DataFrame, power: int) -> None:
    """Draw Supplementary_Figure_2: scale-free fit and mean connectivity against soft-thresholding power."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(mm(150), mm(60)))
    a1.plot(fit["power"], fit["signed_R2"], "o-", color=PALETTE["blue"], markersize=3)
    a1.axhline(float(W.get("scale_free_r2", 0.85)), ls="--", color=PALETTE["grey"], linewidth=0.6)
    a1.set_xlabel("Soft-thresholding power"); a1.set_ylabel("Scale-free topology fit (signed R\u00b2)")
    a2.plot(fit["power"], fit["mean_k"], "o-", color=PALETTE["vermilion"], markersize=3)
    a2.set_xlabel("Soft-thresholding power"); a2.set_ylabel("Mean connectivity")
    a2.set_yscale("log")
    for a in (a1, a2):
        a.axvline(power, color=PALETTE["black"], linewidth=0.6)
    fig.tight_layout()
    for ax, letter in zip((a1, a2), "ab"):
        ax.text(-0.18, 1.06, letter, transform=ax.transAxes, fontweight="bold", fontsize=8)
    save_figure(fig, FIG, "Supplementary_Figure_2")


def main() -> None:
    from dynamicTreeCut import cutreeHybrid
    datExpr = pd.read_pickle(WGCNA_DIR / "datExpr_for_wgcna.pkl")
    samples, genes = datExpr.shape
    signed = str(W["TOM_type"]).lower() == "signed"
    print(f"datExpr: {samples} samples x {genes} genes, signed={signed}")

    t0 = time.time()
    corr = np.corrcoef(datExpr.values.T).astype(np.float32)
    np.nan_to_num(corr, copy=False)
    print(f"correlation done in {time.time() - t0:.0f}s")

    if str(W.get("soft_power", "auto")).lower() == "auto":
        power, fit = pick_soft_power(corr, signed, float(W.get("scale_free_r2", 0.85)))
        fit.to_csv(WGCNA_DIR / "scale_free_fit.tsv", sep="\t", index=False)
        scale_free_figure(fit, power)
        print(f"chosen power {power}")
    else:
        power = int(W["soft_power"]); fit = None

    A = soft_threshold(corr, power, signed).astype(np.float32)
    del corr
    np.fill_diagonal(A, 0.0)
    k = A.sum(axis=1)
    r2, slope = scale_free_fit(k)
    print(f"power {power}: mean k {k.mean():.2f}, max k {k.max():.1f}, scale-free R2 {r2:.3f}")

    t0 = time.time()
    TOM = topological_overlap(A)
    del A
    print(f"TOM done in {time.time() - t0:.0f}s")
    diss = 1.0 - TOM
    del TOM
    np.fill_diagonal(diss, 0.0)
    diss = (diss + diss.T) / 2.0
    np.clip(diss, 0.0, 1.0, out=diss)
    Z = linkage(squareform(diss, checks=False), method="average")

    # Dynamic tree cut with pamStage=False. The PAM stage in the dynamicTreeCut
    # package does not run correctly (it never binds df_apply and the nearest-cluster
    # label is off by one), so the WGCNA PAM stage is implemented below with
    # pamRespectsDendro = FALSE semantics: an unassigned gene joins its nearest module
    # (mean TOM dissimilarity) when that distance is below the module diameter or
    # below maxPamDist (= cutHeight).
    cut = cutreeHybrid(link=Z, distM=diss, deepSplit=int(W.get("deep_split", 2)),
                       minClusterSize=int(W["min_module_size"]), pamStage=False, pamRespectsDendro=False, verbose=0)
    labels0 = np.array(cut["labels"], dtype=int)
    n0 = len(set(labels0) - {0})
    core_labels = labels0.copy()
    heights = Z[:, 2]
    cut_height = float(heights.min() + 0.99 * (heights.max() - heights.min()))
    labels_pam = labels0.copy()
    mods = sorted(set(labels0) - {0})
    diam = {}
    for m in mods:
        idx = np.where(labels0 == m)[0]
        sub = diss[np.ix_(idx, idx)]
        diam[m] = float((sub.sum(axis=1) / max(len(idx) - 1, 1)).max())
    grey = np.where(labels0 == 0)[0]
    if len(grey) and mods:
        D = np.column_stack([diss[np.ix_(grey, np.where(labels0 == m)[0])].mean(axis=1) for m in mods])
        nearest = D.argmin(axis=1)
        nearest_d = D[np.arange(len(grey)), nearest]
        nearest_m = np.array(mods)[nearest]
        assign = (nearest_d < np.array([diam[m] for m in nearest_m])) | (nearest_d < cut_height)
        labels_pam[grey[assign]] = nearest_m[assign]
    pam_used = True
    n_pam = int((labels_pam != 0).sum() - (labels0 != 0).sum())
    print(f"PAM stage assigned {n_pam} of {len(grey)} unassigned genes (cutHeight {cut_height:.4f})")
    labels0 = labels_pam
    labels, ME_df = merge_close_modules(datExpr, labels0, float(W["merge_cut_height"]))
    n_final = len(set(labels) - {0})
    print(f"modules: {n0} initial, {n_final} after merging, grey {int((labels == 0).sum())}")

    core = {}
    for i, m in enumerate(core_labels):
        core[i] = m
    assignments = pd.DataFrame({"gene": datExpr.columns, "module": [f"M{m:02d}" if m else "grey" for m in labels],
                                "core_member_pre_PAM": [bool(core_labels[i] != 0) for i in range(len(labels))]})
    assignments.to_csv(WGCNA_DIR / "gene_module_assignments.tsv", sep="\t", index=False)
    ME_df.to_csv(WGCNA_DIR / "module_eigengenes.tsv", sep="\t")

    # kME: correlation of every gene with every module eigengene
    Xz = (datExpr.values - datExpr.values.mean(axis=0)) / (datExpr.values.std(axis=0) + 1e-9)
    MEz = (ME_df.values - ME_df.values.mean(axis=0)) / (ME_df.values.std(axis=0) + 1e-9)
    kME = pd.DataFrame(Xz.T @ MEz / samples, index=datExpr.columns, columns=[c.replace("ME_", "kME_") for c in ME_df.columns])
    kME.insert(0, "module", assignments["module"].values)
    kME.to_csv(WGCNA_DIR / "gene_kME.tsv", sep="\t")

    np.savez_compressed(WGCNA_DIR / "wgcna_dendrogram.npz", Z=Z, leaf_order=leaves_list(Z),
                        gene_names=np.array(datExpr.columns), labels=labels)
    diag = {
        "samples": samples, "genes": genes, "signed": signed, "soft_power": power,
        "soft_power_selection": "scale-free R2 >= %s" % W.get("scale_free_r2", 0.85) if fit is not None else "fixed",
        "scale_free_R2_at_power": round(float(r2), 3), "mean_connectivity": round(float(k.mean()), 3),
        "max_connectivity": round(float(k.max()), 2), "min_module_size": W["min_module_size"],
        "deep_split": W.get("deep_split", 2), "merge_cut_height": W["merge_cut_height"], "pam_stage_used": pam_used,
        "modules_initial": n0, "genes_assigned_by_PAM": n_pam, "modules_final": n_final, "genes_in_modules": int((labels != 0).sum()), "genes_grey": int((labels == 0).sum()),
    }
    (WGCNA_DIR / "wgcna_diagnostics.txt").write_text("\n".join(f"{a}\t{b}" for a, b in diag.items()) + "\n", encoding="utf-8")
    print(diag)


if __name__ == "__main__":
    # --figure-only redraws Supplementary_Figure_2 from the saved fit table, without refitting the network
    if "--figure-only" in sys.argv:
        f = pd.read_csv(WGCNA_DIR / "scale_free_fit.tsv", sep="\t")
        thr = float(W.get("scale_free_r2", 0.85))
        ok = f[f["signed_R2"] >= thr]
        scale_free_figure(f, int(ok["power"].min()) if len(ok) else int(f.loc[f["signed_R2"].idxmax(), "power"]))
    else:
        main()
