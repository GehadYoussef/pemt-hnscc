"""Published HNSCC gene signatures used as external anchors for the pEMT axis.

None of these were derived here. Each was taken from the source paper's own supplementary material
(gene lists verified against the published files, not reconstructed), so a correlation between one of
them and the classifier axis is an independent statement about what the axis is measuring.

  schinke2022_egfr_emt   171 genes. Schinke et al., Mol Cancer 2022;21:178 (PMID 36076232).
                         Genes concordantly induced by EGF in Kyse30 and FaDu, i.e. the transcriptional
                         programme of EGFR-driven partial EMT. Supplementary Table 1.
  zhou2025_fdeg          46 genes. Zhou et al., Mol Cancer 2025;24:94 (PMID 40121428). "Functional DEGs"
                         of EGFR-mediated local invasion, counter-regulated by cetuximab and/or MEK
                         inhibition. Supplementary Material 5.
  zhou2025_invgrn        59 genes. Same paper, the invasive gene regulatory network (hubs INHBA, SNAI2).
  zhou2025_predictive     9 genes. Same paper: COL17A1, FOSL1, ITGA3, ITGB4, LAMA3, LAMC2, MT2A, PHLDA1,
                         PLEK2. Low expression of each gave significantly increased odds of short PFS on
                         cetuximab in R/M-HNSCC by multivariate logistic regression.
  ourailidis2026_tbs     28 genes. Ourailidis et al., Genome Med 2026 (PMID 41987303). Spatial
                         transcriptomics tumour budding signature in HPV-negative HNSCC; TBS-high
                         squamous lines were selectively MEK-inhibitor sensitive.

The three papers share a senior author (Gires) and are a connected series; their published three-way
intersection is ITGA3, ITGB4, LAMB3, LAMC2, which is reproduced by these files.
"""

from __future__ import annotations

from pathlib import Path

FILES = {
    "schinke2022_egfr_emt": "schinke2022_egfr_emt_genes.txt",
    "zhou2025_fdeg": "zhou2025_cetuximab_response_genes.txt",
    "zhou2025_invgrn": "zhou2025_invGRN_59genes.txt",
    "zhou2025_predictive": "zhou2025_cetuximab_predictive_9genes.txt",
    "ourailidis2026_tbs": "bhatt2026_tumour_budding_28genes.txt",
}

LABELS = {
    "schinke2022_egfr_emt": "EGFR-induced EMT, 171 genes (Schinke 2022)",
    "zhou2025_fdeg": "EGFR invasion fDEGs, 46 genes (Zhou 2025)",
    "zhou2025_invgrn": "Invasive gene network, 59 genes (Zhou 2025)",
    "zhou2025_predictive": "Cetuximab PFS predictors, 9 genes (Zhou 2025)",
    "ourailidis2026_tbs": "Tumour budding signature, 28 genes (Ourailidis 2026)",
}

CITATIONS = {
    "schinke2022_egfr_emt": "Schinke et al., Mol Cancer 2022;21:178 (PMID 36076232)",
    "zhou2025_fdeg": "Zhou et al., Mol Cancer 2025;24:94 (PMID 40121428)",
    "zhou2025_invgrn": "Zhou et al., Mol Cancer 2025;24:94 (PMID 40121428)",
    "zhou2025_predictive": "Zhou et al., Mol Cancer 2025;24:94 (PMID 40121428)",
    "ourailidis2026_tbs": "Ourailidis et al., Genome Med 2026 (PMID 41987303)",
}


def load_published(ref_dir: Path) -> dict[str, list[str]]:
    """Gene lists keyed by signature id. Raises if a file is missing rather than silently skipping."""
    out = {}
    for key, fname in FILES.items():
        p = Path(ref_dir) / fname
        if not p.exists():
            raise FileNotFoundError(f"published signature file missing: {p}")
        out[key] = [l.strip() for l in p.read_text().splitlines() if l.strip()]
    return out
