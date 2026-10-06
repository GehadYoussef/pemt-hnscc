# Changelog

## V1.1

### Classifier

The training pseudobulks are now built the way bulk expression arises. Each pseudobulk sums the
sampled cells' linear counts per 10,000, renormalises the sum to 10,000 and takes the logarithm
(sum-then-log). V1.0 averaged the cells' log-normalised profiles, which is not how a bulk profile is
formed. The random draws are unchanged, so the new pseudobulks contain exactly the same cells as
before. The mean-of-log matrix is still written from the same draws, and
`src/10_cetuximab/08_pseudobulk_construction_sensitivity.py` now refits the classifier on it as the
sensitivity analysis. The sum-then-log classifier gives a pEMT specificity that correlates with the
V1.0 score at 0.98 in TCGA-HNSC and in GSE65021. Its held-out accuracy is higher (0.87 against 0.71
with the training fold's scaler), and its area under the curve for long progression-free survival in
GSE65021 is 0.78 against 0.74.

The classifier benchmark (`src/02_multinomial_pemt_model/05_benchmark_classifiers.py`) now also
evaluates every learner the way bulk cohorts are scored. The held-out dataset is standardised within
itself, and for the linear models only the fitted logistic layer is then applied. Panel d of Figure 1
shows this evaluation beside the earlier one, which uses the training fold's scaler. The new tables are
`results/multinomial_classifier/classifier_benchmark_folds_deployment.tsv` and
`classifier_benchmark_summary_deployment.tsv`, and they are added to the supplementary workbook.

### Revision analyses

A new stage, `src/13_revision/`, adds the analyses requested in review. Each reads the results written
by the earlier stages.

- `01_hartung_knapp_meta.py` repools the survival hazard ratios with the Hartung-Knapp adjustment beside
  the DerSimonian-Laird interval, with every cohort left out in turn.
- `02_core_vs_canonical.py` enters the malignant core and pEMT specificity, each together with the
  canonical Puram pEMT signature, in one Firth model in GSE65021.
- `03_basal_centroid.py` calls the TCGA expression subtypes in GSE65021 and TCGA-HNSC with the published
  centroids and tests the Basal score against outcome under cetuximab. It downloads the centroid file
  and the TCGA classification matrix from the GDC.
- `04_saturation_logodds.py` measures how many tumours sit near the bounds of pEMT specificity and
  repeats the main analyses with a log-odds summary of the same class probabilities.
- `05_gavish_meta_programmes.py` scores the 41 cancer meta-programmes of Gavish et al. 2023 in TCGA-HNSC
  and GSE65021 and records what Tyler and Tirosh 2021 deposited. It downloads Supplementary Table 2 of
  Gavish et al. and matches outdated gene symbols to current HGNC symbols.
- `06_bayesprism_deconvolution.R` and `07_bayesprism_analysis.py` deconvolve TCGA-HNSC and GSE65021 with
  BayesPrism, using GSE181919 as the single-cell reference, and compare the malignant-cell-specific
  scores and the cell-type fractions with the bulk scores, the TCGA subtypes and outcome under
  cetuximab. This stage is optional. It needs R with BayesPrism, the TCGA-HNSC run takes about five
  hours on about 20 cores, and `run_all.sh` and `run_all.ps1` skip it with a message when Rscript or
  BayesPrism is missing.

### Corrections

- `run_all.sh` and `run_all.ps1` ran two scripts before the outputs they read existed. The survival
  meta-analysis (`04_tcga_projection/13`) reads the per-cohort estimates written by `14`, and the
  independent validation and subgroup analyses (`10`, `11`) read the CPTAC-3 scores written by `12`. The
  order is now 09, 12, 10, 11, 14, 13, so a run from an empty `results/` gives the same tables as a
  repeated run.
- Supplementary Figure 11 (`12_figures_and_tables/02_combined_figures.py`) read a Cox model table that no
  stage writes. It now draws the adjusted model M2 of `04_tcga_projection/06_survival_analysis.py`
  (stage, age, site and HPV status).
- `requirements.txt` now pins scikit-learn 1.9.0, the version the V1.1 results were produced with.
- The CRISPR dependency stage (`07_depmap_broad_prism_validation/04_crispr_dependency.py`) was not rerun for V1.1, because it needs the DepMap file `CRISPRGeneEffect.csv`, which is not redistributed. Its outputs are those of V1.0.
- Figure 2c stops its axis at 100%, Figure 3c reports the exact two-sided Mann-Whitney test, and the
  title of Figure 5c reads "Replication in four cohorts", since CPTAC-3 is included.
