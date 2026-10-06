# 06_bayesprism_deconvolution.R
#
# Reference-based deconvolution of TCGA-HNSC and GSE65021 with BayesPrism (Chu et al., Nat Cancer 2022),
# with GSE181919 as the single-cell reference, as an independent check on the bulk scores.
#
# Reference: GSE181919 (Choi et al., Nat Commun 2023) single-cell UMI counts with the authors' cell
#   annotation, primary tumour (CA) and lymph-node (LN) tissue only. Leukoplakia and normal tissue are
#   excluded. Cell types: Malignant, Fibroblast, Endothelial, T, B_Plasma, Myeloid (macrophage and
#   dendritic cell states), Mast, Myocyte. Malignant cells are the tumour type (key) and carry one cell
#   state per patient (patients with fewer than 40 malignant cells pooled into one state), as BayesPrism
#   expects for the tumour compartment. Non-malignant cell states are capped at 3,000 cells by random
#   downsampling (seed 20261005). All malignant cells are kept. The composition before and after
#   downsampling is written to reference_composition.tsv.
#
# Gene filtering follows the BayesPrism tutorial: ribosomal, mitochondrial, MALAT1 and sex-chromosome
#   genes removed, genes expressed in fewer than 5 cells removed, protein-coding genes only. Genes used
#   for deconvolution are the BayesPrism cell-type markers (get.exp.stat / select.marker, p < 0.01,
#   log fold change > 0.1), which the authors recommend when reference and bulk come from different
#   platforms, plus the genes that are scored downstream (the 100 Puram pEMT genes, the 12-gene
#   malignant core and the 14-gene stromal core), so that a malignant-cell-specific expression estimate
#   exists for each of them. Symbols of different vintages are harmonised to current HGNC in all three
#   datasets and the Puram list (CXCR7 = ACKR3, LEPREL1 = P3H2, PRKCDBP = CAVIN3, DFNA5 = GSDME).
#
# Targets
#   TCGA-HNSC: the 520 primaries of the projection (emt_score_panel_scores.tsv), raw STAR gene counts
#     (the Xena file stores log2(count + 1), inverted here and rounded). Ensembl IDs mapped to symbols
#     with the pipeline mapping table, and counts of duplicated symbols summed.
#   GSE65021: Illumina DASL microarray (GPL14951, annotated through GPL10558), quantile-normalised
#     intensities on the deposited linear scale. Probes collapsed to genes by the highest mean log2
#     intensity (the rule of 10_cetuximab/01_cetuximab_cohort.py) and the linear values of that probe
#     used. This is microarray intensity, not counts, and BayesPrism treats it as count-like input.
#
# BayesPrism run: new.prism(key = "Malignant", outlier.cut = 0.01, outlier.fraction = 0.1), run.prism
#   with the default Gibbs settings (chain 1,000, burn-in 500, thinning 2, seed 123) and the updated
#   Gibbs step, on n cores. The TCGA run takes about five hours on about 20 cores.
#
# Purity proxies: ESTIMATE stromal, immune and ESTIMATE scores (tidyestimate, Yoshihara et al. 2013 gene
#   sets) on log2 expression, with the Affymetrix purity transform reported for reference only.
#
# Inputs
#   data/raw/GSE181919/GSE181919_UMI_counts.txt.gz, GSE181919_Barcode_metadata.txt.gz
#   data/raw/tcga_hnsc/expression/TCGA-HNSC.star_counts.tsv, data/references/ensembl_to_symbol_mapping.tsv
#   data/processed/tcga_hnsc/tcga_star_tpm_log2_for_projection.tsv
#   data/raw/external_cohorts/GSE65021_series_matrix.txt.gz, GPL10558.annot.gz
#   config/signatures/puram_pemt.txt, results/arm_scores/arm_gene_sets.tsv
#   results/tcga_projection/emt_score_panel_scores.tsv
#
# Outputs (results/bayesprism/)
#   reference_composition.tsv, reference_markers.tsv, reference_genes_used_<cohort>.tsv
#   <cohort>_fractions.tsv            final theta per cell type (samples x types)
#   <cohort>_fractions_cv.tsv         posterior coefficient of variation of theta
#   <cohort>_fractions_initial.tsv    theta per cell type after the first Gibbs step
#   <cohort>_fractions_state.tsv      initial theta per cell state
#   <cohort>_Z_malignant.tsv.gz       malignant-cell-specific expression (samples x genes, count units)
#   <cohort>_Z_fibroblast.tsv.gz      fibroblast-specific expression
#   <cohort>_estimate.tsv             ESTIMATE scores
#   run_log_<cohort>.txt, session_info.txt
#
# Usage:   Rscript src/13_revision/06_bayesprism_deconvolution.R [cohort ...] [--cores=N] [--cache=ref.rds]
#   cohort is TCGA-HNSC and/or GSE65021 (default both). Then python src/13_revision/07_bayesprism_analysis.py

suppressPackageStartupMessages({
  library(data.table)
  library(Matrix)
  library(BayesPrism)
})

# ---- paths ------------------------------------------------------------------------------------------
args_all <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", args_all[grepl("^--file=", args_all)])
HERE <- if (length(script_path)) dirname(normalizePath(script_path)) else getwd()
ROOT <- normalizePath(file.path(HERE, "..", ".."))
DATA <- file.path(ROOT, "data")
EXT <- file.path(DATA, "raw", "external_cohorts")
D181919 <- file.path(DATA, "raw", "GSE181919")
OUT <- file.path(ROOT, "results", "bayesprism")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

args <- commandArgs(trailingOnly = TRUE)
N_CORES <- 18L
if (any(grepl("^--cores=", args))) N_CORES <- as.integer(sub("^--cores=", "", args[grepl("^--cores=", args)]))
COHORTS <- args[!grepl("^--", args)]
# optional --cache=<file.rds> keeps the processed reference between runs. It is not a result, so keep it
# outside results/
CACHE <- if (any(grepl("^--cache=", args))) sub("^--cache=", "", args[grepl("^--cache=", args)]) else NULL
if (!length(COHORTS)) COHORTS <- c("TCGA-HNSC", "GSE65021")

SEED <- 20261005L
CAP_PER_STATE <- 3000L
MIN_MALIGNANT_PER_PATIENT <- 40L
# the three datasets use symbols of different vintages: harmonise the scored genes to current HGNC
ALIAS <- c(CXCR7 = "ACKR3", LEPREL1 = "P3H2", PRKCDBP = "CAVIN3", DFNA5 = "GSDME")
harmonise <- function(g) {
  hit <- g %in% names(ALIAS) & !(ALIAS[g] %in% g)
  g[hit] <- ALIAS[g[hit]]
  g
}

msg <- function(...) cat(format(Sys.time(), "%H:%M:%S"), ..., "\n", sep = " ")

read_list <- function(path) {
  x <- trimws(readLines(path, warn = FALSE))
  x[nzchar(x) & !startsWith(x, "#")]
}

# ---- genes scored downstream -------------------------------------------------------------------------
puram <- read_list(file.path(ROOT, "config", "signatures", "puram_pemt.txt"))
puram <- harmonise(puram)
sets <- fread(file.path(ROOT, "results", "arm_scores", "arm_gene_sets.tsv"))
split_set <- function(k) trimws(strsplit(sets[score == k, genes], ",")[[1]])
mal_core <- split_set("malignant_arm_core")
str_core <- split_set("stromal_arm_core")
INTEREST <- unique(c(puram, mal_core, str_core))

# ---- 1. reference ------------------------------------------------------------------------------------
build_reference <- function() {
  if (!is.null(CACHE) && file.exists(CACHE)) {
    msg("reference loaded from cache", CACHE)
    return(readRDS(CACHE))
  }
  mf <- file.path(D181919, "GSE181919_Barcode_metadata.txt.gz")
  con <- gzfile(mf, "r")
  mh <- strsplit(readLines(con, n = 1), "\t")[[1]]
  close(con)
  meta <- fread(mf, skip = 1, header = FALSE)            # header line lacks the row-name column
  setnames(meta, c("barcode", mh))
  meta <- meta[tissue.type %in% c("CA", "LN") & cell.type != "Epithelial.cells"]
  type_map <- c(Malignant.cells = "Malignant", Fibroblasts = "Fibroblast", Endothelial.cells = "Endothelial",
                T.cells = "T", B_Plasma.cells = "B_Plasma", Macrophages = "Myeloid",
                Dendritic.cells = "Myeloid", Mast.cells = "Mast", Myocytes = "Myocyte")
  meta[, type := type_map[cell.type]]
  meta[, state := type]
  meta[cell.type == "Macrophages", state := "Myeloid_macrophage"]
  meta[cell.type == "Dendritic.cells", state := "Myeloid_dendritic"]
  mal_n <- meta[type == "Malignant", .N, by = patient.id]
  big <- mal_n[N >= MIN_MALIGNANT_PER_PATIENT, patient.id]
  meta[type == "Malignant", state := ifelse(patient.id %in% big, paste0("Malignant_", patient.id),
                                             "Malignant_pooled")]
  before <- meta[, .(n_cells_CA_LN = .N), by = .(type, state)]

  set.seed(SEED)
  keep <- meta[, {
    if (type[1] != "Malignant" && .N > CAP_PER_STATE) .SD[sample(.N, CAP_PER_STATE)] else .SD
  }, by = state]
  after <- keep[, .(n_cells_used = .N, n_patients = uniqueN(patient.id)), by = .(type, state)]
  comp <- merge(before, after, by = c("type", "state"), all = TRUE)
  setorder(comp, type, state)
  fwrite(comp, file.path(OUT, "reference_composition.tsv"), sep = "\t")
  msg("reference cells used:", nrow(keep), "of", nrow(meta), "CA/LN cells")
  print(comp)

  # stream-free column selection: fread decompresses once and reads only the selected cells
  umi <- file.path(D181919, "GSE181919_UMI_counts.txt.gz")
  con <- gzfile(umi, "r")
  header <- strsplit(readLines(con, n = 1), "\t")[[1]]
  close(con)
  header <- gsub(".", "-", header, fixed = TRUE)   # same rule as 01_single_cell_qc/05_prepare_gse181919.py
  idx <- match(keep$barcode, header)
  if (anyNA(idx)) stop(sum(is.na(idx)), " reference barcodes not found in the UMI matrix header")
  msg("reading", length(idx), "cells from the UMI matrix")
  dt <- fread(umi, skip = 1, header = FALSE, select = c(1L, idx + 1L), nThread = N_CORES, showProgress = FALSE)
  genes <- harmonise(dt[[1]])
  dt[, 1 := NULL]
  X <- t(as.matrix(dt))
  rm(dt)
  gc()
  dimnames(X) <- list(keep$barcode, genes)
  storage.mode(X) <- "double"
  msg("reference matrix", nrow(X), "cells x", ncol(X), "genes")

  X <- cleanup.genes(input = X, input.type = "count.matrix", species = "hs",
                     gene.group = c("Rb", "Mrp", "other_Rb", "chrM", "MALAT1", "chrX", "chrY"), exp.cells = 5)
  Xpc <- select.gene.type(X, gene.type = "protein_coding")
  # BayesPrism's annotation is GENCODE v22, which predates some current symbols (e.g. P3H2): keep the
  # scored genes whatever the annotation says (all are protein-coding)
  extra <- setdiff(intersect(INTEREST, colnames(X)), colnames(Xpc))
  X <- cbind(Xpc, X[, extra, drop = FALSE])
  rm(Xpc)
  msg("after gene cleanup and protein-coding selection:", ncol(X), "genes, scored genes added back:",
      paste(extra, collapse = ", "))

  stat <- get.exp.stat(sc.dat = X[, colSums(X > 0) > 3], cell.type.labels = keep$type,
                       cell.state.labels = keep$state, pseudo.count = 0.1, cell.count.cutoff = 50,
                       n.cores = 1)
  mk <- rbindlist(lapply(names(stat), function(ct) {
    s <- stat[[ct]]
    s <- s[s$pval.up.min < 0.01 & s$min.lfc > 0.1, , drop = FALSE]
    data.table(cell_type = ct, gene = rownames(s), pval_up_min = s$pval.up.min, min_lfc = s$min.lfc)
  }))
  fwrite(mk, file.path(OUT, "reference_markers.tsv"), sep = "\t")
  msg("markers per cell type:")
  print(mk[, .N, by = cell_type])
  ref <- list(X = X, type = keep$type, state = keep$state, markers = unique(mk$gene))
  if (!is.null(CACHE)) saveRDS(ref, CACHE)
  ref
}

# ---- 2. bulk inputs ----------------------------------------------------------------------------------
load_tcga <- function() {
  primaries <- fread(file.path(ROOT, "results", "tcga_projection", "emt_score_panel_scores.tsv"),
                     select = 1L)[[1]]
  f <- file.path(DATA, "raw", "tcga_hnsc", "expression", "TCGA-HNSC.star_counts.tsv")
  dt <- fread(f, nThread = N_CORES, showProgress = FALSE)
  miss <- setdiff(primaries, names(dt))
  if (length(miss)) stop(length(miss), " primaries missing from the STAR count file")
  ens <- sub("\\..*$", "", dt[[1]])
  map <- fread(file.path(DATA, "references", "ensembl_to_symbol_mapping.tsv"))
  sym <- harmonise(map$symbol[match(ens, map$ensembl_id)])
  par_y <- grepl("_PAR_Y$", dt[[1]])
  m <- as.matrix(dt[, ..primaries])
  m <- round(2^m - 1)                         # Xena stores log2(count + 1)
  ok <- !is.na(sym) & nzchar(sym) & !par_y
  m <- rowsum(m[ok, ], sym[ok])               # duplicated symbols summed
  msg("TCGA counts:", ncol(m), "primaries x", nrow(m), "symbols")
  # log2 TPM for ESTIMATE, from the projection matrix used throughout the pipeline
  tpm <- fread(file.path(DATA, "processed", "tcga_hnsc", "tcga_star_tpm_log2_for_projection.tsv"),
               nThread = N_CORES, showProgress = FALSE)
  tpm <- as.matrix(tpm[, ..primaries], rownames = tpm[[1]])
  list(counts = t(m), log2 = tpm)
}

read_series_matrix <- function(path) {
  lines <- readLines(gzfile(path), warn = FALSE)
  b <- which(startsWith(lines, "!series_matrix_table_begin"))
  e <- which(startsWith(lines, "!series_matrix_table_end"))
  tab <- fread(text = paste(lines[(b + 1):(e - 1)], collapse = "\n"))
  m <- as.matrix(tab[, -1])
  rownames(m) <- gsub('"', "", tab[[1]])
  m
}

read_annotation <- function(path) {
  lines <- readLines(gzfile(path), warn = FALSE)
  s <- which(startsWith(lines, "ID\t"))
  e <- which(startsWith(lines, "!platform_table_end"))
  if (!length(e)) e <- length(lines) + 1
  tab <- fread(text = paste(lines[s:(e - 1)], collapse = "\n"), select = c("ID", "Gene symbol"), quote = "")
  tab <- tab[!is.na(`Gene symbol`) & nzchar(`Gene symbol`) & !grepl("///", `Gene symbol`)]
  setNames(harmonise(tab$`Gene symbol`), tab$ID)
}

load_gse65021 <- function() {
  x <- read_series_matrix(file.path(EXT, "GSE65021_series_matrix.txt.gz"))
  sym <- read_annotation(file.path(EXT, "GPL10558.annot.gz"))
  x <- x[rownames(x) %in% names(sym), ]
  lg <- log2(pmax(x, 0) + 1)
  o <- order(rowMeans(lg), decreasing = TRUE)        # highest-mean probe per gene (10_cetuximab/01 rule)
  g <- sym[rownames(x)[o]]
  first <- o[!duplicated(g)]
  lin <- x[first, ]
  rownames(lin) <- sym[rownames(x)[first]]
  msg("GSE65021 linear intensities:", ncol(lin), "samples x", nrow(lin), "genes")
  list(counts = t(pmax(lin, 0)), log2 = log2(lin + 1))
}

# ---- 3. ESTIMATE -------------------------------------------------------------------------------------
run_estimate <- function(log2mat, cohort) {
  if (!requireNamespace("tidyestimate", quietly = TRUE)) return(invisible(NULL))
  df <- data.frame(hgnc_symbol = rownames(log2mat), log2mat, check.names = FALSE)
  df <- tidyestimate::filter_common_genes(df, id = "hgnc_symbol", tidy = TRUE, tell_missing = FALSE,
                                          find_alias = TRUE)
  sc <- tidyestimate::estimate_score(df, is_affymetrix = TRUE)
  setnames(setDT(sc), c("sample", "stromal_estimate", "immune_estimate", "estimate_score", "purity_estimate_affy"))
  fwrite(sc, file.path(OUT, paste0(cohort, "_estimate.tsv")), sep = "\t")
}

# ---- 4. deconvolution --------------------------------------------------------------------------------
deconvolve <- function(ref, bulk, cohort) {
  log_file <- file.path(OUT, paste0("run_log_", cohort, ".txt"))
  sink(log_file, split = TRUE)
  on.exit(sink(), add = TRUE)
  msg("cohort", cohort, "| BayesPrism", as.character(packageVersion("BayesPrism")), "| cores", N_CORES)
  mix <- bulk$counts
  genes <- intersect(union(ref$markers, intersect(INTEREST, colnames(ref$X))), colnames(mix))
  fwrite(data.table(gene = genes, is_marker = genes %in% ref$markers, is_scored = genes %in% INTEREST),
         file.path(OUT, paste0("reference_genes_used_", cohort, ".tsv")), sep = "\t")
  msg("genes used:", length(genes), "(markers", sum(genes %in% ref$markers), ", scored genes",
      sum(genes %in% INTEREST), "of", length(INTEREST), ")")
  msg("scored genes absent from reference or bulk:", paste(setdiff(INTEREST, genes), collapse = ", "))

  pr <- new.prism(reference = ref$X[, genes], mixture = mix[, genes], input.type = "count.matrix",
                  cell.type.labels = ref$type, cell.state.labels = ref$state, key = "Malignant",
                  outlier.cut = 0.01, outlier.fraction = 0.1)
  msg("genes after bulk outlier filter:", ncol(pr@mixture))
  msg("scored genes removed by the outlier filter:",
      paste(setdiff(intersect(INTEREST, genes), colnames(pr@mixture)), collapse = ", "))
  wr <- function(m, name, gz = FALSE) {
    d <- data.table(sample = rownames(m), as.data.frame(m, check.names = FALSE))
    fwrite(d, file.path(OUT, paste0(cohort, "_", name, if (gz) ".tsv.gz" else ".tsv")), sep = "\t")
  }
  # run.prism split into its two steps, with identical settings and seeds, so that the first step is
  # written to disk before the second starts (the TCGA run takes about five hours)
  t0 <- Sys.time()
  gibbs.control <- BayesPrism:::valid.gibbs.control(list(n.cores = N_CORES))
  opt.control <- BayesPrism:::valid.opt.control(list(n.cores = N_CORES))
  stage1 <- if (!is.null(CACHE)) sub("[.]rds$", paste0("_", cohort, "_stage1.rds"), CACHE) else NULL
  stage_files <- function() {   # the Gibbs workers read each sample from the session temp directory
    tmp.dir <- tempdir(check = TRUE)
    for (n in seq_len(nrow(pr@mixture))) {
      X_n <- pr@mixture[n, ]
      save(X_n, file = paste(tmp.dir, "/mixture_", n, ".rdata", sep = ""))
    }
  }
  if (!is.null(stage1) && file.exists(stage1)) {
    msg("first Gibbs step loaded from", stage1)
    ini.ct <- readRDS(stage1)
  } else {
    stage_files()
    ini.cs <- BayesPrism:::run.gibbs(new("gibbsSampler", reference = pr@phi_cellState, X = pr@mixture,
                                         gibbs.control = gibbs.control), final = FALSE)
    ini.ct <- BayesPrism:::mergeK(jointPost.obj = ini.cs, map = pr@map)
    wr(ini.cs@theta, "fractions_state")
    rm(ini.cs)
    gc()
    wr(ini.ct@theta, "fractions_initial")
    wr(ini.ct@Z[, , "Malignant"], "Z_malignant", gz = TRUE)
    wr(ini.ct@Z[, , "Fibroblast"], "Z_fibroblast", gz = TRUE)
    if (!is.null(stage1)) saveRDS(ini.ct, stage1)
    msg("first Gibbs step finished in", round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1), "min")
  }
  psi <- BayesPrism:::updateReference(Z = ini.ct@Z, phi_prime = pr@phi_cellType, map = pr@map,
                                      key = pr@key, opt.control = opt.control)
  stage_files()
  theta_f <- BayesPrism:::run.gibbs(new("gibbsSampler", reference = psi, X = pr@mixture,
                                        gibbs.control = gibbs.control), final = TRUE)
  msg("BayesPrism finished in", round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1), "min")
  wr(theta_f@theta, "fractions")       # final theta, cell-type level (updated Gibbs)
  wr(theta_f@theta.cv, "fractions_cv")
  invisible(theta_f)
}

# ---- main --------------------------------------------------------------------------------------------
main <- function() {
  msg("BayesPrism", as.character(packageVersion("BayesPrism")),
      "| GitHub", packageDescription("BayesPrism")$RemoteSha, "| cohorts", paste(COHORTS, collapse = ", "))
  ref <- build_reference()
  for (cohort in COHORTS) {
    bulk <- switch(cohort, "TCGA-HNSC" = load_tcga(), "GSE65021" = load_gse65021(),
                   stop("unknown cohort ", cohort))
    run_estimate(bulk$log2, cohort)
    deconvolve(ref, bulk, cohort)
    rm(bulk)
    gc()
  }
  writeLines(capture.output(sessionInfo()), file.path(OUT, "session_info.txt"))
  msg("done")
}

main()
