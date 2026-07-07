#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop("Usage: run_dmgwas_external_baseline.R <network.tsv> <gene_p.tsv> <out_dir> <trait> [r]")
}

network_path <- args[[1]]
gene_p_path <- args[[2]]
out_dir <- args[[3]]
trait <- args[[4]]
r_value <- if (length(args) >= 5) as.numeric(args[[5]]) else 0.1

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))

suppressPackageStartupMessages(library(dmGWAS))
suppressPackageStartupMessages(library(igraph))

set.seed(20260725)
network <- read.table(network_path, header = TRUE, sep = "\t", stringsAsFactors = FALSE, quote = "")
geneweight <- read.table(gene_p_path, header = TRUE, sep = "\t", stringsAsFactors = FALSE, quote = "")

required_network <- c("node1", "node2")
required_geneweight <- c("gene", "weight")
if (!all(required_network %in% colnames(network))) {
  stop("Network input must contain node1 and node2 columns")
}
if (!all(required_geneweight %in% colnames(geneweight))) {
  stop("Gene weight input must contain gene and weight columns")
}

network <- unique(network[, required_network])
geneweight <- geneweight[, required_geneweight]
geneweight$weight <- as.numeric(geneweight$weight)
geneweight <- geneweight[is.finite(geneweight$weight), ]
geneweight$weight <- pmin(pmax(geneweight$weight, 1e-16), 1 - 1e-16)
geneweight <- geneweight[geneweight$weight > 0 & geneweight$weight < 1, ]
geneweight <- geneweight[!duplicated(geneweight$gene), ]

covered <- unique(c(network$node1, network$node2))
geneweight <- geneweight[geneweight$gene %in% covered, ]
network <- network[network$node1 %in% geneweight$gene & network$node2 %in% geneweight$gene, ]

run_dir <- file.path(out_dir, paste0(trait, "_dmGWAS_run"))
dir.create(run_dir, recursive = TRUE, showWarnings = FALSE)
old_wd <- getwd()
setwd(run_dir)
on.exit(setwd(old_wd), add = TRUE)

cat("dmGWAS input summary\n")
cat("trait", trait, "\n")
cat("genes", nrow(geneweight), "\n")
cat("edges", nrow(network), "\n")
cat("r", r_value, "\n")

result <- dms(network = network, geneweight = geneweight, expr1 = NULL, expr2 = NULL, d = 1, r = r_value)

ordered <- result$zi.ordered
ordered <- as.data.frame(ordered, stringsAsFactors = FALSE)
colnames(ordered) <- c("seed_gene", "Zm", "Zn", "zcount")
ordered$Zm <- as.numeric(ordered$Zm)
ordered$Zn <- as.numeric(ordered$Zn)
ordered$zcount <- as.numeric(ordered$zcount)

module_rows <- list()
for (i in seq_len(nrow(ordered))) {
  seed <- as.character(ordered$seed_gene[[i]])
  genes <- result$genesets.clear[[seed]]
  if (is.null(genes)) {
    next
  }
  size <- length(genes)
  null_values <- result$genesets.length.null.dis[[as.character(size)]]
  empirical_p <- if (!is.null(null_values) && length(null_values) > 0) {
    (1 + sum(null_values >= ordered$Zm[[i]])) / (1 + length(null_values))
  } else {
    NA_real_
  }
  module_rows[[length(module_rows) + 1]] <- data.frame(
    trait = trait,
    baseline_method = "dmGWAS_3.0_node_only",
    seed_gene = seed,
    module_size = size,
    Zm = ordered$Zm[[i]],
    Zn = ordered$Zn[[i]],
    zcount = ordered$zcount[[i]],
    empirical_p = empirical_p,
    module_genes = paste(genes, collapse = ";"),
    stringsAsFactors = FALSE
  )
}
modules <- if (length(module_rows) > 0) {
  do.call(rbind, module_rows)
} else {
  data.frame()
}
modules <- modules[order(modules$Zn, decreasing = TRUE), ]

summary <- data.frame(
  trait = trait,
  baseline_method = "dmGWAS_3.0_node_only",
  package_version = as.character(utils::packageVersion("dmGWAS")),
  igraph_version = as.character(utils::packageVersion("igraph")),
  n_input_genes = nrow(geneweight),
  n_input_edges = nrow(network),
  n_modules = nrow(modules),
  n_empirical_p_le_0_05 = sum(modules$empirical_p <= 0.05, na.rm = TRUE),
  n_Zn_ge_2_5 = sum(modules$Zn >= 2.5, na.rm = TRUE),
  top_seed = if (nrow(modules) > 0) modules$seed_gene[[1]] else NA_character_,
  top_module_size = if (nrow(modules) > 0) modules$module_size[[1]] else NA_integer_,
  top_Zn = if (nrow(modules) > 0) modules$Zn[[1]] else NA_real_,
  top_empirical_p = if (nrow(modules) > 0) modules$empirical_p[[1]] else NA_real_,
  r = r_value,
  stringsAsFactors = FALSE
)

write.table(modules, file = file.path(out_dir, paste0(trait, ".dmGWAS_modules.tsv")), sep = "\t", quote = FALSE, row.names = FALSE)
write.table(summary, file = file.path(out_dir, paste0(trait, ".dmGWAS_summary.tsv")), sep = "\t", quote = FALSE, row.names = FALSE)
