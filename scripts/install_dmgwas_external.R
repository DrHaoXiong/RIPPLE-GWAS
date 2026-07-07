#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
tool_dir <- if (length(args) >= 1) args[[1]] else "/path/to/ripple_private_workspace/02_environment/tools/dmgwas_external"
user_lib <- Sys.getenv("R_LIBS_USER")
if (user_lib == "") {
  user_lib <- file.path(Sys.getenv("HOME"), "R", paste(R.version$platform, "library", sep = "-"), paste(R.version$major, R.version$minor, sep = "."))
}
dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(user_lib, .libPaths()))

cat("R version:", R.version.string, "\n")
cat("Library paths:\n")
cat(.libPaths(), sep = "\n")
cat("\n")

if (!requireNamespace("igraph", quietly = TRUE)) {
  cat("Installing igraph into", user_lib, "\n")
  install.packages("igraph", repos = "https://cloud.r-project.org", lib = user_lib)
}
if (!requireNamespace("igraph", quietly = TRUE)) {
  stop("igraph installation failed")
}

tarball <- file.path(tool_dir, "dmGWAS_3.0.tar.gz")
if (!file.exists(tarball)) {
  stop("dmGWAS tarball not found: ", tarball)
}

if (!requireNamespace("dmGWAS", quietly = TRUE)) {
  cat("Installing dmGWAS from", tarball, "\n")
  install.packages(tarball, repos = NULL, type = "source", lib = user_lib)
}
if (!requireNamespace("dmGWAS", quietly = TRUE)) {
  stop("dmGWAS installation failed")
}

cat("Installed package versions:\n")
cat("igraph", as.character(utils::packageVersion("igraph")), "\n")
cat("dmGWAS", as.character(utils::packageVersion("dmGWAS")), "\n")
