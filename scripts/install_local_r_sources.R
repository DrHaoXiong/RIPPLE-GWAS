#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
source_dir <- if (length(args) >= 1) args[[1]] else "/path/to/ripple_private_workspace/02_environment/tools/r_cran_sources"
user_lib <- Sys.getenv("R_LIBS_USER")
if (user_lib == "") {
  user_lib <- file.path(Sys.getenv("HOME"), "R", "x86_64-pc-linux-gnu-library", paste(R.version$major, R.version$minor, sep = "."))
}
dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(user_lib, .libPaths()))

packages <- c(
  "cli_3.6.6.tar.gz",
  "glue_1.8.1.tar.gz",
  "rlang_1.3.0.tar.gz",
  "magrittr_2.0.5.tar.gz",
  "pkgconfig_2.0.3.tar.gz",
  "lifecycle_1.0.5.tar.gz",
  "vctrs_0.7.3.tar.gz",
  "cpp11_0.5.5.tar.gz",
  "igraph_2.3.3.tar.gz"
)

for (pkg in packages) {
  tarball <- file.path(source_dir, pkg)
  if (!file.exists(tarball)) {
    stop("Missing local source package: ", tarball)
  }
  cat("Installing", tarball, "\n")
  install.packages(tarball, repos = NULL, type = "source", lib = user_lib)
}

cat("Installed igraph version:", as.character(utils::packageVersion("igraph")), "\n")
