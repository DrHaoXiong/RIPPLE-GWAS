#!/usr/bin/env bash
# Final-scale SCZ no-MHC secondary benchmark for RIPPLE-GWAS V1.
#
# Purpose:
#   Upgrade the SCZ no-MHC secondary cross-domain benchmark from dev500 to
#   final-scale null calibration, using the same null scale as DR_MVP final5000.
#
# Scope:
#   Private workspace only. This is a cross-domain benchmark, not DR-specific
#   biological annotation.
#
# Output:
#   /path/to/ripple_private_workspace/30_analysis/scz_no_mhc_string_final5000
#
# Usage:
#   bash scripts/run_scz_no_mhc_final_scale.sh

set -euo pipefail

REPO="/path/to/ripple_private_workspace/04_private_src/ripple_v1"
PRIVATE_ROOT="/path/to/ripple_private_workspace"
ANALYSIS="${PRIVATE_ROOT}/30_analysis"
LOG_DIR="${ANALYSIS}/logs/scz_no_mhc_final_scale"

mkdir -p "${LOG_DIR}"
cd "${REPO}"

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate ripple

python scripts/run_trait_ld_analysis.py \
  --trait SCZ \
  --gwas "${PRIVATE_ROOT}/20_processed_data/gwas_qc/core_hm3_no_mhc/SCZ.tsv.gz" \
  --out-dir "${ANALYSIS}/scz_no_mhc_string_final5000" \
  --force \
  --graph-name string_ppi \
  --n-null 1000 \
  --n-degree-stratified-null 1000 \
  --n-degree-matched-node-null 5000 \
  --n-degree-graph-null 500 \
  --enable-diffusion \
  --n-diffusion-null 5000 \
  --diffusion-degree-bins 20 \
  --save-null-distributions \
  --score-transform-sensitivity \
  --degree-residualized-sensitivity \
  --n-module-random-null 1000 \
  --n-module-degree-matched-null 1000 \
  --n-module-selection-aware-null 1000 \
  --n-module-degree-graph-null 200 \
  --n-pathway-random-null 1000 \
  --n-pathway-degree-matched-null 1000 \
  --seed 20260702 \
  2>&1 | tee "${LOG_DIR}/scz_no_mhc_string_final5000.log"
