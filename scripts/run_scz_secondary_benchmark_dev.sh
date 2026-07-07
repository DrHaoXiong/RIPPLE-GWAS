#!/usr/bin/env bash
# Dev-scale SCZ secondary cross-domain benchmark for RIPPLE-GWAS V1.
#
# Purpose:
#   Run SCZ default-with-MHC and no-MHC sensitivity at a moderate null scale.
#   These outputs are for cross-trait benchmark triage, not final manuscript lock.
#
# Outputs:
#   /path/to/ripple_private_workspace/30_analysis/scz_with_mhc_string_dev500
#   /path/to/ripple_private_workspace/30_analysis/scz_no_mhc_string_dev500
#
# Usage:
#   bash scripts/run_scz_secondary_benchmark_dev.sh all
#   bash scripts/run_scz_secondary_benchmark_dev.sh with-mhc
#   bash scripts/run_scz_secondary_benchmark_dev.sh no-mhc

set -euo pipefail

REPO="/path/to/ripple_private_workspace/04_private_src/ripple_v1"
PRIVATE_ROOT="/path/to/ripple_private_workspace"
ANALYSIS="${PRIVATE_ROOT}/30_analysis"
LOG_DIR="${ANALYSIS}/logs/scz_secondary_benchmark_dev"

mkdir -p "${LOG_DIR}"
cd "${REPO}"

run_scz_with_mhc() {
  python scripts/run_trait_ld_analysis.py \
    --trait SCZ_WITH_MHC \
    --gwas "${PRIVATE_ROOT}/20_processed_data/gwas_qc/harmonized_hm3_with_mhc/SCZ.tsv.gz" \
    --out-dir "${ANALYSIS}/scz_with_mhc_string_dev500" \
    --force \
    --graph-name string_ppi \
    --n-null 200 \
    --n-degree-stratified-null 200 \
    --n-degree-matched-node-null 500 \
    --n-degree-graph-null 100 \
    --enable-diffusion \
    --n-diffusion-null 500 \
    --diffusion-degree-bins 20 \
    --save-null-distributions \
    --score-transform-sensitivity \
    --degree-residualized-sensitivity \
    --n-module-random-null 300 \
    --n-module-degree-matched-null 300 \
    --n-module-selection-aware-null 300 \
    --n-module-degree-graph-null 50 \
    --n-pathway-random-null 300 \
    --n-pathway-degree-matched-null 300 \
    --seed 20260702 \
    2>&1 | tee "${LOG_DIR}/scz_with_mhc_string_dev500.log"
}

run_scz_no_mhc() {
  python scripts/run_trait_ld_analysis.py \
    --trait SCZ \
    --gwas "${PRIVATE_ROOT}/20_processed_data/gwas_qc/core_hm3_no_mhc/SCZ.tsv.gz" \
    --out-dir "${ANALYSIS}/scz_no_mhc_string_dev500" \
    --force \
    --graph-name string_ppi \
    --n-null 200 \
    --n-degree-stratified-null 200 \
    --n-degree-matched-node-null 500 \
    --n-degree-graph-null 100 \
    --enable-diffusion \
    --n-diffusion-null 500 \
    --diffusion-degree-bins 20 \
    --save-null-distributions \
    --score-transform-sensitivity \
    --degree-residualized-sensitivity \
    --n-module-random-null 300 \
    --n-module-degree-matched-null 300 \
    --n-module-selection-aware-null 300 \
    --n-module-degree-graph-null 50 \
    --n-pathway-random-null 300 \
    --n-pathway-degree-matched-null 300 \
    --seed 20260702 \
    2>&1 | tee "${LOG_DIR}/scz_no_mhc_string_dev500.log"
}

target="${1:-all}"
case "${target}" in
  all)
    run_scz_with_mhc
    run_scz_no_mhc
    ;;
  with-mhc)
    run_scz_with_mhc
    ;;
  no-mhc)
    run_scz_no_mhc
    ;;
  *)
    echo "Unknown target: ${target}" >&2
    exit 2
    ;;
esac
