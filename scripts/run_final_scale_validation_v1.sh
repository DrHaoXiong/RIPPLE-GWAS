#!/usr/bin/env bash
# Final-scale validation launcher for RIPPLE-GWAS V1.
#
# Purpose:
#   Run the targeted validation set needed to move from Final Deliverable v0.1
#   to manuscript-lockable Final Deliverable v1.0.
#
# Scope:
#   Private workspace only. Outputs are written under:
#   /path/to/ripple_private_workspace/30_analysis
#
# Usage:
#   bash scripts/run_final_scale_validation_v1.sh all
#   bash scripts/run_final_scale_validation_v1.sh primary
#   bash scripts/run_final_scale_validation_v1.sh graph
#   bash scripts/run_final_scale_validation_v1.sh synthetic

set -euo pipefail

REPO="/path/to/ripple_private_workspace/04_private_src/ripple_v1"
PRIVATE_ROOT="/path/to/ripple_private_workspace"
ANALYSIS="${PRIVATE_ROOT}/30_analysis"
LOG_DIR="${ANALYSIS}/logs/final_scale_v1"

mkdir -p "${LOG_DIR}"
cd "${REPO}"

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate ripple

run_primary_default() {
  python scripts/run_trait_ld_analysis.py \
    --trait DR_MVP \
    --gwas "${PRIVATE_ROOT}/20_processed_data/gwas_qc/core_hm3_no_mhc/DR_MVP.tsv.gz" \
    --out-dir "${ANALYSIS}/dr_mvp_string_final5000" \
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
    2>&1 | tee "${LOG_DIR}/dr_mvp_string_final5000.log"
}

run_primary_no_mhc_no_apoe() {
  python scripts/run_trait_ld_analysis.py \
    --trait DR_MVP_NO_MHC_NO_APOE \
    --gwas "${PRIVATE_ROOT}/20_processed_data/gwas_qc/core_hm3_no_mhc_no_apoe/DR_MVP.tsv.gz" \
    --out-dir "${ANALYSIS}/dr_mvp_no_mhc_no_apoe_final5000" \
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
    2>&1 | tee "${LOG_DIR}/dr_mvp_no_mhc_no_apoe_final5000.log"
}

run_fvm_weighted_diffusion() {
  python scripts/run_diffusion_kernel.py \
    --trait DR_MVP_FVM_VASCULAR_WEIGHTED \
    --lcc-scores "${ANALYSIS}/dr_mvp_fvm_vascular_weighted_analysis_ready/tables/DR_MVP_FVM_VASCULAR_WEIGHTED.lcc_gene_scores.1000G_LD.residualized.tsv.gz" \
    --out-dir "${ANALYSIS}/dr_mvp_graph_sensitivity/fvm_vascular_weighted_diffusion_final5000" \
    --graph-name fvm_vascular_weighted \
    --graph-edge-list "${PRIVATE_ROOT}/20_processed_data/reference_graphs/fvm_vascular_string/tables/fvm_vascular_weighted_string.edges.tsv.gz" \
    --weighted-laplacian \
    --n-diffusion-null 5000 \
    --diffusion-degree-bins 20 \
    --save-null-distributions \
    --seed 20260616 \
    2>&1 | tee "${LOG_DIR}/fvm_vascular_weighted_diffusion_final5000.log"
}

run_retina_min20_diffusion() {
  python scripts/run_diffusion_kernel.py \
    --trait DR_MVP_RETINA_STRING_MIN20 \
    --lcc-scores "${ANALYSIS}/dr_mvp_retina_string_min20_analysis_ready/tables/DR_MVP_RETINA_STRING_MIN20.lcc_gene_scores.1000G_LD.residualized.tsv.gz" \
    --out-dir "${ANALYSIS}/dr_mvp_graph_sensitivity/retina_string_min20_diffusion_final5000" \
    --graph-name retina_string_min20 \
    --graph-edge-list "${PRIVATE_ROOT}/20_processed_data/reference_graphs/retina_string_filtered_min20/tables/retina_string_filtered.edges.tsv.gz" \
    --n-diffusion-null 5000 \
    --diffusion-degree-bins 20 \
    --save-null-distributions \
    --seed 20260616 \
    2>&1 | tee "${LOG_DIR}/retina_string_min20_diffusion_final5000.log"
}

run_synthetic_validation() {
  python scripts/validate_ripple_graph_statistics.py \
    --out-dir "${ANALYSIS}/synthetic_graph_statistics_validation_final_n100_null500" \
    --n-replicates 100 \
    --n-null 500 \
    --seed 20260616 \
    2>&1 | tee "${LOG_DIR}/synthetic_graph_statistics_validation_final_n100_null500.log"
}

target="${1:-all}"
case "${target}" in
  all)
    run_primary_default
    run_primary_no_mhc_no_apoe
    run_fvm_weighted_diffusion
    run_retina_min20_diffusion
    run_synthetic_validation
    ;;
  primary)
    run_primary_default
    run_primary_no_mhc_no_apoe
    ;;
  graph)
    run_fvm_weighted_diffusion
    run_retina_min20_diffusion
    ;;
  synthetic)
    run_synthetic_validation
    ;;
  dr-default)
    run_primary_default
    ;;
  dr-no-mhc-no-apoe)
    run_primary_no_mhc_no_apoe
    ;;
  fvm-weighted)
    run_fvm_weighted_diffusion
    ;;
  retina-min20)
    run_retina_min20_diffusion
    ;;
  *)
    echo "Unknown target: ${target}" >&2
    exit 2
    ;;
esac
