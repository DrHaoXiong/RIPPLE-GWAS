#!/usr/bin/env bash
set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate ripple

cd /path/to/ripple_private_workspace/04_private_src/ripple_v1

OUT_DIR=/path/to/ripple_private_workspace/30_analysis/dm_retinopathy_exmore_no_mhc_no_apoe_analysis_ready
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/dm_retinopathy_no_mhc_no_apoe_analysis_ready.log"
EXIT_FILE="${LOG_DIR}/dm_retinopathy_no_mhc_no_apoe_analysis_ready.exit"

rm -f "${EXIT_FILE}"

set +e
python scripts/run_trait_ld_analysis.py \
  --trait DM_RETINOPATHY_EXMORE_NO_MHC_NO_APOE \
  --gwas /path/to/ripple_private_workspace/20_processed_data/gwas_qc/core_hm3_no_mhc_no_apoe/DM_RETINOPATHY_EXMORE.tsv.gz \
  --mapping /path/to/ripple_private_workspace/30_analysis/dm_retinopathy_exmore_no_mhc_no_apoe_mvp/tables/DM_RETINOPATHY_EXMORE_NO_MHC_NO_APOE.gene_body_mapping.tsv.gz \
  --ld-cache-overlay-dir /path/to/ripple_private_workspace/30_analysis/dm_retinopathy_exmore_no_mhc_no_apoe_mvp/ld_cache_1000G_EUR_gap_overlay \
  --out-dir "${OUT_DIR}" \
  --force \
  >"${LOG_FILE}" 2>&1
status=$?
set -e

printf '%s\n' "${status}" >"${EXIT_FILE}"
exit "${status}"
