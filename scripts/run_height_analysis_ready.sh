#!/usr/bin/env bash
set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate ripple

cd /path/to/ripple_private_workspace/04_private_src/ripple_v1

OUT_DIR=/path/to/ripple_private_workspace/30_analysis/height_irn_analysis_ready
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/height_analysis_ready.log"
EXIT_FILE="${LOG_DIR}/height_analysis_ready.exit"

rm -f "${EXIT_FILE}"

set +e
python scripts/run_height_ld_null_mvp.py \
  --out-dir "${OUT_DIR}" \
  --force \
  --n-null 100 \
  --ld-cache-overlay-dir /path/to/ripple_private_workspace/30_analysis/height_irn_mvp/ld_cache_1000G_EUR_large_gene_full \
  --n-degree-stratified-null 100 \
  --n-degree-matched-node-null 500 \
  --n-degree-graph-null 20 \
  --degree-graph-nswap-per-edge 1.0 \
  >"${LOG_FILE}" 2>&1
status=$?
set -e

printf '%s\n' "${status}" >"${EXIT_FILE}"
exit "${status}"
