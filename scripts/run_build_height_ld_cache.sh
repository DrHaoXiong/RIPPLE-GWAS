#!/usr/bin/env bash
set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate ripple

cd /path/to/ripple_private_workspace/04_private_src/ripple_v1

LOG_DIR=/path/to/ripple_private_workspace/30_analysis/height_irn_mvp/logs
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/build_height_ld_cache.log"
EXIT_FILE="${LOG_DIR}/build_height_ld_cache.exit"

rm -f "${EXIT_FILE}"

set +e
python scripts/build_height_ld_cache.py \
  --out-dir /path/to/ripple_private_workspace/30_analysis/height_irn_mvp/ld_cache_1000G_EUR \
  --max-snps 1000 \
  --force \
  >"${LOG_FILE}" 2>&1
status=$?
set -e

printf '%s\n' "${status}" >"${EXIT_FILE}"
exit "${status}"
