#!/usr/bin/env bash
set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate ripple

cd /path/to/ripple_private_workspace/04_private_src/ripple_v1

LOG_DIR=/path/to/ripple_private_workspace/20_processed_data/gwas_qc/logs
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/qc_gwas_all.log"
EXIT_FILE="${LOG_DIR}/qc_gwas_all.exit"

rm -f "${EXIT_FILE}"

set +e
python scripts/qc_gwas.py --force --chunksize 500000 >"${LOG_FILE}" 2>&1
status=$?
set -e

printf '%s\n' "${status}" >"${EXIT_FILE}"
exit "${status}"
