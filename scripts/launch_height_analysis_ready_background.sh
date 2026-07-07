#!/usr/bin/env bash
set -euo pipefail

cd /path/to/ripple_private_workspace/04_private_src/ripple_v1

OUT_DIR=/path/to/ripple_private_workspace/30_analysis/height_irn_analysis_ready
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

PID_FILE="${LOG_DIR}/height_analysis_ready.pid"
OUTER_LOG="${LOG_DIR}/height_analysis_ready.outer.log"

rm -f "${PID_FILE}"
setsid bash scripts/run_height_analysis_ready.sh >"${OUTER_LOG}" 2>&1 < /dev/null &
printf '%s\n' "$!" | tee "${PID_FILE}"
