#!/usr/bin/env bash
set -euo pipefail

cd /path/to/ripple_private_workspace/04_private_src/ripple_v1
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ripple

python scripts/run_dmgwas_external_baseline.py \
  --traits DR_MVP DR_MVP_NO_MHC_NO_APOE SCZ \
  --timeout-seconds 0 \
  --force
