# RIPPLE-GWAS Prototype

Python prototype for RIPPLE-GWAS weak-signal graph aggregation and module diagnostics.

Current module-layer development focus: **RIPPLE-D V1.4c**.

V1.4c is a diagnostic layer for separating sparse top-locus pathway overlap from candidate distributed weak-signal module evidence. It is not yet a manuscript-level validated module discovery claim layer.

## Development environment

```bash
wsl -d Ubuntu
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ripple
cd /mnt/d/RIPPLE/RIPPLE_private/04_private_src/ripple_v1
```

## Intended install

```bash
pip install -e .
pytest
```

## Current RIPPLE-D runner

```bash
python scripts/run_v14c_ripple_d_module_rescue.py \
  --traits DR_MVP \
  --n-null 200 \
  --locus-window-grid 50000 100000 250000 500000 1000000 \
  --resume
```

Key V1.4c safeguards:

- module-gene-count-preserving locus nulls are required for manuscript-facing runs;
- locus-membership rank evidence is reported separately from module-specific rank evidence;
- distributed gates use module-specific rank evidence, not whole-locus membership rank alone;
- null replacement sampling is audited with `null_with_replacement_rate` and `null_gene_count_match_degraded`;
- annotation-density matching can be disabled for sensitivity with `--disable-annotation-matching`;
- external LD-block or clumped locus definitions can be supplied through `--locus-id-column`.

## V1 principles

- signed directional gene signal and unsigned association-strength signal are separate streams;
- primary unsigned ranking uses technically residualized normal scores;
- graph degree is controlled by nulls and sensitivity, not primary residualization;
- null replicates must repeat the full observed pipeline.
