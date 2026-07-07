# Scripts

These scripts are included for methodological review and reproducibility
orientation. Many were written during private analysis and may contain default
paths from the development workstation.

For public or collaborator use:

1. Prefer passing explicit `--gwas`, `--out-dir`, `--gene-set-file`,
   `--graph-edge-list`, and reference paths.
2. Do not assume that local default paths exist.
3. Do not commit raw GWAS, LD reference panels, STRING downloads, or generated
   analysis outputs.
4. Treat manuscript-package and benchmark scripts as audit scaffolding unless
   the required private result tables are separately provided.

The core package under `ripple/` is the main review target.
