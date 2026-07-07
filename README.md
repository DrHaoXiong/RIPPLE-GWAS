# RIPPLE-GWAS

RIPPLE-GWAS is a research prototype for calibration-first graph-domain analysis of
GWAS summary statistics.

The current public snapshot is intended for code review and methodological
cross-review. It contains the core Python package, analysis scripts, and tests,
but does not redistribute GWAS summary statistics, LD reference panels, STRING
files, single-cell data, intermediate analysis outputs, or manuscript-private
materials.

## Current Positioning

RIPPLE-GWAS tests whether weak gene-level GWAS signals show graph-domain
aggregation under explicit calibration layers:

- LD-aware gene-level scoring from summary statistics.
- Technical residualization of gene-level association scores.
- Degree-aware graph nulls and degree-preserving topology sensitivity.
- Claim-tier reporting that separates graph-domain aggregation from
  topology-specific discovery.
- Anchored biological module diagnostics with robustness checks.
- RIPPLE-D V1.3 diagnostics that separate raw gene-set enrichment from
  locus-aware distributed weak-signal module evidence.

The current module layer is under active review. Internal diagnostics suggested
that some anchored module signals can be driven by a small number of top-ranked
GWAS genes or loci. The V1.3 RIPPLE-D layer therefore keeps the old
`sqrt(n) * mean` statistic as a raw enrichment component and adds locus-aware
score capping, pseudo-locus collapse, effective-locus contribution diagnostics,
leave-top-locus checks, moderate-locus burden, and locus-aware empirical nulls.
Reviewers should pay particular attention to whether this redesign truly
captures distributed weak-signal architecture rather than sparse top-locus
overlap.

## Repository Contents

- `ripple/`: core Python package.
- `scripts/`: analysis, audit, benchmark, and manuscript-support scripts.
- `tests/`: unit and regression tests.
- `ripple/config/claim_policy.yaml`: machine-readable claim thresholds and
  language guardrails.

## What Is Not Included

The following are intentionally excluded from this public snapshot:

- raw GWAS summary statistics;
- 1000 Genomes / LD caches;
- STRING or other graph reference downloads;
- single-cell expression data;
- analysis outputs under private `30_analysis`;
- manuscript drafts and private review documents.

Several scripts still contain default local paths used during development.
Treat these as reproducibility scaffolds rather than portable defaults. For a
fresh environment, pass explicit command-line paths to the relevant scripts.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The development environment used Python 3.11. Some optional baseline scripts
require additional external software such as R packages, MAGMA, PascalX, or
dmGWAS-compatible tooling.

## Review Focus

External reviewers are especially asked to inspect:

1. LD-aware signed and unsigned gene-score construction.
2. Null generation and whether each null preserves the intended structure.
3. Residualization and degree-control strategy.
4. Percolation and diffusion statistics.
5. Anchored module statistics, top-gene leverage, and selection-aware
   calibration.
6. RIPPLE-D locus-aware distributed module gates and whether they are
   mathematically sufficient to demote top-locus artifacts.
7. Claim-tier policy and whether manuscript-facing language is appropriately
   conservative.

## Status

This is not a stable release. It is a public code-review snapshot of an active
methods project.
