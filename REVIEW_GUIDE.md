# RIPPLE-GWAS Code Review Guide

This snapshot is being shared to invite methodological review. The highest-risk
question is whether the current graph and module statistics capture distributed
weak GWAS signal, or whether they mostly re-label the strongest GWAS loci.

## Core Files

- `ripple/signals/signed.py`: signed LD-aware burden signal.
- `ripple/signals/unsigned.py`: unsigned LD-aware quadratic association score.
- `ripple/signals/residualize.py`: technical residualization.
- `ripple/percolation/`: rank-fraction percolation statistics and nulls.
- `ripple/graph_diffusion.py`: heat-kernel diffusion statistics.
- `ripple/modules/anchored.py`: anchored biological module tests.
- `ripple/modules/discovery.py`: de novo local module discovery diagnostics.
- `ripple/policy.py` and `ripple/config/claim_policy.yaml`: claim thresholds
  and language policy.

## Main Review Questions

1. Are the LD-aware gene scores mathematically calibrated under realistic LD?
2. Are empirical nulls preserving the correct structures: LD, gene size,
   SNP-to-gene overlap, graph degree, and graph topology?
3. Does technical residualization remove nuisance architecture without removing
   biology?
4. Are percolation and diffusion statistics measuring graph-domain aggregation
   rather than degree or top-locus artifacts?
5. Are anchored module tests sufficiently robust to leave-top-k and
   top-locus-dominated signals?
6. Should the V1 module layer be reframed as top-locus-aware pathway
   prioritization rather than weak-signal module discovery?

## Recommended First Tests

```bash
pip install -e ".[dev]"
ruff check .
pytest -q
```

The full data pipeline requires external GWAS summary statistics and reference
resources that are not included in this repository.
