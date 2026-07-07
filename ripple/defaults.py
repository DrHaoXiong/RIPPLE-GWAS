"""Frozen RIPPLE V1 defaults."""

from __future__ import annotations

DEFAULT_LD_SHRINKAGE: float = 0.05
LD_SHRINKAGE_SENSITIVITY: tuple[float, ...] = (0.01, 0.05, 0.10)

RANK_FRACTION_GRID: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20)

PRIMARY_RESIDUALIZATION_COVARIATES: tuple[str, ...] = (
    "log_gene_length",
    "log_mapped_snp_count",
    "log_m_eff",
    "local_ld_score",
    "mappability",
)

DEGREE_RESIDUALIZATION_COVARIATE: str = "log_graph_degree"

REQUIRED_PIPELINE_STAGES: tuple[str, ...] = (
    "snp_to_gene_mapping",
    "ld_aware_signed_or_quadratic_scoring",
    "normal_score_transform",
    "residualization",
    "graph_component_filtering",
    "gsp",
    "percolation",
)

SPECIAL_REGION_LABELS: tuple[str, ...] = ("MHC", "APOE")
