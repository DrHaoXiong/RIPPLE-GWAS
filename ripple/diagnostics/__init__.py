"""Bias and reporting diagnostics."""

from ripple.diagnostics.reporting import (
    build_trait_suitability_diagnostic,
    render_trait_architecture_markdown,
)
from ripple.diagnostics.score_checks import (
    gene_score_clipping_diagnostics,
    gene_score_transform_sensitivity,
    residualization_diagnostics,
)

__all__ = [
    "build_trait_suitability_diagnostic",
    "gene_score_clipping_diagnostics",
    "gene_score_transform_sensitivity",
    "residualization_diagnostics",
    "render_trait_architecture_markdown",
]
