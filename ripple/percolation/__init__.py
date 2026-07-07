"""Rank-fraction percolation."""

from ripple.percolation.auc import percolation_auc, percolation_curve
from ripple.percolation.calibration import (
    classify_percolation_architecture,
    component_row_for_nodes,
    compute_degree_matched_node_percolation_null,
    is_negative_null_result,
    is_positive_null_result,
    prepare_degree_matched_rank_sets,
    summarize_percolation_threshold_robustness,
    summarize_percolation_null,
)
from ripple.percolation.ranking import rank_fraction_grid, rank_nodes_by_score, selected_nodes_at_fraction

__all__ = [
    "classify_percolation_architecture",
    "component_row_for_nodes",
    "compute_degree_matched_node_percolation_null",
    "is_negative_null_result",
    "is_positive_null_result",
    "percolation_auc",
    "percolation_curve",
    "prepare_degree_matched_rank_sets",
    "rank_fraction_grid",
    "rank_nodes_by_score",
    "selected_nodes_at_fraction",
    "summarize_percolation_threshold_robustness",
    "summarize_percolation_null",
]
