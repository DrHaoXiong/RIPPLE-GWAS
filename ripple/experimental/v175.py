"""Conditional matched-locus statistics for the RIPPLE-D V1.7.5 diagnostic.

V1.7.5 is intentionally separate from the frozen V1.7 procedure.  It first
maps each module-locus score to a conditional normal score within the matched
null draws for that locus position.  Full and leave-top-one burdens are then
combined as a conjunction statistic and calibrated by the same null matrix.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from ripple.modules.adaptive import (
    AdaptiveLocusConfig,
    AdaptiveLocusContext,
    _collapse_observed_module,
    _prepare_matched_module_pools,
    _sample_matched_module,
)
from ripple.modules.anchored import AnchoredModuleLibrary, empirical_upper


@dataclass(frozen=True)
class ReducedLibraryResult:
    """A score-independent representative library and its membership audit."""

    library: AnchoredModuleLibrary
    audit: pd.DataFrame


def conditional_rank_normal_scores(
    observed: np.ndarray | pd.Series,
    null_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank-normalize each locus position over observed plus matched nulls.

    Including the observed value in every column's empirical distribution
    preserves exchangeability under the matched-locus null and avoids unstable
    MAD estimates in discrete or capped score strata.
    """

    observed_values = np.asarray(observed, dtype=float)
    null_values = np.asarray(null_matrix, dtype=float)
    if null_values.ndim != 2:
        raise ValueError("null_matrix must be two-dimensional")
    if null_values.shape[1] != observed_values.size:
        raise ValueError("observed and null_matrix must have the same number of loci")
    combined = np.vstack([observed_values, null_values])
    if not np.isfinite(combined).all():
        raise ValueError("conditional rank normalization requires finite values")
    n_rows = combined.shape[0]
    transformed = np.empty_like(combined, dtype=float)
    for column in range(combined.shape[1]):
        ranks = rankdata(combined[:, column], method="average")
        probabilities = (ranks - 0.5) / n_rows
        transformed[:, column] = norm.ppf(probabilities)
    return transformed[0], transformed[1:]


def full_and_leave_top1_stouffer(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return signed full and reselected leave-top-one Stouffer scores row-wise."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("matrix must contain at least one locus column")
    full = values.sum(axis=1) / math.sqrt(values.shape[1])
    if values.shape[1] == 1:
        leave = np.full(values.shape[0], np.nan)
    else:
        leave = (values.sum(axis=1) - values.max(axis=1)) / math.sqrt(values.shape[1] - 1)
    return full, leave


def conditional_joint_statistic(
    observed: np.ndarray | pd.Series,
    null_matrix: np.ndarray,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Calibrate a conjunction of full and leave-top-one conditional burdens.

    The conjunction is ``min(Z_full, Z_leave_top1)``.  A single extreme locus
    can raise the full burden but cannot by itself raise the conjunction.  The
    top locus is reselected in every null row, so leave-top selection is part of
    the calibrated statistic rather than an external hard gate.
    """

    observed_z, null_z = conditional_rank_normal_scores(observed, null_matrix)
    all_z = np.vstack([observed_z, null_z])
    full, leave = full_and_leave_top1_stouffer(all_z)
    if not np.isfinite(leave).all():
        raise ValueError("conditional joint statistic requires at least two loci")
    null_full = full[1:]
    null_leave = leave[1:]
    full_sd = float(np.std(null_full, ddof=1))
    leave_sd = float(np.std(null_leave, ddof=1))
    if full_sd <= 0 or leave_sd <= 0:
        raise ValueError("conditional burden null variance must be positive")
    full_mean = float(np.mean(null_full))
    leave_mean = float(np.mean(null_leave))
    full_standardized = (full - full_mean) / full_sd
    leave_standardized = (leave - leave_mean) / leave_sd
    joint = np.minimum(full_standardized, leave_standardized)
    observed_result = {
        "v175_conditional_full_stouffer_z": float(full_standardized[0]),
        "v175_conditional_leave_top1_stouffer_z": float(leave_standardized[0]),
        "v175_conditional_joint_stat": float(joint[0]),
        "v175_conditional_joint_empirical_p": empirical_upper(joint[1:], joint[0]),
    }
    null_result = {
        "v175_conditional_full_stouffer_z": full_standardized[1:],
        "v175_conditional_leave_top1_stouffer_z": leave_standardized[1:],
        "v175_conditional_joint_stat": joint[1:],
    }
    return observed_result, null_result


def _adaptive_positive_truncated_sum(
    matrix: np.ndarray,
    *,
    fractions: Sequence[float] = (0.10, 0.20, 0.50, 1.0),
    min_loci: int = 3,
) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("matrix must contain locus columns")
    positive = np.sort(np.clip(values, 0.0, None), axis=1)[:, ::-1]
    cumulative = np.cumsum(positive, axis=1)
    n_loci = values.shape[1]
    grid = sorted(
        {
            min(n_loci, max(1, int(min_loci), int(math.ceil(fraction * n_loci))))
            for fraction in fractions
        }
        | {n_loci}
    )
    candidates = np.column_stack([cumulative[:, k - 1] / math.sqrt(k) for k in grid])
    return np.max(candidates, axis=1)


def _remove_rowwise_max(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("leave-top-one requires at least two locus columns")
    order = np.argsort(values, axis=1, kind="stable")
    keep = order[:, :-1]
    return np.take_along_axis(values, keep, axis=1)


def _standardize_against_null(values: np.ndarray) -> np.ndarray:
    null = np.asarray(values[1:], dtype=float)
    sd = float(np.std(null, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("null variance must be positive")
    return (np.asarray(values, dtype=float) - float(np.mean(null))) / sd


def _higher_criticism(matrix: np.ndarray) -> np.ndarray:
    """Return a one-sided Higher-Criticism statistic for each matrix row."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("Higher Criticism requires at least two loci")
    p_values = np.sort(norm.sf(values), axis=1)
    n_loci = values.shape[1]
    expected = np.arange(1, n_loci + 1, dtype=float) / n_loci
    valid = (p_values >= 1.0 / n_loci) & (p_values <= 0.5)
    denominator = np.sqrt(np.clip(p_values * (1.0 - p_values), 1e-12, None))
    hc = math.sqrt(n_loci) * (expected[None, :] - p_values) / denominator
    hc[~valid] = -np.inf
    result = np.max(hc, axis=1)
    result[~np.isfinite(result)] = 0.0
    return result


def conditional_adaptive_robust_omnibus(
    observed: np.ndarray | pd.Series,
    null_matrix: np.ndarray,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Adapt between dense and sparse shifts while requiring top-locus robustness.

    Dense signed Stouffer, sparse positive top-k, and Higher-Criticism
    statistics each form a conjunction with their reselected leave-top-one
    counterpart.  The maximum over all conjunctions is repeated for every null
    row before empirical calibration.
    """

    observed_z, null_z = conditional_rank_normal_scores(observed, null_matrix)
    matrix = np.vstack([observed_z, null_z])
    dense_full, dense_leave = full_and_leave_top1_stouffer(matrix)
    dense_joint = np.minimum(
        _standardize_against_null(dense_full),
        _standardize_against_null(dense_leave),
    )
    sparse_full = _adaptive_positive_truncated_sum(matrix)
    sparse_leave = _adaptive_positive_truncated_sum(_remove_rowwise_max(matrix))
    sparse_joint = np.minimum(
        _standardize_against_null(sparse_full),
        _standardize_against_null(sparse_leave),
    )
    dense_joint_z = _standardize_against_null(dense_joint)
    sparse_joint_z = _standardize_against_null(sparse_joint)
    branches = [dense_joint_z, sparse_joint_z]
    if matrix.shape[1] >= 5:
        hc_full = _higher_criticism(matrix)
        hc_leave = _higher_criticism(_remove_rowwise_max(matrix))
        hc_joint = np.minimum(
            _standardize_against_null(hc_full),
            _standardize_against_null(hc_leave),
        )
        hc_joint_z = _standardize_against_null(hc_joint)
        branches.append(hc_joint_z)
    else:
        hc_joint_z = np.full(matrix.shape[0], np.nan)
    omnibus = np.maximum.reduce(branches)
    observed_result = {
        "v175_dense_robust_joint_z": float(dense_joint_z[0]),
        "v175_sparse_robust_joint_z": float(sparse_joint_z[0]),
        "v175_hc_robust_joint_z": float(hc_joint_z[0]),
        "v175_adaptive_robust_omnibus": float(omnibus[0]),
        "v175_adaptive_robust_omnibus_empirical_p": empirical_upper(omnibus[1:], omnibus[0]),
    }
    null_result = {
        "v175_dense_robust_joint_z": dense_joint_z[1:],
        "v175_sparse_robust_joint_z": sparse_joint_z[1:],
        "v175_hc_robust_joint_z": hc_joint_z[1:],
        "v175_adaptive_robust_omnibus": omnibus[1:],
    }
    return observed_result, null_result


def adaptive_locus_module_test_v175(
    context: AdaptiveLocusContext,
    genes: set[str] | frozenset[str],
    config: AdaptiveLocusConfig,
    *,
    n_null: int,
    rng: np.random.Generator,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Test one fixed module with the V1.7.5 conditional joint statistic."""

    capped, _, _, observed_indices, observed_counts, selected = _collapse_observed_module(
        context, set(genes)
    )
    if len(observed_indices) < 2:
        return {"test_status": "not_tested_fewer_than_two_loci"}, {}
    prepared = _prepare_matched_module_pools(context, observed_indices, observed_counts)
    null_matrix = np.empty((int(n_null), len(observed_indices)), dtype=float)
    audit_keys = (
        "null_exact_match_rate",
        "null_degree_fallback_rate",
        "null_global_fallback_rate",
        "null_reuse_fallback_rate",
        "min_match_pool_size",
        "median_match_pool_size",
        "within_locus_replacement_rate",
    )
    audit_values: dict[str, list[float]] = {key: [] for key in audit_keys}
    for replicate in range(int(n_null)):
        null_capped, _, _, audit = _sample_matched_module(
            context,
            observed_indices,
            observed_counts,
            rng,
            prepared=prepared,
        )
        null_matrix[replicate] = null_capped
        for key in audit_keys:
            audit_values[key].append(float(audit[key]))
    dense_observed, dense_nulls = conditional_joint_statistic(capped, null_matrix)
    observed, nulls = conditional_adaptive_robust_omnibus(capped, null_matrix)
    observed.update(dense_observed)
    nulls.update(dense_nulls)
    row: dict[str, object] = {
        "test_status": "tested",
        "n_present": int(selected["gene_symbol"].nunique()),
        "n_loci": int(len(observed_indices)),
        **observed,
    }
    for key, values in audit_values.items():
        row[key] = float(np.mean(values))
    return row, nulls


class _UnionFind:
    def __init__(self, names: Sequence[str]) -> None:
        self.parent = {name: name for name in names}

    def find(self, name: str) -> str:
        parent = self.parent[name]
        if parent != name:
            self.parent[name] = self.find(parent)
        return self.parent[name]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def reduce_library_by_gene_overlap(
    library: AnchoredModuleLibrary,
    *,
    jaccard_threshold: float = 0.50,
) -> ReducedLibraryResult:
    """Collapse gene-set redundancy without consulting any trait score.

    Connected overlap clusters are formed at the frozen Jaccard threshold.  A
    deterministic medoid is selected by maximum within-cluster mean Jaccard,
    then smaller set size, then lexical module ID.
    """

    if not 0.0 < jaccard_threshold <= 1.0:
        raise ValueError("jaccard_threshold must be in (0, 1]")
    gene_sets = {
        str(name): frozenset(str(gene).upper() for gene in genes)
        for name, genes in library.gene_sets.items()
    }
    names = sorted(gene_sets)
    union_find = _UnionFind(names)
    gene_to_modules: dict[str, list[str]] = defaultdict(list)
    for name, genes in gene_sets.items():
        for gene in genes:
            gene_to_modules[gene].append(name)
    intersections: dict[tuple[str, str], int] = defaultdict(int)
    for modules in gene_to_modules.values():
        ordered = sorted(modules)
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                intersections[(left, right)] += 1
    qualifying_jaccard: dict[tuple[str, str], float] = {}
    for (left, right), intersection in intersections.items():
        union_size = len(gene_sets[left]) + len(gene_sets[right]) - intersection
        jaccard = intersection / union_size if union_size else 0.0
        if jaccard >= jaccard_threshold:
            union_find.union(left, right)
            qualifying_jaccard[(left, right)] = jaccard
    clusters: dict[str, list[str]] = defaultdict(list)
    for name in names:
        clusters[union_find.find(name)].append(name)
    representatives: dict[str, str] = {}
    for members in clusters.values():
        if len(members) == 1:
            representative = members[0]
        else:
            mean_similarity: dict[str, float] = {}
            for name in members:
                similarities = []
                for other in members:
                    if name == other:
                        continue
                    pair = tuple(sorted((name, other)))
                    intersection = intersections.get(pair, 0)
                    union_size = len(gene_sets[name]) + len(gene_sets[other]) - intersection
                    similarities.append(intersection / union_size if union_size else 0.0)
                mean_similarity[name] = float(np.mean(similarities))
            representative = min(
                members,
                key=lambda name: (-mean_similarity[name], len(gene_sets[name]), name),
            )
        for name in members:
            representatives[name] = representative
    representative_names = sorted(set(representatives.values()))
    reduced = AnchoredModuleLibrary(
        gene_sets={name: set(gene_sets[name]) for name in representative_names},
        module_source={name: library.module_source.get(name, "unspecified") for name in representative_names},
        annotation_source_type={
            name: library.annotation_source_type.get(name, "unspecified") for name in representative_names
        },
        module_category={name: library.module_category.get(name, "") for name in representative_names},
    )
    cluster_ids = {
        root: f"V175_OC{index:04d}" for index, root in enumerate(sorted(clusters), start=1)
    }
    rows = []
    for root, members in sorted(clusters.items()):
        representative = representatives[members[0]]
        for name in sorted(members):
            intersection = len(gene_sets[name] & gene_sets[representative])
            union_size = len(gene_sets[name] | gene_sets[representative])
            rows.append(
                {
                    "module_name": name,
                    "overlap_cluster_id": cluster_ids[root],
                    "representative_module": representative,
                    "is_representative": name == representative,
                    "n_genes": len(gene_sets[name]),
                    "gene_jaccard_to_representative": (
                        1.0 if name == representative else intersection / union_size
                    ),
                    "reduction_uses_trait_scores": False,
                    "jaccard_threshold": jaccard_threshold,
                }
            )
    return ReducedLibraryResult(reduced, pd.DataFrame(rows))
