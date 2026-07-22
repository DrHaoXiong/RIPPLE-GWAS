"""Selection-calibrated adaptive locus-burden tests for RIPPLE-D V1.7.

V1.7 treats sparse-to-dense adaptation as one pre-specified test. Component
statistics are standardized against the same matched-locus null draws, and the
component maximum is repeated in every null replicate. Individual components
remain diagnostic and cannot be selected post hoc for a primary claim.
"""

from __future__ import annotations

import math
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ripple.modules.anchored import AnchoredModuleLibrary, bh_fdr, empirical_upper
from ripple.modules.distributed import (
    RippleDConfig,
    contribution_metrics,
    external_locus_audit_table,
    huberize,
    prepare_locus_inputs,
)


V17_COMPONENTS = (
    "symmetric_huber_mean_burden",
    "positive_part_max_burden",
    "adaptive_truncated_sum",
)
V17_LIBRARY_ROLES = {"full_library", "fixed_biological_panel", "diagnostic_subset"}


@dataclass(frozen=True)
class AdaptiveLocusConfig:
    """Frozen V1.7 adaptive-test configuration."""

    ripple_d: RippleDConfig
    topk_fractions: tuple[float, ...] = (0.10, 0.20, 0.50, 1.0)
    topk_min_loci: int = 3
    min_present_genes: int = 5
    q_max: float = 0.10
    dispersion_effective_loci_min: float = 5.0
    leave_top1_empirical_p_max: float = 0.025
    leave_top1_supportive_p_max: float = 0.10
    null_exact_match_rate_min: float = 0.80
    null_global_fallback_rate_max: float = 0.05
    null_reuse_fallback_rate_max: float = 0.0
    within_locus_replacement_rate_max: float = 0.0


@dataclass(frozen=True)
class AdaptiveLocusContext:
    """Precomputed arrays reused by observed and null module evaluations."""

    work: pd.DataFrame
    background: pd.DataFrame
    locus_id_to_index: dict[str, int]
    match_bin_arr: np.ndarray
    degree_bin_arr: np.ndarray
    locus_gene_count_arr: np.ndarray
    all_indices: np.ndarray
    match_pools_idx: Mapping[str, np.ndarray]
    degree_pools_idx: Mapping[int, np.ndarray]
    locus_gene_values: Mapping[int, np.ndarray]
    locus_row_positions: Mapping[int, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class _PreparedMatchedModulePools:
    observed_indices: np.ndarray
    observed_gene_counts: np.ndarray
    exact_pools: tuple[np.ndarray, ...]
    degree_pools: tuple[np.ndarray, ...]
    global_pools: tuple[np.ndarray, ...]


def preregistered_v17_config_pass(config: AdaptiveLocusConfig) -> bool:
    """Return whether the frozen primary V1.7 matching design is in use."""

    locus = config.ripple_d
    return bool(
        locus.locus_id_column
        and locus.locus_collapse == "max"
        and locus.degree_bins == 5
        and locus.property_bins == 2
        and not locus.annotation_matching_enabled
        and locus.require_gene_count_match
        and locus.null_gene_subset_sampling
    )


def adaptive_library_fingerprint(library: AnchoredModuleLibrary) -> str:
    """Return a stable fingerprint over module IDs, genes and provenance."""

    payload = [
        {
            "module_name": name,
            "genes": sorted(str(gene).upper() for gene in genes),
            "module_source": library.module_source.get(name, "unspecified"),
            "annotation_source_type": library.annotation_source_type.get(name, "unspecified"),
            "module_category": library.module_category.get(name, ""),
        }
        for name, genes in sorted(library.gene_sets.items())
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def adaptive_analysis_scope_fingerprint(
    library_fingerprint: str,
    config: AdaptiveLocusConfig,
    *,
    correction_scope_id: str,
    locus_source: str,
    locus_source_version: str,
    genome_build: str,
    ancestry: str,
) -> str:
    """Bind the tested library, statistic, matching and locus provenance."""

    locus = config.ripple_d
    payload = {
        "library_fingerprint": library_fingerprint,
        "correction_scope_id": correction_scope_id,
        "components": V17_COMPONENTS,
        "topk_fractions": config.topk_fractions,
        "topk_min_loci": config.topk_min_loci,
        "leave_top1_empirical_p_max": config.leave_top1_empirical_p_max,
        "leave_top1_supportive_p_max": config.leave_top1_supportive_p_max,
        "score_cap": locus.score_cap,
        "locus_id_column": locus.locus_id_column,
        "locus_definition_name": locus.locus_definition_name,
        "degree_bins": locus.degree_bins,
        "property_bins": locus.property_bins,
        "annotation_matching_enabled": locus.annotation_matching_enabled,
        "require_gene_count_match": locus.require_gene_count_match,
        "null_gene_subset_sampling": locus.null_gene_subset_sampling,
        "locus_source": locus_source,
        "locus_source_version": locus_source_version,
        "genome_build": genome_build,
        "ancestry": ancestry,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def adaptive_topk_grid(n_loci: int, config: AdaptiveLocusConfig) -> tuple[int, ...]:
    """Return the pre-specified truncated-sum grid for a module size."""

    n = int(n_loci)
    if n <= 0:
        return ()
    values = {
        min(n, max(1, int(config.topk_min_loci), int(math.ceil(fraction * n))))
        for fraction in config.topk_fractions
    }
    values.add(n)
    return tuple(sorted(values))


def adaptive_truncated_sum(values: np.ndarray | pd.Series, config: AdaptiveLocusConfig) -> float:
    """Maximum positive truncated sum over the frozen top-k grid."""

    positive = np.sort(np.clip(np.asarray(values, dtype=float), 0.0, None))[::-1]
    positive = positive[np.isfinite(positive)]
    if positive.size == 0:
        return float("nan")
    return float(max(np.sum(positive[:k]) / math.sqrt(k) for k in adaptive_topk_grid(positive.size, config)))


def adaptive_component_statistics(
    capped_max_scores: np.ndarray | pd.Series,
    huber_mean_scores: np.ndarray | pd.Series,
    config: AdaptiveLocusConfig,
) -> dict[str, float]:
    """Compute the frozen V1.7 component family."""

    capped = np.asarray(capped_max_scores, dtype=float)
    huber = np.asarray(huber_mean_scores, dtype=float)
    capped = capped[np.isfinite(capped)]
    huber = huber[np.isfinite(huber)]
    if capped.size == 0 or huber.size == 0:
        return {name: float("nan") for name in V17_COMPONENTS}
    return {
        "symmetric_huber_mean_burden": float(np.sum(huber) / math.sqrt(huber.size)),
        "positive_part_max_burden": float(np.sum(np.clip(capped, 0.0, None)) / math.sqrt(capped.size)),
        "adaptive_truncated_sum": adaptive_truncated_sum(capped, config),
    }


def leave_top1_positive_burden(
    values: np.ndarray | pd.Series,
    raw_values: np.ndarray | pd.Series | None = None,
    *,
    remove_index: int | None = None,
) -> float:
    """Positive burden after removing the strongest uncapped locus."""

    capped = np.asarray(values, dtype=float)
    raw = np.asarray(raw_values if raw_values is not None else values, dtype=float)
    finite = np.isfinite(capped) & np.isfinite(raw)
    capped = capped[finite]
    raw = raw[finite]
    if capped.size <= 1:
        return float("nan")
    remove = int(np.argmax(raw)) if remove_index is None else int(remove_index)
    if remove < 0 or remove >= capped.size:
        raise IndexError("remove_index is outside the locus-score array")
    remainder = np.clip(np.delete(capped, remove), 0.0, None)
    return float(np.sum(remainder) / math.sqrt(remainder.size))


def leave_topk_positive_burden(
    values: np.ndarray | pd.Series,
    raw_values: np.ndarray | pd.Series | None = None,
    *,
    k: int,
    remove_indices: Sequence[int] | None = None,
) -> float:
    """Positive burden after removing ``k`` strongest uncapped loci.

    ``remove_indices`` permits top-conditioned null sensitivity: the null draw
    removes the observed module's top-k locus positions rather than reselecting
    favorable null positions. This is a diagnostic and is not a V1.7 hard gate.
    """

    if k < 1:
        raise ValueError("k must be positive")
    capped = np.asarray(values, dtype=float)
    raw = np.asarray(raw_values if raw_values is not None else values, dtype=float)
    finite = np.isfinite(capped) & np.isfinite(raw)
    capped = capped[finite]
    raw = raw[finite]
    if capped.size <= k:
        return float("nan")
    if remove_indices is None:
        remove = np.argsort(raw, kind="mergesort")[-k:]
    else:
        remove = np.asarray(remove_indices, dtype=int)
        if remove.size != k or np.unique(remove).size != k or (remove < 0).any() or (remove >= capped.size).any():
            raise IndexError("remove_indices must contain k unique valid locus positions")
    remainder = np.clip(np.delete(capped, remove), 0.0, None)
    return float(np.sum(remainder) / math.sqrt(remainder.size))


def selection_calibrated_omnibus(
    observed_components: Mapping[str, float],
    null_components: Mapping[str, np.ndarray],
) -> tuple[float, np.ndarray, dict[str, float]]:
    """Standardize components and repeat their maximum in every null.

    Components with fewer than two finite null values or zero null variance are
    excluded identically from the observed and null maxima.
    """

    columns: list[np.ndarray] = []
    observed_z: list[float] = []
    names: list[str] = []
    n_null: int | None = None
    for name in V17_COMPONENTS:
        values = np.asarray(null_components[name], dtype=float)
        if n_null is None:
            n_null = len(values)
        elif len(values) != n_null:
            raise ValueError("All V1.7 component null arrays must have the same length")
        if not np.isfinite(values).all():
            raise ValueError(f"V1.7 null component {name} contains non-finite values")
        sd = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
        observed = float(observed_components[name])
        if not np.isfinite(observed) or not np.isfinite(sd) or sd <= 0:
            continue
        mean = float(np.mean(values))
        columns.append((values - mean) / sd)
        observed_z.append((observed - mean) / sd)
        names.append(name)
    if not columns:
        return float("nan"), np.full(int(n_null or 0), np.nan), {}
    null_matrix = np.column_stack(columns)
    null_max = np.max(null_matrix, axis=1)
    observed_max = float(np.max(np.asarray(observed_z, dtype=float)))
    return observed_max, null_max, dict(zip(names, observed_z, strict=True))


def prepare_adaptive_locus_context(
    scores: pd.DataFrame,
    library: AnchoredModuleLibrary,
    config: AdaptiveLocusConfig,
) -> AdaptiveLocusContext:
    """Prepare one locus universe for all modules and null replicates."""

    if config.ripple_d.locus_collapse != "max":
        raise ValueError("V1.7 freezes max capped-score locus collapse; other modes are sensitivity analyses")
    if not config.ripple_d.require_gene_count_match or not config.ripple_d.null_gene_subset_sampling:
        raise ValueError("V1.7 requires gene-count matching and within-locus subset sampling")
    eligible_scores = scores.loc[
        np.isfinite(pd.to_numeric(scores["assoc_resid_score"], errors="coerce"))
    ].copy()
    if eligible_scores.empty:
        raise ValueError("V1.7 requires at least one finite assoc_resid_score")
    work, background = prepare_locus_inputs(eligible_scores, library, config.ripple_d)
    duplicated = work["gene_symbol"].astype(str).str.upper().duplicated(keep=False)
    if duplicated.any():
        conflicting = (
            work.loc[duplicated]
            .assign(_gene=work.loc[duplicated, "gene_symbol"].astype(str).str.upper())
            .groupby("_gene", observed=True)["locus_id"]
            .nunique()
        )
        if (conflicting > 1).any():
            genes = ",".join(conflicting.index[conflicting > 1].astype(str)[:5])
            raise ValueError(f"Duplicate gene rows map to multiple loci: {genes}")
        deduplicated_scores = eligible_scores.copy()
        deduplicated_scores["_v17_gene_key"] = deduplicated_scores["gene_symbol"].astype(str).str.upper()
        deduplicated_scores = deduplicated_scores.drop_duplicates(subset=["_v17_gene_key"], keep="first").drop(
            columns="_v17_gene_key"
        )
        work, background = prepare_locus_inputs(deduplicated_scores, library, config.ripple_d)
    work = work.reset_index(drop=True)
    all_loci = background["locus_id"].astype(str).to_numpy(dtype=object)
    locus_id_to_index = {locus_id: idx for idx, locus_id in enumerate(all_loci)}
    match_pools_idx = {
        str(match_bin): np.asarray(
            [locus_id_to_index[str(locus)] for locus in group["locus_id"].astype(str)], dtype=int
        )
        for match_bin, group in background.groupby("match_bin", observed=True)
    }
    degree_pools_idx = {
        int(degree_bin): np.asarray(
            [locus_id_to_index[str(locus)] for locus in group["locus_id"].astype(str)], dtype=int
        )
        for degree_bin, group in background.groupby("degree_bin", observed=True)
    }
    locus_gene_values: dict[int, np.ndarray] = {}
    locus_row_positions: dict[int, np.ndarray] = {}
    for locus_id, group in work.groupby("locus_id", observed=True):
        index = locus_id_to_index.get(str(locus_id))
        if index is None:
            continue
        locus_gene_values[index] = group.loc[
            :, ["ripple_d_capped_score", "ripple_d_huber_score", "assoc_resid_score"]
        ].to_numpy(dtype=float)
        locus_row_positions[index] = group.index.to_numpy(dtype=int)
    return AdaptiveLocusContext(
        work=work,
        background=background,
        locus_id_to_index=locus_id_to_index,
        match_bin_arr=background["match_bin"].astype(str).to_numpy(dtype=object),
        degree_bin_arr=background["degree_bin"].to_numpy(dtype=int),
        locus_gene_count_arr=background["n_locus_genes"].to_numpy(dtype=int),
        all_indices=np.arange(len(background), dtype=int),
        match_pools_idx=match_pools_idx,
        degree_pools_idx=degree_pools_idx,
        locus_gene_values=locus_gene_values,
        locus_row_positions=locus_row_positions,
    )


def context_with_updated_association_scores(
    context: AdaptiveLocusContext,
    updates: Mapping[str, float],
    config: AdaptiveLocusConfig,
) -> AdaptiveLocusContext:
    """Return a score-updated context while retaining immutable matching structure.

    This is intended for real-background spike-ins. V1.7 adaptive null matching
    uses locus membership, gene-counts, and technical match bins, all of which
    are score-independent. The function updates only the observed gene scores,
    capped/Huber transforms, and affected per-locus score arrays. It does not
    alter match pools or reuse a null draw.
    """

    if not updates:
        return context
    normalized = {str(gene).upper(): float(value) for gene, value in updates.items()}
    work = context.work.copy()
    work["assoc_resid_score"] = pd.to_numeric(work["assoc_resid_score"], errors="raise").astype(float)
    gene_keys = work["gene_symbol"].astype(str).str.upper()
    affected = gene_keys.isin(normalized)
    if not affected.any():
        raise ValueError("association-score updates do not overlap the prepared context")
    work.loc[affected, "assoc_resid_score"] = gene_keys.loc[affected].map(normalized).to_numpy(dtype=float)
    work.loc[affected, "ripple_d_capped_score"] = np.minimum(
        work.loc[affected, "assoc_resid_score"].to_numpy(dtype=float), config.ripple_d.score_cap
    )
    work.loc[affected, "ripple_d_huber_score"] = huberize(
        work.loc[affected, "assoc_resid_score"].to_numpy(dtype=float), config.ripple_d.score_cap
    )
    locus_gene_values = dict(context.locus_gene_values)
    score_matrix = work.loc[
        :, ["ripple_d_capped_score", "ripple_d_huber_score", "assoc_resid_score"]
    ].to_numpy(dtype=float)
    affected_rows = np.flatnonzero(affected.to_numpy(dtype=bool))
    affected_locus_indices = {
        context.locus_id_to_index[str(locus_id)]
        for locus_id in work.iloc[affected_rows]["locus_id"].astype(str).unique()
        if str(locus_id) in context.locus_id_to_index
    }
    if context.locus_row_positions:
        for index in affected_locus_indices:
            locus_gene_values[index] = score_matrix[context.locus_row_positions[index]]
    else:
        for locus_id, full_group in work.groupby("locus_id", observed=True, sort=False):
            index = context.locus_id_to_index.get(str(locus_id))
            if index in affected_locus_indices:
                locus_gene_values[index] = full_group.loc[
                    :,
                    [
                        "ripple_d_capped_score",
                        "ripple_d_huber_score",
                        "assoc_resid_score",
                    ],
                ].to_numpy(dtype=float)
    return AdaptiveLocusContext(
        work=work,
        background=context.background,
        locus_id_to_index=context.locus_id_to_index,
        match_bin_arr=context.match_bin_arr,
        degree_bin_arr=context.degree_bin_arr,
        locus_gene_count_arr=context.locus_gene_count_arr,
        all_indices=context.all_indices,
        match_pools_idx=context.match_pools_idx,
        degree_pools_idx=context.degree_pools_idx,
        locus_gene_values=locus_gene_values,
        locus_row_positions=context.locus_row_positions,
    )


def _collapse_observed_module(
    context: AdaptiveLocusContext,
    genes: set[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[int], pd.DataFrame]:
    selected = context.work.loc[
        context.work["gene_symbol"].astype(str).str.upper().isin(genes)
    ].drop_duplicates(subset=["gene_symbol"], keep="first").copy()
    if selected.empty:
        return np.array([]), np.array([]), np.array([]), [], [], selected
    grouped = selected.groupby("locus_id", observed=True)
    capped = grouped["ripple_d_capped_score"].max()
    huber = grouped["ripple_d_huber_score"].mean()
    raw = grouped["assoc_resid_score"].max()
    locus_ids = [locus for locus in capped.index.astype(str) if locus in context.locus_id_to_index]
    indices = [context.locus_id_to_index[locus] for locus in locus_ids]
    counts = [int(grouped.size().loc[locus]) for locus in locus_ids]
    return (
        capped.loc[locus_ids].to_numpy(dtype=float),
        huber.loc[locus_ids].to_numpy(dtype=float),
        raw.loc[locus_ids].to_numpy(dtype=float),
        indices,
        counts,
        selected,
    )


def module_locus_contribution_audit(
    context: AdaptiveLocusContext,
    genes: set[str],
    config: AdaptiveLocusConfig,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Return gene/locus contribution rows and weak-signal diagnostics.

    This audit does not modify the V1.7 statistic. It exposes the exact locus
    collapse and positive contribution decomposition used for nomination.
    """

    selected = context.work.loc[
        context.work["gene_symbol"].astype(str).str.upper().isin(genes)
    ].drop_duplicates(subset=["gene_symbol"], keep="first").copy()
    if selected.empty:
        return pd.DataFrame(), {
            "n_positive_loci": 0.0,
            "fraction_loci_with_uncapped_score_gt_3": float("nan"),
            "fraction_positive_signal_from_uncapped_gt_3": float("nan"),
            "multi_strong_locus_overlap": False,
        }
    grouped = selected.groupby("locus_id", observed=True)
    locus = grouped.agg(
        locus_capped_max=("ripple_d_capped_score", "max"),
        locus_huber_mean=("ripple_d_huber_score", "mean"),
        locus_raw_max=("assoc_resid_score", "max"),
        n_module_genes_in_locus=("gene_symbol", "nunique"),
    ).reset_index()
    positive = np.clip(locus["locus_capped_max"].to_numpy(dtype=float), 0.0, None)
    total = float(positive.sum())
    locus["positive_locus_contribution"] = positive / total if total > 0 else 0.0
    locus["locus_uncapped_gt_score_cap"] = locus["locus_raw_max"] > config.ripple_d.score_cap
    locus["is_top_raw_locus"] = locus["locus_raw_max"].eq(locus["locus_raw_max"].max())
    audit = selected.merge(locus, on="locus_id", how="left", validate="many_to_one")
    audit["gene_positive_score"] = np.clip(audit["ripple_d_capped_score"].to_numpy(dtype=float), 0.0, None)
    fraction_strong = float(locus["locus_uncapped_gt_score_cap"].mean())
    strong_signal = float(positive[locus["locus_uncapped_gt_score_cap"].to_numpy(dtype=bool)].sum() / total) if total > 0 else 0.0
    summary = {
        "n_positive_loci": float(np.count_nonzero(positive > 0)),
        "fraction_loci_with_uncapped_score_gt_3": fraction_strong,
        "fraction_positive_signal_from_uncapped_gt_3": strong_signal,
        "multi_strong_locus_overlap": bool(
            fraction_strong > config.ripple_d.top_tail_fraction_loci_gt3_max
            or strong_signal > config.ripple_d.top_tail_signal_gt3_max
        ),
    }
    return audit, summary


def _sample_matched_module(
    context: AdaptiveLocusContext,
    observed_indices: Sequence[int],
    observed_gene_counts: Sequence[int],
    rng: np.random.Generator,
    *,
    prepared: _PreparedMatchedModulePools | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    pools = prepared or _prepare_matched_module_pools(
        context, observed_indices, observed_gene_counts
    )
    used = np.zeros(len(context.all_indices), dtype=bool)
    used[pools.observed_indices] = True
    sampled: list[int] = []
    counts = {"exact": 0, "degree": 0, "global": 0, "all_with_reuse": 0}
    pool_sizes: list[int] = []
    for exact_pool, degree_pool, global_pool in zip(
        pools.exact_pools, pools.degree_pools, pools.global_pools, strict=True
    ):
        candidates = exact_pool[~used[exact_pool]]
        level = "exact"
        if candidates.size == 0:
            candidates = degree_pool[~used[degree_pool]]
            level = "degree"
        if candidates.size == 0:
            candidates = global_pool[~used[global_pool]]
            level = "global"
        if candidates.size == 0:
            candidates = context.all_indices
            level = "all_with_reuse"
        picked = int(candidates[int(rng.integers(candidates.size))])
        sampled.append(picked)
        used[picked] = True
        counts[level] += 1
        pool_sizes.append(int(candidates.size))
    denom = max(1, len(sampled))
    audit = {
        "null_exact_match_rate": counts["exact"] / denom,
        "null_degree_fallback_rate": counts["degree"] / denom,
        "null_global_fallback_rate": counts["global"] / denom,
        "null_reuse_fallback_rate": counts["all_with_reuse"] / denom,
        "min_match_pool_size": float(np.min(pool_sizes)) if pool_sizes else float("nan"),
        "median_match_pool_size": (
            float(np.median(pool_sizes)) if pool_sizes else float("nan")
        ),
    }
    capped: list[float] = []
    huber: list[float] = []
    raw: list[float] = []
    replacements = 0
    for locus_idx, n_genes in zip(sampled, observed_gene_counts, strict=True):
        values = context.locus_gene_values[int(locus_idx)]
        replace = int(n_genes) > len(values)
        if int(n_genes) == 1:
            subset = values[int(rng.integers(len(values))) :][:1]
        else:
            positions = rng.choice(len(values), size=int(n_genes), replace=replace)
            subset = values[np.asarray(positions, dtype=int)]
        if subset.shape[0] == 1:
            capped.append(float(subset[0, 0]))
            huber.append(float(subset[0, 1]))
            raw.append(float(subset[0, 2]))
        else:
            capped.append(float(np.max(subset[:, 0])))
            huber.append(float(np.mean(subset[:, 1])))
            raw.append(float(np.max(subset[:, 2])))
        replacements += int(replace)
    audit["within_locus_replacement_rate"] = replacements / max(1, len(sampled))
    return (
        np.asarray(capped, dtype=float),
        np.asarray(huber, dtype=float),
        np.asarray(raw, dtype=float),
        audit,
    )


def _prepare_matched_module_pools(
    context: AdaptiveLocusContext,
    observed_indices: Sequence[int],
    observed_gene_counts: Sequence[int],
) -> _PreparedMatchedModulePools:
    indices = np.asarray(observed_indices, dtype=int)
    gene_counts = np.asarray(observed_gene_counts, dtype=int)
    exact_pools: list[np.ndarray] = []
    degree_pools: list[np.ndarray] = []
    global_pools: list[np.ndarray] = []
    global_by_count: dict[int, np.ndarray] = {}
    for observed_idx, min_gene_count in zip(indices, gene_counts, strict=True):
        match_bin = str(context.match_bin_arr[observed_idx])
        exact = context.match_pools_idx.get(match_bin, np.array([], dtype=int))
        degree_bin = int(context.degree_bin_arr[observed_idx])
        degree = context.degree_pools_idx.get(degree_bin, np.array([], dtype=int))
        minimum = int(min_gene_count)
        if minimum > 1:
            exact = exact[context.locus_gene_count_arr[exact] >= minimum]
            degree = degree[context.locus_gene_count_arr[degree] >= minimum]
        if minimum not in global_by_count:
            global_by_count[minimum] = context.all_indices[
                context.locus_gene_count_arr >= minimum
            ]
        exact_pools.append(exact)
        degree_pools.append(degree)
        global_pools.append(global_by_count[minimum])
    return _PreparedMatchedModulePools(
        observed_indices=indices,
        observed_gene_counts=gene_counts,
        exact_pools=tuple(exact_pools),
        degree_pools=tuple(degree_pools),
        global_pools=tuple(global_pools),
    )


def adaptive_locus_module_test(
    context: AdaptiveLocusContext,
    genes: set[str],
    config: AdaptiveLocusConfig,
    *,
    n_null: int,
    rng: np.random.Generator,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Test one fixed module with selection repeated in every null replicate."""

    capped, huber, raw, observed_indices, observed_counts, selected = _collapse_observed_module(context, genes)
    if len(observed_indices) == 0:
        return {"test_status": "not_tested_no_loci"}, {}
    observed = adaptive_component_statistics(capped, huber, config)
    observed_top_position = int(np.argmax(raw))
    observed_top3_positions = np.argsort(raw, kind="mergesort")[-min(3, len(raw)):]
    observed_leave_top1 = leave_top1_positive_burden(
        capped, raw, remove_index=observed_top_position
    )
    nulls = {name: np.empty(int(n_null), dtype=float) for name in V17_COMPONENTS}
    nulls["leave_top1_positive_burden_unconditional"] = np.empty(int(n_null), dtype=float)
    nulls["top_conditioned_leave_top1_positive_burden"] = np.empty(int(n_null), dtype=float)
    nulls["leave_top3_positive_burden_unconditional"] = np.empty(int(n_null), dtype=float)
    nulls["top_conditioned_leave_top3_positive_burden"] = np.empty(int(n_null), dtype=float)
    audit_keys = (
        "null_exact_match_rate",
        "null_degree_fallback_rate",
        "null_global_fallback_rate",
        "null_reuse_fallback_rate",
        "min_match_pool_size",
        "median_match_pool_size",
        "within_locus_replacement_rate",
    )
    audit_sums = {key: 0.0 for key in audit_keys}
    prepared_pools = (
        _prepare_matched_module_pools(context, observed_indices, observed_counts)
        if len(context.match_bin_arr) > max(observed_indices, default=-1)
        else None
    )
    for replicate in range(int(n_null)):
        null_capped, null_huber, null_raw, audit = _sample_matched_module(
            context,
            observed_indices,
            observed_counts,
            rng,
            prepared=prepared_pools,
        )
        stats = adaptive_component_statistics(null_capped, null_huber, config)
        for name in V17_COMPONENTS:
            nulls[name][replicate] = stats[name]
        positive = np.clip(null_capped, 0.0, None)
        positive_sum = float(np.sum(positive))
        top1 = int(np.argmax(null_raw))
        top3 = np.argsort(null_raw, kind="mergesort")[-min(3, len(null_raw)) :]
        nulls["leave_top1_positive_burden_unconditional"][replicate] = (
            (positive_sum - positive[top1]) / math.sqrt(len(positive) - 1)
            if len(positive) > 1
            else float("nan")
        )
        nulls["top_conditioned_leave_top1_positive_burden"][replicate] = (
            (positive_sum - positive[observed_top_position]) / math.sqrt(len(positive) - 1)
            if len(positive) > 1
            else float("nan")
        )
        nulls["leave_top3_positive_burden_unconditional"][replicate] = (
            (positive_sum - float(np.sum(positive[top3]))) / math.sqrt(len(positive) - 3)
            if len(positive) > 3
            else float("nan")
        )
        nulls["top_conditioned_leave_top3_positive_burden"][replicate] = (
            (
                positive_sum
                - float(np.sum(positive[np.asarray(observed_top3_positions, dtype=int)]))
            )
            / math.sqrt(len(positive) - 3)
            if len(positive) > 3
            else float("nan")
        )
        for key in audit_keys:
            audit_sums[key] += float(audit[key])
    omnibus, null_omnibus, component_z = selection_calibrated_omnibus(observed, nulls)
    nulls["v17_adaptive_omnibus"] = null_omnibus
    contributions = contribution_metrics(capped)
    index_to_locus = {index: locus for locus, index in context.locus_id_to_index.items()}
    present_locus_ids = [index_to_locus[index] for index in observed_indices]
    positive_capped = np.clip(capped, 0.0, None)
    positive_total = float(positive_capped.sum())
    raw_top_tail = raw > config.ripple_d.score_cap
    fraction_loci_gt_cap = float(raw_top_tail.mean()) if raw_top_tail.size else float("nan")
    fraction_signal_gt_cap = (
        float(positive_capped[raw_top_tail].sum() / positive_total)
        if positive_total > 0 else 0.0
    )
    row: dict[str, object] = {
        "test_status": "tested",
        "n_present": int(selected["gene_symbol"].nunique()),
        "n_loci": int(len(observed_indices)),
        "present_locus_ids": ",".join(sorted(present_locus_ids)),
        "present_genes": ",".join(sorted(selected["gene_symbol"].astype(str).str.upper().unique())),
        **observed,
        "v17_adaptive_omnibus": omnibus,
        "v17_adaptive_omnibus_empirical_p": empirical_upper(null_omnibus, omnibus),
        "leave_top1_positive_burden": observed_leave_top1,
        "leave_top1_positive_burden_unconditional_empirical_p": empirical_upper(
            nulls["leave_top1_positive_burden_unconditional"], observed_leave_top1
        ),
        "top_conditioned_leave_top1_positive_burden_empirical_p": empirical_upper(
            nulls["top_conditioned_leave_top1_positive_burden"], observed_leave_top1
        ),
        "fraction_loci_with_uncapped_score_gt_3": fraction_loci_gt_cap,
        "fraction_positive_signal_from_uncapped_gt_3": fraction_signal_gt_cap,
        "top_tail_pass": bool(
            fraction_loci_gt_cap <= config.ripple_d.top_tail_fraction_loci_gt3_max
            and fraction_signal_gt_cap <= config.ripple_d.top_tail_signal_gt3_max
        ),
        "leave_top3_positive_burden": leave_topk_positive_burden(capped, raw, k=3),
        "leave_top3_positive_burden_unconditional_empirical_p": empirical_upper(
            nulls["leave_top3_positive_burden_unconditional"],
            leave_topk_positive_burden(capped, raw, k=3),
        ),
        "top_conditioned_leave_top3_positive_burden_empirical_p": empirical_upper(
            nulls["top_conditioned_leave_top3_positive_burden"],
            leave_topk_positive_burden(capped, raw, k=3),
        ),
        **contributions,
    }
    for name in V17_COMPONENTS:
        row[f"{name}_empirical_p"] = empirical_upper(nulls[name], observed[name])
        row[f"{name}_z"] = component_z.get(name, float("nan"))
    for key in audit_keys:
        row[key] = audit_sums[key] / int(n_null)
    return row, nulls


def _claim_status(
    row: pd.Series,
    config: AdaptiveLocusConfig,
    *,
    library_role: str,
    external_locus_pass: bool,
    external_locus_provenance_pass: bool,
    frozen_config_pass: bool,
    scope_registration_pass: bool,
    empirical_resolution_pass: bool,
    scope_claim_authorized: bool,
) -> tuple[str, str]:
    if str(row.get("test_status", "")) != "tested":
        return "not_tested", "insufficient_gene_overlap_or_loci"
    null_pass = bool(
        float(row.get("null_exact_match_rate", 0.0)) >= config.null_exact_match_rate_min
        and float(row.get("null_global_fallback_rate", 1.0)) <= config.null_global_fallback_rate_max
        and float(row.get("null_reuse_fallback_rate", 1.0)) <= config.null_reuse_fallback_rate_max
        and float(row.get("within_locus_replacement_rate", 1.0)) <= config.within_locus_replacement_rate_max
    )
    dispersion_pass = bool(
        float(row.get("top_conditioned_leave_top1_positive_burden_empirical_p", 1.0))
        < config.leave_top1_empirical_p_max
    )
    supportive_dispersion_pass = bool(
        float(row.get("top_conditioned_leave_top1_positive_burden_empirical_p", 1.0))
        < config.leave_top1_supportive_p_max
    )
    q_pass = float(row.get("v17_adaptive_omnibus_bh_q", 1.0)) <= config.q_max
    if (
        q_pass
        and null_pass
        and dispersion_pass
        and external_locus_pass
        and external_locus_provenance_pass
        and frozen_config_pass
        and scope_registration_pass
        and empirical_resolution_pass
        and scope_claim_authorized
        and library_role != "diagnostic_subset"
    ):
        label = "v17_fixed_panel_supported" if library_role == "fixed_biological_panel" else "v17_full_library_candidate"
        return label, "none"
    if (
        q_pass
        and null_pass
        and supportive_dispersion_pass
        and external_locus_pass
        and external_locus_provenance_pass
        and frozen_config_pass
        and scope_registration_pass
        and empirical_resolution_pass
        and scope_claim_authorized
        and library_role != "diagnostic_subset"
    ):
        return "v17_high_confidence_diagnostic", "manuscript_dispersion_gate"
    if float(row.get("v17_adaptive_omnibus_empirical_p", 1.0)) < 0.05 and null_pass and external_locus_pass:
        failed = []
        if not q_pass:
            failed.append("multiplicity")
        if not dispersion_pass:
            failed.append("dispersion")
        if not external_locus_provenance_pass:
            failed.append("external_locus_provenance")
        if not frozen_config_pass:
            failed.append("non_preregistered_config")
        if not scope_registration_pass:
            failed.append("unregistered_library_scope")
        if not scope_claim_authorized:
            failed.append("library_scope_not_authorized_for_claim")
        if not empirical_resolution_pass:
            failed.append("insufficient_empirical_p_resolution")
        if library_role == "diagnostic_subset":
            failed.append("diagnostic_subset_scope")
        return "v17_nominal_diagnostic", ";".join(failed) or "none"
    failed = []
    if not external_locus_pass:
        failed.append("external_locus")
    if not external_locus_provenance_pass:
        failed.append("external_locus_provenance")
    if not frozen_config_pass:
        failed.append("non_preregistered_config")
    if not scope_registration_pass:
        failed.append("unregistered_library_scope")
    if not scope_claim_authorized:
        failed.append("library_scope_not_authorized_for_claim")
    if not empirical_resolution_pass:
        failed.append("insufficient_empirical_p_resolution")
    if library_role == "diagnostic_subset":
        failed.append("diagnostic_subset_scope")
    if not null_pass:
        failed.append("null_quality")
    if not q_pass:
        failed.append("multiplicity")
    if not dispersion_pass:
        failed.append("dispersion")
    return "negative", ";".join(failed) or "no_omnibus_support"


def adaptive_locus_library_test(
    scores: pd.DataFrame,
    library: AnchoredModuleLibrary,
    config: AdaptiveLocusConfig,
    *,
    n_null: int,
    seed: int,
    library_role: str,
    correction_scope_id: str,
    locus_source: str = "unspecified",
    locus_source_version: str = "unspecified",
    genome_build: str = "unspecified",
    ancestry: str = "unspecified",
    construction_script: str = "unspecified",
    retain_nulls: bool = False,
    registered_library_fingerprint: str | None = None,
    registered_claim_level: str = "diagnostic_only",
    per_module_seeded: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, object]]:
    """Test an entire pre-specified library before applying BH correction."""

    if library_role not in V17_LIBRARY_ROLES:
        raise ValueError(f"Unknown V1.7 library_role: {library_role}")
    if n_null < 1:
        raise ValueError("n_null must be positive")
    context = prepare_adaptive_locus_context(scores, library, config)
    locus_audit = external_locus_audit_table(
        context.work,
        locus_id_column=config.ripple_d.locus_id_column,
        locus_source=locus_source,
        locus_source_version=locus_source_version,
        genome_build=genome_build,
        ancestry=ancestry,
        construction_script=construction_script,
    )
    external_pass = bool(locus_audit.iloc[0]["external_locus_audit_pass"])
    provenance_values = (
        locus_source,
        locus_source_version,
        genome_build,
        ancestry,
        construction_script,
    )
    provenance_pass = all(str(value).strip().lower() not in {"", "unspecified", "unknown"} for value in provenance_values)
    frozen_config_pass = preregistered_v17_config_pass(config)
    library_fingerprint = adaptive_library_fingerprint(library)
    scope_registration_pass = bool(
        registered_library_fingerprint
        and str(registered_library_fingerprint).strip().lower() == library_fingerprint
    )
    scope_claim_authorized = registered_claim_level == "manuscript_candidate"
    scope_fingerprint = adaptive_analysis_scope_fingerprint(
        library_fingerprint,
        config,
        correction_scope_id=correction_scope_id,
        locus_source=locus_source,
        locus_source_version=locus_source_version,
        genome_build=genome_build,
        ancestry=ancestry,
    )
    eligible = set(context.work["gene_symbol"].astype(str).str.upper())
    rows: list[dict[str, object]] = []
    retained: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(seed)
    for module_name, genes in library.gene_sets.items():
        present = {str(gene).upper() for gene in genes} & eligible
        base = {
            "module_name": module_name,
            "module_source": library.module_source.get(module_name, "unspecified"),
            "annotation_source_type": library.annotation_source_type.get(module_name, "unspecified"),
            "module_category": library.module_category.get(module_name, ""),
            "library_role": library_role,
            "correction_scope_id": correction_scope_id,
            "n_null": int(n_null),
            "statistic_direction": "greater_is_more_extreme",
            "statistic_family": "v17_adaptive_sparse_dense_locus_omnibus",
            "statistic_selection_rule": "max_over_frozen_components_repeated_in_each_null",
        }
        if len(present) < config.min_present_genes:
            rows.append({**base, "n_present": len(present), "test_status": "not_tested_low_overlap"})
            continue
        module_rng = rng
        if per_module_seeded:
            # V1.7 candidate runs must be invariant to library row order. Legacy
            # callers retain the historical shared-stream behavior by default.
            module_seed = int.from_bytes(
                hashlib.sha256(f"{seed}:{module_name}".encode("utf-8")).digest()[:8],
                "little",
                signed=False,
            )
            module_rng = np.random.default_rng(module_seed)
        result, nulls = adaptive_locus_module_test(
            context, present, config, n_null=n_null, rng=module_rng
        )
        rows.append({**base, **result})
        if retain_nulls:
            retained[module_name] = nulls["v17_adaptive_omnibus"]
    table = pd.DataFrame(rows)
    tested = table["test_status"].eq("tested")
    table["v17_adaptive_omnibus_bh_q"] = np.nan
    table.loc[tested, "v17_adaptive_omnibus_bh_q"] = bh_fdr(
        pd.to_numeric(table.loc[tested, "v17_adaptive_omnibus_empirical_p"], errors="coerce")
    )
    table["n_correction_tests"] = int(tested.sum())
    leave_top1_p = (
        pd.to_numeric(table["top_conditioned_leave_top1_positive_burden_empirical_p"], errors="coerce")
        if "top_conditioned_leave_top1_positive_burden_empirical_p" in table
        else pd.Series(np.nan, index=table.index, dtype=float)
    )
    table["v17_dispersion_gate_pass"] = (leave_top1_p < config.leave_top1_empirical_p_max).fillna(False)
    table["v17_supportive_dispersion_gate_pass"] = (
        leave_top1_p < config.leave_top1_supportive_p_max
    ).fillna(False)
    minimum_empirical_p = 1.0 / (int(n_null) + 1.0)
    single_discovery_resolution_pass = bool(
        int(tested.sum()) > 0 and minimum_empirical_p <= config.q_max / int(tested.sum())
    )
    statuses = table.apply(
        lambda row: _claim_status(
            row,
            config,
            library_role=library_role,
            external_locus_pass=external_pass,
            external_locus_provenance_pass=provenance_pass,
            frozen_config_pass=frozen_config_pass,
            scope_registration_pass=scope_registration_pass,
            empirical_resolution_pass=single_discovery_resolution_pass,
            scope_claim_authorized=scope_claim_authorized,
        ),
        axis=1,
    )
    table["v17_claim_status"] = [status for status, _ in statuses]
    table["v17_downgrade_reason"] = [reason for _, reason in statuses]
    table["external_locus_audit_pass"] = external_pass
    table["external_locus_provenance_pass"] = provenance_pass
    table["v17_preregistered_config_pass"] = frozen_config_pass
    table["library_fingerprint"] = library_fingerprint
    table["registered_library_fingerprint"] = registered_library_fingerprint or ""
    table["scope_registration_pass"] = scope_registration_pass
    table["registered_claim_level"] = registered_claim_level
    table["scope_claim_authorized"] = scope_claim_authorized
    table["analysis_scope_fingerprint"] = scope_fingerprint
    table["minimum_resolvable_empirical_p"] = minimum_empirical_p
    table["single_discovery_resolution_pass"] = single_discovery_resolution_pass
    summary = {
        "n_input_modules": int(len(library.gene_sets)),
        "n_tested_modules": int(tested.sum()),
        "n_correction_tests": int(tested.sum()),
        "n_v17_full_library_candidate": int(table["v17_claim_status"].eq("v17_full_library_candidate").sum()),
        "n_v17_fixed_panel_supported": int(table["v17_claim_status"].eq("v17_fixed_panel_supported").sum()),
        "n_v17_nominal_diagnostic": int(table["v17_claim_status"].eq("v17_nominal_diagnostic").sum()),
        "n_v17_high_confidence_diagnostic": int(
            table["v17_claim_status"].eq("v17_high_confidence_diagnostic").sum()
        ),
        "library_role": library_role,
        "correction_scope_id": correction_scope_id,
        "n_null": int(n_null),
        "seed": int(seed),
        "external_locus_audit_pass": external_pass,
        "external_locus_provenance_pass": provenance_pass,
        "v17_preregistered_config_pass": frozen_config_pass,
        "library_fingerprint": library_fingerprint,
        "registered_library_fingerprint": registered_library_fingerprint or "",
        "scope_registration_pass": scope_registration_pass,
        "registered_claim_level": registered_claim_level,
        "scope_claim_authorized": scope_claim_authorized,
        "analysis_scope_fingerprint": scope_fingerprint,
        "minimum_resolvable_empirical_p": minimum_empirical_p,
        "single_discovery_resolution_pass": single_discovery_resolution_pass,
        "locus_definition": config.ripple_d.locus_definition_name or "unspecified",
        "degree_bins": int(config.ripple_d.degree_bins),
        "property_bins": int(config.ripple_d.property_bins),
        "annotation_matching_enabled": bool(config.ripple_d.annotation_matching_enabled),
        "components": ";".join(V17_COMPONENTS),
    }
    return table, locus_audit, retained, summary
