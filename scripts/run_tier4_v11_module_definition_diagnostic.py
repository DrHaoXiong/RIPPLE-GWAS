#!/usr/bin/env python
"""Diagnose V1.1 candidate module definitions for RIPPLE Tier 4.

This script is intentionally diagnostic. It does not change the frozen V1 claim
policy. It asks whether less brittle module definitions recover oracle
weak-signal modules better than V1 top-rank induced connected components.

Tested candidate definitions:
    1. soft_top_neighborhood_legacy: score-selected radius-k neighborhoods with
       a soft positive-sum statistic. This is retained as a failure-control.
    2. soft_connected_adaptive_neighborhood: connected seed neighborhoods with
       adaptive size and local-background-normalized statistics.
    3. soft_connected_adaptive_delta_neighborhood: a spike-in-only diagnostic
       using spiked-minus-baseline scores to test module geometry without raw
       trait background dominance.
    4. soft_annulus_contrast_neighborhood: connected seed neighborhoods with
       independent annulus-background contrast.
    5. diffusion_localized_neighborhood: score-selected radius-k neighborhoods
       with distance-decayed weights.
    6. community_louvain_anchor: fixed Louvain graph communities.
    7. oracle_gene_set_library_anchor: oracle gene-set library positive control.
    8. oracle_fixed_anchor: known true module upper-bound positive control.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ripple.nulls.score_permutation import assign_degree_bins  # noqa: E402
from run_dr_mvp_module_reselection_null import (  # noqa: E402
    ANALYSIS_ROOT,
    empirical_upper,
    write_table,
    z_score,
)
from run_tier4_design_defect_audit import (  # noqa: E402
    parse_effect_grid,
    score_spikein,
    split_genes,
)
from run_tier4_failure_localization import prepare_scores_and_graph  # noqa: E402


DEFAULT_OUT_DIR = ANALYSIS_ROOT / "tier4_v11_module_definition_diagnostic_v1"
DESIGN_AUDIT_DIR = ANALYSIS_ROOT / "tier4_design_defect_audit_v1"
THIS_SCRIPT = Path(__file__).resolve()


@dataclass(frozen=True)
class Candidate:
    method: str
    candidate_id: str
    node_indices: tuple[int, ...]
    statistic: float
    tau: float | None = None
    seed: str = ""
    inner_radius: int | None = None
    background_radius: int | None = None
    background_n: int | None = None
    score_basis: str = ""


@dataclass(frozen=True)
class SearchContext:
    graph: nx.Graph
    nodes: np.ndarray
    node_to_idx: dict[str, int]
    degrees: np.ndarray
    bins: np.ndarray
    bin_to_indices: dict[int, np.ndarray]
    adjacency: tuple[np.ndarray, ...]
    communities: tuple[tuple[int, ...], ...]
    community_ids: tuple[str, ...]
    oracle_library: tuple[tuple[int, ...], ...]
    oracle_library_ids: tuple[str, ...]
    neighborhood_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]
    layer_cache: dict[tuple[int, int, int, int], tuple[np.ndarray, np.ndarray, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--oracle-modules", type=Path, default=DESIGN_AUDIT_DIR / "tables" / "a1_oracle_modules.tsv")
    parser.add_argument("--n-null", type=int, default=200)
    parser.add_argument("--degree-bins", type=int, default=10)
    parser.add_argument("--effect-grid", default="weak:1.0,moderate:2.5,strong:5.0,very_strong:10.0")
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--n-seeds", type=int, default=50)
    parser.add_argument("--neighborhood-radius", type=int, default=2)
    parser.add_argument("--soft-max-size", type=int, default=80)
    parser.add_argument("--soft-size-grid", default="10,25,50,80")
    parser.add_argument("--soft-inner-radius", type=int, default=2)
    parser.add_argument("--soft-background-radius", type=int, default=3)
    parser.add_argument("--soft-annulus-min-background", type=int, default=30)
    parser.add_argument("--soft-edge-gain-weight", type=float, default=0.10)
    parser.add_argument("--soft-degree-penalty", type=float, default=0.05)
    parser.add_argument("--soft-seed-pool-size", type=int, default=100)
    parser.add_argument("--diffusion-top-size", type=int, default=80)
    parser.add_argument("--tau-grid", default="0.5,1.0,2.0")
    parser.add_argument("--community-min-size", type=int, default=5)
    parser.add_argument("--community-max-size", type=int, default=300)
    parser.add_argument("--community-resolution", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_float_grid(value: str | Iterable[float]) -> tuple[float, ...]:
    if isinstance(value, str):
        out = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    else:
        out = tuple(float(item) for item in value)
    if not out:
        raise ValueError("grid must not be empty")
    if any((not np.isfinite(item)) or item <= 0 for item in out):
        raise ValueError("grid values must be positive finite numbers")
    return out


def parse_int_grid(value: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        out = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    else:
        out = tuple(int(item) for item in value)
    if not out:
        raise ValueError("grid must not be empty")
    if any(item < 1 for item in out):
        raise ValueError("grid values must be positive integers")
    return tuple(sorted(set(out)))


def make_adjacency_index(
    graph: nx.Graph,
    nodes: Sequence[object],
    node_to_idx: Mapping[str, int],
) -> tuple[np.ndarray, ...]:
    """Create deterministic integer adjacency for repeated small-radius searches."""

    adjacency: list[np.ndarray] = []
    for node in nodes:
        neighbor_indices = sorted(
            {
                int(node_to_idx[str(neighbor)])
                for neighbor in graph.neighbors(str(node))
                if str(neighbor) in node_to_idx
            }
        )
        adjacency.append(np.asarray(neighbor_indices, dtype=int))
    return tuple(adjacency)


def finite_summary(null_values: np.ndarray, observed: float) -> dict[str, float | int | bool]:
    finite = np.asarray(null_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "n_null": 0,
            "null_mean": float("nan"),
            "null_sd": float("nan"),
            "z": float("nan"),
            "empirical_p": float("nan"),
            "pass_p05": False,
        }
    p_value = empirical_upper(finite, observed)
    return {
        "n_null": int(finite.size),
        "null_mean": float(np.mean(finite)),
        "null_sd": float(np.std(finite, ddof=1)) if finite.size >= 2 else float("nan"),
        "z": z_score(observed, finite),
        "empirical_p": p_value,
        "pass_p05": bool(np.isfinite(p_value) and p_value <= 0.05),
    }


def positive_sum_stat(values: np.ndarray, indices: Sequence[int], weights: np.ndarray | None = None) -> float:
    if len(indices) == 0:
        return float("-inf")
    idx = np.asarray(indices, dtype=int)
    positive = np.maximum(0.0, values[idx])
    if weights is None:
        return float(positive.sum() / np.sqrt(max(1, positive.size)))
    w = np.asarray(weights, dtype=float)
    denom = float(np.sqrt(np.sum(w * w)))
    if denom <= 0:
        return float("-inf")
    return float(np.dot(w, positive) / denom)


def local_background_z_stat(
    values: np.ndarray,
    candidate_indices: Sequence[int],
    background_indices: Sequence[int],
    *,
    min_background: int = 10,
) -> float:
    """Score a candidate against the local seed neighborhood background."""

    candidate = np.asarray(candidate_indices, dtype=int)
    background = np.asarray(background_indices, dtype=int)
    if candidate.size == 0 or background.size < min_background:
        return float("-inf")
    candidate_set = set(int(idx) for idx in candidate)
    background = np.asarray([int(idx) for idx in background if int(idx) not in candidate_set], dtype=int)
    if background.size < min_background:
        return float("-inf")
    candidate_values = values[candidate]
    background_values = values[background]
    background_sd = float(np.std(background_values, ddof=1)) if background_values.size >= 2 else 0.0
    if background_sd <= 0 or not np.isfinite(background_sd):
        return float("-inf")
    return float((np.mean(candidate_values) - np.mean(background_values)) * np.sqrt(candidate_values.size) / background_sd)


def robust_background_location_scale(
    values: np.ndarray,
    background_indices: Sequence[int],
    *,
    eps: float = 1e-6,
) -> tuple[float, float]:
    """Return robust background median and MAD scale with stable fallback."""

    background = np.asarray(background_indices, dtype=int)
    if background.size == 0:
        return 0.0, float(eps)
    background_values = np.asarray(values[background], dtype=float)
    location = float(np.median(background_values))
    mad = float(np.median(np.abs(background_values - location)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= eps:
        sd = float(np.std(background_values, ddof=1)) if background_values.size >= 2 else 0.0
        scale = sd if np.isfinite(sd) and sd > eps else float(eps)
    return location, float(scale + eps)


def get_neighborhood_layers(
    context: SearchContext,
    seed_idx: int,
    *,
    inner_radius: int,
    background_radius: int,
    min_background: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return seed inner neighborhood and independent annulus background."""

    if background_radius <= inner_radius:
        raise ValueError("background_radius must be greater than inner_radius.")
    key = (int(seed_idx), int(inner_radius), int(background_radius), int(min_background))
    cached = context.layer_cache.get(key)
    if cached is not None:
        return cached
    pairs = bounded_neighborhood_pairs(context, int(seed_idx), max_radius=background_radius)
    inner = np.asarray(
        sorted(
            int(node_idx)
            for node_idx, distance in pairs
            if int(distance) <= inner_radius
        ),
        dtype=int,
    )
    annulus = np.asarray(
        sorted(
            int(node_idx)
            for node_idx, distance in pairs
            if inner_radius < int(distance) <= background_radius
        ),
        dtype=int,
    )
    if annulus.size >= min_background:
        result = (inner, annulus, "annulus")
        context.layer_cache[key] = result
        return result

    inner_set = set(int(idx) for idx in inner)
    seed_bin = int(context.bins[int(seed_idx)])
    degree_background = np.asarray(
        [int(idx) for idx in context.bin_to_indices[seed_bin] if int(idx) not in inner_set],
        dtype=int,
    )
    if degree_background.size >= min_background:
        result = (inner, degree_background, "degree_bin_fallback")
        context.layer_cache[key] = result
        return result

    universe_background = np.asarray(
        [idx for idx in range(len(context.nodes)) if idx not in inner_set],
        dtype=int,
    )
    result = (inner, universe_background, "global_fallback")
    context.layer_cache[key] = result
    return result


def seed_annulus_contrast_score(
    values: np.ndarray,
    inner_indices: Sequence[int],
    background_indices: Sequence[int],
    *,
    top_k: int = 10,
) -> float:
    """Score a seed by top inner-neighborhood enrichment over independent background."""

    inner = np.asarray(inner_indices, dtype=int)
    if inner.size == 0:
        return float("-inf")
    k = min(max(1, int(top_k)), inner.size)
    inner_values = np.asarray(values[inner], dtype=float)
    top = np.partition(inner_values, inner_values.size - k)[-k:]
    location, scale = robust_background_location_scale(values, background_indices)
    return float((np.mean(top) - location) * np.sqrt(k) / scale)


def annulus_module_stat(
    values: np.ndarray,
    candidate_indices: Sequence[int],
    background_indices: Sequence[int],
) -> float:
    candidate = np.asarray(candidate_indices, dtype=int)
    if candidate.size == 0:
        return float("-inf")
    location, scale = robust_background_location_scale(values, background_indices)
    return float((np.mean(values[candidate]) - location) * np.sqrt(candidate.size) / scale)


def precision_recall_jaccard(candidate_indices: Sequence[int], oracle_indices: set[int]) -> tuple[float, float, float]:
    candidate = set(int(idx) for idx in candidate_indices)
    if not candidate and not oracle_indices:
        return 1.0, 1.0, 1.0
    if not candidate:
        return 0.0, 0.0, 0.0
    overlap = len(candidate & oracle_indices)
    precision = float(overlap / len(candidate))
    recall = float(overlap / len(oracle_indices)) if oracle_indices else 0.0
    union = len(candidate | oracle_indices)
    jaccard = float(overlap / union) if union else 0.0
    return precision, recall, jaccard


def rank_order(values: np.ndarray) -> np.ndarray:
    return np.argsort(-np.asarray(values, dtype=float), kind="mergesort")


def local_seed_prefilter_order(
    values: np.ndarray,
    context: SearchContext,
    *,
    top_k: int,
) -> np.ndarray:
    """Rank seeds by a cheap one-hop local enrichment prefilter, not raw score."""

    values = np.asarray(values, dtype=float)
    scores = np.empty(len(context.nodes), dtype=float)
    for idx in range(len(context.nodes)):
        neighbors = context.adjacency[int(idx)]
        if neighbors.size:
            local_indices = np.concatenate((np.asarray([idx], dtype=int), neighbors))
        else:
            local_indices = np.asarray([idx], dtype=int)
        k = min(max(1, int(top_k)), local_indices.size)
        local_values = values[local_indices]
        top = np.partition(local_values, local_values.size - k)[-k:]
        scores[int(idx)] = float(np.mean(top))
    return np.argsort(-scores, kind="mergesort")


def permute_within_bins(values: np.ndarray, groups: Sequence[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    out = values.copy()
    for group in groups:
        if group.size > 1:
            out[group] = out[rng.permutation(group)]
    return out


def get_neighborhood(
    context: SearchContext,
    seed_idx: int,
    *,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    key = (int(seed_idx), int(radius))
    cached = context.neighborhood_cache.get(key)
    if cached is not None:
        return cached
    pairs = sorted(bounded_neighborhood_pairs(context, int(seed_idx), max_radius=radius))
    indices = np.asarray([idx for idx, _ in pairs], dtype=int)
    distances = np.asarray([distance for _, distance in pairs], dtype=float)
    context.neighborhood_cache[key] = (indices, distances)
    return indices, distances


def bounded_neighborhood_pairs(
    context: SearchContext,
    seed_idx: int,
    *,
    max_radius: int,
) -> list[tuple[int, int]]:
    visited = {int(seed_idx)}
    frontier = {int(seed_idx)}
    pairs = [(int(seed_idx), 0)]
    for distance in range(1, int(max_radius) + 1):
        next_frontier: set[int] = set()
        for node_idx in frontier:
            next_frontier.update(int(neighbor_idx) for neighbor_idx in context.adjacency[int(node_idx)])
        next_frontier -= visited
        if not next_frontier:
            break
        pairs.extend((int(node_idx), int(distance)) for node_idx in next_frontier)
        visited.update(next_frontier)
        frontier = next_frontier
    return pairs


def top_by_score(values: np.ndarray, indices: np.ndarray, max_size: int) -> np.ndarray:
    if indices.size <= max_size:
        return indices.astype(int)
    local_values = values[indices]
    selected = np.argpartition(-local_values, max_size - 1)[:max_size]
    chosen = indices[selected]
    return chosen[np.argsort(-values[chosen], kind="mergesort")]


def best_soft_top_neighborhood_legacy(
    values: np.ndarray,
    context: SearchContext,
    *,
    n_seeds: int,
    radius: int,
    max_size: int,
    min_size: int = 5,
) -> Candidate:
    best = Candidate("soft_top_neighborhood_legacy", "", (), float("-inf"))
    for seed_idx in rank_order(values)[:n_seeds]:
        neighborhood, _ = get_neighborhood(context, int(seed_idx), radius=radius)
        if neighborhood.size < min_size:
            continue
        candidate_idx = top_by_score(values, neighborhood, max_size=max_size)
        if candidate_idx.size < min_size:
            continue
        stat = positive_sum_stat(values, candidate_idx)
        if stat > best.statistic:
            best = Candidate(
                "soft_top_neighborhood_legacy",
                f"seed={context.nodes[int(seed_idx)]};radius={radius};top={len(candidate_idx)}",
                tuple(int(idx) for idx in candidate_idx),
                stat,
            )
    return best


def connected_greedy_order(
    values: np.ndarray,
    context: SearchContext,
    seed_idx: int,
    neighborhood: np.ndarray,
    *,
    max_size: int,
) -> np.ndarray:
    """Return a connected greedy expansion order constrained to a seed neighborhood."""

    neighborhood_set = set(int(idx) for idx in neighborhood)
    selected: list[int] = [int(seed_idx)]
    selected_set = {int(seed_idx)}
    frontier = {
        int(neighbor_idx)
        for neighbor_idx in context.adjacency[int(seed_idx)]
        if int(neighbor_idx) in neighborhood_set
    }
    frontier -= selected_set
    while frontier and len(selected) < max_size:
        next_idx = max(
            frontier,
            key=lambda idx: (float(values[int(idx)]), -float(context.degrees[int(idx)]), -int(idx)),
        )
        frontier.remove(next_idx)
        selected.append(int(next_idx))
        selected_set.add(int(next_idx))
        for neighbor_idx in context.adjacency[int(next_idx)]:
            neighbor_idx = int(neighbor_idx)
            if neighbor_idx in neighborhood_set and neighbor_idx not in selected_set:
                frontier.add(neighbor_idx)
    return np.asarray(selected, dtype=int)


def connected_greedy_order_annulus(
    values: np.ndarray,
    context: SearchContext,
    seed_idx: int,
    inner_indices: Sequence[int],
    background_indices: Sequence[int],
    *,
    max_size: int,
    edge_gain_weight: float,
    degree_penalty: float,
) -> np.ndarray:
    """Connected expansion with local contrast, edge gain and degree penalty."""

    inner_set = set(int(idx) for idx in inner_indices)
    selected: list[int] = [int(seed_idx)]
    selected_set = {int(seed_idx)}
    location, scale = robust_background_location_scale(values, background_indices)
    frontier = {
        int(neighbor_idx)
        for neighbor_idx in context.adjacency[int(seed_idx)]
        if int(neighbor_idx) in inner_set
    }
    frontier -= selected_set
    edge_counts = {int(node_idx): 1 for node_idx in frontier}
    local_z_cache = {
        int(idx): (float(values[int(idx)]) - location) / scale
        for idx in inner_set
    }
    while frontier and len(selected) < max_size:
        def priority(node_idx: int) -> tuple[float, float, float, int]:
            edges_to_selected = int(edge_counts.get(int(node_idx), 0))
            local_z = float(local_z_cache[int(node_idx)])
            score = (
                local_z
                + float(edge_gain_weight) * np.log1p(edges_to_selected)
                - float(degree_penalty) * np.log1p(float(context.degrees[int(node_idx)]))
            )
            return (float(score), float(local_z), float(edges_to_selected), -int(node_idx))

        next_idx = max(frontier, key=priority)
        frontier.remove(next_idx)
        edge_counts.pop(int(next_idx), None)
        selected.append(int(next_idx))
        selected_set.add(int(next_idx))
        for neighbor_idx in context.adjacency[int(next_idx)]:
            neighbor_idx = int(neighbor_idx)
            if neighbor_idx not in inner_set or neighbor_idx in selected_set:
                continue
            edge_counts[neighbor_idx] = int(edge_counts.get(neighbor_idx, 0)) + 1
            frontier.add(neighbor_idx)
    return np.asarray(selected, dtype=int)


def best_soft_connected_adaptive_neighborhood(
    values: np.ndarray,
    context: SearchContext,
    *,
    n_seeds: int,
    radius: int,
    size_grid: Sequence[int],
    min_size: int = 5,
) -> Candidate:
    """Find a connected soft neighborhood with adaptive size and local normalization."""

    best = Candidate("soft_connected_adaptive_neighborhood", "", (), float("-inf"))
    max_size = max(int(size) for size in size_grid)
    for seed_idx in rank_order(values)[:n_seeds]:
        neighborhood, _ = get_neighborhood(context, int(seed_idx), radius=radius)
        if neighborhood.size < min_size:
            continue
        order = connected_greedy_order(values, context, int(seed_idx), neighborhood, max_size=max_size)
        for target_size in size_grid:
            if int(target_size) > order.size:
                continue
            candidate_idx = order[: int(target_size)]
            if candidate_idx.size < min_size:
                continue
            stat = local_background_z_stat(values, candidate_idx, neighborhood)
            if stat > best.statistic:
                best = Candidate(
                    "soft_connected_adaptive_neighborhood",
                    f"seed={context.nodes[int(seed_idx)]};radius={radius};size={len(candidate_idx)}",
                    tuple(int(idx) for idx in candidate_idx),
                    stat,
                )
    return best


def best_soft_annulus_contrast_neighborhood(
    values: np.ndarray,
    context: SearchContext,
    *,
    n_seeds: int,
    inner_radius: int,
    background_radius: int,
    min_background: int,
    size_grid: Sequence[int],
    edge_gain_weight: float,
    degree_penalty: float,
    seed_pool_size: int | None = None,
    min_size: int = 5,
) -> Candidate:
    """Find a connected soft neighborhood using independent annulus contrast."""

    best = Candidate("soft_annulus_contrast_neighborhood", "", (), float("-inf"))
    max_size = max(int(size) for size in size_grid)
    if seed_pool_size is None:
        seed_pool_size = max(int(n_seeds) * 10, int(n_seeds))
    seed_pool_size = min(max(int(seed_pool_size), int(n_seeds)), len(context.nodes))
    seed_pool = local_seed_prefilter_order(values, context, top_k=min_size)[:seed_pool_size]
    seed_rows: list[tuple[float, int, np.ndarray, np.ndarray, str]] = []
    for seed_idx in seed_pool:
        inner, background, background_source = get_neighborhood_layers(
            context,
            int(seed_idx),
            inner_radius=inner_radius,
            background_radius=background_radius,
            min_background=min_background,
        )
        if inner.size < min_size:
            continue
        seed_score = seed_annulus_contrast_score(values, inner, background, top_k=min_size)
        seed_rows.append((seed_score, int(seed_idx), inner, background, background_source))
    seed_rows.sort(key=lambda item: (item[0], -int(context.degrees[item[1]]), -item[1]), reverse=True)
    for _, seed_idx, inner, background, background_source in seed_rows[:n_seeds]:
        order = connected_greedy_order_annulus(
            values,
            context,
            seed_idx,
            inner,
            background,
            max_size=max_size,
            edge_gain_weight=edge_gain_weight,
            degree_penalty=degree_penalty,
        )
        for target_size in size_grid:
            if int(target_size) > order.size:
                continue
            candidate_idx = order[: int(target_size)]
            if candidate_idx.size < min_size:
                continue
            stat = annulus_module_stat(values, candidate_idx, background)
            if stat > best.statistic:
                best = Candidate(
                    "soft_annulus_contrast_neighborhood",
                    (
                        f"seed={context.nodes[int(seed_idx)]};inner={inner_radius};"
                        f"background={background_radius};size={len(candidate_idx)};"
                        f"background_source={background_source}"
                    ),
                    tuple(int(idx) for idx in candidate_idx),
                    stat,
                    seed=str(context.nodes[int(seed_idx)]),
                    inner_radius=int(inner_radius),
                    background_radius=int(background_radius),
                    background_n=int(len(background)),
                    score_basis=f"local_one_hop_top{min_size}_seed_prefilter;annulus_contrast:{background_source}",
                )
    return best


def best_soft_connected_adaptive_delta_neighborhood(
    delta_values: np.ndarray,
    context: SearchContext,
    *,
    n_seeds: int,
    radius: int,
    size_grid: Sequence[int],
    min_size: int = 5,
) -> Candidate:
    """Spike-in-only connected soft diagnostic using sparse delta scores."""

    best = Candidate("soft_connected_adaptive_delta_neighborhood", "", (), float("-inf"))
    max_size = max(int(size) for size in size_grid)
    for seed_idx in rank_order(delta_values)[:n_seeds]:
        if delta_values[int(seed_idx)] <= 0:
            continue
        neighborhood, _ = get_neighborhood(context, int(seed_idx), radius=radius)
        if neighborhood.size < min_size:
            continue
        order = connected_greedy_order(delta_values, context, int(seed_idx), neighborhood, max_size=max_size)
        for target_size in size_grid:
            if int(target_size) > order.size:
                continue
            candidate_idx = order[: int(target_size)]
            if candidate_idx.size < min_size:
                continue
            stat = positive_sum_stat(delta_values, candidate_idx)
            if stat > best.statistic:
                best = Candidate(
                    "soft_connected_adaptive_delta_neighborhood",
                    f"seed={context.nodes[int(seed_idx)]};radius={radius};size={len(candidate_idx)}",
                    tuple(int(idx) for idx in candidate_idx),
                    stat,
                )
    return best


def best_diffusion_neighborhood(
    values: np.ndarray,
    context: SearchContext,
    *,
    n_seeds: int,
    radius: int,
    tau_grid: Sequence[float],
    top_size: int,
    min_size: int = 5,
) -> Candidate:
    best = Candidate("diffusion_localized_neighborhood", "", (), float("-inf"), None)
    for seed_idx in rank_order(values)[:n_seeds]:
        neighborhood, distances = get_neighborhood(context, int(seed_idx), radius=radius)
        if neighborhood.size < min_size:
            continue
        for tau in tau_grid:
            weights = np.exp(-distances / float(tau))
            stat = positive_sum_stat(values, neighborhood, weights=weights)
            contributions = weights * np.maximum(0.0, values[neighborhood])
            top_n = min(int(top_size), neighborhood.size)
            if top_n < neighborhood.size:
                selected = np.argpartition(-contributions, top_n - 1)[:top_n]
                candidate_idx = neighborhood[selected]
                candidate_idx = candidate_idx[np.argsort(-contributions[selected], kind="mergesort")]
            else:
                candidate_idx = neighborhood[np.argsort(-contributions, kind="mergesort")]
            if stat > best.statistic:
                best = Candidate(
                    "diffusion_localized_neighborhood",
                    f"seed={context.nodes[int(seed_idx)]};radius={radius};tau={tau:g}",
                    tuple(int(idx) for idx in candidate_idx),
                    stat,
                    float(tau),
                )
    return best


def best_from_fixed_sets(
    values: np.ndarray,
    sets: Sequence[Sequence[int]],
    ids: Sequence[str],
    *,
    method: str,
) -> Candidate:
    best = Candidate(method, "", (), float("-inf"))
    for set_id, indices in zip(ids, sets, strict=True):
        if len(indices) == 0:
            continue
        stat = positive_sum_stat(values, indices)
        if stat > best.statistic:
            best = Candidate(method, str(set_id), tuple(int(idx) for idx in indices), stat)
    return best


def sample_degree_matched_indices(
    target_indices: Sequence[int],
    context: SearchContext,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    profile: dict[int, int] = {}
    for idx in target_indices:
        bin_id = int(context.bins[int(idx)])
        profile[bin_id] = profile.get(bin_id, 0) + 1
    sampled: list[int] = []
    for bin_id, count in sorted(profile.items()):
        candidates = context.bin_to_indices[int(bin_id)]
        replace = int(count) > candidates.size
        sampled.extend(int(idx) for idx in rng.choice(candidates, size=int(count), replace=replace))
    rng.shuffle(sampled)
    return tuple(sampled)


def fixed_degree_matched_null(
    values: np.ndarray,
    target_indices: Sequence[int],
    context: SearchContext,
    *,
    n_replicates: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(n_replicates, dtype=float)
    for replicate in range(n_replicates):
        sampled = sample_degree_matched_indices(target_indices, context, rng)
        out[replicate] = positive_sum_stat(values, sampled)
    return out


def build_context(
    graph: nx.Graph,
    scores: pd.DataFrame,
    oracles: pd.DataFrame,
    *,
    degree_bins: int,
    community_min_size: int,
    community_max_size: int,
    community_resolution: float,
    seed: int,
) -> SearchContext:
    work = scores.copy()
    work["gene_symbol"] = work["gene_symbol"].astype(str)
    nodes = work["gene_symbol"].to_numpy(dtype=object)
    node_to_idx = {str(node): idx for idx, node in enumerate(nodes)}
    degrees = work["graph_degree"].to_numpy(dtype=float)
    bins = assign_degree_bins(pd.Series(degrees), n_bins=degree_bins).to_numpy(dtype=int)
    bin_to_indices = {
        int(bin_id): np.flatnonzero(bins == bin_id)
        for bin_id in sorted(np.unique(bins))
    }
    adjacency = make_adjacency_index(graph, nodes, node_to_idx)
    print("Computing Louvain communities for V1.1 community-anchored diagnostic...", flush=True)
    raw_communities = nx.community.louvain_communities(
        graph,
        resolution=float(community_resolution),
        seed=int(seed),
    )
    community_indices: list[tuple[int, ...]] = []
    community_ids: list[str] = []
    for community_idx, community in enumerate(raw_communities, start=1):
        indices = tuple(sorted(node_to_idx[str(node)] for node in community if str(node) in node_to_idx))
        if community_min_size <= len(indices) <= community_max_size:
            community_indices.append(indices)
            community_ids.append(f"C{community_idx:04d}")
    oracle_sets: list[tuple[int, ...]] = []
    oracle_ids: list[str] = []
    for row in oracles.to_dict(orient="records"):
        indices = tuple(sorted(node_to_idx[gene] for gene in split_genes(row["oracle_genes"]) if gene in node_to_idx))
        if indices:
            oracle_sets.append(indices)
            oracle_ids.append(str(row["oracle_id"]))
    return SearchContext(
        graph=graph,
        nodes=nodes,
        node_to_idx=node_to_idx,
        degrees=degrees,
        bins=bins,
        bin_to_indices=bin_to_indices,
        adjacency=adjacency,
        communities=tuple(community_indices),
        community_ids=tuple(community_ids),
        oracle_library=tuple(oracle_sets),
        oracle_library_ids=tuple(oracle_ids),
        neighborhood_cache={},
        layer_cache={},
    )


def method_nulls(
    values: np.ndarray,
    delta_values: np.ndarray,
    context: SearchContext,
    groups: Sequence[np.ndarray],
    *,
    n_replicates: int,
    seed: int,
    n_seeds: int,
    radius: int,
    soft_max_size: int,
    soft_size_grid: Sequence[int],
    soft_inner_radius: int,
    soft_background_radius: int,
    soft_annulus_min_background: int,
    soft_edge_gain_weight: float,
    soft_degree_penalty: float,
    soft_seed_pool_size: int,
    diffusion_top_size: int,
    tau_grid: Sequence[float],
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = {
        "soft_top_neighborhood_legacy": np.empty(n_replicates, dtype=float),
        "soft_connected_adaptive_neighborhood": np.empty(n_replicates, dtype=float),
        "soft_connected_adaptive_delta_neighborhood": np.empty(n_replicates, dtype=float),
        "soft_annulus_contrast_neighborhood": np.empty(n_replicates, dtype=float),
        "diffusion_localized_neighborhood": np.empty(n_replicates, dtype=float),
        "community_louvain_anchor": np.empty(n_replicates, dtype=float),
        "oracle_gene_set_library_anchor": np.empty(n_replicates, dtype=float),
    }
    for replicate in range(n_replicates):
        permuted = permute_within_bins(values, groups, rng)
        permuted_delta = permute_within_bins(delta_values, groups, rng)
        out["soft_top_neighborhood_legacy"][replicate] = best_soft_top_neighborhood_legacy(
            permuted,
            context,
            n_seeds=n_seeds,
            radius=radius,
            max_size=soft_max_size,
        ).statistic
        out["soft_connected_adaptive_neighborhood"][replicate] = best_soft_connected_adaptive_neighborhood(
            permuted,
            context,
            n_seeds=n_seeds,
            radius=radius,
            size_grid=soft_size_grid,
        ).statistic
        out["soft_connected_adaptive_delta_neighborhood"][replicate] = best_soft_connected_adaptive_delta_neighborhood(
            permuted_delta,
            context,
            n_seeds=n_seeds,
            radius=radius,
            size_grid=soft_size_grid,
        ).statistic
        out["soft_annulus_contrast_neighborhood"][replicate] = best_soft_annulus_contrast_neighborhood(
            permuted,
            context,
            n_seeds=n_seeds,
            inner_radius=soft_inner_radius,
            background_radius=soft_background_radius,
            min_background=soft_annulus_min_background,
            size_grid=soft_size_grid,
            edge_gain_weight=soft_edge_gain_weight,
            degree_penalty=soft_degree_penalty,
            seed_pool_size=soft_seed_pool_size,
        ).statistic
        out["diffusion_localized_neighborhood"][replicate] = best_diffusion_neighborhood(
            permuted,
            context,
            n_seeds=n_seeds,
            radius=radius,
            tau_grid=tau_grid,
            top_size=diffusion_top_size,
        ).statistic
        out["community_louvain_anchor"][replicate] = best_from_fixed_sets(
            permuted,
            context.communities,
            context.community_ids,
            method="community_louvain_anchor",
        ).statistic
        out["oracle_gene_set_library_anchor"][replicate] = best_from_fixed_sets(
            permuted,
            context.oracle_library,
            context.oracle_library_ids,
            method="oracle_gene_set_library_anchor",
        ).statistic
    return out


def candidate_summary_row(
    *,
    scenario: Mapping[str, object],
    method: str,
    candidate: Candidate,
    oracle_indices: set[int],
    null_values: np.ndarray,
    null_type: str,
    selection_scope: str,
    is_oracle_assisted: bool,
    diagnostic_only: bool,
    context: SearchContext,
) -> dict[str, object]:
    precision, recall, jaccard = precision_recall_jaccard(candidate.node_indices, oracle_indices)
    summary = finite_summary(null_values, candidate.statistic)
    genes = ",".join(str(context.nodes[idx]) for idx in candidate.node_indices[:300])
    if len(candidate.node_indices) > 300:
        genes += ",..."
    return {
        "scenario_id": scenario["scenario_id"],
        "oracle_id": scenario["oracle_id"],
        "method": method,
        "module_definition_family": method,
        "module_size": int(scenario["module_size"]),
        "degree_bin": scenario["degree_bin"],
        "architecture": scenario["architecture"],
        "effect_label": scenario["effect_label"],
        "effect_size": float(scenario["effect_size"]),
        "observed_stat": float(candidate.statistic),
        "null_type": null_type,
        "selection_scope": selection_scope,
        "n_null": int(summary["n_null"]),
        "null_mean": summary["null_mean"],
        "null_sd": summary["null_sd"],
        "z": summary["z"],
        "empirical_p": summary["empirical_p"],
        "pass_p05": summary["pass_p05"],
        "best_candidate_id": candidate.candidate_id,
        "candidate_seed": candidate.seed,
        "inner_radius": int(candidate.inner_radius) if candidate.inner_radius is not None else np.nan,
        "background_radius": int(candidate.background_radius)
        if candidate.background_radius is not None
        else np.nan,
        "background_n": int(candidate.background_n) if candidate.background_n is not None else np.nan,
        "candidate_score_basis": candidate.score_basis,
        "best_candidate_size": int(len(candidate.node_indices)),
        "best_candidate_tau": float(candidate.tau) if candidate.tau is not None else np.nan,
        "best_precision": precision,
        "best_recall": recall,
        "best_jaccard": jaccard,
        "is_oracle_assisted": bool(is_oracle_assisted),
        "diagnostic_only": bool(diagnostic_only),
        "best_candidate_genes": genes,
    }


def run_diagnostic(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores, graph = prepare_scores_and_graph()
    oracles = pd.read_csv(args.oracle_modules, sep="\t")
    effect_grid = parse_effect_grid(args.effect_grid)
    tau_grid = parse_float_grid(args.tau_grid)
    soft_size_grid = parse_int_grid(args.soft_size_grid)
    context = build_context(
        graph,
        scores,
        oracles,
        degree_bins=args.degree_bins,
        community_min_size=args.community_min_size,
        community_max_size=args.community_max_size,
        community_resolution=args.community_resolution,
        seed=args.seed,
    )
    groups = [np.flatnonzero(context.bins == bin_id) for bin_id in sorted(np.unique(context.bins))]
    baseline_values = scores["assoc_resid_score"].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    null_rows: list[pd.DataFrame] = []
    scenario_index = 0
    for oracle in oracles.to_dict(orient="records"):
        oracle_genes = set(split_genes(oracle["oracle_genes"]))
        oracle_indices = {context.node_to_idx[gene] for gene in oracle_genes if gene in context.node_to_idx}
        if not oracle_indices:
            continue
        for effect_label, effect_size in effect_grid.items():
            scenario_index += 1
            if args.max_scenarios is not None and scenario_index > args.max_scenarios:
                break
            scenario_id = f"{oracle['oracle_id']}_{effect_label}"
            print(f"V1.1 module-definition diagnostic: {scenario_id}", flush=True)
            spiked = score_spikein(scores, oracle_genes, float(effect_size))
            values = spiked["assoc_resid_score"].to_numpy(dtype=float)
            delta_values = values - baseline_values
            scenario = {
                "scenario_id": scenario_id,
                "oracle_id": str(oracle["oracle_id"]),
                "module_size": int(oracle.get("module_size", oracle.get("target_size", oracle.get("oracle_n_genes")))),
                "degree_bin": str(oracle.get("degree_bin", oracle.get("degree_stratum", ""))),
                "architecture": str(oracle.get("architecture", oracle.get("oracle_type", ""))),
                "effect_label": effect_label,
                "effect_size": float(effect_size),
            }
            soft_legacy = best_soft_top_neighborhood_legacy(
                values,
                context,
                n_seeds=args.n_seeds,
                radius=args.neighborhood_radius,
                max_size=args.soft_max_size,
            )
            soft_connected = best_soft_connected_adaptive_neighborhood(
                values,
                context,
                n_seeds=args.n_seeds,
                radius=args.neighborhood_radius,
                size_grid=soft_size_grid,
            )
            soft_delta = best_soft_connected_adaptive_delta_neighborhood(
                delta_values,
                context,
                n_seeds=args.n_seeds,
                radius=args.neighborhood_radius,
                size_grid=soft_size_grid,
            )
            soft_annulus = best_soft_annulus_contrast_neighborhood(
                values,
                context,
                n_seeds=args.n_seeds,
                inner_radius=args.soft_inner_radius,
                background_radius=args.soft_background_radius,
                min_background=args.soft_annulus_min_background,
                size_grid=soft_size_grid,
                edge_gain_weight=args.soft_edge_gain_weight,
                degree_penalty=args.soft_degree_penalty,
                seed_pool_size=args.soft_seed_pool_size,
            )
            diffusion = best_diffusion_neighborhood(
                values,
                context,
                n_seeds=args.n_seeds,
                radius=args.neighborhood_radius,
                tau_grid=tau_grid,
                top_size=args.diffusion_top_size,
            )
            community = best_from_fixed_sets(
                values,
                context.communities,
                context.community_ids,
                method="community_louvain_anchor",
            )
            oracle_library = best_from_fixed_sets(
                values,
                context.oracle_library,
                context.oracle_library_ids,
                method="oracle_gene_set_library_anchor",
            )
            oracle_fixed = Candidate(
                "oracle_fixed_anchor",
                str(oracle["oracle_id"]),
                tuple(sorted(oracle_indices)),
                positive_sum_stat(values, tuple(sorted(oracle_indices))),
            )
            nulls = method_nulls(
                values,
                delta_values,
                context,
                groups,
                n_replicates=args.n_null,
                seed=args.seed + scenario_index * 100,
                n_seeds=args.n_seeds,
                radius=args.neighborhood_radius,
                soft_max_size=args.soft_max_size,
                soft_size_grid=soft_size_grid,
                soft_inner_radius=args.soft_inner_radius,
                soft_background_radius=args.soft_background_radius,
                soft_annulus_min_background=args.soft_annulus_min_background,
                soft_edge_gain_weight=args.soft_edge_gain_weight,
                soft_degree_penalty=args.soft_degree_penalty,
                soft_seed_pool_size=args.soft_seed_pool_size,
                diffusion_top_size=args.diffusion_top_size,
                tau_grid=tau_grid,
            )
            fixed_null = fixed_degree_matched_null(
                values,
                oracle_fixed.node_indices,
                context,
                n_replicates=args.n_null,
                seed=args.seed + scenario_index * 100 + 51,
            )
            method_candidates = {
                "soft_top_neighborhood_legacy": (
                    soft_legacy,
                    nulls["soft_top_neighborhood_legacy"],
                    "degree_stratified_score_permutation",
                    "legacy_score-selected_seed_neighborhood_max",
                    False,
                    False,
                ),
                "soft_connected_adaptive_neighborhood": (
                    soft_connected,
                    nulls["soft_connected_adaptive_neighborhood"],
                    "degree_stratified_score_permutation",
                    "connected_seed_neighborhood_adaptive_size_local_background_max",
                    False,
                    False,
                ),
                "soft_connected_adaptive_delta_neighborhood": (
                    soft_delta,
                    nulls["soft_connected_adaptive_delta_neighborhood"],
                    "degree_stratified_delta_score_permutation",
                    "spikein_only_delta_connected_seed_neighborhood_adaptive_size",
                    False,
                    True,
                ),
                "soft_annulus_contrast_neighborhood": (
                    soft_annulus,
                    nulls["soft_annulus_contrast_neighborhood"],
                    "degree_stratified_score_permutation_annulus_contrast",
                    "connected_seed_annulus_contrast_adaptive_size",
                    False,
                    False,
                ),
                "diffusion_localized_neighborhood": (
                    diffusion,
                    nulls["diffusion_localized_neighborhood"],
                    "degree_stratified_score_permutation",
                    "score-selected_seed_tau_neighborhood_max",
                    False,
                    False,
                ),
                "community_louvain_anchor": (
                    community,
                    nulls["community_louvain_anchor"],
                    "degree_stratified_score_permutation",
                    "max_over_fixed_louvain_communities",
                    False,
                    False,
                ),
                "oracle_gene_set_library_anchor": (
                    oracle_library,
                    nulls["oracle_gene_set_library_anchor"],
                    "degree_stratified_score_permutation",
                    "max_over_oracle_positive_control_library",
                    True,
                    False,
                ),
                "oracle_fixed_anchor": (
                    oracle_fixed,
                    fixed_null,
                    "degree_matched_node_set",
                    "fixed_true_oracle_module_upper_bound",
                    True,
                    False,
                ),
            }
            for (
                method,
                (candidate, null_values, null_type, selection_scope, oracle_assisted, diagnostic_only),
            ) in method_candidates.items():
                row = candidate_summary_row(
                    scenario=scenario,
                    method=method,
                    candidate=candidate,
                    oracle_indices=oracle_indices,
                    null_values=null_values,
                    null_type=null_type,
                    selection_scope=selection_scope,
                    is_oracle_assisted=oracle_assisted,
                    diagnostic_only=diagnostic_only,
                    context=context,
                )
                rows.append(row)
                candidate_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "method": method,
                        "candidate_id": candidate.candidate_id,
                        "candidate_seed": candidate.seed,
                        "inner_radius": int(candidate.inner_radius) if candidate.inner_radius is not None else np.nan,
                        "background_radius": int(candidate.background_radius)
                        if candidate.background_radius is not None
                        else np.nan,
                        "background_n": int(candidate.background_n) if candidate.background_n is not None else np.nan,
                        "candidate_score_basis": candidate.score_basis,
                        "candidate_size": int(len(candidate.node_indices)),
                        "candidate_tau": float(candidate.tau) if candidate.tau is not None else np.nan,
                        "candidate_genes": row["best_candidate_genes"],
                        "is_oracle_assisted": bool(oracle_assisted),
                        "diagnostic_only": bool(diagnostic_only),
                    }
                )
                null_rows.append(
                    pd.DataFrame(
                        {
                            "scenario_id": scenario_id,
                            "method": method,
                            "replicate": np.arange(len(null_values), dtype=int),
                            "null_stat": np.asarray(null_values, dtype=float),
                            "null_type": null_type,
                        }
                    )
                )
        if args.max_scenarios is not None and scenario_index >= args.max_scenarios:
            break
    summary = pd.DataFrame(rows)
    candidates = pd.DataFrame(candidate_rows)
    null_table = pd.concat(null_rows, ignore_index=True) if null_rows else pd.DataFrame()
    return summary, candidates, null_table


def power_curve(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    grouped = summary.groupby(
        ["method", "is_oracle_assisted", "diagnostic_only", "effect_label", "effect_size"],
        observed=True,
    )
    return grouped.agg(
        n_scenarios=("scenario_id", "count"),
        pass_rate=("pass_p05", "mean"),
        median_empirical_p=("empirical_p", "median"),
        median_z=("z", "median"),
        median_jaccard=("best_jaccard", "median"),
        median_recall=("best_recall", "median"),
        median_candidate_size=("best_candidate_size", "median"),
    ).reset_index()


def recovery_aware_method_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_cols = ["method", "is_oracle_assisted", "diagnostic_only"]
    for (method, oracle_assisted, diagnostic_only), group in summary.groupby(group_cols, observed=True):
        passed = group["pass_p05"].astype(bool)
        jaccard = pd.to_numeric(group["best_jaccard"], errors="coerce")
        rows.append(
            {
                "method": method,
                "is_oracle_assisted": bool(oracle_assisted),
                "diagnostic_only": bool(diagnostic_only),
                "n_scenarios": int(len(group)),
                "pass_count": int(passed.sum()),
                "pass_rate": float(passed.mean()),
                "pass_jaccard_ge_025_count": int((passed & (jaccard >= 0.25)).sum()),
                "pass_jaccard_ge_025_rate": float((passed & (jaccard >= 0.25)).mean()),
                "pass_jaccard_ge_05_count": int((passed & (jaccard >= 0.50)).sum()),
                "pass_jaccard_ge_05_rate": float((passed & (jaccard >= 0.50)).mean()),
                "pass_low_recovery_lt_025_count": int((passed & (jaccard < 0.25)).sum()),
                "median_empirical_p": float(pd.to_numeric(group["empirical_p"], errors="coerce").median()),
                "median_z": float(pd.to_numeric(group["z"], errors="coerce").median()),
                "median_jaccard": float(jaccard.median()),
                "median_recall": float(pd.to_numeric(group["best_recall"], errors="coerce").median()),
                "median_candidate_size": float(pd.to_numeric(group["best_candidate_size"], errors="coerce").median()),
            }
        )
    return pd.DataFrame(rows)


def render_report(
    summary: pd.DataFrame,
    power: pd.DataFrame,
    recovery_summary: pd.DataFrame,
    manifest: Mapping[str, object],
) -> str:
    lines = [
        "# RIPPLE Tier 4 V1.1 Module-Definition Diagnostic",
        "",
        f"Created: {manifest['created_utc']}",
        "",
        "This is a design diagnostic, not a V1 claim upgrade. Oracle-assisted rows are positive controls and should not be interpreted as real-trait discovery evidence.",
        "",
        "## Method-Level Power",
        "",
        "| Method | Oracle assisted | Diagnostic only | Pass rate | Pass + Jaccard >= 0.25 | Pass + Jaccard >= 0.50 | Median P | Median Jaccard |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in recovery_summary.to_dict(orient="records"):
        lines.append(
            f"| {row['method']} | {row['is_oracle_assisted']} | {row['diagnostic_only']} | "
            f"{float(row['pass_rate']):.3f} | "
            f"{int(row['pass_jaccard_ge_025_count'])}/{int(row['n_scenarios'])} | "
            f"{int(row['pass_jaccard_ge_05_count'])}/{int(row['n_scenarios'])} | "
            f"{float(row['median_empirical_p']):.4g} | {float(row['median_jaccard']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Effect-Size Power Curve",
            "",
            "| Method | Diagnostic only | Effect | Pass rate | Median P | Median Jaccard |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in power.to_dict(orient="records"):
        lines.append(
            f"| {row['method']} | {row['diagnostic_only']} | {float(row['effect_size']):.1f} | "
            f"{float(row['pass_rate']):.3f} | "
            f"{float(row['median_empirical_p']):.4g} | {float(row['median_jaccard']):.3f} |"
        )
    non_oracle = summary.loc[~summary["is_oracle_assisted"].astype(bool)].copy()
    best_non_oracle = (
        non_oracle.sort_values(["pass_p05", "empirical_p", "best_jaccard"], ascending=[False, True, False]).head(10)
        if not non_oracle.empty
        else pd.DataFrame()
    )
    lines.extend(
        [
            "",
            "## Top Non-Oracle Candidate Recoveries",
            "",
            "| Scenario | Method | Diagnostic only | P | Z | Jaccard | Recall | Candidate size |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in best_non_oracle.to_dict(orient="records"):
        lines.append(
            f"| {row['scenario_id']} | {row['method']} | {row['diagnostic_only']} | "
            f"{float(row['empirical_p']):.4g} | "
            f"{float(row['z']):.3f} | {float(row['best_jaccard']):.3f} | "
            f"{float(row['best_recall']):.3f} | {int(row['best_candidate_size'])} |"
        )
    lines.extend(
        [
            "",
            "## Working Interpretation",
            "",
            "- If oracle_fixed_anchor passes but non-oracle methods fail, the remaining problem is discovery/search, not score-level detectability.",
            "- If community_louvain_anchor passes with low recovery, graph communities may capture signal statistically but not resolve the true module.",
            "- `soft_top_neighborhood_legacy` is retained as a failure-control; pass-with-low-recovery indicates hub/background capture.",
            "- `soft_connected_adaptive_neighborhood` repairs connectivity, adaptive size, and local background normalization, but low recovery still indicates raw trait-background domination.",
            "- `soft_annulus_contrast_neighborhood` is the V1.1 candidate diagnostic: it ranks seeds by local annulus enrichment and scores connected adaptive modules against an independent annulus/fallback background.",
            "- `soft_connected_adaptive_delta_neighborhood` is spike-in-only. It tests whether the repaired geometry can recover injected modules after subtracting the unspiked baseline; it is not directly available for real-trait discovery.",
            "- If soft or diffusion-localized neighborhoods pass more often than V1 hard components, V1.1 should still require recovery or stability guardrails before replacing induced top-component extraction.",
            "- A non-oracle method that passes P <= 0.05 but has low oracle recovery is not a successful module definition; it is likely detecting unrelated high-score neighborhoods.",
            "- These diagnostics require a fresh Type I calibration before any V1.1 method can replace V1 Tier 4.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.force:
        raise FileExistsError(f"{args.out_dir} exists and is not empty. Use --force.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "reports").mkdir(parents=True, exist_ok=True)
    summary, candidates, nulls = run_diagnostic(args)
    power = power_curve(summary)
    recovery_summary = recovery_aware_method_summary(summary)
    write_table(args.out_dir / "tables" / "v11_module_definition_summary.tsv", summary)
    write_table(args.out_dir / "tables" / "v11_candidate_definitions.tsv", candidates)
    write_table(args.out_dir / "tables" / "v11_module_definition_nulls.tsv.gz", nulls)
    write_table(args.out_dir / "tables" / "v11_method_power_curve.tsv", power)
    write_table(args.out_dir / "tables" / "v11_recovery_aware_method_summary.tsv", recovery_summary)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "analysis_id": "tier4_v11_module_definition_diagnostic_v1",
        "script_path": str(THIS_SCRIPT),
        "oracle_modules": str(args.oracle_modules),
        "n_null": int(args.n_null),
        "degree_bins": int(args.degree_bins),
        "effect_grid": args.effect_grid,
        "n_seeds": int(args.n_seeds),
        "neighborhood_radius": int(args.neighborhood_radius),
        "soft_max_size": int(args.soft_max_size),
        "soft_size_grid": args.soft_size_grid,
        "soft_inner_radius": int(args.soft_inner_radius),
        "soft_background_radius": int(args.soft_background_radius),
        "soft_annulus_min_background": int(args.soft_annulus_min_background),
        "soft_edge_gain_weight": float(args.soft_edge_gain_weight),
        "soft_degree_penalty": float(args.soft_degree_penalty),
        "soft_seed_pool_size": int(args.soft_seed_pool_size),
        "diffusion_top_size": int(args.diffusion_top_size),
        "tau_grid": args.tau_grid,
        "community_min_size": int(args.community_min_size),
        "community_max_size": int(args.community_max_size),
        "community_resolution": float(args.community_resolution),
        "seed": int(args.seed),
        "outputs": {
            "summary": str(args.out_dir / "tables" / "v11_module_definition_summary.tsv"),
            "candidates": str(args.out_dir / "tables" / "v11_candidate_definitions.tsv"),
            "nulls": str(args.out_dir / "tables" / "v11_module_definition_nulls.tsv.gz"),
            "power_curve": str(args.out_dir / "tables" / "v11_method_power_curve.tsv"),
            "recovery_aware_method_summary": str(
                args.out_dir / "tables" / "v11_recovery_aware_method_summary.tsv"
            ),
        },
    }
    (args.out_dir / "tier4_v11_module_definition_diagnostic_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.out_dir / "reports" / "tier4_v11_module_definition_diagnostic_report.md").write_text(
        render_report(summary, power, recovery_summary, manifest),
        encoding="utf-8",
    )
    print(f"Wrote V1.1 module-definition diagnostic outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
