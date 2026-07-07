"""Heat-kernel graph-domain weak-signal aggregation statistics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.sparse.linalg import expm_multiply

from ripple.graph import graph_laplacian
from ripple.nulls.graph_nulls import degree_preserving_graph_replicates
from ripple.nulls.score_permutation import assign_degree_bins

DiffusionScoreMode = Literal["positive", "absolute", "raw", "rank"]
DiffusionNullType = Literal["degree_stratified", "degree_preserving_graph"]

DEFAULT_TAU_GRID: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
DEFAULT_DIFFUSION_EPS: float = 1e-12
DEFAULT_DIFFUSION_BATCH_SIZE: int = 128


def parse_tau_grid(value: str | Iterable[float] | None = None) -> tuple[float, ...]:
    """Parse and validate a fixed tau grid."""

    if value is None:
        return DEFAULT_TAU_GRID
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        taus = tuple(float(part) for part in parts)
    else:
        taus = tuple(float(item) for item in value)
    if not taus:
        raise ValueError("tau grid must contain at least one value.")
    if any((not np.isfinite(tau)) or tau <= 0.0 for tau in taus):
        raise ValueError("all tau values must be positive finite numbers.")
    return taus


def _score_vector_from_values(values: Sequence[float], *, mode: DiffusionScoreMode) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError("score vector must be one-dimensional.")
    if not np.all(np.isfinite(x)):
        raise ValueError("score vector contains non-finite values.")

    if mode == "positive":
        return np.maximum(0.0, x)
    if mode == "absolute":
        return np.abs(x)
    if mode == "raw":
        return x.astype(float)
    if mode == "rank":
        if x.size == 0:
            return x.astype(float)
        ranks = stats.rankdata(x, method="average")
        probs = (ranks - 0.5) / x.size
        return stats.norm.ppf(probs).astype(float)
    raise ValueError("mode must be one of: positive, absolute, raw, rank.")


def aligned_score_vector(
    scores: pd.DataFrame,
    nodes: Sequence[str],
    *,
    node_col: str = "gene_symbol",
    score_col: str = "assoc_resid_score",
    mode: DiffusionScoreMode = "positive",
) -> np.ndarray:
    """Return a graph-node-aligned diffusion score vector."""

    missing = [col for col in (node_col, score_col) if col not in scores.columns]
    if missing:
        raise ValueError(f"scores is missing required columns: {missing}")

    work = scores.loc[:, [node_col, score_col]].copy()
    work[node_col] = work[node_col].astype(str)
    if work[node_col].duplicated().any():
        duplicated = work.loc[work[node_col].duplicated(), node_col].iloc[0]
        raise ValueError(f"node column must be unique; first duplicate: {duplicated}")
    by_node = dict(zip(work[node_col], pd.to_numeric(work[score_col], errors="raise"), strict=True))
    missing_nodes = [str(node) for node in nodes if str(node) not in by_node]
    if missing_nodes:
        raise ValueError(f"score table is missing {len(missing_nodes)} graph nodes.")
    raw = np.array([float(by_node[str(node)]) for node in nodes], dtype=float)
    return _score_vector_from_values(raw, mode=mode)


def heat_kernel_tau_statistics(
    laplacian: sparse.spmatrix,
    score_vector: Sequence[float],
    *,
    tau_grid: Iterable[float] = DEFAULT_TAU_GRID,
    eps: float = DEFAULT_DIFFUSION_EPS,
) -> pd.DataFrame:
    """Compute `T_tau = s^T exp(-tau L) s / (s^T s + eps)` for each tau."""

    taus = parse_tau_grid(tau_grid)
    lap = sparse.csr_matrix(laplacian)
    s = np.asarray(score_vector, dtype=float)
    if s.ndim != 1:
        raise ValueError("score_vector must be one-dimensional.")
    if lap.shape != (s.size, s.size):
        raise ValueError("laplacian dimensions must match score_vector length.")
    denom = float(s @ s) + float(eps)
    rows: list[dict[str, float]] = []
    for tau in taus:
        y = expm_multiply(-float(tau) * lap, s)
        rows.append({"tau": float(tau), "T_tau": float(s @ y) / denom})
    return pd.DataFrame(rows)


def heat_kernel_tau_statistics_matrix(
    laplacian: sparse.spmatrix,
    score_matrix: Sequence[Sequence[float]],
    *,
    tau_grid: Iterable[float] = DEFAULT_TAU_GRID,
    eps: float = DEFAULT_DIFFUSION_EPS,
    batch_size: int = DEFAULT_DIFFUSION_BATCH_SIZE,
) -> np.ndarray:
    """Compute heat-kernel statistics for many score vectors.

    Rows in `score_matrix` are independent score vectors. The returned matrix
    has shape `(n_vectors, n_taus)` and column order matching `tau_grid`.
    """

    taus = parse_tau_grid(tau_grid)
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    lap = sparse.csr_matrix(laplacian)
    scores = np.asarray(score_matrix, dtype=float)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)
    if scores.ndim != 2:
        raise ValueError("score_matrix must be one- or two-dimensional.")
    if lap.shape != (scores.shape[1], scores.shape[1]):
        raise ValueError("laplacian dimensions must match score vector length.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("score_matrix contains non-finite values.")

    out = np.empty((scores.shape[0], len(taus)), dtype=float)
    for start in range(0, scores.shape[0], batch_size):
        stop = min(start + batch_size, scores.shape[0])
        batch = scores[start:stop, :]
        rhs = batch.T
        denom = np.sum(batch * batch, axis=1) + float(eps)
        for tau_idx, tau in enumerate(taus):
            propagated = expm_multiply(-float(tau) * lap, rhs)
            if propagated.ndim == 1:
                propagated = propagated.reshape(-1, 1)
            numerator = np.sum(rhs * propagated, axis=0)
            out[start:stop, tau_idx] = numerator / denom
    return out


def summarize_tau_statistics(tau_stats: pd.DataFrame) -> dict[str, float]:
    """Return the maxT omnibus statistic and its tau."""

    if tau_stats.empty:
        return {"T_max": float("nan"), "tau_at_max": float("nan")}
    idx = int(pd.to_numeric(tau_stats["T_tau"], errors="raise").to_numpy(dtype=float).argmax())
    return {
        "T_max": float(tau_stats.iloc[idx]["T_tau"]),
        "tau_at_max": float(tau_stats.iloc[idx]["tau"]),
    }


def observed_diffusion_statistics(
    graph: nx.Graph,
    scores: pd.DataFrame,
    *,
    score_mode: DiffusionScoreMode = "positive",
    tau_grid: Iterable[float] = DEFAULT_TAU_GRID,
    weighted_laplacian: bool = False,
    edge_weight_column: str = "weight",
    score_col: str = "assoc_resid_score",
) -> tuple[pd.DataFrame, dict[str, float | bool | str]]:
    """Compute observed heat-kernel statistics for a graph and score table."""

    nodes = tuple(str(node) for node in scores["gene_symbol"].astype(str) if graph.has_node(str(node)))
    lap = graph_laplacian(
        graph,
        nodes=nodes,
        kind="normalized",
        weight=edge_weight_column if weighted_laplacian else None,
    )
    s = aligned_score_vector(scores, lap.nodes, score_col=score_col, mode=score_mode)
    tau_stats = heat_kernel_tau_statistics(lap.laplacian, s, tau_grid=tau_grid)
    summary = summarize_tau_statistics(tau_stats)
    summary.update(
        {
            "weighted_laplacian_used": bool(weighted_laplacian),
            "edge_weight_column": edge_weight_column if weighted_laplacian else "",
            "score_mode": score_mode,
        }
    )
    return tau_stats, summary


def _permute_within_bins(values: np.ndarray, bins: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    permuted = values.copy()
    for bin_id in sorted(set(int(item) for item in bins)):
        idx = np.flatnonzero(bins == bin_id)
        if idx.size > 1:
            permuted[idx] = permuted[rng.permutation(idx)]
    return permuted


def _null_summary(null_values: np.ndarray, observed: float) -> dict[str, float | int]:
    finite = np.asarray(null_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "n_null": 0,
            "null_mean": float("nan"),
            "null_sd": float("nan"),
            "z": float("nan"),
            "empirical_p": float("nan"),
        }
    mean = float(np.mean(finite))
    sd = float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
    z = float((observed - mean) / sd) if sd > 0 else float("nan")
    empirical_p = float((1 + np.count_nonzero(finite >= observed)) / (finite.size + 1))
    return {
        "n_null": int(finite.size),
        "null_mean": mean,
        "null_sd": sd,
        "z": z,
        "empirical_p": empirical_p,
    }


def summarize_diffusion_nulls(
    *,
    trait: str,
    graph_name: str,
    score_mode: DiffusionScoreMode,
    null_type: DiffusionNullType,
    observed_tau_stats: pd.DataFrame,
    null_distribution: pd.DataFrame,
    seed: int,
    weighted_laplacian_used: bool = False,
    edge_weight_column: str = "",
    z_threshold: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize tau-specific and maxT diffusion null distributions."""

    observed = observed_tau_stats.rename(columns={"T_tau": "T_tau_observed"}).copy()
    tmax = float(observed["T_tau_observed"].max()) if not observed.empty else float("nan")
    tau_at_max = float(observed.loc[observed["T_tau_observed"].idxmax(), "tau"]) if not observed.empty else float("nan")
    max_null = (
        null_distribution.groupby("replicate", observed=True)["T_max"].first().to_numpy(dtype=float)
        if not null_distribution.empty
        else np.array([], dtype=float)
    )
    max_summary = _null_summary(max_null, tmax)
    summary = pd.DataFrame(
        [
            {
                "trait": trait,
                "graph_name": graph_name,
                "score_mode": score_mode,
                "tau_grid": ",".join(str(float(tau)) for tau in observed["tau"]),
                "T_max": tmax,
                "tau_at_max": tau_at_max,
                "null_type": null_type,
                "null_mean": max_summary["null_mean"],
                "null_sd": max_summary["null_sd"],
                "z": max_summary["z"],
                "empirical_p": max_summary["empirical_p"],
                "n_null": max_summary["n_null"],
                "seed": int(seed),
                "passed": bool(float(max_summary["z"]) >= z_threshold)
                if np.isfinite(float(max_summary["z"]))
                else False,
                "weighted_laplacian_used": bool(weighted_laplacian_used),
                "edge_weight_column": edge_weight_column,
            }
        ]
    )

    tau_rows: list[dict[str, object]] = []
    for row in observed.itertuples(index=False):
        tau = float(row.tau)
        observed_tau = float(row.T_tau_observed)
        null_tau = null_distribution.loc[null_distribution["tau"].astype(float) == tau, "T_tau"].to_numpy(dtype=float)
        tau_summary = _null_summary(null_tau, observed_tau)
        tau_rows.append(
            {
                "trait": trait,
                "graph_name": graph_name,
                "score_mode": score_mode,
                "tau": tau,
                "T_tau_observed": observed_tau,
                "null_type": null_type,
                "null_mean_tau": tau_summary["null_mean"],
                "null_sd_tau": tau_summary["null_sd"],
                "z_tau": tau_summary["z"],
                "empirical_p_tau": tau_summary["empirical_p"],
            }
        )
    return summary, pd.DataFrame(tau_rows)


def degree_stratified_diffusion_null(
    graph: nx.Graph,
    scores: pd.DataFrame,
    *,
    trait: str,
    graph_name: str,
    score_mode: DiffusionScoreMode = "positive",
    tau_grid: Iterable[float] = DEFAULT_TAU_GRID,
    n_replicates: int = 1000,
    seed: int = 12345,
    n_bins: int = 20,
    weighted_laplacian: bool = False,
    edge_weight_column: str = "weight",
    score_col: str = "assoc_resid_score",
    batch_size: int = DEFAULT_DIFFUSION_BATCH_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calibrate diffusion by permuting transformed scores within degree strata."""

    if n_replicates < 0:
        raise ValueError("n_replicates must be nonnegative.")
    nodes = tuple(str(node) for node in scores["gene_symbol"].astype(str) if graph.has_node(str(node)))
    lap = graph_laplacian(
        graph,
        nodes=nodes,
        kind="normalized",
        weight=edge_weight_column if weighted_laplacian else None,
    )
    observed_s = aligned_score_vector(scores, lap.nodes, score_col=score_col, mode=score_mode)
    observed_tau = heat_kernel_tau_statistics(lap.laplacian, observed_s, tau_grid=tau_grid)

    degree = np.asarray([graph.degree(node) for node in lap.nodes], dtype=float)
    bins = assign_degree_bins(pd.Series(degree), n_bins=n_bins).to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    taus = parse_tau_grid(tau_grid)
    for start in range(0, n_replicates, batch_size):
        stop = min(start + batch_size, n_replicates)
        null_scores = np.vstack([_permute_within_bins(observed_s, bins, rng) for _ in range(start, stop)])
        tau_values = heat_kernel_tau_statistics_matrix(
            lap.laplacian,
            null_scores,
            tau_grid=taus,
            batch_size=batch_size,
        )
        tmax_values = np.max(tau_values, axis=1) if tau_values.size else np.array([], dtype=float)
        for local_idx, replicate in enumerate(range(start, stop)):
            for tau_idx, tau in enumerate(taus):
                rows.append(
                    {
                        "trait": trait,
                        "graph_name": graph_name,
                        "score_mode": score_mode,
                        "null_type": "degree_stratified",
                        "replicate": int(replicate),
                        "tau": float(tau),
                        "T_tau": float(tau_values[local_idx, tau_idx]),
                        "T_max": float(tmax_values[local_idx]),
                    }
                )
    null_distribution = pd.DataFrame(rows)
    summary, tau_summary = summarize_diffusion_nulls(
        trait=trait,
        graph_name=graph_name,
        score_mode=score_mode,
        null_type="degree_stratified",
        observed_tau_stats=observed_tau,
        null_distribution=null_distribution,
        seed=seed,
        weighted_laplacian_used=weighted_laplacian,
        edge_weight_column=edge_weight_column if weighted_laplacian else "",
    )
    return summary, tau_summary, null_distribution


def degree_preserving_graph_diffusion_null(
    graph: nx.Graph,
    scores: pd.DataFrame,
    *,
    trait: str,
    graph_name: str,
    score_mode: DiffusionScoreMode = "positive",
    tau_grid: Iterable[float] = DEFAULT_TAU_GRID,
    n_replicates: int = 100,
    seed: int = 12345,
    nswap_per_edge: float = 1.0,
    max_tries_per_swap: float = 20.0,
    score_col: str = "assoc_resid_score",
    cache_path: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calibrate diffusion against unweighted degree-preserving rewired graphs."""

    nodes = tuple(str(node) for node in scores["gene_symbol"].astype(str) if graph.has_node(str(node)))
    observed_lap = graph_laplacian(graph, nodes=nodes, kind="normalized", weight=None)
    observed_s = aligned_score_vector(scores, observed_lap.nodes, score_col=score_col, mode=score_mode)
    observed_tau = heat_kernel_tau_statistics(observed_lap.laplacian, observed_s, tau_grid=tau_grid)

    rows: list[dict[str, object]] = []
    for replicate, null_graph in enumerate(
        degree_preserving_graph_replicates(
            graph,
            n_replicates=n_replicates,
            seed=seed,
            nswap_per_edge=nswap_per_edge,
            max_tries_per_swap=max_tries_per_swap,
            cache_path=cache_path,
        )
    ):
        lap = graph_laplacian(null_graph, nodes=observed_lap.nodes, kind="normalized", weight=None)
        tau_stats = heat_kernel_tau_statistics(lap.laplacian, observed_s, tau_grid=tau_grid)
        tmax = float(tau_stats["T_tau"].max()) if not tau_stats.empty else float("nan")
        for tau_row in tau_stats.itertuples(index=False):
            rows.append(
                {
                    "trait": trait,
                    "graph_name": graph_name,
                    "score_mode": score_mode,
                    "null_type": "degree_preserving_graph",
                    "replicate": int(replicate),
                    "tau": float(tau_row.tau),
                    "T_tau": float(tau_row.T_tau),
                    "T_max": tmax,
                }
            )
    null_distribution = pd.DataFrame(rows)
    summary, tau_summary = summarize_diffusion_nulls(
        trait=trait,
        graph_name=graph_name,
        score_mode=score_mode,
        null_type="degree_preserving_graph",
        observed_tau_stats=observed_tau,
        null_distribution=null_distribution,
        seed=seed,
        weighted_laplacian_used=False,
        edge_weight_column="",
    )
    return summary, tau_summary, null_distribution
