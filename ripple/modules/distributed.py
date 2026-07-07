"""Locus-aware distributed module statistics for RIPPLE-D diagnostics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ripple.modules.anchored import (
    AnchoredModuleLibrary,
    bh_fdr,
    empirical_upper,
    sqrt_n_mean_stat,
    z_score,
)


DEFAULT_SCORE_CAP = 3.0
DEFAULT_LOCUS_WINDOW_BP = 500_000
DEFAULT_EFFECTIVE_LOCI_TARGET = 5.0
DEFAULT_TOP1_CONTRIBUTION_MAX = 0.35
DEFAULT_TOP5_CONTRIBUTION_MAX = 0.70
MODERATE_SCORE_LOW = 1.0
MODERATE_SCORE_HIGH = 3.0


@dataclass(frozen=True)
class RippleDConfig:
    """Configuration for RIPPLE-D module diagnostics."""

    score_cap: float = DEFAULT_SCORE_CAP
    locus_window_bp: int = DEFAULT_LOCUS_WINDOW_BP
    effective_loci_target: float = DEFAULT_EFFECTIVE_LOCI_TARGET
    top1_contribution_max: float = DEFAULT_TOP1_CONTRIBUTION_MAX
    top5_contribution_max: float = DEFAULT_TOP5_CONTRIBUTION_MAX
    moderate_score_low: float = MODERATE_SCORE_LOW
    moderate_score_high: float = MODERATE_SCORE_HIGH
    locus_collapse: str = "max"
    degree_bins: int = 10
    property_bins: int = 4


def finite_or_nan(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def cap_upper(values: np.ndarray | pd.Series, cap: float = DEFAULT_SCORE_CAP) -> np.ndarray:
    """Cap only the upper tail of positive association scores."""

    arr = np.asarray(values, dtype=float)
    return np.minimum(arr, float(cap))


def huberize(values: np.ndarray | pd.Series, cap: float = DEFAULT_SCORE_CAP) -> np.ndarray:
    """Symmetric Huberization for sensitivity diagnostics."""

    arr = np.asarray(values, dtype=float)
    cap = float(cap)
    return np.clip(arr, -cap, cap)


def _required_score_columns(scores: pd.DataFrame) -> None:
    required = {"gene_symbol", "assoc_resid_score", "chrom", "gene_start", "gene_end"}
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"scores is missing required columns for RIPPLE-D: {missing}")


def assign_pseudo_loci(scores: pd.DataFrame, *, window_bp: int = DEFAULT_LOCUS_WINDOW_BP) -> pd.DataFrame:
    """Assign deterministic coordinate-based pseudo-loci.

    Genes on the same chromosome are expanded by ``window_bp`` on both sides and
    merged when the expanded intervals overlap. This is an intentionally simple
    substitute for an external LD-block file and is recorded in outputs as a
    pseudo-locus definition.
    """

    _required_score_columns(scores)
    work = scores.copy()
    work["gene_symbol"] = work["gene_symbol"].astype(str).str.upper()
    work["chrom"] = work["chrom"].astype(str)
    work["gene_start"] = pd.to_numeric(work["gene_start"], errors="coerce")
    work["gene_end"] = pd.to_numeric(work["gene_end"], errors="coerce")
    work["locus_id"] = ""
    work["locus_start"] = np.nan
    work["locus_end"] = np.nan

    valid = work["gene_start"].notna() & work["gene_end"].notna() & work["chrom"].notna()
    invalid = work.loc[~valid]
    for idx, row in invalid.iterrows():
        gene = str(row["gene_symbol"]).upper()
        work.loc[idx, "locus_id"] = f"UNMAPPED:{gene}"
        work.loc[idx, "locus_start"] = finite_or_nan(row.get("gene_start"))
        work.loc[idx, "locus_end"] = finite_or_nan(row.get("gene_end"))

    for chrom, group in work.loc[valid].sort_values(["chrom", "gene_start", "gene_end"]).groupby("chrom"):
        locus_counter = 0
        current_start: float | None = None
        current_end: float | None = None
        current_indices: list[int] = []

        def flush() -> None:
            nonlocal locus_counter, current_start, current_end, current_indices
            if not current_indices:
                return
            locus_counter += 1
            locus_id = f"chr{chrom}:L{locus_counter:05d}"
            work.loc[current_indices, "locus_id"] = locus_id
            work.loc[current_indices, "locus_start"] = float(current_start)
            work.loc[current_indices, "locus_end"] = float(current_end)
            current_indices = []
            current_start = None
            current_end = None

        for idx, row in group.iterrows():
            start = max(0.0, float(row["gene_start"]) - float(window_bp))
            end = float(row["gene_end"]) + float(window_bp)
            if current_end is None or start > current_end:
                flush()
                current_start = start
                current_end = end
                current_indices = [int(idx)]
            else:
                current_end = max(float(current_end), end)
                current_indices.append(int(idx))
        flush()

    return work


def add_ripple_d_score_columns(scores: pd.DataFrame, config: RippleDConfig) -> pd.DataFrame:
    work = scores.copy()
    work["assoc_resid_score"] = pd.to_numeric(work["assoc_resid_score"], errors="coerce")
    work["ripple_d_capped_score"] = cap_upper(work["assoc_resid_score"], config.score_cap)
    work["ripple_d_huber_score"] = huberize(work["assoc_resid_score"], config.score_cap)
    descending = work["assoc_resid_score"].rank(method="average", ascending=False)
    work["score_rank_fraction"] = descending / len(work) if len(work) else np.nan
    work["score_rank_percentile"] = 1.0 - work["score_rank_fraction"]
    return work


def _collapse_locus_scores(selected: pd.DataFrame, *, score_col: str, collapse: str = "max") -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame(columns=["locus_id", "locus_score", "n_genes_in_locus"])
    if collapse == "max":
        agg = selected.groupby("locus_id", observed=True).agg(
            locus_score=(score_col, "max"),
            n_genes_in_locus=("gene_symbol", "nunique"),
        )
    elif collapse == "huber_mean":
        agg = selected.groupby("locus_id", observed=True).agg(
            locus_score=(score_col, "mean"),
            n_genes_in_locus=("gene_symbol", "nunique"),
        )
    else:
        raise ValueError(f"Unknown locus collapse method: {collapse}")
    return agg.reset_index()


def locus_robust_stat(locus_scores: np.ndarray | pd.Series) -> float:
    values = np.asarray(locus_scores, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.sum(values) / np.sqrt(values.size))


def contribution_metrics(locus_scores: np.ndarray | pd.Series) -> dict[str, float]:
    values = np.asarray(locus_scores, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "positive_locus_signal_sum": 0.0,
            "n_effective_loci": 0.0,
            "top1_locus_contribution": 1.0,
            "top5_locus_contribution": 1.0,
        }
    positive = np.clip(values, 0.0, None)
    total = float(np.sum(positive))
    if total <= 0:
        return {
            "positive_locus_signal_sum": 0.0,
            "n_effective_loci": 0.0,
            "top1_locus_contribution": 1.0,
            "top5_locus_contribution": 1.0,
        }
    proportions = positive / total
    ordered = np.sort(proportions)[::-1]
    return {
        "positive_locus_signal_sum": total,
        "n_effective_loci": float(1.0 / np.sum(proportions**2)),
        "top1_locus_contribution": float(ordered[0]),
        "top5_locus_contribution": float(np.sum(ordered[:5])),
    }


def ripple_d_stat(locus_scores: np.ndarray | pd.Series, config: RippleDConfig) -> float:
    base = locus_robust_stat(locus_scores)
    if not np.isfinite(base):
        return float("nan")
    metrics = contribution_metrics(locus_scores)
    n_eff = metrics["n_effective_loci"]
    top1 = metrics["top1_locus_contribution"]
    eff_penalty = min(1.0, n_eff / float(config.effective_loci_target))
    top_penalty = min(1.0, float(config.top1_contribution_max) / top1) if top1 > 0 else 1.0
    return float(base * eff_penalty * top_penalty)


def leave_top_locus_stat(locus_scores: pd.Series, *, k: int) -> float:
    values = pd.to_numeric(locus_scores, errors="coerce").dropna().sort_values(ascending=False)
    if values.shape[0] <= int(k):
        return float("nan")
    return locus_robust_stat(values.iloc[int(k) :].to_numpy(dtype=float))


def leave_top_gene_stat(values: np.ndarray | pd.Series, *, k: int) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= int(k):
        return float("nan")
    return sqrt_n_mean_stat(np.sort(arr)[::-1][int(k) :])


def moderate_locus_burden(
    selected: pd.DataFrame,
    config: RippleDConfig,
    *,
    score_col: str = "assoc_resid_score",
) -> int:
    if selected.empty:
        return 0
    scores = pd.to_numeric(selected[score_col], errors="coerce")
    moderate = selected.loc[
        scores.gt(config.moderate_score_low) & scores.lt(config.moderate_score_high),
        ["locus_id"],
    ]
    return int(moderate["locus_id"].nunique())


def _quantile_bins(values: pd.Series, n_bins: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    if numeric.nunique(dropna=True) <= 1:
        return pd.Series(np.zeros(len(numeric), dtype=int), index=values.index)
    ranks = numeric.rank(method="first")
    return pd.qcut(ranks, q=min(int(n_bins), len(numeric)), labels=False, duplicates="drop").astype(int)


def build_locus_background(
    scores: pd.DataFrame,
    library: AnchoredModuleLibrary | None,
    config: RippleDConfig,
) -> pd.DataFrame:
    """Collapse the analysis background to one row per pseudo-locus."""

    work = scores.copy()
    annotation_counts = pd.Series(0, index=work["gene_symbol"].astype(str).str.upper(), dtype=int)
    if library is not None:
        counts: dict[str, int] = {}
        for genes in library.gene_sets.values():
            for gene in genes:
                key = str(gene).upper()
                counts[key] = counts.get(key, 0) + 1
        annotation_counts = work["gene_symbol"].astype(str).str.upper().map(counts).fillna(0).astype(int)
    work["annotation_count"] = annotation_counts.to_numpy(dtype=int)
    for column in ["graph_degree", "gene_length", "n_mapped_snps", "local_ld_score"]:
        if column not in work.columns:
            work[column] = 0.0
        work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0.0)

    locus_scores = _collapse_locus_scores(
        work,
        score_col="ripple_d_capped_score",
        collapse=config.locus_collapse,
    ).rename(columns={"locus_score": "genome_locus_score"})
    agg = work.groupby("locus_id", observed=True).agg(
        chrom=("chrom", "first"),
        locus_start=("locus_start", "min"),
        locus_end=("locus_end", "max"),
        n_locus_genes=("gene_symbol", "nunique"),
        mean_graph_degree=("graph_degree", "mean"),
        max_graph_degree=("graph_degree", "max"),
        mean_gene_length=("gene_length", "mean"),
        mean_mapped_snp_count=("n_mapped_snps", "mean"),
        mean_local_ld_score=("local_ld_score", "mean"),
        annotation_density=("annotation_count", "mean"),
        moderate_locus=("assoc_resid_score", lambda x: int(((x > 1.0) & (x < 3.0)).any())),
    )
    out = agg.reset_index().merge(locus_scores[["locus_id", "genome_locus_score"]], on="locus_id", how="left")
    out["score_rank"] = out["genome_locus_score"].rank(method="first", ascending=False)
    out["score_rank_fraction"] = out["score_rank"] / len(out)
    out["degree_bin"] = _quantile_bins(out["mean_graph_degree"], config.degree_bins)
    out["gene_count_bin"] = _quantile_bins(out["n_locus_genes"], config.property_bins)
    out["annotation_bin"] = _quantile_bins(out["annotation_density"], config.property_bins)
    out["ld_bin"] = _quantile_bins(out["mean_local_ld_score"], config.property_bins)
    out["snp_count_bin"] = _quantile_bins(out["mean_mapped_snp_count"], config.property_bins)
    out["match_bin"] = (
        out["degree_bin"].astype(str)
        + "|"
        + out["gene_count_bin"].astype(str)
        + "|"
        + out["annotation_bin"].astype(str)
        + "|"
        + out["ld_bin"].astype(str)
        + "|"
        + out["snp_count_bin"].astype(str)
    )
    return out


def _sample_from_bin(
    locus_background: pd.DataFrame,
    observed_row: pd.Series,
    rng: np.random.Generator,
    *,
    exclude: set[str] | None = None,
    match_pools: Mapping[str, np.ndarray] | None = None,
    degree_pools: Mapping[int, np.ndarray] | None = None,
    all_loci: np.ndarray | None = None,
) -> str:
    exclude = exclude or set()
    if match_pools is not None:
        candidates = np.asarray(match_pools.get(str(observed_row["match_bin"]), []), dtype=object)
    else:
        candidates = locus_background.loc[locus_background["match_bin"].eq(observed_row["match_bin"]), "locus_id"].to_numpy(
            dtype=object
        )
    if candidates.size:
        candidates = np.asarray([candidate for candidate in candidates if str(candidate) not in exclude], dtype=object)
    if candidates.size == 0:
        if degree_pools is not None:
            candidates = np.asarray(degree_pools.get(int(observed_row["degree_bin"]), []), dtype=object)
        else:
            candidates = locus_background.loc[
                locus_background["degree_bin"].eq(observed_row["degree_bin"]), "locus_id"
            ].to_numpy(dtype=object)
        candidates = np.asarray([candidate for candidate in candidates if str(candidate) not in exclude], dtype=object)
    if candidates.size == 0:
        candidates = (
            np.asarray(all_loci, dtype=object)
            if all_loci is not None
            else locus_background["locus_id"].to_numpy(dtype=object)
        )
        candidates = np.asarray([candidate for candidate in candidates if str(candidate) not in exclude], dtype=object)
    if candidates.size == 0:
        candidates = locus_background["locus_id"].to_numpy(dtype=object)
    return str(rng.choice(candidates))


def _sample_locus_matched_set(
    locus_background: pd.DataFrame,
    observed_loci: Sequence[str],
    rng: np.random.Generator,
    *,
    match_pools: Mapping[str, np.ndarray] | None = None,
    degree_pools: Mapping[int, np.ndarray] | None = None,
    all_loci: np.ndarray | None = None,
) -> list[str]:
    by_locus = locus_background.set_index("locus_id", drop=False)
    sampled: list[str] = []
    observed_set = {str(locus_id) for locus_id in observed_loci}
    used: set[str] = set(observed_set)
    for locus_id in observed_loci:
        if locus_id not in by_locus.index:
            continue
        picked = _sample_from_bin(
            locus_background,
            by_locus.loc[locus_id],
            rng,
            exclude=used,
            match_pools=match_pools,
            degree_pools=degree_pools,
            all_loci=all_loci,
        )
        sampled.append(picked)
        used.add(picked)
    return sampled


def _module_locus_scores(
    scores: pd.DataFrame,
    genes: Sequence[str],
    config: RippleDConfig,
    *,
    score_col: str = "ripple_d_capped_score",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    query = {str(gene).upper() for gene in genes}
    selected = scores.loc[scores["gene_symbol"].astype(str).str.upper().isin(query)].copy()
    locus_table = _collapse_locus_scores(selected, score_col=score_col, collapse=config.locus_collapse)
    return selected, locus_table


def summarize_module_distribution(
    scores: pd.DataFrame,
    module_genes: Sequence[str],
    config: RippleDConfig,
) -> dict[str, object]:
    selected, locus_table = _module_locus_scores(scores, module_genes, config)
    raw_values = selected["assoc_resid_score"].to_numpy(dtype=float)
    capped_values = selected["ripple_d_capped_score"].to_numpy(dtype=float)
    locus_scores = locus_table["locus_score"].to_numpy(dtype=float)
    contribution = contribution_metrics(locus_scores)
    ripple_d_value = ripple_d_stat(locus_scores, config)
    rank_stat = (
        float(selected["score_rank_fraction"].mean()) if "score_rank_fraction" in selected and not selected.empty else float("nan")
    )
    top_gene = selected.sort_values("assoc_resid_score", ascending=False).head(1)
    top_locus = locus_table.sort_values("locus_score", ascending=False).head(1)
    return {
        "n_present": int(selected["gene_symbol"].nunique()),
        "n_loci": int(locus_table["locus_id"].nunique()),
        "raw_gene_stat": sqrt_n_mean_stat(raw_values),
        "capped_gene_stat": sqrt_n_mean_stat(capped_values),
        "locus_robust_stat": locus_robust_stat(locus_scores),
        "ripple_d_stat": ripple_d_value,
        "moderate_locus_burden": moderate_locus_burden(selected, config),
        "rank_enrichment_stat": rank_stat,
        "leave_top1_gene_stat": leave_top_gene_stat(capped_values, k=1),
        "leave_top5_gene_stat": leave_top_gene_stat(capped_values, k=5),
        "leave_top10_gene_stat": leave_top_gene_stat(capped_values, k=10),
        "leave_top1_locus_stat": leave_top_locus_stat(locus_table.set_index("locus_id")["locus_score"], k=1),
        "leave_top3_locus_stat": leave_top_locus_stat(locus_table.set_index("locus_id")["locus_score"], k=3),
        "leave_top5_locus_stat": leave_top_locus_stat(locus_table.set_index("locus_id")["locus_score"], k=5),
        "top1_gene": str(top_gene.iloc[0]["gene_symbol"]) if not top_gene.empty else "",
        "top1_gene_score": finite_or_nan(top_gene.iloc[0]["assoc_resid_score"]) if not top_gene.empty else float("nan"),
        "top1_locus": str(top_locus.iloc[0]["locus_id"]) if not top_locus.empty else "",
        "top1_locus_score": finite_or_nan(top_locus.iloc[0]["locus_score"]) if not top_locus.empty else float("nan"),
        **contribution,
    }


def _null_stats_from_loci(locus_background: pd.DataFrame, sampled_loci: Sequence[str], config: RippleDConfig) -> dict[str, float]:
    rows = locus_background.loc[locus_background["locus_id"].isin(set(sampled_loci))]
    scores = rows["genome_locus_score"].to_numpy(dtype=float)
    return {
        "locus_robust_stat": locus_robust_stat(scores),
        "ripple_d_stat": ripple_d_stat(scores, config),
        "moderate_locus_burden": float(rows["moderate_locus"].sum()),
        "leave_top1_locus_stat": leave_top_locus_stat(pd.Series(scores), k=1),
    }


def _locus_matched_nulls(
    locus_background: pd.DataFrame,
    observed_loci: Sequence[str],
    config: RippleDConfig,
    *,
    n_null: int,
    rng: np.random.Generator,
    match_pools: Mapping[str, np.ndarray] | None = None,
    degree_pools: Mapping[int, np.ndarray] | None = None,
    all_loci: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    out = {
        "locus_robust_stat": np.empty(n_null, dtype=float),
        "ripple_d_stat": np.empty(n_null, dtype=float),
        "moderate_locus_burden": np.empty(n_null, dtype=float),
        "leave_top1_locus_stat": np.empty(n_null, dtype=float),
    }
    for idx in range(n_null):
        sampled = _sample_locus_matched_set(
            locus_background,
            observed_loci,
            rng,
            match_pools=match_pools,
            degree_pools=degree_pools,
            all_loci=all_loci,
        )
        stats = _null_stats_from_loci(locus_background, sampled, config)
        for name, value in stats.items():
            out[name][idx] = value
    return out


def _top_locus_conditioned_nulls(
    locus_background: pd.DataFrame,
    observed_loci: Sequence[str],
    config: RippleDConfig,
    *,
    n_null: int,
    rng: np.random.Generator,
    match_pools: Mapping[str, np.ndarray] | None = None,
    degree_pools: Mapping[int, np.ndarray] | None = None,
    all_loci: np.ndarray | None = None,
    top_fraction: float = 0.01,
) -> np.ndarray:
    by_locus = locus_background.set_index("locus_id", drop=False)
    observed = by_locus.loc[[locus for locus in observed_loci if locus in by_locus.index]].copy()
    if observed.empty:
        return np.full(n_null, np.nan, dtype=float)
    top_cutoff = max(1, int(math.ceil(len(locus_background) * top_fraction)))
    observed_top = observed.loc[observed["score_rank"].le(top_cutoff)]
    if observed_top.empty:
        return np.full(n_null, np.nan, dtype=float)
    top_pool = locus_background.loc[locus_background["score_rank"].le(top_cutoff), "locus_id"].astype(str)
    remaining_observed = [str(locus) for locus in observed_loci if str(locus) not in set(observed_top["locus_id"].astype(str))]
    out = np.empty(n_null, dtype=float)
    for idx in range(n_null):
        forced = str(rng.choice(top_pool.to_numpy(dtype=object)))
        sampled = [forced]
        sampled.extend(
            _sample_locus_matched_set(
                locus_background,
                remaining_observed,
                rng,
                match_pools=match_pools,
                degree_pools=degree_pools,
                all_loci=all_loci,
            )
        )
        stats = _null_stats_from_loci(locus_background, sampled, config)
        out[idx] = stats["leave_top1_locus_stat"]
    return out


def _locus_score_permutation_nulls(
    locus_background: pd.DataFrame,
    observed_loci: Sequence[str],
    config: RippleDConfig,
    *,
    n_null: int,
    rng: np.random.Generator,
    match_score_pools: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    by_locus = locus_background.set_index("locus_id", drop=False)
    observed = by_locus.loc[[locus for locus in observed_loci if locus in by_locus.index]].copy()
    out = np.empty(n_null, dtype=float)
    for idx in range(n_null):
        sampled_scores: list[float] = []
        for row in observed.itertuples(index=False):
            pool = (
                np.asarray(match_score_pools.get(str(getattr(row, "match_bin")), []), dtype=float)
                if match_score_pools is not None
                else locus_background.loc[
                    locus_background["match_bin"].eq(getattr(row, "match_bin")),
                    "genome_locus_score",
                ].to_numpy(dtype=float)
            )
            if pool.size == 0:
                pool = locus_background["genome_locus_score"].to_numpy(dtype=float)
            sampled_scores.append(float(rng.choice(pool)))
        out[idx] = ripple_d_stat(np.asarray(sampled_scores, dtype=float), config)
    return out


def _filter_pool(pool: np.ndarray, exclude: set[int]) -> np.ndarray:
    if not exclude:
        return pool
    return np.asarray([int(idx) for idx in pool if int(idx) not in exclude], dtype=int)


def _sample_index_from_pools(
    observed_idx: int,
    *,
    match_bin_arr: np.ndarray,
    degree_bin_arr: np.ndarray,
    match_pools_idx: Mapping[str, np.ndarray],
    degree_pools_idx: Mapping[int, np.ndarray],
    all_indices: np.ndarray,
    exclude: set[int],
    rng: np.random.Generator,
) -> int:
    match_bin = str(match_bin_arr[int(observed_idx)])
    candidates = _filter_pool(match_pools_idx.get(match_bin, np.array([], dtype=int)), exclude)
    if candidates.size == 0:
        degree_bin = int(degree_bin_arr[int(observed_idx)])
        candidates = _filter_pool(degree_pools_idx.get(degree_bin, np.array([], dtype=int)), exclude)
    if candidates.size == 0:
        candidates = _filter_pool(all_indices, exclude)
    if candidates.size == 0:
        candidates = all_indices
    return int(rng.choice(candidates))


def _sample_locus_matched_indices_fast(
    observed_indices: Sequence[int],
    *,
    match_bin_arr: np.ndarray,
    degree_bin_arr: np.ndarray,
    match_pools_idx: Mapping[str, np.ndarray],
    degree_pools_idx: Mapping[int, np.ndarray],
    all_indices: np.ndarray,
    rng: np.random.Generator,
) -> list[int]:
    observed_set = {int(idx) for idx in observed_indices}
    used = set(observed_set)
    sampled: list[int] = []
    for observed_idx in observed_indices:
        picked = _sample_index_from_pools(
            int(observed_idx),
            match_bin_arr=match_bin_arr,
            degree_bin_arr=degree_bin_arr,
            match_pools_idx=match_pools_idx,
            degree_pools_idx=degree_pools_idx,
            all_indices=all_indices,
            exclude=used,
            rng=rng,
        )
        sampled.append(picked)
        used.add(picked)
    return sampled


def _stats_from_index_scores(scores: np.ndarray, moderate_flags: np.ndarray, indices: Sequence[int], config: RippleDConfig) -> dict[str, float]:
    idx = np.asarray(indices, dtype=int)
    values = scores[idx]
    return {
        "locus_robust_stat": locus_robust_stat(values),
        "ripple_d_stat": ripple_d_stat(values, config),
        "moderate_locus_burden": float(np.sum(moderate_flags[idx])),
        "leave_top1_locus_stat": leave_top_locus_stat(pd.Series(values), k=1),
    }


def _locus_matched_nulls_fast(
    observed_indices: Sequence[int],
    config: RippleDConfig,
    *,
    scores: np.ndarray,
    moderate_flags: np.ndarray,
    match_bin_arr: np.ndarray,
    degree_bin_arr: np.ndarray,
    match_pools_idx: Mapping[str, np.ndarray],
    degree_pools_idx: Mapping[int, np.ndarray],
    all_indices: np.ndarray,
    n_null: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    out = {
        "locus_robust_stat": np.empty(n_null, dtype=float),
        "ripple_d_stat": np.empty(n_null, dtype=float),
        "moderate_locus_burden": np.empty(n_null, dtype=float),
        "leave_top1_locus_stat": np.empty(n_null, dtype=float),
    }
    for idx in range(n_null):
        sampled = _sample_locus_matched_indices_fast(
            observed_indices,
            match_bin_arr=match_bin_arr,
            degree_bin_arr=degree_bin_arr,
            match_pools_idx=match_pools_idx,
            degree_pools_idx=degree_pools_idx,
            all_indices=all_indices,
            rng=rng,
        )
        stats = _stats_from_index_scores(scores, moderate_flags, sampled, config)
        for name, value in stats.items():
            out[name][idx] = value
    return out


def _top_locus_conditioned_nulls_fast(
    observed_indices: Sequence[int],
    config: RippleDConfig,
    *,
    scores: np.ndarray,
    moderate_flags: np.ndarray,
    score_rank_arr: np.ndarray,
    match_bin_arr: np.ndarray,
    degree_bin_arr: np.ndarray,
    match_pools_idx: Mapping[str, np.ndarray],
    degree_pools_idx: Mapping[int, np.ndarray],
    all_indices: np.ndarray,
    n_null: int,
    rng: np.random.Generator,
    top_fraction: float = 0.01,
) -> np.ndarray:
    if len(observed_indices) == 0:
        return np.full(n_null, np.nan, dtype=float)
    top_cutoff = max(1, int(math.ceil(len(score_rank_arr) * top_fraction)))
    top_indices = all_indices[score_rank_arr <= top_cutoff]
    observed_top = [int(idx) for idx in observed_indices if score_rank_arr[int(idx)] <= top_cutoff]
    if len(observed_top) == 0 or top_indices.size == 0:
        return np.full(n_null, np.nan, dtype=float)
    observed_top_set = set(observed_top)
    remaining = [int(idx) for idx in observed_indices if int(idx) not in observed_top_set]
    out = np.empty(n_null, dtype=float)
    for idx in range(n_null):
        forced = int(rng.choice(top_indices))
        sampled = [forced]
        sampled.extend(
            _sample_locus_matched_indices_fast(
                remaining,
                match_bin_arr=match_bin_arr,
                degree_bin_arr=degree_bin_arr,
                match_pools_idx=match_pools_idx,
                degree_pools_idx=degree_pools_idx,
                all_indices=all_indices,
                rng=rng,
            )
        )
        out[idx] = _stats_from_index_scores(scores, moderate_flags, sampled, config)["leave_top1_locus_stat"]
    return out


def _locus_score_permutation_nulls_fast(
    observed_indices: Sequence[int],
    config: RippleDConfig,
    *,
    match_bin_arr: np.ndarray,
    match_score_pools: Mapping[str, np.ndarray],
    all_scores: np.ndarray,
    n_null: int,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.empty(n_null, dtype=float)
    for idx in range(n_null):
        sampled_scores: list[float] = []
        for observed_idx in observed_indices:
            pool = match_score_pools.get(str(match_bin_arr[int(observed_idx)]), np.array([], dtype=float))
            if pool.size == 0:
                pool = all_scores
            sampled_scores.append(float(rng.choice(pool)))
        out[idx] = ripple_d_stat(np.asarray(sampled_scores, dtype=float), config)
    return out


def classify_distributed_module(row: Mapping[str, object], config: RippleDConfig) -> str:
    locus_p = finite_or_nan(row.get("locus_robust_empirical_p"))
    moderate_p = finite_or_nan(row.get("moderate_locus_burden_empirical_p"))
    leave_p = finite_or_nan(row.get("leave_top1_locus_empirical_p"))
    raw_p = finite_or_nan(row.get("raw_gene_empirical_p"))
    n_eff = finite_or_nan(row.get("n_effective_loci"))
    top1 = finite_or_nan(row.get("top1_locus_contribution"))
    top5 = finite_or_nan(row.get("top5_locus_contribution"))

    passes = (
        locus_p < 0.05
        and moderate_p < 0.10
        and leave_p < 0.10
        and n_eff >= config.effective_loci_target
        and top1 <= config.top1_contribution_max
        and top5 <= config.top5_contribution_max
    )
    if passes:
        return "distributed_weak_signal_module_candidate"
    if raw_p < 0.05 and (top1 > config.top1_contribution_max or n_eff < config.effective_loci_target):
        return "top_locus_dominant_module"
    if raw_p < 0.05 and locus_p < 0.10:
        return "sparse_locus_pathway_overlap"
    if raw_p < 0.05:
        return "raw_gene_set_enrichment_only"
    return "negative"


def ripple_d_module_tests(
    scores: pd.DataFrame,
    library: AnchoredModuleLibrary,
    *,
    config: RippleDConfig | None = None,
    min_present: int = 5,
    n_null: int = 200,
    seed: int = 20260713,
    return_null_details: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Run fixed-library RIPPLE-D diagnostics."""

    config = config or RippleDConfig()
    work = assign_pseudo_loci(scores, window_bp=config.locus_window_bp)
    work = add_ripple_d_score_columns(work, config)
    locus_background = build_locus_background(work, library, config)
    gene_to_row = {str(row.gene_symbol).upper(): row for row in work.itertuples(index=False)}
    rng = np.random.default_rng(seed)
    all_loci = locus_background["locus_id"].to_numpy(dtype=object)
    all_locus_indices = np.arange(len(locus_background), dtype=int)
    locus_id_to_index = {str(locus_id): int(idx) for idx, locus_id in enumerate(all_loci)}
    genome_locus_scores = locus_background["genome_locus_score"].to_numpy(dtype=float)
    moderate_flags = locus_background["moderate_locus"].to_numpy(dtype=float)
    match_bin_arr = locus_background["match_bin"].astype(str).to_numpy(dtype=object)
    degree_bin_arr = locus_background["degree_bin"].to_numpy(dtype=int)
    score_rank_arr = locus_background["score_rank"].to_numpy(dtype=float)
    match_pools = {
        str(match_bin): group["locus_id"].to_numpy(dtype=object)
        for match_bin, group in locus_background.groupby("match_bin", observed=True)
    }
    degree_pools = {
        int(degree_bin): group["locus_id"].to_numpy(dtype=object)
        for degree_bin, group in locus_background.groupby("degree_bin", observed=True)
    }
    match_score_pools = {
        str(match_bin): group["genome_locus_score"].to_numpy(dtype=float)
        for match_bin, group in locus_background.groupby("match_bin", observed=True)
    }
    match_pools_idx = {
        str(match_bin): np.asarray([locus_id_to_index[str(locus)] for locus in pool], dtype=int)
        for match_bin, pool in match_pools.items()
    }
    degree_pools_idx = {
        int(degree_bin): np.asarray([locus_id_to_index[str(locus)] for locus in pool], dtype=int)
        for degree_bin, pool in degree_pools.items()
    }

    rows: list[dict[str, object]] = []
    locus_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    for module_idx, (module_name, genes) in enumerate(library.gene_sets.items(), start=1):
        module_id = f"D{module_idx:04d}"
        query = {str(gene).upper() for gene in genes}
        present = sorted(gene for gene in query if gene in gene_to_row)
        dropped = sorted(query - set(present))
        base = {
            "module_id": module_id,
            "module_name": str(module_name),
            "module_source": library.module_source.get(module_name, "unspecified"),
            "annotation_source_type": library.annotation_source_type.get(module_name, "internal_support"),
            "module_category": library.module_category.get(module_name, "unspecified"),
            "n_query_genes": int(len(query)),
            "n_present": int(len(present)),
            "n_missing": int(len(dropped)),
            "present_genes": ",".join(present),
            "dropped_genes": ",".join(dropped),
            "statistic_direction": "greater_is_more_extreme",
            "candidate_score_basis": "fixed_anchored_library;locus_aware_ripple_d",
            "raw_component_name": "raw_enrichment_component",
            "locus_definition": f"gene_coordinate_pseudo_locus_pm_{config.locus_window_bp}bp",
        }
        if len(present) < min_present:
            rows.append({**base, "module_status": "not_tested_low_overlap", "module_label": "not_tested"})
            continue

        observed = summarize_module_distribution(work, present, config)
        selected, locus_table = _module_locus_scores(work, present, config)
        observed_loci = locus_table["locus_id"].astype(str).tolist()
        observed_locus_indices = [locus_id_to_index[locus] for locus in observed_loci if locus in locus_id_to_index]
        raw_gene_null = np.empty(n_null, dtype=float)
        locus_nulls = _locus_matched_nulls_fast(
            observed_locus_indices,
            config,
            scores=genome_locus_scores,
            moderate_flags=moderate_flags,
            match_bin_arr=match_bin_arr,
            degree_bin_arr=degree_bin_arr,
            match_pools_idx=match_pools_idx,
            degree_pools_idx=degree_pools_idx,
            all_indices=all_locus_indices,
            n_null=n_null,
            rng=rng,
        )
        top_conditioned_null = _top_locus_conditioned_nulls_fast(
            observed_locus_indices,
            config,
            scores=genome_locus_scores,
            moderate_flags=moderate_flags,
            score_rank_arr=score_rank_arr,
            match_bin_arr=match_bin_arr,
            degree_bin_arr=degree_bin_arr,
            match_pools_idx=match_pools_idx,
            degree_pools_idx=degree_pools_idx,
            all_indices=all_locus_indices,
            n_null=n_null,
            rng=rng,
        )
        permutation_null = _locus_score_permutation_nulls_fast(
            observed_locus_indices,
            config,
            match_bin_arr=match_bin_arr,
            match_score_pools=match_score_pools,
            all_scores=genome_locus_scores,
            n_null=n_null,
            rng=rng,
        )
        # Raw gene burden null is intentionally retained as a legacy contrast.
        all_scores = work["assoc_resid_score"].to_numpy(dtype=float)
        for idx in range(n_null):
            sampled = rng.choice(all_scores, size=len(present), replace=len(present) > len(all_scores))
            raw_gene_null[idx] = sqrt_n_mean_stat(sampled)

        row = {
            **base,
            **observed,
            "raw_gene_empirical_p": empirical_upper(raw_gene_null, observed["raw_gene_stat"]),
            "locus_robust_null_mean": float(np.nanmean(locus_nulls["locus_robust_stat"])),
            "locus_robust_null_sd": float(np.nanstd(locus_nulls["locus_robust_stat"], ddof=1)),
            "locus_robust_z": z_score(observed["locus_robust_stat"], locus_nulls["locus_robust_stat"]),
            "locus_robust_empirical_p": empirical_upper(
                locus_nulls["locus_robust_stat"],
                observed["locus_robust_stat"],
            ),
            "ripple_d_null_mean": float(np.nanmean(locus_nulls["ripple_d_stat"])),
            "ripple_d_null_sd": float(np.nanstd(locus_nulls["ripple_d_stat"], ddof=1)),
            "ripple_d_z": z_score(observed["ripple_d_stat"], locus_nulls["ripple_d_stat"]),
            "ripple_d_empirical_p": empirical_upper(locus_nulls["ripple_d_stat"], observed["ripple_d_stat"]),
            "moderate_locus_burden_empirical_p": empirical_upper(
                locus_nulls["moderate_locus_burden"],
                observed["moderate_locus_burden"],
            ),
            "leave_top1_locus_empirical_p": empirical_upper(
                locus_nulls["leave_top1_locus_stat"],
                observed["leave_top1_locus_stat"],
            ),
            "top_locus_conditioned_leave_top1_p": empirical_upper(
                top_conditioned_null,
                observed["leave_top1_locus_stat"],
            ),
            "locus_score_permutation_ripple_d_p": empirical_upper(permutation_null, observed["ripple_d_stat"]),
            "n_locus_matched_null": int(n_null),
            "n_top_locus_conditioned_null": int(np.isfinite(top_conditioned_null).sum()),
            "n_locus_score_permutation_null": int(n_null),
            "module_status": "negative",
            "module_label": "negative",
        }
        label = classify_distributed_module(row, config)
        row["module_status"] = label
        row["module_label"] = label
        rows.append(row)

        locus_detail = selected.merge(locus_table, on="locus_id", how="left", suffixes=("", "_collapsed"))
        for detail in locus_detail.itertuples(index=False):
            locus_rows.append(
                {
                    "module_id": module_id,
                    "module_name": str(module_name),
                    "gene_symbol": str(detail.gene_symbol),
                    "locus_id": str(detail.locus_id),
                    "assoc_resid_score": finite_or_nan(detail.assoc_resid_score),
                    "ripple_d_capped_score": finite_or_nan(detail.ripple_d_capped_score),
                    "locus_score": finite_or_nan(detail.locus_score),
                }
            )
        if return_null_details:
            for null_type, stat_name, values in [
                ("raw_gene_random_null", "raw_gene_stat", raw_gene_null),
                ("locus_matched_competitive_null", "locus_robust_stat", locus_nulls["locus_robust_stat"]),
                ("locus_matched_competitive_null", "ripple_d_stat", locus_nulls["ripple_d_stat"]),
                (
                    "locus_matched_competitive_null",
                    "moderate_locus_burden",
                    locus_nulls["moderate_locus_burden"],
                ),
                (
                    "locus_matched_competitive_null",
                    "leave_top1_locus_stat",
                    locus_nulls["leave_top1_locus_stat"],
                ),
                ("top_locus_conditioned_null", "leave_top1_locus_stat", top_conditioned_null),
                ("locus_score_permutation_null", "ripple_d_stat", permutation_null),
            ]:
                for replicate, value in enumerate(values):
                    if np.isfinite(value):
                        null_rows.append(
                            {
                                "module_id": module_id,
                                "module_name": str(module_name),
                                "replicate": int(replicate),
                                "null_type": null_type,
                                "statistic_name": stat_name,
                                "statistic_direction": "greater_is_more_extreme",
                                "statistic_value": float(value),
                            }
                        )

    modules = pd.DataFrame(rows)
    if not modules.empty and "locus_robust_empirical_p" in modules:
        modules["locus_robust_fdr"] = bh_fdr(modules["locus_robust_empirical_p"].to_numpy(dtype=float))
        modules["ripple_d_fdr"] = bh_fdr(modules["ripple_d_empirical_p"].to_numpy(dtype=float))
        modules = modules.sort_values(
            ["module_status", "locus_robust_empirical_p", "ripple_d_empirical_p", "ripple_d_stat"],
            ascending=[True, True, True, False],
        ).reset_index(drop=True)
        modules["ripple_d_module_rank"] = np.arange(1, len(modules) + 1)

    summary = {
        "n_input_modules": int(len(library.gene_sets)),
        "n_tested_modules": int(modules["module_status"].ne("not_tested_low_overlap").sum())
        if "module_status" in modules
        else 0,
        "n_distributed_weak_signal_module_candidate": int(
            modules["module_status"].eq("distributed_weak_signal_module_candidate").sum()
        )
        if "module_status" in modules
        else 0,
        "n_top_locus_dominant_module": int(modules["module_status"].eq("top_locus_dominant_module").sum())
        if "module_status" in modules
        else 0,
        "n_raw_gene_set_enrichment_only": int(modules["module_status"].eq("raw_gene_set_enrichment_only").sum())
        if "module_status" in modules
        else 0,
        "n_background_genes": int(work["gene_symbol"].nunique()),
        "n_background_loci": int(locus_background["locus_id"].nunique()),
        "n_null": int(n_null),
        "score_cap": float(config.score_cap),
        "locus_window_bp": int(config.locus_window_bp),
        "locus_definition": f"gene_coordinate_pseudo_locus_pm_{config.locus_window_bp}bp",
        "null_details_returned": bool(return_null_details),
    }
    return modules, pd.DataFrame(locus_rows), pd.DataFrame(null_rows), summary
