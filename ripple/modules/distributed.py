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
DEFAULT_RIPPLE_D_P_MAX = 0.05
DEFAULT_POSITIVE_LOCUS_P_MAX = 0.05
DEFAULT_RANK_LOCUS_P_MAX = 0.05


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
    null_gene_subset_sampling: bool = True
    ripple_d_p_max: float = DEFAULT_RIPPLE_D_P_MAX
    positive_locus_p_max: float = DEFAULT_POSITIVE_LOCUS_P_MAX
    rank_locus_p_max: float = DEFAULT_RANK_LOCUS_P_MAX
    annotation_matching_enabled: bool = True
    locus_id_column: str | None = None
    locus_definition_name: str | None = None
    require_gene_count_match: bool = True
    manuscript_mode: bool = True


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


def assign_predefined_loci(scores: pd.DataFrame, *, locus_id_column: str) -> pd.DataFrame:
    """Assign loci from an externally supplied locus column.

    This provides the V1.4c interface for EUR LD-block or clumped-GWAS locus
    sensitivity without changing the module statistic itself.
    """

    _required_score_columns(scores)
    if locus_id_column not in scores.columns:
        raise ValueError(f"Requested locus_id_column {locus_id_column!r} is not present in scores")
    work = scores.copy()
    work["gene_symbol"] = work["gene_symbol"].astype(str).str.upper()
    work["chrom"] = work["chrom"].astype(str)
    work["gene_start"] = pd.to_numeric(work["gene_start"], errors="coerce")
    work["gene_end"] = pd.to_numeric(work["gene_end"], errors="coerce")
    provided = work[locus_id_column].astype(str).replace({"": np.nan, "nan": np.nan, "None": np.nan})
    fallback = work["gene_symbol"].astype(str).map(lambda gene: f"UNMAPPED:{gene}")
    work["locus_id"] = provided.fillna(fallback)
    grouped = work.groupby("locus_id", observed=True)
    starts = grouped["gene_start"].transform("min")
    ends = grouped["gene_end"].transform("max")
    work["locus_start"] = starts
    work["locus_end"] = ends
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


def _collapse_score_column(collapse: str) -> str:
    if collapse in {"max", "max_capped", "mean_capped", "positive_mean_capped"}:
        return "ripple_d_capped_score"
    if collapse in {"huber_mean", "mean_huber"}:
        return "ripple_d_huber_score"
    raise ValueError(f"Unknown locus collapse method: {collapse}")


def _collapse_locus_scores(selected: pd.DataFrame, *, score_col: str, collapse: str = "max") -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame(columns=["locus_id", "locus_score", "n_genes_in_locus"])
    if collapse in {"max", "max_capped"}:
        agg = selected.groupby("locus_id", observed=True).agg(
            locus_score=(score_col, "max"),
            n_genes_in_locus=("gene_symbol", "nunique"),
        )
    elif collapse in {"huber_mean", "mean_huber", "mean_capped"}:
        agg = selected.groupby("locus_id", observed=True).agg(
            locus_score=(score_col, "mean"),
            n_genes_in_locus=("gene_symbol", "nunique"),
        )
    elif collapse == "positive_mean_capped":
        work = selected.copy()
        work["_positive_score"] = np.clip(pd.to_numeric(work[score_col], errors="coerce"), 0.0, None)
        agg = work.groupby("locus_id", observed=True).agg(
            locus_score=("_positive_score", "mean"),
            n_genes_in_locus=("gene_symbol", "nunique"),
        )
    else:
        raise ValueError(f"Unknown locus collapse method: {collapse}")
    return agg.reset_index()


def collapse_values(values: np.ndarray | pd.Series, *, collapse: str = "max") -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    if collapse in {"max", "max_capped"}:
        return float(np.max(arr))
    if collapse in {"huber_mean", "mean_huber", "mean_capped"}:
        return float(np.mean(arr))
    if collapse == "positive_mean_capped":
        return float(np.mean(np.clip(arr, 0.0, None)))
    raise ValueError(f"Unknown locus collapse method: {collapse}")


def locus_robust_stat(locus_scores: np.ndarray | pd.Series) -> float:
    values = np.asarray(locus_scores, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.sum(values) / np.sqrt(values.size))


def positive_locus_robust_stat(locus_scores: np.ndarray | pd.Series) -> float:
    """Positive-part locus statistic for broad modules with passenger genes."""

    values = np.asarray(locus_scores, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    positive = np.clip(values, 0.0, None)
    return float(np.sum(positive) / np.sqrt(values.size))


def rank_locus_enrichment_stat(rank_fractions: np.ndarray | pd.Series) -> float:
    """Return a high-is-more-enriched statistic from locus rank fractions.

    ``score_rank_fraction`` is small for top-ranked loci. This transform makes
    stronger rank enrichment larger, avoiding a direction mismatch in null
    calibration and downstream reports.
    """

    values = np.asarray(rank_fractions, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(1.0 - np.mean(values))


def score_rank_fractions(values: np.ndarray | pd.Series, background_scores: np.ndarray | pd.Series) -> np.ndarray:
    """Rank arbitrary module-specific scores against genome-wide locus scores."""

    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    background = np.asarray(background_scores, dtype=float)
    background = background[np.isfinite(background)]
    if vals.size == 0 or background.size == 0:
        return np.array([], dtype=float)
    sorted_background = np.sort(background)[::-1]
    ranks = np.searchsorted(-sorted_background, -vals, side="right")
    return np.clip(ranks / float(sorted_background.size), 0.0, 1.0)


def module_specific_rank_enrichment_stat(
    locus_scores: np.ndarray | pd.Series,
    background_scores: np.ndarray | pd.Series,
) -> float:
    """Rank enrichment based on module-gene-specific collapsed locus scores."""

    return rank_locus_enrichment_stat(score_rank_fractions(locus_scores, background_scores))


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

    score_col = _collapse_score_column(config.locus_collapse)
    locus_scores = _collapse_locus_scores(
        work,
        score_col=score_col,
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
        moderate_locus=(
            "assoc_resid_score",
            lambda x: int(((x > config.moderate_score_low) & (x < config.moderate_score_high)).any()),
        ),
    )
    out = agg.reset_index().merge(locus_scores[["locus_id", "genome_locus_score"]], on="locus_id", how="left")
    out["score_rank"] = out["genome_locus_score"].rank(method="first", ascending=False)
    out["score_rank_fraction"] = out["score_rank"] / len(out)
    out["degree_bin"] = _quantile_bins(out["mean_graph_degree"], config.degree_bins)
    out["gene_count_bin"] = _quantile_bins(out["n_locus_genes"], config.property_bins)
    out["annotation_bin"] = _quantile_bins(out["annotation_density"], config.property_bins)
    out["ld_bin"] = _quantile_bins(out["mean_local_ld_score"], config.property_bins)
    out["snp_count_bin"] = _quantile_bins(out["mean_mapped_snp_count"], config.property_bins)
    parts = [
        out["degree_bin"].astype(str)
        + "|"
        + out["gene_count_bin"].astype(str)
    ]
    if config.annotation_matching_enabled:
        parts.append(out["annotation_bin"].astype(str))
    parts.extend([out["ld_bin"].astype(str), out["snp_count_bin"].astype(str)])
    out["match_bin"] = parts[0]
    for part in parts[1:]:
        out["match_bin"] = out["match_bin"] + "|" + part
    return out


def locus_background_audit_table(locus_background: pd.DataFrame) -> pd.DataFrame:
    """Return one-row and per-bin diagnostics for a pseudo-locus background."""

    if locus_background.empty:
        return pd.DataFrame(
            [
                {
                    "audit_level": "overall",
                    "group": "all",
                    "n_loci": 0,
                    "median_genes_per_locus": float("nan"),
                    "p90_genes_per_locus": float("nan"),
                    "p95_genes_per_locus": float("nan"),
                    "max_genes_per_locus": float("nan"),
                    "median_locus_span_bp": float("nan"),
                    "p95_locus_span_bp": float("nan"),
                    "max_locus_span_bp": float("nan"),
                }
            ]
        )
    work = locus_background.copy()
    work["locus_span_bp"] = pd.to_numeric(work["locus_end"], errors="coerce") - pd.to_numeric(
        work["locus_start"], errors="coerce"
    )

    def summarize(group: pd.DataFrame, *, audit_level: str, group_name: str) -> dict[str, object]:
        return {
            "audit_level": audit_level,
            "group": group_name,
            "n_loci": int(group["locus_id"].nunique()),
            "median_genes_per_locus": float(group["n_locus_genes"].median()),
            "p90_genes_per_locus": float(group["n_locus_genes"].quantile(0.90)),
            "p95_genes_per_locus": float(group["n_locus_genes"].quantile(0.95)),
            "max_genes_per_locus": int(group["n_locus_genes"].max()),
            "median_locus_span_bp": float(group["locus_span_bp"].median()),
            "p95_locus_span_bp": float(group["locus_span_bp"].quantile(0.95)),
            "max_locus_span_bp": float(group["locus_span_bp"].max()),
        }

    rows = [summarize(work, audit_level="overall", group_name="all")]
    for chrom, group in work.groupby("chrom", observed=True):
        rows.append(summarize(group, audit_level="chromosome", group_name=str(chrom)))
    return pd.DataFrame(rows)


def prepare_locus_inputs(
    scores: pd.DataFrame,
    library: AnchoredModuleLibrary | None,
    config: RippleDConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the exact scored gene table and locus background used by RIPPLE-D."""

    if config.locus_id_column:
        work = assign_predefined_loci(scores, locus_id_column=config.locus_id_column)
    else:
        work = assign_pseudo_loci(scores, window_bp=config.locus_window_bp)
    work = add_ripple_d_score_columns(work, config)
    locus_background = build_locus_background(work, library, config)
    return work, locus_background


def prepare_locus_background(
    scores: pd.DataFrame,
    library: AnchoredModuleLibrary | None,
    config: RippleDConfig,
) -> pd.DataFrame:
    """Build the exact locus background used by RIPPLE-D module tests."""

    _, locus_background = prepare_locus_inputs(scores, library, config)
    return locus_background


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
    *,
    locus_background: pd.DataFrame | None = None,
) -> dict[str, object]:
    selected, locus_table = _module_locus_scores(scores, module_genes, config)
    raw_values = selected["assoc_resid_score"].to_numpy(dtype=float)
    capped_values = selected["ripple_d_capped_score"].to_numpy(dtype=float)
    locus_scores = locus_table["locus_score"].to_numpy(dtype=float)
    contribution = contribution_metrics(locus_scores)
    ripple_d_value = ripple_d_stat(locus_scores, config)
    if locus_background is not None and not locus_table.empty:
        by_locus = locus_background.set_index("locus_id", drop=False)
        rank_values = by_locus.loc[
            [locus for locus in locus_table["locus_id"].astype(str) if locus in by_locus.index],
            "score_rank_fraction",
        ].to_numpy(dtype=float)
        background_scores = locus_background["genome_locus_score"].to_numpy(dtype=float)
    elif "score_rank_fraction" in selected and not selected.empty:
        rank_values = selected["score_rank_fraction"].to_numpy(dtype=float)
        background_scores = scores["ripple_d_capped_score"].to_numpy(dtype=float)
    else:
        rank_values = np.array([], dtype=float)
        background_scores = np.array([], dtype=float)
    locus_membership_rank_stat = rank_locus_enrichment_stat(rank_values)
    module_specific_rank_stat = module_specific_rank_enrichment_stat(locus_scores, background_scores)
    top_gene = selected.sort_values("assoc_resid_score", ascending=False).head(1)
    top_locus = locus_table.sort_values("locus_score", ascending=False).head(1)
    return {
        "n_present": int(selected["gene_symbol"].nunique()),
        "n_loci": int(locus_table["locus_id"].nunique()),
        "raw_gene_stat": sqrt_n_mean_stat(raw_values),
        "capped_gene_stat": sqrt_n_mean_stat(capped_values),
        "locus_robust_stat": locus_robust_stat(locus_scores),
        "positive_locus_robust_stat": positive_locus_robust_stat(locus_scores),
        "ripple_d_stat": ripple_d_value,
        "moderate_locus_burden": moderate_locus_burden(selected, config),
        "locus_membership_rank_enrichment_stat": locus_membership_rank_stat,
        "module_specific_rank_enrichment_stat": module_specific_rank_stat,
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


def _null_stats_from_loci_legacy_full_locus_max(
    locus_background: pd.DataFrame, sampled_loci: Sequence[str], config: RippleDConfig
) -> dict[str, float]:
    """Legacy whole-locus-max null statistic retained only for comparison."""

    rows = locus_background.loc[locus_background["locus_id"].isin(set(sampled_loci))]
    scores = rows["genome_locus_score"].to_numpy(dtype=float)
    rank_fractions = rows["score_rank_fraction"].to_numpy(dtype=float)
    background_scores = locus_background["genome_locus_score"].to_numpy(dtype=float)
    return {
        "locus_robust_stat": locus_robust_stat(scores),
        "positive_locus_robust_stat": positive_locus_robust_stat(scores),
        "ripple_d_stat": ripple_d_stat(scores, config),
        "locus_membership_rank_enrichment_stat": rank_locus_enrichment_stat(rank_fractions),
        "module_specific_rank_enrichment_stat": module_specific_rank_enrichment_stat(scores, background_scores),
        "moderate_locus_burden": float(rows["moderate_locus"].sum()),
        "leave_top1_locus_stat": leave_top_locus_stat(pd.Series(scores), k=1),
    }


def _locus_matched_nulls_legacy_full_locus_max(
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
        "positive_locus_robust_stat": np.empty(n_null, dtype=float),
        "ripple_d_stat": np.empty(n_null, dtype=float),
        "locus_membership_rank_enrichment_stat": np.empty(n_null, dtype=float),
        "module_specific_rank_enrichment_stat": np.empty(n_null, dtype=float),
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
        stats = _null_stats_from_loci_legacy_full_locus_max(locus_background, sampled, config)
        for name, value in stats.items():
            out[name][idx] = value
    return out


def _top_locus_conditioned_nulls_legacy_full_locus_max(
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
        stats = _null_stats_from_loci_legacy_full_locus_max(locus_background, sampled, config)
        out[idx] = stats["leave_top1_locus_stat"]
    return out


def _locus_score_permutation_nulls_legacy_full_locus_max(
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


def _filter_pool_by_gene_count(pool: np.ndarray, locus_gene_count_arr: np.ndarray | None, min_gene_count: int) -> np.ndarray:
    if locus_gene_count_arr is None or min_gene_count <= 1 or pool.size == 0:
        return pool
    filtered = np.asarray([int(idx) for idx in pool if int(locus_gene_count_arr[int(idx)]) >= int(min_gene_count)], dtype=int)
    return filtered if filtered.size else pool


def _filter_pool_with_level(pool: np.ndarray, exclude: set[int]) -> tuple[np.ndarray, str]:
    filtered = _filter_pool(pool, exclude)
    return filtered, "exact"


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
    min_gene_count: int = 1,
    locus_gene_count_arr: np.ndarray | None = None,
) -> tuple[int, str, int]:
    match_bin = str(match_bin_arr[int(observed_idx)])
    exact_pool = match_pools_idx.get(match_bin, np.array([], dtype=int))
    candidates = _filter_pool(exact_pool, exclude)
    candidates = _filter_pool_by_gene_count(candidates, locus_gene_count_arr, min_gene_count)
    level = "exact"
    if candidates.size == 0:
        degree_bin = int(degree_bin_arr[int(observed_idx)])
        degree_pool = degree_pools_idx.get(degree_bin, np.array([], dtype=int))
        candidates = _filter_pool(degree_pool, exclude)
        candidates = _filter_pool_by_gene_count(candidates, locus_gene_count_arr, min_gene_count)
        level = "degree"
    if candidates.size == 0:
        candidates = _filter_pool(all_indices, exclude)
        candidates = _filter_pool_by_gene_count(candidates, locus_gene_count_arr, min_gene_count)
        level = "global"
    if candidates.size == 0:
        candidates = all_indices
        level = "all_with_reuse"
    return int(rng.choice(candidates)), level, int(candidates.size)


def _sample_locus_matched_indices_fast(
    observed_indices: Sequence[int],
    *,
    match_bin_arr: np.ndarray,
    degree_bin_arr: np.ndarray,
    match_pools_idx: Mapping[str, np.ndarray],
    degree_pools_idx: Mapping[int, np.ndarray],
    all_indices: np.ndarray,
    rng: np.random.Generator,
    extra_exclude: set[int] | None = None,
    observed_gene_counts: Sequence[int] | None = None,
    locus_gene_count_arr: np.ndarray | None = None,
) -> tuple[list[int], dict[str, float]]:
    observed_set = {int(idx) for idx in observed_indices}
    used = set(observed_set)
    if extra_exclude:
        used.update(int(idx) for idx in extra_exclude)
    sampled: list[int] = []
    counts = {"exact": 0, "degree": 0, "global": 0, "all_with_reuse": 0}
    pool_sizes: list[int] = []
    gene_counts = list(observed_gene_counts) if observed_gene_counts is not None else [1] * len(observed_indices)
    for observed_idx, min_gene_count in zip(observed_indices, gene_counts, strict=False):
        picked, level, pool_size = _sample_index_from_pools(
            int(observed_idx),
            match_bin_arr=match_bin_arr,
            degree_bin_arr=degree_bin_arr,
            match_pools_idx=match_pools_idx,
            degree_pools_idx=degree_pools_idx,
            all_indices=all_indices,
            exclude=used,
            rng=rng,
            min_gene_count=int(min_gene_count) if locus_gene_count_arr is not None else 1,
            locus_gene_count_arr=locus_gene_count_arr,
        )
        sampled.append(picked)
        used.add(picked)
        counts[level] = counts.get(level, 0) + 1
        pool_sizes.append(pool_size)
    denom = max(1, len(observed_indices))
    audit = {
        "null_exact_match_rate": counts.get("exact", 0) / denom,
        "null_degree_fallback_rate": counts.get("degree", 0) / denom,
        "null_global_fallback_rate": counts.get("global", 0) / denom,
        "null_reuse_fallback_rate": counts.get("all_with_reuse", 0) / denom,
        "min_match_pool_size": float(np.min(pool_sizes)) if pool_sizes else float("nan"),
        "median_match_pool_size": float(np.median(pool_sizes)) if pool_sizes else float("nan"),
    }
    return sampled, audit


def _sample_locus_subset_score(
    locus_idx: int,
    n_genes: int,
    *,
    locus_gene_score_pools: Mapping[int, np.ndarray],
    locus_gene_raw_pools: Mapping[int, np.ndarray],
    config: RippleDConfig,
    rng: np.random.Generator,
) -> tuple[float, float, bool, int, bool]:
    score_pool = np.asarray(locus_gene_score_pools[int(locus_idx)], dtype=float)
    raw_pool = np.asarray(locus_gene_raw_pools[int(locus_idx)], dtype=float)
    n = max(1, int(n_genes))
    insufficient = n > score_pool.size
    replace = insufficient
    positions = rng.choice(np.arange(score_pool.size), size=n, replace=replace)
    subset_scores = score_pool[positions]
    subset_raw = raw_pool[positions]
    return (
        collapse_values(subset_scores, collapse=config.locus_collapse),
        float(((subset_raw > config.moderate_score_low) & (subset_raw < config.moderate_score_high)).any()),
        bool(replace),
        int(max(0, n - int(score_pool.size))) if replace else 0,
        bool(insufficient),
    )


def _stats_from_index_scores(
    scores: np.ndarray,
    moderate_flags: np.ndarray,
    rank_fraction_arr: np.ndarray,
    background_scores: np.ndarray,
    indices: Sequence[int],
    config: RippleDConfig,
    *,
    rng: np.random.Generator | None = None,
    observed_gene_counts: Sequence[int] | None = None,
    locus_gene_score_pools: Mapping[int, np.ndarray] | None = None,
    locus_gene_raw_pools: Mapping[int, np.ndarray] | None = None,
    replacement_audit: list[dict[str, float]] | None = None,
) -> dict[str, float]:
    idx = np.asarray(indices, dtype=int)
    if (
        config.null_gene_subset_sampling
        and
        observed_gene_counts is not None
        and locus_gene_score_pools is not None
        and locus_gene_raw_pools is not None
        and rng is not None
    ):
        sampled_scores: list[float] = []
        sampled_moderate: list[float] = []
        replacement_flags: list[bool] = []
        replacement_draws: list[int] = []
        insufficient_flags: list[bool] = []
        for locus_idx, n_genes in zip(idx, observed_gene_counts, strict=False):
            score, moderate, used_replacement, n_replacement_draws, insufficient = _sample_locus_subset_score(
                int(locus_idx),
                int(n_genes),
                locus_gene_score_pools=locus_gene_score_pools,
                locus_gene_raw_pools=locus_gene_raw_pools,
                config=config,
                rng=rng,
            )
            sampled_scores.append(score)
            sampled_moderate.append(moderate)
            replacement_flags.append(used_replacement)
            replacement_draws.append(n_replacement_draws)
            insufficient_flags.append(insufficient)
        values = np.asarray(sampled_scores, dtype=float)
        moderate_values = np.asarray(sampled_moderate, dtype=float)
        if replacement_audit is not None:
            denom = max(1, len(replacement_flags))
            replacement_audit.append(
                {
                    "null_with_replacement_rate": float(np.sum(replacement_flags) / denom),
                    "null_mean_replacement_draws": float(np.mean(replacement_draws)) if replacement_draws else 0.0,
                    "null_loci_with_insufficient_gene_pool_rate": float(np.sum(insufficient_flags) / denom),
                }
            )
    else:
        values = scores[idx]
        moderate_values = moderate_flags[idx]
    rank_values = rank_fraction_arr[idx]
    return {
        "locus_robust_stat": locus_robust_stat(values),
        "positive_locus_robust_stat": positive_locus_robust_stat(values),
        "ripple_d_stat": ripple_d_stat(values, config),
        "locus_membership_rank_enrichment_stat": rank_locus_enrichment_stat(rank_values),
        "module_specific_rank_enrichment_stat": module_specific_rank_enrichment_stat(values, background_scores),
        "moderate_locus_burden": float(np.sum(moderate_values)),
        "leave_top1_locus_stat": leave_top_locus_stat(pd.Series(values), k=1),
    }


def _locus_matched_nulls_fast(
    observed_indices: Sequence[int],
    config: RippleDConfig,
    *,
    scores: np.ndarray,
    moderate_flags: np.ndarray,
    rank_fraction_arr: np.ndarray,
    background_scores: np.ndarray,
    match_bin_arr: np.ndarray,
    degree_bin_arr: np.ndarray,
    match_pools_idx: Mapping[str, np.ndarray],
    degree_pools_idx: Mapping[int, np.ndarray],
    all_indices: np.ndarray,
    observed_gene_counts: Sequence[int],
    locus_gene_count_arr: np.ndarray,
    locus_gene_score_pools: Mapping[int, np.ndarray],
    locus_gene_raw_pools: Mapping[int, np.ndarray],
    n_null: int,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    out = {
        "locus_robust_stat": np.empty(n_null, dtype=float),
        "positive_locus_robust_stat": np.empty(n_null, dtype=float),
        "ripple_d_stat": np.empty(n_null, dtype=float),
        "locus_membership_rank_enrichment_stat": np.empty(n_null, dtype=float),
        "module_specific_rank_enrichment_stat": np.empty(n_null, dtype=float),
        "moderate_locus_burden": np.empty(n_null, dtype=float),
        "leave_top1_locus_stat": np.empty(n_null, dtype=float),
    }
    audits: list[dict[str, float]] = []
    replacement_audits: list[dict[str, float]] = []
    for idx in range(n_null):
        sampled, audit = _sample_locus_matched_indices_fast(
            observed_indices,
            match_bin_arr=match_bin_arr,
            degree_bin_arr=degree_bin_arr,
            match_pools_idx=match_pools_idx,
            degree_pools_idx=degree_pools_idx,
            all_indices=all_indices,
            rng=rng,
            observed_gene_counts=observed_gene_counts if config.require_gene_count_match else None,
            locus_gene_count_arr=locus_gene_count_arr if config.require_gene_count_match else None,
        )
        audits.append(audit)
        replacement_audit: list[dict[str, float]] = []
        stats = _stats_from_index_scores(
            scores,
            moderate_flags,
            rank_fraction_arr,
            background_scores,
            sampled,
            config,
            rng=rng,
            observed_gene_counts=observed_gene_counts,
            locus_gene_score_pools=locus_gene_score_pools,
            locus_gene_raw_pools=locus_gene_raw_pools,
            replacement_audit=replacement_audit,
        )
        replacement_audits.extend(replacement_audit)
        for name, value in stats.items():
            out[name][idx] = value
    summary = {
        key: float(np.nanmean([audit[key] for audit in audits])) if audits else float("nan")
        for key in [
            "null_exact_match_rate",
            "null_degree_fallback_rate",
            "null_global_fallback_rate",
            "null_reuse_fallback_rate",
            "min_match_pool_size",
            "median_match_pool_size",
        ]
    }
    for key in [
        "null_with_replacement_rate",
        "null_mean_replacement_draws",
        "null_loci_with_insufficient_gene_pool_rate",
    ]:
        summary[key] = float(np.nanmean([audit[key] for audit in replacement_audits])) if replacement_audits else 0.0
    return out, summary


def _top_locus_conditioned_nulls_fast(
    observed_indices: Sequence[int],
    config: RippleDConfig,
    *,
    scores: np.ndarray,
    moderate_flags: np.ndarray,
    rank_fraction_arr: np.ndarray,
    background_scores: np.ndarray,
    score_rank_arr: np.ndarray,
    match_bin_arr: np.ndarray,
    degree_bin_arr: np.ndarray,
    match_pools_idx: Mapping[str, np.ndarray],
    degree_pools_idx: Mapping[int, np.ndarray],
    all_indices: np.ndarray,
    observed_gene_counts: Sequence[int],
    locus_gene_count_arr: np.ndarray,
    locus_gene_score_pools: Mapping[int, np.ndarray],
    locus_gene_raw_pools: Mapping[int, np.ndarray],
    n_null: int,
    rng: np.random.Generator,
    top_fraction: float = 0.01,
) -> tuple[np.ndarray, dict[str, float]]:
    if len(observed_indices) == 0:
        return np.full(n_null, np.nan, dtype=float), {}
    top_cutoff = max(1, int(math.ceil(len(score_rank_arr) * top_fraction)))
    top_indices = all_indices[score_rank_arr <= top_cutoff]
    observed_top = [int(idx) for idx in observed_indices if score_rank_arr[int(idx)] <= top_cutoff]
    if len(observed_top) == 0 or top_indices.size == 0:
        return np.full(n_null, np.nan, dtype=float), {}
    top_set = {int(idx) for idx in top_indices}
    top_match_pools_idx = {
        key: np.asarray([int(idx) for idx in pool if int(idx) in top_set], dtype=int)
        for key, pool in match_pools_idx.items()
    }
    top_degree_pools_idx = {
        key: np.asarray([int(idx) for idx in pool if int(idx) in top_set], dtype=int)
        for key, pool in degree_pools_idx.items()
    }
    observed_top_set = set(observed_top)
    observed_pairs = list(zip([int(idx) for idx in observed_indices], list(observed_gene_counts), strict=False))
    top_counts = [int(count) for idx, count in observed_pairs if int(idx) in observed_top_set]
    remaining_pairs = [(int(idx), int(count)) for idx, count in observed_pairs if int(idx) not in observed_top_set]
    remaining = [idx for idx, _ in remaining_pairs]
    remaining_counts = [count for _, count in remaining_pairs]
    out = np.empty(n_null, dtype=float)
    top_audits: list[dict[str, float]] = []
    for idx in range(n_null):
        forced_list, top_audit = _sample_locus_matched_indices_fast(
            observed_top,
            match_bin_arr=match_bin_arr,
            degree_bin_arr=degree_bin_arr,
            match_pools_idx=top_match_pools_idx,
            degree_pools_idx=top_degree_pools_idx,
            all_indices=top_indices,
            rng=rng,
            observed_gene_counts=top_counts if config.require_gene_count_match else None,
            locus_gene_count_arr=locus_gene_count_arr if config.require_gene_count_match else None,
        )
        top_audits.append(top_audit)
        sampled = list(forced_list)
        remaining_sampled, _ = _sample_locus_matched_indices_fast(
            remaining,
            match_bin_arr=match_bin_arr,
            degree_bin_arr=degree_bin_arr,
            match_pools_idx=match_pools_idx,
            degree_pools_idx=degree_pools_idx,
            all_indices=all_indices,
            rng=rng,
            extra_exclude=set(forced_list),
            observed_gene_counts=remaining_counts if config.require_gene_count_match else None,
            locus_gene_count_arr=locus_gene_count_arr if config.require_gene_count_match else None,
        )
        sampled.extend(remaining_sampled)
        sampled_gene_counts = top_counts + remaining_counts
        out[idx] = _stats_from_index_scores(
            scores,
            moderate_flags,
            rank_fraction_arr,
            background_scores,
            sampled,
            config,
            rng=rng,
            observed_gene_counts=sampled_gene_counts,
            locus_gene_score_pools=locus_gene_score_pools,
            locus_gene_raw_pools=locus_gene_raw_pools,
        )["leave_top1_locus_stat"]
    audit = {
        f"top_conditioned_{key}": float(np.nanmean([entry[key] for entry in top_audits])) if top_audits else 0.0
        for key in [
            "null_exact_match_rate",
            "null_degree_fallback_rate",
            "null_global_fallback_rate",
            "null_reuse_fallback_rate",
        ]
    }
    return out, audit


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
    ripple_p = finite_or_nan(row.get("ripple_d_empirical_p"))
    positive_p = finite_or_nan(row.get("positive_locus_empirical_p"))
    rank_p = finite_or_nan(row.get("module_specific_rank_empirical_p"))
    moderate_p = finite_or_nan(row.get("moderate_locus_burden_empirical_p"))
    leave_p = finite_or_nan(row.get("leave_top1_locus_empirical_p"))
    raw_p = finite_or_nan(row.get("raw_gene_empirical_p"))
    n_eff = finite_or_nan(row.get("n_effective_loci"))
    top1 = finite_or_nan(row.get("top1_locus_contribution"))
    top5 = finite_or_nan(row.get("top5_locus_contribution"))

    locus_supported = locus_p < 0.05 or ripple_p < config.ripple_d_p_max
    positive_supported = positive_p < config.positive_locus_p_max
    rank_supported = rank_p < config.rank_locus_p_max
    leave_supported = leave_p < 0.10
    moderate_supported = moderate_p < 0.10
    contribution_supported = (
        n_eff >= config.effective_loci_target
        and top1 <= config.top1_contribution_max
        and top5 <= config.top5_contribution_max
    )
    top1_not_dominant = top1 <= config.top1_contribution_max

    passes = (
        locus_p < 0.05
        and ripple_p < config.ripple_d_p_max
        and moderate_supported
        and leave_supported
        and contribution_supported
        and (positive_supported or rank_supported)
    )
    if passes:
        return "distributed_weak_signal_module_candidate"
    if (
        locus_supported
        and top1_not_dominant
        and n_eff >= config.effective_loci_target
        and (positive_supported or rank_supported or leave_supported or moderate_supported)
    ):
        return "mixed_sparse_distributed_candidate"
    if locus_supported and moderate_supported and top1_not_dominant:
        return "moderate_locus_supported_module"
    if locus_supported and rank_supported and top1_not_dominant:
        return "module_specific_rank_supported_module"
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
    precomputed_work: pd.DataFrame | None = None,
    precomputed_locus_background: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Run fixed-library RIPPLE-D diagnostics."""

    config = config or RippleDConfig()
    if config.manuscript_mode and not config.null_gene_subset_sampling:
        raise ValueError("manuscript_mode requires null_gene_subset_sampling=True")
    if precomputed_work is None and precomputed_locus_background is None:
        work, locus_background = prepare_locus_inputs(scores, library, config)
    else:
        if precomputed_work is None:
            raise ValueError("precomputed_work must be supplied with precomputed_locus_background")
        work = precomputed_work
        locus_background = (
            precomputed_locus_background
            if precomputed_locus_background is not None
            else build_locus_background(work, library, config)
        )
    gene_to_row = {str(row.gene_symbol).upper(): row for row in work.itertuples(index=False)}
    rng = np.random.default_rng(seed)
    all_loci = locus_background["locus_id"].to_numpy(dtype=object)
    all_locus_indices = np.arange(len(locus_background), dtype=int)
    locus_id_to_index = {str(locus_id): int(idx) for idx, locus_id in enumerate(all_loci)}
    genome_locus_scores = locus_background["genome_locus_score"].to_numpy(dtype=float)
    moderate_flags = locus_background["moderate_locus"].to_numpy(dtype=float)
    locus_gene_count_arr = locus_background["n_locus_genes"].to_numpy(dtype=int)
    match_bin_arr = locus_background["match_bin"].astype(str).to_numpy(dtype=object)
    degree_bin_arr = locus_background["degree_bin"].to_numpy(dtype=int)
    score_rank_arr = locus_background["score_rank"].to_numpy(dtype=float)
    score_rank_fraction_arr = locus_background["score_rank_fraction"].to_numpy(dtype=float)
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
    locus_gene_score_pools = {
        locus_id_to_index[str(locus_id)]: group["ripple_d_capped_score"].to_numpy(dtype=float)
        for locus_id, group in work.groupby("locus_id", observed=True)
        if str(locus_id) in locus_id_to_index
    }
    locus_gene_raw_pools = {
        locus_id_to_index[str(locus_id)]: group["assoc_resid_score"].to_numpy(dtype=float)
        for locus_id, group in work.groupby("locus_id", observed=True)
        if str(locus_id) in locus_id_to_index
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
            "locus_definition": config.locus_definition_name
            or (
                f"external_locus_column:{config.locus_id_column}"
                if config.locus_id_column
                else f"gene_coordinate_pseudo_locus_pm_{config.locus_window_bp}bp"
            ),
            "locus_window_bp": int(config.locus_window_bp) if not config.locus_id_column else "",
            "annotation_matching_enabled": bool(config.annotation_matching_enabled),
            "rank_evidence_interpretation": "module_specific_rank_required_for_distributed_gate;locus_membership_rank_reported_only",
            "ld_block_locus_sensitivity_status": "not_tested" if not config.locus_id_column else "external_locus_column_used",
            "pseudo_locus_window_stability_status": "not_tested",
        }
        if len(present) < min_present:
            rows.append({**base, "module_status": "not_tested_low_overlap", "module_label": "not_tested"})
            continue

        observed = summarize_module_distribution(work, present, config, locus_background=locus_background)
        selected, locus_table = _module_locus_scores(work, present, config)
        observed_loci = locus_table["locus_id"].astype(str).tolist()
        observed_locus_indices = [locus_id_to_index[locus] for locus in observed_loci if locus in locus_id_to_index]
        observed_gene_counts = [
            int(count)
            for locus, count in zip(
                locus_table["locus_id"].astype(str),
                locus_table["n_genes_in_locus"].astype(int),
                strict=True,
            )
            if locus in locus_id_to_index
        ]
        raw_gene_null = np.empty(n_null, dtype=float)
        locus_nulls, locus_null_audit = _locus_matched_nulls_fast(
            observed_locus_indices,
            config,
            scores=genome_locus_scores,
            moderate_flags=moderate_flags,
            rank_fraction_arr=score_rank_fraction_arr,
            background_scores=genome_locus_scores,
            match_bin_arr=match_bin_arr,
            degree_bin_arr=degree_bin_arr,
            match_pools_idx=match_pools_idx,
            degree_pools_idx=degree_pools_idx,
            all_indices=all_locus_indices,
            observed_gene_counts=observed_gene_counts,
            locus_gene_count_arr=locus_gene_count_arr,
            locus_gene_score_pools=locus_gene_score_pools,
            locus_gene_raw_pools=locus_gene_raw_pools,
            n_null=n_null,
            rng=rng,
        )
        top_conditioned_null, top_conditioned_audit = _top_locus_conditioned_nulls_fast(
            observed_locus_indices,
            config,
            scores=genome_locus_scores,
            moderate_flags=moderate_flags,
            rank_fraction_arr=score_rank_fraction_arr,
            background_scores=genome_locus_scores,
            score_rank_arr=score_rank_arr,
            match_bin_arr=match_bin_arr,
            degree_bin_arr=degree_bin_arr,
            match_pools_idx=match_pools_idx,
            degree_pools_idx=degree_pools_idx,
            all_indices=all_locus_indices,
            observed_gene_counts=observed_gene_counts,
            locus_gene_count_arr=locus_gene_count_arr,
            locus_gene_score_pools=locus_gene_score_pools,
            locus_gene_raw_pools=locus_gene_raw_pools,
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
            "positive_locus_null_mean": float(np.nanmean(locus_nulls["positive_locus_robust_stat"])),
            "positive_locus_null_sd": float(np.nanstd(locus_nulls["positive_locus_robust_stat"], ddof=1)),
            "positive_locus_z": z_score(
                observed["positive_locus_robust_stat"],
                locus_nulls["positive_locus_robust_stat"],
            ),
            "positive_locus_empirical_p": empirical_upper(
                locus_nulls["positive_locus_robust_stat"],
                observed["positive_locus_robust_stat"],
            ),
            "ripple_d_null_mean": float(np.nanmean(locus_nulls["ripple_d_stat"])),
            "ripple_d_null_sd": float(np.nanstd(locus_nulls["ripple_d_stat"], ddof=1)),
            "ripple_d_z": z_score(observed["ripple_d_stat"], locus_nulls["ripple_d_stat"]),
            "ripple_d_empirical_p": empirical_upper(locus_nulls["ripple_d_stat"], observed["ripple_d_stat"]),
            "locus_membership_rank_null_mean": float(
                np.nanmean(locus_nulls["locus_membership_rank_enrichment_stat"])
            ),
            "locus_membership_rank_null_sd": float(
                np.nanstd(locus_nulls["locus_membership_rank_enrichment_stat"], ddof=1)
            ),
            "locus_membership_rank_z": z_score(
                observed["locus_membership_rank_enrichment_stat"],
                locus_nulls["locus_membership_rank_enrichment_stat"],
            ),
            "locus_membership_rank_empirical_p": empirical_upper(
                locus_nulls["locus_membership_rank_enrichment_stat"],
                observed["locus_membership_rank_enrichment_stat"],
            ),
            "module_specific_rank_null_mean": float(
                np.nanmean(locus_nulls["module_specific_rank_enrichment_stat"])
            ),
            "module_specific_rank_null_sd": float(
                np.nanstd(locus_nulls["module_specific_rank_enrichment_stat"], ddof=1)
            ),
            "module_specific_rank_z": z_score(
                observed["module_specific_rank_enrichment_stat"],
                locus_nulls["module_specific_rank_enrichment_stat"],
            ),
            "module_specific_rank_empirical_p": empirical_upper(
                locus_nulls["module_specific_rank_enrichment_stat"],
                observed["module_specific_rank_enrichment_stat"],
            ),
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
            "null_gene_subset_sampling": bool(config.null_gene_subset_sampling),
            "n_module_genes_per_locus": ",".join(str(count) for count in observed_gene_counts),
            **locus_null_audit,
            **top_conditioned_audit,
            "null_gene_count_match_degraded": bool(locus_null_audit.get("null_with_replacement_rate", 0.0) > 0.05),
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
                    "n_module_genes_in_locus": int(detail.n_genes_in_locus),
                }
            )
        if return_null_details:
            for null_type, stat_name, values in [
                ("raw_gene_random_null", "raw_gene_stat", raw_gene_null),
                ("locus_matched_competitive_null", "locus_robust_stat", locus_nulls["locus_robust_stat"]),
                (
                    "locus_matched_competitive_null",
                    "positive_locus_robust_stat",
                    locus_nulls["positive_locus_robust_stat"],
                ),
                ("locus_matched_competitive_null", "ripple_d_stat", locus_nulls["ripple_d_stat"]),
                (
                    "locus_matched_competitive_null",
                    "locus_membership_rank_enrichment_stat",
                    locus_nulls["locus_membership_rank_enrichment_stat"],
                ),
                (
                    "locus_matched_competitive_null",
                    "module_specific_rank_enrichment_stat",
                    locus_nulls["module_specific_rank_enrichment_stat"],
                ),
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
        modules["positive_locus_fdr"] = bh_fdr(modules["positive_locus_empirical_p"].to_numpy(dtype=float))
        modules["ripple_d_fdr"] = bh_fdr(modules["ripple_d_empirical_p"].to_numpy(dtype=float))
        modules["locus_membership_rank_fdr"] = bh_fdr(
            modules["locus_membership_rank_empirical_p"].to_numpy(dtype=float)
        )
        modules["module_specific_rank_fdr"] = bh_fdr(modules["module_specific_rank_empirical_p"].to_numpy(dtype=float))
        modules = modules.sort_values(
            [
                "module_status",
                "locus_robust_empirical_p",
                "ripple_d_empirical_p",
                "positive_locus_empirical_p",
                "module_specific_rank_empirical_p",
                "ripple_d_stat",
            ],
            ascending=[True, True, True, True, True, False],
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
        "n_mixed_sparse_distributed_candidate": int(
            modules["module_status"].eq("mixed_sparse_distributed_candidate").sum()
        )
        if "module_status" in modules
        else 0,
        "n_moderate_locus_supported_module": int(
            modules["module_status"].eq("moderate_locus_supported_module").sum()
        )
        if "module_status" in modules
        else 0,
        "n_module_specific_rank_supported_module": int(
            modules["module_status"].eq("module_specific_rank_supported_module").sum()
        )
        if "module_status" in modules
        else 0,
        "n_top_locus_dominant_module": int(modules["module_status"].eq("top_locus_dominant_module").sum())
        if "module_status" in modules
        else 0,
        "n_sparse_locus_pathway_overlap": int(modules["module_status"].eq("sparse_locus_pathway_overlap").sum())
        if "module_status" in modules
        else 0,
        "n_raw_gene_set_enrichment_only": int(modules["module_status"].eq("raw_gene_set_enrichment_only").sum())
        if "module_status" in modules
        else 0,
        "n_background_genes": int(work["gene_symbol"].nunique()),
        "n_background_loci": int(locus_background["locus_id"].nunique()),
        "median_genes_per_locus": float(locus_background["n_locus_genes"].median()),
        "p95_genes_per_locus": float(locus_background["n_locus_genes"].quantile(0.95)),
        "max_genes_per_locus": int(locus_background["n_locus_genes"].max()),
        "max_locus_span_bp": float((locus_background["locus_end"] - locus_background["locus_start"]).max()),
        "n_null_gene_count_match_degraded": int(modules.get("null_gene_count_match_degraded", pd.Series(dtype=bool)).sum())
        if not modules.empty
        else 0,
        "n_null": int(n_null),
        "score_cap": float(config.score_cap),
        "locus_window_bp": int(config.locus_window_bp),
        "locus_definition": config.locus_definition_name
        or (
            f"external_locus_column:{config.locus_id_column}"
            if config.locus_id_column
            else f"gene_coordinate_pseudo_locus_pm_{config.locus_window_bp}bp"
        ),
        "annotation_matching_enabled": bool(config.annotation_matching_enabled),
        "locus_id_column": config.locus_id_column or "",
        "null_details_returned": bool(return_null_details),
        "null_gene_subset_sampling": bool(config.null_gene_subset_sampling),
        "require_gene_count_match": bool(config.require_gene_count_match),
    }
    return modules, pd.DataFrame(locus_rows), pd.DataFrame(null_rows), summary
