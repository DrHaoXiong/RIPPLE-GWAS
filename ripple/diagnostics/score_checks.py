"""Gene-score transform, clipping, and residualization diagnostics."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from ripple.graph_diffusion import observed_diffusion_statistics


def _safe_corr(x: pd.Series, y: pd.Series, *, method: str) -> float:
    table = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(table) < 3 or table["x"].nunique() <= 1 or table["y"].nunique() <= 1:
        return float("nan")
    return float(table["x"].corr(table["y"], method=method))


def gene_score_clipping_diagnostics(
    table: pd.DataFrame,
    *,
    trait: str,
    p_col: str = "assoc_p_g",
    clipped_col: str = "assoc_p_g_clipped",
    p_clip_min: float = 1e-300,
    p_clip_max: float = 1.0 - 1e-16,
) -> pd.DataFrame:
    """Summarize low and high P-value clipping."""

    if p_col not in table.columns:
        raise ValueError(f"table is missing {p_col}.")
    p_raw = pd.to_numeric(table[p_col], errors="coerce")
    if clipped_col in table.columns:
        p_clipped = pd.to_numeric(table[clipped_col], errors="coerce")
    else:
        p_clipped = p_raw.clip(lower=p_clip_min, upper=p_clip_max)
    p_clip_max = min(float(p_clip_max), np.nextafter(1.0, 0.0))
    clipped_low = p_raw <= float(p_clip_min)
    clipped_high = p_raw >= p_clip_max
    changed = p_raw.ne(p_clipped)
    n_total = int(p_raw.notna().sum())
    return pd.DataFrame(
        [
            {
                "trait": trait,
                "p_clip_min": float(p_clip_min),
                "p_clip_max": float(p_clip_max),
                "n_genes": n_total,
                "n_clipped_low": int(clipped_low.sum()),
                "n_clipped_high": int(clipped_high.sum()),
                "fraction_clipped_low": float(clipped_low.sum() / n_total) if n_total else 0.0,
                "fraction_clipped_high": float(clipped_high.sum() / n_total) if n_total else 0.0,
                "n_clipped_any": int(changed.sum()),
                "fraction_clipped_any": float(changed.sum() / n_total) if n_total else 0.0,
            }
        ]
    )


def residualization_diagnostics(
    table: pd.DataFrame,
    *,
    trait: str,
    score_type: str = "assoc_normal_score_g",
    residualized_col: str = "assoc_resid_score",
    covariates: Iterable[str] = (
        "log_gene_length",
        "log_mapped_snp_count",
        "log_m_eff",
        "local_ld_score",
        "mappability",
        "graph_degree",
    ),
) -> pd.DataFrame:
    """Compute pre/post residualization correlations with technical covariates."""

    missing = [col for col in (score_type, residualized_col) if col not in table.columns]
    if missing:
        raise ValueError(f"table is missing score columns: {missing}")
    rows: list[dict[str, object]] = []
    for covariate in covariates:
        if covariate not in table.columns:
            continue
        rows.append(
            {
                "trait": trait,
                "score_type": score_type,
                "covariate": covariate,
                "pearson_before": _safe_corr(table[score_type], table[covariate], method="pearson"),
                "spearman_before": _safe_corr(table[score_type], table[covariate], method="spearman"),
                "pearson_after": _safe_corr(table[residualized_col], table[covariate], method="pearson"),
                "spearman_after": _safe_corr(table[residualized_col], table[covariate], method="spearman"),
            }
        )
    return pd.DataFrame(rows)


def _rank_normal(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="raise")
    ranks = stats.rankdata(x, method="average")
    probs = (ranks - 0.5) / len(ranks)
    return pd.Series(stats.norm.ppf(probs), index=values.index)


def _top_set(values: pd.Series, fraction: float) -> set[int]:
    n_top = max(1, int(np.ceil(len(values) * fraction)))
    return set(values.sort_values(ascending=False).head(n_top).index)


def _jaccard(a: set[int], b: set[int]) -> float:
    return float(len(a & b) / len(a | b)) if a or b else float("nan")


def gene_score_transform_sensitivity(
    table: pd.DataFrame,
    *,
    trait: str,
    graph=None,
    default_col: str = "assoc_normal_score_g",
    p_col: str = "assoc_p_g_clipped",
) -> pd.DataFrame:
    """Compare score transforms against the default normal-score transform.

    Downstream recalibration can be expensive, so this diagnostic always reports
    correlations and top-k overlaps. If a graph is supplied, it also reports the
    observed diffusion maxT for each transform without null calibration.
    """

    missing = [col for col in (default_col, p_col) if col not in table.columns]
    if missing:
        raise ValueError(f"table is missing required columns: {missing}")
    default = pd.to_numeric(table[default_col], errors="raise")
    p_values = pd.to_numeric(table[p_col], errors="raise")
    transforms = {
        "normal_score": default,
        "minuslog10p": -np.log10(p_values),
        "rank_normal": _rank_normal(default),
    }
    default_top_1 = _top_set(default, 0.01)
    default_top_5 = _top_set(default, 0.05)
    rows: list[dict[str, object]] = []
    for name, transformed in transforms.items():
        diffusion_tmax = float("nan")
        if graph is not None:
            tmp = table.copy()
            tmp["assoc_resid_score"] = transformed.to_numpy(dtype=float)
            try:
                _, diffusion = observed_diffusion_statistics(graph, tmp, score_mode="positive")
                diffusion_tmax = float(diffusion["T_max"])
            except ValueError:
                diffusion_tmax = float("nan")
        rows.append(
            {
                "trait": trait,
                "score_transform": name,
                "n_genes": int(len(table)),
                "correlation_with_default_pearson": _safe_corr(transformed, default, method="pearson"),
                "correlation_with_default_spearman": _safe_corr(transformed, default, method="spearman"),
                "top_1pct_jaccard": _jaccard(_top_set(transformed, 0.01), default_top_1),
                "top_5pct_jaccard": _jaccard(_top_set(transformed, 0.05), default_top_5),
                "percolation_delta_auc": float("nan"),
                "diffusion_T_max": diffusion_tmax,
            }
        )
    return pd.DataFrame(rows)
