"""Technical residualization for unsigned association-strength scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from ripple.defaults import DEGREE_RESIDUALIZATION_COVARIATE, PRIMARY_RESIDUALIZATION_COVARIATES


@dataclass(frozen=True)
class ResidualizationResult:
    """Residualized association scores and fitted null moments."""

    residualized_score: np.ndarray
    raw_residual: np.ndarray
    mu0: np.ndarray
    sigma0: np.ndarray
    covariate_names: tuple[str, ...]
    beta: np.ndarray
    covariate_center: np.ndarray
    covariate_scale: np.ndarray
    method: str
    includes_degree: bool


def _as_1d_float_array(values: np.ndarray | list[float] | tuple[float, ...], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _as_2d_float_array(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def validate_primary_covariates(
    covariate_names: Iterable[str],
    *,
    allow_degree: bool = False,
) -> tuple[str, ...]:
    """Validate V1 residualization covariates.

    Degree is deliberately excluded from primary residualization. Set
    `allow_degree=True` only for the mandatory degree-residualized sensitivity mode.
    """

    names = tuple(str(name) for name in covariate_names)
    if not names:
        raise ValueError("At least one residualization covariate is required.")
    if DEGREE_RESIDUALIZATION_COVARIATE in names and not allow_degree:
        raise ValueError(
            f"{DEGREE_RESIDUALIZATION_COVARIATE} is not allowed in primary residualization."
        )
    return names


def covariate_matrix_from_frame(
    table: pd.DataFrame,
    covariate_names: Iterable[str] = PRIMARY_RESIDUALIZATION_COVARIATES,
    *,
    allow_degree: bool = False,
) -> np.ndarray:
    """Extract a finite covariate matrix from a DataFrame."""

    names = validate_primary_covariates(covariate_names, allow_degree=allow_degree)
    missing = [name for name in names if name not in table.columns]
    if missing:
        raise ValueError(f"Missing residualization covariates: {missing}")
    return _as_2d_float_array(table.loc[:, names].to_numpy(dtype=float), "covariates")


def standardize_covariates(
    covariates: np.ndarray,
    *,
    center: np.ndarray | None = None,
    scale: np.ndarray | None = None,
    scale_floor: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center and scale covariates with a floor for constant columns."""

    x = _as_2d_float_array(covariates, "covariates")
    if center is None:
        center = np.mean(x, axis=0)
    else:
        center = _as_1d_float_array(center, "center")
    if scale is None:
        scale = np.std(x, axis=0, ddof=0)
    else:
        scale = _as_1d_float_array(scale, "scale")
    if center.shape[0] != x.shape[1] or scale.shape[0] != x.shape[1]:
        raise ValueError("center and scale must match covariate columns.")

    safe_scale = np.where(scale <= scale_floor, 1.0, scale)
    return (x - center) / safe_scale, center.astype(float), safe_scale.astype(float)


def design_matrix(covariates: np.ndarray) -> np.ndarray:
    """Return intercept-augmented standardized covariate design matrix."""

    x = _as_2d_float_array(covariates, "covariates")
    return np.column_stack([np.ones(x.shape[0]), x])


def residualize_from_null(
    observed_scores: np.ndarray | list[float] | tuple[float, ...],
    null_scores: np.ndarray,
    covariates: np.ndarray,
    covariate_names: Iterable[str] = PRIMARY_RESIDUALIZATION_COVARIATES,
    *,
    allow_degree: bool = False,
    sigma_floor: float = 1e-6,
) -> ResidualizationResult:
    """Residualize observed scores using null-estimated `mu0(C)` and `sigma0(C)`.

    `null_scores` must be shaped `(n_null_replicates, n_genes)`.
    """

    names = validate_primary_covariates(covariate_names, allow_degree=allow_degree)
    y_obs = _as_1d_float_array(observed_scores, "observed_scores")
    y_null = _as_2d_float_array(null_scores, "null_scores")
    if y_null.shape[1] != y_obs.size:
        raise ValueError("null_scores must have shape (n_null, n_genes).")
    if y_null.shape[0] < 2:
        raise ValueError("At least two null replicates are required to estimate sigma0.")

    x_std, center, scale = standardize_covariates(covariates)
    if x_std.shape[0] != y_obs.size:
        raise ValueError("covariates rows must match observed_scores.")
    if x_std.shape[1] != len(names):
        raise ValueError("covariate_names length must match covariate columns.")

    x_design = design_matrix(x_std)
    stacked_x = np.tile(x_design, (y_null.shape[0], 1))
    stacked_y = y_null.reshape(-1)
    beta, *_ = np.linalg.lstsq(stacked_x, stacked_y, rcond=None)

    mu0 = x_design @ beta
    null_residuals = y_null - mu0[None, :]
    sigma0 = np.std(null_residuals, axis=0, ddof=1)
    global_sigma = float(np.std(null_residuals.reshape(-1), ddof=1))
    sigma0 = np.where(sigma0 <= sigma_floor, max(global_sigma, sigma_floor), sigma0)

    raw_residual = y_obs - mu0
    residualized = raw_residual / sigma0
    return ResidualizationResult(
        residualized_score=residualized.astype(float),
        raw_residual=raw_residual.astype(float),
        mu0=mu0.astype(float),
        sigma0=sigma0.astype(float),
        covariate_names=names,
        beta=beta.astype(float),
        covariate_center=center,
        covariate_scale=scale,
        method="null_estimated_linear_mean_per_gene_sigma",
        includes_degree=DEGREE_RESIDUALIZATION_COVARIATE in names,
    )


def residualize_with_regression(
    scores: np.ndarray | list[float] | tuple[float, ...],
    covariates: np.ndarray,
    covariate_names: Iterable[str] = PRIMARY_RESIDUALIZATION_COVARIATES,
    *,
    allow_degree: bool = False,
    sigma_floor: float = 1e-6,
) -> ResidualizationResult:
    """Fast fallback residualization using observed-score regression only."""

    names = validate_primary_covariates(covariate_names, allow_degree=allow_degree)
    y = _as_1d_float_array(scores, "scores")
    x_std, center, scale = standardize_covariates(covariates)
    if x_std.shape[0] != y.size:
        raise ValueError("covariates rows must match scores.")
    if x_std.shape[1] != len(names):
        raise ValueError("covariate_names length must match covariate columns.")

    x_design = design_matrix(x_std)
    beta, *_ = np.linalg.lstsq(x_design, y, rcond=None)
    mu = x_design @ beta
    raw_residual = y - mu
    sigma = float(np.std(raw_residual, ddof=1))
    sigma = max(sigma, sigma_floor)
    sigma0 = np.full_like(y, sigma, dtype=float)
    residualized = raw_residual / sigma

    return ResidualizationResult(
        residualized_score=residualized.astype(float),
        raw_residual=raw_residual.astype(float),
        mu0=mu.astype(float),
        sigma0=sigma0,
        covariate_names=names,
        beta=beta.astype(float),
        covariate_center=center,
        covariate_scale=scale,
        method="observed_regression_fallback",
        includes_degree=DEGREE_RESIDUALIZATION_COVARIATE in names,
    )


def append_residualized_score(
    table: pd.DataFrame,
    *,
    score_col: str = "assoc_normal_score_g",
    output_col: str = "assoc_resid_score",
    covariate_names: Iterable[str] = PRIMARY_RESIDUALIZATION_COVARIATES,
    null_scores: np.ndarray | None = None,
    allow_degree: bool = False,
) -> tuple[pd.DataFrame, ResidualizationResult]:
    """Append a residualized association score to a gene-level table."""

    if score_col not in table.columns:
        raise ValueError(f"Missing score column: {score_col}")
    names = validate_primary_covariates(covariate_names, allow_degree=allow_degree)
    covariates = covariate_matrix_from_frame(table, names, allow_degree=allow_degree)
    scores = _as_1d_float_array(table[score_col].to_numpy(dtype=float), score_col)

    if null_scores is None:
        result = residualize_with_regression(
            scores,
            covariates,
            names,
            allow_degree=allow_degree,
        )
    else:
        result = residualize_from_null(
            scores,
            null_scores,
            covariates,
            names,
            allow_degree=allow_degree,
        )

    out = table.copy()
    out[output_col] = result.residualized_score
    return out, result
