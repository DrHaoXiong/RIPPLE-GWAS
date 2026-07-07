"""Unsigned LD-aware quadratic association signal.

The V1 default unsigned stream is:

    Q_g = z_g^T W_g z_g

with nonnegative PSD weights, defaulting to `W_g = diag(w_sg^2)`. Under the
LD null, `Q_g` is approximated as a weighted mixture of chi-square_1 variables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import optimize, stats

from ripple.defaults import DEFAULT_LD_SHRINKAGE
from ripple.signals.signed import shrink_ld

QuadraticPValueMethod = Literal["saddlepoint", "liu", "satterthwaite", "davies"]


@dataclass(frozen=True)
class QuadraticAssociationResult:
    """Result for one gene-level unsigned quadratic association calculation."""

    statistic: float
    assoc_p_g: float
    assoc_p_g_clipped: float
    assoc_minuslog10p_g: float
    assoc_normal_score_g: float
    is_p_clipped: bool
    mixture_eigenvalues: np.ndarray
    method: str
    n_snps: int
    shrinkage: float


def _as_1d_float_array(values: np.ndarray | list[float] | tuple[float, ...], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _as_square_float_matrix(values: np.ndarray, name: str) -> np.ndarray:
    mat = np.asarray(values, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"{name} must be a square matrix.")
    if not np.all(np.isfinite(mat)):
        raise ValueError(f"{name} must contain only finite values.")
    return 0.5 * (mat + mat.T)


def default_weight_matrix(weights: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    """Return the V1 default `W = diag(w_sg^2)`."""

    w_arr = _as_1d_float_array(weights, "weights")
    return np.diag(np.square(w_arr))


def validate_psd_weight_matrix(weight_matrix: np.ndarray, *, tol: float = 1e-10) -> np.ndarray:
    """Validate and symmetrize a nonnegative PSD quadratic-form weight matrix."""

    matrix = _as_square_float_matrix(weight_matrix, "weight_matrix")
    eigvals = np.linalg.eigvalsh(matrix)
    if np.min(eigvals) < -tol:
        raise ValueError("weight_matrix must be positive semidefinite.")
    return matrix


def quadratic_statistic(z: np.ndarray | list[float] | tuple[float, ...], weight_matrix: np.ndarray) -> float:
    """Compute `Q = z^T W z`."""

    z_arr = _as_1d_float_array(z, "z")
    w_mat = validate_psd_weight_matrix(weight_matrix)
    if w_mat.shape[0] != z_arr.size:
        raise ValueError("weight_matrix dimensions must match z.")
    return float(z_arr @ w_mat @ z_arr)


def mixture_eigenvalues(
    ld: np.ndarray,
    weight_matrix: np.ndarray,
    *,
    shrinkage: float = DEFAULT_LD_SHRINKAGE,
    eigen_tol: float = 1e-12,
) -> np.ndarray:
    """Return eigenvalues of the LD-weighted quadratic-form null mixture.

    If `z ~ N(0, R*)`, then `z^T W z` has the same nonzero eigenvalues as
    `sqrt(R*) W sqrt(R*)`.
    """

    r_star = shrink_ld(ld, shrinkage=shrinkage)
    w_mat = validate_psd_weight_matrix(weight_matrix)
    if r_star.shape != w_mat.shape:
        raise ValueError("LD and weight_matrix dimensions must match.")

    r_vals, r_vecs = np.linalg.eigh(r_star)
    r_vals = np.clip(r_vals, 0.0, None)
    sqrt_r = (r_vecs * np.sqrt(r_vals)) @ r_vecs.T
    form = sqrt_r @ w_mat @ sqrt_r
    null_form = 0.5 * (form + form.T)
    eigvals = np.linalg.eigvalsh(null_form)
    eigvals = np.real(eigvals)
    eigvals = eigvals[eigvals > eigen_tol]
    if eigvals.size == 0:
        raise ValueError("quadratic null mixture has no positive eigenvalues.")
    return np.sort(eigvals)[::-1]


def clip_p_value(p_value: float, epsilon: float = 1e-15, upper_epsilon: float | None = None) -> tuple[float, bool]:
    """Clip a P value to finite transform bounds.

    The lower tail can require very small values such as ``1e-300``. The upper
    tail cannot use the same epsilon because ``1 - 1e-300`` rounds to ``1.0`` in
    double precision. When ``upper_epsilon`` is omitted, use at least ``1e-16``
    for the upper-tail distance from one.
    """

    if not np.isfinite(p_value):
        raise ValueError("p_value must be finite.")
    if not 0.0 <= p_value <= 1.0:
        raise ValueError("p_value must be in [0, 1].")
    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must be in (0, 0.5).")
    upper_eps = max(float(epsilon), 1e-16) if upper_epsilon is None else float(upper_epsilon)
    if not 0.0 < upper_eps < 0.5:
        raise ValueError("upper_epsilon must be in (0, 0.5).")
    lower = max(float(epsilon), np.nextafter(0.0, 1.0))
    upper = min(1.0 - upper_eps, np.nextafter(1.0, 0.0))
    clipped = min(max(float(p_value), lower), upper)
    return clipped, clipped != float(p_value)


def normal_score_from_p_value(
    p_value: float,
    epsilon: float = 1e-15,
    upper_epsilon: float | None = None,
) -> tuple[float, float, bool]:
    """Return `Phi^-1(1 - P_clip)` plus the clipped P value and clip flag."""

    clipped, was_clipped = clip_p_value(p_value, epsilon=epsilon, upper_epsilon=upper_epsilon)
    return float(stats.norm.isf(clipped)), clipped, was_clipped


def normal_scores_from_p_values(
    p_values: np.ndarray | list[float] | tuple[float, ...],
    epsilon: float = 1e-15,
    upper_epsilon: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized `Phi^-1(1 - P_clip)` for an array of P values."""

    p_arr = np.asarray(p_values, dtype=float)
    if not np.all(np.isfinite(p_arr)):
        raise ValueError("p_values must be finite.")
    if np.any((p_arr < 0.0) | (p_arr > 1.0)):
        raise ValueError("p_values must be in [0, 1].")
    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must be in (0, 0.5).")
    upper_eps = max(float(epsilon), 1e-16) if upper_epsilon is None else float(upper_epsilon)
    if not 0.0 < upper_eps < 0.5:
        raise ValueError("upper_epsilon must be in (0, 0.5).")
    lower = max(float(epsilon), np.nextafter(0.0, 1.0))
    upper = min(1.0 - upper_eps, np.nextafter(1.0, 0.0))
    clipped = np.clip(p_arr, lower, upper)
    scores = stats.norm.isf(clipped)
    return np.asarray(scores, dtype=float), clipped, clipped != p_arr


def _satterthwaite_sf(statistic: float, lambdas: np.ndarray) -> float:
    mean = float(np.sum(lambdas))
    second = float(np.sum(np.square(lambdas)))
    if mean <= 0.0 or second <= 0.0:
        raise ValueError("mixture eigenvalues must have positive first and second moments.")
    df = mean * mean / second
    scale = second / mean
    return float(stats.chi2.sf(statistic / scale, df=df))


def _satterthwaite_sf_array(statistic: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    mean = float(np.sum(lambdas))
    second = float(np.sum(np.square(lambdas)))
    if mean <= 0.0 or second <= 0.0:
        raise ValueError("mixture eigenvalues must have positive first and second moments.")
    df = mean * mean / second
    scale = second / mean
    return np.asarray(stats.chi2.sf(statistic / scale, df=df), dtype=float)


def _liu_sf(statistic: float, lambdas: np.ndarray) -> float:
    c1 = float(np.sum(lambdas))
    c2 = float(np.sum(lambdas**2))
    c3 = float(np.sum(lambdas**3))
    c4 = float(np.sum(lambdas**4))
    if c1 <= 0.0 or c2 <= 0.0 or c3 <= 0.0:
        raise ValueError("mixture eigenvalues must have positive moments.")

    s1 = c3 / (c2 ** 1.5)
    s2 = c4 / (c2**2)
    if s1 <= 0.0:
        return _satterthwaite_sf(statistic, lambdas)

    if s1 * s1 > s2:
        a = 1.0 / (s1 - np.sqrt(max(s1 * s1 - s2, 0.0)))
        delta = s1 * a**3 - a**2
        df = a**2 - 2.0 * delta
    else:
        a = 1.0 / s1
        delta = 0.0
        df = c2**3 / (c3**2)

    if df <= 0.0 or a <= 0.0 or delta < 0.0:
        return _satterthwaite_sf(statistic, lambdas)

    standardized_q = (statistic - c1) / np.sqrt(2.0 * c2)
    transformed_q = standardized_q * (np.sqrt(2.0) * a) + df + delta
    return float(stats.ncx2.sf(transformed_q, df=df, nc=delta))


def _liu_sf_array(statistic: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    c1 = float(np.sum(lambdas))
    c2 = float(np.sum(lambdas**2))
    c3 = float(np.sum(lambdas**3))
    c4 = float(np.sum(lambdas**4))
    if c1 <= 0.0 or c2 <= 0.0 or c3 <= 0.0:
        raise ValueError("mixture eigenvalues must have positive moments.")

    s1 = c3 / (c2 ** 1.5)
    s2 = c4 / (c2**2)
    if s1 <= 0.0:
        return _satterthwaite_sf_array(statistic, lambdas)

    if s1 * s1 > s2:
        a = 1.0 / (s1 - np.sqrt(max(s1 * s1 - s2, 0.0)))
        delta = s1 * a**3 - a**2
        df = a**2 - 2.0 * delta
    else:
        a = 1.0 / s1
        delta = 0.0
        df = c2**3 / (c3**2)

    if df <= 0.0 or a <= 0.0 or delta < 0.0:
        return _satterthwaite_sf_array(statistic, lambdas)

    standardized_q = (statistic - c1) / np.sqrt(2.0 * c2)
    transformed_q = standardized_q * (np.sqrt(2.0) * a) + df + delta
    return np.asarray(stats.ncx2.sf(transformed_q, df=df, nc=delta), dtype=float)


def _saddlepoint_sf(statistic: float, lambdas: np.ndarray) -> float:
    if statistic <= 0.0:
        return 1.0

    lambdas = np.asarray(lambdas, dtype=float)
    mean = float(np.sum(lambdas))
    if np.isclose(statistic, mean, rtol=1e-10, atol=1e-12):
        return 0.5

    max_lambda = float(np.max(lambdas))

    def k(t: float) -> float:
        return float(-0.5 * np.sum(np.log1p(-2.0 * t * lambdas)))

    def k_prime(t: float) -> float:
        return float(np.sum(lambdas / (1.0 - 2.0 * t * lambdas)))

    def k_second(t: float) -> float:
        denom = 1.0 - 2.0 * t * lambdas
        return float(2.0 * np.sum((lambdas**2) / (denom**2)))

    if statistic > mean:
        lower = 0.0
        upper = (1.0 / (2.0 * max_lambda)) * (1.0 - 1e-12)
    else:
        upper = 0.0
        lower = -1.0
        while k_prime(lower) > statistic:
            lower *= 2.0
            if lower < -1e12:
                return 1.0

    try:
        saddle_t = optimize.brentq(lambda t: k_prime(t) - statistic, lower, upper, maxiter=200)
    except ValueError:
        return _liu_sf(statistic, lambdas)

    signed = 1.0 if saddle_t > 0.0 else -1.0
    w_sq = max(2.0 * (saddle_t * statistic - k(saddle_t)), 0.0)
    w = signed * np.sqrt(w_sq)
    u = saddle_t * np.sqrt(k_second(saddle_t))

    if np.isclose(w, 0.0) or np.isclose(u, 0.0):
        return _liu_sf(statistic, lambdas)

    cdf = stats.norm.cdf(w) + stats.norm.pdf(w) * (1.0 / w - 1.0 / u)
    return float(np.clip(1.0 - cdf, 0.0, 1.0))


def quadratic_p_value(
    statistic: float,
    lambdas: np.ndarray | list[float] | tuple[float, ...],
    *,
    method: QuadraticPValueMethod = "saddlepoint",
) -> float:
    """Approximate `P(sum(lambda_j chi2_1) >= statistic)`.

    `davies` is intentionally explicit but not implemented in pure Python here.
    Use `saddlepoint` or `liu` for the current private V1 foundation.
    """

    if not np.isfinite(statistic) or statistic < 0.0:
        raise ValueError("statistic must be a finite nonnegative value.")
    lam = _as_1d_float_array(lambdas, "lambdas")
    lam = lam[lam > 0.0]
    if lam.size == 0:
        raise ValueError("lambdas must contain at least one positive value.")

    if method == "saddlepoint":
        p_value = _saddlepoint_sf(float(statistic), lam)
    elif method == "liu":
        p_value = _liu_sf(float(statistic), lam)
    elif method == "satterthwaite":
        p_value = _satterthwaite_sf(float(statistic), lam)
    elif method == "davies":
        raise NotImplementedError(
            "Davies exact mixture-chi-square p-values require an external implementation; "
            "use method='saddlepoint' or method='liu' in the V1 private prototype."
        )
    else:
        raise ValueError(f"Unknown quadratic p-value method: {method}")

    return float(np.clip(p_value, 0.0, 1.0))


def quadratic_p_values(
    statistics: np.ndarray | list[float] | tuple[float, ...],
    lambdas: np.ndarray | list[float] | tuple[float, ...],
    *,
    method: QuadraticPValueMethod = "saddlepoint",
) -> np.ndarray:
    """Vectorized mixture-chi-square upper-tail P values for fixed lambdas."""

    stats_arr = np.asarray(statistics, dtype=float)
    if not np.all(np.isfinite(stats_arr)) or np.any(stats_arr < 0.0):
        raise ValueError("statistics must contain finite nonnegative values.")
    lam = _as_1d_float_array(lambdas, "lambdas")
    lam = lam[lam > 0.0]
    if lam.size == 0:
        raise ValueError("lambdas must contain at least one positive value.")

    flat = stats_arr.reshape(-1)
    if method == "satterthwaite":
        p_values = _satterthwaite_sf_array(flat, lam)
    elif method == "liu":
        p_values = _liu_sf_array(flat, lam)
    elif method == "saddlepoint":
        p_values = np.asarray([_saddlepoint_sf(float(value), lam) for value in flat], dtype=float)
    elif method == "davies":
        raise NotImplementedError(
            "Davies exact mixture-chi-square p-values require an external implementation; "
            "use method='saddlepoint' or method='liu' in the V1 private prototype."
        )
    else:
        raise ValueError(f"Unknown quadratic p-value method: {method}")

    return np.clip(p_values.reshape(stats_arr.shape), 0.0, 1.0)


def quadratic_association(
    z: np.ndarray | list[float] | tuple[float, ...],
    ld: np.ndarray,
    *,
    weights: np.ndarray | list[float] | tuple[float, ...] | None = None,
    weight_matrix: np.ndarray | None = None,
    shrinkage: float = DEFAULT_LD_SHRINKAGE,
    method: QuadraticPValueMethod = "saddlepoint",
    p_clip_epsilon: float = 1e-15,
) -> QuadraticAssociationResult:
    """Compute the RIPPLE V1 unsigned gene association score."""

    z_arr = _as_1d_float_array(z, "z")
    if (weights is None) == (weight_matrix is None):
        raise ValueError("Provide exactly one of weights or weight_matrix.")

    if weight_matrix is None:
        w_mat = default_weight_matrix(weights if weights is not None else ())
    else:
        w_mat = validate_psd_weight_matrix(weight_matrix)

    if w_mat.shape[0] != z_arr.size:
        raise ValueError("weight dimensions must match z.")

    statistic = quadratic_statistic(z_arr, w_mat)
    lambdas = mixture_eigenvalues(ld, w_mat, shrinkage=shrinkage)
    p_value = quadratic_p_value(statistic, lambdas, method=method)
    normal_score, clipped_p, is_clipped = normal_score_from_p_value(
        p_value,
        epsilon=p_clip_epsilon,
    )

    return QuadraticAssociationResult(
        statistic=statistic,
        assoc_p_g=p_value,
        assoc_p_g_clipped=clipped_p,
        assoc_minuslog10p_g=float(-np.log10(clipped_p)),
        assoc_normal_score_g=normal_score,
        is_p_clipped=is_clipped,
        mixture_eigenvalues=lambdas,
        method=method,
        n_snps=int(z_arr.size),
        shrinkage=float(shrinkage),
    )
