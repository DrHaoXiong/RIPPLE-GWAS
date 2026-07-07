"""Signed LD-aware burden signal.

The V1 default signed stream is:

    x_g^dir = (w_g^T z_g) / sqrt(w_g^T R_g* w_g)
    R_g* = (1 - lambda) R_g + lambda I

This score is directional only when the input SNP Z scores and weights are
directionally harmonized. Positional weights should be interpreted as an
allelic signed burden, not expression activation/repression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ripple.defaults import DEFAULT_LD_SHRINKAGE


SignedStatus = Literal["available", "unstable"]


@dataclass(frozen=True)
class SignedBurdenResult:
    """Result for one gene-level signed burden calculation."""

    score: float
    numerator: float
    denominator_variance: float
    denominator: float
    n_snps: int
    shrinkage: float
    status: SignedStatus


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


def shrink_ld(ld: np.ndarray, shrinkage: float = DEFAULT_LD_SHRINKAGE) -> np.ndarray:
    """Return `R* = (1 - lambda)R + lambda I`."""

    ld = _as_square_float_matrix(ld, "LD matrix")
    if not 0 <= shrinkage <= 1:
        raise ValueError("LD shrinkage must be in [0, 1].")
    return (1.0 - shrinkage) * ld + shrinkage * np.eye(ld.shape[0])


def signed_ld_burden(
    z: np.ndarray | list[float] | tuple[float, ...],
    weights: np.ndarray | list[float] | tuple[float, ...],
    ld: np.ndarray,
    *,
    shrinkage: float = DEFAULT_LD_SHRINKAGE,
    denominator_tol: float = 1e-12,
) -> SignedBurdenResult:
    """Compute the RIPPLE V1 signed LD-aware gene burden score.

    Parameters
    ----------
    z:
        Allele-harmonized signed SNP Z-score vector for one gene.
    weights:
        SNP-to-gene weights aligned to `z`. Positional weights are usually
        nonnegative; eQTL/TWAS weights may be signed after harmonization.
    ld:
        SNP LD correlation matrix aligned to `z`.
    shrinkage:
        LD shrinkage `lambda` in `R* = (1 - lambda)R + lambda I`.
    denominator_tol:
        Minimum allowed `w^T R* w`. Values at or below this threshold are
        reported as unstable with `score = nan`.
    """

    z_arr = _as_1d_float_array(z, "z")
    w_arr = _as_1d_float_array(weights, "weights")
    if z_arr.shape != w_arr.shape:
        raise ValueError("z and weights must have the same length.")

    r_star = shrink_ld(ld, shrinkage=shrinkage)
    if r_star.shape[0] != z_arr.size:
        raise ValueError("LD matrix dimensions must match z and weights.")

    numerator = float(w_arr @ z_arr)
    denominator_variance = float(w_arr @ r_star @ w_arr)

    if denominator_variance <= denominator_tol:
        return SignedBurdenResult(
            score=float("nan"),
            numerator=numerator,
            denominator_variance=denominator_variance,
            denominator=float("nan"),
            n_snps=int(z_arr.size),
            shrinkage=float(shrinkage),
            status="unstable",
        )

    denominator = float(np.sqrt(denominator_variance))
    score = numerator / denominator
    return SignedBurdenResult(
        score=float(score),
        numerator=numerator,
        denominator_variance=denominator_variance,
        denominator=denominator,
        n_snps=int(z_arr.size),
        shrinkage=float(shrinkage),
        status="available",
    )
