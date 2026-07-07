"""Pre-specified graph-frequency bands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SpectralBand:
    """Closed-open graph-frequency band, except the final infinite upper bound."""

    label: str
    lower: float
    upper: float


DEFAULT_NORMALIZED_LAPLACIAN_BANDS: tuple[SpectralBand, ...] = (
    SpectralBand("low", 0.0, 0.25),
    SpectralBand("mid", 0.25, 1.0),
    SpectralBand("high", 1.0, 2.0000001),
)


def assign_frequency_bands(
    eigenvalues: Iterable[float],
    bands: tuple[SpectralBand, ...] = DEFAULT_NORMALIZED_LAPLACIAN_BANDS,
) -> pd.DataFrame:
    """Assign Laplacian eigenvalues to pre-specified frequency bands."""

    lambdas = np.asarray(tuple(eigenvalues), dtype=float)
    rows: list[dict[str, object]] = []
    for idx, value in enumerate(lambdas):
        if value < 0.0 and np.isclose(value, 0.0, atol=1e-10):
            value = 0.0
        label = "unassigned"
        for band in bands:
            if band.lower <= value < band.upper:
                label = band.label
                break
        rows.append({"eigen_index": idx, "eigenvalue": float(value), "band": label})
    return pd.DataFrame(rows, columns=["eigen_index", "eigenvalue", "band"])


def band_energy_table(
    eigenvalues: Iterable[float],
    coefficients: Iterable[float],
    bands: tuple[SpectralBand, ...] = DEFAULT_NORMALIZED_LAPLACIAN_BANDS,
) -> pd.DataFrame:
    """Summarize squared spectral coefficient energy by band."""

    lambdas = np.asarray(tuple(eigenvalues), dtype=float)
    coeffs = np.asarray(tuple(coefficients), dtype=float)
    if lambdas.shape != coeffs.shape:
        raise ValueError("eigenvalues and coefficients must have the same shape.")

    assignments = assign_frequency_bands(lambdas, bands)
    assignments["energy"] = np.square(coeffs)
    total = float(assignments["energy"].sum())
    rows = []
    for label, group in assignments.groupby("band", observed=True, sort=False):
        energy = float(group["energy"].sum())
        rows.append(
            {
                "band": label,
                "n_frequencies": int(len(group)),
                "energy": energy,
                "energy_fraction": float(energy / total) if total > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows, columns=["band", "n_frequencies", "energy", "energy_fraction"])
