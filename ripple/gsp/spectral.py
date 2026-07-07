"""Graph spectral endpoints for graph signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg


@dataclass(frozen=True)
class SpectralDecomposition:
    """Truncated or full Laplacian eigensystem."""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    nodes: tuple[str, ...]
    method: str


@dataclass(frozen=True)
class SpectralSignal:
    """Graph signal projected onto a Laplacian eigensystem."""

    nodes: tuple[str, ...]
    signal: np.ndarray
    eigenvalues: np.ndarray
    coefficients: np.ndarray
    energy: np.ndarray
    smoothness: float
    retained_energy_fraction: float


def laplacian_eigendecomposition(
    laplacian: sparse.spmatrix | np.ndarray,
    *,
    nodes: Iterable[str],
    n_components: int | None = 128,
    which: str = "SM",
) -> SpectralDecomposition:
    """Compute a full or truncated symmetric Laplacian eigendecomposition."""

    node_order = tuple(str(node) for node in nodes)
    matrix = sparse.csr_matrix(laplacian)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("laplacian must be square.")
    if matrix.shape[0] != len(node_order):
        raise ValueError("nodes length must match laplacian dimensions.")

    n = matrix.shape[0]
    if n == 0:
        return SpectralDecomposition(
            eigenvalues=np.array([], dtype=float),
            eigenvectors=np.empty((0, 0), dtype=float),
            nodes=node_order,
            method="empty",
        )

    if n_components is None or n_components >= n - 1:
        dense = matrix.toarray()
        values, vectors = np.linalg.eigh(0.5 * (dense + dense.T))
        method = "full"
    else:
        k = max(1, int(n_components))
        values, vectors = sparse_linalg.eigsh(matrix, k=k, which=which)
        order = np.argsort(values)
        values = values[order]
        vectors = vectors[:, order]
        method = f"eigsh_{which}_{k}"

    values = np.real(values)
    vectors = np.real(vectors)
    return SpectralDecomposition(
        eigenvalues=values.astype(float),
        eigenvectors=vectors.astype(float),
        nodes=node_order,
        method=method,
    )


def project_graph_signal(
    signal: Iterable[float],
    decomposition: SpectralDecomposition,
    *,
    laplacian: sparse.spmatrix | np.ndarray | None = None,
) -> SpectralSignal:
    """Project a node-aligned signal onto graph Fourier components."""

    x = np.asarray(tuple(signal), dtype=float)
    if x.ndim != 1:
        raise ValueError("signal must be one-dimensional.")
    if x.size != len(decomposition.nodes):
        raise ValueError("signal length must match decomposition nodes.")
    if not np.all(np.isfinite(x)):
        raise ValueError("signal must contain only finite values.")

    centered = x - float(np.mean(x))
    coefficients = decomposition.eigenvectors.T @ centered
    energy = np.square(coefficients)
    total_signal_energy = float(np.sum(np.square(centered)))
    retained_energy = float(np.sum(energy))
    retained_fraction = retained_energy / total_signal_energy if total_signal_energy > 0 else 0.0

    if laplacian is None:
        smoothness = float(np.sum(decomposition.eigenvalues * energy))
    else:
        matrix = sparse.csr_matrix(laplacian)
        smoothness = float(centered @ matrix @ centered)

    return SpectralSignal(
        nodes=decomposition.nodes,
        signal=centered.astype(float),
        eigenvalues=decomposition.eigenvalues,
        coefficients=coefficients.astype(float),
        energy=energy.astype(float),
        smoothness=smoothness,
        retained_energy_fraction=float(retained_fraction),
    )
