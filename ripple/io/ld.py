"""LD reference IO and per-gene LD matrix retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PlinkReference:
    """PLINK binary reference prefix and sidecar paths."""

    prefix: Path
    bed: Path
    bim: Path
    fam: Path


@dataclass(frozen=True)
class LDMatrix:
    """LD matrix aligned to `snp_ids`."""

    matrix: np.ndarray
    snp_ids: tuple[str, ...]


@dataclass(frozen=True)
class LDSummary:
    """Compact LD-derived technical covariates."""

    m_eff: float
    local_ld_score: float


def resolve_plink_reference(prefix: str | Path) -> PlinkReference:
    """Resolve and validate a PLINK binary reference prefix."""

    prefix = Path(prefix)
    if prefix.suffix in {".bed", ".bim", ".fam"}:
        prefix = prefix.with_suffix("")
    reference = PlinkReference(
        prefix=prefix,
        bed=prefix.with_suffix(".bed"),
        bim=prefix.with_suffix(".bim"),
        fam=prefix.with_suffix(".fam"),
    )
    missing = [str(path) for path in (reference.bed, reference.bim, reference.fam) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing PLINK reference files: {missing}")
    return reference


def read_bim(path_or_prefix: str | Path) -> pd.DataFrame:
    """Read a PLINK `.bim` file or binary reference prefix."""

    path = Path(path_or_prefix)
    if path.suffix != ".bim":
        path = path.with_suffix(".bim")
    if not path.exists():
        raise FileNotFoundError(path)

    columns = ["chrom", "snp_id", "cm", "pos", "allele1", "allele2"]
    out = pd.read_csv(path, sep=r"\s+", header=None, names=columns, dtype={"chrom": str})
    out["snp_id"] = out["snp_id"].astype(str)
    out["pos"] = pd.to_numeric(out["pos"], errors="raise").astype(int)
    return out


def read_fam(path_or_prefix: str | Path) -> pd.DataFrame:
    """Read a PLINK `.fam` file or binary reference prefix."""

    path = Path(path_or_prefix)
    if path.suffix != ".fam":
        path = path.with_suffix(".fam")
    if not path.exists():
        raise FileNotFoundError(path)

    columns = ["family_id", "individual_id", "paternal_id", "maternal_id", "sex", "phenotype"]
    return pd.read_csv(path, sep=r"\s+", header=None, names=columns, dtype=str)


def filter_bim_to_snps(
    bim: pd.DataFrame,
    snp_ids: Iterable[str],
    *,
    preserve_input_order: bool = True,
) -> pd.DataFrame:
    """Subset BIM metadata to requested SNP IDs."""

    requested = tuple(str(snp) for snp in snp_ids)
    if not requested:
        raise ValueError("snp_ids must not be empty.")

    indexed = bim.drop_duplicates("snp_id").set_index("snp_id", drop=False)
    present = [snp for snp in requested if snp in indexed.index]
    if preserve_input_order:
        return indexed.loc[present].reset_index(drop=True)
    return bim.loc[bim["snp_id"].astype(str).isin(set(requested))].reset_index(drop=True)


def align_ld_matrix(
    matrix: np.ndarray,
    matrix_snp_ids: Iterable[str],
    target_snp_ids: Iterable[str],
    *,
    allow_missing: bool = False,
) -> LDMatrix:
    """Align an LD matrix from `matrix_snp_ids` order to `target_snp_ids` order."""

    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("LD matrix must be square.")

    source = tuple(str(snp) for snp in matrix_snp_ids)
    target = tuple(str(snp) for snp in target_snp_ids)
    if len(source) != arr.shape[0]:
        raise ValueError("matrix_snp_ids length must match LD matrix dimensions.")
    if not target:
        raise ValueError("target_snp_ids must not be empty.")

    source_lookup = {snp: i for i, snp in enumerate(source)}
    missing = [snp for snp in target if snp not in source_lookup]
    if missing and not allow_missing:
        raise KeyError(f"SNPs missing from LD matrix: {missing}")

    kept = [snp for snp in target if snp in source_lookup]
    indices = [source_lookup[snp] for snp in kept]
    aligned = arr[np.ix_(indices, indices)]
    aligned = 0.5 * (aligned + aligned.T)
    np.fill_diagonal(aligned, 1.0)
    return LDMatrix(matrix=aligned, snp_ids=tuple(kept))


def read_square_ld_matrix(path: str | Path, snp_ids: Iterable[str]) -> LDMatrix:
    """Read a plain-text square LD matrix and attach SNP IDs."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    matrix = np.loadtxt(path, dtype=float)
    snp_tuple = tuple(str(snp) for snp in snp_ids)
    if matrix.shape != (len(snp_tuple), len(snp_tuple)):
        raise ValueError("LD matrix shape does not match snp_ids length.")
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 1.0)
    return LDMatrix(matrix=matrix, snp_ids=snp_tuple)


def correlation_matrix_from_genotypes(genotypes: np.ndarray) -> np.ndarray:
    """Compute a SNP-by-SNP correlation matrix from genotype dosages.

    Input is shaped `(n_samples, n_snps)`. Missing values are mean-imputed per
    SNP. Monomorphic SNPs receive zero off-diagonal correlation and unit
    diagonal.
    """

    x = np.asarray(genotypes, dtype=float)
    if x.ndim != 2:
        raise ValueError("genotypes must be a two-dimensional array.")
    if x.shape[1] == 0:
        raise ValueError("genotypes must contain at least one SNP.")
    if x.shape[1] == 1:
        return np.ones((1, 1), dtype=float)

    means = np.nanmean(x, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    missing = np.isnan(x)
    if missing.any():
        x = x.copy()
        row_idx, col_idx = np.where(missing)
        x[row_idx, col_idx] = means[col_idx]

    centered = x - np.mean(x, axis=0)
    scale = np.std(centered, axis=0, ddof=1)
    informative = scale > 0
    z = np.zeros_like(centered, dtype=float)
    z[:, informative] = centered[:, informative] / scale[informative]
    denom = max(x.shape[0] - 1, 1)
    corr = (z.T @ z) / denom
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)
    return corr.astype(float)


def summarize_ld_matrix(ld: np.ndarray) -> LDSummary:
    """Return participation-ratio M_eff and mean LD score for an LD matrix."""

    matrix = np.asarray(ld, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("LD matrix must be square.")
    if matrix.shape[0] == 0:
        raise ValueError("LD matrix must not be empty.")
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 1.0)
    trace = float(np.trace(matrix))
    fro_sq = float(np.sum(np.square(matrix)))
    m_eff = (trace * trace / fro_sq) if fro_sq > 0 else 1.0
    local_ld_score = float(np.mean(np.sum(np.square(matrix), axis=1)))
    return LDSummary(m_eff=float(m_eff), local_ld_score=local_ld_score)


def write_snp_list(path: str | Path, snp_ids: Iterable[str]) -> Path:
    """Write one SNP ID per line."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = tuple(str(snp) for snp in snp_ids)
    if not ids:
        raise ValueError("snp_ids must not be empty.")
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    return path


def compute_plink_ld_matrix(
    bfile_prefix: str | Path,
    snp_ids: Iterable[str],
    *,
    plink_executable: str = "plink",
    out_dir: str | Path | None = None,
    keep_intermediate: bool = False,
) -> LDMatrix:
    """Compute a square LD matrix using PLINK 1.9 `--r square`.

    The returned matrix is aligned to the SNP order retained from the reference
    `.bim` file after applying the requested SNP set. Missing requested SNPs are
    silently absent from the returned `LDMatrix.snp_ids`; callers should compare
    requested and returned IDs when strict completeness is required.
    """

    reference = resolve_plink_reference(bfile_prefix)
    requested = tuple(str(snp) for snp in snp_ids)
    if not requested:
        raise ValueError("snp_ids must not be empty.")

    bim = read_bim(reference.prefix)
    retained_bim = filter_bim_to_snps(bim, requested, preserve_input_order=False)
    retained_ids = tuple(retained_bim["snp_id"].astype(str))
    if not retained_ids:
        raise ValueError("None of the requested SNPs are present in the PLINK reference.")

    if out_dir is None:
        temp_context = tempfile.TemporaryDirectory()
        work_dir = Path(temp_context.name)
    else:
        temp_context = None
        work_dir = Path(out_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    try:
        snplist = write_snp_list(work_dir / "extract.snplist", retained_ids)
        out_prefix = work_dir / "plink_ld"
        command = [
            plink_executable,
            "--bfile",
            str(reference.prefix),
            "--extract",
            str(snplist),
            "--r",
            "square",
            "--out",
            str(out_prefix),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(
                "PLINK LD computation failed with exit code "
                f"{completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )

        ld_path = out_prefix.with_suffix(".ld")
        if not ld_path.exists():
            raise FileNotFoundError(f"Expected PLINK LD output not found: {ld_path}")
        result = read_square_ld_matrix(ld_path, retained_ids)

        if keep_intermediate and out_dir is None:
            raise ValueError("keep_intermediate=True requires out_dir.")
        return result
    finally:
        if temp_context is not None:
            temp_context.cleanup()
