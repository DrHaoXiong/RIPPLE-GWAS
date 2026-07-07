#!/usr/bin/env python
"""Run HEIGHT_IRN pilot with per-gene 1000G EUR LD cache and null replicates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ripple.defaults import DEFAULT_LD_SHRINKAGE, RANK_FRACTION_GRID  # noqa: E402
from ripple.graph import graph_laplacian  # noqa: E402
from ripple.gsp import band_energy_table, laplacian_eigendecomposition, project_graph_signal  # noqa: E402
from ripple.io.annotations import read_magma_gene_loc  # noqa: E402
from ripple.mapping.weights import add_positional_weights  # noqa: E402
from ripple.nulls import (  # noqa: E402
    degree_preserving_graph_replicates,
    degree_stratified_permuted_scores,
    graph_component_summary,
)
from ripple.percolation import (  # noqa: E402
    classify_percolation_architecture,
    compute_degree_matched_node_percolation_null,
    percolation_auc,
    percolation_curve,
    prepare_degree_matched_rank_sets,
    rank_nodes_by_score,
    summarize_percolation_null,
)
from ripple.signals.residualize import append_residualized_score  # noqa: E402
from ripple.signals.safety import append_safety_columns, summarize_clipping  # noqa: E402
from ripple.signals.signed import shrink_ld  # noqa: E402
from ripple.signals.unsigned import (  # noqa: E402
    normal_score_from_p_value,
    normal_scores_from_p_values,
    quadratic_p_value,
    quadratic_p_values,
)
from run_height_mvp import (  # noqa: E402
    DEFAULT_GENE_LOC,
    DEFAULT_GWAS,
    DEFAULT_STRING_INFO,
    DEFAULT_STRING_LINKS,
    build_string_graph,
    fast_positional_map_by_gene,
    load_height_gwas,
    write_table,
)


PRIVATE_ROOT = Path("/path/to/ripple_private_workspace")
DEFAULT_OUT_DIR = PRIVATE_ROOT / "30_analysis" / "height_irn_ld_null_mvp"
DEFAULT_MAPPING = (
    PRIVATE_ROOT
    / "30_analysis"
    / "height_irn_mvp"
    / "tables"
    / "HEIGHT_IRN.gene_body_mapping.tsv.gz"
)
DEFAULT_LD_CACHE = PRIVATE_ROOT / "30_analysis" / "height_irn_mvp" / "ld_cache_1000G_EUR"
LD_SCORING_PAYLOAD_SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gwas", type=Path, default=DEFAULT_GWAS)
    parser.add_argument("--gene-loc", type=Path, default=DEFAULT_GENE_LOC)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--ld-cache-dir", type=Path, default=DEFAULT_LD_CACHE)
    parser.add_argument("--ld-cache-overlay-dir", type=Path, action="append", default=[])
    parser.add_argument("--ld-score-payload-cache", type=Path, default=None)
    parser.add_argument("--string-links", type=Path, default=DEFAULT_STRING_LINKS)
    parser.add_argument("--string-info", type=Path, default=DEFAULT_STRING_INFO)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--upstream-bp", type=int, default=0)
    parser.add_argument("--downstream-bp", type=int, default=0)
    parser.add_argument("--string-min-score", type=int, default=400)
    parser.add_argument("--eigen-components", type=int, default=128)
    parser.add_argument("--n-null", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--quadratic-method", choices=["liu", "satterthwaite"], default="liu")
    parser.add_argument("--p-clip-epsilon", type=float, default=1e-300)
    parser.add_argument("--ld-shrinkage", type=float, default=DEFAULT_LD_SHRINKAGE)
    parser.add_argument("--allow-identity-fallback", action="store_true", default=True)
    parser.add_argument("--n-degree-stratified-null", type=int, default=0)
    parser.add_argument("--degree-stratified-bins", type=int, default=10)
    parser.add_argument("--n-degree-matched-node-null", type=int, default=0)
    parser.add_argument("--degree-matched-bins", type=int, default=10)
    parser.add_argument("--n-degree-graph-null", type=int, default=0)
    parser.add_argument("--degree-graph-nswap-per-edge", type=float, default=1.0)
    parser.add_argument("--degree-graph-max-tries-per-swap", type=float, default=20.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_outputs(out_dir: Path, *, force: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        raise FileExistsError(f"{out_dir} exists and is not empty. Use --force to overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)


def ld_cache_path(cache_dir: Path, gene_id: str) -> Path:
    return cache_dir / "matrices" / f"{gene_id}.ld.npz"


def load_single_ld_entry(cache_dir: Path, gene_id: str) -> dict[str, object]:
    path = ld_cache_path(cache_dir, gene_id)
    if not path.exists():
        return {"status": "missing_cache", "path": str(path)}
    with np.load(path, allow_pickle=True) as data:
        return {
            "gene_id": str(data["gene_id"]),
            "gene_symbol": str(data["gene_symbol"]),
            "snp_ids": tuple(str(x) for x in data["snp_ids"]),
            "ld": np.asarray(data["ld"], dtype=float),
            "m_eff": float(data["m_eff"]),
            "local_ld_score": float(data["local_ld_score"]),
            "status": str(data["status"]),
            "missing_snps": tuple(str(x) for x in data["missing_snps"]),
            "path": str(path),
        }


def load_ld_entry(cache_dirs: Sequence[Path], gene_id: str) -> dict[str, object]:
    """Load a gene LD entry, preferring computed entries from earlier cache dirs."""

    first_existing: dict[str, object] | None = None
    missing_paths: list[str] = []
    for cache_dir in cache_dirs:
        entry = load_single_ld_entry(cache_dir, gene_id)
        if entry["status"] == "missing_cache":
            missing_paths.append(str(entry["path"]))
            continue
        if first_existing is None:
            first_existing = entry
        if entry["status"] == "computed":
            return entry

    if first_existing is not None:
        return first_existing
    return {"status": "missing_cache", "path": ";".join(missing_paths)}


def weighted_ld_mixture_eigenvalues(
    ld: np.ndarray,
    weights: np.ndarray,
    *,
    shrinkage: float,
    eigen_tol: float = 1e-12,
) -> np.ndarray:
    r_star = shrink_ld(ld, shrinkage=shrinkage)
    abs_w = np.abs(weights)
    form = (abs_w[:, None] * r_star) * abs_w[None, :]
    eigvals = np.linalg.eigvalsh(0.5 * (form + form.T))
    eigvals = eigvals[eigvals > eigen_tol]
    if eigvals.size == 0:
        return np.array([float(np.sum(np.square(weights)))])
    return np.sort(eigvals)[::-1]


def p_to_normal(p_value: float, epsilon: float) -> tuple[float, float, bool]:
    score, clipped, was_clipped = normal_score_from_p_value(p_value, epsilon=epsilon)
    if not np.isfinite(score):
        clipped = min(max(clipped, epsilon), np.nextafter(1.0, 0.0))
        score = float(stats.norm.isf(clipped))
        was_clipped = True
    return score, clipped, was_clipped


def build_null_z_matrix(z: np.ndarray, *, n_null: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((n_null, z.size), dtype=np.float32)
    for idx in range(n_null):
        out[idx, :] = z[rng.permutation(z.size)].astype(np.float32)
    return out


def build_null_z_matrix_for_snps(
    z: np.ndarray,
    *,
    n_target_snps: int,
    n_null: int,
    seed: int,
) -> np.ndarray:
    """Build SNP-permutation null Z values only for SNPs used downstream."""

    if n_target_snps < 0:
        raise ValueError("n_target_snps must be nonnegative.")
    if n_target_snps > z.size:
        raise ValueError("n_target_snps cannot exceed the GWAS SNP count.")
    rng = np.random.default_rng(seed)
    out = np.empty((n_null, n_target_snps), dtype=np.float32)
    for idx in range(n_null):
        sampled = rng.choice(z.size, size=n_target_snps, replace=False)
        out[idx, :] = z[sampled].astype(np.float32)
    return out


def mapping_cache_signature(mapping: pd.DataFrame) -> str:
    """Return a stable signature for SNP-to-gene weights used by LD scoring."""

    columns = ["gene_id", "gene_symbol", "chrom", "gene_start", "gene_end", "snp_id", "weight"]
    missing = [col for col in columns if col not in mapping.columns]
    if missing:
        raise ValueError(f"mapping is missing required columns for payload cache: {missing}")
    work = mapping.loc[:, columns].copy()
    for col in ["gene_id", "gene_symbol", "chrom", "snp_id"]:
        work[col] = work[col].astype(str)
    for col in ["gene_start", "gene_end", "weight"]:
        work[col] = pd.to_numeric(work[col], errors="raise")
    work = work.sort_values(columns, kind="mergesort").reset_index(drop=True)
    hashed = pd.util.hash_pandas_object(work, index=False).to_numpy(dtype=np.uint64)
    digest = hashlib.sha256()
    digest.update("|".join(columns).encode("utf-8"))
    digest.update(hashed.tobytes())
    return digest.hexdigest()


def ld_cache_dirs_signature(ld_cache_dirs: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for cache_dir in ld_cache_dirs:
        digest.update(str(Path(cache_dir)).encode("utf-8"))
        manifest = Path(cache_dir) / "manifest.tsv"
        if manifest.exists():
            stat = manifest.stat()
            digest.update(str(stat.st_size).encode("utf-8"))
            digest.update(str(int(stat.st_mtime_ns)).encode("utf-8"))
    return digest.hexdigest()


def string_sequence_signature(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def build_ld_scoring_payloads(
    mapping: pd.DataFrame,
    *,
    snp_to_idx: dict[str, int],
    snp_to_null_idx: dict[str, int],
    ld_cache_dirs: Sequence[Path],
    shrinkage: float,
    allow_identity_fallback: bool,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Precompute trait-invariant LD scoring payloads for each mapped gene."""

    payloads: list[dict[str, object]] = []
    status_counts: dict[str, int] = {}
    grouped = list(mapping.groupby(["gene_id", "gene_symbol", "chrom", "gene_start", "gene_end"], observed=True, sort=True))

    for counter, ((gene_id, gene_symbol, chrom, gene_start, gene_end), group) in enumerate(grouped, start=1):
        gene_id = str(gene_id)
        weights = group.groupby("snp_id", observed=True, sort=False)["weight"].sum()
        weights.index = weights.index.astype(str)
        entry = load_ld_entry(ld_cache_dirs, gene_id)
        ld_status = str(entry["status"])

        if ld_status == "computed":
            cache_snps = [
                snp
                for snp in entry["snp_ids"]
                if snp in weights.index and snp in snp_to_idx and snp in snp_to_null_idx
            ]
            if not cache_snps:
                ld_status = "identity_fallback_no_cache_overlap"
            else:
                snp_ids = tuple(cache_snps)
                ld = np.asarray(entry["ld"], dtype=float)
                cache_lookup = {snp: idx for idx, snp in enumerate(entry["snp_ids"])}
                keep = [cache_lookup[snp] for snp in snp_ids]
                ld = ld[np.ix_(keep, keep)]
                m_eff = float(entry["m_eff"])
                local_ld_score = float(entry["local_ld_score"])
        if ld_status != "computed":
            if not allow_identity_fallback:
                raise RuntimeError(f"LD cache unavailable for {gene_id}: {ld_status}")
            snp_ids = tuple(snp for snp in weights.index if snp in snp_to_idx and snp in snp_to_null_idx)
            ld = np.eye(len(snp_ids), dtype=float)
            m_eff = float(len(snp_ids))
            local_ld_score = 1.0
            ld_status = f"identity_fallback_{ld_status}"

        status_counts[ld_status] = status_counts.get(ld_status, 0) + 1
        if not snp_ids:
            continue

        w = weights.loc[list(snp_ids)].to_numpy(dtype=float)
        r_star = shrink_ld(ld, shrinkage=shrinkage)
        payloads.append(
            {
                "gene_id": gene_id,
                "gene_symbol": str(gene_symbol),
                "chrom": str(chrom),
                "gene_start": int(gene_start),
                "gene_end": int(gene_end),
                "snp_ids": tuple(str(snp) for snp in snp_ids),
                "weights": w,
                "denominator_variance": float(w @ r_star @ w),
                "lambdas": weighted_ld_mixture_eigenvalues(ld, w, shrinkage=shrinkage),
                "m_eff": m_eff,
                "local_ld_score": local_ld_score,
                "ld_status": ld_status,
                "ld_cache_path": str(entry["path"]),
            }
        )
        if counter % 1000 == 0 or counter == len(grouped):
            print(f"Prepared LD scoring payloads for {counter:,}/{len(grouped):,} genes", flush=True)
    return payloads, status_counts


def write_ld_scoring_payload_cache(path: Path, payloads: list[dict[str, object]], metadata: dict[str, object]) -> None:
    """Write compact reusable LD scoring payloads."""

    path.parent.mkdir(parents=True, exist_ok=True)
    snp_offsets = [0]
    lambda_offsets = [0]
    snp_flat: list[str] = []
    weights_flat: list[float] = []
    lambda_flat: list[float] = []
    for payload in payloads:
        snps = tuple(str(snp) for snp in payload["snp_ids"])
        weights = np.asarray(payload["weights"], dtype=float)
        lambdas = np.asarray(payload["lambdas"], dtype=float)
        snp_flat.extend(snps)
        weights_flat.extend(float(value) for value in weights)
        lambda_flat.extend(float(value) for value in lambdas)
        snp_offsets.append(len(snp_flat))
        lambda_offsets.append(len(lambda_flat))

    np.savez_compressed(
        path,
        schema_version=np.array(LD_SCORING_PAYLOAD_SCHEMA_VERSION, dtype=np.int16),
        metadata_json=np.array(json.dumps(metadata, sort_keys=True), dtype=object),
        gene_ids=np.array([payload["gene_id"] for payload in payloads], dtype=object),
        gene_symbols=np.array([payload["gene_symbol"] for payload in payloads], dtype=object),
        chroms=np.array([payload["chrom"] for payload in payloads], dtype=object),
        gene_starts=np.array([payload["gene_start"] for payload in payloads], dtype=np.int64),
        gene_ends=np.array([payload["gene_end"] for payload in payloads], dtype=np.int64),
        denominator_variances=np.array([payload["denominator_variance"] for payload in payloads], dtype=float),
        m_eff=np.array([payload["m_eff"] for payload in payloads], dtype=float),
        local_ld_scores=np.array([payload["local_ld_score"] for payload in payloads], dtype=float),
        ld_statuses=np.array([payload["ld_status"] for payload in payloads], dtype=object),
        ld_cache_paths=np.array([payload["ld_cache_path"] for payload in payloads], dtype=object),
        snp_offsets=np.array(snp_offsets, dtype=np.int64),
        snp_ids=np.array(snp_flat, dtype=object),
        weights=np.array(weights_flat, dtype=float),
        lambda_offsets=np.array(lambda_offsets, dtype=np.int64),
        lambdas=np.array(lambda_flat, dtype=float),
    )


def read_ld_scoring_payload_cache(path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Load reusable LD scoring payloads written by `write_ld_scoring_payload_cache`."""

    with np.load(path, allow_pickle=True) as data:
        schema_version = int(np.asarray(data["schema_version"]).item())
        if schema_version != LD_SCORING_PAYLOAD_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported LD scoring payload schema version {schema_version}; "
                f"expected {LD_SCORING_PAYLOAD_SCHEMA_VERSION}."
            )
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        gene_ids = np.asarray(data["gene_ids"], dtype=object)
        gene_symbols = np.asarray(data["gene_symbols"], dtype=object)
        chroms = np.asarray(data["chroms"], dtype=object)
        gene_starts = np.asarray(data["gene_starts"], dtype=np.int64)
        gene_ends = np.asarray(data["gene_ends"], dtype=np.int64)
        denominator_variances = np.asarray(data["denominator_variances"], dtype=float)
        m_eff = np.asarray(data["m_eff"], dtype=float)
        local_ld_scores = np.asarray(data["local_ld_scores"], dtype=float)
        ld_statuses = np.asarray(data["ld_statuses"], dtype=object)
        ld_cache_paths = np.asarray(data["ld_cache_paths"], dtype=object)
        snp_offsets = np.asarray(data["snp_offsets"], dtype=np.int64)
        snp_ids = np.asarray(data["snp_ids"], dtype=object)
        weights = np.asarray(data["weights"], dtype=float)
        lambda_offsets = np.asarray(data["lambda_offsets"], dtype=np.int64)
        lambdas = np.asarray(data["lambdas"], dtype=float)

    payloads: list[dict[str, object]] = []
    for idx, gene_id in enumerate(gene_ids):
        snp_start, snp_end = int(snp_offsets[idx]), int(snp_offsets[idx + 1])
        lambda_start, lambda_end = int(lambda_offsets[idx]), int(lambda_offsets[idx + 1])
        payloads.append(
            {
                "gene_id": str(gene_id),
                "gene_symbol": str(gene_symbols[idx]),
                "chrom": str(chroms[idx]),
                "gene_start": int(gene_starts[idx]),
                "gene_end": int(gene_ends[idx]),
                "snp_ids": tuple(str(snp) for snp in snp_ids[snp_start:snp_end]),
                "weights": weights[snp_start:snp_end].astype(float, copy=True),
                "denominator_variance": float(denominator_variances[idx]),
                "lambdas": lambdas[lambda_start:lambda_end].astype(float, copy=True),
                "m_eff": float(m_eff[idx]),
                "local_ld_score": float(local_ld_scores[idx]),
                "ld_status": str(ld_statuses[idx]),
                "ld_cache_path": str(ld_cache_paths[idx]),
            }
        )
    return payloads, metadata


def payload_cache_is_compatible(
    metadata: dict[str, object],
    *,
    mapping_signature: str,
    mapped_snp_ids_signature: str,
    ld_dirs_signature: str,
    shrinkage: float,
    allow_identity_fallback: bool,
) -> bool:
    if int(metadata.get("schema_version", -1)) != LD_SCORING_PAYLOAD_SCHEMA_VERSION:
        return False
    if metadata.get("mapping_signature") != mapping_signature:
        return False
    if metadata.get("ld_cache_dirs_signature") != ld_dirs_signature:
        return False
    if metadata.get("mapped_snp_ids_signature") != mapped_snp_ids_signature:
        return False
    if not np.isclose(float(metadata.get("ld_shrinkage", np.nan)), float(shrinkage)):
        return False
    if not allow_identity_fallback and bool(metadata.get("has_identity_fallback", False)):
        return False
    return True


def compute_ld_cached_gene_scores(
    gwas: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    ld_cache_dirs: Sequence[Path],
    n_null: int,
    seed: int,
    method: str,
    p_clip_epsilon: float,
    shrinkage: float,
    allow_identity_fallback: bool,
    payload_cache_path: Path | None = None,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, object]]:
    """Compute observed and SNP-permutation null gene scores with cached LD."""

    gwas_indexed = gwas.drop_duplicates("snp_id").reset_index(drop=True)
    snp_to_idx = {snp: idx for idx, snp in enumerate(gwas_indexed["snp_id"].astype(str))}
    z_values = gwas_indexed["z"].to_numpy(dtype=float)
    mapped_snp_ids = [
        snp
        for snp in mapping["snp_id"].dropna().astype(str).drop_duplicates()
        if snp in snp_to_idx
    ]
    snp_to_null_idx = {snp: idx for idx, snp in enumerate(mapped_snp_ids)}
    mapped_snp_ids_sig = string_sequence_signature(mapped_snp_ids)
    null_z = (
        build_null_z_matrix_for_snps(
            z_values,
            n_target_snps=len(mapped_snp_ids),
            n_null=n_null,
            seed=seed,
        )
        if n_null > 0
        else np.empty((0, len(mapped_snp_ids)), dtype=np.float32)
    )

    mapping_sig = mapping_cache_signature(mapping)
    ld_dirs_sig = ld_cache_dirs_signature(ld_cache_dirs)
    payload_cache_report: dict[str, object] = {
        "enabled": payload_cache_path is not None,
        "path": str(payload_cache_path) if payload_cache_path is not None else None,
        "status": "disabled",
    }
    payloads: list[dict[str, object]] | None = None
    status_counts: dict[str, int] = {}
    if payload_cache_path is not None and payload_cache_path.exists():
        try:
            cached_payloads, metadata = read_ld_scoring_payload_cache(payload_cache_path)
            if payload_cache_is_compatible(
                metadata,
                mapping_signature=mapping_sig,
                mapped_snp_ids_signature=mapped_snp_ids_sig,
                ld_dirs_signature=ld_dirs_sig,
                shrinkage=shrinkage,
                allow_identity_fallback=allow_identity_fallback,
            ):
                payloads = cached_payloads
                payload_cache_report.update(
                    {
                        "status": "loaded",
                        "n_payloads": int(len(payloads)),
                    }
                )
                print(f"Loaded LD scoring payload cache: {payload_cache_path}", flush=True)
            else:
                payload_cache_report["status"] = "ignored_incompatible"
                print(f"Ignoring incompatible LD scoring payload cache: {payload_cache_path}", flush=True)
        except Exception as exc:  # pragma: no cover - defensive cache fallback
            payload_cache_report.update({"status": "ignored_read_error", "error": str(exc)})
            print(f"Ignoring unreadable LD scoring payload cache {payload_cache_path}: {exc}", flush=True)

    if payloads is None:
        payloads, status_counts = build_ld_scoring_payloads(
            mapping,
            snp_to_idx=snp_to_idx,
            snp_to_null_idx=snp_to_null_idx,
            ld_cache_dirs=ld_cache_dirs,
            shrinkage=shrinkage,
            allow_identity_fallback=allow_identity_fallback,
        )
        payload_cache_report.setdefault("status", "built")
        if payload_cache_report["status"] in {"disabled", "ignored_incompatible", "ignored_read_error"}:
            payload_cache_report["status"] = "built"
        if payload_cache_path is not None:
            metadata = {
                "schema_version": LD_SCORING_PAYLOAD_SCHEMA_VERSION,
                "mapping_signature": mapping_sig,
                "mapped_snp_ids_signature": mapped_snp_ids_sig,
                "ld_cache_dirs_signature": ld_dirs_sig,
                "ld_cache_dirs": [str(path) for path in ld_cache_dirs],
                "ld_shrinkage": float(shrinkage),
                "allow_identity_fallback": bool(allow_identity_fallback),
                "has_identity_fallback": any(str(payload["ld_status"]).startswith("identity_fallback") for payload in payloads),
                "n_payloads": int(len(payloads)),
                "n_mapped_snps_for_null_matrix": int(len(mapped_snp_ids)),
            }
            write_ld_scoring_payload_cache(payload_cache_path, payloads, metadata)
            payload_cache_report.update({"status": "built_and_written", "n_payloads": int(len(payloads))})
            print(f"Wrote LD scoring payload cache: {payload_cache_path}", flush=True)
    else:
        for payload in payloads:
            ld_status = str(payload["ld_status"])
            status_counts[ld_status] = status_counts.get(ld_status, 0) + 1

    rows: list[dict[str, object]] = []
    null_score_rows: list[np.ndarray] = []

    for counter, payload in enumerate(payloads, start=1):
        gene_id = str(payload["gene_id"])
        gene_symbol = str(payload["gene_symbol"])
        chrom = str(payload["chrom"])
        gene_start = int(payload["gene_start"])
        gene_end = int(payload["gene_end"])
        snp_ids = tuple(str(snp) for snp in payload["snp_ids"])
        w = np.asarray(payload["weights"], dtype=float)
        gwas_idx = np.array([snp_to_idx[snp] for snp in snp_ids], dtype=int)
        null_idx = np.array([snp_to_null_idx[snp] for snp in snp_ids], dtype=int)
        z = z_values[gwas_idx]
        denom_var = float(payload["denominator_variance"])
        denom = float(np.sqrt(denom_var)) if denom_var > 0 else np.nan
        numerator = float(w @ z)
        x_dir = numerator / denom if denom > 0 else np.nan
        w2 = np.square(w)
        q_obs = float(np.sum(w2 * np.square(z)))
        lambdas = np.asarray(payload["lambdas"], dtype=float)
        p_obs = quadratic_p_value(q_obs, lambdas, method=method)
        normal_obs, clipped_obs, was_clipped = p_to_normal(p_obs, p_clip_epsilon)

        if n_null > 0:
            z_null = null_z[:, null_idx].astype(float, copy=False)
            q_null = np.square(z_null) @ w2
            p_null = quadratic_p_values(q_null, lambdas, method=method)
            null_scores = normal_scores_from_p_values(p_null, epsilon=p_clip_epsilon)[0]
        else:
            null_scores = np.empty(0, dtype=float)
        null_score_rows.append(null_scores)

        rows.append(
            {
                "gene_id": gene_id,
                "gene_symbol": gene_symbol,
                "chrom": chrom,
                "gene_start": int(gene_start),
                "gene_end": int(gene_end),
                "n_mapped_snps": int(len(snp_ids)),
                "signed_numerator": numerator,
                "denominator_variance": denom_var,
                "denominator": denom,
                "x_g_dir": x_dir,
                "quadratic_statistic": q_obs,
                "assoc_p_g": p_obs,
                "assoc_p_g_clipped": clipped_obs,
                "assoc_minuslog10p_g": -np.log10(clipped_obs),
                "assoc_normal_score_g": normal_obs,
                "is_p_clipped": was_clipped,
                "m_eff": float(payload["m_eff"]),
                "local_ld_score": float(payload["local_ld_score"]),
                "ld_status": str(payload["ld_status"]),
                "ld_cache_path": str(payload["ld_cache_path"]),
                "ld_shrinkage": shrinkage,
            }
        )

        if counter % 1000 == 0 or counter == len(payloads):
            print(f"Scored {counter:,}/{len(payloads):,} genes", flush=True)

    gene_scores = pd.DataFrame(rows)
    null_matrix = np.vstack(null_score_rows).T if null_score_rows else np.empty((n_null, 0))
    gene_scores["gene_length"] = gene_scores["gene_end"] - gene_scores["gene_start"] + 1
    gene_scores["log_gene_length"] = np.log1p(gene_scores["gene_length"])
    gene_scores["log_mapped_snp_count"] = np.log1p(gene_scores["n_mapped_snps"])
    gene_scores["log_m_eff"] = np.log1p(gene_scores["m_eff"])
    gene_scores["mappability"] = 1.0
    gene_scores = append_safety_columns(
        gene_scores.rename(columns={"gene_start": "start", "gene_end": "end"}),
        chrom_col="chrom",
        start_col="start",
        end_col="end",
    ).rename(columns={"start": "gene_start", "end": "gene_end"})
    report = {
        "n_genes_scored": int(len(gene_scores)),
        "n_null": int(n_null),
        "n_gwas_snps_for_null_source": int(len(z_values)),
        "n_mapped_snps_for_null_matrix": int(len(mapped_snp_ids)),
        "seed": int(seed),
        "ld_status_counts": status_counts,
        "ld_scoring_payload_cache": payload_cache_report,
    }
    return gene_scores, null_matrix, report


def compute_degree_stratified_percolation_null(
    graph,
    lcc_scores: pd.DataFrame,
    *,
    n_replicates: int,
    seed: int,
    n_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    permuted = degree_stratified_permuted_scores(
        lcc_scores,
        score_col="assoc_resid_score",
        degree_col="graph_degree",
        n_replicates=n_replicates,
        seed=seed,
        n_bins=n_bins,
    )
    null_base = lcc_scores.loc[:, ["gene_symbol"]].copy()
    auc_rows: list[dict[str, float | int]] = []
    curve_rows: list[pd.DataFrame] = []
    for idx in range(n_replicates):
        null_table = null_base.copy()
        null_table["assoc_resid_score"] = permuted[idx, :]
        null_ranking = rank_nodes_by_score(null_table, node_col="gene_symbol", score_col="assoc_resid_score")
        null_curve = percolation_curve(graph, null_ranking, RANK_FRACTION_GRID, node_col="gene_symbol")
        null_auc = percolation_auc(null_curve)
        auc_rows.append({"replicate": idx, "percolation_auc": null_auc})
        null_curve["replicate"] = idx
        curve_rows.append(null_curve)
    auc_table = pd.DataFrame(auc_rows)
    curve_table = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()
    return auc_table, curve_table


def compute_degree_preserving_graph_percolation_null(
    graph,
    ranking: pd.DataFrame,
    *,
    n_replicates: int,
    seed: int,
    nswap_per_edge: float,
    max_tries_per_swap: float,
    cache_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    auc_rows: list[dict[str, float | int]] = []
    curve_rows: list[pd.DataFrame] = []
    graph_rows: list[dict[str, float | int]] = []
    null_graphs = degree_preserving_graph_replicates(
        graph,
        n_replicates=n_replicates,
        seed=seed,
        nswap_per_edge=nswap_per_edge,
        max_tries_per_swap=max_tries_per_swap,
        cache_path=cache_path,
    )
    for idx, null_graph in enumerate(null_graphs):
        null_curve = percolation_curve(null_graph, ranking, RANK_FRACTION_GRID, node_col="gene_symbol")
        null_auc = percolation_auc(null_curve)
        auc_rows.append({"replicate": idx, "percolation_auc": null_auc})
        null_curve["replicate"] = idx
        curve_rows.append(null_curve)
        graph_rows.append({"replicate": idx, **graph_component_summary(null_graph)})
        if (idx + 1) % 5 == 0 or idx + 1 == n_replicates:
            print(f"Computed {idx + 1:,}/{n_replicates:,} degree-preserving graph nulls", flush=True)
    auc_table = pd.DataFrame(auc_rows)
    curve_table = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()
    graph_table = pd.DataFrame(graph_rows)
    return auc_table, curve_table, graph_table


def main() -> None:
    args = parse_args()
    ensure_outputs(args.out_dir, force=args.force)
    tables_dir = args.out_dir / "tables"
    reports_dir = args.out_dir / "reports"

    print("Loading HEIGHT GWAS", flush=True)
    gwas = load_height_gwas(args.gwas)

    if args.mapping.exists():
        print("Loading existing SNP-to-gene mapping", flush=True)
        mapping = pd.read_csv(
            args.mapping,
            sep="\t",
            compression="infer",
            dtype={"gene_id": str, "gene_symbol": str, "chrom": str, "snp_id": str},
        )
    else:
        print("Mapping file missing; rebuilding from gene locations", flush=True)
        genes = read_magma_gene_loc(args.gene_loc).table
        mapping = fast_positional_map_by_gene(
            gwas,
            genes,
            upstream_bp=args.upstream_bp,
            downstream_bp=args.downstream_bp,
        )
        mapping = add_positional_weights(mapping)
    mapping["weight"] = pd.to_numeric(mapping["weight"], errors="raise").astype(float)

    print("Computing LD-aware observed and null gene scores", flush=True)
    ld_cache_dirs = tuple(args.ld_cache_overlay_dir) + (args.ld_cache_dir,)
    gene_scores, null_scores, score_report = compute_ld_cached_gene_scores(
        gwas,
        mapping,
        ld_cache_dirs=ld_cache_dirs,
        n_null=args.n_null,
        seed=args.seed,
        method=args.quadratic_method,
        p_clip_epsilon=args.p_clip_epsilon,
        shrinkage=args.ld_shrinkage,
        allow_identity_fallback=args.allow_identity_fallback,
        payload_cache_path=args.ld_score_payload_cache or tables_dir / "HEIGHT_IRN.ld_scoring_payload_cache.npz",
    )
    write_table(tables_dir / "HEIGHT_IRN.gene_scores.1000G_LD.tsv.gz", gene_scores)
    np.savez_compressed(
        tables_dir / "HEIGHT_IRN.null_assoc_normal_scores.npz",
        null_scores=null_scores.astype(np.float32),
        gene_ids=gene_scores["gene_id"].astype(str).to_numpy(dtype=object),
        gene_symbols=gene_scores["gene_symbol"].astype(str).to_numpy(dtype=object),
    )

    print("Building STRING graph and filtering to LCC scored genes", flush=True)
    gene_symbols = tuple(gene_scores["gene_symbol"].dropna().astype(str).unique())
    _, graph_pre = build_string_graph(args, gene_symbols)
    lcc_nodes = tuple(sorted(str(node) for node in graph_pre.largest_component.nodes()))
    lcc_mask = gene_scores["gene_symbol"].astype(str).isin(lcc_nodes).to_numpy()
    lcc_scores = gene_scores.loc[lcc_mask].copy().sort_values("gene_symbol").reset_index(drop=True)
    lcc_gene_ids = lcc_scores["gene_id"].astype(str).tolist()
    gene_id_to_idx = {gene_id: idx for idx, gene_id in enumerate(gene_scores["gene_id"].astype(str))}
    lcc_indices = np.array([gene_id_to_idx[gene_id] for gene_id in lcc_gene_ids], dtype=int)
    null_lcc_scores = null_scores[:, lcc_indices]
    degree = dict(graph_pre.largest_component.degree())
    lcc_scores["graph_degree"] = lcc_scores["gene_symbol"].map(degree).astype(int)

    print("Residualizing observed scores with null-estimated moments", flush=True)
    lcc_scores, residualization = append_residualized_score(lcc_scores, null_scores=null_lcc_scores)
    null_resid = (null_lcc_scores - residualization.mu0[None, :]) / residualization.sigma0[None, :]
    write_table(tables_dir / "HEIGHT_IRN.lcc_gene_scores.1000G_LD.residualized.tsv.gz", lcc_scores)
    np.savez_compressed(
        tables_dir / "HEIGHT_IRN.null_residualized_scores.npz",
        null_resid=null_resid.astype(np.float32),
        gene_ids=np.array(lcc_gene_ids, dtype=object),
        gene_symbols=lcc_scores["gene_symbol"].astype(str).to_numpy(dtype=object),
    )

    print("Computing GSP and observed/null percolation", flush=True)
    nodes = tuple(lcc_scores["gene_symbol"].astype(str))
    lap = graph_laplacian(graph_pre.largest_component, nodes=nodes, kind="normalized")
    decomp = laplacian_eigendecomposition(
        lap.laplacian,
        nodes=nodes,
        n_components=min(args.eigen_components, max(1, len(nodes) - 2)),
    )
    signal = project_graph_signal(lcc_scores["assoc_resid_score"].to_numpy(dtype=float), decomp, laplacian=lap.laplacian)
    band_table = band_energy_table(signal.eigenvalues, signal.coefficients)
    write_table(tables_dir / "HEIGHT_IRN.gsp_band_energy.1000G_LD.tsv", band_table)

    ranking = rank_nodes_by_score(lcc_scores, node_col="gene_symbol", score_col="assoc_resid_score")
    observed_curve = percolation_curve(graph_pre.largest_component, ranking, RANK_FRACTION_GRID, node_col="gene_symbol")
    observed_auc = percolation_auc(observed_curve)
    write_table(tables_dir / "HEIGHT_IRN.percolation_curve.1000G_LD.observed.tsv", observed_curve)

    null_auc_rows: list[dict[str, float | int]] = []
    null_curve_rows: list[pd.DataFrame] = []
    null_base = lcc_scores.loc[:, ["gene_symbol"]].copy()
    for idx in range(args.n_null):
        null_table = null_base.copy()
        null_table["assoc_resid_score"] = null_resid[idx, :]
        null_ranking = rank_nodes_by_score(null_table, node_col="gene_symbol", score_col="assoc_resid_score")
        null_curve = percolation_curve(graph_pre.largest_component, null_ranking, RANK_FRACTION_GRID, node_col="gene_symbol")
        null_auc = percolation_auc(null_curve)
        null_auc_rows.append({"replicate": idx, "percolation_auc": null_auc})
        null_curve["replicate"] = idx
        null_curve_rows.append(null_curve)

    null_auc_table = pd.DataFrame(null_auc_rows)
    null_curve_table = pd.concat(null_curve_rows, ignore_index=True) if null_curve_rows else pd.DataFrame()
    write_table(tables_dir / "HEIGHT_IRN.percolation_auc.1000G_LD.null.tsv", null_auc_table)
    write_table(tables_dir / "HEIGHT_IRN.percolation_curves.1000G_LD.null.tsv", null_curve_table)
    expected_null_auc = float(null_auc_table["percolation_auc"].mean()) if not null_auc_table.empty else float("nan")
    delta_perc = observed_auc - expected_null_auc

    degree_strat_auc_table = pd.DataFrame()
    degree_strat_curve_table = pd.DataFrame()
    if args.n_degree_stratified_null > 0:
        print("Computing degree-stratified score percolation null", flush=True)
        degree_strat_auc_table, degree_strat_curve_table = compute_degree_stratified_percolation_null(
            graph_pre.largest_component,
            lcc_scores,
            n_replicates=args.n_degree_stratified_null,
            seed=args.seed + 101,
            n_bins=args.degree_stratified_bins,
        )
        write_table(
            tables_dir / "HEIGHT_IRN.percolation_auc.1000G_LD.degree_stratified_null.tsv",
            degree_strat_auc_table,
        )
        write_table(
            tables_dir / "HEIGHT_IRN.percolation_curves.1000G_LD.degree_stratified_null.tsv",
            degree_strat_curve_table,
        )

    degree_matched_auc_table = pd.DataFrame()
    degree_matched_curve_table = pd.DataFrame()
    degree_profile_table = pd.DataFrame()
    if args.n_degree_matched_node_null > 0:
        print("Computing degree-matched node percolation null", flush=True)
        selected_bin_counts, bin_to_nodes, degree_profile_table = prepare_degree_matched_rank_sets(
            lcc_scores,
            ranking,
            RANK_FRACTION_GRID,
            node_col="gene_symbol",
            degree_col="graph_degree",
            n_bins=args.degree_matched_bins,
        )
        degree_matched_auc_table, degree_matched_curve_table = compute_degree_matched_node_percolation_null(
            graph_pre.largest_component,
            selected_bin_counts,
            bin_to_nodes,
            n_replicates=args.n_degree_matched_node_null,
            seed=args.seed + 303,
            progress_interval=100,
        )
        write_table(
            tables_dir / "HEIGHT_IRN.percolation_auc.1000G_LD.degree_matched_node_null.tsv",
            degree_matched_auc_table,
        )
        write_table(
            tables_dir / "HEIGHT_IRN.percolation_curves.1000G_LD.degree_matched_node_null.tsv",
            degree_matched_curve_table,
        )
        write_table(tables_dir / "HEIGHT_IRN.degree_profile.by_rank_fraction.tsv", degree_profile_table)

    degree_graph_auc_table = pd.DataFrame()
    degree_graph_curve_table = pd.DataFrame()
    degree_graph_diagnostics = pd.DataFrame()
    if args.n_degree_graph_null > 0:
        print("Computing degree-preserving graph percolation null", flush=True)
        degree_graph_auc_table, degree_graph_curve_table, degree_graph_diagnostics = (
            compute_degree_preserving_graph_percolation_null(
                graph_pre.largest_component,
                ranking,
                n_replicates=args.n_degree_graph_null,
                seed=args.seed + 202,
                nswap_per_edge=args.degree_graph_nswap_per_edge,
                max_tries_per_swap=args.degree_graph_max_tries_per_swap,
            )
        )
        write_table(
            tables_dir / "HEIGHT_IRN.percolation_auc.1000G_LD.degree_preserving_graph_null.tsv",
            degree_graph_auc_table,
        )
        write_table(
            tables_dir / "HEIGHT_IRN.percolation_curves.1000G_LD.degree_preserving_graph_null.tsv",
            degree_graph_curve_table,
        )
        write_table(
            tables_dir / "HEIGHT_IRN.degree_preserving_graph_null.diagnostics.tsv",
            degree_graph_diagnostics,
        )

    clipping = summarize_clipping(gene_scores["is_p_clipped"])
    snp_null_summary = summarize_percolation_null(null_auc_table, observed_auc)
    degree_strat_summary = summarize_percolation_null(degree_strat_auc_table, observed_auc)
    degree_matched_summary = summarize_percolation_null(degree_matched_auc_table, observed_auc)
    degree_graph_summary = summarize_percolation_null(degree_graph_auc_table, observed_auc)
    architecture = classify_percolation_architecture(
        snp_permutation_null=snp_null_summary,
        degree_stratified_null=degree_strat_summary,
        degree_matched_node_null=degree_matched_summary,
        degree_preserving_graph_null=degree_graph_summary,
    )
    remaining_required_work: list[str] = []
    identity_status_counts = {
        key: value for key, value in score_report["ld_status_counts"].items() if key.startswith("identity_fallback")
    }
    if args.n_null < 100:
        remaining_required_work.append("increase SNP-permutation null replicate count to at least 100")
    if identity_status_counts:
        remaining_required_work.append("replace remaining identity LD fallback genes with full or block LD sensitivity")
    if args.n_degree_stratified_null <= 0:
        remaining_required_work.append("add degree-stratified score nulls")
    if args.n_degree_matched_node_null <= 0:
        remaining_required_work.append("add degree-matched node nulls")
    if args.n_degree_graph_null <= 0:
        remaining_required_work.append("add degree-preserving graph nulls")

    summary = {
        "created_utc": datetime.now(UTC).isoformat(),
        "trait": "HEIGHT_IRN",
        "mode": "ld_cached_with_snp_permutation_and_degree_nulls",
        "ld_mode": "1000G_EUR_per_gene_cache_with_optional_overlays",
        "ld_cache_dir": str(args.ld_cache_dir),
        "ld_cache_overlay_dirs": [str(path) for path in args.ld_cache_overlay_dir],
        "ld_cache_dirs_in_priority_order": [str(path) for path in ld_cache_dirs],
        "ld_shrinkage": args.ld_shrinkage,
        "quadratic_method": args.quadratic_method,
        "n_null": args.n_null,
        "n_degree_stratified_null": args.n_degree_stratified_null,
        "degree_stratified_bins": args.degree_stratified_bins,
        "n_degree_matched_node_null": args.n_degree_matched_node_null,
        "degree_matched_bins": args.degree_matched_bins,
        "n_degree_graph_null": args.n_degree_graph_null,
        "degree_graph_nswap_per_edge": args.degree_graph_nswap_per_edge,
        "degree_graph_max_tries_per_swap": args.degree_graph_max_tries_per_swap,
        "seed": args.seed,
        "n_gwas_snps": int(len(gwas)),
        "n_gene_scores": int(len(gene_scores)),
        "n_lcc_scored_genes": int(len(lcc_scores)),
        "score_report": score_report,
        "graph_coverage_report": asdict(graph_pre.coverage_report),
        "residualization_method": residualization.method,
        "gsp_method": decomp.method,
        "gsp_smoothness": signal.smoothness,
        "gsp_retained_energy_fraction": signal.retained_energy_fraction,
        "percolation_auc_observed": observed_auc,
        "percolation_auc_null_mean": expected_null_auc,
        "percolation_auc_null_sd": snp_null_summary["sd"],
        "delta_perc": delta_perc,
        "snp_permutation_null_summary": snp_null_summary,
        "degree_stratified_null_summary": degree_strat_summary,
        "degree_matched_node_null_summary": degree_matched_summary,
        "degree_preserving_graph_null_summary": degree_graph_summary,
        "percolation_architecture": architecture,
        "delta_perc_vs_degree_stratified_null": observed_auc - float(degree_strat_summary["mean"]),
        "delta_perc_vs_degree_matched_node_null": observed_auc - float(degree_matched_summary["mean"]),
        "delta_perc_vs_degree_preserving_graph_null": observed_auc - float(degree_graph_summary["mean"]),
        "p_clipping_summary": asdict(clipping),
        "outputs": {
            "gene_scores": str(tables_dir / "HEIGHT_IRN.gene_scores.1000G_LD.tsv.gz"),
            "lcc_gene_scores": str(tables_dir / "HEIGHT_IRN.lcc_gene_scores.1000G_LD.residualized.tsv.gz"),
            "null_assoc_scores": str(tables_dir / "HEIGHT_IRN.null_assoc_normal_scores.npz"),
            "null_residualized_scores": str(tables_dir / "HEIGHT_IRN.null_residualized_scores.npz"),
            "observed_percolation_curve": str(tables_dir / "HEIGHT_IRN.percolation_curve.1000G_LD.observed.tsv"),
            "null_percolation_auc": str(tables_dir / "HEIGHT_IRN.percolation_auc.1000G_LD.null.tsv"),
            "degree_stratified_null_auc": str(
                tables_dir / "HEIGHT_IRN.percolation_auc.1000G_LD.degree_stratified_null.tsv"
            ),
            "degree_matched_node_null_auc": str(
                tables_dir / "HEIGHT_IRN.percolation_auc.1000G_LD.degree_matched_node_null.tsv"
            ),
            "degree_matched_node_null_curves": str(
                tables_dir / "HEIGHT_IRN.percolation_curves.1000G_LD.degree_matched_node_null.tsv"
            ),
            "degree_profile": str(tables_dir / "HEIGHT_IRN.degree_profile.by_rank_fraction.tsv"),
            "degree_preserving_graph_null_auc": str(
                tables_dir / "HEIGHT_IRN.percolation_auc.1000G_LD.degree_preserving_graph_null.tsv"
            ),
            "degree_preserving_graph_diagnostics": str(
                tables_dir / "HEIGHT_IRN.degree_preserving_graph_null.diagnostics.tsv"
            ),
        },
        "remaining_required_work": remaining_required_work,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_json = json.dumps(summary, indent=2, sort_keys=True)
    (reports_dir / "HEIGHT_IRN.ld_null_mvp_summary.json").write_text(summary_json, encoding="utf-8")
    (reports_dir / "HEIGHT_IRN.analysis_ready_summary.json").write_text(summary_json, encoding="utf-8")
    report_md = [
        "# HEIGHT_IRN RIPPLE Analysis-Ready Run",
        "",
        "This run uses per-gene 1000G EUR LD cache, SNP-permutation nulls, and optional degree nulls.",
        "",
        f"- Gene scores: {summary['n_gene_scores']:,}",
        f"- LCC scored genes: {summary['n_lcc_scored_genes']:,}",
        f"- SNP-permutation null replicates: {args.n_null}",
        f"- Degree-stratified null replicates: {args.n_degree_stratified_null}",
        f"- Degree-matched node null replicates: {args.n_degree_matched_node_null}",
        f"- Degree-preserving graph null replicates: {args.n_degree_graph_null}",
        f"- Architecture class: {architecture['architecture_class']}",
        f"- Observed percolation AUC: {observed_auc:.6f}",
        f"- Mean null percolation AUC: {expected_null_auc:.6f}",
        f"- Delta_perc: {delta_perc:.6f}",
        f"- SNP-null Z: {snp_null_summary['z']:.6f}",
        f"- Delta_perc vs degree-stratified null: {summary['delta_perc_vs_degree_stratified_null']:.6f}",
        f"- Degree-matched node null Z: {degree_matched_summary['z']:.6f}",
        f"- Delta_perc vs degree-preserving graph null: {summary['delta_perc_vs_degree_preserving_graph_null']:.6f}",
        f"- Degree-preserving graph null Z: {degree_graph_summary['z']:.6f}",
        f"- GSP retained energy fraction: {signal.retained_energy_fraction:.6f}",
    ]
    report_text = "\n".join(report_md) + "\n"
    (reports_dir / "HEIGHT_IRN.ld_null_mvp_report.md").write_text(report_text, encoding="utf-8")
    (reports_dir / "HEIGHT_IRN.analysis_ready_report.md").write_text(report_text, encoding="utf-8")
    print(f"Wrote LD-null MVP outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
