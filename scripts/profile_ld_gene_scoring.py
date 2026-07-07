#!/usr/bin/env python
"""Profile LD-aware gene scoring hot paths on a bounded real-data subset.

This is a performance-maintenance utility, not part of the manuscript pipeline.
It compares the legacy scalar null-score kernel with the vectorized kernel used
by `compute_ld_cached_gene_scores`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ripple.defaults import DEFAULT_LD_SHRINKAGE  # noqa: E402
from ripple.signals.signed import shrink_ld  # noqa: E402
from ripple.signals.unsigned import normal_scores_from_p_values, quadratic_p_value, quadratic_p_values  # noqa: E402
from run_height_ld_null_mvp import (  # noqa: E402
    build_null_z_matrix_for_snps,
    compute_ld_cached_gene_scores,
    load_ld_entry,
    p_to_normal,
    weighted_ld_mixture_eigenvalues,
)
from run_height_mvp import load_height_gwas, write_table  # noqa: E402


PRIVATE_ROOT = Path("/path/to/ripple_private_workspace")
DEFAULT_GWAS = PRIVATE_ROOT / "20_processed_data" / "gwas_qc" / "core_hm3_no_mhc" / "HEIGHT_IRN.tsv.gz"
DEFAULT_MAPPING = PRIVATE_ROOT / "30_analysis" / "height_irn_mvp" / "tables" / "HEIGHT_IRN.gene_body_mapping.tsv.gz"
DEFAULT_LD_CACHE = PRIVATE_ROOT / "30_analysis" / "height_irn_mvp" / "ld_cache_1000G_EUR"
DEFAULT_OUT_DIR = PRIVATE_ROOT / "30_analysis" / "performance_optimization_v1" / "ld_scoring_profile"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gwas", type=Path, default=DEFAULT_GWAS)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--ld-cache-dir", type=Path, default=DEFAULT_LD_CACHE)
    parser.add_argument("--ld-cache-overlay-dir", type=Path, action="append", default=[])
    parser.add_argument("--payload-cache", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-genes", type=int, default=300)
    parser.add_argument("--n-null", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--method", choices=["liu", "satterthwaite", "saddlepoint"], default="liu")
    parser.add_argument("--p-clip-epsilon", type=float, default=1e-300)
    parser.add_argument("--ld-shrinkage", type=float, default=DEFAULT_LD_SHRINKAGE)
    parser.add_argument("--allow-identity-fallback", action="store_true", default=True)
    return parser.parse_args()


def load_mapping_subset(path: Path, *, n_genes: int) -> pd.DataFrame:
    mapping = pd.read_csv(
        path,
        sep="\t",
        compression="infer",
        dtype={"gene_id": str, "gene_symbol": str, "chrom": str, "snp_id": str},
    )
    gene_ids = mapping["gene_id"].drop_duplicates().head(n_genes).astype(str)
    return mapping.loc[mapping["gene_id"].astype(str).isin(set(gene_ids))].copy()


def resolve_gene_ld_payloads(
    gwas: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    ld_cache_dirs: tuple[Path, ...],
    shrinkage: float,
    allow_identity_fallback: bool,
) -> tuple[list[dict[str, object]], np.ndarray, dict[str, int], int]:
    gwas_indexed = gwas.drop_duplicates("snp_id").reset_index(drop=True)
    snp_to_idx = {snp: idx for idx, snp in enumerate(gwas_indexed["snp_id"].astype(str))}
    z_values = gwas_indexed["z"].to_numpy(dtype=float)
    mapped_snp_ids = [
        snp
        for snp in mapping["snp_id"].dropna().astype(str).drop_duplicates()
        if snp in snp_to_idx
    ]
    snp_to_null_idx = {snp: idx for idx, snp in enumerate(mapped_snp_ids)}
    payloads: list[dict[str, object]] = []
    status_counts: dict[str, int] = {}
    grouped = mapping.groupby(["gene_id", "gene_symbol", "chrom", "gene_start", "gene_end"], observed=True, sort=True)

    for (gene_id, gene_symbol, *_), group in grouped:
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
        if ld_status != "computed":
            if not allow_identity_fallback:
                raise RuntimeError(f"LD cache unavailable for {gene_id}: {ld_status}")
            snp_ids = tuple(snp for snp in weights.index if snp in snp_to_idx and snp in snp_to_null_idx)
            ld = np.eye(len(snp_ids), dtype=float)
            ld_status = f"identity_fallback_{ld_status}"

        status_counts[ld_status] = status_counts.get(ld_status, 0) + 1
        if not snp_ids:
            continue

        w = weights.loc[list(snp_ids)].to_numpy(dtype=float)
        r_star = shrink_ld(ld, shrinkage=shrinkage)
        lambdas = weighted_ld_mixture_eigenvalues(ld, w, shrinkage=shrinkage)
        payloads.append(
            {
                "gene_id": gene_id,
                "gene_symbol": str(gene_symbol),
                "idx": np.array([snp_to_idx[snp] for snp in snp_ids], dtype=int),
                "null_idx": np.array([snp_to_null_idx[snp] for snp in snp_ids], dtype=int),
                "w2": np.square(w),
                "denominator_variance": float(w @ r_star @ w),
                "lambdas": lambdas,
            }
        )
    return payloads, z_values, status_counts, len(mapped_snp_ids)


def benchmark_null_kernels(
    payloads: list[dict[str, object]],
    null_z: np.ndarray,
    *,
    method: str,
    p_clip_epsilon: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, object]] = []
    scalar_elapsed = 0.0
    vectorized_elapsed = 0.0
    max_abs_diff = 0.0

    for payload in payloads:
        idx = np.asarray(payload["null_idx"], dtype=int)
        w2 = np.asarray(payload["w2"], dtype=float)
        lambdas = np.asarray(payload["lambdas"], dtype=float)

        start = time.perf_counter()
        scalar_scores = np.empty(null_z.shape[0], dtype=float)
        for null_idx in range(null_z.shape[0]):
            z_null = null_z[null_idx, idx].astype(float)
            q_null = float(np.sum(w2 * np.square(z_null)))
            p_null = quadratic_p_value(q_null, lambdas, method=method)
            scalar_scores[null_idx] = p_to_normal(p_null, p_clip_epsilon)[0]
        scalar_time = time.perf_counter() - start

        start = time.perf_counter()
        z_null = null_z[:, idx].astype(float, copy=False)
        q_null = np.square(z_null) @ w2
        p_null = quadratic_p_values(q_null, lambdas, method=method)
        vector_scores = normal_scores_from_p_values(p_null, epsilon=p_clip_epsilon)[0]
        vector_time = time.perf_counter() - start

        diff = float(np.max(np.abs(scalar_scores - vector_scores))) if len(vector_scores) else 0.0
        max_abs_diff = max(max_abs_diff, diff)
        scalar_elapsed += scalar_time
        vectorized_elapsed += vector_time
        rows.append(
            {
                "gene_id": payload["gene_id"],
                "gene_symbol": payload["gene_symbol"],
                "n_snps": int(len(idx)),
                "scalar_elapsed_seconds": scalar_time,
                "vectorized_elapsed_seconds": vector_time,
                "speedup": scalar_time / vector_time if vector_time > 0 else np.nan,
                "max_abs_score_diff": diff,
            }
        )

    summary = {
        "scalar_elapsed_seconds": float(scalar_elapsed),
        "vectorized_elapsed_seconds": float(vectorized_elapsed),
        "speedup": float(scalar_elapsed / vectorized_elapsed) if vectorized_elapsed > 0 else float("nan"),
        "max_abs_score_diff": float(max_abs_diff),
    }
    return pd.DataFrame(rows), summary


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ld_cache_dirs = tuple(args.ld_cache_overlay_dir) + (args.ld_cache_dir,)

    gwas = load_height_gwas(args.gwas)
    mapping = load_mapping_subset(args.mapping, n_genes=args.n_genes)
    payloads, z_values, status_counts, n_mapped_snps = resolve_gene_ld_payloads(
        gwas,
        mapping,
        ld_cache_dirs=ld_cache_dirs,
        shrinkage=args.ld_shrinkage,
        allow_identity_fallback=args.allow_identity_fallback,
    )
    null_z = build_null_z_matrix_for_snps(
        z_values,
        n_target_snps=n_mapped_snps,
        n_null=args.n_null,
        seed=args.seed,
    )

    kernel_table, kernel_summary = benchmark_null_kernels(
        payloads,
        null_z,
        method=args.method,
        p_clip_epsilon=args.p_clip_epsilon,
    )
    write_table(args.out_dir / "ld_null_kernel_profile.tsv", kernel_table)

    start = time.perf_counter()
    gene_scores, null_matrix, score_report = compute_ld_cached_gene_scores(
        gwas,
        mapping,
        ld_cache_dirs=ld_cache_dirs,
        n_null=args.n_null,
        seed=args.seed,
        method=args.method,
        p_clip_epsilon=args.p_clip_epsilon,
        shrinkage=args.ld_shrinkage,
        allow_identity_fallback=args.allow_identity_fallback,
        payload_cache_path=args.payload_cache,
    )
    full_elapsed = time.perf_counter() - start
    write_table(args.out_dir / "optimized_subset_gene_scores.tsv.gz", gene_scores)
    np.savez_compressed(args.out_dir / "optimized_subset_null_scores.npz", null_scores=null_matrix.astype(np.float32))

    summary = {
        "created_utc": datetime.now(UTC).isoformat(),
        "purpose": "ld_gene_scoring_performance_profile",
        "gwas": str(args.gwas),
        "mapping": str(args.mapping),
        "ld_cache_dirs": [str(path) for path in ld_cache_dirs],
        "payload_cache": str(args.payload_cache) if args.payload_cache is not None else None,
        "n_requested_genes": int(args.n_genes),
        "n_profiled_genes": int(len(payloads)),
        "n_mapped_snps_for_null_matrix": int(n_mapped_snps),
        "n_null": int(args.n_null),
        "method": args.method,
        "kernel_summary": kernel_summary,
        "optimized_compute_ld_cached_gene_scores_seconds": float(full_elapsed),
        "optimized_genes_scored": int(len(gene_scores)),
        "optimized_null_matrix_shape": list(null_matrix.shape),
        "ld_status_counts_profile_payloads": status_counts,
        "score_report": score_report,
    }
    (args.out_dir / "ld_scoring_profile_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
