#!/usr/bin/env python
"""Run SNP-overlap-preserving graph aggregation sensitivity nulls.

This review-oriented sensitivity complements the primary LD-aware SNP/gene-score
pipeline null. It preserves the observed SNP-to-gene mapping incidence: one
permuted SNP score is reused for every gene receiving that SNP, so overlapping
gene mappings remain coupled within each replicate. The statistic is an
approximate mapped-SNP burden proxy, not a replacement for the primary
LD-calibrated quadratic gene score.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ripple.graph import graph_laplacian  # noqa: E402
from ripple.graph_diffusion import heat_kernel_tau_statistics_matrix, parse_tau_grid  # noqa: E402
from ripple.percolation import percolation_auc, percolation_curve, rank_nodes_by_score  # noqa: E402

PRIVATE_ROOT = (
    Path("D:/path/to/ripple_private_workspace")
    if Path("D:/path/to/ripple_private_workspace").exists()
    else Path("/path/to/ripple_private_workspace")
)
MANUSCRIPT_ROOT = (
    Path("D:/RIPPLE/RIPPLE_manuscript")
    if Path("D:/RIPPLE/RIPPLE_manuscript").exists()
    else Path("/path/to/ripple_manuscript_workspace")
)
ANALYSIS_ROOT = PRIVATE_ROOT / "30_analysis"
QC_REPORT_ROOT = PRIVATE_ROOT / "20_processed_data" / "gwas_qc" / "qc_reports"
DEFAULT_OUT_DIR = ANALYSIS_ROOT / "review_driven_revision_v1" / "overlap_preserving_null_sensitivity"
DEFAULT_SUPPLEMENT_DIR = MANUSCRIPT_ROOT / "supplementary_files"
RANK_FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20)

ANALYSES: dict[str, dict[str, Any]] = {
    "DR_MVP": {
        "analysis_id": "DR_MVP_default_final5000",
        "trait": "DR_MVP",
        "analysis_dir": ANALYSIS_ROOT / "dr_mvp_string_final5000",
        "mapping_trait": "DR_MVP",
        "gwas_report_trait": "DR_MVP",
        "gwas_output_key": "no_mhc",
        "edge_trait": "DR_MVP",
        "edge_dir": ANALYSIS_ROOT / "dr_mvp_string_final5000",
    },
    "DR_MVP_NO_MHC_NO_APOE": {
        "analysis_id": "DR_MVP_no_MHC_no_APOE_final5000",
        "trait": "DR_MVP_NO_MHC_NO_APOE",
        "analysis_dir": ANALYSIS_ROOT / "dr_mvp_no_mhc_no_apoe_final5000",
        "mapping_trait": "DR_MVP_NO_MHC_NO_APOE",
        "gwas_report_trait": "DR_MVP",
        "gwas_output_key": "no_mhc_no_apoe",
        "edge_trait": "DR_MVP",
        "edge_dir": ANALYSIS_ROOT / "dr_mvp_string_final5000",
    },
    "SCZ": {
        "analysis_id": "SCZ_no_MHC_final5000",
        "trait": "SCZ",
        "analysis_dir": ANALYSIS_ROOT / "scz_no_mhc_string_final5000",
        "mapping_trait": "SCZ",
        "gwas_report_trait": "SCZ",
        "gwas_output_key": "no_mhc",
        "edge_trait": "SCZ",
        "edge_dir": ANALYSIS_ROOT / "scz_no_mhc_string_final5000",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--supplement-dir", type=Path, default=DEFAULT_SUPPLEMENT_DIR)
    parser.add_argument("--traits", nargs="+", default=["DR_MVP", "DR_MVP_NO_MHC_NO_APOE", "SCZ"])
    parser.add_argument("--n-null", type=int, default=500)
    parser.add_argument("--degree-bins", type=int, default=10)
    parser.add_argument("--tau-grid", default="0.25,0.5,1.0,2.0,4.0")
    parser.add_argument("--diffusion-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--copy-to-supplement", action="store_true", default=True)
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def read_tsv(path: Path, *, usecols: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t", compression="infer", usecols=usecols)


def write_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False, compression="infer")


def gwas_path(report_trait: str, output_key: str) -> Path:
    report_path = QC_REPORT_ROOT / f"{report_trait}.qc_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return Path(report["outputs"][output_key])


def lcc_scores_path(analysis: dict[str, Any]) -> Path:
    trait = str(analysis["trait"])
    return Path(analysis["analysis_dir"]) / "tables" / f"{trait}.lcc_gene_scores.1000G_LD.residualized.tsv.gz"


def mapping_path(analysis: dict[str, Any]) -> Path:
    trait = str(analysis["mapping_trait"])
    return Path(analysis["analysis_dir"]) / "tables" / f"{trait}.gene_body_mapping.tsv.gz"


def edge_path(analysis: dict[str, Any]) -> Path:
    trait = str(analysis["edge_trait"])
    return Path(analysis["edge_dir"]) / "tables" / f"{trait}.analysis_graph_edges.tsv.gz"


def graph_from_edges(path: Path, genes: set[str]) -> nx.Graph:
    edges = read_tsv(path)
    edges = edges.rename(columns={edges.columns[0]: "node1", edges.columns[1]: "node2"})
    graph = nx.Graph()
    graph.add_nodes_from(genes)
    for row in edges[["node1", "node2"]].dropna().itertuples(index=False):
        a = str(row.node1).upper()
        b = str(row.node2).upper()
        if a in genes and b in genes and a != b:
            graph.add_edge(a, b)
    graph.remove_nodes_from([node for node, degree in dict(graph.degree()).items() if degree == 0])
    return graph


def design_matrix(scores: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    covariates = ["log_gene_length", "log_mapped_snp_count", "log_m_eff", "local_ld_score", "mappability"]
    x = scores[covariates].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    med = np.nanmedian(x, axis=0)
    x = np.where(np.isfinite(x), x, med)
    mean = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd <= 0] = 1.0
    x = (x - mean) / sd
    x = np.column_stack([np.ones(x.shape[0]), x])
    return x, np.linalg.pinv(x)


def residualize_proxy(q_values: np.ndarray, x: np.ndarray, pinv_x: np.ndarray) -> np.ndarray:
    y = np.log1p(np.clip(np.asarray(q_values, dtype=float), 0.0, None))
    resid = y - x @ (pinv_x @ y)
    sd = float(np.std(resid, ddof=1))
    return resid / sd if sd > 0 else resid


def build_mapping_matrix(
    mapping: pd.DataFrame,
    scores: pd.DataFrame,
    gwas: pd.DataFrame,
) -> tuple[sparse.csr_matrix, pd.DataFrame, np.ndarray, np.ndarray]:
    scores = scores.copy()
    scores["gene_symbol"] = scores["gene_symbol"].astype(str).str.upper()
    scores["gene_id"] = scores["gene_id"].astype(str)
    gwas = gwas.loc[:, ["snp_id", "chrom", "z"]].dropna().copy()
    gwas["snp_id"] = gwas["snp_id"].astype(str)
    gwas["z"] = pd.to_numeric(gwas["z"], errors="coerce")
    gwas = gwas.dropna(subset=["z"]).drop_duplicates("snp_id")
    snp_to_row = {snp: idx for idx, snp in enumerate(gwas["snp_id"].to_numpy(dtype=object))}
    gene_to_col = {gene_id: idx for idx, gene_id in enumerate(scores["gene_id"].to_numpy(dtype=object))}

    work = mapping.loc[:, ["snp_id", "gene_id", "weight"]].copy()
    work["snp_id"] = work["snp_id"].astype(str)
    work["gene_id"] = work["gene_id"].astype(str)
    work["snp_row"] = work["snp_id"].map(snp_to_row)
    work["gene_col"] = work["gene_id"].map(gene_to_col)
    work = work.dropna(subset=["snp_row", "gene_col"])
    work["weight"] = pd.to_numeric(work["weight"], errors="coerce").fillna(1.0)
    rows = work["snp_row"].to_numpy(dtype=int)
    cols = work["gene_col"].to_numpy(dtype=int)
    vals = np.square(work["weight"].to_numpy(dtype=float))
    matrix = sparse.csr_matrix((vals, (rows, cols)), shape=(len(gwas), len(scores)))
    weight_sum = np.asarray(matrix.sum(axis=0)).ravel()
    weight_sum = np.where(weight_sum > 0, weight_sum, 1.0)
    norm = 1.0 / np.sqrt(weight_sum)
    matrix = matrix @ sparse.diags(norm, format="csr")
    chrom = pd.to_numeric(gwas["chrom"], errors="coerce").fillna(-1).to_numpy(dtype=int)
    z2 = np.square(gwas["z"].to_numpy(dtype=float))
    return matrix.tocsr(), scores, z2, chrom


def proxy_scores_from_z2(
    z2: np.ndarray,
    mapping_matrix: sparse.csr_matrix,
    x: np.ndarray,
    pinv_x: np.ndarray,
) -> np.ndarray:
    q = np.asarray(z2 @ mapping_matrix).ravel()
    return residualize_proxy(q, x, pinv_x)


def permute_z2_within_chrom(z2: np.ndarray, chrom: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = z2.copy()
    for chrom_id in np.unique(chrom):
        idx = np.flatnonzero(chrom == chrom_id)
        if idx.size > 1:
            out[idx] = out[rng.permutation(idx)]
    return out


def percolation_stat(graph: nx.Graph, scores: pd.DataFrame, values: np.ndarray) -> float:
    table = pd.DataFrame({"gene_symbol": scores["gene_symbol"].to_numpy(dtype=object), "score": values})
    ranking = rank_nodes_by_score(table, node_col="gene_symbol", score_col="score")
    curve = percolation_curve(graph, ranking, RANK_FRACTIONS, node_col="gene_symbol")
    return percolation_auc(curve)


def summarize_null(observed: float, null_values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(null_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    mean = float(np.mean(finite))
    sd = float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
    z = float((observed - mean) / sd) if sd > 0 else float("nan")
    p = float((1 + np.count_nonzero(finite >= observed)) / (1 + finite.size))
    return {"observed_value": observed, "null_mean": mean, "null_sd": sd, "z": z, "empirical_p": p, "n_null": int(finite.size)}


def stable_seed_offset(text: str, modulo: int = 100_000) -> int:
    """Deterministic seed offset independent of Python's randomized hash seed."""

    return sum((idx + 1) * ord(char) for idx, char in enumerate(text)) % modulo


def run_analysis(args: argparse.Namespace, key: str) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    analysis = ANALYSES[key]
    trait = str(analysis["trait"])
    analysis_id = str(analysis["analysis_id"])
    rng = np.random.default_rng(args.seed + stable_seed_offset(key))

    scores = read_tsv(lcc_scores_path(analysis))
    scores["gene_symbol"] = scores["gene_symbol"].astype(str).str.upper()
    genes = set(scores["gene_symbol"].astype(str))
    graph = graph_from_edges(edge_path(analysis), genes)
    scores = scores[scores["gene_symbol"].isin(set(graph.nodes()))].drop_duplicates("gene_symbol").reset_index(drop=True)
    mapping = read_tsv(mapping_path(analysis), usecols=["snp_id", "gene_id", "weight"])
    gwas = read_tsv(gwas_path(str(analysis["gwas_report_trait"]), str(analysis["gwas_output_key"])), usecols=["snp_id", "chrom", "z"])

    mapping_matrix, scores, z2, chrom = build_mapping_matrix(mapping, scores, gwas)
    x, pinv_x = design_matrix(scores)
    observed_scores = proxy_scores_from_z2(z2, mapping_matrix, x, pinv_x)
    observed_perc = percolation_stat(graph, scores, observed_scores)

    null_scores = np.empty((args.n_null, len(scores)), dtype=np.float32)
    null_perc = np.empty(args.n_null, dtype=float)
    for idx in range(args.n_null):
        permuted = permute_z2_within_chrom(z2, chrom, rng)
        values = proxy_scores_from_z2(permuted, mapping_matrix, x, pinv_x)
        null_scores[idx, :] = values.astype(np.float32)
        null_perc[idx] = percolation_stat(graph, scores, values)

    lap = graph_laplacian(graph, nodes=tuple(scores["gene_symbol"].astype(str)), kind="normalized")
    observed_matrix = np.maximum(observed_scores.reshape(1, -1), 0.0)
    null_matrix = np.maximum(null_scores.astype(float), 0.0)
    tau_grid = parse_tau_grid(args.tau_grid)
    observed_tau = heat_kernel_tau_statistics_matrix(lap.laplacian, observed_matrix, tau_grid=tau_grid, batch_size=1)[0]
    null_tau = heat_kernel_tau_statistics_matrix(
        lap.laplacian,
        null_matrix,
        tau_grid=tau_grid,
        batch_size=args.diffusion_batch_size,
    )
    observed_tmax = float(np.max(observed_tau))
    null_tmax = np.max(null_tau, axis=1)

    rows: list[dict[str, Any]] = []
    for statistic_name, observed, null_values in [
        ("overlap_preserving_proxy_percolation_auc", observed_perc, null_perc),
        ("overlap_preserving_proxy_diffusion_Tmax", observed_tmax, null_tmax),
    ]:
        summary = summarize_null(float(observed), null_values)
        claim_status = "supportive" if float(summary["z"]) >= 2.5 else "negative"
        rows.append(
            {
                "trait": trait,
                "analysis_id": analysis_id,
                "statistic_name": statistic_name,
                "statistic_direction": "greater_is_more_extreme",
                "null_type": "snp_label_permutation_within_chromosome_preserving_snp_to_gene_overlap",
                **summary,
                "threshold": 2.5,
                "claim_status": claim_status,
                "graph_id": "STRING_default",
                "score_stream": "overlap_preserving_proxy_unsigned_burden",
                "n_graph_genes": int(len(scores)),
                "n_graph_edges": int(graph.number_of_edges()),
                "n_gwas_snps": int(len(z2)),
                "n_mapping_edges": int(mapping_matrix.nnz),
                "tau_grid": ",".join(str(item) for item in tau_grid) if "diffusion" in statistic_name else "",
                "method_scope": "review_sensitivity_not_primary_ld_quadratic_score",
                "interpretation": (
                    "Sensitivity null preserves shared SNP assignments across overlapping genes but uses a proxy "
                    "mapped-SNP burden score; it does not replace the primary LD-aware quadratic pipeline null."
                ),
                "source_result_path": str(args.out_dir / "overlap_preserving_null_sensitivity_summary.tsv"),
                "script_path": str(Path(__file__).resolve()),
                "seed": int(args.seed),
                "timestamp": now_utc(),
            }
        )
    null_rows = pd.concat(
        [
            pd.DataFrame(
                {
                    "trait": trait,
                    "analysis_id": analysis_id,
                    "replicate": np.arange(args.n_null),
                    "statistic_name": "overlap_preserving_proxy_percolation_auc",
                    "statistic_value": null_perc,
                }
            ),
            pd.DataFrame(
                {
                    "trait": trait,
                    "analysis_id": analysis_id,
                    "replicate": np.arange(args.n_null),
                    "statistic_name": "overlap_preserving_proxy_diffusion_Tmax",
                    "statistic_value": null_tmax,
                }
            ),
        ],
        ignore_index=True,
    )
    return rows, null_rows


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    all_nulls: list[pd.DataFrame] = []
    for key in args.traits:
        if key not in ANALYSES:
            raise ValueError(f"Unknown trait key {key}; available: {sorted(ANALYSES)}")
        rows, nulls = run_analysis(args, key)
        all_rows.extend(rows)
        all_nulls.append(nulls)
    summary = pd.DataFrame(all_rows)
    null_dist = pd.concat(all_nulls, ignore_index=True) if all_nulls else pd.DataFrame()
    summary_path = args.out_dir / "overlap_preserving_null_sensitivity_summary.tsv"
    null_path = args.out_dir / "overlap_preserving_null_sensitivity_nulls.tsv.gz"
    write_table(summary_path, summary)
    write_table(null_path, null_dist)
    manifest = {
        "created_utc": now_utc(),
        "script_path": str(Path(__file__).resolve()),
        "traits": args.traits,
        "n_null": int(args.n_null),
        "summary": str(summary_path),
        "null_distribution": str(null_path),
        "method_boundary": "review sensitivity only; proxy score preserving SNP-to-gene overlap",
    }
    (args.out_dir / "overlap_preserving_null_sensitivity_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    if args.copy_to_supplement:
        write_table(args.supplement_dir / "overlap_preserving_null_sensitivity_summary.tsv", summary)
    print(f"Wrote overlap-preserving null sensitivity to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
