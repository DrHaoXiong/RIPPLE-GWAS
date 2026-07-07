#!/usr/bin/env python
"""Run a trait-generic RIPPLE LD-aware analysis with calibrated percolation reporting."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
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

from ripple.claims import build_claim_tier_table  # noqa: E402
from ripple.defaults import DEFAULT_LD_SHRINKAGE, RANK_FRACTION_GRID  # noqa: E402
from ripple.diagnostics import (  # noqa: E402
    build_trait_suitability_diagnostic,
    gene_score_clipping_diagnostics,
    gene_score_transform_sensitivity,
    render_trait_architecture_markdown,
    residualization_diagnostics,
)
from ripple.graph import graph_laplacian  # noqa: E402
from ripple.graph_diffusion import (  # noqa: E402
    DEFAULT_TAU_GRID,
    degree_preserving_graph_diffusion_null,
    degree_stratified_diffusion_null,
    parse_tau_grid,
)
from ripple.gsp import band_energy_table, laplacian_eigendecomposition, project_graph_signal  # noqa: E402
from ripple.io.annotations import read_magma_gene_loc  # noqa: E402
from ripple.io.graph import networkx_to_edge_list, read_edge_list, write_edge_list  # noqa: E402
from ripple.mapping.weights import add_positional_weights  # noqa: E402
from ripple.modules import (  # noqa: E402
    load_gene_sets,
    render_module_discovery_report,
    run_local_module_discovery,
)
from ripple.percolation import (  # noqa: E402
    classify_percolation_architecture,
    compute_degree_matched_node_percolation_null,
    percolation_auc,
    percolation_curve,
    prepare_degree_matched_rank_sets,
    rank_nodes_by_score,
    summarize_percolation_threshold_robustness,
    summarize_percolation_null,
)
from ripple.signals.residualize import append_residualized_score  # noqa: E402
from ripple.signals.safety import summarize_clipping  # noqa: E402
from run_height_ld_null_mvp import (  # noqa: E402
    compute_degree_preserving_graph_percolation_null,
    compute_degree_stratified_percolation_null,
    compute_ld_cached_gene_scores,
    ensure_outputs,
)
from run_height_mvp import (  # noqa: E402
    DEFAULT_GENE_LOC,
    DEFAULT_STRING_INFO,
    DEFAULT_STRING_LINKS,
    build_string_graph,
    fast_positional_map_by_gene,
    load_height_gwas,
    write_table,
)


PRIVATE_ROOT = Path("/path/to/ripple_private_workspace")
DEFAULT_GWAS_DIR = PRIVATE_ROOT / "20_processed_data" / "gwas_qc" / "core_hm3_no_mhc"
DEFAULT_BASE_LD_CACHE = PRIVATE_ROOT / "30_analysis" / "height_irn_mvp" / "ld_cache_1000G_EUR"
DEFAULT_LARGE_GENE_LD_OVERLAY = (
    PRIVATE_ROOT / "30_analysis" / "height_irn_mvp" / "ld_cache_1000G_EUR_large_gene_full"
)


def trait_slug(trait: str) -> str:
    return trait.lower().replace("-", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trait", required=True)
    parser.add_argument("--gwas", type=Path, default=None)
    parser.add_argument("--gene-loc", type=Path, default=DEFAULT_GENE_LOC)
    parser.add_argument("--mapping", type=Path, default=None)
    parser.add_argument("--ld-cache-dir", type=Path, default=DEFAULT_BASE_LD_CACHE)
    parser.add_argument("--ld-cache-overlay-dir", type=Path, action="append", default=[])
    parser.add_argument("--ld-score-payload-cache", type=Path, default=None)
    parser.add_argument("--graph-name", default="string_ppi")
    parser.add_argument("--graph-edge-list", type=Path, default=None)
    parser.add_argument("--string-links", type=Path, default=DEFAULT_STRING_LINKS)
    parser.add_argument("--string-info", type=Path, default=DEFAULT_STRING_INFO)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--upstream-bp", type=int, default=0)
    parser.add_argument("--downstream-bp", type=int, default=0)
    parser.add_argument("--string-min-score", type=int, default=400)
    parser.add_argument("--eigen-components", type=int, default=128)
    parser.add_argument("--n-null", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--quadratic-method", choices=["liu", "satterthwaite"], default="liu")
    parser.add_argument("--p-clip-epsilon", type=float, default=1e-300)
    parser.add_argument("--ld-shrinkage", type=float, default=DEFAULT_LD_SHRINKAGE)
    parser.add_argument("--allow-identity-fallback", action="store_true", default=True)
    parser.add_argument("--n-degree-stratified-null", type=int, default=100)
    parser.add_argument("--degree-stratified-bins", type=int, default=10)
    parser.add_argument("--n-degree-matched-node-null", type=int, default=500)
    parser.add_argument("--degree-matched-bins", type=int, default=10)
    parser.add_argument("--n-degree-graph-null", type=int, default=20)
    parser.add_argument("--degree-graph-nswap-per-edge", type=float, default=1.0)
    parser.add_argument("--degree-graph-max-tries-per-swap", type=float, default=20.0)
    parser.add_argument("--degree-graph-cache", type=Path, default=None)
    parser.add_argument("--enable-diffusion", action="store_true")
    parser.add_argument(
        "--diffusion-score-mode",
        choices=["positive", "absolute", "raw", "rank"],
        default="positive",
    )
    parser.add_argument("--tau-grid", default=",".join(str(tau) for tau in DEFAULT_TAU_GRID))
    parser.add_argument(
        "--diffusion-null",
        choices=["degree_stratified", "degree_preserving_graph", "both"],
        default="degree_stratified",
    )
    parser.add_argument("--n-diffusion-null", type=int, default=1000)
    parser.add_argument("--diffusion-degree-bins", type=int, default=20)
    parser.add_argument("--diffusion-batch-size", type=int, default=128)
    parser.add_argument("--weighted-laplacian", action="store_true")
    parser.add_argument("--graph-family-maxt", action="store_true")
    parser.add_argument("--save-null-distributions", action="store_true")
    parser.add_argument("--degree-residualized-sensitivity", action="store_true")
    parser.add_argument("--score-transform-sensitivity", action="store_true")
    parser.add_argument("--gene-set-file", type=Path, default=None)
    parser.add_argument("--no-default-dr-panel", action="store_true")
    parser.add_argument("--min-module-size", type=int, default=5)
    parser.add_argument("--min-module-subthreshold-genes", type=int, default=3)
    parser.add_argument("--max-local-modules", type=int, default=20)
    parser.add_argument("--n-module-random-null", type=int, default=200)
    parser.add_argument("--n-module-degree-matched-null", type=int, default=200)
    parser.add_argument("--n-module-degree-graph-null", type=int, default=20)
    parser.add_argument("--n-module-selection-aware-null", type=int, default=200)
    parser.add_argument("--n-pathway-random-null", type=int, default=200)
    parser.add_argument("--n-pathway-degree-matched-null", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_or_build_mapping(args: argparse.Namespace, gwas: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    trait = str(args.trait)
    mapping_path = args.mapping or tables_dir / f"{trait}.gene_body_mapping.tsv.gz"
    if mapping_path.exists():
        print(f"Loading SNP-to-gene mapping: {mapping_path}", flush=True)
        return pd.read_csv(
            mapping_path,
            sep="\t",
            compression="infer",
            dtype={"gene_id": str, "gene_symbol": str, "chrom": str, "snp_id": str},
        )

    print("Mapping file missing; rebuilding from gene locations", flush=True)
    genes = read_magma_gene_loc(args.gene_loc).table
    genes = genes[genes["chrom"].astype(str).isin({str(i) for i in range(1, 23)})].copy()
    mapping = fast_positional_map_by_gene(
        gwas,
        genes,
        upstream_bp=args.upstream_bp,
        downstream_bp=args.downstream_bp,
    )
    mapping = add_positional_weights(mapping)
    write_table(mapping_path, mapping)
    return mapping


def load_analysis_graph(args: argparse.Namespace, gene_symbols: tuple[str, ...]):
    """Load either the default STRING graph or a custom canonical edge list."""

    if args.graph_edge_list is not None:
        graph_edges = read_edge_list(args.graph_edge_list, source=args.graph_name)
        from ripple.graph import preprocess_reference_graph  # noqa: PLC0415

        return graph_edges, preprocess_reference_graph(graph_edges.edges, gene_universe=gene_symbols)
    return build_string_graph(args, gene_symbols)


def main() -> None:
    args = parse_args()
    trait = str(args.trait)
    gwas_path = args.gwas or DEFAULT_GWAS_DIR / f"{trait}.tsv.gz"
    out_dir = args.out_dir or PRIVATE_ROOT / "30_analysis" / f"{trait_slug(trait)}_analysis_ready"
    ensure_outputs(out_dir, force=args.force)
    tables_dir = out_dir / "tables"
    reports_dir = out_dir / "reports"

    print(f"Loading GWAS: {gwas_path}", flush=True)
    gwas = load_height_gwas(gwas_path)
    mapping = load_or_build_mapping(args, gwas, tables_dir)
    mapping["weight"] = pd.to_numeric(mapping["weight"], errors="raise").astype(float)

    print("Computing LD-aware observed and null gene scores", flush=True)
    overlay_dirs = list(args.ld_cache_overlay_dir)
    if DEFAULT_LARGE_GENE_LD_OVERLAY.exists() and DEFAULT_LARGE_GENE_LD_OVERLAY not in overlay_dirs:
        overlay_dirs.append(DEFAULT_LARGE_GENE_LD_OVERLAY)
    ld_cache_dirs = tuple(overlay_dirs) + (args.ld_cache_dir,)
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
        payload_cache_path=args.ld_score_payload_cache or tables_dir / f"{trait}.ld_scoring_payload_cache.npz",
    )
    write_table(tables_dir / f"{trait}.gene_scores.1000G_LD.tsv.gz", gene_scores)
    np.savez_compressed(
        tables_dir / f"{trait}.null_assoc_normal_scores.npz",
        null_scores=null_scores.astype(np.float32),
        gene_ids=gene_scores["gene_id"].astype(str).to_numpy(dtype=object),
        gene_symbols=gene_scores["gene_symbol"].astype(str).to_numpy(dtype=object),
    )

    print(f"Building {args.graph_name} graph and filtering to LCC scored genes", flush=True)
    gene_symbols = tuple(gene_scores["gene_symbol"].dropna().astype(str).unique())
    graph_edges, graph_pre = load_analysis_graph(args, gene_symbols)
    analysis_graph_edges_path = tables_dir / f"{trait}.analysis_graph_edges.tsv.gz"
    write_edge_list(
        analysis_graph_edges_path,
        networkx_to_edge_list(graph_pre.largest_component),
        source=f"{trait}_{args.graph_name}_largest_connected_component",
    )
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
    write_table(tables_dir / f"{trait}.lcc_gene_scores.1000G_LD.residualized.tsv.gz", lcc_scores)
    np.savez_compressed(
        tables_dir / f"{trait}.null_residualized_scores.npz",
        null_resid=null_resid.astype(np.float32),
        gene_ids=np.array(lcc_gene_ids, dtype=object),
        gene_symbols=lcc_scores["gene_symbol"].astype(str).to_numpy(dtype=object),
    )
    clipping_diagnostics = gene_score_clipping_diagnostics(
        gene_scores,
        trait=trait,
        p_clip_min=args.p_clip_epsilon,
    )
    residual_diag = residualization_diagnostics(lcc_scores, trait=trait)
    write_table(tables_dir / f"{trait}.gene_score_clipping_diagnostics.tsv", clipping_diagnostics)
    write_table(tables_dir / f"{trait}.residualization_diagnostics.tsv", residual_diag)
    transform_sensitivity = pd.DataFrame()
    if args.score_transform_sensitivity:
        transform_sensitivity = gene_score_transform_sensitivity(
            lcc_scores,
            trait=trait,
            graph=graph_pre.largest_component,
        )
        write_table(tables_dir / f"{trait}.gene_score_transform_sensitivity.tsv", transform_sensitivity)

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
    write_table(tables_dir / f"{trait}.gsp_band_energy.1000G_LD.tsv", band_table)

    ranking = rank_nodes_by_score(lcc_scores, node_col="gene_symbol", score_col="assoc_resid_score")
    observed_curve = percolation_curve(graph_pre.largest_component, ranking, RANK_FRACTION_GRID, node_col="gene_symbol")
    observed_auc = percolation_auc(observed_curve)
    write_table(tables_dir / f"{trait}.percolation_curve.1000G_LD.observed.tsv", observed_curve)

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
    write_table(tables_dir / f"{trait}.percolation_auc.1000G_LD.null.tsv", null_auc_table)
    write_table(tables_dir / f"{trait}.percolation_curves.1000G_LD.null.tsv", null_curve_table)

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
        write_table(tables_dir / f"{trait}.percolation_auc.1000G_LD.degree_stratified_null.tsv", degree_strat_auc_table)
        write_table(tables_dir / f"{trait}.percolation_curves.1000G_LD.degree_stratified_null.tsv", degree_strat_curve_table)

    degree_matched_auc_table = pd.DataFrame()
    degree_matched_curve_table = pd.DataFrame()
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
        write_table(tables_dir / f"{trait}.percolation_auc.1000G_LD.degree_matched_node_null.tsv", degree_matched_auc_table)
        write_table(
            tables_dir / f"{trait}.percolation_curves.1000G_LD.degree_matched_node_null.tsv",
            degree_matched_curve_table,
        )
        write_table(tables_dir / f"{trait}.degree_profile.by_rank_fraction.tsv", degree_profile_table)

    degree_graph_auc_table = pd.DataFrame()
    degree_graph_curve_table = pd.DataFrame()
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
                cache_path=args.degree_graph_cache,
            )
        )
        write_table(tables_dir / f"{trait}.percolation_auc.1000G_LD.degree_preserving_graph_null.tsv", degree_graph_auc_table)
        write_table(
            tables_dir / f"{trait}.percolation_curves.1000G_LD.degree_preserving_graph_null.tsv",
            degree_graph_curve_table,
        )
        write_table(tables_dir / f"{trait}.degree_preserving_graph_null.diagnostics.tsv", degree_graph_diagnostics)

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
    robustness_tables = [
        summarize_percolation_threshold_robustness(
            observed_curve,
            null_curve_table,
            trait=trait,
            graph_name=str(args.graph_name),
            null_type="snp_pipeline_null",
        )
    ]
    if not degree_strat_curve_table.empty:
        robustness_tables.append(
            summarize_percolation_threshold_robustness(
                observed_curve,
                degree_strat_curve_table,
                trait=trait,
                graph_name=str(args.graph_name),
                null_type="degree_stratified_score_null",
            )
        )
    if not degree_matched_curve_table.empty:
        robustness_tables.append(
            summarize_percolation_threshold_robustness(
                observed_curve,
                degree_matched_curve_table,
                trait=trait,
                graph_name=str(args.graph_name),
                null_type="degree_matched_node_null",
            )
        )
    if not degree_graph_curve_table.empty:
        robustness_tables.append(
            summarize_percolation_threshold_robustness(
                observed_curve,
                degree_graph_curve_table,
                trait=trait,
                graph_name=str(args.graph_name),
                null_type="degree_preserving_graph_null",
            )
        )
    percolation_robustness = pd.concat(robustness_tables, ignore_index=True)
    percolation_robustness_path = tables_dir / f"{trait}.percolation_threshold_robustness.tsv"
    write_table(percolation_robustness_path, percolation_robustness)

    diffusion_summary_table = pd.DataFrame()
    diffusion_tau_table = pd.DataFrame()
    diffusion_null_distribution = pd.DataFrame()
    diffusion_summary_path = tables_dir / f"{trait}.diffusion_kernel_summary.tsv"
    diffusion_tau_path = tables_dir / f"{trait}.diffusion_kernel_tau_stats.tsv"
    diffusion_null_path = tables_dir / f"{trait}.diffusion_kernel_null_distribution.tsv.gz"
    if args.enable_diffusion:
        print("Computing heat-kernel diffusion weak-signal statistic", flush=True)
        tau_grid = parse_tau_grid(args.tau_grid)
        diffusion_summaries: list[pd.DataFrame] = []
        diffusion_tau_summaries: list[pd.DataFrame] = []
        diffusion_nulls: list[pd.DataFrame] = []
        if args.diffusion_null in {"degree_stratified", "both"}:
            summary_part, tau_part, null_part = degree_stratified_diffusion_null(
                graph_pre.largest_component,
                lcc_scores,
                trait=trait,
                graph_name=str(args.graph_name),
                score_mode=args.diffusion_score_mode,
                tau_grid=tau_grid,
                n_replicates=args.n_diffusion_null,
                seed=args.seed + 1101,
                n_bins=args.diffusion_degree_bins,
                weighted_laplacian=args.weighted_laplacian,
                batch_size=args.diffusion_batch_size,
            )
            diffusion_summaries.append(summary_part)
            diffusion_tau_summaries.append(tau_part)
            diffusion_nulls.append(null_part)
        if args.diffusion_null in {"degree_preserving_graph", "both"}:
            summary_part, tau_part, null_part = degree_preserving_graph_diffusion_null(
                graph_pre.largest_component,
                lcc_scores,
                trait=trait,
                graph_name=str(args.graph_name),
                score_mode=args.diffusion_score_mode,
                tau_grid=tau_grid,
                n_replicates=args.n_diffusion_null,
                seed=args.seed + 1201,
                nswap_per_edge=args.degree_graph_nswap_per_edge,
                max_tries_per_swap=args.degree_graph_max_tries_per_swap,
                cache_path=args.degree_graph_cache,
            )
            diffusion_summaries.append(summary_part)
            diffusion_tau_summaries.append(tau_part)
            diffusion_nulls.append(null_part)
        diffusion_summary_table = pd.concat(diffusion_summaries, ignore_index=True) if diffusion_summaries else pd.DataFrame()
        diffusion_tau_table = pd.concat(diffusion_tau_summaries, ignore_index=True) if diffusion_tau_summaries else pd.DataFrame()
        diffusion_null_distribution = pd.concat(diffusion_nulls, ignore_index=True) if diffusion_nulls else pd.DataFrame()
        write_table(diffusion_summary_path, diffusion_summary_table)
        write_table(diffusion_tau_path, diffusion_tau_table)
        if args.save_null_distributions:
            write_table(diffusion_null_path, diffusion_null_distribution)
    expected_null_auc = float(null_auc_table["percolation_auc"].mean()) if not null_auc_table.empty else float("nan")
    delta_perc = observed_auc - expected_null_auc
    identity_status_counts = {
        key: value for key, value in score_report["ld_status_counts"].items() if key.startswith("identity_fallback")
    }
    remaining_required_work: list[str] = []
    if identity_status_counts:
        remaining_required_work.append("replace remaining identity LD fallback genes with full or block LD sensitivity")
    if args.graph_family_maxt:
        remaining_required_work.append("graph-family maxT requested but not implemented as a V1 default output")
    if args.weighted_laplacian and args.diffusion_null in {"degree_preserving_graph", "both"}:
        remaining_required_work.append(
            "weighted diffusion was requested; weighted topology-specific graph nulls are not implemented in V1"
        )

    print("Running local weak-signal module discovery", flush=True)
    gene_sets = load_gene_sets(
        args.gene_set_file,
        include_default_dr_panel=not args.no_default_dr_panel,
    )
    local_modules, local_module_nulls, pathway_tests = run_local_module_discovery(
        graph_pre.largest_component,
        lcc_scores,
        gene_sets=gene_sets,
        seed=args.seed + 707,
        min_module_size=args.min_module_size,
        min_subthreshold_genes=args.min_module_subthreshold_genes,
        max_modules=args.max_local_modules,
        n_module_random=args.n_module_random_null,
        n_module_degree_matched=args.n_module_degree_matched_null,
        n_module_degree_graph=args.n_module_degree_graph_null,
        n_module_selection_aware=args.n_module_selection_aware_null,
        selection_null_scores=null_resid,
        n_pathway_random=args.n_pathway_random_null,
        n_pathway_degree_matched=args.n_pathway_degree_matched_null,
        degree_bins=args.degree_matched_bins,
    )
    global_module_gate_pass = bool(architecture.get("degree_matched_node_positive", False))
    if not local_modules.empty:
        local_modules = local_modules.copy()
        is_local_component = local_modules["n_genes"].astype(int) < 200
        local_modules["passes_global_module_gate"] = global_module_gate_pass
        local_modules["is_reportable_calibrated_module"] = (
            local_modules["is_calibrated_weak_signal_module"].astype(bool) & is_local_component & global_module_gate_pass
        )
        local_modules["is_reportable_topology_specific_module"] = (
            local_modules["is_topology_specific_module"].astype(bool) & is_local_component & global_module_gate_pass
        )
    local_module_path = tables_dir / f"{trait}.local_modules.tsv"
    local_module_null_path = tables_dir / f"{trait}.local_module_nulls.tsv"
    pathway_path = tables_dir / f"{trait}.pathway_subgraph_tests.tsv"
    write_table(local_module_path, local_modules)
    write_table(local_module_null_path, local_module_nulls)
    write_table(pathway_path, pathway_tests)
    top_modules = (
        local_modules.loc[
            local_modules.get("is_reportable_calibrated_module", pd.Series(False, index=local_modules.index))
            .astype(bool)
            .to_numpy()
        ]
        if not local_modules.empty
        else local_modules
    )
    local_module_summary = {
        "graph_name": str(args.graph_name),
        "global_module_gate_pass": global_module_gate_pass,
        "n_candidate_modules": int(len(local_modules)),
        "n_module_level_calibrated_candidates": int(
            local_modules.get("is_calibrated_weak_signal_module", pd.Series(dtype=bool)).sum()
        )
        if not local_modules.empty
        else 0,
        "n_broad_calibrated_components": int(
            (
                local_modules.get("is_calibrated_weak_signal_module", pd.Series(dtype=bool)).astype(bool)
                & (local_modules.get("n_genes", pd.Series(dtype=int)).astype(int) >= 200)
            ).sum()
        )
        if not local_modules.empty
        else 0,
        "n_calibrated_modules": int(
            local_modules.get("is_reportable_calibrated_module", pd.Series(dtype=bool)).sum()
        )
        if not local_modules.empty
        else 0,
        "n_topology_specific_modules": int(
            local_modules.get("is_reportable_topology_specific_module", pd.Series(dtype=bool)).sum()
        )
        if not local_modules.empty
        else 0,
        "top_modules": top_modules.head(5).to_dict(orient="records") if not top_modules.empty else [],
    }
    claim_tiers = build_claim_tier_table(
        trait=trait,
        graph_name=str(args.graph_name),
        observed_percolation_auc=observed_auc,
        snp_permutation_null=snp_null_summary,
        degree_matched_node_null=degree_matched_summary,
        degree_preserving_graph_null=degree_graph_summary,
        diffusion_summary=diffusion_summary_table,
        local_module_summary=local_module_summary,
    )
    claim_tiers_path = tables_dir / f"{trait}.claim_tiers.tsv"
    write_table(claim_tiers_path, claim_tiers)

    summary = {
        "created_utc": datetime.now(UTC).isoformat(),
        "trait": trait,
        "mode": "trait_ld_cached_with_snp_permutation_and_degree_nulls",
        "gwas": str(gwas_path),
        "mapping": str(args.mapping or tables_dir / f"{trait}.gene_body_mapping.tsv.gz"),
        "ld_cache_dir": str(args.ld_cache_dir),
        "ld_cache_overlay_dirs": [str(path) for path in overlay_dirs],
        "ld_cache_dirs_in_priority_order": [str(path) for path in ld_cache_dirs],
        "ld_score_payload_cache": str(args.ld_score_payload_cache or tables_dir / f"{trait}.ld_scoring_payload_cache.npz"),
        "ld_shrinkage": args.ld_shrinkage,
        "quadratic_method": args.quadratic_method,
        "n_null": args.n_null,
        "n_degree_stratified_null": args.n_degree_stratified_null,
        "n_degree_matched_node_null": args.n_degree_matched_node_null,
        "n_degree_graph_null": args.n_degree_graph_null,
        "degree_graph_cache": str(args.degree_graph_cache) if args.degree_graph_cache is not None else None,
        "enable_diffusion": bool(args.enable_diffusion),
        "diffusion_score_mode": args.diffusion_score_mode,
        "diffusion_tau_grid": args.tau_grid,
        "diffusion_null": args.diffusion_null,
        "n_diffusion_null": args.n_diffusion_null,
        "diffusion_batch_size": args.diffusion_batch_size,
        "weighted_laplacian": bool(args.weighted_laplacian),
        "graph_family_maxt_requested": bool(args.graph_family_maxt),
        "graph_name": str(args.graph_name),
        "graph_edge_list": str(args.graph_edge_list) if args.graph_edge_list is not None else None,
        "analysis_graph_edges": str(analysis_graph_edges_path),
        "graph_load_report": asdict(graph_edges.report),
        "n_module_random_null": args.n_module_random_null,
        "n_module_degree_matched_null": args.n_module_degree_matched_null,
        "n_module_degree_graph_null": args.n_module_degree_graph_null,
        "n_module_selection_aware_null": args.n_module_selection_aware_null,
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
        "diffusion_kernel_summary": diffusion_summary_table.to_dict(orient="records")
        if not diffusion_summary_table.empty
        else [],
        "percolation_architecture": architecture,
        "claim_tiers": claim_tiers.to_dict(orient="records"),
        "local_module_summary": local_module_summary,
        "p_clipping_summary": asdict(clipping),
        "outputs": {
            "gene_scores": str(tables_dir / f"{trait}.gene_scores.1000G_LD.tsv.gz"),
            "lcc_gene_scores": str(tables_dir / f"{trait}.lcc_gene_scores.1000G_LD.residualized.tsv.gz"),
            "null_assoc_scores": str(tables_dir / f"{trait}.null_assoc_normal_scores.npz"),
            "null_residualized_scores": str(tables_dir / f"{trait}.null_residualized_scores.npz"),
            "analysis_graph_edges": str(analysis_graph_edges_path),
            "observed_percolation_curve": str(tables_dir / f"{trait}.percolation_curve.1000G_LD.observed.tsv"),
            "null_percolation_auc": str(tables_dir / f"{trait}.percolation_auc.1000G_LD.null.tsv"),
            "degree_stratified_null_auc": str(tables_dir / f"{trait}.percolation_auc.1000G_LD.degree_stratified_null.tsv"),
            "degree_matched_node_null_auc": str(tables_dir / f"{trait}.percolation_auc.1000G_LD.degree_matched_node_null.tsv"),
            "degree_preserving_graph_null_auc": str(
                tables_dir / f"{trait}.percolation_auc.1000G_LD.degree_preserving_graph_null.tsv"
            ),
            "percolation_threshold_robustness": str(percolation_robustness_path),
            "claim_tiers": str(claim_tiers_path),
            "gene_score_clipping_diagnostics": str(tables_dir / f"{trait}.gene_score_clipping_diagnostics.tsv"),
            "residualization_diagnostics": str(tables_dir / f"{trait}.residualization_diagnostics.tsv"),
            "gene_score_transform_sensitivity": str(tables_dir / f"{trait}.gene_score_transform_sensitivity.tsv")
            if args.score_transform_sensitivity
            else None,
            "diffusion_kernel_summary": str(diffusion_summary_path) if args.enable_diffusion else None,
            "diffusion_kernel_tau_stats": str(diffusion_tau_path) if args.enable_diffusion else None,
            "diffusion_kernel_null_distribution": str(diffusion_null_path)
            if args.enable_diffusion and args.save_null_distributions
            else None,
            "local_modules": str(local_module_path),
            "local_module_nulls": str(local_module_null_path),
            "pathway_subgraph_tests": str(pathway_path),
        },
        "remaining_required_work": remaining_required_work,
    }
    trait_suitability = build_trait_suitability_diagnostic(summary)
    summary["trait_suitability"] = trait_suitability
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_text = json.dumps(summary, indent=2, sort_keys=True)
    (reports_dir / f"{trait}.analysis_ready_summary.json").write_text(summary_text, encoding="utf-8")
    (reports_dir / f"{trait}.trait_suitability.json").write_text(
        json.dumps(trait_suitability, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    architecture_report = render_trait_architecture_markdown(summary)
    (reports_dir / f"{trait}.architecture_report.md").write_text(architecture_report, encoding="utf-8")
    (reports_dir / f"{trait}.analysis_ready_report.md").write_text(architecture_report, encoding="utf-8")
    module_report = render_module_discovery_report(
        trait=trait,
        graph_name=str(args.graph_name),
        modules=local_modules,
        pathway=pathway_tests,
        global_gate_pass=global_module_gate_pass,
    )
    (reports_dir / f"{trait}.module_discovery_report.md").write_text(module_report, encoding="utf-8")
    print(f"Wrote trait analysis outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
