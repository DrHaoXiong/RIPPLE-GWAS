#!/usr/bin/env python
"""Run DR_MVP trait-specificity diagnostics against metabolic covariates.

This is a labeled diagnostic script, not a replacement for the primary
analysis-ready pipeline. It asks whether the DR_MVP RIPPLE signal remains after
gene-level residualization against T2D and/or BMI residualized gene scores.

Default output:
    /path/to/ripple_private_workspace/30_analysis/dr_mvp_trait_specificity
"""

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

from ripple.defaults import RANK_FRACTION_GRID  # noqa: E402
from ripple.diagnostics import build_trait_suitability_diagnostic, render_trait_architecture_markdown  # noqa: E402
from ripple.graph import graph_laplacian  # noqa: E402
from ripple.gsp import band_energy_table, laplacian_eigendecomposition, project_graph_signal  # noqa: E402
from ripple.modules import load_gene_sets, render_module_discovery_report, run_local_module_discovery  # noqa: E402
from ripple.percolation import (  # noqa: E402
    classify_percolation_architecture,
    compute_degree_matched_node_percolation_null,
    percolation_auc,
    percolation_curve,
    prepare_degree_matched_rank_sets,
    rank_nodes_by_score,
    summarize_percolation_null,
)
from run_height_ld_null_mvp import (  # noqa: E402
    compute_degree_preserving_graph_percolation_null,
    compute_degree_stratified_percolation_null,
)
from run_height_mvp import DEFAULT_STRING_INFO, DEFAULT_STRING_LINKS, build_string_graph  # noqa: E402

PRIVATE_ROOT = Path("/path/to/ripple_private_workspace")
ANALYSIS_ROOT = PRIVATE_ROOT / "30_analysis"

DEFAULT_ANALYSES = {
    "DR_MVP": ANALYSIS_ROOT / "dr_mvp_analysis_ready",
    "T2D": ANALYSIS_ROOT / "t2d_analysis_ready",
    "BMI_IRN": ANALYSIS_ROOT / "bmi_irn_analysis_ready",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-trait", default="DR_MVP")
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_ANALYSES["DR_MVP"])
    parser.add_argument("--t2d-dir", type=Path, default=DEFAULT_ANALYSES["T2D"])
    parser.add_argument("--bmi-dir", type=Path, default=DEFAULT_ANALYSES["BMI_IRN"])
    parser.add_argument("--out-dir", type=Path, default=ANALYSIS_ROOT / "dr_mvp_trait_specificity")
    parser.add_argument("--string-links", type=Path, default=DEFAULT_STRING_LINKS)
    parser.add_argument("--string-info", type=Path, default=DEFAULT_STRING_INFO)
    parser.add_argument("--string-min-score", type=int, default=400)
    parser.add_argument("--n-degree-stratified-null", type=int, default=100)
    parser.add_argument("--n-degree-matched-node-null", type=int, default=500)
    parser.add_argument("--n-degree-graph-null", type=int, default=20)
    parser.add_argument("--n-module-random-null", type=int, default=200)
    parser.add_argument("--n-module-degree-matched-null", type=int, default=200)
    parser.add_argument("--n-module-degree-graph-null", type=int, default=20)
    parser.add_argument("--n-module-selection-aware-null", type=int, default=100)
    parser.add_argument("--n-pathway-random-null", type=int, default=200)
    parser.add_argument("--n-pathway-degree-matched-null", type=int, default=200)
    parser.add_argument("--degree-bins", type=int, default=10)
    parser.add_argument("--eigen-components", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def trait_table_path(analysis_dir: Path, trait: str) -> Path:
    return analysis_dir / "tables" / f"{trait}.lcc_gene_scores.1000G_LD.residualized.tsv.gz"


def null_scores_path(analysis_dir: Path, trait: str) -> Path:
    return analysis_dir / "tables" / f"{trait}.null_residualized_scores.npz"


def write_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if path.suffix == ".gz" else None
    table.to_csv(path, sep="\t", index=False, compression=compression)


def ensure_output_dir(path: Path, *, force: bool) -> None:
    if path.exists() and any(path.iterdir()) and not force:
        raise FileExistsError(f"Refusing to overwrite non-empty output directory without --force: {path}")
    path.mkdir(parents=True, exist_ok=True)
    for child in ("tables", "reports"):
        (path / child).mkdir(parents=True, exist_ok=True)


def load_lcc_scores(analysis_dir: Path, trait: str) -> pd.DataFrame:
    path = trait_table_path(analysis_dir, trait)
    if not path.exists():
        raise FileNotFoundError(path)
    table = pd.read_csv(path, sep="\t", compression="infer")
    table["gene_symbol"] = table["gene_symbol"].astype(str)
    table["assoc_resid_score"] = pd.to_numeric(table["assoc_resid_score"], errors="raise").astype(float)
    table["graph_degree"] = pd.to_numeric(table["graph_degree"], errors="raise").astype(int)
    return table


def load_null_scores(analysis_dir: Path, trait: str, gene_symbols: pd.Series) -> np.ndarray:
    path = null_scores_path(analysis_dir, trait)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        null_resid = np.asarray(data["null_resid"], dtype=float)
        null_gene_symbols = np.asarray(data["gene_symbols"]).astype(str)
    requested = gene_symbols.astype(str).to_numpy()
    if np.array_equal(null_gene_symbols, requested):
        return null_resid
    index_by_gene = {gene: idx for idx, gene in enumerate(null_gene_symbols)}
    missing = [gene for gene in requested if gene not in index_by_gene]
    if missing:
        raise ValueError(f"Null residualized score matrix is missing {len(missing)} requested genes.")
    order = np.array([index_by_gene[gene] for gene in requested], dtype=int)
    return null_resid[:, order]


def standardize_columns(table: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    values = table[columns].to_numpy(dtype=float).copy()
    stats: dict[str, dict[str, float]] = {}
    for idx, col in enumerate(columns):
        mean = float(np.mean(values[:, idx]))
        sd = float(np.std(values[:, idx], ddof=1))
        if sd <= 0 or not np.isfinite(sd):
            sd = 1.0
        values[:, idx] = (values[:, idx] - mean) / sd
        stats[col] = {"mean": mean, "sd": sd}
    return values, stats


def residualize_against_covariates(
    observed: np.ndarray,
    null_scores: np.ndarray,
    covariates: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    covariate_cols = list(covariates.columns)
    if not covariate_cols:
        return observed.copy(), null_scores.copy(), {"covariates": [], "observed_r2": 0.0}
    x_cov, cov_stats = standardize_columns(covariates, covariate_cols)
    design = np.column_stack([np.ones(len(covariates)), x_cov])
    pinv = np.linalg.pinv(design)

    beta_obs = pinv @ observed
    fitted_obs = design @ beta_obs
    residual_obs = observed - fitted_obs
    ss_total = float(np.sum((observed - np.mean(observed)) ** 2))
    ss_resid = float(np.sum(residual_obs**2))
    observed_r2 = float(1.0 - (ss_resid / ss_total)) if ss_total > 0 else float("nan")

    beta_null = pinv @ null_scores.T
    fitted_null = (design @ beta_null).T
    residual_null = null_scores - fitted_null
    return residual_obs, residual_null, {
        "covariates": covariate_cols,
        "covariate_standardization": cov_stats,
        "coefficients": dict(zip(["intercept", *covariate_cols], beta_obs.astype(float), strict=True)),
        "observed_r2": observed_r2,
    }


def add_reportable_module_flags(modules: pd.DataFrame, *, global_gate_pass: bool) -> pd.DataFrame:
    if modules.empty:
        return modules
    out = modules.copy()
    is_local_component = out["n_genes"].astype(int) < 200
    out["passes_global_module_gate"] = bool(global_gate_pass)
    out["is_reportable_calibrated_module"] = (
        out["is_calibrated_weak_signal_module"].astype(bool) & is_local_component & bool(global_gate_pass)
    )
    out["is_reportable_topology_specific_module"] = (
        out["is_topology_specific_module"].astype(bool) & is_local_component & bool(global_gate_pass)
    )
    return out


def summarize_modules(graph_name: str, modules: pd.DataFrame, *, global_gate_pass: bool) -> dict[str, object]:
    if modules.empty:
        return {
            "graph_name": graph_name,
            "global_module_gate_pass": bool(global_gate_pass),
            "n_candidate_modules": 0,
            "n_module_level_calibrated_candidates": 0,
            "n_broad_calibrated_components": 0,
            "n_calibrated_modules": 0,
            "n_topology_specific_modules": 0,
            "top_modules": [],
        }
    reportable = modules.loc[modules["is_reportable_calibrated_module"].astype(bool)].copy()
    return {
        "graph_name": graph_name,
        "global_module_gate_pass": bool(global_gate_pass),
        "n_candidate_modules": int(len(modules)),
        "n_module_level_calibrated_candidates": int(modules["is_calibrated_weak_signal_module"].sum()),
        "n_broad_calibrated_components": int(
            (modules["is_calibrated_weak_signal_module"].astype(bool) & (modules["n_genes"].astype(int) >= 200)).sum()
        ),
        "n_calibrated_modules": int(modules["is_reportable_calibrated_module"].sum()),
        "n_topology_specific_modules": int(modules["is_reportable_topology_specific_module"].sum()),
        "top_modules": reportable.head(5).to_dict(orient="records") if not reportable.empty else [],
    }


def compute_score_similarity(aligned: pd.DataFrame, *, target_col: str, covariate_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for covariate_col in covariate_cols:
        pearson = aligned[[target_col, covariate_col]].corr(method="pearson").iloc[0, 1]
        spearman = aligned[[target_col, covariate_col]].corr(method="spearman").iloc[0, 1]
        rows.append(
            {
                "covariate": covariate_col,
                "pearson": float(pearson),
                "spearman": float(spearman),
                "n_genes": int(len(aligned)),
            }
        )
    return pd.DataFrame(rows)


def compute_top_overlap(aligned: pd.DataFrame, *, target_col: str, covariate_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fraction in RANK_FRACTION_GRID:
        n_top = max(1, int(round(float(fraction) * len(aligned))))
        target_top = set(aligned.nlargest(n_top, target_col)["gene_symbol"])
        for covariate_col in covariate_cols:
            cov_top = set(aligned.nlargest(n_top, covariate_col)["gene_symbol"])
            intersection = target_top & cov_top
            union = target_top | cov_top
            rows.append(
                {
                    "rank_fraction": float(fraction),
                    "n_top": int(n_top),
                    "covariate": covariate_col,
                    "n_overlap": int(len(intersection)),
                    "jaccard": float(len(intersection) / len(union)) if union else 0.0,
                    "overlap_genes": ",".join(sorted(intersection)),
                }
            )
    return pd.DataFrame(rows)


def run_mode(
    *,
    mode_name: str,
    mode_label: str,
    base_scores: pd.DataFrame,
    observed_scores: np.ndarray,
    null_scores: np.ndarray,
    residualization_info: dict[str, object],
    graph,
    graph_coverage_report: dict[str, object],
    graph_load_report: dict[str, object],
    decomp,
    laplacian,
    gene_sets: dict[str, set[str]],
    args: argparse.Namespace,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores = base_scores.copy()
    scores["assoc_resid_score"] = observed_scores
    scores["graph_degree"] = scores["gene_symbol"].map(dict(graph.degree())).astype(int)

    signal = project_graph_signal(scores["assoc_resid_score"].to_numpy(dtype=float), decomp, laplacian=laplacian)
    band_table = band_energy_table(signal.eigenvalues, signal.coefficients)

    ranking = rank_nodes_by_score(scores, node_col="gene_symbol", score_col="assoc_resid_score")
    observed_curve = percolation_curve(graph, ranking, RANK_FRACTION_GRID, node_col="gene_symbol")
    observed_auc = percolation_auc(observed_curve)

    null_auc_rows: list[dict[str, float | int]] = []
    null_curve_rows: list[pd.DataFrame] = []
    null_base = scores.loc[:, ["gene_symbol"]].copy()
    for idx in range(null_scores.shape[0]):
        null_table = null_base.copy()
        null_table["assoc_resid_score"] = null_scores[idx, :]
        null_ranking = rank_nodes_by_score(null_table, node_col="gene_symbol", score_col="assoc_resid_score")
        null_curve = percolation_curve(graph, null_ranking, RANK_FRACTION_GRID, node_col="gene_symbol")
        null_auc_rows.append({"replicate": idx, "percolation_auc": percolation_auc(null_curve)})
        null_curve["replicate"] = idx
        null_curve_rows.append(null_curve)
    null_auc_table = pd.DataFrame(null_auc_rows)
    null_curve_table = pd.concat(null_curve_rows, ignore_index=True) if null_curve_rows else pd.DataFrame()

    degree_strat_auc_table, degree_strat_curve_table = compute_degree_stratified_percolation_null(
        graph,
        scores,
        n_replicates=args.n_degree_stratified_null,
        seed=args.seed + 101,
        n_bins=args.degree_bins,
    )
    selected_bin_counts, bin_to_nodes, degree_profile_table = prepare_degree_matched_rank_sets(
        scores,
        ranking,
        RANK_FRACTION_GRID,
        node_col="gene_symbol",
        degree_col="graph_degree",
        n_bins=args.degree_bins,
    )
    degree_matched_auc_table, degree_matched_curve_table = compute_degree_matched_node_percolation_null(
        graph,
        selected_bin_counts,
        bin_to_nodes,
        n_replicates=args.n_degree_matched_node_null,
        seed=args.seed + 303,
    )
    degree_graph_auc_table, degree_graph_curve_table, degree_graph_diagnostics = (
        compute_degree_preserving_graph_percolation_null(
            graph,
            ranking,
            n_replicates=args.n_degree_graph_null,
            seed=args.seed + 202,
            nswap_per_edge=1.0,
            max_tries_per_swap=20.0,
        )
    )

    snp_summary = summarize_percolation_null(null_auc_table, observed_auc)
    degree_strat_summary = summarize_percolation_null(degree_strat_auc_table, observed_auc)
    degree_matched_summary = summarize_percolation_null(degree_matched_auc_table, observed_auc)
    degree_graph_summary = summarize_percolation_null(degree_graph_auc_table, observed_auc)
    architecture = classify_percolation_architecture(
        snp_permutation_null=snp_summary,
        degree_stratified_null=degree_strat_summary,
        degree_matched_node_null=degree_matched_summary,
        degree_preserving_graph_null=degree_graph_summary,
    )

    local_modules, local_module_nulls, pathway_tests = run_local_module_discovery(
        graph,
        scores,
        gene_sets=gene_sets,
        seed=args.seed + 707,
        n_module_random=args.n_module_random_null,
        n_module_degree_matched=args.n_module_degree_matched_null,
        n_module_degree_graph=args.n_module_degree_graph_null,
        n_module_selection_aware=args.n_module_selection_aware_null,
        selection_null_scores=null_scores,
        n_pathway_random=args.n_pathway_random_null,
        n_pathway_degree_matched=args.n_pathway_degree_matched_null,
        degree_bins=args.degree_bins,
    )
    global_gate_pass = bool(architecture.get("degree_matched_node_positive", False))
    local_modules = add_reportable_module_flags(local_modules, global_gate_pass=global_gate_pass)
    local_module_summary = summarize_modules("string_ppi", local_modules, global_gate_pass=global_gate_pass)
    summary: dict[str, object] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "diagnostic": "trait_specificity",
        "mode_name": mode_name,
        "mode_label": mode_label,
        "trait": f"{args.target_trait}_{mode_name}",
        "target_trait": args.target_trait,
        "residualization_against_covariates": residualization_info,
        "n_null": int(null_scores.shape[0]),
        "n_lcc_scored_genes": int(len(scores)),
        "graph_name": "string_ppi",
        "graph_load_report": graph_load_report,
        "graph_coverage_report": graph_coverage_report,
        "gsp_method": decomp.method,
        "gsp_smoothness": signal.smoothness,
        "gsp_retained_energy_fraction": signal.retained_energy_fraction,
        "percolation_auc_observed": observed_auc,
        "percolation_auc_null_mean": float(null_auc_table["percolation_auc"].mean()),
        "percolation_auc_null_sd": snp_summary["sd"],
        "delta_perc": float(observed_auc - float(null_auc_table["percolation_auc"].mean())),
        "snp_permutation_null_summary": snp_summary,
        "degree_stratified_null_summary": degree_strat_summary,
        "degree_matched_node_null_summary": degree_matched_summary,
        "degree_preserving_graph_null_summary": degree_graph_summary,
        "percolation_architecture": architecture,
        "local_module_summary": local_module_summary,
        "p_clipping_summary": {
            "n_clipped": int(scores["is_p_clipped"].sum()) if "is_p_clipped" in scores.columns else 0,
            "n_total": int(len(scores)),
            "fraction_clipped": float(scores["is_p_clipped"].mean()) if "is_p_clipped" in scores.columns else 0.0,
        },
    }
    summary["trait_suitability"] = build_trait_suitability_diagnostic(summary)

    mode_dir = args.out_dir / "tables" / mode_name
    write_table(mode_dir / f"{mode_name}.scores.tsv.gz", scores)
    write_table(mode_dir / f"{mode_name}.gsp_band_energy.tsv", band_table)
    write_table(mode_dir / f"{mode_name}.percolation_curve.observed.tsv", observed_curve)
    write_table(mode_dir / f"{mode_name}.percolation_auc.null.tsv", null_auc_table)
    write_table(mode_dir / f"{mode_name}.percolation_curves.null.tsv", null_curve_table)
    write_table(mode_dir / f"{mode_name}.percolation_auc.degree_stratified_null.tsv", degree_strat_auc_table)
    write_table(mode_dir / f"{mode_name}.percolation_curves.degree_stratified_null.tsv", degree_strat_curve_table)
    write_table(mode_dir / f"{mode_name}.percolation_auc.degree_matched_node_null.tsv", degree_matched_auc_table)
    write_table(mode_dir / f"{mode_name}.percolation_curves.degree_matched_node_null.tsv", degree_matched_curve_table)
    write_table(mode_dir / f"{mode_name}.degree_profile.by_rank_fraction.tsv", degree_profile_table)
    write_table(mode_dir / f"{mode_name}.percolation_auc.degree_preserving_graph_null.tsv", degree_graph_auc_table)
    write_table(mode_dir / f"{mode_name}.percolation_curves.degree_preserving_graph_null.tsv", degree_graph_curve_table)
    write_table(mode_dir / f"{mode_name}.degree_preserving_graph_null.diagnostics.tsv", degree_graph_diagnostics)
    write_table(mode_dir / f"{mode_name}.local_modules.tsv", local_modules)
    write_table(mode_dir / f"{mode_name}.local_module_nulls.tsv", local_module_nulls)
    write_table(mode_dir / f"{mode_name}.pathway_subgraph_tests.tsv", pathway_tests)
    (args.out_dir / "reports" / f"{mode_name}.summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.out_dir / "reports" / f"{mode_name}.architecture_report.md").write_text(
        render_trait_architecture_markdown(summary),
        encoding="utf-8",
    )
    (args.out_dir / "reports" / f"{mode_name}.module_discovery_report.md").write_text(
        render_module_discovery_report(
            trait=f"{args.target_trait}_{mode_name}",
            graph_name="string_ppi",
            modules=local_modules,
            pathway=pathway_tests,
            global_gate_pass=global_gate_pass,
        ),
        encoding="utf-8",
    )
    return summary, local_modules, local_module_nulls, pathway_tests, scores


def summary_row(summary: dict[str, object]) -> dict[str, object]:
    suitability = summary["trait_suitability"]
    modules = summary["local_module_summary"]
    return {
        "mode_name": summary["mode_name"],
        "mode_label": summary["mode_label"],
        "covariates": ",".join(summary["residualization_against_covariates"].get("covariates", [])),
        "covariate_observed_r2": summary["residualization_against_covariates"].get("observed_r2", 0.0),
        "architecture_class": summary["percolation_architecture"]["architecture_class"],
        "suitability_verdict": suitability["verdict"],
        "snp_null_z": summary["snp_permutation_null_summary"]["z"],
        "degree_stratified_z": summary["degree_stratified_null_summary"]["z"],
        "degree_matched_z": summary["degree_matched_node_null_summary"]["z"],
        "degree_preserving_graph_z": summary["degree_preserving_graph_null_summary"]["z"],
        "observed_auc": summary["percolation_auc_observed"],
        "delta_perc": summary["delta_perc"],
        "n_candidate_modules": modules["n_candidate_modules"],
        "n_module_level_calibrated_candidates": modules["n_module_level_calibrated_candidates"],
        "n_broad_calibrated_components": modules["n_broad_calibrated_components"],
        "n_reportable_modules": modules["n_calibrated_modules"],
        "n_topology_specific_modules": modules["n_topology_specific_modules"],
    }


def render_report(summary_table: pd.DataFrame, similarity: pd.DataFrame, top_overlap: pd.DataFrame) -> str:
    lines = [
        "# DR_MVP Trait Specificity Diagnostic",
        "",
        "Purpose: test whether DR_MVP RIPPLE signal remains after gene-level residualization against T2D and/or BMI.",
        "",
        "## Mode Summary",
        "",
        "| Mode | Covariates | R2 removed | Degree Z | Graph Z | Architecture | Reportable modules | Broad components |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for row in summary_table.itertuples(index=False):
        lines.append(
            "| "
            f"{row.mode_name} | {row.covariates or 'none'} | {float(row.covariate_observed_r2):.3f} | "
            f"{float(row.degree_matched_z):.3f} | {float(row.degree_preserving_graph_z):.3f} | "
            f"{row.architecture_class} | {int(row.n_reportable_modules)} | "
            f"{int(row.n_broad_calibrated_components)} |"
        )
    lines.extend(["", "## Baseline Cross-Trait Similarity", ""])
    lines.extend(["| Covariate | Pearson | Spearman | n genes |", "|---|---:|---:|---:|"])
    for row in similarity.itertuples(index=False):
        lines.append(f"| {row.covariate} | {float(row.pearson):.3f} | {float(row.spearman):.3f} | {int(row.n_genes)} |")
    lines.extend(["", "## Top-Rank Overlap With DR_MVP", ""])
    lines.extend(["| Fraction | Covariate | n overlap | Jaccard |", "|---:|---|---:|---:|"])
    for row in top_overlap.itertuples(index=False):
        lines.append(
            f"| {float(row.rank_fraction):.2f} | {row.covariate} | {int(row.n_overlap)} | {float(row.jaccard):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "- If degree-matched Z and reportable modules collapse after T2D/BMI residualization, the original signal is largely metabolic-background driven.",
            "- If signal remains with different core genes/pathways, it supports a DR-specific residual component.",
            "- If topology-specific modules remain zero, claims should stay at calibrated weak-signal module or degree-aware aggregation level.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.out_dir, force=args.force)
    tables_dir = args.out_dir / "tables"
    reports_dir = args.out_dir / "reports"

    target = load_lcc_scores(args.target_dir, args.target_trait)
    t2d = load_lcc_scores(args.t2d_dir, "T2D").loc[:, ["gene_symbol", "assoc_resid_score"]].rename(
        columns={"assoc_resid_score": "T2D"}
    )
    bmi = load_lcc_scores(args.bmi_dir, "BMI_IRN").loc[:, ["gene_symbol", "assoc_resid_score"]].rename(
        columns={"assoc_resid_score": "BMI_IRN"}
    )
    aligned = (
        target.merge(t2d, on="gene_symbol", how="inner")
        .merge(bmi, on="gene_symbol", how="inner")
        .sort_values("gene_symbol")
        .reset_index(drop=True)
    )
    graph_args = argparse.Namespace(
        string_links=args.string_links,
        string_info=args.string_info,
        string_min_score=args.string_min_score,
    )
    graph_edges, graph_pre = build_string_graph(graph_args, tuple(aligned["gene_symbol"]))
    lcc_nodes = set(str(node) for node in graph_pre.largest_component.nodes())
    aligned = aligned[aligned["gene_symbol"].isin(lcc_nodes)].sort_values("gene_symbol").reset_index(drop=True)
    target_null = load_null_scores(args.target_dir, args.target_trait, aligned["gene_symbol"])
    degree = dict(graph_pre.largest_component.degree())
    aligned["graph_degree"] = aligned["gene_symbol"].map(degree).astype(int)

    nodes = tuple(aligned["gene_symbol"].astype(str))
    lap = graph_laplacian(graph_pre.largest_component, nodes=nodes, kind="normalized")
    decomp = laplacian_eigendecomposition(
        lap.laplacian,
        nodes=nodes,
        n_components=min(args.eigen_components, max(1, len(nodes) - 2)),
    )
    gene_sets = load_gene_sets()
    graph_coverage_report = asdict(graph_pre.coverage_report)
    graph_load_report = asdict(graph_edges.report)
    base_scores = aligned.drop(columns=["T2D", "BMI_IRN"]).copy()
    observed = aligned["assoc_resid_score"].to_numpy(dtype=float)

    similarity = compute_score_similarity(aligned, target_col="assoc_resid_score", covariate_cols=["T2D", "BMI_IRN"])
    top_overlap = compute_top_overlap(aligned, target_col="assoc_resid_score", covariate_cols=["T2D", "BMI_IRN"])
    write_table(tables_dir / "cross_trait_score_similarity.tsv", similarity)
    write_table(tables_dir / "top_rank_overlap.tsv", top_overlap)

    modes = {
        "baseline_common": ("No cross-trait residualization on common LCC genes", []),
        "resid_t2d": ("DR_MVP residualized against T2D gene scores", ["T2D"]),
        "resid_bmi": ("DR_MVP residualized against BMI gene scores", ["BMI_IRN"]),
        "resid_t2d_bmi": ("DR_MVP residualized against T2D and BMI gene scores", ["T2D", "BMI_IRN"]),
    }
    summaries: list[dict[str, object]] = []
    for mode_name, (mode_label, covariate_cols) in modes.items():
        print(f"Running specificity mode: {mode_name}", flush=True)
        if covariate_cols:
            obs_mode, null_mode, residualization = residualize_against_covariates(
                observed,
                target_null,
                aligned.loc[:, covariate_cols],
            )
        else:
            obs_mode = observed.copy()
            null_mode = target_null.copy()
            residualization = {"covariates": [], "observed_r2": 0.0}
        summary, _, _, _, _ = run_mode(
            mode_name=mode_name,
            mode_label=mode_label,
            base_scores=base_scores,
            observed_scores=obs_mode,
            null_scores=null_mode,
            residualization_info=residualization,
            graph=graph_pre.largest_component,
            graph_coverage_report=graph_coverage_report,
            graph_load_report=graph_load_report,
            decomp=decomp,
            laplacian=lap.laplacian,
            gene_sets=gene_sets,
            args=args,
        )
        summaries.append(summary)

    summary_table = pd.DataFrame(summary_row(summary) for summary in summaries)
    write_table(tables_dir / "trait_specificity_summary.tsv", summary_table)
    (reports_dir / "trait_specificity_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (reports_dir / "trait_specificity_report.md").write_text(
        render_report(summary_table, similarity, top_overlap),
        encoding="utf-8",
    )
    print(f"Wrote trait-specificity diagnostic outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
