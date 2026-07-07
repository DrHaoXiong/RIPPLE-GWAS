#!/usr/bin/env python
"""Diagnose why DR degree-matched percolation is weak."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_height_mvp import DEFAULT_STRING_INFO, DEFAULT_STRING_LINKS, build_string_graph  # noqa: E402

PRIVATE_ROOT = Path("/path/to/ripple_private_workspace")
ANALYSIS_ROOT = PRIVATE_ROOT / "30_analysis"


TRAIT_SUMMARIES = {
    "HEIGHT_IRN": ANALYSIS_ROOT
    / "height_irn_analysis_ready"
    / "reports"
    / "HEIGHT_IRN.analysis_ready_summary.json",
    "BMI_IRN": ANALYSIS_ROOT / "bmi_irn_analysis_ready" / "reports" / "BMI_IRN.analysis_ready_summary.json",
    "T2D": ANALYSIS_ROOT / "t2d_analysis_ready" / "reports" / "T2D.analysis_ready_summary.json",
    "DM_RETINOPATHY_EXMORE": ANALYSIS_ROOT
    / "dm_retinopathy_exmore_analysis_ready"
    / "reports"
    / "DM_RETINOPATHY_EXMORE.analysis_ready_summary.json",
    "DM_RETINOPATHY_EXMORE_WITH_MHC": ANALYSIS_ROOT
    / "dm_retinopathy_exmore_with_mhc_analysis_ready"
    / "reports"
    / "DM_RETINOPATHY_EXMORE_WITH_MHC.analysis_ready_summary.json",
    "DM_RETINOPATHY_EXMORE_NO_MHC_NO_APOE": ANALYSIS_ROOT
    / "dm_retinopathy_exmore_no_mhc_no_apoe_analysis_ready"
    / "reports"
    / "DM_RETINOPATHY_EXMORE_NO_MHC_NO_APOE.analysis_ready_summary.json",
}

HEIGHT_DEGREE_DIAGNOSTICS = ANALYSIS_ROOT / (
    "height_irn_analysis_ready/reports/HEIGHT_IRN.degree_topology_diagnostics_summary.json"
)


GENE_SCORE_TABLES = {
    "DM_RETINOPATHY_EXMORE": ANALYSIS_ROOT
    / "dm_retinopathy_exmore_analysis_ready"
    / "tables"
    / "DM_RETINOPATHY_EXMORE.lcc_gene_scores.1000G_LD.residualized.tsv.gz",
    "DM_RETINOPATHY_EXMORE_WITH_MHC": ANALYSIS_ROOT
    / "dm_retinopathy_exmore_with_mhc_analysis_ready"
    / "tables"
    / "DM_RETINOPATHY_EXMORE_WITH_MHC.lcc_gene_scores.1000G_LD.residualized.tsv.gz",
    "DM_RETINOPATHY_EXMORE_NO_MHC_NO_APOE": ANALYSIS_ROOT
    / "dm_retinopathy_exmore_no_mhc_no_apoe_analysis_ready"
    / "tables"
    / "DM_RETINOPATHY_EXMORE_NO_MHC_NO_APOE.lcc_gene_scores.1000G_LD.residualized.tsv.gz",
}


PATHWAY_GENE_SETS = {
    "complement": {
        "C1QA",
        "C1QB",
        "C1QC",
        "C1R",
        "C1S",
        "C2",
        "C3",
        "C4A",
        "C4B",
        "C5",
        "C6",
        "C7",
        "C8A",
        "C8B",
        "C8G",
        "C9",
        "CFB",
        "CFH",
        "CFI",
        "CD46",
        "CD55",
        "CD59",
        "SERPING1",
        "MASP1",
        "MASP2",
    },
    "vegf_angiogenesis": {
        "VEGFA",
        "VEGFB",
        "VEGFC",
        "VEGFD",
        "KDR",
        "FLT1",
        "FLT4",
        "PGF",
        "HIF1A",
        "ANGPT1",
        "ANGPT2",
        "TEK",
        "TIE1",
        "NRP1",
        "NRP2",
        "DLL4",
        "NOTCH1",
        "PDGFB",
        "PDGFRB",
        "ESM1",
    },
    "ecm_basement_membrane": {
        "COL1A1",
        "COL1A2",
        "COL4A1",
        "COL4A2",
        "COL4A3",
        "COL4A4",
        "COL4A5",
        "LAMA1",
        "LAMA2",
        "LAMA3",
        "LAMA4",
        "LAMA5",
        "LAMB1",
        "LAMB2",
        "LAMC1",
        "FN1",
        "ELN",
        "MMP2",
        "MMP9",
        "TIMP1",
        "TIMP2",
        "LOX",
        "SPARC",
        "THBS1",
        "ITGA5",
        "ITGB1",
    },
    "endothelial_vascular": {
        "PECAM1",
        "VWF",
        "CDH5",
        "ENG",
        "ESAM",
        "ICAM1",
        "VCAM1",
        "SELE",
        "FLT1",
        "KDR",
        "TEK",
        "ROBO4",
        "CLDN5",
        "PLVAP",
        "EMCN",
        "MCAM",
        "EDN1",
        "NOS3",
        "SOX17",
        "ERG",
    },
}


@dataclass(frozen=True)
class NullSummary:
    z: float
    delta: float
    mean: float
    sd: float
    p_upper: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ANALYSIS_ROOT / "dr_signal_diagnosis")
    parser.add_argument("--string-links", type=Path, default=DEFAULT_STRING_LINKS)
    parser.add_argument("--string-info", type=Path, default=DEFAULT_STRING_INFO)
    parser.add_argument("--string-min-score", type=int, default=400)
    parser.add_argument("--n-random", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260614)
    return parser.parse_args()


def write_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if path.suffix == ".gz" else None
    table.to_csv(path, sep="\t", index=False, compression=compression)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_null(summary: dict[str, object], key: str) -> NullSummary:
    value = summary.get(key, {})
    if not isinstance(value, dict):
        value = {}
    return NullSummary(
        z=float(value.get("z", np.nan)),
        delta=float(value.get("delta", np.nan)),
        mean=float(value.get("mean", np.nan)),
        sd=float(value.get("sd", np.nan)),
        p_upper=float(value.get("empirical_p_upper", np.nan)),
    )


def summary_row(trait: str, summary: dict[str, object]) -> dict[str, object]:
    snp = compact_null(summary, "snp_permutation_null_summary")
    strat = compact_null(summary, "degree_stratified_null_summary")
    matched = compact_null(summary, "degree_matched_node_null_summary")
    graph = compact_null(summary, "degree_preserving_graph_null_summary")
    clipping = summary.get("p_clipping_summary", {})
    if not isinstance(clipping, dict):
        clipping = {}
    arch = summary.get("percolation_architecture", {})
    if not isinstance(arch, dict):
        arch = {}
    return {
        "trait": trait,
        "architecture_class": arch.get("architecture_class", ""),
        "n_gene_scores": int(summary.get("n_gene_scores", 0)),
        "n_lcc_scored_genes": int(summary.get("n_lcc_scored_genes", 0)),
        "observed_auc": float(summary.get("percolation_auc_observed", np.nan)),
        "snp_null_mean_auc": snp.mean,
        "snp_null_z": snp.z,
        "snp_null_delta": snp.delta,
        "degree_stratified_z": strat.z,
        "degree_stratified_delta": strat.delta,
        "degree_matched_z": matched.z,
        "degree_matched_delta": matched.delta,
        "degree_preserving_graph_z": graph.z,
        "degree_preserving_graph_delta": graph.delta,
        "gsp_retained_energy_fraction": float(summary.get("gsp_retained_energy_fraction", np.nan)),
        "p_clipped": int(clipping.get("n_clipped", 0)),
        "p_clipped_fraction": float(clipping.get("fraction_clipped", np.nan)),
    }


def add_rank_columns(scores: pd.DataFrame) -> pd.DataFrame:
    out = scores.copy()
    out["gene_symbol"] = out["gene_symbol"].astype(str)
    out["assoc_resid_score"] = pd.to_numeric(out["assoc_resid_score"], errors="raise")
    out["graph_degree"] = pd.to_numeric(out["graph_degree"], errors="coerce").fillna(0).astype(int)
    out = out.sort_values("assoc_resid_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out["rank_fraction"] = out["rank"] / len(out)
    return out


def empirical_upper(null_values: np.ndarray, observed: float) -> float:
    return float((np.count_nonzero(null_values >= observed) + 1) / (len(null_values) + 1))


def empirical_lower(null_values: np.ndarray, observed: float) -> float:
    return float((np.count_nonzero(null_values <= observed) + 1) / (len(null_values) + 1))


def z_score(observed: float, null_values: np.ndarray) -> float:
    sd = float(np.std(null_values, ddof=1))
    if sd <= 0:
        return float("nan")
    return float((observed - float(np.mean(null_values))) / sd)


def pathway_enrichment_rows(
    trait: str,
    scores: pd.DataFrame,
    rng: np.random.Generator,
    n_random: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    universe = scores["gene_symbol"].to_numpy(dtype=str)
    score_values = scores["assoc_resid_score"].to_numpy(dtype=float)
    rank_values = scores["rank_fraction"].to_numpy(dtype=float)
    score_by_gene = dict(zip(universe, score_values, strict=True))
    rank_by_gene = dict(zip(universe, rank_values, strict=True))
    all_genes = set(universe.tolist())

    for set_name, gene_set in PATHWAY_GENE_SETS.items():
        present = sorted(gene_set & all_genes)
        missing = sorted(gene_set - all_genes)
        n_present = len(present)
        if n_present < 3:
            rows.append(
                {
                    "trait": trait,
                    "gene_set": set_name,
                    "n_query_genes": len(gene_set),
                    "n_present": n_present,
                    "n_missing": len(missing),
                    "observed_mean_score": np.nan,
                    "score_z": np.nan,
                    "score_empirical_p_upper": np.nan,
                    "observed_mean_rank_fraction": np.nan,
                    "rank_enrichment_z": np.nan,
                    "rank_empirical_p_lower": np.nan,
                    "top_present_genes": ",".join(present),
                    "missing_genes": ",".join(missing),
                }
            )
            continue

        present_scores = np.array([score_by_gene[gene] for gene in present], dtype=float)
        present_ranks = np.array([rank_by_gene[gene] for gene in present], dtype=float)
        observed_mean_score = float(np.mean(present_scores))
        observed_mean_rank = float(np.mean(present_ranks))
        random_idx = np.vstack(
            [rng.choice(len(universe), size=n_present, replace=False) for _ in range(n_random)]
        )
        random_mean_scores = score_values[random_idx].mean(axis=1)
        random_mean_ranks = rank_values[random_idx].mean(axis=1)
        top_present = sorted(
            present,
            key=lambda gene: (score_by_gene[gene], -rank_by_gene[gene]),
            reverse=True,
        )[:8]
        rows.append(
            {
                "trait": trait,
                "gene_set": set_name,
                "n_query_genes": len(gene_set),
                "n_present": n_present,
                "n_missing": len(missing),
                "observed_mean_score": observed_mean_score,
                "score_z": z_score(observed_mean_score, random_mean_scores),
                "score_empirical_p_upper": empirical_upper(random_mean_scores, observed_mean_score),
                "observed_mean_rank_fraction": observed_mean_rank,
                "rank_enrichment_z": -z_score(observed_mean_rank, random_mean_ranks),
                "rank_empirical_p_lower": empirical_lower(random_mean_ranks, observed_mean_rank),
                "top_present_genes": ",".join(top_present),
                "missing_genes": ",".join(missing),
            }
        )
    return rows


def component_stats(graph: nx.Graph, nodes: list[str]) -> tuple[int, int, int, float, int]:
    present = [node for node in nodes if node in graph]
    if not present:
        return 0, 0, 0, 0.0, 0
    subgraph = graph.subgraph(present)
    components = list(nx.connected_components(subgraph))
    largest = max((len(component) for component in components), default=0)
    largest_fraction = largest / len(present) if present else 0.0
    return len(present), subgraph.number_of_edges(), len(components), largest_fraction, largest


def subgraph_rows(
    scores_by_trait: dict[str, pd.DataFrame],
    graph: nx.Graph,
    rng: np.random.Generator,
    n_random: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    main_scores = scores_by_trait["DM_RETINOPATHY_EXMORE"]
    graph_nodes = set(graph.nodes())
    all_lcc_genes = [gene for gene in main_scores["gene_symbol"].astype(str).tolist() if gene in graph_nodes]

    for set_name, gene_set in PATHWAY_GENE_SETS.items():
        nodes = sorted(gene_set & set(all_lcc_genes))
        n_present, n_edges, n_components, largest_fraction, largest_size = component_stats(graph, nodes)
        if n_present < 3:
            rows.append(
                {
                    "gene_set": set_name,
                    "n_present_in_main_lcc": n_present,
                    "n_induced_edges": n_edges,
                    "n_components": n_components,
                    "largest_component_size": largest_size,
                    "largest_component_fraction": largest_fraction,
                    "edge_count_z_random": np.nan,
                    "edge_count_empirical_p_upper": np.nan,
                }
            )
            continue

        random_edges = np.empty(n_random, dtype=float)
        for idx in range(n_random):
            sample = rng.choice(all_lcc_genes, size=n_present, replace=False).tolist()
            _, edge_count, _, _, _ = component_stats(graph, sample)
            random_edges[idx] = edge_count
        rows.append(
            {
                "gene_set": set_name,
                "n_present_in_main_lcc": n_present,
                "n_induced_edges": n_edges,
                "n_components": n_components,
                "largest_component_size": largest_size,
                "largest_component_fraction": largest_fraction,
                "edge_count_z_random": z_score(float(n_edges), random_edges),
                "edge_count_empirical_p_upper": empirical_upper(random_edges, float(n_edges)),
            }
        )
    return rows


def top_gene_rows(scores_by_trait: dict[str, pd.DataFrame], n_top: int = 25) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    columns = [
        "gene_symbol",
        "rank",
        "rank_fraction",
        "assoc_resid_score",
        "assoc_p_g",
        "assoc_minuslog10p_g",
        "graph_degree",
        "n_mapped_snps",
        "m_eff",
        "is_special_region",
        "special_region_labels",
    ]
    for trait, scores in scores_by_trait.items():
        available_columns = [column for column in columns if column in scores.columns]
        top = scores.loc[:, available_columns].head(n_top).copy()
        top.insert(0, "trait", trait)
        rows.append(top)
    return pd.concat(rows, ignore_index=True)


def build_markdown_report(
    out_path: Path,
    comparison: pd.DataFrame,
    dr_sensitivity: pd.DataFrame,
    pathway: pd.DataFrame,
    subgraph: pd.DataFrame,
) -> None:
    def fmt(value: object, digits: int = 3) -> str:
        try:
            if pd.isna(value):
                return "NA"
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return str(value)

    bmi = comparison.loc[comparison["trait"] == "BMI_IRN"].iloc[0]
    t2d = comparison.loc[comparison["trait"] == "T2D"].iloc[0]
    dr_main = comparison.loc[comparison["trait"] == "DM_RETINOPATHY_EXMORE"].iloc[0]
    dr_mhc = comparison.loc[comparison["trait"] == "DM_RETINOPATHY_EXMORE_WITH_MHC"].iloc[0]
    dr_no = comparison.loc[comparison["trait"] == "DM_RETINOPATHY_EXMORE_NO_MHC_NO_APOE"].iloc[0]
    pathway_main = pathway[pathway["trait"] == "DM_RETINOPATHY_EXMORE"].copy()
    best_pathway = pathway_main.sort_values("score_z", ascending=False).head(1)
    if best_pathway.empty:
        best_pathway_text = "No pathway had enough genes for testing."
    else:
        row = best_pathway.iloc[0]
        best_pathway_text = (
            f"Best quick pathway signal was `{row['gene_set']}` "
            f"(score Z={fmt(row['score_z'])}, rank Z={fmt(row['rank_enrichment_z'])}, "
            f"p_score={fmt(row['score_empirical_p_upper'])})."
        )

    lines = [
        "# DR Degree-Matched Signal Diagnosis",
        "",
        "## Executive read",
        "",
        (
            "DR remains weak after degree-aware calibration. The main no-MHC analysis has "
            f"degree-matched Z={fmt(dr_main['degree_matched_z'])}; with MHC increases this to "
            f"{fmt(dr_mhc['degree_matched_z'])}, still below the frozen Z>=2 threshold; "
            f"removing both MHC and APOE gives Z={fmt(dr_no['degree_matched_z'])}."
        ),
        "",
        (
            "The same pipeline is strongly positive for metabolic benchmarks: "
            f"BMI degree-matched Z={fmt(bmi['degree_matched_z'])} and "
            f"T2D degree-matched Z={fmt(t2d['degree_matched_z'])}. This argues against a "
            "generic pipeline failure and points to DR-specific power, phenotype, or graph-layer limitations."
        ),
        "",
        best_pathway_text,
        "",
        "## Architecture comparison",
        "",
        "| Trait | Class | SNP Z | Degree-strat Z | Degree-matched Z | Degree-pres graph Z | Clipped P |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in comparison.iterrows():
        lines.append(
            "| "
            f"{row['trait']} | {row['architecture_class']} | {fmt(row['snp_null_z'])} | "
            f"{fmt(row['degree_stratified_z'])} | {fmt(row['degree_matched_z'])} | "
            f"{fmt(row['degree_preserving_graph_z'])} | {int(row['p_clipped'])} |"
        )

    lines.extend(
        [
            "",
            "## DR sensitivity read",
            "",
            "| DR version | Observed AUC | SNP Z | Degree-matched Z | Degree-pres graph Z | Class |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in dr_sensitivity.iterrows():
        lines.append(
            "| "
            f"{row['trait']} | {fmt(row['observed_auc'], 6)} | {fmt(row['snp_null_z'])} | "
            f"{fmt(row['degree_matched_z'])} | {fmt(row['degree_preserving_graph_z'])} | "
            f"{row['architecture_class']} |"
        )

    lines.extend(
        [
            "",
            "## Pathway quick tests",
            "",
            "These are hypothesis-oriented gene-set checks, not a replacement for a curated pathway benchmark.",
            "",
            "| Trait | Gene set | n | Score Z | Rank Z | p_score | Top present genes |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in pathway.iterrows():
        lines.append(
            "| "
            f"{row['trait']} | {row['gene_set']} | {int(row['n_present'])} | "
            f"{fmt(row['score_z'])} | {fmt(row['rank_enrichment_z'])} | "
            f"{fmt(row['score_empirical_p_upper'])} | {row['top_present_genes']} |"
        )

    lines.extend(
        [
            "",
            "## STRING induced subgraphs",
            "",
            "| Gene set | n | Edges | Components | LCC fraction | Edge Z | p_edge |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in subgraph.iterrows():
        lines.append(
            "| "
            f"{row['gene_set']} | {int(row['n_present_in_main_lcc'])} | "
            f"{int(row['n_induced_edges'])} | {int(row['n_components'])} | "
            f"{fmt(row['largest_component_fraction'])} | {fmt(row['edge_count_z_random'])} | "
            f"{fmt(row['edge_count_empirical_p_upper'])} |"
        )

    lines.extend(
        [
            "",
            "## Working interpretation",
            "",
            "1. DR has only weak global gene-score aggregation in this FinnGen R13 phenotype under the frozen V1 defaults.",
            "2. MHC inclusion creates some broad signal but does not create robust degree-aware graph aggregation.",
            "3. Removing APOE/MHC does not rescue the signal, so the low main result is not explained by APOE/MHC masking.",
            "4. The negative degree-preserving graph Z means the observed STRING topology is not adding module excess beyond degree-preserving rewired graphs; this is consistent with degree/topology sensitivity rather than a strong disease module.",
            "5. If pathway quick tests are also weak, the next highest-value action is to evaluate a larger or more specific DR GWAS. If one pathway is enriched, the next action is a retina/endothelial graph before replacing the GWAS.",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_architecture_comparison_markdown(path: Path, comparison: pd.DataFrame) -> None:
    def fmt(value: object, digits: int = 3) -> str:
        try:
            if pd.isna(value):
                return "NA"
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return str(value)

    lines = [
        "# RIPPLE Trait Architecture Comparison",
        "",
        "| Trait | Class | SNP Z | Degree-strat Z | Degree-matched Z | Degree-pres graph Z |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in comparison.iterrows():
        lines.append(
            "| "
            f"{row['trait']} | {row['architecture_class']} | {fmt(row['snp_null_z'])} | "
            f"{fmt(row['degree_stratified_z'])} | {fmt(row['degree_matched_z'])} | "
            f"{fmt(row['degree_preserving_graph_z'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    tables_dir = args.out_dir / "tables"
    reports_dir = args.out_dir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    summaries = {trait: load_json(path) for trait, path in TRAIT_SUMMARIES.items()}
    if HEIGHT_DEGREE_DIAGNOSTICS.exists():
        height_diag = load_json(HEIGHT_DEGREE_DIAGNOSTICS)
        height_summary = summaries["HEIGHT_IRN"]
        height_summary["snp_permutation_null_summary"] = height_diag["snp_permutation_null"]
        height_summary["degree_stratified_null_summary"] = height_diag["degree_stratified_null"]
        height_summary["degree_matched_node_null_summary"] = height_diag["degree_matched_node_null"]
        height_summary["degree_preserving_graph_null_summary"] = height_diag["degree_preserving_graph_null"]
        height_summary["percolation_architecture"] = height_diag["percolation_architecture"]
    comparison = pd.DataFrame([summary_row(trait, summary) for trait, summary in summaries.items()])
    write_table(ANALYSIS_ROOT / "trait_architecture_comparison.tsv", comparison)
    write_architecture_comparison_markdown(ANALYSIS_ROOT / "trait_architecture_comparison.md", comparison)
    write_table(tables_dir / "trait_architecture_comparison.tsv", comparison)
    dr_sensitivity = comparison[comparison["trait"].str.startswith("DM_RETINOPATHY_EXMORE")].copy()
    write_table(tables_dir / "dr_sensitivity_comparison.tsv", dr_sensitivity)

    scores_by_trait = {
        trait: add_rank_columns(pd.read_csv(path, sep="\t", compression="infer"))
        for trait, path in GENE_SCORE_TABLES.items()
    }
    top_genes = top_gene_rows(scores_by_trait)
    write_table(tables_dir / "dr_top25_lcc_genes_by_variant.tsv", top_genes)

    pathway_rows_all: list[dict[str, object]] = []
    for trait, scores in scores_by_trait.items():
        pathway_rows_all.extend(pathway_enrichment_rows(trait, scores, rng, args.n_random))
    pathway = pd.DataFrame(pathway_rows_all)
    write_table(tables_dir / "dr_pathway_rank_score_enrichment.tsv", pathway)

    main_scores = scores_by_trait["DM_RETINOPATHY_EXMORE"]
    graph_args = argparse.Namespace(
        string_links=args.string_links,
        string_info=args.string_info,
        string_min_score=args.string_min_score,
    )
    _, graph_pre = build_string_graph(graph_args, tuple(main_scores["gene_symbol"].astype(str)))
    graph = graph_pre.largest_component
    subgraph = pd.DataFrame(subgraph_rows(scores_by_trait, graph, rng, args.n_random))
    write_table(tables_dir / "dr_pathway_string_induced_subgraphs.tsv", subgraph)

    build_markdown_report(
        reports_dir / "DR_degree_matched_signal_diagnosis.md",
        comparison,
        dr_sensitivity,
        pathway,
        subgraph,
    )
    print(f"Wrote DR diagnosis outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
