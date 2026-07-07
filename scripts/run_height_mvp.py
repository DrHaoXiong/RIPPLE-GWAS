#!/usr/bin/env python
"""Run the observed-only HEIGHT_IRN end-to-end RIPPLE pilot.

This pilot deliberately uses `ld_mode=identity_pilot_observed_only`: it runs the
complete data plumbing and graph endpoints, but does not yet build per-gene
1000G LD matrices. The next integration step should replace this with cached
per-gene LD.
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ripple.defaults import RANK_FRACTION_GRID  # noqa: E402
from ripple.graph import graph_laplacian, preprocess_reference_graph  # noqa: E402
from ripple.gsp import band_energy_table, laplacian_eigendecomposition, project_graph_signal  # noqa: E402
from ripple.io.annotations import read_magma_gene_loc  # noqa: E402
from ripple.io.graph import read_string_gene_graph  # noqa: E402
from ripple.mapping.weights import add_positional_weights, summarize_mapping  # noqa: E402
from ripple.percolation import percolation_auc, percolation_curve, rank_nodes_by_score  # noqa: E402
from ripple.signals.residualize import append_residualized_score  # noqa: E402
from ripple.signals.safety import append_safety_columns, summarize_clipping  # noqa: E402
from ripple.signals.unsigned import normal_score_from_p_value, quadratic_p_value  # noqa: E402


PRIVATE_ROOT = Path("/path/to/ripple_private_workspace")
DEFAULT_GWAS = PRIVATE_ROOT / "20_processed_data" / "gwas_qc" / "core_hm3_no_mhc" / "HEIGHT_IRN.tsv.gz"
DEFAULT_GENE_LOC = (
    PRIVATE_ROOT
    / "10_raw_data"
    / "reference"
    / "genes"
    / "magma_gene_locations"
    / "NCBI37.3"
    / "NCBI37.3.gene.loc"
)
DEFAULT_STRING_LINKS = (
    PRIVATE_ROOT
    / "10_raw_data"
    / "reference"
    / "graphs"
    / "string_v12"
    / "9606.protein.physical.links.v12.0.txt.gz"
)
DEFAULT_STRING_INFO = (
    PRIVATE_ROOT
    / "10_raw_data"
    / "reference"
    / "graphs"
    / "string_v12"
    / "9606.protein.info.v12.0.txt.gz"
)
DEFAULT_OUT_DIR = PRIVATE_ROOT / "30_analysis" / "height_irn_mvp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gwas", type=Path, default=DEFAULT_GWAS)
    parser.add_argument("--gene-loc", type=Path, default=DEFAULT_GENE_LOC)
    parser.add_argument("--string-links", type=Path, default=DEFAULT_STRING_LINKS)
    parser.add_argument("--string-info", type=Path, default=DEFAULT_STRING_INFO)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--upstream-bp", type=int, default=0)
    parser.add_argument("--downstream-bp", type=int, default=0)
    parser.add_argument("--string-min-score", type=int, default=400)
    parser.add_argument("--eigen-components", type=int, default=128)
    parser.add_argument("--quadratic-method", choices=["liu", "satterthwaite"], default="liu")
    parser.add_argument("--p-clip-epsilon", type=float, default=1e-300)
    parser.add_argument("--rebuild-mapping", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def normalize_chrom(chrom: object) -> str:
    value = str(chrom)
    return value[3:] if value.lower().startswith("chr") else value


def load_height_gwas(path: Path) -> pd.DataFrame:
    """Load the QC'ed HEIGHT GWAS columns needed for gene scoring."""

    columns = ["snp_id", "chrom", "pos", "z", "beta", "se", "p_value", "eaf", "maf"]
    gwas = pd.read_csv(
        path,
        sep="\t",
        compression="infer",
        usecols=columns,
        dtype={"snp_id": str, "chrom": str},
    )
    gwas["chrom"] = gwas["chrom"].map(normalize_chrom)
    gwas["pos"] = pd.to_numeric(gwas["pos"], errors="raise").astype(int)
    gwas["z"] = pd.to_numeric(gwas["z"], errors="raise").astype(float)
    return gwas


def fast_positional_map_by_gene(
    snps: pd.DataFrame,
    genes: pd.DataFrame,
    *,
    upstream_bp: int = 0,
    downstream_bp: int = 0,
) -> pd.DataFrame:
    """Map SNPs to gene intervals by iterating genes and binary-searching SNPs."""

    if upstream_bp < 0 or downstream_bp < 0:
        raise ValueError("upstream_bp and downstream_bp must be nonnegative.")

    snp_by_chrom: dict[str, pd.DataFrame] = {}
    for chrom, group in snps.loc[:, ["snp_id", "chrom", "pos"]].groupby("chrom", observed=True):
        snp_by_chrom[str(chrom)] = group.sort_values("pos").reset_index(drop=True)

    rows: list[pd.DataFrame] = []
    for gene in genes.itertuples(index=False):
        chrom = normalize_chrom(gene.chrom)
        snp_chr = snp_by_chrom.get(chrom)
        if snp_chr is None or snp_chr.empty:
            continue
        gene_start = int(gene.start)
        gene_end = int(gene.end)
        map_start = max(0, gene_start - int(upstream_bp))
        map_end = gene_end + int(downstream_bp)
        positions = snp_chr["pos"].to_numpy(dtype=int)
        left = int(np.searchsorted(positions, map_start, side="left"))
        right = int(np.searchsorted(positions, map_end, side="right"))
        if right <= left:
            continue
        subset = snp_chr.iloc[left:right].copy()
        subset["gene_id"] = str(gene.gene_id)
        subset["gene_symbol"] = str(gene.gene_symbol)
        subset["gene_start"] = gene_start
        subset["gene_end"] = gene_end
        subset["map_start"] = map_start
        subset["map_end"] = map_end
        subset["distance_to_gene"] = 0
        rows.append(
            subset.loc[
                :,
                [
                    "snp_id",
                    "gene_id",
                    "gene_symbol",
                    "chrom",
                    "pos",
                    "gene_start",
                    "gene_end",
                    "map_start",
                    "map_end",
                    "distance_to_gene",
                ],
            ].rename(columns={"pos": "snp_pos"})
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "snp_id",
                "gene_id",
                "gene_symbol",
                "chrom",
                "snp_pos",
                "gene_start",
                "gene_end",
                "map_start",
                "map_end",
                "distance_to_gene",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def write_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False, compression="infer")


def compute_identity_gene_scores(
    gwas: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    method: str,
    p_clip_epsilon: float,
) -> pd.DataFrame:
    """Compute observed-only gene scores using identity LD for the pilot."""

    work = mapping.merge(gwas.loc[:, ["snp_id", "z"]], on="snp_id", how="inner", validate="many_to_one")
    if work.empty:
        raise RuntimeError("No SNP-gene mappings overlap the HEIGHT GWAS.")
    work["weight"] = pd.to_numeric(work["weight"], errors="raise").astype(float)
    work["z"] = pd.to_numeric(work["z"], errors="raise").astype(float)
    work["wz"] = work["weight"] * work["z"]
    work["w2"] = np.square(work["weight"])
    work["w2z2"] = work["w2"] * np.square(work["z"])

    aggregate = (
        work.groupby(["gene_id", "gene_symbol", "chrom", "gene_start", "gene_end"], observed=True)
        .agg(
            n_mapped_snps=("snp_id", "nunique"),
            signed_numerator=("wz", "sum"),
            denominator_variance=("w2", "sum"),
            quadratic_statistic=("w2z2", "sum"),
        )
        .reset_index()
    )
    aggregate["denominator"] = np.sqrt(aggregate["denominator_variance"])
    aggregate["x_g_dir"] = aggregate["signed_numerator"] / aggregate["denominator"]
    aggregate["m_eff"] = aggregate["n_mapped_snps"].astype(float)

    p_values: list[float] = []
    p_clipped: list[float] = []
    normal_scores: list[float] = []
    clipped_flags: list[bool] = []
    for gene_id, group in work.groupby("gene_id", observed=True, sort=False):
        stat = float(group["w2z2"].sum())
        lambdas = group["w2"].to_numpy(dtype=float)
        p_value = quadratic_p_value(stat, lambdas, method=method)
        normal_score, clipped_p, was_clipped = normal_score_from_p_value(
            p_value,
            epsilon=p_clip_epsilon,
        )
        p_values.append(p_value)
        p_clipped.append(clipped_p)
        normal_scores.append(normal_score)
        clipped_flags.append(was_clipped)

    p_frame = pd.DataFrame(
        {
            "gene_id": [str(gene_id) for gene_id, _ in work.groupby("gene_id", observed=True, sort=False)],
            "assoc_p_g": p_values,
            "assoc_p_g_clipped": p_clipped,
            "assoc_normal_score_g": normal_scores,
            "is_p_clipped": clipped_flags,
        }
    )
    out = aggregate.merge(p_frame, on="gene_id", how="left", validate="one_to_one")
    out["assoc_minuslog10p_g"] = -np.log10(out["assoc_p_g_clipped"])
    out["gene_length"] = out["gene_end"] - out["gene_start"] + 1
    out["log_gene_length"] = np.log1p(out["gene_length"])
    out["log_mapped_snp_count"] = np.log1p(out["n_mapped_snps"])
    out["log_m_eff"] = np.log1p(out["m_eff"])
    out["local_ld_score"] = 1.0
    out["mappability"] = 1.0
    out["ld_mode"] = "identity_pilot_observed_only"
    return append_safety_columns(
        out.rename(columns={"gene_start": "start", "gene_end": "end"}),
        chrom_col="chrom",
        start_col="start",
        end_col="end",
    ).rename(columns={"start": "gene_start", "end": "gene_end"})


def build_string_graph(args: argparse.Namespace, gene_symbols: tuple[str, ...]):
    graph_edges = read_string_gene_graph(
        args.string_links,
        args.string_info,
        min_score=args.string_min_score,
    )
    return graph_edges, preprocess_reference_graph(graph_edges.edges, gene_universe=gene_symbols)


def write_markdown_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# HEIGHT_IRN RIPPLE V1 MVP",
        "",
        "This is an observed-only end-to-end pilot run.",
        "",
        "Important caveat: LD mode is `identity_pilot_observed_only`; per-gene 1000G LD cache and null replicates are not yet included.",
        "",
        "## Key Counts",
        "",
        f"- SNPs loaded: {summary['n_gwas_snps']:,}",
        f"- SNP-gene mapping rows: {summary['n_mapping_rows']:,}",
        f"- Gene scores: {summary['n_gene_scores']:,}",
        f"- STRING LCC genes with scores: {summary['n_lcc_scored_genes']:,}",
        f"- Percolation observed AUC: {summary['percolation_auc_observed']:.6f}",
        f"- GSP retained energy fraction: {summary['gsp_retained_energy_fraction']:.6f}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_outputs(args: argparse.Namespace) -> None:
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.force:
        raise FileExistsError(f"{args.out_dir} exists and is not empty. Use --force to overwrite.")
    args.out_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    ensure_outputs(args)
    tables_dir = args.out_dir / "tables"
    reports_dir = args.out_dir / "reports"

    print("Loading HEIGHT GWAS", flush=True)
    gwas = load_height_gwas(args.gwas)

    print("Loading MAGMA gene locations", flush=True)
    genes = read_magma_gene_loc(args.gene_loc).table
    genes = genes[genes["chrom"].astype(str).isin({str(i) for i in range(1, 23)})].copy()

    mapping_path = tables_dir / "HEIGHT_IRN.gene_body_mapping.tsv.gz"
    if mapping_path.exists() and not args.rebuild_mapping:
        mapping = pd.read_csv(mapping_path, sep="\t", compression="infer", dtype={"gene_id": str, "gene_symbol": str})
    else:
        print("Building positional SNP-to-gene mapping", flush=True)
        mapping = fast_positional_map_by_gene(
            gwas,
            genes,
            upstream_bp=args.upstream_bp,
            downstream_bp=args.downstream_bp,
        )
        mapping = add_positional_weights(mapping)
        write_table(mapping_path, mapping)

    mapping_summary = summarize_mapping(mapping)
    print(f"Mapped {mapping_summary.n_snps:,} SNPs to {mapping_summary.n_genes:,} genes", flush=True)

    gene_scores_path = tables_dir / "HEIGHT_IRN.gene_scores.identity_pilot.tsv.gz"
    print("Computing identity-LD gene scores", flush=True)
    gene_scores = compute_identity_gene_scores(
        gwas,
        mapping,
        method=args.quadratic_method,
        p_clip_epsilon=args.p_clip_epsilon,
    )
    write_table(gene_scores_path, gene_scores)

    print("Building STRING graph and largest connected component", flush=True)
    gene_symbols = tuple(gene_scores["gene_symbol"].dropna().astype(str).unique())
    graph_edges, graph_pre = build_string_graph(args, gene_symbols)
    write_table(tables_dir / "STRING.physical.min400.edges.tsv.gz", graph_edges.edges)
    write_table(tables_dir / "STRING.component_table.tsv", graph_pre.component_table)

    lcc_nodes = tuple(sorted(str(node) for node in graph_pre.largest_component.nodes()))
    lcc_scores = gene_scores[gene_scores["gene_symbol"].astype(str).isin(lcc_nodes)].copy()
    degree = dict(graph_pre.largest_component.degree())
    lcc_scores["graph_degree"] = lcc_scores["gene_symbol"].map(degree).astype(int)
    lcc_scores = lcc_scores.sort_values("gene_symbol").reset_index(drop=True)

    print("Residualizing association scores on technical covariates", flush=True)
    lcc_scores, residualization = append_residualized_score(lcc_scores)
    write_table(tables_dir / "HEIGHT_IRN.lcc_gene_scores.residualized.tsv.gz", lcc_scores)

    print("Constructing normalized Laplacian and GSP summary", flush=True)
    nodes = tuple(lcc_scores["gene_symbol"].astype(str))
    lap = graph_laplacian(graph_pre.largest_component, nodes=nodes, kind="normalized")
    decomp = laplacian_eigendecomposition(
        lap.laplacian,
        nodes=nodes,
        n_components=min(args.eigen_components, max(1, len(nodes) - 2)),
    )
    signal = project_graph_signal(lcc_scores["assoc_resid_score"].to_numpy(dtype=float), decomp, laplacian=lap.laplacian)
    eigen_table = pd.DataFrame({"eigenvalue": signal.eigenvalues, "coefficient": signal.coefficients, "energy": signal.energy})
    band_table = band_energy_table(signal.eigenvalues, signal.coefficients)
    write_table(tables_dir / "HEIGHT_IRN.gsp_eigen.tsv", eigen_table)
    write_table(tables_dir / "HEIGHT_IRN.gsp_band_energy.tsv", band_table)

    print("Computing observed-only percolation curve", flush=True)
    ranking = rank_nodes_by_score(lcc_scores, node_col="gene_symbol", score_col="assoc_resid_score")
    curve = percolation_curve(graph_pre.largest_component, ranking, RANK_FRACTION_GRID, node_col="gene_symbol")
    auc = percolation_auc(curve)
    write_table(tables_dir / "HEIGHT_IRN.percolation_curve.observed.tsv", curve)
    write_table(tables_dir / "HEIGHT_IRN.ranking.tsv.gz", ranking)

    clipping = summarize_clipping(gene_scores["is_p_clipped"])
    summary = {
        "created_utc": datetime.now(UTC).isoformat(),
        "trait": "HEIGHT_IRN",
        "mode": "observed_only_mvp",
        "ld_mode": "identity_pilot_observed_only",
        "gwas_path": str(args.gwas),
        "gene_loc_path": str(args.gene_loc),
        "string_links_path": str(args.string_links),
        "string_min_score": args.string_min_score,
        "upstream_bp": args.upstream_bp,
        "downstream_bp": args.downstream_bp,
        "quadratic_method": args.quadratic_method,
        "n_gwas_snps": int(len(gwas)),
        "n_mapping_rows": int(len(mapping)),
        "mapping_summary": asdict(mapping_summary),
        "n_gene_scores": int(len(gene_scores)),
        "n_lcc_scored_genes": int(len(lcc_scores)),
        "graph_coverage_report": asdict(graph_pre.coverage_report),
        "residualization_method": residualization.method,
        "residualization_covariates": residualization.covariate_names,
        "gsp_method": decomp.method,
        "gsp_smoothness": signal.smoothness,
        "gsp_retained_energy_fraction": signal.retained_energy_fraction,
        "percolation_auc_observed": auc,
        "percolation_delta": None,
        "p_clipping_summary": asdict(clipping),
        "outputs": {
            "mapping": str(mapping_path),
            "gene_scores": str(gene_scores_path),
            "lcc_gene_scores": str(tables_dir / "HEIGHT_IRN.lcc_gene_scores.residualized.tsv.gz"),
            "gsp_band_energy": str(tables_dir / "HEIGHT_IRN.gsp_band_energy.tsv"),
            "percolation_curve": str(tables_dir / "HEIGHT_IRN.percolation_curve.observed.tsv"),
        },
        "remaining_required_work": [
            "replace identity LD with cached per-gene 1000G EUR LD matrices",
            "add null replicates and Delta_perc calibration",
            "add degree-stratified and degree-preserving null diagnostics",
        ],
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "HEIGHT_IRN.mvp_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown_report(reports_dir / "HEIGHT_IRN.mvp_report.md", summary)
    print(f"Wrote MVP outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
