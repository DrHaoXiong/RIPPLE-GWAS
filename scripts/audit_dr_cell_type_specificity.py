#!/usr/bin/env python
"""Audit the DR cell-type specificity enrichment layer.

This script is intentionally diagnostic. It tests whether apparent scRNA
cell-type enrichment is driven by rare cell-type marker sets, low-information
genes, special regions, or a small number of extreme GWAS gene scores.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from run_dr_cell_type_specificity_enrichment import (  # noqa: E402
    ANALYSIS_ROOT,
    DEFAULT_MARKERS,
    TRAITS,
    add_degree_bins,
    degree_matched_mean_null,
    empirical_p,
    stable_offset,
)

PRIVATE_ROOT = Path("/path/to/ripple_private_workspace")
DEFAULT_ENRICHMENT = PRIVATE_ROOT / "30_analysis" / "dr_cell_type_specificity_v0_1"
DEFAULT_MARKER_METRICS = (
    PRIVATE_ROOT
    / "20_processed_data"
    / "reference_expression"
    / "dr_cell_type_specificity_v0_1"
    / "tables"
    / "dr_scrna_cell_type_markers.tsv"
)
DEFAULT_OUT_DIR = PRIVATE_ROOT / "30_analysis" / "dr_cell_type_specificity_v0_1_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enrichment-dir", type=Path, default=DEFAULT_ENRICHMENT)
    parser.add_argument("--marker-gene-sets", type=Path, default=DEFAULT_MARKERS)
    parser.add_argument("--marker-metrics", type=Path, default=DEFAULT_MARKER_METRICS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-null", type=int, default=5000)
    parser.add_argument("--degree-bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260730)
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def write_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False)


def load_scores_full(key: str) -> pd.DataFrame:
    config = TRAITS[key]
    path = ANALYSIS_ROOT / config["analysis_dir"] / "tables" / config["score_file"]
    scores = pd.read_csv(path, sep="\t")
    scores["gene_symbol"] = scores["gene_symbol"].astype(str).str.upper()
    scores["assoc_resid_score"] = pd.to_numeric(scores["assoc_resid_score"], errors="coerce")
    scores["graph_degree"] = pd.to_numeric(scores["graph_degree"], errors="coerce").fillna(0.0)
    for column in ["is_special_region", "is_low_information"]:
        if column not in scores:
            scores[column] = False
        scores[column] = scores[column].fillna(False).astype(bool)
    return scores.dropna(subset=["assoc_resid_score"]).drop_duplicates("gene_symbol")


def choose_audit_targets(enrichment: pd.DataFrame) -> pd.DataFrame:
    supported = enrichment.loc[
        enrichment["cell_context_status"].isin(["context_supported", "nominal_suggestive"])
    ].copy()
    best_by_trait = (
        enrichment.sort_values(["trait", "degree_matched_empirical_p", "degree_matched_z"], ascending=[True, True, False])
        .groupby("trait", observed=True)
        .head(1)
    )
    targets = pd.concat([supported, best_by_trait], ignore_index=True).drop_duplicates(["trait", "gene_set"])
    keep_traits = {"DR_MVP", "DR_MVP_NO_MHC_NO_APOE", "T2D", "SCZ", "HEIGHT_IRN"}
    return targets.loc[targets["trait"].isin(keep_traits)].reset_index(drop=True)


def sensitivity_rows(
    trait_key: str,
    gene_set: str,
    markers: pd.DataFrame,
    marker_metrics: pd.DataFrame,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    scores_full = load_scores_full(trait_key)
    genes = set(markers.loc[markers["gene_set"].eq(gene_set), "gene_symbol"].astype(str).str.upper())
    marker_meta = markers.loc[markers["gene_set"].eq(gene_set)].iloc[0].to_dict()
    metrics = marker_metrics.loc[marker_metrics["marker_set"].eq(gene_set)]
    n_target_cells = int(metrics["n_target_cells"].max()) if "n_target_cells" in metrics and not metrics.empty else np.nan
    present_full = scores_full.loc[scores_full["gene_symbol"].isin(genes)].copy()
    rows = []
    scenarios = [
        ("baseline", scores_full, present_full),
        (
            "exclude_special_region",
            scores_full.loc[~scores_full["is_special_region"]].copy(),
            present_full.loc[~present_full["is_special_region"]].copy(),
        ),
        (
            "exclude_low_information",
            scores_full.loc[~scores_full["is_low_information"]].copy(),
            present_full.loc[~present_full["is_low_information"]].copy(),
        ),
        (
            "exclude_special_and_low_information",
            scores_full.loc[~scores_full["is_special_region"] & ~scores_full["is_low_information"]].copy(),
            present_full.loc[~present_full["is_special_region"] & ~present_full["is_low_information"]].copy(),
        ),
    ]
    for scenario, universe, selected in scenarios:
        rows.append(compute_row(trait_key, gene_set, marker_meta, universe, selected, scenario, args, n_target_cells))
    for k in [1, 5, 10]:
        selected = present_full.sort_values("assoc_resid_score", ascending=False).iloc[k:].copy()
        rows.append(compute_row(trait_key, gene_set, marker_meta, scores_full, selected, f"leave_top_{k}_selected_genes", args, n_target_cells))
    selected = present_full.copy()
    cap = float(scores_full["assoc_resid_score"].quantile(0.99))
    selected["assoc_resid_score"] = np.minimum(selected["assoc_resid_score"], cap)
    universe = scores_full.copy()
    universe["assoc_resid_score"] = np.minimum(universe["assoc_resid_score"], cap)
    rows.append(compute_row(trait_key, gene_set, marker_meta, universe, selected, "winsorize_global_p99", args, n_target_cells))
    return rows


def compute_row(
    trait_key: str,
    gene_set: str,
    marker_meta: dict[str, object],
    universe: pd.DataFrame,
    selected: pd.DataFrame,
    scenario: str,
    args: argparse.Namespace,
    n_target_cells: int | float,
) -> dict[str, object]:
    universe = add_degree_bins(universe.loc[:, ["gene_symbol", "assoc_resid_score", "graph_degree"]].copy(), args.degree_bins)
    selected = selected.loc[:, ["gene_symbol", "assoc_resid_score", "graph_degree"]].copy()
    selected = selected.merge(universe[["gene_symbol", "degree_bin"]], on="gene_symbol", how="inner")
    if selected.empty:
        observed = null_mean = null_sd = z = p_value = np.nan
    else:
        rng = np.random.default_rng(args.seed + stable_offset(f"{trait_key}:{gene_set}:{scenario}"))
        observed = float(selected["assoc_resid_score"].mean())
        null_values = degree_matched_mean_null(universe, selected, n_null=args.n_null, rng=rng)
        null_mean = float(np.nanmean(null_values))
        null_sd = float(np.nanstd(null_values, ddof=1))
        z = (observed - null_mean) / null_sd if null_sd > 0 else np.nan
        p_value = empirical_p(observed, null_values)
    top = selected.sort_values("assoc_resid_score", ascending=False).head(10)
    positive_scores = selected.loc[selected["assoc_resid_score"] > 0, "assoc_resid_score"]
    positive_sum = float(positive_scores.sum()) if not selected.empty else np.nan
    top5_positive_sum = float(top.head(5).loc[top.head(5)["assoc_resid_score"] > 0, "assoc_resid_score"].sum()) if not selected.empty else np.nan
    return {
        "trait": TRAITS[trait_key]["trait"],
        "analysis_id": trait_key,
        "gene_set": gene_set,
        "dataset_id": marker_meta.get("dataset_id", ""),
        "tissue_context": marker_meta.get("tissue_context", ""),
        "condition_scope": marker_meta.get("condition_scope", ""),
        "cell_type": marker_meta.get("cell_type", ""),
        "scenario": scenario,
        "n_target_cells_for_marker_set": n_target_cells,
        "n_present_genes": int(selected.shape[0]),
        "observed_mean_score": observed,
        "null_mean": null_mean,
        "null_sd": null_sd,
        "degree_matched_z": z,
        "degree_matched_empirical_p": p_value,
        "top_gene_symbols": ",".join(top["gene_symbol"].astype(str).head(10)),
        "top_gene_scores": ",".join(f"{value:.4g}" for value in top["assoc_resid_score"].head(10)),
        "top5_positive_score_sum": top5_positive_sum,
        "positive_score_sum": positive_sum,
        "top5_positive_score_contribution": (
            top5_positive_sum / positive_sum if positive_sum and np.isfinite(positive_sum) and positive_sum > 0 else np.nan
        ),
        "n_null": int(args.n_null),
        "statistic_direction": "greater_is_more_extreme",
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["tables", "reports"]:
        (args.out_dir / subdir).mkdir(parents=True, exist_ok=True)
    enrichment = pd.read_csv(args.enrichment_dir / "tables" / "dr_cell_type_marker_enrichment.tsv", sep="\t")
    markers = pd.read_csv(args.marker_gene_sets, sep="\t")
    marker_metrics = pd.read_csv(args.marker_metrics, sep="\t")
    markers["gene_symbol"] = markers["gene_symbol"].astype(str).str.upper()
    targets = choose_audit_targets(enrichment)
    rows: list[dict[str, object]] = []
    for target in targets.to_dict(orient="records"):
        trait_key = str(target["analysis_id"])
        gene_set = str(target["gene_set"])
        if trait_key not in TRAITS:
            continue
        rows.extend(sensitivity_rows(trait_key, gene_set, markers, marker_metrics, args))
    audit = pd.DataFrame(rows)
    write_table(args.out_dir / "tables" / "dr_cell_type_specificity_sensitivity_audit.tsv", audit)
    target_table = targets.loc[
        :,
        [
            "trait",
            "analysis_id",
            "gene_set",
            "dataset_id",
            "tissue_context",
            "condition_scope",
            "cell_type",
            "degree_matched_z",
            "degree_matched_empirical_p",
            "degree_matched_fdr_bh",
            "cell_context_status",
        ],
    ].copy()
    write_table(args.out_dir / "tables" / "dr_cell_type_specificity_audit_targets.tsv", target_table)
    manifest = {
        "created_utc": now_utc(),
        "script_path": str(Path(__file__).resolve()),
        "enrichment_dir": str(args.enrichment_dir),
        "marker_gene_sets": str(args.marker_gene_sets),
        "marker_metrics": str(args.marker_metrics),
        "out_dir": str(args.out_dir),
        "n_null": int(args.n_null),
        "audit_note": "Diagnostic only; identifies sensitivity to rare cell types, low-information genes, and extreme gene-score outliers.",
    }
    (args.out_dir / "reports" / "dr_cell_type_specificity_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# DR cell-type specificity audit",
        "",
        f"Created: {manifest['created_utc']}",
        "",
        "This audit tests whether cell-type enrichment conclusions are sensitive to rare cell-type marker sets,",
        "special regions, low-information genes, and a small number of extreme GWAS gene scores.",
        "",
        "| Trait | Gene set | Scenario | n genes | Z | empirical P | top5 positive contribution |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in audit.to_dict(orient="records"):
        lines.append(
            f"| {row['trait']} | {row['gene_set']} | {row['scenario']} | {int(row['n_present_genes'])} | "
            f"{fmt(row['degree_matched_z'])} | {fmt(row['degree_matched_empirical_p'])} | "
            f"{fmt(row['top5_positive_score_contribution'])} |"
        )
    (args.out_dir / "reports" / "dr_cell_type_specificity_audit_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote DR cell-type specificity audit to {args.out_dir}", flush=True)


def fmt(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return f"{float(number):.4g}" if np.isfinite(number) else ""


if __name__ == "__main__":
    main()
