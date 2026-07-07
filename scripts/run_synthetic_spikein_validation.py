#!/usr/bin/env python
"""Run RIPPLE synthetic spike-in validation scenarios."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ripple.validation import run_synthetic_spikein_validation  # noqa: E402

PRIVATE_ROOT = Path("/path/to/ripple_private_workspace")
DEFAULT_OUT_DIR = PRIVATE_ROOT / "30_analysis" / "synthetic_spikein_validation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--n-score-null", type=int, default=200)
    parser.add_argument("--n-degree-stratified-null", type=int, default=200)
    parser.add_argument("--n-degree-matched-node-null", type=int, default=500)
    parser.add_argument("--n-degree-graph-null", type=int, default=100)
    parser.add_argument("--n-module-selection-aware-null", type=int, default=200)
    parser.add_argument("--degree-bins", type=int, default=10)
    return parser.parse_args()


def write_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False)


def render_report(summary: pd.DataFrame) -> str:
    lines = [
        "# RIPPLE Synthetic Spike-In Validation",
        "",
        "## Scenario Summary",
        "",
        "| Scenario | Architecture | Suitability | SNP Z | Degree-strat Z | Degree-matched Z | Graph Z | Calibrated modules |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| "
            f"{row['scenario']} | {row['architecture_class']} | {row['suitability_verdict']} | "
            f"{float(row['snp_null_z']):.3f} | {float(row['degree_stratified_z']):.3f} | "
            f"{float(row['degree_matched_z']):.3f} | {float(row['degree_preserving_graph_z']):.3f} | "
            f"{int(row['n_calibrated_modules'])} |"
        )
    lines.extend(
        [
            "",
            "## Expected Behavior",
            "",
            "- Null should not show stable positive calibrated signal.",
            "- Dispersed signal can exceed score nulls but should not reliably create graph aggregation.",
            "- Degree-biased signal should be reduced by degree-aware calibration.",
            "- Module spike-in should produce the strongest degree-matched and topology-specific evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    tables_dir = args.out_dir / "tables"
    reports_dir = args.out_dir / "reports"
    summary, detail_tables = run_synthetic_spikein_validation(
        seed=args.seed,
        n_score_null=args.n_score_null,
        n_degree_stratified_null=args.n_degree_stratified_null,
        n_degree_matched_node_null=args.n_degree_matched_node_null,
        n_degree_graph_null=args.n_degree_graph_null,
        n_module_selection_aware_null=args.n_module_selection_aware_null,
        degree_bins=args.degree_bins,
    )
    write_table(tables_dir / "synthetic_spikein_summary.tsv", summary)
    for name, table in detail_tables.items():
        write_table(tables_dir / f"{name}.tsv", table)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "synthetic_spikein_validation_report.md").write_text(render_report(summary), encoding="utf-8")
    print(f"Wrote synthetic spike-in validation outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
