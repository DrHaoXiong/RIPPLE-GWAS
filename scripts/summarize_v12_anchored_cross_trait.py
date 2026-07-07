#!/usr/bin/env python
"""Summarize RIPPLE V1.2 broad anchored module runs across traits."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PRIVATE_ROOT = Path("/path/to/ripple_private_workspace")
DEFAULT_ROOT = PRIVATE_ROOT / "30_analysis" / "tier4_v12_anchored_broad_reactome_go_cross_trait_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--top-n", type=int, default=25)
    return parser.parse_args()


def load_trait_result(result_dir: Path) -> tuple[dict[str, object], pd.DataFrame]:
    reports = result_dir / "reports"
    tables = result_dir / "tables"
    summary_paths = sorted(reports.glob("*.v12_anchored_module_summary.json"))
    table_paths = sorted(tables.glob("*.v12_anchored_module_tests.tsv"))
    if len(summary_paths) != 1 or len(table_paths) != 1:
        raise FileNotFoundError(f"Expected one summary and one module table in {result_dir}")
    summary = json.loads(summary_paths[0].read_text(encoding="utf-8"))
    modules = pd.read_csv(table_paths[0], sep="\t")
    modules["trait"] = str(summary.get("trait", result_dir.name))
    modules["analysis_id"] = result_dir.name
    modules["source_result_path"] = str(table_paths[0])
    return summary, modules


def compact_status_counts(modules: pd.DataFrame) -> dict[str, int]:
    tested = modules.loc[modules["module_status"].ne("not_tested_low_overlap")]
    return {
        "n_tested_rows": int(len(tested)),
        "n_fixed_degree_supported_rows": int(tested["module_status"].eq("fixed_degree_supported").sum()),
        "n_familywise_supported_rows": int(
            tested["module_status"].eq("anchored_familywise_supported").sum()
        ),
    }


def summarize_root(root_dir: Path, *, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    summaries: list[dict[str, object]] = []
    top_rows: list[pd.DataFrame] = []
    supported_rows: list[pd.DataFrame] = []
    result_dirs = [
        path
        for path in root_dir.iterdir()
        if path.is_dir()
        and (path / "reports").is_dir()
        and (path / "tables").is_dir()
        and list((path / "reports").glob("*.v12_anchored_module_summary.json"))
        and list((path / "tables").glob("*.v12_anchored_module_tests.tsv"))
    ]
    for result_dir in sorted(result_dirs):
        summary, modules = load_trait_result(result_dir)
        counts = compact_status_counts(modules)
        tested = modules.loc[modules["module_status"].ne("not_tested_low_overlap")].copy()
        tested = tested.sort_values(
            ["library_familywise_p", "degree_matched_empirical_p", "observed_value"],
            ascending=[True, True, False],
        )
        top = tested.head(top_n).copy()
        top["top_rank_within_trait"] = range(1, len(top) + 1)
        top_rows.append(top)
        supported = tested.loc[
            tested["module_status"].isin(["fixed_degree_supported", "anchored_familywise_supported"])
        ].copy()
        supported_rows.append(supported)
        best = tested.iloc[0].to_dict() if not tested.empty else {}
        summaries.append(
            {
                "trait": summary.get("trait", result_dir.name),
                "analysis_id": result_dir.name,
                "n_modules_total": summary.get("n_modules_total"),
                "n_tested_modules": summary.get("n_tested_modules"),
                "n_not_tested_low_overlap": summary.get("n_not_tested_low_overlap"),
                "n_external_gene_sets": summary.get("n_external_gene_sets"),
                "n_degree_matched_null": summary.get("n_degree_matched_null"),
                "n_score_permutation_null": summary.get("n_score_permutation_null"),
                **counts,
                "best_module_name": best.get("module_name", ""),
                "best_module_category": best.get("module_category", ""),
                "best_module_source": best.get("module_source", ""),
                "best_degree_matched_z": best.get("degree_matched_z", ""),
                "best_degree_matched_p": best.get("degree_matched_empirical_p", ""),
                "best_library_familywise_p": best.get("library_familywise_p", ""),
                "best_module_status": best.get("module_status", ""),
                "source_result_path": str(result_dir),
            }
        )

    summary_table = pd.DataFrame(summaries).sort_values("trait").reset_index(drop=True)
    top_table = pd.concat(top_rows, ignore_index=True) if top_rows else pd.DataFrame()
    supported_table = pd.concat(supported_rows, ignore_index=True) if supported_rows else pd.DataFrame()
    report = render_report(summary_table, top_table, supported_table)
    return summary_table, top_table, report


def render_report(summary: pd.DataFrame, top: pd.DataFrame, supported: pd.DataFrame) -> str:
    null_pairs = (
        summary[["n_degree_matched_null", "n_score_permutation_null"]]
        .dropna()
        .drop_duplicates()
        .to_dict(orient="records")
    )
    if len(null_pairs) == 1:
        null_text = (
            f"This diagnostic used {int(null_pairs[0]['n_degree_matched_null'])} degree-matched nulls "
            f"and {int(null_pairs[0]['n_score_permutation_null'])} score-permutation nulls."
        )
    else:
        null_text = (
            "This diagnostic used trait-specific null counts; see the summary table for "
            "degree-matched and score-permutation null counts."
        )
    lines = [
        "# RIPPLE V1.2 Broad Anchored Module Cross-Trait Report",
        "",
        f"Created: {datetime.now(UTC).isoformat()}",
        "",
        null_text,
        "Familywise P values should be interpreted as diagnostic module-prioritization evidence "
        "unless the corresponding claim policy and Type I calibration have been frozen.",
        "",
        "## Trait Summary",
        "",
        "| Trait | Tested modules | Fixed degree-supported | Familywise-supported | Best module | Best family P |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"| {row['trait']} | {int(row['n_tested_modules'])} | "
            f"{int(row['n_fixed_degree_supported_rows'])} | {int(row['n_familywise_supported_rows'])} | "
            f"{row['best_module_name']} | {float(row['best_library_familywise_p']):.4g} |"
        )
    lines.extend(
        [
            "",
            "## Top Module Themes",
            "",
            "| Trait | Rank | Module | Category | Source | degree Z | degree P | family P | Status |",
            "|---|---:|---|---|---|---:|---:|---:|---|",
        ]
    )
    display = top.loc[top["top_rank_within_trait"].le(5)].copy() if not top.empty else pd.DataFrame()
    for row in display.to_dict(orient="records"):
        lines.append(
            f"| {row['trait']} | {int(row['top_rank_within_trait'])} | {row['module_name']} | "
            f"{row.get('module_category', '')} | {row.get('module_source', '')} | "
            f"{float(row['degree_matched_z']):.3f} | {float(row['degree_matched_empirical_p']):.4g} | "
            f"{float(row['library_familywise_p']):.4g} | {row['module_status']} |"
        )
    lines.extend(
        [
            "",
            "## Supported Module Counts By Category",
            "",
        ]
    )
    if supported.empty:
        lines.append("No fixed degree-supported modules were observed in the smoke runs.")
    else:
        category_counts = (
            supported.groupby(["trait", "module_category", "module_source"], observed=True)
            .size()
            .reset_index(name="n_supported")
            .sort_values(["trait", "n_supported"], ascending=[True, False])
        )
        lines.extend(
            [
                "| Trait | Category | Source | Supported modules |",
                "|---|---|---|---:|",
            ]
        )
        for row in category_counts.to_dict(orient="records"):
            lines.append(
                f"| {row['trait']} | {row['module_category']} | {row['module_source']} | "
                f"{int(row['n_supported'])} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Broad Reactome/GO anchored testing is trait-agnostic and suitable for cross-trait diagnostics.",
            "- Familywise positives should be treated as prioritization signals unless separately claim-calibrated.",
            "- If top themes differ by trait, the anchored layer has plausible general-purpose behavior.",
            "- If the same broad categories dominate all traits, further degree/size/statistic calibration is needed.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    tables_dir = args.root_dir / "tables"
    reports_dir = args.root_dir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary, top, report = summarize_root(args.root_dir, top_n=args.top_n)
    summary.to_csv(tables_dir / "cross_trait_anchored_summary.tsv", sep="\t", index=False)
    top.to_csv(tables_dir / "cross_trait_top_modules.tsv", sep="\t", index=False)
    (reports_dir / "cross_trait_anchored_report.md").write_text(report + "\n", encoding="utf-8")
    if "smoke" in args.root_dir.name:
        (reports_dir / "cross_trait_anchored_smoke_report.md").write_text(report + "\n", encoding="utf-8")
    print(f"Wrote cross-trait summary to {args.root_dir}", flush=True)


if __name__ == "__main__":
    main()
