#!/usr/bin/env python
"""Run frozen Tier 4 full reselection nulls for cross-trait local modules.

This script runs after the RIPPLE V1 module-layer claim policy freeze. It does
not rerun GWAS/LD/global analysis and does not alter the DR_MVP frozen
module-reselection outputs. It reuses the validated full-reselection engine from
`run_dr_mvp_module_reselection_null.py` and applies the same rank grid, module
filters, max-module statistic and empirical P formula to eligible cross-trait
analyses.
"""

from __future__ import annotations

import argparse
import json
import sys
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

from run_dr_mvp_module_reselection_null import (  # noqa: E402
    AnalysisSpec,
    run_one,
    write_table,
)


PRIVATE_ROOT = Path("/path/to/ripple_private_workspace")
ANALYSIS_ROOT = PRIVATE_ROOT / "30_analysis"
DEFAULT_OUT_DIR = ANALYSIS_ROOT / "module_reselection_null_cross_trait_v1"
THIS_SCRIPT = Path(__file__).resolve()


CROSS_TRAIT_SPECS = {
    "SCZ_no_MHC_final5000": AnalysisSpec(
        trait="SCZ",
        analysis_id="SCZ_no_MHC_final5000",
        analysis_dir=ANALYSIS_ROOT / "scz_no_mhc_string_final5000",
        graph_edges_path=ANALYSIS_ROOT
        / "scz_no_mhc_string_final5000"
        / "tables"
        / "SCZ.analysis_graph_edges.tsv.gz",
    ),
    "HEIGHT_IRN_analysis_ready": AnalysisSpec(
        trait="HEIGHT_IRN",
        analysis_id="HEIGHT_IRN_analysis_ready",
        analysis_dir=ANALYSIS_ROOT / "height_irn_analysis_ready",
        graph_edges_path=ANALYSIS_ROOT
        / "height_irn_analysis_ready"
        / "tables"
        / "HEIGHT_IRN.analysis_graph_edges.tsv.gz",
    ),
    "BMI_IRN_analysis_ready": AnalysisSpec(
        trait="BMI_IRN",
        analysis_id="BMI_IRN_analysis_ready",
        analysis_dir=ANALYSIS_ROOT / "bmi_irn_analysis_ready",
        graph_edges_path=ANALYSIS_ROOT / "bmi_irn_analysis_ready" / "tables" / "BMI_IRN.analysis_graph_edges.tsv.gz",
    ),
    "T2D_analysis_ready": AnalysisSpec(
        trait="T2D",
        analysis_id="T2D_analysis_ready",
        analysis_dir=ANALYSIS_ROOT / "t2d_analysis_ready",
        graph_edges_path=ANALYSIS_ROOT / "t2d_analysis_ready" / "tables" / "T2D.analysis_graph_edges.tsv.gz",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--analysis-id", action="append", default=[])
    parser.add_argument("--all-cross-traits", action="store_true")
    parser.add_argument("--n-reselection-null", type=int, default=5000)
    parser.add_argument("--engine", choices=["fast", "reference"], default="fast")
    parser.add_argument("--degree-bins", type=int, default=10)
    parser.add_argument("--min-module-size", type=int, default=5)
    parser.add_argument("--min-module-subthreshold-genes", type=int, default=3)
    parser.add_argument("--max-local-modules", type=int, default=20)
    parser.add_argument("--broad-component-min-size", type=int, default=200)
    parser.add_argument("--broad-component-fraction", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def select_specs(args: argparse.Namespace) -> list[AnalysisSpec]:
    if args.all_cross_traits or not args.analysis_id:
        return list(CROSS_TRAIT_SPECS.values())
    requested = set(args.analysis_id)
    unknown = sorted(requested - set(CROSS_TRAIT_SPECS))
    if unknown:
        raise ValueError(f"Unknown analysis_id values: {unknown}")
    return [CROSS_TRAIT_SPECS[name] for name in CROSS_TRAIT_SPECS if name in requested]


def add_frozen_module_policy_status(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    labels = out.get("recommended_module_claim_after_reselection", pd.Series(dtype=str)).astype(str)
    out["module_layer_claim_status"] = np.where(
        labels.eq("calibrated_candidate_module"),
        "selection_calibrated_module",
        np.where(
            labels.eq("post_hoc_candidate_module"),
            "post_hoc_candidate_only",
            "no_local_module_support",
        ),
    )
    out["module_layer_policy_version"] = "RIPPLE_V1_Tier4_policy_2026-07-03"
    out["policy_freeze_timing"] = "after_DR_MVP_before_cross_trait_module_reselection"
    return out


def render_report(summary: pd.DataFrame, manifest: dict[str, object]) -> str:
    lines = [
        "# Cross-Trait Local Module Full Reselection Null",
        "",
        f"Created: {manifest['created_utc']}",
        "",
        "This analysis applies the frozen RIPPLE V1 Tier 4 module-layer policy after the DR_MVP policy freeze.",
        "",
        "All null replicates repeat ranking, top-fraction selection, connected-component extraction, module filtering and max-module statistic collection under degree-stratified score permutations.",
        "",
        "| Analysis | Modules | Selection-calibrated | Post hoc only | No local support | Min score P |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if summary.empty:
        lines.append("| none | 0 | 0 | 0 | 0 | NA |")
    else:
        for analysis_id, group in summary.groupby("analysis_id", observed=True):
            status = group["module_layer_claim_status"].astype(str)
            min_p = pd.to_numeric(group["full_reselection_score_p"], errors="coerce").min()
            p_text = "NA" if not np.isfinite(min_p) else f"{float(min_p):.4g}"
            lines.append(
                "| "
                f"{analysis_id} | {len(group):,} | "
                f"{int((status == 'selection_calibrated_module').sum()):,} | "
                f"{int((status == 'post_hoc_candidate_only').sum()):,} | "
                f"{int((status == 'no_local_module_support').sum()):,} | "
                f"{p_text} |"
            )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `selection_calibrated_module` requires full reselection max-module empirical P <= 0.05.",
            "- `post_hoc_candidate_only` is exploratory biological follow-up and does not support module-level statistical discovery.",
            "- This analysis does not upgrade any module to `topology_specific_module`; that status also requires module-level degree-preserving graph-null support.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.force:
        raise FileExistsError(f"{args.out_dir} exists and is not empty. Use --force.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "reports").mkdir(parents=True, exist_ok=True)

    specs = select_specs(args)
    summaries: list[pd.DataFrame] = []
    null_tables: list[pd.DataFrame] = []
    skipped: list[dict[str, object]] = []
    for spec in specs:
        missing = [path for path in (spec.scores_path, spec.modules_path, spec.graph_edges_path) if not path.exists()]
        if missing:
            skipped.append(
                {
                    "trait": spec.trait,
                    "analysis_id": spec.analysis_id,
                    "reason": "missing_required_input",
                    "missing_paths": ";".join(str(path) for path in missing),
                }
            )
            print(f"Skipping {spec.analysis_id}: missing {len(missing)} required inputs", flush=True)
            continue
        summary, nulls = run_one(args, spec)
        summary = add_frozen_module_policy_status(summary)
        nulls = nulls.assign(
            policy_freeze_timing="after_DR_MVP_before_cross_trait_module_reselection",
            script_path=str(THIS_SCRIPT),
        )
        write_table(args.out_dir / "tables" / f"{spec.trait}.module_full_reselection_summary.tsv", summary)
        write_table(args.out_dir / "tables" / f"{spec.trait}.module_full_reselection_null.tsv", nulls)
        summaries.append(summary)
        null_tables.append(nulls)

    combined_summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    combined_nulls = pd.concat(null_tables, ignore_index=True) if null_tables else pd.DataFrame()
    skipped_table = pd.DataFrame(skipped)
    write_table(args.out_dir / "tables" / "module_full_reselection_summary.all_traits.tsv", combined_summary)
    write_table(args.out_dir / "tables" / "module_full_reselection_null.all_traits.tsv.gz", combined_nulls)
    write_table(args.out_dir / "tables" / "module_full_reselection_skipped.tsv", skipped_table)

    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "analysis_id": "module_reselection_null_cross_trait_v1",
        "policy_freeze_timing": "after_DR_MVP_before_cross_trait_module_reselection",
        "traits": [spec.trait for spec in specs],
        "analysis_ids": [spec.analysis_id for spec in specs],
        "skipped": skipped,
        "n_full_reselection_null": args.n_reselection_null,
        "null_source": "degree_stratified_score_permutation",
        "degree_bins": args.degree_bins,
        "rank_fraction_grid": "0.01,0.02,0.05,0.10,0.15,0.20",
        "min_module_size": args.min_module_size,
        "min_module_subthreshold_genes": args.min_module_subthreshold_genes,
        "max_local_modules": args.max_local_modules,
        "broad_component_min_size": args.broad_component_min_size,
        "broad_component_fraction": args.broad_component_fraction,
        "seed": args.seed,
        "script_path": str(THIS_SCRIPT),
        "output_summary": str(args.out_dir / "tables" / "module_full_reselection_summary.all_traits.tsv"),
        "output_null": str(args.out_dir / "tables" / "module_full_reselection_null.all_traits.tsv.gz"),
    }
    (args.out_dir / "module_full_reselection_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.out_dir / "reports" / "module_full_reselection_report.md").write_text(
        render_report(combined_summary, manifest),
        encoding="utf-8",
    )
    print(f"Wrote cross-trait full reselection null outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
