#!/usr/bin/env python
"""Backfill RIPPLE V1 final reporting under the Z >= 2.5 manuscript gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ripple.claims import FINAL_MANUSCRIPT_Z_THRESHOLD, SUPPORTIVE_Z_THRESHOLD  # noqa: E402
from run_height_mvp import write_table  # noqa: E402


PRIVATE_ROOT = (
    Path("D:/path/to/ripple_private_workspace")
    if Path("D:/path/to/ripple_private_workspace").exists()
    else Path("/path/to/ripple_private_workspace")
)
DEFAULT_OUT_DIR = PRIVATE_ROOT / "30_analysis" / "final_reporting_z2p5_backfill"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def classify_z(z: float | None) -> str:
    if z is None or not np.isfinite(float(z)):
        return "not_applicable"
    z = float(z)
    if z >= FINAL_MANUSCRIPT_Z_THRESHOLD:
        return "final_positive"
    if z >= SUPPORTIVE_Z_THRESHOLD:
        return "supportive_only"
    return "not_positive"


def bool_z(z: float | None, threshold: float) -> bool:
    return bool(z is not None and np.isfinite(float(z)) and float(z) >= threshold)


def summary_to_claim_rows(
    *,
    analysis_id: str,
    summary: dict[str, Any],
) -> list[dict[str, object]]:
    trait = str(summary["trait"])
    graph = str(summary["graph_name"])
    observed_auc = float(summary["percolation_auc_observed"])
    local = summary.get("local_module_summary", {})
    rows: list[dict[str, object]] = []

    specs = [
        (
            "TIER_0_gene_signal",
            "percolation_auc_snp_pipeline_null",
            summary.get("snp_permutation_null_summary", {}),
            observed_auc,
        ),
        (
            "TIER_1_degree_calibrated_aggregation",
            "degree_calibrated_top_rank_aggregation",
            summary.get("degree_matched_node_null_summary", {}),
            observed_auc,
        ),
        (
            "TIER_3_topology_specific_support",
            "degree_preserving_graph_percolation",
            summary.get("degree_preserving_graph_null_summary", {}),
            observed_auc,
        ),
    ]
    for tier, statistic, stat_summary, observed in specs:
        z = stat_summary.get("z")
        rows.append(
            {
                "analysis_id": analysis_id,
                "trait": trait,
                "graph_name": graph,
                "tier": tier,
                "statistic": statistic,
                "observed": observed,
                "null_mean": stat_summary.get("mean"),
                "null_sd": stat_summary.get("sd"),
                "z": z,
                "empirical_p": stat_summary.get("empirical_p_upper", stat_summary.get("empirical_p")),
                "final_passed": bool_z(z, FINAL_MANUSCRIPT_Z_THRESHOLD),
                "supportive_passed": bool_z(z, SUPPORTIVE_Z_THRESHOLD),
                "claim_status": classify_z(z),
                "final_z_threshold": FINAL_MANUSCRIPT_Z_THRESHOLD,
                "supportive_z_threshold": SUPPORTIVE_Z_THRESHOLD,
            }
        )

    for item in summary.get("diffusion_kernel_summary", []) or []:
        z = item.get("z")
        rows.append(
            {
                "analysis_id": analysis_id,
                "trait": trait,
                "graph_name": str(item.get("graph_name", graph)),
                "tier": "TIER_2_graph_domain_aggregation",
                "statistic": f"diffusion_kernel_Tmax_{item.get('null_type', 'null')}",
                "observed": item.get("T_max"),
                "null_mean": item.get("null_mean"),
                "null_sd": item.get("null_sd"),
                "z": z,
                "empirical_p": item.get("empirical_p"),
                "final_passed": bool_z(z, FINAL_MANUSCRIPT_Z_THRESHOLD),
                "supportive_passed": bool_z(z, SUPPORTIVE_Z_THRESHOLD),
                "claim_status": classify_z(z),
                "final_z_threshold": FINAL_MANUSCRIPT_Z_THRESHOLD,
                "supportive_z_threshold": SUPPORTIVE_Z_THRESHOLD,
            }
        )

    n_modules = int(local.get("n_calibrated_modules", 0) or 0)
    n_topology = int(local.get("n_topology_specific_modules", 0) or 0)
    rows.append(
        {
            "analysis_id": analysis_id,
            "trait": trait,
            "graph_name": graph,
            "tier": "TIER_4_local_calibrated_modules",
            "statistic": "local_module_count",
            "observed": n_modules,
            "null_mean": np.nan,
            "null_sd": np.nan,
            "z": np.nan,
            "empirical_p": np.nan,
            "final_passed": n_modules > 0,
            "supportive_passed": n_modules > 0,
            "claim_status": "module_fwer_positive" if n_modules > 0 else "not_positive",
            "final_z_threshold": np.nan,
            "supportive_z_threshold": np.nan,
            "n_topology_specific_modules": n_topology,
        }
    )
    return rows


def diffusion_file_to_row(*, analysis_id: str, path: Path) -> dict[str, object]:
    table = pd.read_csv(path, sep="\t")
    if table.empty:
        raise ValueError(f"Empty diffusion summary: {path}")
    item = table.iloc[0].to_dict()
    z = float(item["z"])
    return {
        "analysis_id": analysis_id,
        "trait": item.get("trait"),
        "graph_name": item.get("graph_name"),
        "tier": "TIER_2_graph_domain_aggregation",
        "statistic": f"diffusion_kernel_Tmax_{item.get('null_type', 'null')}",
        "observed": item.get("T_max"),
        "null_mean": item.get("null_mean"),
        "null_sd": item.get("null_sd"),
        "z": z,
        "empirical_p": item.get("empirical_p"),
        "final_passed": bool_z(z, FINAL_MANUSCRIPT_Z_THRESHOLD),
        "supportive_passed": bool_z(z, SUPPORTIVE_Z_THRESHOLD),
        "claim_status": classify_z(z),
        "final_z_threshold": FINAL_MANUSCRIPT_Z_THRESHOLD,
        "supportive_z_threshold": SUPPORTIVE_Z_THRESHOLD,
    }


def row_value(rows: pd.DataFrame, tier: str, col: str) -> object:
    match = rows.loc[rows["tier"] == tier]
    if match.empty:
        return np.nan
    return match.iloc[0].get(col, np.nan)


def is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def format_z(value: object) -> str:
    if is_missing(value):
        return "NA"
    return f"{float(value):.3f}"


def format_status_z(status: object, z: object) -> str:
    if is_missing(status) or str(status) == "not_applicable":
        return "NA"
    if is_missing(z):
        return str(status)
    return f"{status} ({format_z(z)})"


def format_count(value: object) -> str:
    if is_missing(value):
        return "NA"
    return str(int(float(value)))


def render_report(dashboard: pd.DataFrame, trait_specificity: pd.DataFrame) -> str:
    lines = [
        "# RIPPLE V1 Final Reporting Backfill: Z >= 2.5",
        "",
        "Final Z-calibrated positive claims require `Z >= 2.5`.",
        "Results with `2.0 <= Z < 2.5` are supportive sensitivity evidence.",
        "Tier 4 module claims remain controlled by module-level empirical/FWER criteria.",
        "",
        "## Main Dashboard",
        "",
        "| Analysis | Tier 1 | Tier 2 | Tier 3 | Modules | Conclusion |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in dashboard.itertuples(index=False):
        lines.append(
            f"| {row.analysis_id} | {format_status_z(row.tier1_status, row.tier1_z)} | "
            f"{format_status_z(row.tier2_status, row.tier2_z)} | "
            f"{format_status_z(row.tier3_status, row.tier3_z)} | "
            f"{format_count(row.reportable_modules)} | {row.final_conclusion} |"
        )
    lines.extend(
        [
            "",
            "## Trait Specificity",
            "",
            "| Mode | SNP-null Z | Degree-matched Z | Degree-stratified Z | Graph-null Z | Modules | Final? |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in trait_specificity.itertuples(index=False):
        lines.append(
            f"| {row.mode} | {row.snp_z:.3f} | {row.degree_matched_z:.3f} | "
            f"{row.degree_stratified_z:.3f} | {row.degree_graph_z:.3f} | "
            f"{row.reportable_modules} | {row.degree_matched_status == 'final_positive'} |"
        )
    lines.extend(
        [
            "",
            "## Main Interpretation",
            "",
            "- DR_MVP default STRING remains final-positive for Tier 1 and Tier 2.",
            "- DR_MVP no-MHC-no-APOE remains final-positive for Tier 1 and Tier 2.",
            "- FVM vascular weighted diffusion remains final-positive sensitivity evidence.",
            "- Retina STRING min20 is downgraded to supportive-only sensitivity evidence.",
            "- Default STRING topology-specific support remains negative.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.force:
        raise FileExistsError(f"{args.out_dir} exists and is not empty. Use --force.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    analysis_root = PRIVATE_ROOT / "30_analysis"
    main_specs = [
        (
            "DR_MVP_default_final5000",
            analysis_root / "dr_mvp_string_final5000" / "reports" / "DR_MVP.analysis_ready_summary.json",
        ),
        (
            "DR_MVP_no_MHC_no_APOE_final5000",
            analysis_root
            / "dr_mvp_no_mhc_no_apoe_final5000"
            / "reports"
            / "DR_MVP_NO_MHC_NO_APOE.analysis_ready_summary.json",
        ),
    ]
    claim_rows: list[dict[str, object]] = []
    for analysis_id, path in main_specs:
        claim_rows.extend(summary_to_claim_rows(analysis_id=analysis_id, summary=load_json(path)))

    sensitivity_specs = [
        (
            "FVM_vascular_weighted_diffusion_final5000",
            analysis_root
            / "dr_mvp_graph_sensitivity"
            / "fvm_vascular_weighted_diffusion_final5000"
            / "DR_MVP_FVM_VASCULAR_WEIGHTED.diffusion_kernel_summary.tsv",
        ),
        (
            "retina_string_min20_diffusion_final5000",
            analysis_root
            / "dr_mvp_graph_sensitivity"
            / "retina_string_min20_diffusion_final5000"
            / "DR_MVP_RETINA_STRING_MIN20.diffusion_kernel_summary.tsv",
        ),
    ]
    for analysis_id, path in sensitivity_specs:
        claim_rows.append(diffusion_file_to_row(analysis_id=analysis_id, path=path))

    claim_table = pd.DataFrame(claim_rows)
    write_table(args.out_dir / "final_claim_tiers_z2p5.tsv", claim_table)

    dashboard_rows: list[dict[str, object]] = []
    for analysis_id, group in claim_table.groupby("analysis_id", sort=False):
        tier1_z = row_value(group, "TIER_1_degree_calibrated_aggregation", "z")
        tier2_z = row_value(group, "TIER_2_graph_domain_aggregation", "z")
        tier3_z = row_value(group, "TIER_3_topology_specific_support", "z")
        modules = row_value(group, "TIER_4_local_calibrated_modules", "observed")
        tier1_status = row_value(group, "TIER_1_degree_calibrated_aggregation", "claim_status")
        tier2_status = row_value(group, "TIER_2_graph_domain_aggregation", "claim_status")
        tier3_status = row_value(group, "TIER_3_topology_specific_support", "claim_status")
        if analysis_id == "retina_string_min20_diffusion_final5000":
            conclusion = "supportive tissue-context sensitivity only"
        elif analysis_id == "FVM_vascular_weighted_diffusion_final5000":
            conclusion = "final-positive weighted graph-domain sensitivity"
        elif tier1_status == "final_positive" and tier2_status == "final_positive":
            conclusion = "final-positive degree-calibrated and graph-domain support"
        else:
            conclusion = "not final-positive"
        dashboard_rows.append(
            {
                "analysis_id": analysis_id,
                "trait": group.iloc[0].get("trait", np.nan),
                "graph_name": group.iloc[0].get("graph_name", np.nan),
                "tier1_z": tier1_z,
                "tier1_status": tier1_status,
                "tier2_z": tier2_z,
                "tier2_status": tier2_status,
                "tier3_z": tier3_z,
                "tier3_status": tier3_status,
                "reportable_modules": modules,
                "final_conclusion": conclusion,
            }
        )
    dashboard = pd.DataFrame(dashboard_rows)
    write_table(args.out_dir / "final_dashboard_z2p5.tsv", dashboard)

    ts_path = analysis_root / "dr_mvp_trait_specificity" / "reports" / "trait_specificity_summary.json"
    ts_data = json.loads(ts_path.read_text(encoding="utf-8"))
    ts_rows = []
    for item in ts_data:
        degree_z = float(item["degree_matched_node_null_summary"]["z"])
        ts_rows.append(
            {
                "mode": item["mode_name"],
                "snp_z": float(item["snp_permutation_null_summary"]["z"]),
                "degree_matched_z": degree_z,
                "degree_matched_status": classify_z(degree_z),
                "degree_stratified_z": float(item["degree_stratified_null_summary"]["z"]),
                "degree_graph_z": float(item["degree_preserving_graph_null_summary"]["z"]),
                "reportable_modules": int(item["local_module_summary"]["n_calibrated_modules"]),
            }
        )
    trait_specificity = pd.DataFrame(ts_rows)
    write_table(args.out_dir / "trait_specificity_z2p5.tsv", trait_specificity)

    report = render_report(dashboard, trait_specificity)
    (args.out_dir / "final_reporting_z2p5_backfill_report.md").write_text(report, encoding="utf-8")
    print(f"Wrote Z>=2.5 final reporting backfill to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
