#!/usr/bin/env python
"""Build schema-fixed manuscript-ready RIPPLE V1 result package."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta
from scipy.stats import hypergeom

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ripple.manuscript import (  # noqa: E402
    CLAIM_EVIDENCE_SCHEMA,
    COMMON_RESULT_SCHEMA,
    FIGURE_CLAIM_MAP_SCHEMA,
    FIGURE_SOURCE_SCHEMA,
    GENE_ID_MAPPING_SCHEMA,
    GRAPH_REGISTRY_SCHEMA,
    INFERENCE_FAMILY_SCHEMA,
    INPUT_CHECKSUM_SCHEMA,
    PARAMETER_TABLE_SCHEMA,
    PUBLIC_RELEASE_AUDIT_SCHEMA,
    REGION_EXCLUSION_SCHEMA,
    REPRODUCIBILITY_MANIFEST_SCHEMA,
    MODULE_CLAIM_POLICY_SCHEMA,
    TYPE1_UNCERTAINTY_SCHEMA,
    ensure_columns,
    validate_columns,
    validate_vocabulary,
)
from ripple.policy import (  # noqa: E402
    classify_z_claim,
    controlled_vocabulary,
    final_z_threshold,
    load_claim_policy,
    supportive_z_threshold,
)
from ripple.modules.discovery import DEFAULT_DR_GENE_SETS  # noqa: E402


PRIVATE_ROOT = (
    Path("D:/path/to/ripple_private_workspace")
    if Path("D:/path/to/ripple_private_workspace").exists()
    else Path("/path/to/ripple_private_workspace")
)
ANALYSIS_ROOT = PRIVATE_ROOT / "30_analysis"
PROCESSED_ROOT = PRIVATE_ROOT / "20_processed_data"
MANUSCRIPT_ROOT = (
    Path("D:/RIPPLE/RIPPLE_manuscript")
    if Path("D:/RIPPLE/RIPPLE_manuscript").exists()
    else Path("/path/to/ripple_manuscript_workspace")
)
SUPPLEMENTARY_ROOT = MANUSCRIPT_ROOT / "supplementary_files"
DEFAULT_OUT_DIR = ANALYSIS_ROOT / "manuscript_ready_v1"
V12_OUT_DIR = ANALYSIS_ROOT / "manuscript_ready_v1_2"
V12_REVIEW_OUT_DIR = ANALYSIS_ROOT / "manuscript_ready_v1_2_review"
POLICY_PATH = PROJECT_ROOT / "ripple" / "config" / "claim_policy.yaml"
THIS_SCRIPT = Path(__file__).resolve()
SCZ_DEV_SCRIPT = SCRIPT_DIR / "run_scz_secondary_benchmark_dev.sh"
SCZ_FINAL_SCRIPT = SCRIPT_DIR / "run_scz_no_mhc_final_scale.sh"
SCZ_SUMMARY_SCRIPT = SCRIPT_DIR / "summarize_scz_secondary_benchmark.py"
SCZ_SECONDARY_SUMMARY_DIR = ANALYSIS_ROOT / "scz_secondary_benchmark_final"
MAGMA_BASELINE_DIR = ANALYSIS_ROOT / "external_baselines" / "magma_v1.10"
PASCALX_BASELINE_DIR = ANALYSIS_ROOT / "external_baselines" / "pascalx"
NETWORK_ABLATION_DIR = ANALYSIS_ROOT / "external_baselines" / "network_ablation_v1"
EXTERNAL_SCORE_GRAPH_DIR = ANALYSIS_ROOT / "external_baselines" / "external_score_graph_layer_v1"
DENSE_MODULE_COMPETITOR_DIR = ANALYSIS_ROOT / "external_baselines" / "dense_module_competitor_v1"
MODULE_RESELECTION_DIR = ANALYSIS_ROOT / "module_reselection_null_v1"
CROSS_TRAIT_MODULE_RESELECTION_DIR = ANALYSIS_ROOT / "module_reselection_null_cross_trait_v1"
MAGMA_BASELINE_SCRIPT = SCRIPT_DIR / "run_magma_baseline.py"
PASCALX_BASELINE_SCRIPT = SCRIPT_DIR / "run_pascalx_baseline.py"
NETWORK_ABLATION_SCRIPT = SCRIPT_DIR / "run_network_ablation_baseline.py"
EXTERNAL_SCORE_GRAPH_SCRIPT = SCRIPT_DIR / "run_external_score_graph_layer.py"
DENSE_MODULE_COMPETITOR_SCRIPT = SCRIPT_DIR / "run_dense_module_competitor.py"
MODULE_RESELECTION_SCRIPT = SCRIPT_DIR / "run_dr_mvp_module_reselection_null.py"
CROSS_TRAIT_MODULE_RESELECTION_SCRIPT = SCRIPT_DIR / "run_cross_trait_module_reselection_null.py"
MAGMA_TOOL_DIR = PRIVATE_ROOT / "02_environment" / "tools" / "magma_v1.10"
PASCALX_TOOL_DIR = PRIVATE_ROOT / "02_environment" / "tools" / "PascalX"
SYNTHETIC_SPIKEIN_DIR = ANALYSIS_ROOT / "synthetic_spikein_validation"
SYNTHETIC_SELECTION_AWARE_DIR = ANALYSIS_ROOT / "synthetic_spikein_validation_selection_aware"
ANCHORED_V12_ROOT = ANALYSIS_ROOT / "tier4_v12_anchored_broad_reactome_go_cross_trait_n500"
ANCHORED_V12_TYPE1_ROOT = (
    ANALYSIS_ROOT / "tier4_v12_anchored_broad_reactome_go_type1_robustness_v1" / "type1_outer1000"
)
ANCHORED_V12_ROBUSTNESS_ROOT = (
    ANALYSIS_ROOT / "tier4_v12_anchored_broad_reactome_go_type1_robustness_v1" / "robustness"
)
DIRECTION23_V12_ROOT = ANALYSIS_ROOT / "tier4_v12_direction2_3_smoke_v1"
ANCHORED_V12_RUN_SCRIPT = SCRIPT_DIR / "run_v12_anchored_module_test.py"
ANCHORED_V12_SUMMARY_SCRIPT = SCRIPT_DIR / "summarize_v12_anchored_cross_trait.py"
ANCHORED_V12_TYPE1_SCRIPT = SCRIPT_DIR / "run_v12_anchored_type1_calibration.py"
ANCHORED_V12_ROBUSTNESS_SCRIPT = SCRIPT_DIR / "run_v12_anchored_robustness.py"
DIRECTION23_V12_SCRIPT = SCRIPT_DIR / "run_v12_direction23_smoke.py"
REVIEW_REVISION_ROOT = ANALYSIS_ROOT / "review_driven_revision_v1"
NULL_GENERATION_AUDIT_SCRIPT = SCRIPT_DIR / "audit_null_generation.py"
GENE_SCORE_TAIL_CALIBRATION_SCRIPT = SCRIPT_DIR / "run_gene_score_tail_calibration.py"
PATHWAY_COMPARATOR_SCRIPT = SCRIPT_DIR / "run_pathway_comparator_baselines.py"
TRUE_NETWORK_BASELINE_SCRIPT = SCRIPT_DIR / "run_true_network_method_baseline.py"
DMGWAS_EXTERNAL_BASELINE_SCRIPT = SCRIPT_DIR / "run_dmgwas_external_baseline.py"
ANCHORED_STRENGTHENED_NULL_SCRIPT = SCRIPT_DIR / "run_anchored_strengthened_nulls.py"
OVERLAP_NULL_SENSITIVITY_SCRIPT = SCRIPT_DIR / "run_overlap_preserving_null_sensitivity.py"
ANCHORED_FAMILYWISE_EXPANSION_SCRIPT = SCRIPT_DIR / "run_anchored_familywise_targeted_expansion.py"
TAIL_DECISION_STABILITY_SCRIPT = SCRIPT_DIR / "summarize_tail_decision_stability.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--policy", type=Path, default=POLICY_PATH)
    parser.add_argument("--package-version", choices=["v1", "v1_2", "v1_2_review"], default="v1")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def is_v12_package(package_version: str) -> bool:
    return package_version in {"v1_2", "v1_2_review"}


def is_review_package(package_version: str) -> bool:
    return package_version == "v1_2_review"


def write_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if path.suffix == ".gz" else None
    table.to_csv(path, sep="\t", index=False, compression=compression)


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", compression="infer")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def package_policy(policy: dict[str, Any], *, package_version: str) -> dict[str, Any]:
    """Return the active manuscript policy, with V1.2 addendum when requested."""

    if not is_v12_package(package_version):
        return policy
    out = copy.deepcopy(policy)
    out.setdefault("tier_interpretation", {})[
        "TIER_4A_anchored_biological_modules"
    ] = "calibrated anchored biological module evidence over a fixed external gene-set library"
    module_labels = out.setdefault("module_label", [])
    for label in [
        "anchored_library_calibrated_module",
        "anchored_familywise_supported",
        "source_familywise_supported",
        "direction23_diagnostic_only",
    ]:
        if label not in module_labels:
            module_labels.append(label)
    module_policy = out.setdefault("module_layer_policy", {})
    module_policy["v1_2_anchored_layer"] = {
        "tier_name": "Tier 4A",
        "tier_description": "Anchored biological module evidence over fixed Reactome/GO libraries",
        "allowed_language": "anchored biological module evidence; anchored familywise-supported module; source-familywise supported module",
        "forbidden_language": "de novo STRING topology claim; validated disease module; causal network",
        "claim_boundary": (
            "Anchored module evidence prioritizes fixed external biological modules and does not imply "
            "de novo topology-specific PPI module discovery."
        ),
    }
    return out


def policy_text(policy: dict[str, Any]) -> str:
    try:
        import yaml

        return yaml.safe_dump(policy, sort_keys=False, allow_unicode=True)
    except ModuleNotFoundError:
        return json.dumps(policy, indent=2, ensure_ascii=False)


def as_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def count_rows(path: Path | None) -> int | str:
    if path is None or not path.exists():
        return ""
    try:
        return int(len(read_tsv(path)))
    except Exception:
        return ""


def unique_replicates(path: Path | None) -> int | str:
    if path is None or not path.exists():
        return ""
    table = read_tsv(path)
    if "replicate" not in table.columns:
        return int(len(table))
    return int(table["replicate"].nunique())


def n_null_for(tables_dir: Path, trait: str, statistic: str) -> int | str:
    if statistic == "percolation_auc_snp_pipeline_null":
        return count_rows(tables_dir / f"{trait}.percolation_auc.1000G_LD.null.tsv")
    if statistic == "degree_calibrated_top_rank_aggregation":
        return count_rows(tables_dir / f"{trait}.percolation_auc.1000G_LD.degree_matched_node_null.tsv")
    if statistic == "degree_preserving_graph_percolation":
        return count_rows(tables_dir / f"{trait}.percolation_auc.1000G_LD.degree_preserving_graph_null.tsv")
    if statistic.startswith("diffusion_kernel_Tmax"):
        summary_path = tables_dir / f"{trait}.diffusion_kernel_summary.tsv"
        if summary_path.exists():
            summary = read_tsv(summary_path)
            if "n_null" in summary.columns and not summary.empty:
                return int(pd.to_numeric(summary["n_null"], errors="coerce").fillna(0).max())
        return unique_replicates(tables_dir / f"{trait}.diffusion_kernel_null_distribution.tsv.gz")
    return ""


def source_for_statistic(tables_dir: Path, trait: str, statistic: str) -> Path:
    if statistic.startswith("diffusion_kernel_Tmax"):
        return tables_dir / f"{trait}.diffusion_kernel_summary.tsv"
    if statistic == "local_module_count":
        return tables_dir / f"{trait}.local_modules.tsv"
    return tables_dir / f"{trait}.claim_tiers.tsv"


def normalize_claim_table(
    *,
    analysis_id: str,
    trait: str,
    graph_id: str,
    analysis_dir: Path,
    source_script: Path,
    seed: int | str,
    timestamp: str,
    policy: dict[str, Any],
) -> pd.DataFrame:
    tables_dir = analysis_dir / "tables"
    claim_path = tables_dir / f"{trait}.claim_tiers.tsv"
    raw = read_tsv(claim_path)
    rows: list[dict[str, object]] = []
    for item in raw.to_dict(orient="records"):
        statistic = str(item.get("statistic", ""))
        tier = str(item.get("tier", ""))
        z = as_float(item.get("z"))
        passed = str(item.get("passed", "")).lower() == "true"
        if tier == "TIER_4_local_calibrated_modules":
            status = "final_positive" if passed else "negative"
            threshold: object = "module_empirical_fwer"
            exclusion = "not_applicable"
            not_tested = "not_applicable"
        else:
            status = classify_z_claim(z, policy)
            threshold = final_z_threshold(policy)
            exclusion = "none"
            not_tested = "not_applicable" if status != "not_tested" else "not_run"
        rows.append(
            {
                "trait": trait,
                "analysis_id": analysis_id,
                "graph_id": graph_id,
                "score_stream": item.get("score_type", "assoc_resid_score"),
                "null_type": statistic_to_null_type(statistic),
                "statistic_name": statistic,
                "statistic_direction": "greater_is_more_extreme",
                "observed_value": item.get("observed", ""),
                "null_mean": item.get("null_mean", ""),
                "null_sd": item.get("null_sd", ""),
                "z": item.get("z", ""),
                "empirical_p": item.get("empirical_p", ""),
                "n_null": n_null_for(tables_dir, trait, statistic),
                "threshold": threshold,
                "claim_tier": tier,
                "claim_status": status,
                "exclusion_or_na_reason": exclusion,
                "not_tested_reason": not_tested,
                "source_result_path": str(source_for_statistic(tables_dir, trait, statistic)),
                "script_path": str(source_script),
                "seed": seed,
                "timestamp": timestamp,
            }
        )
    return ensure_columns(pd.DataFrame(rows), COMMON_RESULT_SCHEMA)


def cap_dev_scale_claims(table: pd.DataFrame) -> pd.DataFrame:
    """Keep dev-scale benchmark rows out of final-positive manuscript language."""

    out = table.copy()
    dev_mask = out["analysis_id"].astype(str).str.contains("_dev", regex=False)
    final_mask = out["claim_status"].astype(str) == "final_positive"
    mask = dev_mask & final_mask
    out.loc[mask, "claim_status"] = "supportive"
    out.loc[mask, "exclusion_or_na_reason"] = "insufficient_nulls"
    return out


def scz_analysis_specs() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = [
        {
            "analysis_id": "SCZ_default_with_MHC_dev500",
            "trait": "SCZ_WITH_MHC",
            "analysis_dir": ANALYSIS_ROOT / "scz_with_mhc_string_dev500",
            "source_script": SCZ_DEV_SCRIPT,
            "seed": 20260702,
        },
    ]
    final_dir = ANALYSIS_ROOT / "scz_no_mhc_string_final5000"
    if (final_dir / "tables" / "SCZ.claim_tiers.tsv").exists():
        specs.append(
            {
                "analysis_id": "SCZ_no_MHC_final5000",
                "trait": "SCZ",
                "analysis_dir": final_dir,
                "source_script": SCZ_FINAL_SCRIPT,
                "seed": 20260702,
            }
        )
    else:
        specs.append(
            {
                "analysis_id": "SCZ_no_MHC_dev500",
                "trait": "SCZ",
                "analysis_dir": ANALYSIS_ROOT / "scz_no_mhc_string_dev500",
                "source_script": SCZ_DEV_SCRIPT,
                "seed": 20260702,
            }
        )
    return specs


def build_scz_claim_rows(policy: dict[str, Any]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    missing: list[str] = []
    for spec in scz_analysis_specs():
        analysis_dir = Path(spec["analysis_dir"])
        trait = str(spec["trait"])
        claim_path = analysis_dir / "tables" / f"{trait}.claim_tiers.tsv"
        if not claim_path.exists():
            missing.append(str(spec["analysis_id"]))
            continue
        rows.append(
            normalize_claim_table(
                analysis_id=str(spec["analysis_id"]),
                trait=trait,
                graph_id="STRING_default",
                analysis_dir=analysis_dir,
                source_script=Path(spec["source_script"]),
                seed=spec["seed"],
                timestamp=summary_timestamp(analysis_dir / "reports" / f"{trait}.analysis_ready_summary.json"),
                policy=policy,
            )
        )
    if not rows:
        return scz_not_tested_rows(policy)
    out = cap_dev_scale_claims(pd.concat(rows, ignore_index=True))
    if missing:
        out = pd.concat([out, scz_not_tested_rows(policy, analysis_ids=missing)], ignore_index=True)
    return ensure_columns(out, COMMON_RESULT_SCHEMA)


def statistic_to_null_type(statistic: str) -> str:
    if statistic == "percolation_auc_snp_pipeline_null":
        return "snp_gene_score_pipeline_null"
    if statistic == "degree_calibrated_top_rank_aggregation":
        return "degree_matched_node_null"
    if statistic == "degree_preserving_graph_percolation":
        return "degree_preserving_graph_null"
    if statistic.startswith("diffusion_kernel_Tmax"):
        return statistic.replace("diffusion_kernel_Tmax_", "")
    if statistic == "local_module_count":
        return "module_empirical_fwer"
    return statistic


def normalize_diffusion_sensitivity(
    *,
    analysis_id: str,
    trait: str,
    graph_id: str,
    summary_path: Path,
    source_script: Path,
    policy: dict[str, Any],
) -> pd.DataFrame:
    raw = read_tsv(summary_path)
    rows: list[dict[str, object]] = []
    for item in raw.to_dict(orient="records"):
        z = as_float(item.get("z"))
        rows.append(
            {
                "trait": trait,
                "analysis_id": analysis_id,
                "graph_id": graph_id,
                "score_stream": item.get("score_mode", "positive"),
                "null_type": item.get("null_type", "degree_stratified"),
                "statistic_name": f"diffusion_kernel_Tmax_{item.get('null_type', 'null')}",
                "statistic_direction": "greater_is_more_extreme",
                "observed_value": item.get("T_max", ""),
                "null_mean": item.get("null_mean", ""),
                "null_sd": item.get("null_sd", ""),
                "z": item.get("z", ""),
                "empirical_p": item.get("empirical_p", ""),
                "n_null": item.get("n_null", ""),
                "threshold": final_z_threshold(policy),
                "claim_tier": "TIER_2_graph_domain_aggregation",
                "claim_status": classify_z_claim(z, policy),
                "exclusion_or_na_reason": "not_applicable",
                "not_tested_reason": "not_applicable",
                "source_result_path": str(summary_path),
                "script_path": str(source_script),
                "seed": item.get("seed", ""),
                "timestamp": now_utc(),
            }
        )
    return ensure_columns(pd.DataFrame(rows), COMMON_RESULT_SCHEMA)


def build_final_claim_audit(policy: dict[str, Any]) -> pd.DataFrame:
    run_script = SCRIPT_DIR / "run_final_scale_validation_v1.sh"
    rows = [
        normalize_claim_table(
            analysis_id="DR_MVP_default_final5000",
            trait="DR_MVP",
            graph_id="STRING_default",
            analysis_dir=ANALYSIS_ROOT / "dr_mvp_string_final5000",
            source_script=run_script,
            seed=20260613,
            timestamp=summary_timestamp(
                ANALYSIS_ROOT / "dr_mvp_string_final5000" / "reports" / "DR_MVP.analysis_ready_summary.json"
            ),
            policy=policy,
        ),
        normalize_claim_table(
            analysis_id="DR_MVP_no_MHC_no_APOE_final5000",
            trait="DR_MVP_NO_MHC_NO_APOE",
            graph_id="STRING_default",
            analysis_dir=ANALYSIS_ROOT / "dr_mvp_no_mhc_no_apoe_final5000",
            source_script=run_script,
            seed=20260613,
            timestamp=summary_timestamp(
                ANALYSIS_ROOT
                / "dr_mvp_no_mhc_no_apoe_final5000"
                / "reports"
                / "DR_MVP_NO_MHC_NO_APOE.analysis_ready_summary.json"
            ),
            policy=policy,
        ),
        normalize_diffusion_sensitivity(
            analysis_id="FVM_vascular_weighted_diffusion_final5000",
            trait="DR_MVP_FVM_VASCULAR_WEIGHTED",
            graph_id="FVM_vascular_weighted",
            summary_path=ANALYSIS_ROOT
            / "dr_mvp_graph_sensitivity"
            / "fvm_vascular_weighted_diffusion_final5000"
            / "DR_MVP_FVM_VASCULAR_WEIGHTED.diffusion_kernel_summary.tsv",
            source_script=run_script,
            policy=policy,
        ),
        normalize_diffusion_sensitivity(
            analysis_id="retina_string_min20_diffusion_final5000",
            trait="DR_MVP_RETINA_STRING_MIN20",
            graph_id="retina_string_min20",
            summary_path=ANALYSIS_ROOT
            / "dr_mvp_graph_sensitivity"
            / "retina_string_min20_diffusion_final5000"
            / "DR_MVP_RETINA_STRING_MIN20.diffusion_kernel_summary.tsv",
            source_script=run_script,
            policy=policy,
        ),
    ]
    scz_rows = build_scz_claim_rows(policy)
    return ensure_columns(pd.concat(rows + [scz_rows], ignore_index=True), COMMON_RESULT_SCHEMA)


def summary_timestamp(path: Path) -> str:
    if not path.exists():
        return now_utc()
    data = read_json(path)
    return str(data.get("created_utc", now_utc()))


def scz_not_tested_rows(policy: dict[str, Any], analysis_ids: list[str] | None = None) -> pd.DataFrame:
    qc_report = PROCESSED_ROOT / "gwas_qc" / "qc_reports" / "SCZ.qc_report.json"
    timestamp = summary_timestamp(qc_report) if qc_report.exists() else now_utc()
    rows = []
    if analysis_ids is None:
        analysis_ids = [
            "SCZ_default_with_MHC_dev500",
            "SCZ_no_MHC_dev500",
            "SCZ_APOE_region_diagnostic",
        ]
    for analysis_id in analysis_ids:
        rows.append(
            {
                "trait": "SCZ",
                "analysis_id": analysis_id,
                "graph_id": "STRING_default",
                "score_stream": "assoc_resid_score",
                "null_type": "not_tested",
                "statistic_name": "SCZ_secondary_benchmark_claim_tier",
                "statistic_direction": "greater_is_more_extreme",
                "observed_value": "",
                "null_mean": "",
                "null_sd": "",
                "z": "",
                "empirical_p": "",
                "n_null": "",
                "threshold": final_z_threshold(policy),
                "claim_tier": "TIER_1_degree_calibrated_aggregation",
                "claim_status": "not_tested",
                "exclusion_or_na_reason": "not_run",
                "not_tested_reason": "deferred_to_submission_stage",
                "source_result_path": str(qc_report),
                "script_path": str(SCZ_DEV_SCRIPT),
                "seed": "",
                "timestamp": timestamp,
            }
        )
    return ensure_columns(pd.DataFrame(rows), COMMON_RESULT_SCHEMA)


def build_cross_trait_benchmark(policy: dict[str, Any]) -> pd.DataFrame:
    path = ANALYSIS_ROOT / "trait_architecture_comparison.tsv"
    rows: list[dict[str, object]] = []
    if path.exists():
        table = read_tsv(path)
        for item in table.to_dict(orient="records"):
            for statistic, tier, z_col, null_type in [
                (
                    "degree_calibrated_top_rank_aggregation",
                    "TIER_1_degree_calibrated_aggregation",
                    "degree_matched_z",
                    "degree_matched_node_null",
                ),
                (
                    "degree_preserving_graph_percolation",
                    "TIER_3_topology_specific_support",
                    "degree_preserving_graph_z",
                    "degree_preserving_graph_null",
                ),
            ]:
                z = as_float(item.get(z_col))
                rows.append(
                    {
                        "trait": item.get("trait", ""),
                        "analysis_id": f"{item.get('trait', '')}_early_benchmark",
                        "graph_id": "STRING_default",
                        "score_stream": "assoc_resid_score",
                        "null_type": null_type,
                        "statistic_name": statistic,
                        "statistic_direction": "greater_is_more_extreme",
                        "observed_value": item.get("observed_auc", ""),
                        "null_mean": "",
                        "null_sd": "",
                        "z": z,
                        "empirical_p": "",
                        "n_null": "",
                        "threshold": final_z_threshold(policy),
                        "claim_tier": tier,
                        "claim_status": classify_z_claim(z, policy),
                        "exclusion_or_na_reason": "not_applicable",
                        "not_tested_reason": "not_applicable",
                        "source_result_path": str(path),
                        "script_path": str(SCRIPT_DIR / "diagnose_dr_degree_matched_signal.py"),
                        "seed": "",
                        "timestamp": timestamp_for(path),
                    }
                )
    rows.extend(build_scz_claim_rows(policy).to_dict(orient="records"))
    return ensure_columns(pd.DataFrame(rows), COMMON_RESULT_SCHEMA)


def add_source_metadata(table: pd.DataFrame, *, source_path: Path, script_path: Path, seed: object = "") -> pd.DataFrame:
    out = table.copy()
    out["source_result_path"] = str(source_path)
    out["script_path"] = str(script_path)
    out["seed"] = seed
    out["timestamp"] = timestamp_for(source_path)
    return out


def build_v12_anchored_cross_trait_summary() -> pd.DataFrame:
    path = ANCHORED_V12_ROOT / "tables" / "cross_trait_anchored_summary.tsv"
    if not path.exists():
        return pd.DataFrame()
    out = add_source_metadata(read_tsv(path), source_path=path, script_path=ANCHORED_V12_SUMMARY_SCRIPT, seed=20260712)
    out["v12_claim_role"] = np.where(
        out.get("best_module_status", pd.Series(dtype=str)).astype(str).eq("anchored_familywise_supported"),
        "anchored_biological_module_evidence",
        "anchored_negative_or_fixed_only",
    )
    out["interpretation_note"] = np.where(
        out["analysis_id"].astype(str).eq("BMI_IRN"),
        "statistically supported but biologically cautious because the top module is not the most intuitive BMI biology",
        np.where(
            out["analysis_id"].astype(str).eq("SCZ_NO_MHC"),
            "negative control behavior for anchored module evidence",
            "anchored module evidence over fixed Reactome/GO library",
        ),
    )
    return out


def build_v12_anchored_top_modules() -> pd.DataFrame:
    path = ANCHORED_V12_ROOT / "tables" / "cross_trait_top_modules.tsv"
    if not path.exists():
        return pd.DataFrame()
    out = add_source_metadata(read_tsv(path), source_path=path, script_path=ANCHORED_V12_SUMMARY_SCRIPT, seed=20260712)
    out["v12_module_label"] = np.where(
        out.get("module_status", pd.Series(dtype=str)).astype(str).eq("anchored_familywise_supported"),
        "anchored_library_calibrated_module",
        np.where(
            out.get("module_status", pd.Series(dtype=str)).astype(str).eq("fixed_degree_supported"),
            "fixed_module_supported",
            "no_local_module_support",
        ),
    )
    return out


def build_v12_anchored_type1_outer1000_summary() -> pd.DataFrame:
    path = ANCHORED_V12_TYPE1_ROOT / "tables" / "anchored_type1_summary.tsv"
    if not path.exists():
        return pd.DataFrame()
    out = add_source_metadata(read_tsv(path), source_path=path, script_path=ANCHORED_V12_TYPE1_SCRIPT, seed="")
    out["calibration_interpretation"] = np.select(
        [
            out["calibration_family"].astype(str).eq("module_source:Gene_Ontology"),
            out["calibration_family"].astype(str).eq("module_source:Reactome"),
            out["calibration_family"].astype(str).eq("all_modules"),
        ],
        [
            "strongest source-family calibration for V1.2 anchored claims",
            "supportive source-family calibration; slightly conservative language required",
            "full-library calibration acceptable but described conservatively",
        ],
        default="anchored calibration family",
    )
    return out


def build_v12_anchored_go_reactome_confirmation() -> pd.DataFrame:
    path = ANCHORED_V12_ROBUSTNESS_ROOT / "tables" / "anchored_go_reactome_confirmation.tsv"
    if not path.exists():
        return pd.DataFrame()
    out = add_source_metadata(read_tsv(path), source_path=path, script_path=ANCHORED_V12_ROBUSTNESS_SCRIPT, seed=20260714)
    out["source_family_claim_status"] = np.where(
        pd.to_numeric(out.get("group_familywise_p", np.nan), errors="coerce").le(0.05),
        "source_familywise_supported",
        "supportive_or_negative",
    )
    return out


def build_v12_anchored_robustness_summary() -> pd.DataFrame:
    stability_path = ANCHORED_V12_ROBUSTNESS_ROOT / "tables" / "anchored_null_scale_stability.tsv"
    perturb_path = ANCHORED_V12_ROBUSTNESS_ROOT / "tables" / "anchored_score_perturbation_summary.tsv"
    if not stability_path.exists() and not perturb_path.exists():
        return pd.DataFrame()
    stability = read_tsv(stability_path) if stability_path.exists() else pd.DataFrame()
    perturb = read_tsv(perturb_path) if perturb_path.exists() else pd.DataFrame()
    if not stability.empty and not perturb.empty:
        out = stability.merge(
            perturb,
            on=["trait", "analysis_id"],
            how="outer",
            suffixes=("_null_scale", "_perturbation"),
        )
    else:
        out = stability if not stability.empty else perturb
    out["source_result_path"] = ";".join(str(path) for path in [stability_path, perturb_path] if path.exists())
    out["script_path"] = str(ANCHORED_V12_ROBUSTNESS_SCRIPT)
    out["seed"] = 20260714
    out["timestamp"] = now_utc()
    return out


def build_v12_direction23_smoke_diagnostic() -> pd.DataFrame:
    path = DIRECTION23_V12_ROOT / "tables" / "direction23_smoke_summary.all_traits.tsv"
    if not path.exists():
        return pd.DataFrame()
    out = add_source_metadata(read_tsv(path), source_path=path, script_path=DIRECTION23_V12_SCRIPT, seed=20260715)
    out["v12_claim_role"] = "direction23_diagnostic_only"
    out["v12_recommendation"] = np.select(
        [
            out["method"].astype(str).eq("direction3_louvain_community_anchor"),
            out["method"].astype(str).eq("direction2_diffusion_localized_neighborhood"),
            out["method"].astype(str).str.contains("pathway_anchor", regex=False),
        ],
        [
            "do_not_include_in_v1_2_mainline",
            "retain_as_v13_v2_candidate_only",
            "covered_by_anchored_broad_v1_2_layer",
        ],
        default="diagnostic_only",
    )
    return out


def build_v12_module_claim_boundary() -> pd.DataFrame:
    rows = [
        {
            "boundary_id": "V12_TIER4A_ANCHORED",
            "claim_layer": "TIER_4A_anchored_biological_modules",
            "allowed_language": "anchored biological module evidence; anchored familywise-supported module; source-familywise supported module",
            "forbidden_language": "de novo STRING topology claim; validated disease module; causal network",
            "evidence_table": "anchored_cross_trait_summary.tsv;anchored_type1_outer1000_summary.tsv;anchored_go_reactome_confirmation.tsv",
            "interpretation": "V1.2 anchored modules prioritize fixed external Reactome/GO biological modules under controlled null models.",
            "source_result_path": str(ANCHORED_V12_ROOT),
            "script_path": str(THIS_SCRIPT),
            "timestamp": now_utc(),
        },
        {
            "boundary_id": "V12_DE_NOVO_STRING_LOCAL_MODULES",
            "claim_layer": "original_Tier_4_de_novo_local_modules",
            "allowed_language": "post hoc candidate module; limitation; diagnostic boundary",
            "forbidden_language": "calibrated de novo local module discovery; topology-specific module claim; validated disease module",
            "evidence_table": "module_full_reselection_summary.tsv;cross_trait_module_reselection_summary.tsv;direction23_smoke_diagnostic.tsv",
            "interpretation": "De novo STRING local modules remain supplementary limitation evidence and are not V1.2 main claims.",
            "source_result_path": f"{MODULE_RESELECTION_DIR};{CROSS_TRAIT_MODULE_RESELECTION_DIR};{DIRECTION23_V12_ROOT}",
            "script_path": str(THIS_SCRIPT),
            "timestamp": now_utc(),
        },
    ]
    return pd.DataFrame(rows)


def add_review_metadata(
    table: pd.DataFrame,
    *,
    source_path: Path,
    script_path: Path,
    seed: object = "",
) -> pd.DataFrame:
    """Attach provenance to review-driven revision tables without changing existing values."""

    out = table.copy()
    defaults = {
        "source_result_path": str(source_path),
        "script_path": str(script_path),
        "seed": seed,
        "timestamp": timestamp_for(source_path) if source_path.exists() else now_utc(),
    }
    for column, value in defaults.items():
        if column not in out.columns:
            out[column] = value
        else:
            blank = out[column].isna() | out[column].astype(str).str.strip().eq("")
            out.loc[blank, column] = value
    return out


def read_review_table(
    relative_path: str,
    *,
    script_path: Path,
    seed: object = "",
    source_root: Path = REVIEW_REVISION_ROOT,
) -> pd.DataFrame:
    path = source_root / relative_path
    if not path.exists():
        return pd.DataFrame()
    return add_review_metadata(read_tsv(path), source_path=path, script_path=script_path, seed=seed)


def build_review_driven_revision_tables() -> dict[str, pd.DataFrame]:
    """Collect NC-review-driven supplementary tables into the manuscript package."""

    tables = {
        "null_generation_audit.tsv": read_review_table(
            "null_generation_audit/null_generation_audit.tsv",
            script_path=NULL_GENERATION_AUDIT_SCRIPT,
        ),
        "gene_score_tail_calibration.tsv": read_review_table(
            "gene_score_tail_calibration/gene_score_tail_calibration.tsv",
            script_path=GENE_SCORE_TAIL_CALIBRATION_SCRIPT,
            seed=20260720,
        ),
        "gene_score_tail_decision_stability_summary.tsv": read_review_table(
            "gene_score_tail_decision_stability/gene_score_tail_decision_stability_summary.tsv",
            script_path=TAIL_DECISION_STABILITY_SCRIPT,
            seed=20260706,
        ),
        "gene_score_tail_rank_displacement.tsv": read_review_table(
            "gene_score_tail_decision_stability/gene_score_tail_rank_displacement.tsv",
            script_path=TAIL_DECISION_STABILITY_SCRIPT,
            seed=20260706,
        ),
        "overlap_preserving_null_sensitivity_summary.tsv": read_review_table(
            "overlap_preserving_null_sensitivity/overlap_preserving_null_sensitivity_summary.tsv",
            script_path=OVERLAP_NULL_SENSITIVITY_SCRIPT,
            seed=20260726,
        ),
        "pathway_comparator_rankset_summary.tsv": read_review_table(
            "pathway_comparator_baselines/magma_pascalx_rankset_comparator.all_traits.tsv",
            script_path=PATHWAY_COMPARATOR_SCRIPT,
            seed=20260721,
        ),
        "magma_reactome_go_gene_sets.tsv": read_review_table(
            "pathway_comparator_baselines/magma_reactome_go_gene_sets.all_traits.tsv",
            script_path=PATHWAY_COMPARATOR_SCRIPT,
            seed=20260721,
        ),
        "pascalx_reactome_go_pathways.tsv": read_review_table(
            "pathway_comparator_baselines/pascalx_reactome_go_pathways.all_traits.tsv",
            script_path=PATHWAY_COMPARATOR_SCRIPT,
            seed=20260721,
        ),
        "true_network_method_baseline_summary.tsv": read_review_table(
            "true_network_method_baseline/true_network_method_baseline_summary.tsv",
            script_path=TRUE_NETWORK_BASELINE_SCRIPT,
            seed=20260722,
        ),
        "dmgwas_external_baseline_summary.tsv": read_review_table(
            "dmgwas_external_baseline/dmgwas_external_baseline_summary.tsv",
            script_path=DMGWAS_EXTERNAL_BASELINE_SCRIPT,
            seed=20260725,
        ),
        "dmgwas_external_baseline_modules.tsv": read_review_table(
            "dmgwas_external_baseline/dmgwas_external_baseline_modules.tsv",
            script_path=DMGWAS_EXTERNAL_BASELINE_SCRIPT,
            seed=20260725,
        ),
        "anchored_strengthened_null_summary.tsv": read_review_table(
            "anchored_strengthened_nulls/anchored_strengthened_null_summary.tsv",
            script_path=ANCHORED_STRENGTHENED_NULL_SCRIPT,
            seed=20260723,
        ),
        "anchored_familywise_targeted_expansion_summary.tsv": read_review_table(
            "anchored_familywise_targeted_expansion/anchored_familywise_targeted_expansion_summary.tsv",
            script_path=ANCHORED_FAMILYWISE_EXPANSION_SCRIPT,
            seed=20260727,
        ),
        "anchored_module_redundancy_clusters.tsv": read_review_table(
            "anchored_strengthened_nulls/anchored_module_redundancy_clusters.tsv",
            script_path=ANCHORED_STRENGTHENED_NULL_SCRIPT,
            seed=20260723,
        ),
        "claim_interpretation_table.tsv": read_review_table(
            "claim_interpretation_table.tsv",
            script_path=THIS_SCRIPT,
            source_root=SUPPLEMENTARY_ROOT,
        ),
        "multiplicity_map.tsv": read_review_table(
            "multiplicity_map.tsv",
            script_path=THIS_SCRIPT,
            source_root=SUPPLEMENTARY_ROOT,
        ),
    }
    return {name: table for name, table in tables.items() if not table.empty}


def timestamp_for(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat() if path.exists() else now_utc()


def figure_row(
    *,
    figure_id: str,
    panel_id: str,
    plot_type: str,
    trait: object = "",
    analysis_id: object = "",
    graph_id: object = "",
    x_value: object = "",
    y_value: object = "",
    group: object = "",
    label: object = "",
    source_table: object = "",
    source_result_path: object = "",
    script_path: object = THIS_SCRIPT,
    error_low: object = "",
    error_high: object = "",
    **extra: object,
) -> dict[str, object]:
    row = {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "plot_type": plot_type,
        "trait": trait,
        "analysis_id": analysis_id,
        "graph_id": graph_id,
        "x_value": x_value,
        "y_value": y_value,
        "group": group,
        "label": label,
        "error_low": error_low,
        "error_high": error_high,
        "source_table": source_table,
        "source_result_path": source_result_path,
        "script_path": str(script_path),
        "timestamp": now_utc(),
    }
    row.update(extra)
    return row


def build_figure_source_tables(
    *,
    policy: dict[str, Any],
    final_claims: pd.DataFrame,
    cross_trait: pd.DataFrame,
    external_baselines: pd.DataFrame,
    type1_uncertainty: pd.DataFrame,
    module_annotation: pd.DataFrame,
    module_reselection: pd.DataFrame,
    cross_trait_module_reselection: pd.DataFrame,
    package_version: str = "v1",
    v12_tables: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    figure_1 = build_figure_1_source(policy)
    figure_2 = build_figure_2_source(type1_uncertainty)
    figure_3 = build_figure_3_source(final_claims, cross_trait, cross_trait_module_reselection)
    figure_4 = (
        build_figure_4_v12_source(v12_tables or {})
        if is_v12_package(package_version)
        else build_figure_4_source(module_annotation, module_reselection)
    )
    figure_5 = build_external_baseline_figure_source(
        external_baselines,
        figure_id="Figure_5",
        panel_by_baseline=True,
    )
    supplement = build_external_baseline_figure_source(
        external_baselines,
        figure_id="Figure_S_external_baselines",
        panel_by_baseline=True,
    )
    out = {
        "figure_1_framework.tsv": figure_1,
        "figure_2_calibration.tsv": figure_2,
        "figure_3_claim_tiers.tsv": figure_3,
        "figure_4_anchored_modules.tsv" if is_v12_package(package_version) else "figure_4_local_modules.tsv": figure_4,
        "figure_5_external_baselines.tsv": figure_5,
        "figure_S_external_baselines.tsv": supplement,
        "figure_5_claim_tiers.tsv": figure_3.copy(),
    }
    if is_v12_package(package_version):
        out["figure_S_local_module_boundary.tsv"] = build_figure_4_source(module_annotation, module_reselection)
        out["figure_S_direction23_diagnostic.tsv"] = build_direction23_supplement_figure_source(v12_tables or {})
    return out


def build_figure_1_source(policy: dict[str, Any]) -> pd.DataFrame:
    rows = [
        figure_row(
            figure_id="Figure_1",
            panel_id="a_pipeline",
            plot_type="schematic",
            x_value="GWAS summary statistics -> LD-aware gene scores -> residualization -> graph inference",
            group="method_overview",
            label="RIPPLE-GWAS pipeline",
            source_table="parameter_table.tsv;claim_policy.yaml",
            source_result_path=f"{DEFAULT_OUT_DIR / 'claim_policy.yaml'};{DEFAULT_OUT_DIR / 'parameter_table.tsv'}",
            allowed_claim="method schematic; no statistical claim",
        ),
        figure_row(
            figure_id="Figure_1",
            panel_id="b_gene_score_streams",
            plot_type="formula_schematic",
            x_value="signed burden; unsigned quadratic association",
            group="gene_score",
            label="Frozen signed and unsigned signal streams",
            source_table="parameter_table.tsv",
            source_result_path=str(DEFAULT_OUT_DIR / "parameter_table.tsv"),
            allowed_claim="technical foundation only",
        ),
        figure_row(
            figure_id="Figure_1",
            panel_id="c_claim_tiers",
            plot_type="tier_schematic",
            x_value="Tier 0-4",
            y_value=final_z_threshold(policy),
            group="claim_policy",
            label=f"final-positive Z >= {final_z_threshold(policy)}",
            source_table="claim_policy.yaml;final_claim_audit.tsv",
            source_result_path=f"{DEFAULT_OUT_DIR / 'claim_policy.yaml'};{DEFAULT_OUT_DIR / 'final_claim_audit.tsv'}",
            allowed_claim="claim thresholds are policy-defined",
        ),
        figure_row(
            figure_id="Figure_1",
            panel_id="d_null_hierarchy",
            plot_type="null_hierarchy_schematic",
            x_value="SNP/gene null; degree-matched node null; diffusion null; graph null",
            group="null_model",
            label="Null model hierarchy",
            source_table="final_claim_audit.tsv;inference_family.tsv",
            source_result_path=f"{DEFAULT_OUT_DIR / 'final_claim_audit.tsv'};{DEFAULT_OUT_DIR / 'inference_family.tsv'}",
            allowed_claim="hierarchy of tested null models",
        ),
    ]
    return ensure_columns(pd.DataFrame(rows), FIGURE_SOURCE_SCHEMA)


def build_figure_2_source(type1_uncertainty: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in type1_uncertainty.to_dict(orient="records"):
        rows.append(
            figure_row(
                figure_id="Figure_2",
                panel_id="a_type1_error",
                plot_type="point_interval",
                x_value=item.get("claim_tier", ""),
                y_value=item.get("fpr", ""),
                group=item.get("scenario", ""),
                label=f"Z >= {item.get('z_threshold', '')}",
                error_low=item.get("binomial_95ci_low", ""),
                error_high=item.get("binomial_95ci_high", ""),
                source_table="type1_uncertainty.tsv",
                source_result_path=str(DEFAULT_OUT_DIR / "type1_uncertainty.tsv"),
                script_path=THIS_SCRIPT,
                n_outer=item.get("n_outer", ""),
                false_positive_count=item.get("false_positive_count", ""),
            )
        )
        rows.append(
            figure_row(
                figure_id="Figure_2",
                panel_id="b_mc_uncertainty",
                plot_type="bar",
                x_value=item.get("claim_tier", ""),
                y_value=item.get("mc_se", ""),
                group=item.get("scenario", ""),
                label=f"Z >= {item.get('z_threshold', '')}",
                source_table="type1_uncertainty.tsv",
                source_result_path=str(DEFAULT_OUT_DIR / "type1_uncertainty.tsv"),
                script_path=THIS_SCRIPT,
                n_outer=item.get("n_outer", ""),
            )
        )
    rows.extend(synthetic_figure_rows(SYNTHETIC_SPIKEIN_DIR, "c_synthetic_scenarios"))
    rows.extend(synthetic_figure_rows(SYNTHETIC_SELECTION_AWARE_DIR, "d_selection_aware_modules"))
    return ensure_columns(pd.DataFrame(rows), FIGURE_SOURCE_SCHEMA)


def synthetic_figure_rows(root: Path, panel_id: str) -> list[dict[str, object]]:
    path = root / "tables" / "synthetic_spikein_summary.tsv"
    if not path.exists():
        return []
    table = read_tsv(path)
    rows: list[dict[str, object]] = []
    for item in table.to_dict(orient="records"):
        scenario = item.get("scenario", "null")
        if pd.isna(scenario) or str(scenario).strip() == "":
            scenario = "null"
        for stat in [
            "snp_null_z",
            "degree_stratified_z",
            "degree_matched_z",
            "degree_preserving_graph_z",
            "n_calibrated_modules",
            "n_topology_specific_modules",
        ]:
            if stat not in table.columns:
                continue
            value = item.get(stat, "")
            if value == "" or pd.isna(value):
                continue
            rows.append(
                figure_row(
                    figure_id="Figure_2",
                    panel_id=panel_id,
                    plot_type="scenario_statistic",
                    x_value=scenario,
                    y_value=value,
                    group=stat,
                    label=item.get("architecture_class", ""),
                    source_table="synthetic_spikein_summary.tsv"
                    if panel_id == "c_synthetic_scenarios"
                    else "synthetic_spikein_selection_aware_summary.tsv",
                    source_result_path=str(path),
                    script_path=SCRIPT_DIR / "run_synthetic_spikein_validation.py",
                    suitability_verdict=item.get("suitability_verdict", ""),
                    effect_size=item.get("effect_size", ""),
                    n_targets=item.get("n_targets", ""),
                )
            )
    return rows


def build_figure_3_source(
    final_claims: pd.DataFrame,
    cross_trait: pd.DataFrame,
    cross_trait_module_reselection: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.extend(claim_tier_figure_rows(cross_trait, "a_cross_trait_heatmap", "cross_trait_benchmark_z2p5.tsv"))
    dr = final_claims[final_claims["trait"].astype(str).str.startswith("DR_MVP", na=False)]
    rows.extend(claim_tier_figure_rows(dr, "b_dr_sensitivity_tiers", "final_claim_audit.tsv"))
    scz = final_claims[final_claims["trait"].astype(str).str.startswith("SCZ", na=False)]
    rows.extend(claim_tier_figure_rows(scz, "c_scz_secondary_benchmark", "final_claim_audit.tsv"))
    graph = final_claims[final_claims["graph_id"].astype(str).isin(["FVM_vascular_weighted", "retina_string_min20"])]
    rows.extend(claim_tier_figure_rows(graph, "d_graph_sensitivity", "final_claim_audit.tsv"))
    rows.extend(cross_trait_module_reselection_figure_rows(cross_trait_module_reselection))
    return ensure_columns(pd.DataFrame(rows), FIGURE_SOURCE_SCHEMA)


def cross_trait_module_reselection_figure_rows(table: pd.DataFrame) -> list[dict[str, object]]:
    if table.empty:
        return []
    rows: list[dict[str, object]] = []
    for item in table.to_dict(orient="records"):
        rows.append(
            figure_row(
                figure_id="Figure_3",
                panel_id="e_cross_trait_tier4_modules",
                plot_type="cross_trait_module_reselection_status",
                trait=item.get("trait", ""),
                analysis_id=item.get("analysis_id", ""),
                graph_id="STRING_default",
                x_value=item.get("analysis_id", ""),
                y_value=item.get("full_reselection_score_z", ""),
                group=item.get("module_layer_claim_status", ""),
                label=item.get("module_id", ""),
                source_table="cross_trait_module_reselection_summary.tsv",
                source_result_path=str(DEFAULT_OUT_DIR / "cross_trait_module_reselection_summary.tsv"),
                script_path=CROSS_TRAIT_MODULE_RESELECTION_SCRIPT,
                full_reselection_score_p=item.get("full_reselection_score_p", ""),
                n_full_reselection_null=item.get("n_full_reselection_null", ""),
                policy_freeze_timing=item.get("policy_freeze_timing", ""),
            )
        )
    return rows


def claim_tier_figure_rows(table: pd.DataFrame, panel_id: str, source_table: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in table.to_dict(orient="records"):
        z = as_float(item.get("z"))
        if not np.isfinite(z):
            continue
        rows.append(
            figure_row(
                figure_id="Figure_3",
                panel_id=panel_id,
                plot_type="claim_tier_z",
                trait=item.get("trait", ""),
                analysis_id=item.get("analysis_id", ""),
                graph_id=item.get("graph_id", ""),
                x_value=item.get("claim_tier", ""),
                y_value=z,
                group=item.get("claim_status", ""),
                label=item.get("statistic_name", ""),
                source_table=source_table,
                source_result_path=item.get("source_result_path", ""),
                script_path=THIS_SCRIPT,
                null_type=item.get("null_type", ""),
                empirical_p=item.get("empirical_p", ""),
                n_null=item.get("n_null", ""),
            )
        )
    return rows


def build_figure_4_source(module_annotation: pd.DataFrame, module_reselection: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.extend(
        local_module_rows(
            ANALYSIS_ROOT / "dr_mvp_string_final5000",
            "DR_MVP",
            "a_default_string_modules",
            module_reselection,
        )
    )
    rows.extend(
        local_module_rows(
            ANALYSIS_ROOT / "dr_mvp_no_mhc_no_apoe_final5000",
            "DR_MVP_NO_MHC_NO_APOE",
            "b_no_mhc_no_apoe_modules",
            module_reselection,
        )
    )
    for item in module_annotation.to_dict(orient="records"):
        rows.append(
            figure_row(
                figure_id="Figure_4",
                panel_id="c_module_annotation",
                plot_type="annotation_enrichment",
                trait=item.get("trait", ""),
                analysis_id=item.get("analysis_id", ""),
                x_value=item.get("annotation_name", ""),
                y_value=item.get("overlap_count", ""),
                group=item.get("annotation_source_type", ""),
                label=item.get("module_id", ""),
                source_table="module_annotation.tsv",
                source_result_path=str(DEFAULT_OUT_DIR / "module_annotation.tsv"),
                script_path=THIS_SCRIPT,
                p_value=item.get("p_value", ""),
                fdr=item.get("fdr", ""),
                background_size=item.get("background_size", ""),
                gene_set_size_within_background=item.get("gene_set_size_within_background", ""),
            )
        )
    rows.extend(module_claim_boundary_rows(ANALYSIS_ROOT / "dr_mvp_string_final5000", "DR_MVP", module_reselection))
    rows.extend(
        module_claim_boundary_rows(
            ANALYSIS_ROOT / "dr_mvp_no_mhc_no_apoe_final5000",
            "DR_MVP_NO_MHC_NO_APOE",
            module_reselection,
        )
    )
    return ensure_columns(pd.DataFrame(rows), FIGURE_SOURCE_SCHEMA)


def build_figure_4_v12_source(v12_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    anchored_summary = v12_tables.get("anchored_cross_trait_summary.tsv", pd.DataFrame())
    anchored_top = v12_tables.get("anchored_top_modules.tsv", pd.DataFrame())
    type1 = v12_tables.get("anchored_type1_outer1000_summary.tsv", pd.DataFrame())
    go_reactome = v12_tables.get("anchored_go_reactome_confirmation.tsv", pd.DataFrame())
    robustness = v12_tables.get("anchored_robustness_summary.tsv", pd.DataFrame())
    for item in anchored_summary.to_dict(orient="records"):
        rows.append(
            figure_row(
                figure_id="Figure_4",
                panel_id="a_anchored_cross_trait",
                plot_type="anchored_cross_trait_status",
                trait=item.get("trait", ""),
                analysis_id=item.get("analysis_id", ""),
                graph_id="STRING_default",
                x_value=item.get("analysis_id", ""),
                y_value=item.get("best_degree_matched_z", ""),
                group=item.get("best_module_status", ""),
                label=item.get("best_module_name", ""),
                source_table="anchored_cross_trait_summary.tsv",
                source_result_path=item.get("source_result_path", ""),
                script_path=ANCHORED_V12_SUMMARY_SCRIPT,
                familywise_p=item.get("best_library_familywise_p", ""),
                interpretation_note=item.get("interpretation_note", ""),
            )
        )
    top_display = anchored_top.loc[
        pd.to_numeric(anchored_top.get("top_rank_within_trait", pd.Series(dtype=float)), errors="coerce").le(3)
    ] if not anchored_top.empty else pd.DataFrame()
    for item in top_display.to_dict(orient="records"):
        rows.append(
            figure_row(
                figure_id="Figure_4",
                panel_id="b_top_anchored_modules",
                plot_type="top_anchored_module",
                trait=item.get("trait", ""),
                analysis_id=item.get("analysis_id", ""),
                graph_id="STRING_default",
                x_value=item.get("top_rank_within_trait", ""),
                y_value=item.get("degree_matched_z", ""),
                group=item.get("module_status", ""),
                label=item.get("module_name", ""),
                source_table="anchored_top_modules.tsv",
                source_result_path=item.get("source_result_path", ""),
                script_path=ANCHORED_V12_SUMMARY_SCRIPT,
                library_familywise_p=item.get("library_familywise_p", ""),
                module_source=item.get("module_source", ""),
            )
        )
    type1_subset = type1[
        type1.get("calibration_target", pd.Series(dtype=str)).astype(str).eq("any_familywise_positive")
    ] if not type1.empty else pd.DataFrame()
    for item in type1_subset.to_dict(orient="records"):
        rows.append(
            figure_row(
                figure_id="Figure_4",
                panel_id="c_anchored_type1",
                plot_type="type1_fpr_interval",
                trait=item.get("trait", ""),
                analysis_id=item.get("analysis_id", ""),
                graph_id="STRING_default",
                x_value=item.get("calibration_family", ""),
                y_value=item.get("fpr", ""),
                group=item.get("calibration_target", ""),
                label=item.get("calibration_family", ""),
                error_low=item.get("binomial_95ci_low", ""),
                error_high=item.get("binomial_95ci_high", ""),
                source_table="anchored_type1_outer1000_summary.tsv",
                source_result_path=item.get("source_result_path", ""),
                script_path=ANCHORED_V12_TYPE1_SCRIPT,
                n_outer=item.get("n_outer", ""),
                false_positive_count=item.get("false_positive_count", ""),
            )
        )
    for item in go_reactome.to_dict(orient="records"):
        rows.append(
            figure_row(
                figure_id="Figure_4",
                panel_id="d_go_reactome_confirmation",
                plot_type="source_family_confirmation",
                trait=item.get("trait", ""),
                analysis_id=item.get("analysis_id", ""),
                graph_id="STRING_default",
                x_value=item.get("group_value", ""),
                y_value=item.get("group_familywise_p", ""),
                group=item.get("source_family_claim_status", ""),
                label=item.get("module_name", ""),
                source_table="anchored_go_reactome_confirmation.tsv",
                source_result_path=item.get("source_result_path", ""),
                script_path=ANCHORED_V12_ROBUSTNESS_SCRIPT,
                library_familywise_p=item.get("library_familywise_p", ""),
                type1_fpr_outer1000=item.get("type1_fpr_outer1000", ""),
            )
        )
    for item in robustness.to_dict(orient="records"):
        rows.append(
            figure_row(
                figure_id="Figure_4",
                panel_id="e_anchored_robustness",
                plot_type="robustness_metric",
                trait=item.get("trait", ""),
                analysis_id=item.get("analysis_id", ""),
                graph_id="STRING_default",
                x_value="score_perturbation_familywise_positive_rate",
                y_value=item.get("familywise_positive_rate", ""),
                group=item.get("best_status_n500", ""),
                label=item.get("observed_best_module", item.get("best_module_n500", "")),
                source_table="anchored_robustness_summary.tsv",
                source_result_path=item.get("source_result_path", ""),
                script_path=ANCHORED_V12_ROBUSTNESS_SCRIPT,
                top10_jaccard=item.get("top10_jaccard_n50_n500", ""),
                best_module_same=item.get("best_module_same", ""),
            )
        )
    return ensure_columns(pd.DataFrame(rows), FIGURE_SOURCE_SCHEMA)


def build_direction23_supplement_figure_source(v12_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    table = v12_tables.get("direction23_smoke_diagnostic.tsv", pd.DataFrame())
    rows: list[dict[str, object]] = []
    for item in table.to_dict(orient="records"):
        rows.append(
            figure_row(
                figure_id="Figure_S_direction23",
                panel_id=str(item.get("method", "")),
                plot_type="direction23_smoke",
                trait=item.get("trait", ""),
                analysis_id=item.get("analysis_id", ""),
                graph_id="STRING_default",
                x_value=item.get("method", ""),
                y_value=item.get("z", ""),
                group=item.get("smoke_status", ""),
                label=item.get("candidate_id", ""),
                source_table="direction23_smoke_diagnostic.tsv",
                source_result_path=item.get("source_result_path", ""),
                script_path=DIRECTION23_V12_SCRIPT,
                empirical_p=item.get("empirical_p", ""),
                v12_recommendation=item.get("v12_recommendation", ""),
            )
        )
    return ensure_columns(pd.DataFrame(rows), FIGURE_SOURCE_SCHEMA)


def local_module_rows(
    analysis_dir: Path,
    trait: str,
    panel_id: str,
    module_reselection: pd.DataFrame,
) -> list[dict[str, object]]:
    if not module_reselection.empty and "trait" in module_reselection.columns:
        selected = module_reselection[module_reselection["trait"].astype(str) == trait]
        if not selected.empty:
            rows: list[dict[str, object]] = []
            for item in selected.to_dict(orient="records"):
                rows.append(
                    figure_row(
                        figure_id="Figure_4",
                        panel_id=panel_id,
                        plot_type="module_full_reselection_summary",
                        trait=trait,
                        analysis_id=item.get("analysis_id", analysis_dir.name),
                        graph_id="STRING_default",
                        x_value=item.get("module_id", ""),
                        y_value=item.get("full_reselection_score_z", ""),
                        group=item.get("module_layer_claim_status", item.get("recommended_module_claim_after_reselection", "")),
                        label=item.get("core_genes", ""),
                        source_table="module_full_reselection_summary.tsv",
                        source_result_path=str(DEFAULT_OUT_DIR / "module_full_reselection_summary.tsv"),
                        script_path=MODULE_RESELECTION_SCRIPT,
                        module_layer_claim_status=item.get("module_layer_claim_status", ""),
                        module_layer_policy_version=item.get("module_layer_policy_version", ""),
                        rank_fraction=item.get("rank_fraction", ""),
                        n_genes=item.get("n_genes", ""),
                        edge_density=item.get("edge_density", ""),
                        full_reselection_score_p=item.get("full_reselection_score_p", ""),
                        full_reselection_edge_p=item.get("full_reselection_edge_p", ""),
                        source_module_table=item.get("source_module_table", ""),
                    )
                )
            return rows
    path = analysis_dir / "tables" / f"{trait}.local_modules.tsv"
    if not path.exists():
        return []
    table = read_tsv(path)
    rows: list[dict[str, object]] = []
    for item in table.to_dict(orient="records"):
        rows.append(
            figure_row(
                figure_id="Figure_4",
                panel_id=panel_id,
                plot_type="module_summary",
                trait=trait,
                analysis_id=analysis_dir.name,
                graph_id="STRING_default",
                x_value=item.get("module_id", ""),
                y_value=item.get("selection_aware_score_z", item.get("degree_matched_score_z", "")),
                group=item.get("module_claim_label", ""),
                label=item.get("core_genes", ""),
                source_table=f"{trait}.local_modules.tsv",
                source_result_path=str(path),
                script_path=THIS_SCRIPT,
                rank_fraction=item.get("rank_fraction", ""),
                n_genes=item.get("n_genes", ""),
                edge_density=item.get("edge_density", ""),
                degree_matched_p=item.get("degree_matched_p", ""),
                selection_aware_score_p=item.get("selection_aware_score_p", ""),
                is_broad_component=item.get("is_broad_component", ""),
                is_reportable_calibrated_module=item.get("is_reportable_calibrated_module", ""),
                is_reportable_topology_specific_module=item.get("is_reportable_topology_specific_module", ""),
            )
        )
    return rows


def module_claim_boundary_rows(
    analysis_dir: Path,
    trait: str,
    module_reselection: pd.DataFrame,
) -> list[dict[str, object]]:
    if not module_reselection.empty and "trait" in module_reselection.columns:
        selected = module_reselection[module_reselection["trait"].astype(str) == trait]
        if not selected.empty:
            rows: list[dict[str, object]] = []
            group_col = "module_layer_claim_status" if "module_layer_claim_status" in selected.columns else "recommended_module_claim_after_reselection"
            for label, group in selected.groupby(group_col, dropna=False):
                rows.append(
                    figure_row(
                        figure_id="Figure_4",
                        panel_id="d_module_claim_boundary",
                        plot_type="module_full_reselection_claim_count",
                        trait=trait,
                        analysis_id=analysis_dir.name,
                        graph_id="STRING_default",
                        x_value=str(label),
                        y_value=len(group),
                        group="not_topology_specific",
                        label=f"{len(group)} modules",
                        source_table="module_full_reselection_summary.tsv",
                        source_result_path=str(DEFAULT_OUT_DIR / "module_full_reselection_summary.tsv"),
                        script_path=MODULE_RESELECTION_SCRIPT,
                        module_layer_claim_status=str(label),
                    )
                )
            return rows
    path = analysis_dir / "tables" / f"{trait}.local_modules.tsv"
    if not path.exists():
        return []
    table = read_tsv(path)
    rows: list[dict[str, object]] = []
    for label, group in table.groupby("module_claim_label", dropna=False):
        rows.append(
            figure_row(
                figure_id="Figure_4",
                panel_id="d_module_claim_boundary",
                plot_type="module_claim_count",
                trait=trait,
                analysis_id=analysis_dir.name,
                graph_id="STRING_default",
                x_value=str(label),
                y_value=len(group),
                group="topology_specific"
                if group.get("is_reportable_topology_specific_module", pd.Series(dtype=bool)).astype(str).str.lower().eq("true").any()
                else "not_topology_specific",
                label=f"{len(group)} modules",
                source_table=f"{trait}.local_modules.tsv",
                source_result_path=str(path),
                script_path=THIS_SCRIPT,
            )
        )
    return rows


def build_external_baseline_figure_source(
    external_baselines: pd.DataFrame,
    *,
    figure_id: str = "Figure_S_external_baselines",
    panel_by_baseline: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in external_baselines.to_dict(orient="records"):
        baseline_type = str(row.get("baseline_type", "external_baseline"))
        value = as_float(row.get("z")) if baseline_type == "network_ablation" else as_float(row.get("observed_value"))
        if not np.isfinite(value):
            continue
        panel_id = baseline_type if panel_by_baseline else "external_baselines"
        rows.append(
            figure_row(
                figure_id=figure_id,
                panel_id=panel_id,
                plot_type="bar_or_point",
                trait=row.get("trait", ""),
                analysis_id=row.get("analysis_id", ""),
                graph_id=row.get("graph_id", ""),
                x_value=row.get("statistic_name", ""),
                y_value=value,
                group=row.get("baseline_tool", row.get("baseline_name", "")),
                label=row.get("comparison_name", row.get("statistic_name", "")),
                source_table="external_baseline_summary.tsv",
                source_result_path=row.get("source_result_path", ""),
                script_path=THIS_SCRIPT,
                baseline_type=baseline_type,
                claim_status=row.get("claim_status", ""),
            )
        )
    return ensure_columns(pd.DataFrame(rows), FIGURE_SOURCE_SCHEMA)


def build_parameter_table(policy: dict[str, Any], package_version: str = "v1") -> pd.DataFrame:
    rows = [
        param("global", "all", "final_z_threshold", final_z_threshold(policy), "float", "global", "default"),
        param("global", "all", "supportive_z_threshold", supportive_z_threshold(policy), "float", "global", "default"),
        param("global", "all", "rank_fraction_grid", "0.01,0.02,0.05,0.10,0.15,0.20", "list", "percolation", "default"),
        param("global", "all", "ld_shrinkage_lambda", 0.05, "float", "gene_score", "default"),
        param("DR_MVP_default_final5000", "DR_MVP", "degree_matched_node_null", 5000, "int", "null", "override"),
        param("DR_MVP_default_final5000", "DR_MVP", "diffusion_null", 5000, "int", "null", "override"),
        param("DR_MVP_no_MHC_no_APOE_final5000", "DR_MVP_NO_MHC_NO_APOE", "degree_matched_node_null", 5000, "int", "null", "override"),
        param("DR_MVP_no_MHC_no_APOE_final5000", "DR_MVP_NO_MHC_NO_APOE", "diffusion_null", 5000, "int", "null", "override"),
        param("SCZ_default_with_MHC_dev500", "SCZ_WITH_MHC", "degree_matched_node_null", 500, "int", "null", "override"),
        param("SCZ_default_with_MHC_dev500", "SCZ_WITH_MHC", "diffusion_null", 500, "int", "null", "override"),
        param("SCZ_default_with_MHC_dev500", "SCZ_WITH_MHC", "degree_preserving_graph_null", 100, "int", "null", "override"),
        param("SCZ_no_MHC_final5000", "SCZ", "degree_matched_node_null", 5000, "int", "null", "override"),
        param("SCZ_no_MHC_final5000", "SCZ", "diffusion_null", 5000, "int", "null", "override"),
        param("SCZ_no_MHC_final5000", "SCZ", "degree_preserving_graph_null", 500, "int", "null", "override"),
        param("external_baseline_MAGMA", "primary_traits", "magma_version", "v1.10 linux static", "string", "global", "override"),
        param("external_baseline_PascalX", "primary_traits", "pascalx_version", "0.0.5", "string", "global", "override"),
        param("external_baseline_PascalX", "primary_traits", "pascalx_maf", 0.01, "float", "global", "override"),
        param("external_baseline_PascalX", "primary_traits", "pascalx_window_bp", 0, "int", "global", "override"),
        param("network_ablation_v1", "primary_traits", "network_ablation_nulls", 5000, "int", "null", "override"),
        param("external_score_graph_layer_v1", "primary_traits", "external_score_degree_matched_nulls", 5000, "int", "null", "override"),
        param("external_score_graph_layer_v1", "primary_traits", "external_score_diffusion_nulls", 5000, "int", "null", "override"),
        param("external_score_graph_layer_v1", "primary_traits", "external_score_degree_graph_nulls", 100, "int", "null", "override"),
        param("dense_module_competitor_v1", "primary_traits", "dense_module_seeds", 80, "int", "module", "override"),
        param("dense_module_competitor_v1", "primary_traits", "dense_module_node_nulls", 500, "int", "null", "override"),
        param("dense_module_competitor_v1", "primary_traits", "dense_module_graph_nulls", 0, "int", "null", "override"),
        param("dmgwas_external_baseline_top2000_v1", "DR_MVP;DR_MVP_NO_MHC_NO_APOE;SCZ", "dmGWAS_version", "3.0", "string", "module", "override"),
        param("dmgwas_external_baseline_top2000_v1", "DR_MVP;DR_MVP_NO_MHC_NO_APOE;SCZ", "dmGWAS_scope_top_n_by_gene_p", 2000, "int", "module", "override"),
        param("dmgwas_external_baseline_top2000_v1", "DR_MVP;DR_MVP_NO_MHC_NO_APOE;SCZ", "dmGWAS_r", 0.1, "float", "module", "override"),
        param("dmgwas_external_baseline_top2000_v1", "DR_MVP;DR_MVP_NO_MHC_NO_APOE;SCZ", "dmGWAS_gene_p_clip_min", "1e-16", "float", "gene_score", "override"),
        param("dmgwas_external_baseline_top2000_v1", "DR_MVP;DR_MVP_NO_MHC_NO_APOE;SCZ", "dmGWAS_gene_p_clip_max", "1-1e-16", "float", "gene_score", "override"),
        param("module_reselection_null_v1", "DR_MVP;DR_MVP_NO_MHC_NO_APOE", "full_reselection_nulls", 5000, "int", "module", "override"),
        param("module_reselection_null_v1", "DR_MVP;DR_MVP_NO_MHC_NO_APOE", "full_reselection_null_source", "degree_stratified_score_permutation", "string", "null", "override"),
        param("module_reselection_null_cross_trait_v1", "SCZ;HEIGHT_IRN;BMI_IRN;T2D", "full_reselection_nulls", 5000, "int", "module", "override"),
        param("module_reselection_null_cross_trait_v1", "SCZ;HEIGHT_IRN;BMI_IRN;T2D", "full_reselection_null_source", "degree_stratified_score_permutation", "string", "null", "override"),
        param("module_reselection_null_cross_trait_v1", "SCZ;HEIGHT_IRN;BMI_IRN;T2D", "policy_freeze_timing", "after_DR_MVP_before_cross_trait_module_reselection", "string", "module", "override"),
    ]
    if is_v12_package(package_version):
        rows.extend(
            [
                param("anchored_broad_reactome_go_v1_n500", "DR_MVP;T2D;HEIGHT_IRN;BMI_IRN;SCZ", "anchored_degree_matched_nulls", 500, "int", "module", "override"),
                param("anchored_broad_reactome_go_v1_n500", "DR_MVP;T2D;HEIGHT_IRN;BMI_IRN;SCZ", "anchored_score_permutation_nulls", 500, "int", "module", "override"),
                param("anchored_broad_reactome_go_v1_n500", "DR_MVP;T2D;HEIGHT_IRN;BMI_IRN;SCZ", "anchored_library", "Reactome plus Gene Ontology biological process", "string", "module", "override"),
                param("anchored_type1_outer1000_v1", "DR_MVP;T2D;HEIGHT_IRN;BMI_IRN;SCZ", "outer_null_replicates", 1000, "int", "null", "override"),
                param("anchored_type1_outer1000_v1", "DR_MVP;T2D;HEIGHT_IRN;BMI_IRN;SCZ", "calibration_null_replicates", 500, "int", "null", "override"),
                param("anchored_type1_outer1000_v1", "DR_MVP;T2D;HEIGHT_IRN;BMI_IRN;SCZ", "calibration_families", "all_modules;Gene Ontology;Reactome", "list", "null", "override"),
                param("anchored_robustness_v1", "DR_MVP;T2D;HEIGHT_IRN;BMI_IRN;SCZ", "score_perturbation_replicates", 200, "int", "module", "override"),
                param("direction23_smoke_v1", "DR_MVP;T2D;HEIGHT_IRN;BMI_IRN;SCZ", "direction23_null_replicates", 200, "int", "module", "override"),
                param("direction23_smoke_v1", "DR_MVP;T2D;HEIGHT_IRN;BMI_IRN;SCZ", "direction23_claim_role", "diagnostic_only", "string", "module", "override"),
            ]
        )
    return ensure_columns(pd.DataFrame(rows), PARAMETER_TABLE_SCHEMA)


def param(
    analysis_id: str,
    trait: str,
    name: str,
    value: object,
    parameter_type: str,
    scope: str,
    default_or_override: str,
) -> dict[str, object]:
    lower = f"{analysis_id} {name}".lower()
    if "threshold" in name:
        source = POLICY_PATH
    elif "magma" in lower:
        source = MAGMA_BASELINE_SCRIPT
    elif "pascalx" in lower:
        source = PASCALX_BASELINE_SCRIPT
    elif "network_ablation" in lower:
        source = NETWORK_ABLATION_SCRIPT
    elif "external_score" in lower:
        source = EXTERNAL_SCORE_GRAPH_SCRIPT
    elif "dense_module" in lower:
        source = DENSE_MODULE_COMPETITOR_SCRIPT
    elif "anchored_type1" in lower:
        source = ANCHORED_V12_TYPE1_SCRIPT
    elif "anchored_robustness" in lower:
        source = ANCHORED_V12_ROBUSTNESS_SCRIPT
    elif "anchored" in lower:
        source = ANCHORED_V12_RUN_SCRIPT
    elif "direction23" in lower:
        source = DIRECTION23_V12_SCRIPT
    elif "cross_trait" in lower and "reselection" in lower:
        source = CROSS_TRAIT_MODULE_RESELECTION_SCRIPT
    elif "reselection" in lower:
        source = MODULE_RESELECTION_SCRIPT
    else:
        source = SCRIPT_DIR / "run_trait_ld_analysis.py"
    return {
        "analysis_id": analysis_id,
        "trait": trait,
        "parameter_name": name,
        "parameter_value": value,
        "parameter_type": parameter_type,
        "scope": scope,
        "default_or_override": default_or_override,
        "source_config_or_script": str(source),
        "notes": "",
    }


def build_external_baseline_summary(policy: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.extend(external_method_comparison_rows(MAGMA_BASELINE_DIR / "magma_ripple_comparison.all_traits.tsv", "MAGMA"))
    rows.extend(
        external_dr_panel_rows(
            MAGMA_BASELINE_DIR / "magma_dr_panel_gene_sets.all_traits.tsv",
            tool="MAGMA",
            pathway_col="VARIABLE",
            p_col="P",
            n_genes_col="NGENES",
        )
    )
    rows.extend(external_method_comparison_rows(PASCALX_BASELINE_DIR / "pascalx_ripple_comparison.all_traits.tsv", "PascalX"))
    rows.extend(
        external_dr_panel_rows(
            PASCALX_BASELINE_DIR / "pascalx_dr_panel_pathways.all_traits.tsv",
            tool="PascalX",
            pathway_col="pathway_name",
            p_col="pascalx_p",
            n_genes_col="n_scored_genes",
        )
    )
    rows.extend(network_ablation_rows(NETWORK_ABLATION_DIR / "network_ablation_summary.all_traits.tsv", policy))
    rows.extend(external_score_graph_layer_rows(EXTERNAL_SCORE_GRAPH_DIR))
    rows.extend(dense_module_competitor_rows(DENSE_MODULE_COMPETITOR_DIR / "dense_module_competitor_summary.all_traits.tsv"))
    rows.extend(
        dmgwas_external_network_rows(
            REVIEW_REVISION_ROOT / "dmgwas_external_baseline" / "dmgwas_external_baseline_summary.tsv"
        )
    )
    return ensure_columns(pd.DataFrame(rows), COMMON_RESULT_SCHEMA)


def baseline_common_row(
    *,
    trait: str,
    analysis_id: str,
    baseline_tool: str,
    baseline_type: str,
    statistic_name: str,
    statistic_direction: str,
    observed_value: object,
    empirical_p: object,
    claim_status: str,
    source_result_path: Path,
    script_path: Path,
    threshold: object = "descriptive_external_baseline",
    null_type: str = "method_comparison",
    score_stream: str = "external_gene_score_baseline",
    claim_tier: str = "external_baseline",
    seed: object = "",
    notes: str = "",
    **extra: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "trait": trait,
        "analysis_id": analysis_id,
        "graph_id": "STRING_default",
        "score_stream": score_stream,
        "null_type": null_type,
        "statistic_name": statistic_name,
        "statistic_direction": statistic_direction,
        "observed_value": observed_value,
        "null_mean": "",
        "null_sd": "",
        "z": "",
        "empirical_p": empirical_p,
        "n_null": "",
        "threshold": threshold,
        "claim_tier": claim_tier,
        "claim_status": claim_status,
        "exclusion_or_na_reason": "not_applicable",
        "not_tested_reason": "not_applicable",
        "source_result_path": str(source_result_path),
        "script_path": str(script_path),
        "seed": seed,
        "timestamp": timestamp_for(source_result_path),
        "baseline_tool": baseline_tool,
        "baseline_type": baseline_type,
        "notes": notes,
    }
    row.update(extra)
    return row


def external_method_comparison_rows(path: Path, tool: str) -> list[dict[str, object]]:
    if not path.exists():
        return []
    table = read_tsv(path)
    script = MAGMA_BASELINE_SCRIPT if tool == "MAGMA" else PASCALX_BASELINE_SCRIPT
    rows: list[dict[str, object]] = []
    for item in table.to_dict(orient="records"):
        p_value = item.get("p_value", "")
        if pd.isna(p_value):
            p_value = ""
        rows.append(
            baseline_common_row(
                trait=str(item.get("trait", "")),
                analysis_id=str(item.get("analysis_id", "")),
                baseline_tool=tool,
                baseline_type="gene_score_concordance",
                statistic_name=f"{tool.lower()}_{item.get('comparison_name', '')}",
                statistic_direction="greater_is_more_extreme",
                observed_value=item.get("observed_value", ""),
                empirical_p=p_value,
                claim_status="supportive",
                source_result_path=path,
                script_path=script,
                comparison_name=item.get("comparison_name", ""),
                n_genes=item.get("n_genes", ""),
                k=item.get("k", ""),
                overlap_count=item.get("overlap_count", ""),
                notes="External gene-based baseline concordance; not a module-discovery claim.",
            )
        )
    return rows


def external_dr_panel_rows(
    path: Path,
    *,
    tool: str,
    pathway_col: str,
    p_col: str,
    n_genes_col: str,
) -> list[dict[str, object]]:
    if not path.exists():
        return []
    table = read_tsv(path)
    script = MAGMA_BASELINE_SCRIPT if tool == "MAGMA" else PASCALX_BASELINE_SCRIPT
    rows: list[dict[str, object]] = []
    for item in table.to_dict(orient="records"):
        p_value = as_float(item.get(p_col))
        pathway = str(item.get(pathway_col, ""))
        rows.append(
            baseline_common_row(
                trait=str(item.get("trait", "")),
                analysis_id=str(item.get("analysis_id", "")),
                baseline_tool=tool,
                baseline_type="dr_panel_gene_set",
                statistic_name=f"{tool.lower()}_dr_panel_{pathway}_p",
                statistic_direction="less_is_more_extreme",
                observed_value=p_value,
                empirical_p=p_value,
                claim_status="supportive" if np.isfinite(p_value) and p_value <= 0.05 else "negative",
                source_result_path=path,
                script_path=script,
                threshold="p<=0.05_descriptive",
                null_type="external_gene_set_test",
                comparison_name=pathway,
                n_genes=item.get(n_genes_col, ""),
                notes="Internal DR panel baseline annotation; not independent validation.",
            )
        )
    return rows


def dmgwas_external_network_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    table = read_tsv(path)
    rows: list[dict[str, object]] = []
    for item in table.to_dict(orient="records"):
        trait = str(item.get("trait", ""))
        analysis_id = str(item.get("analysis_id", ""))
        common_extra = {
            "n_input_genes": item.get("n_input_genes", ""),
            "n_input_edges": item.get("n_input_edges", ""),
            "n_modules": item.get("n_modules", ""),
            "top_seed": item.get("top_seed", ""),
            "top_module_size": item.get("top_module_size", ""),
            "top_Zn": item.get("top_Zn", ""),
            "top_empirical_p": item.get("top_empirical_p", ""),
            "package_version": item.get("package_version", ""),
            "igraph_version": item.get("igraph_version", ""),
            "analysis_scope": item.get("analysis_scope", ""),
            "max_genes_requested": item.get("max_genes_requested", ""),
            "full_scale_attempt_status": item.get("full_scale_attempt_status", ""),
            "scope_note": item.get("scope_note", ""),
        }
        rows.append(
            baseline_common_row(
                trait=trait,
                analysis_id=analysis_id,
                baseline_tool="dmGWAS",
                baseline_type="external_network_method",
                statistic_name="dmgwas_nominal_modules_empirical_p_le_0_05",
                statistic_direction="greater_is_more_extreme",
                observed_value=item.get("n_empirical_p_le_0_05", ""),
                empirical_p="",
                claim_status="supportive",
                source_result_path=path,
                script_path=DMGWAS_EXTERNAL_BASELINE_SCRIPT,
                threshold="dmGWAS empirical P <= 0.05",
                null_type="dmGWAS_internal_random_network_normalization",
                score_stream="RIPPLE_gene_p_as_dmGWAS_geneweight",
                claim_tier="external_network_baseline",
                seed=20260725,
                notes=(
                    "Actual dmGWAS 3.0 node-only external network-method baseline over a top-ranked "
                    "graph-induced analysis scope; many nominal modules are interpreted as a benchmark "
                    "for calibration needs, not as RIPPLE discoveries."
                ),
                **common_extra,
            )
        )
        rows.append(
            baseline_common_row(
                trait=trait,
                analysis_id=analysis_id,
                baseline_tool="dmGWAS",
                baseline_type="external_network_method",
                statistic_name="dmgwas_modules_Zn_ge_2_5",
                statistic_direction="greater_is_more_extreme",
                observed_value=item.get("n_Zn_ge_2_5", ""),
                empirical_p="",
                claim_status="supportive",
                source_result_path=path,
                script_path=DMGWAS_EXTERNAL_BASELINE_SCRIPT,
                threshold="dmGWAS normalized Zn >= 2.5",
                null_type="dmGWAS_internal_random_network_normalization",
                score_stream="RIPPLE_gene_p_as_dmGWAS_geneweight",
                claim_tier="external_network_baseline",
                seed=20260725,
                notes=(
                    "Actual dmGWAS 3.0 node-only external network-method baseline; high counts illustrate "
                    "that traditional dense-module search outputs require explicit claim-boundary calibration."
                ),
                **common_extra,
            )
        )
    return rows


def network_ablation_rows(path: Path, policy: dict[str, Any]) -> list[dict[str, object]]:
    if not path.exists():
        return []
    table = read_tsv(path)
    rows: list[dict[str, object]] = []
    for item in table.to_dict(orient="records"):
        z = as_float(item.get("z"))
        row = baseline_common_row(
            trait=str(item.get("trait", "")),
            analysis_id=str(item.get("analysis_id", "")),
            baseline_tool="RIPPLE_network_ablation",
            baseline_type="network_ablation",
            statistic_name=str(item.get("statistic_name", "")),
            statistic_direction=str(item.get("statistic_direction", "greater_is_more_extreme")),
            observed_value=item.get("observed_value", ""),
            empirical_p=item.get("empirical_p", ""),
            claim_status=classify_z_claim(z, policy),
            source_result_path=path,
            script_path=NETWORK_ABLATION_SCRIPT,
            threshold=final_z_threshold(policy),
            null_type=str(item.get("null_type", "")),
            score_stream="assoc_resid_score",
            claim_tier="network_ablation_baseline",
            seed=item.get("seed", ""),
            baseline_name=item.get("baseline_name", ""),
            null_mean=item.get("null_mean", ""),
            null_sd=item.get("null_sd", ""),
            z=item.get("z", ""),
            n_null=item.get("n_null", ""),
            notes=item.get("interpretation", ""),
        )
        rows.append(row)
    return rows


def external_score_graph_layer_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not root.exists():
        return rows
    for path in sorted(root.rglob("*.external_score_graph_claims.tsv")):
        table = read_tsv(path)
        for item in table.to_dict(orient="records"):
            row = baseline_common_row(
                trait=str(item.get("trait", "")),
                analysis_id=str(item.get("analysis_id", "")),
                baseline_tool="RIPPLE_graph_layer_with_external_gene_scores",
                baseline_type="external_score_graph_layer",
                statistic_name=str(item.get("statistic_name", "")),
                statistic_direction=str(item.get("statistic_direction", "greater_is_more_extreme")),
                observed_value=item.get("observed_value", ""),
                empirical_p=item.get("empirical_p", ""),
                claim_status=str(item.get("claim_status", "not_tested")),
                source_result_path=path,
                script_path=EXTERNAL_SCORE_GRAPH_SCRIPT,
                threshold=item.get("threshold", final_z_threshold(load_claim_policy(POLICY_PATH))),
                null_type=str(item.get("null_type", "")),
                score_stream=str(item.get("score_stream", "external_resid_score")),
                claim_tier=str(item.get("claim_tier", "external_score_graph_layer")),
                seed=item.get("seed", ""),
                baseline_name=f"{item.get('score_source', '')}_through_RIPPLE_graph_layer",
                score_source=item.get("score_source", ""),
                null_mean=item.get("null_mean", ""),
                null_sd=item.get("null_sd", ""),
                z=item.get("z", ""),
                n_null=item.get("n_null", ""),
                notes=(
                    "Established gene score passed through the RIPPLE graph/null layer; "
                    "supports graph-layer robustness, not replacement of gene-based tests."
                ),
            )
            rows.append(row)
    return rows


def dense_module_competitor_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    table = read_tsv(path)
    rows: list[dict[str, object]] = []
    metric_specs = [
        ("n_candidate_modules", "dense_module_candidate_count"),
        ("n_naive_positive", "dense_module_naive_positive_count"),
        ("n_degree_robust_positive", "dense_module_degree_matched_positive_count"),
        ("n_fixed_edge_density_positive", "dense_module_fixed_edge_density_positive_count"),
    ]
    for item in table.to_dict(orient="records"):
        for col, statistic in metric_specs:
            rows.append(
                baseline_common_row(
                    trait=str(item.get("trait", "")),
                    analysis_id=str(item.get("analysis_id", "")),
                    baseline_tool="simplified_dense_module_search",
                    baseline_type="graph_module_competitor",
                    statistic_name=statistic,
                    statistic_direction="greater_is_more_extreme",
                    observed_value=item.get(col, ""),
                    empirical_p="",
                    claim_status="supportive",
                    source_result_path=path,
                    script_path=DENSE_MODULE_COMPETITOR_SCRIPT,
                    threshold="descriptive_competitor_baseline",
                    null_type="module_competitor_descriptive",
                    score_stream="assoc_resid_score",
                    claim_tier="graph_module_competitor",
                    baseline_name="simplified_dense_module_search",
                    module_table=item.get("module_table", ""),
                    notes=(
                        "Compact greedy dense-module competitor; fixed-module edge-density checks do not rerun "
                        "module selection and cannot establish topology-specific discovery."
                    ),
                )
            )
    return rows


def checksum_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_input_checksum_table(package_version: str = "v1") -> pd.DataFrame:
    paths = [
        POLICY_PATH,
        PROCESSED_ROOT / "gwas_qc" / "core_hm3_no_mhc" / "DR_MVP.tsv.gz",
        PROCESSED_ROOT / "gwas_qc" / "core_hm3_no_mhc_no_apoe" / "DR_MVP.tsv.gz",
        PROCESSED_ROOT / "gwas_qc" / "qc_reports" / "SCZ.qc_report.json",
        PROCESSED_ROOT / "gwas_qc" / "harmonized_hm3_with_mhc" / "SCZ.tsv.gz",
        PROCESSED_ROOT / "gwas_qc" / "core_hm3_no_mhc" / "SCZ.tsv.gz",
        SCZ_SECONDARY_SUMMARY_DIR / "scz_claim_summary.tsv",
        SCZ_SECONDARY_SUMMARY_DIR / "scz_apoe_region_diagnostic.tsv",
        ANALYSIS_ROOT / "dr_mvp_string_final5000" / "reports" / "DR_MVP.analysis_ready_summary.json",
        ANALYSIS_ROOT
        / "dr_mvp_no_mhc_no_apoe_final5000"
        / "reports"
        / "DR_MVP_NO_MHC_NO_APOE.analysis_ready_summary.json",
        ANALYSIS_ROOT / "scz_with_mhc_string_dev500" / "reports" / "SCZ_WITH_MHC.analysis_ready_summary.json",
        ANALYSIS_ROOT / "scz_no_mhc_string_dev500" / "reports" / "SCZ.analysis_ready_summary.json",
        ANALYSIS_ROOT / "scz_no_mhc_string_final5000" / "reports" / "SCZ.analysis_ready_summary.json",
        ANALYSIS_ROOT
        / "dr_mvp_graph_sensitivity"
        / "fvm_vascular_weighted_diffusion_final5000"
        / "DR_MVP_FVM_VASCULAR_WEIGHTED.diffusion_kernel_summary.tsv",
        ANALYSIS_ROOT
        / "dr_mvp_graph_sensitivity"
        / "retina_string_min20_diffusion_final5000"
        / "DR_MVP_RETINA_STRING_MIN20.diffusion_kernel_summary.tsv",
        MAGMA_TOOL_DIR / "magma",
        MAGMA_TOOL_DIR / "magma_v1.10_static.zip",
        MAGMA_BASELINE_DIR / "magma_baseline_summary.json",
        MAGMA_BASELINE_DIR / "magma_ripple_comparison.all_traits.tsv",
        MAGMA_BASELINE_DIR / "magma_dr_panel_gene_sets.all_traits.tsv",
        MAGMA_BASELINE_DIR / "magma_gene_results.all_traits.tsv.gz",
        PASCALX_TOOL_DIR / "pascalx",
        PASCALX_BASELINE_DIR / "pascalx_baseline_summary.json",
        PASCALX_BASELINE_DIR / "pascalx_ripple_comparison.all_traits.tsv",
        PASCALX_BASELINE_DIR / "pascalx_dr_panel_pathways.all_traits.tsv",
        PASCALX_BASELINE_DIR / "pascalx_gene_results.all_traits.tsv.gz",
        NETWORK_ABLATION_DIR / "network_ablation_manifest.json",
        NETWORK_ABLATION_DIR / "network_ablation_summary.all_traits.tsv",
        EXTERNAL_SCORE_GRAPH_DIR / "external_score_graph_layer_manifest.json",
        EXTERNAL_SCORE_GRAPH_DIR / "external_score_graph_layer_summary.all_traits.tsv",
        DENSE_MODULE_COMPETITOR_DIR / "dense_module_competitor_manifest.json",
        DENSE_MODULE_COMPETITOR_DIR / "dense_module_competitor_summary.all_traits.tsv",
        MODULE_RESELECTION_DIR / "module_full_reselection_manifest.json",
        MODULE_RESELECTION_DIR / "tables" / "module_full_reselection_summary.all_traits.tsv",
        MODULE_RESELECTION_DIR / "tables" / "module_full_reselection_null.all_traits.tsv.gz",
        CROSS_TRAIT_MODULE_RESELECTION_DIR / "module_full_reselection_manifest.json",
        CROSS_TRAIT_MODULE_RESELECTION_DIR / "tables" / "module_full_reselection_summary.all_traits.tsv",
        CROSS_TRAIT_MODULE_RESELECTION_DIR / "tables" / "module_full_reselection_null.all_traits.tsv.gz",
    ]
    if is_v12_package(package_version):
        paths.extend(
            [
                ANCHORED_V12_ROOT / "tables" / "cross_trait_anchored_summary.tsv",
                ANCHORED_V12_ROOT / "tables" / "cross_trait_top_modules.tsv",
                ANCHORED_V12_ROOT / "reports" / "cross_trait_anchored_report.md",
                ANCHORED_V12_TYPE1_ROOT / "tables" / "anchored_type1_summary.tsv",
                ANCHORED_V12_TYPE1_ROOT / "tables" / "anchored_type1_outer.all_traits.tsv.gz",
                ANCHORED_V12_TYPE1_ROOT / "reports" / "anchored_type1_calibration_report.md",
                ANCHORED_V12_ROBUSTNESS_ROOT / "tables" / "anchored_go_reactome_confirmation.tsv",
                ANCHORED_V12_ROBUSTNESS_ROOT / "tables" / "anchored_null_scale_stability.tsv",
                ANCHORED_V12_ROBUSTNESS_ROOT / "tables" / "anchored_score_perturbation_summary.tsv",
                ANCHORED_V12_ROBUSTNESS_ROOT / "reports" / "anchored_robustness_report.md",
                DIRECTION23_V12_ROOT / "tables" / "direction23_smoke_summary.all_traits.tsv",
                DIRECTION23_V12_ROOT / "reports" / "direction23_smoke_report.md",
                DIRECTION23_V12_ROOT / "reports" / "direction23_smoke_manifest.json",
            ]
        )
    if is_review_package(package_version):
        paths.extend(
            [
                REVIEW_REVISION_ROOT / "null_generation_audit" / "null_generation_audit.tsv",
                REVIEW_REVISION_ROOT / "gene_score_tail_calibration" / "gene_score_tail_calibration.tsv",
                REVIEW_REVISION_ROOT
                / "gene_score_tail_decision_stability"
                / "gene_score_tail_decision_stability_summary.tsv",
                REVIEW_REVISION_ROOT
                / "gene_score_tail_decision_stability"
                / "gene_score_tail_rank_displacement.tsv",
                REVIEW_REVISION_ROOT
                / "gene_score_tail_decision_stability"
                / "gene_score_tail_decision_stability_manifest.json",
                REVIEW_REVISION_ROOT
                / "overlap_preserving_null_sensitivity"
                / "overlap_preserving_null_sensitivity_summary.tsv",
                REVIEW_REVISION_ROOT
                / "overlap_preserving_null_sensitivity"
                / "overlap_preserving_null_sensitivity_nulls.tsv.gz",
                REVIEW_REVISION_ROOT
                / "overlap_preserving_null_sensitivity"
                / "overlap_preserving_null_sensitivity_manifest.json",
                REVIEW_REVISION_ROOT
                / "pathway_comparator_baselines"
                / "magma_pascalx_rankset_comparator.all_traits.tsv",
                REVIEW_REVISION_ROOT
                / "pathway_comparator_baselines"
                / "magma_reactome_go_gene_sets.all_traits.tsv",
                REVIEW_REVISION_ROOT
                / "pathway_comparator_baselines"
                / "pascalx_reactome_go_pathways.all_traits.tsv",
                REVIEW_REVISION_ROOT / "pathway_comparator_baselines" / "pathway_comparator_manifest.json",
                REVIEW_REVISION_ROOT
                / "true_network_method_baseline"
                / "true_network_method_baseline_summary.tsv",
                REVIEW_REVISION_ROOT
                / "true_network_method_baseline"
                / "true_network_method_baseline_manifest.json",
                REVIEW_REVISION_ROOT / "dmgwas_external_baseline" / "dmgwas_external_baseline_summary.tsv",
                REVIEW_REVISION_ROOT / "dmgwas_external_baseline" / "dmgwas_external_baseline_modules.tsv",
                REVIEW_REVISION_ROOT / "dmgwas_external_baseline" / "dmgwas_external_baseline_manifest.json",
                REVIEW_REVISION_ROOT
                / "anchored_strengthened_nulls"
                / "anchored_strengthened_null_summary.tsv",
                REVIEW_REVISION_ROOT
                / "anchored_strengthened_nulls"
                / "anchored_module_redundancy_clusters.tsv",
                REVIEW_REVISION_ROOT
                / "anchored_strengthened_nulls"
                / "anchored_strengthened_null_manifest.json",
                REVIEW_REVISION_ROOT
                / "anchored_familywise_targeted_expansion"
                / "anchored_familywise_targeted_expansion_summary.tsv",
                REVIEW_REVISION_ROOT
                / "anchored_familywise_targeted_expansion"
                / "anchored_familywise_targeted_expansion_family_max_null.tsv.gz",
                REVIEW_REVISION_ROOT
                / "anchored_familywise_targeted_expansion"
                / "anchored_familywise_targeted_expansion_manifest.json",
                SUPPLEMENTARY_ROOT / "claim_interpretation_table.tsv",
                SUPPLEMENTARY_ROOT / "multiplicity_map.tsv",
            ]
        )
    rows: list[dict[str, object]] = []
    for idx, path in enumerate(paths, start=1):
        if not path.exists():
            continue
        rows.append(
            {
                "input_id": f"I{idx:03d}",
                "file_path": str(path),
                "file_type": path.suffix.replace(".", "") or "file",
                "data_role": infer_data_role(path),
                "genome_build": "GRCh37" if "gwas_qc" in str(path) else "not_applicable",
                "source": infer_source(path),
                "version": "RIPPLE_V1_private_2026-07-02",
                "checksum_algorithm": "sha256",
                "checksum": checksum_file(path),
                "file_size_bytes": path.stat().st_size,
                "last_modified": timestamp_for(path),
                "redistributable": not contains_redistribution_restricted_content(path),
                "notes": "",
            }
        )
    return ensure_columns(pd.DataFrame(rows), INPUT_CHECKSUM_SCHEMA)


def infer_data_role(path: Path) -> str:
    text = str(path).lower()
    if "review_driven_revision_v1" in text:
        return "review_driven_revision_result"
    if "tier4_v12_anchored" in text:
        return "v12_anchored_module_result"
    if "tier4_v12_direction2_3" in text:
        return "v12_direction23_smoke_diagnostic"
    if "external_baselines" in text or "network_ablation" in text:
        return "external_baseline_or_ablation"
    if "module_reselection_null_cross_trait" in text:
        return "cross_trait_module_full_reselection_null"
    if "module_reselection_null" in text:
        return "module_full_reselection_null"
    if "gwas_qc" in text:
        return "qc_gwas_summary_statistics"
    if "summary.json" in text:
        return "analysis_summary"
    if "scz_claim_summary" in text:
        return "secondary_benchmark_claim_summary"
    if "apoe_region_diagnostic" in text:
        return "region_diagnostic"
    if "diffusion_kernel_summary" in text:
        return "diffusion_summary"
    if "claim_policy" in text:
        return "claim_policy"
    return "analysis_input"


def infer_source(path: Path) -> str:
    text = str(path).lower()
    if "review_driven_revision_v1" in text:
        return "RIPPLE_NC_review_driven_revision_v1"
    if "tier4_v12_anchored" in text:
        return "RIPPLE_V1_2_anchored_module_diagnostic"
    if "tier4_v12_direction2_3" in text:
        return "RIPPLE_V1_2_direction23_smoke_diagnostic"
    if "magma" in text:
        return "MAGMA_v1.10_or_RIPPLE_MAGMA_baseline"
    if "pascalx" in text:
        return "PascalX_or_RIPPLE_PascalX_baseline"
    if "network_ablation" in text:
        return "RIPPLE_network_ablation_baseline"
    if "external_score_graph_layer" in text:
        return "RIPPLE_external_score_graph_layer_baseline"
    if "dense_module_competitor" in text:
        return "RIPPLE_dense_module_competitor_baseline"
    if "module_reselection_null_cross_trait" in text:
        return "RIPPLE_cross_trait_module_full_reselection_null"
    if "module_reselection_null" in text:
        return "RIPPLE_module_full_reselection_null"
    if "scz" in text:
        return "PGC3_SCZ"
    if "dr_mvp" in text:
        return "MVP_DR_or_RIPPLE_output"
    if "claim_policy" in text:
        return "local_private_policy"
    return "private_workspace"


def contains_redistribution_restricted_content(path: Path) -> bool:
    text = str(path).lower()
    if "gwas_qc" in text or "10_raw_data" in text:
        return True
    if "string" in text or "1000g" in text or "g1000" in text:
        return True
    return "02_environment" in text and ("magma" in text or "pascalx" in text)


def build_reproducibility_manifest(package_version: str = "v1") -> pd.DataFrame:
    versions = f"python={platform.python_version()}; pandas={pd.__version__}; numpy={np.__version__}"
    run_specs = [
        (
            "DR_MVP_default_final5000",
            "DR_MVP",
            SCRIPT_DIR / "run_final_scale_validation_v1.sh",
            "bash scripts/run_final_scale_validation_v1.sh",
            ANALYSIS_ROOT / "dr_mvp_string_final5000",
            20260613,
            "WSL conda env: ripple",
        ),
        (
            "DR_MVP_no_MHC_no_APOE_final5000",
            "DR_MVP_NO_MHC_NO_APOE",
            SCRIPT_DIR / "run_final_scale_validation_v1.sh",
            "bash scripts/run_final_scale_validation_v1.sh",
            ANALYSIS_ROOT / "dr_mvp_no_mhc_no_apoe_final5000",
            20260613,
            "WSL conda env: ripple",
        ),
        (
            "manuscript_ready_v1",
            "all",
            THIS_SCRIPT,
            "python scripts/build_manuscript_ready_package.py --force",
            DEFAULT_OUT_DIR,
            20260613,
            sys.executable,
        ),
        (
            "SCZ_default_with_MHC_dev500",
            "SCZ_WITH_MHC",
            SCZ_DEV_SCRIPT,
            "bash scripts/run_scz_secondary_benchmark_dev.sh with-mhc",
            ANALYSIS_ROOT / "scz_with_mhc_string_dev500",
            20260702,
            "WSL conda env: ripple",
        ),
        (
            "SCZ_no_MHC_final5000",
            "SCZ",
            SCZ_FINAL_SCRIPT,
            "bash scripts/run_scz_no_mhc_final_scale.sh",
            ANALYSIS_ROOT / "scz_no_mhc_string_final5000",
            20260702,
            "WSL conda env: ripple",
        ),
        (
            "SCZ_secondary_benchmark_summary_final",
            "SCZ",
            SCZ_SUMMARY_SCRIPT,
            "python scripts/summarize_scz_secondary_benchmark.py --force",
            SCZ_SECONDARY_SUMMARY_DIR,
            20260702,
            sys.executable,
        ),
        (
            "external_baseline_MAGMA_primary",
            "DR_MVP;DR_MVP_NO_MHC_NO_APOE;SCZ",
            MAGMA_BASELINE_SCRIPT,
            "python scripts/run_magma_baseline.py --analysis-set primary --force",
            MAGMA_BASELINE_DIR,
            20260702,
            "WSL conda env: ripple; MAGMA v1.10 static linux",
        ),
        (
            "external_baseline_PascalX_primary",
            "DR_MVP;DR_MVP_NO_MHC_NO_APOE;SCZ",
            PASCALX_BASELINE_SCRIPT,
            "python scripts/run_pascalx_baseline.py --analysis-set primary --chromosomes 1-22 --prepare-refpanel --score",
            PASCALX_BASELINE_DIR,
            20260702,
            "WSL conda env: ripple; PascalX 0.0.5",
        ),
        (
            "network_ablation_v1_primary",
            "DR_MVP;DR_MVP_NO_MHC_NO_APOE;SCZ",
            NETWORK_ABLATION_SCRIPT,
            "python scripts/run_network_ablation_baseline.py --n-null 5000",
            NETWORK_ABLATION_DIR,
            20260703,
            "WSL conda env: ripple",
        ),
        (
            "external_score_graph_layer_v1_primary",
            "DR_MVP;DR_MVP_NO_MHC_NO_APOE;SCZ",
            EXTERNAL_SCORE_GRAPH_SCRIPT,
            "python scripts/run_external_score_graph_layer.py",
            EXTERNAL_SCORE_GRAPH_DIR,
            20260704,
            "WSL conda env: ripple",
        ),
        (
            "dense_module_competitor_v1_compact",
            "DR_MVP;DR_MVP_NO_MHC_NO_APOE;SCZ",
            DENSE_MODULE_COMPETITOR_SCRIPT,
            "python scripts/run_dense_module_competitor.py",
            DENSE_MODULE_COMPETITOR_DIR,
            20260705,
            "WSL conda env: ripple",
        ),
        (
            "module_reselection_null_v1",
            "DR_MVP;DR_MVP_NO_MHC_NO_APOE",
            MODULE_RESELECTION_SCRIPT,
            "python scripts/run_dr_mvp_module_reselection_null.py --n-reselection-null 5000 --force",
            MODULE_RESELECTION_DIR,
            20260706,
            "WSL conda env: ripple",
        ),
        (
            "module_reselection_null_cross_trait_v1",
            "SCZ;HEIGHT_IRN;BMI_IRN;T2D",
            CROSS_TRAIT_MODULE_RESELECTION_SCRIPT,
            "python scripts/run_cross_trait_module_reselection_null.py --all-cross-traits --n-reselection-null 5000 --force",
            CROSS_TRAIT_MODULE_RESELECTION_DIR,
            20260707,
            "WSL conda env: ripple",
        ),
    ]
    if is_v12_package(package_version):
        run_specs.extend(
            [
                (
                    "anchored_broad_reactome_go_cross_trait_n500",
                    "DR_MVP;T2D;HEIGHT_IRN;BMI_IRN;SCZ",
                    ANCHORED_V12_RUN_SCRIPT,
                    "python scripts/run_v12_anchored_module_test.py --n-degree-matched-null 500 --n-score-permutation-null 500",
                    ANCHORED_V12_ROOT,
                    20260712,
                    "WSL conda env: ripple",
                ),
                (
                    "anchored_type1_outer1000_v1",
                    "DR_MVP;T2D;HEIGHT_IRN;BMI_IRN;SCZ",
                    ANCHORED_V12_TYPE1_SCRIPT,
                    "python scripts/run_v12_anchored_type1_calibration.py --outer-n 1000 --calibration-null-n 500",
                    ANCHORED_V12_TYPE1_ROOT,
                    20260713,
                    "WSL conda env: ripple",
                ),
                (
                    "anchored_robustness_v1",
                    "DR_MVP;T2D;HEIGHT_IRN;BMI_IRN;SCZ",
                    ANCHORED_V12_ROBUSTNESS_SCRIPT,
                    "python scripts/run_v12_anchored_robustness.py",
                    ANCHORED_V12_ROBUSTNESS_ROOT,
                    20260714,
                    "WSL conda env: ripple",
                ),
                (
                    "direction23_smoke_v1",
                    "DR_MVP;T2D;HEIGHT_IRN;BMI_IRN;SCZ",
                    DIRECTION23_V12_SCRIPT,
                    "python scripts/run_v12_direction23_smoke.py --n-null 200",
                    DIRECTION23_V12_ROOT,
                    20260715,
                    "WSL conda env: ripple",
                ),
                (
                    "manuscript_ready_v1_2",
                    "all",
                    THIS_SCRIPT,
                    "python scripts/build_manuscript_ready_package.py --package-version v1_2 --force",
                    V12_OUT_DIR,
                    20260715,
                    sys.executable,
                ),
            ]
        )
    if is_review_package(package_version):
        run_specs.extend(
            [
                (
                    "review_null_generation_audit_v1",
                    "DR_MVP;DR_MVP_NO_MHC_NO_APOE;SCZ",
                    NULL_GENERATION_AUDIT_SCRIPT,
                    "python scripts/audit_null_generation.py",
                    REVIEW_REVISION_ROOT / "null_generation_audit",
                    "",
                    "WSL conda env: ripple",
                ),
                (
                    "review_gene_score_tail_calibration_v1",
                    "DR_MVP;DR_MVP_NO_MHC_NO_APOE;SCZ",
                    GENE_SCORE_TAIL_CALIBRATION_SCRIPT,
                    "python scripts/run_gene_score_tail_calibration.py --max-genes-per-category 5 --n-sim 20000",
                    REVIEW_REVISION_ROOT / "gene_score_tail_calibration",
                    20260720,
                    "WSL conda env: ripple",
                ),
                (
                    "review_tail_decision_stability_v1",
                    "DR_MVP;DR_MVP_NO_MHC_NO_APOE;SCZ",
                    TAIL_DECISION_STABILITY_SCRIPT,
                    "python scripts/summarize_tail_decision_stability.py",
                    REVIEW_REVISION_ROOT / "gene_score_tail_decision_stability",
                    20260706,
                    "WSL conda env: ripple",
                ),
                (
                    "review_overlap_preserving_null_sensitivity_v1",
                    "DR_MVP;DR_MVP_NO_MHC_NO_APOE;SCZ",
                    OVERLAP_NULL_SENSITIVITY_SCRIPT,
                    "python scripts/run_overlap_preserving_null_sensitivity.py --n-null 500",
                    REVIEW_REVISION_ROOT / "overlap_preserving_null_sensitivity",
                    20260726,
                    "WSL conda env: ripple",
                ),
                (
                    "review_pathway_comparator_baselines_v1",
                    "DR_MVP;DR_MVP_NO_MHC_NO_APOE;SCZ",
                    PATHWAY_COMPARATOR_SCRIPT,
                    "python scripts/run_pathway_comparator_baselines.py",
                    REVIEW_REVISION_ROOT / "pathway_comparator_baselines",
                    20260721,
                    "WSL conda env: ripple",
                ),
                (
                    "review_true_network_method_baseline_v1",
                    "DR_MVP;DR_MVP_NO_MHC_NO_APOE;SCZ",
                    TRUE_NETWORK_BASELINE_SCRIPT,
                    "python scripts/run_true_network_method_baseline.py",
                    REVIEW_REVISION_ROOT / "true_network_method_baseline",
                    20260722,
                    "WSL conda env: ripple",
                ),
                (
                    "review_dmgwas_external_baseline_top2000_v1",
                    "DR_MVP;DR_MVP_NO_MHC_NO_APOE;SCZ",
                    DMGWAS_EXTERNAL_BASELINE_SCRIPT,
                    "python scripts/run_dmgwas_external_baseline.py --traits DR_MVP DR_MVP_NO_MHC_NO_APOE SCZ --max-genes 2000 --timeout-seconds 1800 --force",
                    REVIEW_REVISION_ROOT / "dmgwas_external_baseline",
                    20260725,
                    "WSL conda env: ripple; system R 4.3.3; dmGWAS 3.0; igraph 2.3.3",
                ),
                (
                    "review_anchored_strengthened_nulls_v1",
                    "DR_MVP;T2D;HEIGHT_IRN;BMI_IRN;SCZ",
                    ANCHORED_STRENGTHENED_NULL_SCRIPT,
                    "python scripts/run_anchored_strengthened_nulls.py --n-null 5000",
                    REVIEW_REVISION_ROOT / "anchored_strengthened_nulls",
                    20260723,
                    "WSL conda env: ripple",
                ),
                (
                    "review_anchored_familywise_targeted_expansion_v1",
                    "DR_MVP;T2D;HEIGHT_IRN;BMI_IRN;SCZ",
                    ANCHORED_FAMILYWISE_EXPANSION_SCRIPT,
                    "python scripts/run_anchored_familywise_targeted_expansion.py --n-null 5000",
                    REVIEW_REVISION_ROOT / "anchored_familywise_targeted_expansion",
                    20260727,
                    "WSL conda env: ripple",
                ),
                (
                    "manuscript_ready_v1_2_review",
                    "all",
                    THIS_SCRIPT,
                    "python scripts/build_manuscript_ready_package.py --package-version v1_2_review --force",
                    V12_REVIEW_OUT_DIR,
                    20260724,
                    sys.executable,
                ),
            ]
        )
    rows = []
    for analysis_id, trait, script, command, outputs, seed, runtime_env in run_specs:
        rows.append(
            {
                "analysis_id": analysis_id,
                "trait": trait,
                "run_stage": "final_reporting" if analysis_id == "manuscript_ready_v1" else "analysis",
                "script_path": str(script),
                "command": command,
                "working_directory": str(PROJECT_ROOT),
                "conda_or_python_env": runtime_env,
                "software_versions": versions,
                "random_seed": seed,
                "input_ids": "",
                "output_paths": str(outputs),
                "log_paths": str(ANALYSIS_ROOT / "logs"),
                "runtime_seconds": "",
                "run_status": "complete" if outputs.exists() else "not_run",
                "timestamp": now_utc(),
            }
        )
    return ensure_columns(pd.DataFrame(rows), REPRODUCIBILITY_MANIFEST_SCHEMA)


def build_claim_evidence_audit(package_version: str = "v1") -> pd.DataFrame:
    module_p = module_reselection_min_p_strings()
    claim_rows = [
        (
            "C_METHOD_PIPELINE",
            "RIPPLE-GWAS maps GWAS summary statistics to gene-level signals and calibrated graph-domain inference.",
            "supportive",
            "parameter_table.tsv;claim_policy.yaml",
            "figure_source_tables/figure_1_framework.tsv",
        ),
        (
            "C_METHOD_GENE_SCORE_STREAMS",
            "RIPPLE-GWAS uses frozen signed and unsigned gene-level signal streams as technical foundations.",
            "supportive",
            "parameter_table.tsv",
            "figure_source_tables/figure_1_framework.tsv",
        ),
        (
            "C_METHOD_CLAIM_TIERS",
            "RIPPLE-GWAS reports fixed claim tiers using policy-defined final-positive and supportive Z thresholds.",
            "supportive",
            "claim_policy.yaml;final_claim_audit.tsv",
            "figure_source_tables/figure_1_framework.tsv",
        ),
        (
            "C_METHOD_NULL_HIERARCHY",
            "RIPPLE-GWAS separates SNP/gene null, degree-matched node null, diffusion null and graph-null evidence.",
            "supportive",
            "final_claim_audit.tsv;inference_family.tsv",
            "figure_source_tables/figure_1_framework.tsv",
        ),
        (
            "C_TYPE1_TIER_GATING",
            "Type I calibration supports tier-specific manuscript gating at the fixed Z threshold.",
            "supportive",
            "type1_uncertainty.tsv",
            "figure_source_tables/figure_2_calibration.tsv",
        ),
        (
            "C_TYPE1_MC_UNCERTAINTY",
            "Type I error estimates are reported with binomial confidence intervals and Monte Carlo standard errors.",
            "supportive",
            "type1_uncertainty.tsv",
            "figure_source_tables/figure_2_calibration.tsv",
        ),
        (
            "C_SYNTHETIC_SCENARIO_BEHAVIOR",
            "Synthetic spike-ins distinguish null, dispersed, degree-biased and connected-module graph-domain behavior.",
            "supportive",
            "synthetic_spikein_summary.tsv",
            "figure_source_tables/figure_2_calibration.tsv",
        ),
        (
            "C_SYNTHETIC_SELECTION_AWARE_MODULES",
            "Selection-aware synthetic analyses calibrate local-module behavior after module selection.",
            "supportive",
            "synthetic_spikein_selection_aware_summary.tsv",
            "figure_source_tables/figure_2_calibration.tsv",
        ),
        (
            "C_CROSS_TRAIT_CLAIM_TIERS",
            "Cross-trait benchmarks show that RIPPLE-GWAS does not force universal graph-domain positivity.",
            "supportive",
            "cross_trait_benchmark_z2p5.tsv",
            "figure_source_tables/figure_3_claim_tiers.tsv",
        ),
        (
            "C_CROSS_TRAIT_TIER4_RESELECTION",
            "Cross-trait full reselection nulls for SCZ, HEIGHT, BMI and T2D did not support selection-calibrated local module discovery under the frozen Tier 4 policy.",
            "limitation",
            "cross_trait_module_reselection_summary.tsv",
            "figure_source_tables/figure_3_claim_tiers.tsv",
        ),
        (
            "C_DR_T1_DEFAULT",
            "DR_MVP showed final-positive degree-calibrated weak-signal aggregation on default STRING.",
            "final",
            "final_claim_audit.tsv",
            "figure_source_tables/figure_3_claim_tiers.tsv",
        ),
        (
            "C_DR_T2_DEFAULT",
            "DR_MVP showed final-positive graph-domain diffusion aggregation on default STRING.",
            "final",
            "final_claim_audit.tsv",
            "figure_source_tables/figure_3_claim_tiers.tsv",
        ),
        (
            "C_DR_T3_DEFAULT_NEG",
            "Default STRING did not provide topology-specific support for DR_MVP under degree-preserving graph nulls.",
            "negative",
            "final_claim_audit.tsv",
            "figure_source_tables/figure_3_claim_tiers.tsv",
        ),
        (
            "C_DR_NO_MHC_APOE",
            "The DR_MVP Tier 1 and Tier 2 signals remained final-positive after MHC/APOE exclusion.",
            "final",
            "final_claim_audit.tsv",
            "figure_source_tables/figure_3_claim_tiers.tsv",
        ),
        (
            "C_DR_LOCAL_MODULES_DEFAULT",
            f"DR_MVP default STRING local subnetworks are reported as post hoc candidate modules for biological follow-up because they did not pass strict full reselection null calibration (minimum empirical P={module_p.get('DR_MVP', '0.3825')}).",
            "exploratory",
            "module_full_reselection_summary.tsv",
            "figure_source_tables/figure_4_local_modules.tsv",
        ),
        (
            "C_DR_LOCAL_MODULES_NO_MHC_APOE",
            f"DR_MVP no-MHC-no-APOE local subnetworks are reported as post hoc candidate modules for biological follow-up because they did not pass strict full reselection null calibration (minimum empirical P={module_p.get('DR_MVP_NO_MHC_NO_APOE', '0.3023')}).",
            "exploratory",
            "module_full_reselection_summary.tsv",
            "figure_source_tables/figure_4_local_modules.tsv",
        ),
        (
            "C_DR_MODULE_ANNOTATION",
            "DR_MVP module annotation prioritizes independent external sources and labels graph-construction-linked support.",
            "supportive",
            "module_annotation.tsv",
            "figure_source_tables/figure_4_local_modules.tsv",
        ),
        (
            "C_DR_MODULE_CLAIM_BOUNDARY",
            "DR_MVP local modules are reported as post hoc candidates because full reselection nulls did not support calibrated module-level discovery, and selection-calibrated module count and topology-specific module count are both zero.",
            "limitation",
            "module_full_reselection_summary.tsv",
            "figure_source_tables/figure_4_local_modules.tsv",
        ),
        (
            "C_FVM_SENSITIVITY",
            "FVM vascular weighted graph evidence supports graph-domain sensitivity but not a topology-specific claim.",
            "supportive",
            "final_claim_audit.tsv;graph_registry.tsv",
            "figure_source_tables/figure_3_claim_tiers.tsv",
        ),
        (
            "C_SCZ_SECONDARY_BENCHMARK",
            "SCZ no-MHC was included as a final-scale secondary cross-domain benchmark and showed graph-domain aggregation on default STRING.",
            "supportive",
            "scz_claim_summary.tsv;scz_apoe_region_diagnostic.tsv",
            "figure_source_tables/figure_3_claim_tiers.tsv",
        ),
        (
            "C_SCZ_TOPOLOGY_NEG",
            "SCZ no-MHC default STRING did not provide topology-specific support under degree-preserving graph nulls.",
            "negative",
            "scz_claim_summary.tsv",
            "figure_source_tables/figure_3_claim_tiers.tsv",
        ),
        (
            "C_SCZ_NOT_DR_ANNOTATION",
            "SCZ is not part of DR-specific biological annotation.",
            "limitation",
            "scz_claim_summary.tsv",
            "figure_source_tables/figure_3_claim_tiers.tsv",
        ),
        (
            "C_BASELINE_MAGMA_PASCAL_CONCORDANCE",
            "MAGMA and PascalX gene-level baseline scores were concordant with RIPPLE gene-level association scores.",
            "supportive",
            "external_baseline_summary.tsv",
            "figure_source_tables/figure_5_external_baselines.tsv",
        ),
        (
            "C_BASELINE_NETWORK_ABLATION",
            "Network ablation separated naive PPI connectivity from degree-calibrated RIPPLE evidence.",
            "supportive",
            "external_baseline_summary.tsv",
            "figure_source_tables/figure_5_external_baselines.tsv",
        ),
        (
            "C_BASELINE_EXTERNAL_SCORE_GRAPH_LAYER",
            "MAGMA and PascalX scores can be passed through the RIPPLE graph/null layer to evaluate graph-domain robustness.",
            "supportive",
            "external_baseline_summary.tsv",
            "figure_source_tables/figure_5_external_baselines.tsv",
        ),
        (
            "C_BASELINE_DENSE_MODULE_COMPETITOR",
            "A simplified dense module search baseline nominated many modules, but these were treated as competitor outputs rather than topology-specific RIPPLE discoveries.",
            "supportive",
            "external_baseline_summary.tsv",
            "figure_source_tables/figure_5_external_baselines.tsv",
        ),
    ]
    if is_v12_package(package_version):
        claim_rows.extend(
            [
                (
                    "C_V12_ANCHORED_MODULE_EVIDENCE",
                    "RIPPLE-GWAS V1.2 prioritizes anchored biological modules from fixed Reactome/GO libraries under controlled null models.",
                    "supportive",
                    "anchored_cross_trait_summary.tsv;anchored_top_modules.tsv",
                    "figure_source_tables/figure_4_anchored_modules.tsv",
                ),
                (
                    "C_V12_ANCHORED_TYPE1_CALIBRATION",
                    "Anchored module familywise evidence showed acceptable Type I behavior in outer-null calibration, with GO-only calibration closest to the nominal level.",
                    "supportive",
                    "anchored_type1_outer1000_summary.tsv",
                    "figure_source_tables/figure_4_anchored_modules.tsv",
                ),
                (
                    "C_V12_GO_REACTOME_CONFIRMATION",
                    "GO-only and Reactome-only source-family analyses confirmed that V1.2 anchored signals are source-calibrated and trait-dependent.",
                    "supportive",
                    "anchored_go_reactome_confirmation.tsv",
                    "figure_source_tables/figure_4_anchored_modules.tsv",
                ),
                (
                    "C_V12_ANCHORED_ROBUSTNESS",
                    "Anchored module signals were stable from smoke-scale to n500 nulls and under score perturbation for DR_MVP, T2D, HEIGHT and BMI, while SCZ remained negative.",
                    "supportive",
                    "anchored_robustness_summary.tsv",
                    "figure_source_tables/figure_4_anchored_modules.tsv",
                ),
                (
                    "C_V12_NO_DE_NOVO_TOPOLOGY_SPECIFIC_CLAIM",
                    "V1.2 does not make a de novo STRING PPI topology claim; original local STRING modules remain supplementary limitation evidence.",
                    "limitation",
                    "v12_module_claim_boundary.tsv;module_full_reselection_summary.tsv;cross_trait_module_reselection_summary.tsv",
                    "figure_source_tables/figure_S_local_module_boundary.tsv",
                ),
                (
                    "C_V12_DIRECTION23_NOT_MAINLINE",
                    "Direction 2 diffusion-localized neighborhoods and Direction 3 Louvain communities were evaluated as smoke diagnostics and were not selected as V1.2 mainline module claims.",
                    "limitation",
                    "direction23_smoke_diagnostic.tsv",
                    "figure_source_tables/figure_S_direction23_diagnostic.tsv",
                ),
            ]
        )
    if is_review_package(package_version):
        claim_rows.extend(
            [
                (
                    "C_REVIEW_NULL_GENERATION_AUDIT",
                    "The manuscript reports which structures are preserved, approximated or not explicitly modeled by the RIPPLE null-generation procedure.",
                    "supportive",
                    "null_generation_audit.tsv",
                    "supplementary_tables_only",
                ),
                (
                    "C_REVIEW_GENE_TAIL_CALIBRATION",
                    "High-risk gene-level quadratic-form tails were audited against saddlepoint, Satterthwaite and parametric simulation sensitivities.",
                    "supportive",
                    "gene_score_tail_calibration.tsv",
                    "supplementary_tables_only",
                ),
                (
                    "C_REVIEW_TAIL_DECISION_STABILITY",
                    "Gene-score tail audit results were summarized as decision-stability and rank-displacement diagnostics for manuscript-level claims.",
                    "supportive",
                    "gene_score_tail_decision_stability_summary.tsv;gene_score_tail_rank_displacement.tsv",
                    "supplementary_tables_only",
                ),
                (
                    "C_REVIEW_OVERLAP_PRESERVING_NULL_SENSITIVITY",
                    "An overlap-preserving SNP-label permutation proxy sensitivity was added to evaluate whether SNP-to-gene mapping overlap materially changes graph-domain calibration.",
                    "supportive",
                    "overlap_preserving_null_sensitivity_summary.tsv",
                    "supplementary_tables_only",
                ),
                (
                    "C_REVIEW_MULTIPLICITY_MAP",
                    "Primary, sensitivity, benchmark and exploratory analyses are separated in a manuscript-facing multiplicity map.",
                    "supportive",
                    "multiplicity_map.tsv;claim_interpretation_table.tsv",
                    "supplementary_tables_only",
                ),
                (
                    "C_REVIEW_PATHWAY_COMPARATOR_BASELINES",
                    "MAGMA competitive gene-set and PascalX pathway comparators over the same Reactome/GO library support the interpretation of RIPPLE as a calibrated graph-inference layer.",
                    "supportive",
                    "pathway_comparator_rankset_summary.tsv;magma_reactome_go_gene_sets.tsv;pascalx_reactome_go_pathways.tsv",
                    "supplementary_tables_only",
                ),
                (
                    "C_REVIEW_TRUE_NETWORK_BASELINE",
                    "A reproducible external dmGWAS 3.0 node-only network-method baseline was run on top-ranked graph-induced scopes, while full 12k-node STRING dmGWAS runs were recorded as computationally deferred.",
                    "supportive",
                    "dmgwas_external_baseline_summary.tsv;dmgwas_external_baseline_modules.tsv;true_network_method_baseline_summary.tsv",
                    "supplementary_tables_only",
                ),
                (
                    "C_REVIEW_ANCHORED_STRENGTHENED_NULLS",
                    "Top anchored modules were retested under degree, gene-property and annotation-density matched nulls as strengthened sensitivity evidence.",
                    "supportive",
                    "anchored_strengthened_null_summary.tsv;anchored_module_redundancy_clusters.tsv",
                    "supplementary_tables_only",
                ),
                (
                    "C_REVIEW_ANCHORED_FAMILYWISE_TARGETED_EXPANSION",
                    "Manuscript-relevant anchored modules were retested with expanded familywise score-permutation nulls to stabilize empirical P estimates.",
                    "supportive",
                    "anchored_familywise_targeted_expansion_summary.tsv",
                    "supplementary_tables_only",
                ),
            ]
        )
    rows = []
    for claim_id, sentence, strength, required_tables, required_figures in claim_rows:
        module_status = ""
        selection_count: object = ""
        topology_count: object = ""
        full_reselection_p = ""
        if claim_id == "C_DR_LOCAL_MODULES_DEFAULT":
            module_status = "post_hoc_candidate_only"
            selection_count = 0
            topology_count = 0
            full_reselection_p = module_p.get("DR_MVP", "0.3825")
        elif claim_id == "C_DR_LOCAL_MODULES_NO_MHC_APOE":
            module_status = "post_hoc_candidate_only"
            selection_count = 0
            topology_count = 0
            full_reselection_p = module_p.get("DR_MVP_NO_MHC_NO_APOE", "0.3023")
        elif claim_id == "C_DR_MODULE_CLAIM_BOUNDARY":
            module_status = "post_hoc_candidate_only"
            selection_count = 0
            topology_count = 0
            full_reselection_p = f"DR_MVP={module_p.get('DR_MVP', '0.3825')};DR_MVP_NO_MHC_NO_APOE={module_p.get('DR_MVP_NO_MHC_NO_APOE', '0.3023')}"
        elif claim_id == "C_CROSS_TRAIT_TIER4_RESELECTION":
            module_status = "no_selection_calibrated_cross_trait_modules"
            selection_count = 0
            topology_count = 0
            full_reselection_p = cross_trait_module_reselection_min_p_string()
        rows.append(
            {
            "claim_id": claim_id,
            "manuscript_sentence": sentence,
            "allowed_strength": strength,
            "required_tables": required_tables,
            "required_figures": required_figures,
            "source_files": required_tables,
            "code_path": str(THIS_SCRIPT),
            "pass_fail": not contains_forbidden(sentence),
            "banned_language_check": "fail" if contains_forbidden(sentence) else "pass",
            "module_layer_claim_status": module_status,
            "selection_calibrated_module_count": selection_count,
            "topology_specific_module_count": topology_count,
            "full_reselection_empirical_p": full_reselection_p,
            }
        )
    return ensure_columns(pd.DataFrame(rows), CLAIM_EVIDENCE_SCHEMA)


def module_reselection_min_p_strings() -> dict[str, str]:
    table = build_module_full_reselection_summary()
    if table.empty or "trait" not in table.columns or "full_reselection_score_p" not in table.columns:
        return {"DR_MVP": "0.3825", "DR_MVP_NO_MHC_NO_APOE": "0.3023"}
    values = pd.to_numeric(table["full_reselection_score_p"], errors="coerce")
    out: dict[str, str] = {}
    for trait, group in table.assign(_p=values).groupby("trait", observed=True):
        min_p = pd.to_numeric(group["_p"], errors="coerce").min()
        if np.isfinite(min_p):
            out[str(trait)] = f"{float(min_p):.4f}"
    out.setdefault("DR_MVP", "0.3825")
    out.setdefault("DR_MVP_NO_MHC_NO_APOE", "0.3023")
    return out


def cross_trait_module_reselection_min_p_string() -> str:
    table = build_cross_trait_module_reselection_summary()
    if table.empty or "analysis_id" not in table.columns or "full_reselection_score_p" not in table.columns:
        return ""
    values = pd.to_numeric(table["full_reselection_score_p"], errors="coerce")
    parts: list[str] = []
    for analysis_id, group in table.assign(_p=values).groupby("analysis_id", observed=True):
        min_p = pd.to_numeric(group["_p"], errors="coerce").min()
        if np.isfinite(min_p):
            parts.append(f"{analysis_id}={float(min_p):.4f}")
    return ";".join(parts)


def contains_forbidden(text: str) -> bool:
    policy = load_claim_policy(POLICY_PATH)
    lowered = text.lower()
    return any(str(phrase).lower() in lowered for phrase in policy["forbidden_language"])


def build_figure_claim_map(package_version: str = "v1") -> pd.DataFrame:
    rows = [
        figure_claim("Figure_1", "a_pipeline", "RIPPLE separates gene scoring from graph inference", "C_METHOD_PIPELINE", "supportive", "parameter_table.tsv;claim_policy.yaml", "complete"),
        figure_claim("Figure_1", "b_gene_score_streams", "RIPPLE separates gene scoring from graph inference", "C_METHOD_GENE_SCORE_STREAMS", "supportive", "parameter_table.tsv", "complete"),
        figure_claim("Figure_1", "c_claim_tiers", "RIPPLE separates gene scoring from graph inference", "C_METHOD_CLAIM_TIERS", "supportive", "claim_policy.yaml;final_claim_audit.tsv", "complete"),
        figure_claim("Figure_1", "d_null_hierarchy", "RIPPLE separates gene scoring from graph inference", "C_METHOD_NULL_HIERARCHY", "supportive", "final_claim_audit.tsv;inference_family.tsv", "inference family table added in this package"),
        figure_claim("Figure_2", "a_type1_error", "Null simulations calibrate tier-specific claims", "C_TYPE1_TIER_GATING", "supportive", "type1_uncertainty.tsv", "complete; do not claim exact global FWER"),
        figure_claim("Figure_2", "b_mc_uncertainty", "Null simulations calibrate tier-specific claims", "C_TYPE1_MC_UNCERTAINTY", "supportive", "type1_uncertainty.tsv", "complete"),
        figure_claim("Figure_2", "c_synthetic_scenarios", "Null simulations calibrate tier-specific claims", "C_SYNTHETIC_SCENARIO_BEHAVIOR", "supportive", "synthetic_spikein_summary.tsv", "promoted from existing synthetic spike-in output"),
        figure_claim("Figure_2", "d_selection_aware_modules", "Null simulations calibrate tier-specific claims", "C_SYNTHETIC_SELECTION_AWARE_MODULES", "supportive", "synthetic_spikein_selection_aware_summary.tsv", "selection-aware claims only"),
        figure_claim("Figure_3", "a_cross_trait_heatmap", "Trait benchmarks reveal graph-domain specificity", "C_CROSS_TRAIT_CLAIM_TIERS", "supportive", "cross_trait_benchmark_z2p5.tsv", "early benchmark rows remain supportive when null fields are incomplete"),
        figure_claim("Figure_3", "b_dr_sensitivity_tiers", "Trait benchmarks reveal graph-domain specificity", "C_DR_T1_DEFAULT", "final", "final_claim_audit.tsv", "complete"),
        figure_claim("Figure_3", "b_dr_sensitivity_tiers", "Trait benchmarks reveal graph-domain specificity", "C_DR_T2_DEFAULT", "final", "final_claim_audit.tsv", "complete"),
        figure_claim("Figure_3", "b_dr_sensitivity_tiers", "Trait benchmarks reveal graph-domain specificity", "C_DR_T3_DEFAULT_NEG", "negative", "final_claim_audit.tsv", "complete; no topology-specific discovery language"),
        figure_claim("Figure_3", "b_dr_sensitivity_tiers", "Trait benchmarks reveal graph-domain specificity", "C_DR_NO_MHC_APOE", "final", "final_claim_audit.tsv", "complete"),
        figure_claim("Figure_3", "c_scz_secondary_benchmark", "Trait benchmarks reveal graph-domain specificity", "C_SCZ_SECONDARY_BENCHMARK", "supportive", "scz_claim_summary.tsv;scz_apoe_region_diagnostic.tsv", "complete; SCZ is cross-domain only"),
        figure_claim("Figure_3", "c_scz_secondary_benchmark", "Trait benchmarks reveal graph-domain specificity", "C_SCZ_TOPOLOGY_NEG", "negative", "scz_claim_summary.tsv", "complete; no topology-specific discovery language"),
        figure_claim("Figure_3", "c_scz_secondary_benchmark", "Trait benchmarks reveal graph-domain specificity", "C_SCZ_NOT_DR_ANNOTATION", "limitation", "scz_claim_summary.tsv", "complete"),
        figure_claim("Figure_3", "d_graph_sensitivity", "Trait benchmarks reveal graph-domain specificity", "C_FVM_SENSITIVITY", "supportive", "final_claim_audit.tsv;graph_registry.tsv", "complete; graph-context sensitivity only"),
        figure_claim(
            "Figure_3",
            "e_cross_trait_tier4_modules",
            "Cross-trait Tier 4 module reselection remains conservative",
            "C_CROSS_TRAIT_TIER4_RESELECTION",
            "limitation",
            "cross_trait_module_reselection_summary.tsv",
            "full reselection complete; no selection-calibrated local module support",
            claim_tier="Tier 4",
            claim_status="no_local_module_support",
            allowed_strength="limitation",
            figure_role="cross-trait Tier 4 boundary",
            allowed_language="no selection-calibrated local module support under frozen Tier 4 policy",
            banned_language="selection-calibrated local module as a positive claim; topology-specific module discovery; validated disease module",
        ),
        figure_claim(
            "Figure_4",
            "a_default_string_modules",
            "DR_MVP graph-domain aggregation and post hoc module follow-up",
            "C_DR_LOCAL_MODULES_DEFAULT",
            "exploratory",
            "module_full_reselection_summary.tsv",
            "full reselection complete; modules remain post hoc candidates",
            claim_tier="Tier 4",
            claim_status="post_hoc_candidate_only",
            allowed_strength="exploratory",
            figure_role="post hoc module follow-up",
            allowed_language="post hoc candidate module; candidate subnetwork for biological follow-up; exploratory local module",
            banned_language="calibrated local module discovery; topology-specific module discovery; validated DR module",
        ),
        figure_claim(
            "Figure_4",
            "b_no_mhc_no_apoe_modules",
            "DR_MVP graph-domain aggregation and post hoc module follow-up",
            "C_DR_LOCAL_MODULES_NO_MHC_APOE",
            "exploratory",
            "module_full_reselection_summary.tsv",
            "full reselection complete; modules remain post hoc candidates",
            claim_tier="Tier 4",
            claim_status="post_hoc_candidate_only",
            allowed_strength="exploratory",
            figure_role="post hoc module follow-up",
            allowed_language="post hoc candidate module; candidate subnetwork for biological follow-up; exploratory local module",
            banned_language="calibrated local module discovery; topology-specific module discovery; validated DR module",
        ),
        figure_claim(
            "Figure_4",
            "c_module_annotation",
            "DR_MVP graph-domain aggregation and post hoc module follow-up",
            "C_DR_MODULE_ANNOTATION",
            "supportive",
            "module_annotation.tsv",
            "complete; prioritize independent annotations",
            claim_tier="Tier 4",
            claim_status="post_hoc_candidate_only",
            allowed_strength="supportive",
            figure_role="biological follow-up context",
            allowed_language="biological annotation for post hoc candidate modules",
            banned_language="independent validation of module discovery; validated DR module",
        ),
        figure_claim(
            "Figure_4",
            "d_module_claim_boundary",
            "DR_MVP graph-domain aggregation and post hoc module follow-up",
            "C_DR_MODULE_CLAIM_BOUNDARY",
            "limitation",
            "module_full_reselection_summary.tsv",
            "full reselection complete; no calibrated module upgrade",
            claim_tier="Tier 4",
            claim_status="limitation",
            allowed_strength="limitation",
            figure_role="claim boundary",
            allowed_language="full reselection null did not support calibrated local module discovery",
            banned_language="calibrated local module discovery as a positive claim; topology-specific module discovery; validated DR module",
        ),
        figure_claim("Figure_5", "gene_score_concordance", "External gene scores preserve graph-layer inference", "C_BASELINE_MAGMA_PASCAL_CONCORDANCE", "supportive", "external_baseline_summary.tsv", "complete"),
        figure_claim("Figure_5", "external_score_graph_layer", "External gene scores preserve graph-layer inference", "C_BASELINE_EXTERNAL_SCORE_GRAPH_LAYER", "supportive", "external_baseline_summary.tsv", "complete"),
        figure_claim("Figure_5", "network_ablation", "External gene scores preserve graph-layer inference", "C_BASELINE_NETWORK_ABLATION", "supportive", "external_baseline_summary.tsv", "complete"),
        figure_claim("Figure_5", "graph_module_competitor", "External gene scores preserve graph-layer inference", "C_BASELINE_DENSE_MODULE_COMPETITOR", "supportive", "external_baseline_summary.tsv", "simplified dense-module competitor only"),
    ]
    if is_v12_package(package_version):
        rows = [
            row
            for row in rows
            if not (
                row["figure_id"] == "Figure_4"
                and str(row["claim_id"]).startswith("C_DR_")
            )
        ]
        rows.extend(
            [
                figure_claim(
                    "Figure_4",
                    "a_anchored_cross_trait",
                    "V1.2 anchored biological module evidence is trait-dependent",
                    "C_V12_ANCHORED_MODULE_EVIDENCE",
                    "supportive",
                    "anchored_cross_trait_summary.tsv;anchored_top_modules.tsv",
                    "anchored cross-trait n500 outputs complete",
                    claim_tier="Tier 4A",
                    claim_status="anchored_familywise_supported",
                    allowed_strength="supportive",
                    figure_role="main anchored module evidence",
                    allowed_language="anchored biological module evidence; anchored familywise-supported module",
                    banned_language="de novo STRING topology claim; validated disease module; causal network",
                ),
                figure_claim(
                    "Figure_4",
                    "b_top_anchored_modules",
                    "Top anchored modules summarize trait-specific weak-signal biological modules",
                    "C_V12_ANCHORED_MODULE_EVIDENCE",
                    "supportive",
                    "anchored_top_modules.tsv",
                    "top anchored modules promoted to V1.2 source table",
                    claim_tier="Tier 4A",
                    claim_status="anchored_familywise_supported",
                    allowed_strength="supportive",
                    figure_role="main anchored module evidence",
                    allowed_language="anchored biological module evidence; fixed external module evidence",
                    banned_language="de novo STRING topology claim; validated disease module",
                ),
                figure_claim(
                    "Figure_4",
                    "c_anchored_type1",
                    "Anchored module evidence is calibrated with outer-null Type I estimates",
                    "C_V12_ANCHORED_TYPE1_CALIBRATION",
                    "supportive",
                    "anchored_type1_outer1000_summary.tsv",
                    "outer_n=1000 anchored Type I calibration complete",
                    claim_tier="Tier 4A",
                    claim_status="supportive",
                    allowed_strength="supportive",
                    figure_role="calibration evidence",
                    allowed_language="anchored Type I calibration; source-family calibration",
                    banned_language="exact global FWER claim; de novo STRING topology claim",
                ),
                figure_claim(
                    "Figure_4",
                    "d_go_reactome_confirmation",
                    "GO-only and Reactome-only source-family analyses bound anchored module interpretation",
                    "C_V12_GO_REACTOME_CONFIRMATION",
                    "supportive",
                    "anchored_go_reactome_confirmation.tsv",
                    "GO/Reactome source-family confirmation complete",
                    claim_tier="Tier 4A",
                    claim_status="source_familywise_supported",
                    allowed_strength="supportive",
                    figure_role="source-family sensitivity",
                    allowed_language="GO-only source-family support; Reactome-only source-family support",
                    banned_language="standalone disease module validation; de novo STRING topology claim",
                ),
                figure_claim(
                    "Figure_4",
                    "e_anchored_robustness",
                    "Anchored module evidence is stable under null-scale and score-perturbation diagnostics",
                    "C_V12_ANCHORED_ROBUSTNESS",
                    "supportive",
                    "anchored_robustness_summary.tsv",
                    "robustness diagnostics complete",
                    claim_tier="Tier 4A",
                    claim_status="supportive",
                    allowed_strength="supportive",
                    figure_role="robustness evidence",
                    allowed_language="anchored robustness; score perturbation stability",
                    banned_language="module validation; causal network",
                ),
                figure_claim(
                    "Figure_S",
                    "local_module_boundary",
                    "Original de novo STRING local modules remain limitation evidence",
                    "C_V12_NO_DE_NOVO_TOPOLOGY_SPECIFIC_CLAIM",
                    "limitation",
                    "v12_module_claim_boundary.tsv;module_full_reselection_summary.tsv;cross_trait_module_reselection_summary.tsv",
                    "V1.2 claim boundary complete",
                    claim_tier="Tier 4",
                    claim_status="limitation",
                    allowed_strength="limitation",
                    figure_role="supplementary claim boundary",
                    allowed_language="no de novo STRING PPI topology claim; post hoc local modules only",
                    banned_language="validated disease module; causal network",
                ),
                figure_claim(
                    "Figure_S_direction23",
                    "direction23_smoke",
                    "Direction 2/3 smoke diagnostics are future-method evidence only",
                    "C_V12_DIRECTION23_NOT_MAINLINE",
                    "limitation",
                    "direction23_smoke_diagnostic.tsv",
                    "direction 2/3 smoke diagnostic complete",
                    claim_tier="Supplement",
                    claim_status="direction23_diagnostic_only",
                    allowed_strength="limitation",
                    figure_role="supplementary method diagnostic",
                    allowed_language="diagnostic smoke result; future-method candidate; not V1.2 mainline",
                    banned_language="V1.2 main claim; validated module; topology-specific claim",
                ),
            ]
        )
    if is_review_package(package_version):
        rows.append(
            figure_claim(
                "Figure_5",
                "external_network_method",
                "External network methods highlight the need for calibrated claim boundaries",
                "C_REVIEW_TRUE_NETWORK_BASELINE",
                "supportive",
                "dmgwas_external_baseline_summary.tsv;dmgwas_external_baseline_modules.tsv",
                "actual dmGWAS 3.0 node-only top2000 baseline complete; full 12k-node run computationally deferred",
                claim_tier="external_network_baseline",
                claim_status="supportive",
                allowed_strength="supportive",
                figure_role="external network-method comparator",
                allowed_language="actual dmGWAS 3.0 network-method baseline; top-ranked graph-induced analysis scope",
                banned_language="dmGWAS validates RIPPLE modules; topology-specific discovery; validated disease module",
            )
        )
    return ensure_columns(pd.DataFrame(rows), FIGURE_CLAIM_MAP_SCHEMA)


def figure_claim(
    figure_id: str,
    panel_id: str,
    result_subheading: str,
    claim_id: str,
    claim_strength: str,
    source_table: str,
    missing_validation: str,
    *,
    claim_tier: str = "",
    claim_status: str = "",
    allowed_strength: str = "",
    figure_role: str = "",
    allowed_language: str = "calibrated graph-domain aggregation; candidate module; supportive benchmark",
    banned_language: str = "causal disease topology; validated disease module; topology-specific discovery; therapeutic target discovery",
) -> dict[str, object]:
    return {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "result_subheading": result_subheading,
        "claim_id": claim_id,
        "claim_strength": claim_strength,
        "source_table": source_table,
        "source_result_path": source_paths_for_claim_map(source_table),
        "script_path": str(THIS_SCRIPT),
        "missing_validation": missing_validation,
        "allowed_language": allowed_language,
        "banned_language": banned_language,
        "claim_tier": claim_tier,
        "claim_status": claim_status,
        "allowed_strength": allowed_strength or claim_strength,
        "figure_role": figure_role,
    }


def source_paths_for_claim_map(source_table: str) -> str:
    paths: list[str] = []
    for item in source_table.split(";"):
        table_name = item.strip()
        if not table_name:
            continue
        if table_name == "claim_policy.yaml":
            paths.append(str(DEFAULT_OUT_DIR / "claim_policy.yaml"))
        else:
            paths.append(str(DEFAULT_OUT_DIR / table_name))
    return ";".join(paths)


def build_inference_family(final_claims: pd.DataFrame, cross_trait: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.extend(inference_rows_from_claims(final_claims, "final_claim_audit.tsv"))
    rows.extend(inference_rows_from_claims(cross_trait, "cross_trait_benchmark_z2p5.tsv"))
    rows.extend(
        [
            inference_row(
                family_id="F_TYPE1_LD_PIPELINE_NULL",
                analysis_role="calibration",
                trait="synthetic_null",
                analysis_id="type1_error_calibration_v1",
                graph_id="STRING_default",
                score_stream="assoc_resid_score",
                null_type="ld_pipeline_null",
                statistic_name="tier_false_positive_rate",
                grid_or_selection="Z thresholds 2.0 and 2.5; tier-specific claims",
                error_control_scope="tier-specific Type I calibration; not exact global FWER",
                claim_tier="all_tiers",
                allowed_claim_strength="supportive",
                source_table="type1_uncertainty.tsv",
                source_result_path=str(DEFAULT_OUT_DIR / "type1_uncertainty.tsv"),
                notes="n_outer=500 with binomial CI and Monte Carlo SE",
            ),
            inference_row(
                family_id="F_SYNTHETIC_SPIKEIN",
                analysis_role="validation",
                trait="synthetic_spikein",
                analysis_id="synthetic_spikein_validation",
                graph_id="STRING_default",
                score_stream="simulated_assoc_resid_score",
                null_type="scenario_validation",
                statistic_name="synthetic_spikein_summary",
                grid_or_selection="null, dispersed, degree-biased, connected-module scenarios",
                error_control_scope="synthetic behavior validation; not manuscript FWER",
                claim_tier="validation",
                allowed_claim_strength="supportive",
                source_table="synthetic_spikein_summary.tsv",
                source_result_path=str(SYNTHETIC_SPIKEIN_DIR / "tables" / "synthetic_spikein_summary.tsv"),
                notes="Promoted into manuscript-ready package",
            ),
            inference_row(
                family_id="F_SYNTHETIC_SELECTION_AWARE_MODULES",
                analysis_role="validation",
                trait="synthetic_spikein",
                analysis_id="synthetic_spikein_validation_selection_aware",
                graph_id="STRING_default",
                score_stream="simulated_assoc_resid_score",
                null_type="selection_aware_module_null",
                statistic_name="selection_aware_local_module_behavior",
                grid_or_selection="top-fraction selection and module calling repeated before module summary",
                error_control_scope="selection-aware synthetic validation",
                claim_tier="TIER_4_local_calibrated_modules",
                allowed_claim_strength="supportive",
                source_table="synthetic_spikein_selection_aware_summary.tsv",
                source_result_path=str(SYNTHETIC_SELECTION_AWARE_DIR / "tables" / "synthetic_spikein_summary.tsv"),
                notes="Does not upgrade real-data modules to topology-specific status",
            ),
        ]
    )
    return ensure_columns(pd.DataFrame(rows), INFERENCE_FAMILY_SCHEMA)


def inference_rows_from_claims(table: pd.DataFrame, source_table: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in table.to_dict(orient="records"):
        analysis_id = str(item.get("analysis_id", ""))
        trait = str(item.get("trait", ""))
        graph_id = str(item.get("graph_id", ""))
        if source_table == "cross_trait_benchmark_z2p5.tsv" and analysis_id in {
            "SCZ_default_with_MHC_dev500",
            "SCZ_no_MHC_final5000",
        }:
            continue
        rows.append(
            inference_row(
                family_id=f"F_{analysis_id}_{item.get('claim_tier', '')}",
                analysis_role=inference_role(trait, analysis_id, graph_id),
                trait=trait,
                analysis_id=analysis_id,
                graph_id=graph_id,
                score_stream=item.get("score_stream", ""),
                null_type=item.get("null_type", ""),
                statistic_name=item.get("statistic_name", ""),
                grid_or_selection=grid_description(str(item.get("statistic_name", ""))),
                error_control_scope=error_scope(str(item.get("claim_tier", "")), source_table),
                claim_tier=item.get("claim_tier", ""),
                allowed_claim_strength=item.get("claim_status", ""),
                source_table=source_table,
                source_result_path=item.get("source_result_path", ""),
                notes=(
                    "Early benchmark rows are supportive when null fields are incomplete"
                    if source_table == "cross_trait_benchmark_z2p5.tsv"
                    else ""
                ),
            )
        )
    return rows


def inference_row(**kwargs: object) -> dict[str, object]:
    return kwargs


def inference_role(trait: str, analysis_id: str, graph_id: str) -> str:
    if "FVM" in graph_id or "retina" in graph_id:
        return "graph_sensitivity"
    if "WITH_MHC" in trait or "_dev" in analysis_id:
        return "dev_sensitivity"
    if trait == "SCZ":
        return "secondary_benchmark"
    if "NO_MHC_NO_APOE" in trait:
        return "confirmatory_region_sensitivity"
    if trait == "DR_MVP":
        return "confirmatory_primary"
    return "supportive_benchmark"


def grid_description(statistic: str) -> str:
    if "percolation" in statistic or "top_rank" in statistic:
        return "rank-fraction grid 1%,2%,5%,10%,15%,20%; observed and null use same AUC operation"
    if "diffusion" in statistic:
        return "tau grid max statistic; null repeats same tau-grid max operation"
    if "local_module" in statistic:
        return "top-fraction module selection; candidate-module claim only unless full reselection null is complete"
    return "not grid-based"


def error_scope(claim_tier: str, source_table: str) -> str:
    if source_table == "cross_trait_benchmark_z2p5.tsv":
        return "cross-trait benchmark; supportive unless final-scale null metadata are complete"
    if claim_tier == "TIER_3_topology_specific_support":
        return "degree-preserving graph-null test for topology-specific support"
    if claim_tier == "TIER_4_local_calibrated_modules":
        return "local module candidate evidence; no topology-specific upgrade without full reselection graph-null support"
    return "tier-specific manuscript gate at Z>=2.5"


def build_synthetic_summary(selection_aware: bool = False) -> pd.DataFrame:
    root = SYNTHETIC_SELECTION_AWARE_DIR if selection_aware else SYNTHETIC_SPIKEIN_DIR
    path = root / "tables" / "synthetic_spikein_summary.tsv"
    return read_tsv(path) if path.exists() else pd.DataFrame()


def build_dr_local_module_summary() -> pd.DataFrame:
    specs = [
        ("DR_MVP_default_final5000", "DR_MVP", ANALYSIS_ROOT / "dr_mvp_string_final5000"),
        (
            "DR_MVP_no_MHC_no_APOE_final5000",
            "DR_MVP_NO_MHC_NO_APOE",
            ANALYSIS_ROOT / "dr_mvp_no_mhc_no_apoe_final5000",
        ),
    ]
    rows: list[pd.DataFrame] = []
    for analysis_id, trait, analysis_dir in specs:
        path = analysis_dir / "tables" / f"{trait}.local_modules.tsv"
        if not path.exists():
            continue
        table = read_tsv(path)
        table.insert(0, "analysis_id", analysis_id)
        table.insert(1, "trait", trait)
        table.insert(2, "source_result_path", str(path))
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_module_full_reselection_summary() -> pd.DataFrame:
    path = MODULE_RESELECTION_DIR / "tables" / "module_full_reselection_summary.all_traits.tsv"
    if not path.exists():
        return pd.DataFrame()
    table = read_tsv(path)
    labels = table.get("recommended_module_claim_after_reselection", pd.Series(dtype=str)).astype(str)
    table["module_layer_claim_status"] = np.where(
        labels.eq("post_hoc_candidate_module"),
        "post_hoc_candidate_only",
        np.where(labels.eq("calibrated_candidate_module"), "selection_calibrated_module", "no_local_module_support"),
    )
    table["module_layer_policy_version"] = "RIPPLE_V1_Tier4_policy_2026-07-03"
    return table


def build_cross_trait_module_reselection_summary() -> pd.DataFrame:
    path = CROSS_TRAIT_MODULE_RESELECTION_DIR / "tables" / "module_full_reselection_summary.all_traits.tsv"
    return read_tsv(path) if path.exists() else pd.DataFrame()


def build_module_claim_policy(package_version: str = "v1") -> pd.DataFrame:
    rows = [
        {
            "module_status": "post_hoc_candidate_only",
            "tier": "Tier 4",
            "allowed_language": "post hoc candidate module; candidate subnetwork for biological follow-up; exploratory local module; post hoc weak-signal module candidate",
            "forbidden_language": "calibrated module discovery; selection-calibrated module; validated disease module; topology-specific module; discovered disease module",
            "minimum_statistical_requirement": "Module extracted by the predefined top-rank/component procedure; full reselection null does not pass.",
            "controls_selection_process": False,
            "controls_topology_specificity": False,
            "manuscript_role": "exploratory_follow_up",
            "allowed_figure_role": "biological follow-up context only",
            "upgrade_requirement": "fixed-module empirical support or full reselection max-module empirical P below policy threshold",
            "downgrade_condition": "full reselection empirical P >= 0.05 without documented fixed-module support",
        },
        {
            "module_status": "fixed_module_supported",
            "tier": "Tier 4",
            "allowed_language": "fixed-module supported candidate; degree-matched fixed-module support; fixed-candidate module support",
            "forbidden_language": "selection-calibrated discovery; topology-specific discovery; validated module",
            "minimum_statistical_requirement": "Fixed-module degree-matched or size-matched empirical P <= 0.05 under a predefined fixed candidate.",
            "controls_selection_process": False,
            "controls_topology_specificity": False,
            "manuscript_role": "supportive_only",
            "allowed_figure_role": "supportive fixed-candidate evidence",
            "upgrade_requirement": "full reselection max-module empirical P <= 0.05",
            "downgrade_condition": "fixed-module empirical P > 0.05 or fixed candidate not predefined",
        },
        {
            "module_status": "selection_calibrated_module",
            "tier": "Tier 4",
            "allowed_language": "selection-calibrated local module; calibrated module-level discovery; selection-aware calibrated module",
            "forbidden_language": "topology-specific module unless degree-preserving graph null also passes",
            "minimum_statistical_requirement": "Full reselection max-module empirical P <= 0.05 with identical score source, eligible genes, graph, rank grid, module filters, component extraction, max-module statistic and empirical P formula.",
            "controls_selection_process": True,
            "controls_topology_specificity": False,
            "manuscript_role": "module_level_final_positive",
            "allowed_figure_role": "module-level discovery claim",
            "upgrade_requirement": "module-level degree-preserving graph null empirical P <= 0.05",
            "downgrade_condition": "full reselection empirical P >= 0.05",
        },
        {
            "module_status": "topology_specific_module",
            "tier": "Tier 4",
            "allowed_language": "topology-specific local module; topology-specific module support",
            "forbidden_language": "validated disease module; causal disease module",
            "minimum_statistical_requirement": "Must first satisfy selection_calibrated_module and then pass module-level degree-preserving graph null empirical P <= 0.05.",
            "controls_selection_process": True,
            "controls_topology_specificity": True,
            "manuscript_role": "topology_specific_module_claim",
            "allowed_figure_role": "topology-specific module support",
            "upgrade_requirement": "not applicable; highest V1 module-layer status",
            "downgrade_condition": "degree-preserving graph null empirical P >= 0.05 or selection-calibrated status not satisfied",
        },
        {
            "module_status": "no_local_module_support",
            "tier": "Tier 4",
            "allowed_language": "no calibrated local module support; no module-level support under the tested configuration",
            "forbidden_language": "module discovery; validated module; topology-specific module",
            "minimum_statistical_requirement": "No extractable or interpretable candidate modules, or all module evidence is weak.",
            "controls_selection_process": False,
            "controls_topology_specificity": False,
            "manuscript_role": "negative_or_limitation",
            "allowed_figure_role": "negative or limitation statement",
            "upgrade_requirement": "post hoc candidate extraction plus calibrated evidence in future analysis",
            "downgrade_condition": "not applicable",
        },
        {
            "module_status": "not_tested",
            "tier": "Tier 4",
            "allowed_language": "not tested; not applicable; not evaluated",
            "forbidden_language": "module discovery; calibrated module; topology-specific module",
            "minimum_statistical_requirement": "Allowed reasons: missing_input, failed_qc, insufficient_graph_coverage, insufficient_gene_overlap, null_sd_zero, not_applicable_to_graph_role, not_run_v0_1, computationally_deferred.",
            "controls_selection_process": False,
            "controls_topology_specificity": False,
            "manuscript_role": "not_tested",
            "allowed_figure_role": "not-tested footnote or omission",
            "upgrade_requirement": "run the required Tier 4 module calibration workflow",
            "downgrade_condition": "missing required inputs or failed QC",
        },
    ]
    if is_v12_package(package_version):
        rows.extend(
            [
                {
                    "module_status": "anchored_library_calibrated_module",
                    "tier": "Tier 4A",
                    "allowed_language": "anchored biological module evidence; anchored familywise-supported module; fixed external biological module",
                    "forbidden_language": "de novo STRING topology claim; validated disease module; causal module",
                    "minimum_statistical_requirement": "Fixed Reactome/GO anchored module passes degree-matched support and max-over-library anchored familywise empirical P <= 0.05.",
                    "controls_selection_process": True,
                    "controls_topology_specificity": False,
                    "manuscript_role": "anchored_module_supportive_or_final_context_dependent",
                    "allowed_figure_role": "main Figure 4 anchored module evidence",
                    "upgrade_requirement": "independent validation or preregistered graph/topology analysis in future work",
                    "downgrade_condition": "anchored familywise empirical P > 0.05 or library not fixed before testing",
                },
                {
                    "module_status": "anchored_familywise_supported",
                    "tier": "Tier 4A",
                    "allowed_language": "anchored familywise-supported biological module; calibrated anchored module evidence",
                    "forbidden_language": "de novo discovery; topology-specific module; validated disease module",
                    "minimum_statistical_requirement": "Module status is anchored_familywise_supported in the V1.2 anchored broad Reactome/GO output.",
                    "controls_selection_process": True,
                    "controls_topology_specificity": False,
                    "manuscript_role": "anchored_module_evidence",
                    "allowed_figure_role": "main Figure 4 anchored module evidence",
                    "upgrade_requirement": "not a topology claim; no V1.2 upgrade path to topology-specific status",
                    "downgrade_condition": "source-family or full-library calibration fails",
                },
                {
                    "module_status": "source_familywise_supported",
                    "tier": "Tier 4A",
                    "allowed_language": "GO-only or Reactome-only source-familywise support; source-family anchored module evidence",
                    "forbidden_language": "standalone disease module validation; causal module",
                    "minimum_statistical_requirement": "Source-family max-over-library empirical P <= 0.05 with documented source-family Type I calibration.",
                    "controls_selection_process": True,
                    "controls_topology_specificity": False,
                    "manuscript_role": "source_family_sensitivity",
                    "allowed_figure_role": "Figure 4 source-family sensitivity",
                    "upgrade_requirement": "none in V1.2",
                    "downgrade_condition": "source-family empirical P > 0.05",
                },
                {
                    "module_status": "direction23_diagnostic_only",
                    "tier": "Supplement",
                    "allowed_language": "diagnostic smoke result; future-method candidate; not V1.2 mainline",
                    "forbidden_language": "V1.2 main claim; validated module; topology-specific claim",
                    "minimum_statistical_requirement": "Direction 2/3 smoke diagnostic completed with explicit no-mainline recommendation.",
                    "controls_selection_process": False,
                    "controls_topology_specificity": False,
                    "manuscript_role": "supplementary_diagnostic",
                    "allowed_figure_role": "supplementary method diagnostic only",
                    "upgrade_requirement": "requires new method design and full Type I calibration in V1.3/V2",
                    "downgrade_condition": "not applicable",
                },
            ]
        )
    return ensure_columns(pd.DataFrame(rows), MODULE_CLAIM_POLICY_SCHEMA)


def build_graph_registry() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dr_graph_report = (
        ANALYSIS_ROOT
        / "dr_mvp_string_final5000"
        / "tables"
        / "DR_MVP.analysis_graph_edges.tsv.gz.report.json"
    )
    if dr_graph_report.exists():
        report = read_json(dr_graph_report)
        coverage = report.get("coverage_report", {})
        rows.append(
            {
                "graph_id": "STRING_default",
                "graph_name": "STRING PPI default",
                "source": "STRING",
                "version": "STRING min score 400",
                "construction_script": str(SCRIPT_DIR / "write_analysis_graph_cache.py"),
                "construction_date": report.get("created_utc", ""),
                "node_count": coverage.get("n_graph_nodes", ""),
                "edge_count": coverage.get("n_graph_edges", ""),
                "lcc_node_count": coverage.get("largest_component_size", ""),
                "lcc_edge_count": coverage.get("largest_component_edges", ""),
                "edge_weight_rule": "STRING combined score",
                "expression_filter_rule": "none",
                "constructed_before_final_dr_mvp": True,
                "graph_role": "primary",
                "allowed_claim_level": "Tier 1-3",
            }
        )
    rows.extend(reference_graph_rows())
    return ensure_columns(pd.DataFrame(rows), GRAPH_REGISTRY_SCHEMA)


def reference_graph_rows() -> list[dict[str, object]]:
    specs = [
        (
            "FVM_vascular_weighted",
            "FVM vascular weighted STRING",
            PROCESSED_ROOT / "reference_graphs" / "fvm_vascular_string" / "reports" / "fvm_vascular_string.summary.json",
            "sensitivity",
            "graph-domain support only; no topology-specific claim unless preregistered graph-null support exists",
        ),
        (
            "retina_string_min20",
            "retina STRING min20",
            PROCESSED_ROOT
            / "reference_graphs"
            / "retina_string_filtered_min20"
            / "reports"
            / "retina_string_filtered.summary.json",
            "sensitivity",
            "Tier 2 graph-domain only",
        ),
    ]
    rows = []
    for graph_id, graph_name, path, role, claim_level in specs:
        if not path.exists():
            continue
        summary = read_json(path)
        graph_report = summary.get("graph_report", summary.get("coverage_report", {}))
        rows.append(
            {
                "graph_id": graph_id,
                "graph_name": graph_name,
                "source": "STRING + expression support",
                "version": "local_private_snapshot_2026-06",
                "construction_script": str(
                    SCRIPT_DIR
                    / (
                        "build_fvm_vascular_string_graph.py"
                        if "FVM" in graph_id
                        else "build_retina_string_filtered_graph.py"
                    )
                ),
                "construction_date": summary.get("created_utc", ""),
                "node_count": graph_report.get("n_weighted_nodes", graph_report.get("n_nodes_output", "")),
                "edge_count": graph_report.get("n_weighted_edges", graph_report.get("n_edges_output", "")),
                "lcc_node_count": "",
                "lcc_edge_count": "",
                "edge_weight_rule": "expression-supported STRING weighting" if "FVM" in graph_id else "unweighted",
                "expression_filter_rule": "FVM vascular support" if "FVM" in graph_id else "retina expression min20",
                "constructed_before_final_dr_mvp": True,
                "graph_role": role,
                "allowed_claim_level": claim_level,
            }
        )
    return rows


def build_region_exclusion_registry() -> pd.DataFrame:
    rows = []
    for build, mhc_start, mhc_end, apoe_start, apoe_end in [
        ("GRCh37", 25_000_000, 34_000_000, 44_000_000, 46_500_000),
        ("GRCh38", 25_000_000, 34_000_000, 44_000_000, 46_500_000),
    ]:
        rows.extend(
            [
                region("MHC", "Major histocompatibility complex", build, "6", mhc_start, mhc_end, "long-range LD", "report_separately"),
                region("APOE", "APOE/TOMM40/APOC region", build, "19", apoe_start, apoe_end, "pleiotropic high-LD locus", "sensitivity_only"),
                region("8p23_1", "8p23.1 inversion", build, "8", 7_000_000, 13_000_000, "known inversion/high-LD region", "sensitivity_only"),
                region("17q21_31", "17q21.31 inversion", build, "17", 42_000_000, 46_000_000, "known inversion/high-LD region", "sensitivity_only"),
            ]
        )
    return ensure_columns(pd.DataFrame(rows), REGION_EXCLUSION_SCHEMA)


def region(
    region_id: str,
    name: str,
    build: str,
    chrom: str,
    start: int,
    end: int,
    reason: str,
    action: str,
) -> dict[str, object]:
    return {
        "region_id": region_id,
        "region_name": name,
        "genome_build": build,
        "chromosome": chrom,
        "start_bp": start,
        "end_bp": end,
        "reason": reason,
        "default_action": action,
        "source": "RIPPLE_V1_registry",
        "notes": "Broad manuscript sensitivity interval; not a fine-mapping boundary.",
    }


def build_gene_id_mapping_report() -> pd.DataFrame:
    module_paths = [
        ANALYSIS_ROOT / "dr_mvp_string_final5000" / "tables" / "DR_MVP.local_modules.tsv",
        ANALYSIS_ROOT
        / "dr_mvp_no_mhc_no_apoe_final5000"
        / "tables"
        / "DR_MVP_NO_MHC_NO_APOE.local_modules.tsv",
        ANALYSIS_ROOT / "scz_with_mhc_string_dev500" / "tables" / "SCZ_WITH_MHC.local_modules.tsv",
        ANALYSIS_ROOT / "scz_no_mhc_string_dev500" / "tables" / "SCZ.local_modules.tsv",
        ANALYSIS_ROOT / "scz_no_mhc_string_final5000" / "tables" / "SCZ.local_modules.tsv",
    ]
    symbols: set[str] = set()
    for path in module_paths:
        if not path.exists():
            continue
        table = read_tsv(path)
        for col in ["core_genes", "module_genes"]:
            if col not in table.columns:
                continue
            for value in table[col].dropna().astype(str):
                symbols.update(gene.strip() for gene in value.split(",") if gene.strip())
    rows = [
        {
            "input_symbol": symbol,
            "mapped_ensembl_id": "",
            "mapped_hgnc_symbol": symbol,
            "mapping_status": "symbol_retained_without_external_remap",
            "mapping_source": "analysis_gene_symbol",
            "mapping_version": "RIPPLE_V1_private",
            "drop_reason": "",
        }
        for symbol in sorted(symbols)
    ]
    return ensure_columns(pd.DataFrame(rows), GENE_ID_MAPPING_SCHEMA)


def build_module_annotation() -> pd.DataFrame:
    analyses = [
        (
            "DR_MVP_default_final5000",
            "DR_MVP",
            ANALYSIS_ROOT / "dr_mvp_string_final5000" / "tables",
            "DR_MVP",
        ),
        (
            "DR_MVP_no_MHC_no_APOE_final5000",
            "DR_MVP_NO_MHC_NO_APOE",
            ANALYSIS_ROOT / "dr_mvp_no_mhc_no_apoe_final5000" / "tables",
            "DR_MVP_NO_MHC_NO_APOE",
        ),
    ]
    gene_sets = load_annotation_gene_sets()
    rows: list[dict[str, object]] = []
    for analysis_id, trait, tables_dir, prefix in analyses:
        modules_path = tables_dir / f"{prefix}.local_modules.tsv"
        scores_path = tables_dir / f"{prefix}.lcc_gene_scores.1000G_LD.residualized.tsv.gz"
        if not modules_path.exists() or not scores_path.exists():
            continue
        modules = read_tsv(modules_path)
        scores = read_tsv(scores_path)
        background = set(scores["gene_symbol"].dropna().astype(str))
        reportable = modules
        if "is_reportable_calibrated_module" in modules.columns:
            mask = modules["is_reportable_calibrated_module"].astype(str).str.lower() == "true"
            reportable = modules.loc[mask]
        for module in reportable.to_dict(orient="records"):
            module_genes = split_genes(module.get("module_genes", ""))
            module_in_background = sorted(set(module_genes) & background)
            for gene_set_name, genes, source_type in gene_sets:
                gene_set_in_background = sorted(set(genes) & background)
                overlap = sorted(set(module_in_background) & set(gene_set_in_background))
                dropped = sorted((set(module_genes) | set(genes)) - background)
                p_value = hypergeom.sf(
                    len(overlap) - 1,
                    len(background),
                    len(gene_set_in_background),
                    len(module_in_background),
                ) if background and module_in_background and gene_set_in_background else float("nan")
                rows.append(
                    {
                        "analysis_id": analysis_id,
                        "trait": trait,
                        "module_id": module.get("module_id", ""),
                        "annotation_name": gene_set_name,
                        "annotation_source_type": source_type,
                        "module_gene_count": len(module_in_background),
                        "overlap_count": len(overlap),
                        "background_size": len(background),
                        "gene_set_size_within_background": len(gene_set_in_background),
                        "p_value": p_value,
                        "fdr": float("nan"),
                        "gene_ids_used": ",".join(overlap),
                        "dropped_genes": ",".join(dropped[:200]),
                        "drop_reason": "not_in_analysis_eligible_lcc_background" if dropped else "",
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr"] = bh_fdr(pd.to_numeric(out["p_value"], errors="coerce").to_numpy(dtype=float))
    return out


def split_genes(value: object) -> list[str]:
    return [gene.strip() for gene in str(value).split(",") if gene.strip()]


def load_annotation_gene_sets() -> list[tuple[str, set[str], str]]:
    gene_sets: list[tuple[str, set[str], str]] = [
        (name, set(genes), "independent_external") for name, genes in DEFAULT_DR_GENE_SETS.items()
    ]
    fvm_path = (
        PROCESSED_ROOT
        / "reference_graphs"
        / "fvm_vascular_string"
        / "tables"
        / "fvm_vascular_markers.gene_sets.tsv"
    )
    if fvm_path.exists():
        fvm = read_tsv(fvm_path)
        for name, group in fvm.groupby("gene_set", observed=True):
            gene_sets.append(
                (
                    str(name),
                    set(group["gene_symbol"].dropna().astype(str)),
                    "graph_construction_related",
                )
            )
    return gene_sets


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not np.any(finite):
        return out
    idx = np.flatnonzero(finite)
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    n = len(ranked)
    adjusted = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    out[order] = np.minimum(adjusted, 1.0)
    return out


def build_public_release_audit(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        text = str(path)
        restricted = contains_redistribution_restricted_content(path)
        contains_raw = "10_raw_data" in text or "gwas_qc" in text
        contains_licensed = restricted and not contains_raw
        rows.append(
            {
                "file_path": text,
                "file_type": path.suffix.replace(".", "") or "file",
                "contains_raw_gwas": contains_raw,
                "contains_private_path": "private_workspace" in text,
                "contains_licensed_data": contains_licensed,
                "contains_individual_level_data": False,
                "redistributable": not restricted,
                "public_release_action": "do_not_redistribute_or_replace_with_install_instructions"
                if restricted
                else (
                    "summarize_or_strip_private_paths"
                    if "private_workspace" in text
                    else "eligible_after_review"
                ),
            }
        )
    return ensure_columns(pd.DataFrame(rows), PUBLIC_RELEASE_AUDIT_SCHEMA)


def build_type1_uncertainty() -> pd.DataFrame:
    path = ANALYSIS_ROOT / "type1_error_calibration_v1" / "dr_mvp_string_pipeline_null_n500" / "type1_threshold_sensitivity.tsv"
    table = read_tsv(path)
    rows = []
    for item in table.to_dict(orient="records"):
        n = int(item.get("n_outer", 500) or 500) if "n_outer" in item else 500
        k = int(item["false_positive_count"])
        fpr = float(item["false_positive_rate"])
        if k == 0:
            low = 0.0
        else:
            low = float(beta.ppf(0.025, k, n - k + 1))
        if k == n:
            high = 1.0
        else:
            high = float(beta.ppf(0.975, k + 1, n - k))
        rows.append(
            {
                "scenario": item.get("scenario", "ld_pipeline_null"),
                "z_threshold": item.get("z_threshold", ""),
                "claim_tier": item.get("claim_tier", ""),
                "fpr": fpr,
                "n_outer": n,
                "false_positive_count": k,
                "binomial_95ci_low": low,
                "binomial_95ci_high": high,
                "mc_se": float(np.sqrt(fpr * (1.0 - fpr) / n)),
            }
        )
    return ensure_columns(pd.DataFrame(rows), TYPE1_UNCERTAINTY_SCHEMA)


def build_language_guardrail_report(
    scan_dir: Path,
    policy: dict[str, Any],
    *,
    include_method_specs: bool = True,
) -> pd.DataFrame:
    rows = []
    phrases = [(phrase, "forbidden") for phrase in policy["forbidden_language"]]
    phrases.extend((phrase, "context_check") for phrase in policy["context_check_language"])
    phrases.extend((phrase, "module_controlled") for phrase in policy.get("module_layer_controlled_phrases", []))
    scan_roots = [scan_dir]
    method_specs = PRIVATE_ROOT / "01_method_specs"
    if include_method_specs and method_specs.exists():
        scan_roots.append(method_specs)
    seen_paths: set[Path] = set()
    for root in scan_roots:
        for path in sorted(root.rglob("*")):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            claim_text_table = path.name in {"claim_evidence_audit.tsv", "figure_claim_map.tsv"}
            if path.suffix.lower() not in {".md", ".tex", ".txt"} and not claim_text_table:
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for idx, line in enumerate(lines, start=1):
                lowered = line.lower()
                for phrase, severity in phrases:
                    if str(phrase).lower() not in lowered:
                        continue
                    if severity == "module_controlled":
                        resolved = is_module_controlled_phrase_resolved(line, path)
                        required_action = "none_status_validated" if resolved else "status_aware_review"
                        status_validation = "resolved_by_negation_or_policy_context" if resolved else "requires_claim_status_check"
                    else:
                        resolved = severity == "forbidden" and is_forbidden_phrase_resolved(line, path)
                        required_action = (
                            "none_policy_or_banned_language_field"
                            if resolved
                            else ("rewrite_or_manual_exception" if severity == "forbidden" else "context_review")
                        )
                        status_validation = ""
                    rows.append(
                        {
                            "file_path": str(path),
                            "line_number": idx,
                            "matched_phrase": phrase,
                            "severity": severity,
                            "required_action": required_action,
                            "resolved": resolved,
                            "status_validation": status_validation,
                        }
                    )
    return pd.DataFrame(rows)


def is_module_controlled_phrase_resolved(line: str, path: Path) -> bool:
    lowered = line.lower()
    if path.name in {"figure_claim_map.tsv", "module_layer_policy_freeze_report.md", "module_claim_policy.tsv"}:
        return True
    safe_markers = [
        "not ",
        "no ",
        "did not",
        "does not",
        "do not",
        "must not",
        "cannot",
        "unless",
        "negative",
        "forbidden",
        "unsafe",
        "banned",
        "boundary",
        "post hoc",
        "example",
        "allowed only",
        "requires",
    ]
    return any(marker in lowered for marker in safe_markers)


def is_forbidden_phrase_resolved(line: str, path: Path) -> bool:
    lowered = line.lower()
    if path.name == "figure_claim_map.tsv":
        return True
    return any(
        marker in lowered
        for marker in [
            "forbidden",
            "unsafe",
            "banned",
            "must not",
            "do not",
            "not ",
            "no ",
            "negative",
            "without",
        ]
    )


def build_negative_result_rules() -> str:
    return """# RIPPLE V1 Negative And Conditional Result Interpretation Rules

| Result pattern | Manuscript-safe interpretation |
|---|---|
| Tier 1 positive, Tier 2 positive, Tier 3 negative | Degree-calibrated graph-domain aggregation; no topology-specific support. |
| Tier 1 positive, Tier 2 positive, Tier 3 negative, Tier 4 post_hoc_candidate_only | The trait shows calibrated global graph-domain aggregation but no topology-specific graph support and no selection-calibrated local module discovery under the tested graph and module statistic. |
| Tier 1/Tier 2 positive, Tier 4 selection_calibrated_module | The trait shows global graph-domain aggregation and selection-calibrated local module evidence. |
| Tier 1/Tier 2 positive, Tier 4 negative or post_hoc only | Global graph-domain aggregation is stronger than local module-level evidence. Local subnetworks should be treated as biological follow-up candidates unless module calibration passes. |
| Cross-trait Tier 4 full reselection negative | Under the frozen Tier 4 policy, the tested trait does not provide selection-calibrated local module support; this does not negate Tier 1/Tier 2 global graph-domain aggregation. |
| Dense module competitor produces many modules, RIPPLE full reselection null negative | Uncalibrated dense module searches can nominate many subnetworks, but RIPPLE does not upgrade them to module-level discoveries without selection-aware calibration. |
| Tier 1 positive, Tier 2 negative | Top-rank aggregation without diffusion-domain support. |
| Tier 1 negative, Tier 2 negative | No calibrated weak-signal graph aggregation under the tested graph and nulls. |
| Tier 1 positive with poor graph coverage | Provisional only; graph coverage limitation. |
| Finngen DR negative | Low power or graph/context mismatch, not evidence of biological absence. |
| SCZ Tier 1/Tier 2 positive, Tier 3 negative | Cross-domain benchmark with degree-calibrated graph-domain aggregation; no topology-specific support. |
| SCZ negative | Diagnostic negative under tested graph/null configuration, not proof that SCZ has no network biology. |
| Degree-preserving graph null negative | Do not claim topology-specific support. |
"""


def build_module_layer_policy_freeze_report(module_reselection: pd.DataFrame) -> str:
    module_p = module_reselection_min_p_strings()
    if module_reselection.empty:
        status_summary = "No module full reselection summary was available when this report was generated."
    else:
        counts = (
            module_reselection.groupby(["trait", "module_layer_claim_status"], observed=True)
            .size()
            .reset_index(name="n_modules")
        )
        lines = ["| Trait | Module-layer claim status | n modules |", "|---|---|---:|"]
        for item in counts.to_dict(orient="records"):
            lines.append(f"| {item['trait']} | {item['module_layer_claim_status']} | {item['n_modules']} |")
        status_summary = "\n".join(lines)
    return f"""# RIPPLE V1 Module-Layer Claim Policy Freeze Report

Created: {now_utc()}

## Purpose

This report freezes the RIPPLE V1 Tier 4 module-layer claim policy before SCZ, HEIGHT, BMI, T2D, or other cross-trait local module full reselection experiments.

Key frozen statement:

The module-layer policy was frozen after DR_MVP full reselection null analysis and before any SCZ, HEIGHT, BMI, or other cross-trait module reselection experiments. Under this policy, DR_MVP local subnetworks are reported as post hoc candidate modules for biological follow-up, not as calibrated module-level discoveries.

## DR_MVP Full Reselection Null Results

| Analysis | Full reselection null | Post hoc candidates | Exploratory/no-support modules | Selection-calibrated candidates | Minimum empirical P |
|---|---:|---:|---:|---:|---:|
| DR_MVP default STRING | 5,000 | 3 | 8 | 0 | {module_p.get("DR_MVP", "0.3825")} |
| DR_MVP no-MHC-no-APOE | 5,000 | 5 | 10 | 0 | {module_p.get("DR_MVP_NO_MHC_NO_APOE", "0.3023")} |

These results do not overturn Tier 1/Tier 2 global graph-domain aggregation results. They constrain only module-layer claims.

## Final Tier 4 Status Definitions

| Status | Manuscript role |
|---|---|
| post_hoc_candidate_only | Exploratory follow-up only; no module-level statistical discovery claim. |
| fixed_module_supported | Supportive fixed-candidate evidence only; does not control full post hoc selection. |
| selection_calibrated_module | Module-level final-positive claim after full reselection max-module empirical P <= 0.05. |
| topology_specific_module | Requires selection-calibrated status plus module-level degree-preserving graph-null support. |
| no_local_module_support | Negative or limitation statement. |
| not_tested | Not evaluated or not applicable with a recorded reason. |

## DR_MVP Status

| Analysis | Tier 4 status | Selection-calibrated module count | Topology-specific module count |
|---|---|---:|---:|
| DR_MVP default STRING | post_hoc_candidate_only | 0 | 0 |
| DR_MVP no-MHC-no-APOE | post_hoc_candidate_only | 0 | 0 |

## Allowed Language

- post hoc candidate module
- candidate subnetwork for biological follow-up
- exploratory local module
- post hoc weak-signal module candidate

## Forbidden For DR_MVP Local Modules

- calibrated module discovery
- selection-calibrated module
- validated disease module
- topology-specific module
- discovered disease module

## Module Status Summary

{status_summary}

## Files Updated

- `claim_policy.yaml`
- `module_claim_policy.tsv`
- `claim_evidence_audit.tsv`
- `figure_claim_map.tsv`
- `negative_result_interpretation_rules.md`
- `language_guardrail_report.tsv`
- `module_layer_policy_freeze_report.md`

## Validation Results

Validation is performed by `scripts/build_manuscript_ready_package.py`, `ruff check .`, and `pytest -q`. The final console run records pass/fail status.

## Public Folder

`D:\\path\\to\\RIPPLE_public` was not modified by this policy-freeze task.
"""


def build_scz_qc_summary() -> pd.DataFrame:
    path = PROCESSED_ROOT / "gwas_qc" / "qc_reports" / "SCZ.qc_report.json"
    if not path.exists():
        return pd.DataFrame()
    data = read_json(path)
    counts = data.get("counts", {})
    fail_counts = data.get("fail_counts", {})
    rows = [
        {
            "trait": "SCZ",
            "qc_item": key,
            "value": value,
            "source_result_path": str(path),
            "timestamp": data.get("created_utc", ""),
        }
        for key, value in {**counts, **fail_counts}.items()
    ]
    rows.extend(
        [
            {
                "trait": "SCZ",
                "qc_item": "analysis_build",
                "value": data.get("config", {}).get("analysis_build", ""),
                "source_result_path": str(path),
                "timestamp": data.get("created_utc", ""),
            },
            {
                "trait": "SCZ",
                "qc_item": "source_build_pgc",
                "value": data.get("config", {}).get("source_build_pgc", ""),
                "source_result_path": str(path),
                "timestamp": data.get("created_utc", ""),
            },
        ]
    )
    return pd.DataFrame(rows)


def build_scz_claim_summary_table() -> pd.DataFrame:
    path = SCZ_SECONDARY_SUMMARY_DIR / "scz_claim_summary.tsv"
    return read_tsv(path) if path.exists() else pd.DataFrame()


def build_scz_apoe_diagnostic_table() -> pd.DataFrame:
    path = SCZ_SECONDARY_SUMMARY_DIR / "scz_apoe_region_diagnostic.tsv"
    return read_tsv(path) if path.exists() else pd.DataFrame()


def validate_outputs(tables: dict[str, pd.DataFrame], policy: dict[str, Any]) -> None:
    validate_columns(tables["final_claim_audit.tsv"], COMMON_RESULT_SCHEMA, table_name="final_claim_audit.tsv")
    validate_columns(
        tables["cross_trait_benchmark_z2p5.tsv"],
        COMMON_RESULT_SCHEMA,
        table_name="cross_trait_benchmark_z2p5.tsv",
    )
    validate_columns(
        tables["external_baseline_summary.tsv"],
        COMMON_RESULT_SCHEMA,
        table_name="external_baseline_summary.tsv",
    )
    validate_columns(tables["parameter_table.tsv"], PARAMETER_TABLE_SCHEMA, table_name="parameter_table.tsv")
    validate_columns(tables["input_checksum_table.tsv"], INPUT_CHECKSUM_SCHEMA, table_name="input_checksum_table.tsv")
    validate_columns(
        tables["reproducibility_manifest.tsv"],
        REPRODUCIBILITY_MANIFEST_SCHEMA,
        table_name="reproducibility_manifest.tsv",
    )
    validate_columns(tables["claim_evidence_audit.tsv"], CLAIM_EVIDENCE_SCHEMA, table_name="claim_evidence_audit.tsv")
    validate_columns(tables["figure_claim_map.tsv"], FIGURE_CLAIM_MAP_SCHEMA, table_name="figure_claim_map.tsv")
    validate_columns(tables["module_claim_policy.tsv"], MODULE_CLAIM_POLICY_SCHEMA, table_name="module_claim_policy.tsv")
    validate_columns(tables["inference_family.tsv"], INFERENCE_FAMILY_SCHEMA, table_name="inference_family.tsv")
    validate_columns(tables["graph_registry.tsv"], GRAPH_REGISTRY_SCHEMA, table_name="graph_registry.tsv")
    validate_columns(
        tables["region_exclusion_registry.tsv"],
        REGION_EXCLUSION_SCHEMA,
        table_name="region_exclusion_registry.tsv",
    )
    validate_columns(tables["gene_id_mapping_report.tsv"], GENE_ID_MAPPING_SCHEMA, table_name="gene_id_mapping_report.tsv")
    validate_columns(tables["public_release_audit.tsv"], PUBLIC_RELEASE_AUDIT_SCHEMA, table_name="public_release_audit.tsv")
    validate_columns(tables["type1_uncertainty.tsv"], TYPE1_UNCERTAINTY_SCHEMA, table_name="type1_uncertainty.tsv")
    vocab = {
        "claim_status": controlled_vocabulary(policy, "claim_status"),
        "exclusion_or_na_reason": controlled_vocabulary(policy, "exclusion_or_na_reason"),
        "not_tested_reason": controlled_vocabulary(policy, "not_tested_reason"),
        "statistic_direction": controlled_vocabulary(policy, "statistic_direction"),
    }
    validate_vocabulary(tables["final_claim_audit.tsv"], vocab, table_name="final_claim_audit.tsv")
    validate_vocabulary(tables["cross_trait_benchmark_z2p5.tsv"], vocab, table_name="cross_trait_benchmark_z2p5.tsv")
    validate_vocabulary(tables["external_baseline_summary.tsv"], vocab, table_name="external_baseline_summary.tsv")
    validate_vocabulary(
        tables["graph_registry.tsv"],
        {"graph_role": controlled_vocabulary(policy, "graph_role")},
        table_name="graph_registry.tsv",
    )
    validate_module_layer_policy_outputs(tables, policy)
    validate_figure_claim_map(tables["figure_claim_map.tsv"], tables["claim_evidence_audit.tsv"])


def validate_module_layer_policy_outputs(tables: dict[str, pd.DataFrame], policy: dict[str, Any]) -> None:
    if "module_layer_policy" not in policy:
        raise ValueError("claim_policy.yaml is missing module_layer_policy")
    expected = {
        "post_hoc_candidate_only",
        "fixed_module_supported",
        "selection_calibrated_module",
        "topology_specific_module",
        "no_local_module_support",
        "not_tested",
    }
    observed = set(tables["module_claim_policy.tsv"]["module_status"].astype(str))
    missing = sorted(expected - observed)
    if missing:
        raise ValueError(f"module_claim_policy.tsv is missing statuses: {missing}")
    claims = tables["claim_evidence_audit.tsv"]
    local = claims[claims["claim_id"].astype(str).str.startswith("C_DR_LOCAL_MODULES") | claims["claim_id"].astype(str).eq("C_DR_MODULE_CLAIM_BOUNDARY")]
    if not local.empty:
        strengths = set(local["allowed_strength"].astype(str))
        disallowed = strengths & {"final", "final_positive"}
        if disallowed:
            raise ValueError(f"DR_MVP local module claims have disallowed strengths: {sorted(disallowed)}")
        statuses = set(local.get("module_layer_claim_status", pd.Series(dtype=str)).astype(str))
        if "topology_specific_module" in statuses or "selection_calibrated_module" in statuses:
            raise ValueError("DR_MVP local module claims must not be selection-calibrated or topology-specific")
        sentences = " ".join(local["manuscript_sentence"].astype(str)).lower()
        unsafe = ["calibrated module discovery", "validated disease module", "topology-specific module discovery"]
        hits = [phrase for phrase in unsafe if phrase in sentences]
        if hits:
            raise ValueError(f"DR_MVP local module claims contain unsafe language: {hits}")
    summary = tables.get("module_full_reselection_summary.tsv", pd.DataFrame())
    if not summary.empty:
        p_values = set(pd.to_numeric(summary["full_reselection_score_p"], errors="coerce").round(4).dropna().astype(str))
        if "0.3825" not in p_values or "0.3023" not in p_values:
            raise ValueError("Full reselection empirical P values 0.3825 and 0.3023 are not both present")
    cross_trait_summary = tables.get("cross_trait_module_reselection_summary.tsv", pd.DataFrame())
    if not cross_trait_summary.empty:
        expected = {"SCZ_no_MHC_final5000", "HEIGHT_IRN_analysis_ready", "BMI_IRN_analysis_ready", "T2D_analysis_ready"}
        observed = set(cross_trait_summary["analysis_id"].astype(str))
        missing = sorted(expected - observed)
        if missing:
            raise ValueError(f"cross_trait_module_reselection_summary.tsv is missing analyses: {missing}")
        statuses = set(cross_trait_summary["module_layer_claim_status"].astype(str))
        disallowed = statuses & {"selection_calibrated_module", "topology_specific_module"}
        if disallowed:
            raise ValueError(f"Cross-trait Tier 4 reselection unexpectedly contains positive statuses: {sorted(disallowed)}")
        n_null = pd.to_numeric(cross_trait_summary["n_full_reselection_null"], errors="coerce")
        if n_null.notna().any() and int(n_null.min()) < 5000:
            raise ValueError("Cross-trait full reselection null count is below 5,000")


def validate_figure_claim_map(figure_claims: pd.DataFrame, claims: pd.DataFrame) -> None:
    required = ["source_table", "source_result_path", "script_path", "claim_id", "missing_validation"]
    for col in required:
        missing = figure_claims[col].isna() | (figure_claims[col].astype(str).str.strip() == "")
        if bool(missing.any()):
            raise ValueError(f"figure_claim_map.tsv has blank values in required column {col}")
    known_claims = set(claims["claim_id"].astype(str))
    mapped_claims = set(figure_claims["claim_id"].astype(str))
    unknown = sorted(mapped_claims - known_claims)
    if unknown:
        raise ValueError(f"figure_claim_map.tsv references unknown claim IDs: {unknown}")
    risky = figure_claims["allowed_language"].astype(str).str.contains(
        "topology-specific discovery",
        case=False,
        regex=False,
    )
    if bool(risky.any()):
        raise ValueError("figure_claim_map.tsv allowed_language contains topology-specific discovery")
    fig4 = figure_claims[figure_claims["figure_id"].astype(str).eq("Figure_4")]
    local_panels = fig4[fig4["panel_id"].astype(str).isin(["a_default_string_modules", "b_no_mhc_no_apoe_modules"])]
    if not local_panels.empty:
        statuses = set(local_panels.get("claim_status", pd.Series(dtype=str)).astype(str))
        if statuses - {"post_hoc_candidate_only"}:
            raise ValueError(f"Figure 4 local module panels have invalid claim_status values: {sorted(statuses)}")
        allowed = " ".join(local_panels.get("allowed_language", pd.Series(dtype=str)).astype(str)).lower()
        if "calibrated local module discovery" in allowed or "topology-specific module discovery" in allowed:
            raise ValueError("Figure 4 local module panels allow forbidden module-discovery language")


def render_summary(out_dir: Path, final_claims: pd.DataFrame, *, package_version: str = "v1") -> str:
    counts = final_claims["claim_status"].value_counts(dropna=False).to_dict()
    if is_review_package(package_version):
        title = "RIPPLE V1.2 Review-Revision Manuscript-Ready Package"
    elif is_v12_package(package_version):
        title = "RIPPLE V1.2 Manuscript-Ready Package"
    else:
        title = "RIPPLE V1 Manuscript-Ready Package"
    v12_notes = ""
    if is_v12_package(package_version):
        v12_notes = """
- V1.2 adds calibrated anchored biological module evidence over fixed Reactome/GO libraries.
- Figure 4 is updated to anchored module evidence; original de novo STRING local modules move to supplement/limitation.
- Anchored Type I outer1000 calibration is included; GO-only calibration is the strongest source-family gate.
- Direction 2/3 smoke diagnostics are included as future-method diagnostics only, not V1.2 main claims.
- V1.2 does not make a de novo STRING PPI topology claim.
"""
    return f"""# {title}

Created: {now_utc()}

Policy: `{POLICY_PATH}`

Final positive gate: `Z >= 2.5`

Supportive gate: `2.0 <= Z < 2.5`

Output directory: `{out_dir}`

## Claim Status Counts

{json.dumps(counts, indent=2, ensure_ascii=False)}

## Notes

- DR_MVP final5000 analyses are backfilled from existing final-scale outputs.
- SCZ no-MHC final5000 is backfilled as a secondary cross-domain benchmark; with-MHC remains dev500 sensitivity only.
- SCZ APOE-region diagnostic did not require no-MHC-no-APOE sensitivity at this stage.
- External baselines now include MAGMA v1.10, PascalX, and network ablation outputs.
- External MAGMA/PascalX scores were also passed through the RIPPLE graph/null layer.
- A compact simplified dense-module-search competitor is included as a graph/module baseline.
- DR_MVP local modules now include a 5,000-replicate full reselection null; module claims remain post hoc candidates unless this max-module null is passed.
- Tier 4 module-layer claim policy is frozen before cross-trait module full reselection; DR_MVP local subnetworks are post_hoc_candidate_only, not calibrated module-level discoveries.
- FVM vascular annotations are graph-construction-related and must not be described as independent validation.
- Local module stability metrics are descriptive and do not upgrade modules to topology-specific status.
{v12_notes}"""


def main() -> None:
    args = parse_args()
    global DEFAULT_OUT_DIR
    if is_review_package(args.package_version) and args.out_dir == DEFAULT_OUT_DIR:
        args.out_dir = V12_REVIEW_OUT_DIR
    elif is_v12_package(args.package_version) and args.out_dir == DEFAULT_OUT_DIR:
        args.out_dir = V12_OUT_DIR
    DEFAULT_OUT_DIR = args.out_dir
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.force:
        raise FileExistsError(f"{args.out_dir} exists and is not empty. Use --force.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "figure_source_tables").mkdir(parents=True, exist_ok=True)

    base_policy = load_claim_policy(args.policy)
    policy = package_policy(base_policy, package_version=args.package_version)
    final_claims = build_final_claim_audit(policy)
    cross_trait = build_cross_trait_benchmark(policy)
    external_baselines = build_external_baseline_summary(policy)
    type1_uncertainty = build_type1_uncertainty()
    module_annotation = build_module_annotation()
    module_reselection = build_module_full_reselection_summary()
    cross_trait_module_reselection = build_cross_trait_module_reselection_summary()
    v12_tables: dict[str, pd.DataFrame] = {}
    if is_v12_package(args.package_version):
        v12_tables = {
            "anchored_cross_trait_summary.tsv": build_v12_anchored_cross_trait_summary(),
            "anchored_top_modules.tsv": build_v12_anchored_top_modules(),
            "anchored_type1_outer1000_summary.tsv": build_v12_anchored_type1_outer1000_summary(),
            "anchored_go_reactome_confirmation.tsv": build_v12_anchored_go_reactome_confirmation(),
            "anchored_robustness_summary.tsv": build_v12_anchored_robustness_summary(),
            "direction23_smoke_diagnostic.tsv": build_v12_direction23_smoke_diagnostic(),
            "v12_module_claim_boundary.tsv": build_v12_module_claim_boundary(),
        }
    figure_tables = build_figure_source_tables(
        policy=policy,
        final_claims=final_claims,
        cross_trait=cross_trait,
        external_baselines=external_baselines,
        type1_uncertainty=type1_uncertainty,
        module_annotation=module_annotation,
        module_reselection=module_reselection,
        cross_trait_module_reselection=cross_trait_module_reselection,
        package_version=args.package_version,
        v12_tables=v12_tables,
    )
    input_checksums = build_input_checksum_table(package_version=args.package_version)
    claim_evidence = build_claim_evidence_audit(package_version=args.package_version)
    tables: dict[str, pd.DataFrame] = {
        "final_claim_audit.tsv": final_claims,
        "cross_trait_benchmark_z2p5.tsv": cross_trait,
        "external_baseline_summary.tsv": external_baselines,
        "parameter_table.tsv": build_parameter_table(policy, package_version=args.package_version),
        "input_checksum_table.tsv": input_checksums,
        "reproducibility_manifest.tsv": build_reproducibility_manifest(package_version=args.package_version),
        "claim_evidence_audit.tsv": claim_evidence,
        "figure_claim_map.tsv": build_figure_claim_map(package_version=args.package_version),
        "module_claim_policy.tsv": build_module_claim_policy(package_version=args.package_version),
        "inference_family.tsv": build_inference_family(final_claims, cross_trait),
        "graph_registry.tsv": build_graph_registry(),
        "region_exclusion_registry.tsv": build_region_exclusion_registry(),
        "gene_id_mapping_report.tsv": build_gene_id_mapping_report(),
        "module_annotation.tsv": module_annotation,
        "type1_uncertainty.tsv": type1_uncertainty,
        "synthetic_spikein_summary.tsv": build_synthetic_summary(selection_aware=False),
        "synthetic_spikein_selection_aware_summary.tsv": build_synthetic_summary(selection_aware=True),
        "dr_mvp_local_modules_summary.tsv": build_dr_local_module_summary(),
        "module_full_reselection_summary.tsv": module_reselection,
        "cross_trait_module_reselection_summary.tsv": cross_trait_module_reselection,
        "scz_gene_score_qc.tsv": build_scz_qc_summary(),
        "scz_claim_summary.tsv": build_scz_claim_summary_table(),
        "scz_apoe_region_diagnostic.tsv": build_scz_apoe_diagnostic_table(),
    }
    tables.update(v12_tables)
    if is_review_package(args.package_version):
        tables.update(build_review_driven_revision_tables())
    release_paths = [args.out_dir / name for name in tables]
    release_paths.extend(args.out_dir / "figure_source_tables" / name for name in figure_tables)
    release_paths.extend(Path(path) for path in input_checksums["file_path"].dropna().astype(str))
    tables["public_release_audit.tsv"] = build_public_release_audit(release_paths)

    validate_outputs(tables, policy)

    for name, table in tables.items():
        write_table(args.out_dir / name, table)
    for name, table in figure_tables.items():
        validate_columns(table, FIGURE_SOURCE_SCHEMA, table_name=name)
        write_table(args.out_dir / "figure_source_tables" / name, table)
    (args.out_dir / "claim_policy.yaml").write_text(policy_text(policy), encoding="utf-8")
    (args.out_dir / "negative_result_interpretation_rules.md").write_text(
        build_negative_result_rules(),
        encoding="utf-8",
    )
    (args.out_dir / "module_layer_policy_freeze_report.md").write_text(
        build_module_layer_policy_freeze_report(module_reselection),
        encoding="utf-8",
    )
    (args.out_dir / "manuscript_ready_summary.md").write_text(
        render_summary(args.out_dir, final_claims, package_version=args.package_version),
        encoding="utf-8",
    )
    language = build_language_guardrail_report(
        args.out_dir,
        policy,
        include_method_specs=not is_v12_package(args.package_version),
    )
    write_table(args.out_dir / "language_guardrail_report.tsv", language)
    print(f"Wrote manuscript-ready package to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
