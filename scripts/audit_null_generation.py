#!/usr/bin/env python
"""Audit what the RIPPLE-GWAS null pipeline preserves for manuscript review."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PRIVATE_ROOT = (
    Path("D:/path/to/ripple_private_workspace")
    if Path("D:/path/to/ripple_private_workspace").exists()
    else Path("/path/to/ripple_private_workspace")
)
MANUSCRIPT_ROOT = (
    Path("D:/RIPPLE/RIPPLE_manuscript")
    if Path("D:/RIPPLE/RIPPLE_manuscript").exists()
    else Path("/path/to/ripple_manuscript_workspace")
)
ANALYSIS_ROOT = PRIVATE_ROOT / "30_analysis"
DEFAULT_OUT_DIR = ANALYSIS_ROOT / "review_driven_revision_v1" / "null_generation_audit"
DEFAULT_SUPPLEMENT_DIR = MANUSCRIPT_ROOT / "supplementary_files"


DEFAULT_ANALYSES = {
    "DR_MVP": ANALYSIS_ROOT / "dr_mvp_string_final5000",
    "DR_MVP_NO_MHC_NO_APOE": ANALYSIS_ROOT / "dr_mvp_no_mhc_no_apoe_final5000",
    "SCZ": ANALYSIS_ROOT / "scz_no_mhc_string_final5000",
    "HEIGHT_IRN": ANALYSIS_ROOT / "height_irn_analysis_ready",
    "BMI_IRN": ANALYSIS_ROOT / "bmi_irn_analysis_ready",
    "T2D": ANALYSIS_ROOT / "t2d_analysis_ready",
    "FINNGEN_DR": ANALYSIS_ROOT / "dm_retinopathy_exmore_analysis_ready",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--supplement-dir", type=Path, default=DEFAULT_SUPPLEMENT_DIR)
    parser.add_argument("--analysis", action="append", default=[])
    parser.add_argument("--copy-to-supplement", action="store_true", default=True)
    return parser.parse_args()


def read_summary(analysis_dir: Path, trait: str) -> dict[str, Any]:
    candidates = sorted((analysis_dir / "reports").glob(f"{trait}*.analysis_ready_summary.json"))
    if not candidates:
        candidates = sorted((analysis_dir / "reports").glob("*.analysis_ready_summary.json"))
    if not candidates:
        return {}
    return json.loads(candidates[0].read_text(encoding="utf-8"))


def compact_json(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def analysis_rows(trait: str, analysis_dir: Path) -> list[dict[str, Any]]:
    summary = read_summary(analysis_dir, trait)
    outputs = summary.get("outputs", {}) if isinstance(summary, dict) else {}
    score_report = summary.get("score_report", {}) if isinstance(summary, dict) else {}
    coverage = summary.get("graph_coverage_report", {}) if isinstance(summary, dict) else {}
    return [
        {
            "trait": trait,
            "analysis_dir": str(analysis_dir),
            "audit_item": "null_replicate_identity",
            "preservation_status": "preserved",
            "evidence": (
                "Observed and SNP-pipeline null scores use compute_ld_cached_gene_scores; "
                "downstream percolation reranks each null vector and repeats the rank-fraction AUC."
            ),
            "observed_metrics": compact_json(
                {
                    "n_null": summary.get("n_null"),
                    "null_assoc_scores": outputs.get("null_assoc_scores"),
                    "null_residualized_scores": outputs.get("null_residualized_scores"),
                }
            ),
            "reviewer_risk": "low_if_summary_paths_exist",
            "manuscript_language": "Null replicates repeated LD-aware scoring, normal-score transformation, residualization, graph filtering and graph statistics.",
            "action_required": "none_if_output_files_exist",
        },
        {
            "trait": trait,
            "analysis_dir": str(analysis_dir),
            "audit_item": "local_ld_structure",
            "preservation_status": "preserved_with_reference_ld",
            "evidence": "Each gene statistic is calibrated using its 1000G EUR per-gene LD cache and the same LD shrinkage as observed.",
            "observed_metrics": compact_json(
                {
                    "ld_shrinkage": summary.get("ld_shrinkage"),
                    "ld_status_counts": score_report.get("ld_status_counts"),
                    "ld_cache_dirs": summary.get("ld_cache_dirs_in_priority_order"),
                }
            ),
            "reviewer_risk": "moderate_for_identity_fallback_or_ancestry_mismatch",
            "manuscript_language": "The null preserves local reference-LD calibration within each mapped gene but does not model trait-specific LD beyond the reference panel.",
            "action_required": "report_identity_fallback_counts_and_ancestry_reference",
        },
        {
            "trait": trait,
            "analysis_dir": str(analysis_dir),
            "audit_item": "snp_to_gene_mapping_overlap",
            "preservation_status": "partly_preserved",
            "evidence": "The observed mapping and weights are reused, but null SNP Z values are sampled into mapped SNP slots; this preserves per-gene SNP count and weights but not all cross-gene SNP-sharing correlations.",
            "observed_metrics": compact_json({"mapping": summary.get("mapping")}),
            "reviewer_risk": "moderate",
            "manuscript_language": (
                "The primary null preserves the analysis mapping schema and per-gene mapped SNP burden; "
                "overlapping-gene correlation is approximated rather than exactly preserved. A separate "
                "overlap-preserving SNP-label permutation proxy sensitivity is reported for primary review traits."
            ),
            "action_required": "state_as_limitation_and_cross_reference_overlap_preserving_sensitivity",
        },
        {
            "trait": trait,
            "analysis_dir": str(analysis_dir),
            "audit_item": "gene_size_snp_density_covariates",
            "preservation_status": "preserved_by_reuse_and_residualization",
            "evidence": "Gene-level technical covariates are fixed and null-estimated residualization uses the same gene table.",
            "observed_metrics": compact_json({"residualization_method": summary.get("residualization_method")}),
            "reviewer_risk": "low",
            "manuscript_language": "Gene length, mapped SNP count, M_eff, local LD score and mappability are fixed gene-level covariates in observed and null analyses.",
            "action_required": "none",
        },
        {
            "trait": trait,
            "analysis_dir": str(analysis_dir),
            "audit_item": "graph_covered_gene_universe",
            "preservation_status": "preserved",
            "evidence": "Observed and null graph statistics are computed after alignment to the same score-covered graph largest connected component.",
            "observed_metrics": compact_json(
                {
                    "n_lcc_scored_genes": summary.get("n_lcc_scored_genes"),
                    "graph_covered_gene_fraction": coverage.get("graph_covered_gene_fraction"),
                    "largest_component_gene_fraction": coverage.get("largest_component_gene_fraction"),
                }
            ),
            "reviewer_risk": "low_if_coverage_reported",
            "manuscript_language": "All graph nulls and observed graph statistics use the same score-available graph universe.",
            "action_required": "report_graph_coverage",
        },
        {
            "trait": trait,
            "analysis_dir": str(analysis_dir),
            "audit_item": "special_region_exclusions",
            "preservation_status": "preserved_by_analysis_input",
            "evidence": "With-MHC, no-MHC and no-MHC-no-APOE analyses use separate QC inputs before gene scoring.",
            "observed_metrics": compact_json({"gwas": summary.get("gwas")}),
            "reviewer_risk": "low_if_region_registry_reported",
            "manuscript_language": "Region exclusions are applied before mapping and scoring in sensitivity analyses.",
            "action_required": "cross_reference_region_exclusion_registry",
        },
        {
            "trait": trait,
            "analysis_dir": str(analysis_dir),
            "audit_item": "sample_size_heterogeneity",
            "preservation_status": "not_explicitly_modeled",
            "evidence": "The current SNP-pipeline null samples observed Z statistics and does not explicitly resimulate per-SNP effective sample size heterogeneity.",
            "observed_metrics": compact_json({"mode": summary.get("mode")}),
            "reviewer_risk": "moderate",
            "manuscript_language": "Trait-specific sample-size heterogeneity is inherited through observed Z-score sampling but is not separately parameterized in the null.",
            "action_required": "state_as_limitation",
        },
    ]


def selected_analyses(requested: list[str]) -> dict[str, Path]:
    if not requested:
        return DEFAULT_ANALYSES
    out = {}
    for name in requested:
        if name not in DEFAULT_ANALYSES:
            raise ValueError(f"Unknown analysis {name}; available: {sorted(DEFAULT_ANALYSES)}")
        out[name] = DEFAULT_ANALYSES[name]
    return out


def write_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False)


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for trait, analysis_dir in selected_analyses(args.analysis).items():
        if analysis_dir.exists():
            rows.extend(analysis_rows(trait, analysis_dir))
    table = pd.DataFrame(rows)
    table["script_path"] = str(Path(__file__).resolve())
    table["timestamp"] = datetime.now(UTC).isoformat()
    out_path = args.out_dir / "null_generation_audit.tsv"
    write_table(out_path, table)
    if args.copy_to_supplement:
        write_table(args.supplement_dir / "null_generation_audit.tsv", table)
    print(f"Wrote null generation audit to {out_path}", flush=True)


if __name__ == "__main__":
    main()
