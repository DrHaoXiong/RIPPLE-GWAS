#!/usr/bin/env python
"""Build a public-safe V1.6b failure-mode audit package.

The package contains compact derived result tables and a focused review prompt.
It intentionally excludes raw GWAS, LD matrices, full gene-score files, and
private absolute paths.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

PRIVATE_ROOT = Path("/mnt/d/RIPPLE/RIPPLE_private")
DEFAULT_ANALYSIS_ROOT = PRIVATE_ROOT / "30_analysis"
DEFAULT_OUT_DIR = DEFAULT_ANALYSIS_ROOT / "v16b_failure_mode_audit_package_public"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t", compression="infer")


def write_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False)


def existing_columns(table: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in table.columns]


def compact_high8_specificity() -> pd.DataFrame:
    path = (
        DEFAULT_ANALYSIS_ROOT
        / "tier4_v16_claim_readiness_hardening_v0_1"
        / "v16b_high8_refinement_n5000"
        / "tables"
        / "DR_MVP_V16B_HIGH8_N5000.DR_vs_T2D_BMI_specificity.tsv"
    )
    table = read_table(path)
    columns = [
        "module_name",
        "specificity_class",
        "biological_theme",
        "theme_reason",
        "dr_v16b_claim_status",
        "dr_resid_t2d_bmi_v16b_claim_status",
        "t2d_v16b_claim_status",
        "bmi_v16b_claim_status",
        "dr_ripple_d_v16_z",
        "dr_resid_t2d_bmi_ripple_d_v16_z",
        "t2d_ripple_d_v16_z",
        "bmi_ripple_d_v16_z",
        "dr_ripple_d_v16_empirical_p",
        "dr_resid_t2d_bmi_ripple_d_v16_empirical_p",
        "dr_ripple_d_q_full_library",
        "dr_module_specific_rank_q_full_library",
        "dr_n_loci",
        "dr_n_effective_loci",
        "dr_top1_gene",
        "dr_top1_locus_contribution",
        "dr_top5_locus_contribution",
        "n_extreme_score_ge5",
        "n_upper_tail_score_ge3",
        "n_moderate_score_1_to_3",
        "diabetes_prior_gene_overlap_n",
        "diabetes_prior_gene_overlap",
        "dr_microvascular_prior_gene_overlap_n",
        "dr_microvascular_prior_gene_overlap",
        "top10_module_genes_by_score",
    ]
    out = table.loc[:, existing_columns(table, columns)].copy()
    out.insert(0, "audit_source", "dr_mvp_v16b_high8_n5000_specificity")
    return out


def compact_dr_specific_library() -> pd.DataFrame:
    path = (
        DEFAULT_ANALYSIS_ROOT
        / "dr_specific_library_v1_v16b_n5000"
        / "tables"
        / "dr_specific_library_v1_v16b_cross_context_summary.tsv"
    )
    table = read_table(path)
    columns = [
        "module_name",
        "specificity_class",
        "category",
        "panel_role",
        "description",
        "dr_v16b_claim_status",
        "dr_resid_t2d_bmi_v16b_claim_status",
        "t2d_v16b_claim_status",
        "bmi_v16b_claim_status",
        "dr_ripple_d_v16_z",
        "dr_resid_t2d_bmi_ripple_d_v16_z",
        "t2d_ripple_d_v16_z",
        "bmi_ripple_d_v16_z",
        "dr_ripple_d_v16_empirical_p",
        "dr_resid_t2d_bmi_ripple_d_v16_empirical_p",
        "dr_ripple_d_q_full_library",
        "dr_resid_t2d_bmi_ripple_d_q_full_library",
        "dr_module_specific_rank_q_full_library",
        "dr_resid_t2d_bmi_module_specific_rank_q_full_library",
        "dr_top1_gene",
        "dr_top1_gene_score",
        "dr_n_loci",
        "dr_n_effective_loci",
        "dr_top1_locus_contribution",
        "dr_top5_locus_contribution",
        "dr_v16b_downgrade_reason",
        "dr_resid_t2d_bmi_v16b_downgrade_reason",
    ]
    out = table.loc[:, existing_columns(table, columns)].copy()
    out.insert(0, "audit_source", "dr_specific_library_v1_retinal_only_n5000")
    return out


def compact_type1() -> pd.DataFrame:
    path = (
        DEFAULT_ANALYSIS_ROOT
        / "tier4_v16_claim_readiness_hardening_v0_1"
        / "null_matching_type1_5x2_ann_off_v16b_outer100_null200"
        / "tables"
        / "v16_null_matching_type1_summary.tsv"
    )
    table = read_table(path)
    table.insert(0, "audit_source", "v16b_type1_null_matching_5x2_annotation_off_outer100_null200")
    return table


def compact_synthetic() -> pd.DataFrame:
    path = (
        DEFAULT_ANALYSIS_ROOT
        / "tier4_v16_claim_readiness_hardening_v0_1"
        / "synthetic_nullmatch_5x2_ann_off_v16b_broadguard_n1000"
        / "tables"
        / "v16_synthetic_validation.tsv"
    )
    table = read_table(path)
    columns = [
        "scenario",
        "expected_min_tier",
        "oracle_v16_claim_status",
        "oracle_v16b_claim_status",
        "behavior_passed",
        "v16b_behavior_passed",
        "n_loci",
        "n_effective_loci",
        "top1_locus_contribution",
        "top5_locus_contribution",
        "ripple_d_v16_z",
        "ripple_d_v16_empirical_p",
        "ripple_d_q_full_library",
        "module_specific_rank_q_full_library",
        "leave_top1_locus_empirical_p",
        "leave_top3_locus_empirical_p",
        "top_locus_conditioned_leave_top1_p",
        "top_locus_conditioned_leave_top3_p",
        "null_quality_balanced_pass",
        "multiplicity_pass",
        "top_tail_pass",
        "leave_topk_pass",
        "v16b_broad_module_pass",
        "v16b_downgrade_reason",
        "n_null",
        "degree_bins",
        "property_bins",
        "annotation_matching_enabled",
        "seed",
    ]
    out = table.loc[:, existing_columns(table, columns)].copy()
    out.insert(0, "audit_source", "v16b_synthetic_broadguard_n1000")
    return out


def compact_sensitivity_completion() -> pd.DataFrame:
    path = (
        DEFAULT_ANALYSIS_ROOT
        / "tier4_v16_claim_readiness_hardening_v0_1"
        / "v16b_candidate_sensitivity_completion_v0_1"
        / "run"
        / "tables"
        / "DR_MVP_V16B_HIGH33.v16b_sensitivity_completion_summary.tsv"
    )
    table = read_table(path)
    columns = [
        "module_name",
        "base_v16b_claim_status",
        "base_ripple_d_q_full_library",
        "base_module_specific_rank_q_full_library",
        "base_ripple_d_v16_empirical_p",
        "base_null_quality_balanced_pass",
        "base_multiplicity_pass",
        "base_top_tail_pass",
        "base_leave_topk_pass",
        "base_n_loci",
        "base_n_effective_loci",
        "base_top1_locus_contribution",
        "base_top5_locus_contribution",
        "annotation_on_v16b_claim_status",
        "annotation_sensitivity_balanced_pass",
        "annotation_sensitivity_balanced_status",
        "pseudo_window_tested_n",
        "pseudo_window_pass_n",
        "pseudo_window_high_confidence_n",
        "pseudo_window_stability_balanced_pass",
        "v16b_sensitivity_completed_status",
    ]
    out = table.loc[:, existing_columns(table, columns)].copy()
    out.insert(0, "audit_source", "dr_mvp_v16b_high33_sensitivity_completion")
    return out


def build_manifest(generated_at: str) -> pd.DataFrame:
    rows = [
        {
            "table_file": "tables/dr_mvp_v16b_high8_specificity_compact.tsv",
            "description": "Eight V1.6b sensitivity-completed candidates refined at n_null=5000 and contrasted against T2D/BMI and DR residualized against T2D/BMI.",
            "contains_raw_gwas": "false",
            "contains_gene_scores": "false",
            "contains_private_paths": "false",
            "generated_at_utc": generated_at,
        },
        {
            "table_file": "tables/dr_specific_library_v1_cross_context_compact.tsv",
            "description": "Retinal-only DR-specific fixed-library V1.6b n5000 results across DR_MVP, DR residualized, T2D and BMI.",
            "contains_raw_gwas": "false",
            "contains_gene_scores": "false",
            "contains_private_paths": "false",
            "generated_at_utc": generated_at,
        },
        {
            "table_file": "tables/v16b_synthetic_validation_compact.tsv",
            "description": "Synthetic V1.6b stress scenarios for top-locus artifacts, distributed signals, broad modules and sparse retinal modules.",
            "contains_raw_gwas": "false",
            "contains_gene_scores": "false",
            "contains_private_paths": "false",
            "generated_at_utc": generated_at,
        },
        {
            "table_file": "tables/v16b_type1_calibration_compact.tsv",
            "description": "Diagnostic Type I calibration for V1.6b null matching, outer=100 and n_null=200.",
            "contains_raw_gwas": "false",
            "contains_gene_scores": "false",
            "contains_private_paths": "false",
            "generated_at_utc": generated_at,
        },
        {
            "table_file": "tables/v16b_sensitivity_completion_compact.tsv",
            "description": "Annotation and pseudo-window sensitivity completion for 33 V1.6b high-confidence candidates.",
            "contains_raw_gwas": "false",
            "contains_gene_scores": "false",
            "contains_private_paths": "false",
            "generated_at_utc": generated_at,
        },
    ]
    return pd.DataFrame(rows)


def build_questions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "priority": "P0",
                "review_area": "statistic_target",
                "question": "Does RIPPLE-D V1.6b still optimize a burden-like pathway score rather than a genuine distributed weak-signal module objective?",
                "code_focus": "ripple/modules/distributed.py",
            },
            {
                "priority": "P0",
                "review_area": "locus_collapse",
                "question": "Does max(capped gene score) per locus discard useful multi-gene moderate signals or create winner-take-all behavior?",
                "code_focus": "prepare_locus_inputs, locus collapse utilities, contribution diagnostics",
            },
            {
                "priority": "P0",
                "review_area": "null_matching",
                "question": "Do degree/property/locus matched nulls overmatch real biology or create hidden false-negative pressure?",
                "code_focus": "locus matched null generation and V1.6b balanced null-quality gates",
            },
            {
                "priority": "P0",
                "review_area": "synthetic_real_gap",
                "question": "Why does V1.6b recover distributed synthetic signals but not DR-specific retinal modules in real DR_MVP?",
                "code_focus": "scripts/run_v16_ripple_d_synthetic_validation.py and real-data V1.6b workflows",
            },
            {
                "priority": "P1",
                "review_area": "claim_policy",
                "question": "Are top-tail, leave-top-k, multiplicity and pseudo-window gates calibrated as discovery gates or are they too defensive for weak-signal biology?",
                "code_focus": "add_ripple_d_v16_claim_readiness",
            },
            {
                "priority": "P1",
                "review_area": "module_definition",
                "question": "Should binary curated gene sets be replaced or supplemented by weighted cell-type/expression-supported modules?",
                "code_focus": "DR-specific library construction and module membership handling",
            },
            {
                "priority": "P1",
                "review_area": "trait_specificity",
                "question": "Does residualizing DR against T2D/BMI remove confounding only, or also remove real DR biology mediated by diabetes pathways?",
                "code_focus": "DR vs T2D/BMI specificity and residualized score workflow",
            },
        ]
    )


def write_readme(path: Path) -> None:
    text = """# RIPPLE-D V1.6b Failure-Mode Audit Package

This package is designed for focused external review of why RIPPLE-D V1.6b has not yet recovered a strong DR-specific weak-signal module narrative.

## What Is Included

- Compact derived tables only.
- No raw GWAS summary statistics.
- No LD matrices or genotype reference files.
- No full gene-score tables.
- No private absolute file paths.

## Current Empirical Pattern

1. V1.6b improves the false-negative problem relative to strict V1.6.
2. Synthetic distributed weak-signal scenarios are recovered under V1.6b.
3. DR_MVP broad GO/Reactome candidates contain some multi-locus evidence, but many are generic cell-cycle, senescence, mitochondrial, or lipid modules.
4. A prespecified retinal-only DR-specific library does not produce high-confidence V1.6b DR modules at n_null=5000.
5. The key unresolved question is whether this reflects DR_MVP biology/power, module definition, statistic design, null calibration, or implementation error.

## Files

- `tables/dr_mvp_v16b_high8_specificity_compact.tsv`
- `tables/dr_specific_library_v1_cross_context_compact.tsv`
- `tables/v16b_synthetic_validation_compact.tsv`
- `tables/v16b_type1_calibration_compact.tsv`
- `tables/v16b_sensitivity_completion_compact.tsv`
- `tables/v16b_failure_mode_questions.tsv`
- `tables/v16b_failure_mode_manifest.tsv`
- `AI_REVIEW_PROMPT.md`

## Review Goal

Do not perform a generic code review. The target is to identify mathematical or implementation choices that prevent RIPPLE-D from detecting realistic distributed weak-signal modules in real GWAS data despite synthetic recovery.
"""
    path.write_text(text, encoding="utf-8")


def write_prompt(path: Path) -> None:
    text = """# AI Review Prompt: RIPPLE-D V1.6b Failure-Mode Audit

You are reviewing the RIPPLE-GWAS public repository, focusing on RIPPLE-D V1.6b. Your task is not to give a generic code review. Your task is to identify what mathematical, statistical, or implementation choices may be preventing RIPPLE-D from supporting a strong distributed weak-signal module narrative in real DR GWAS data.

## Repository Context

RIPPLE-GWAS is intended to test graph-domain aggregation and module-level distributed weak-signal patterns from GWAS summary statistics. The V1.6b layer attempts to rescue weak-signal module detection using locus-aware score capping, locus collapse, contribution dispersion diagnostics, leave-top-locus tests, matched locus nulls, multiplicity correction, annotation sensitivity, and pseudo-window/external locus sensitivity.

## Empirical Problem

Observed pattern:

1. Synthetic distributed 8-locus and 15-locus scenarios pass V1.6b high-confidence behavior.
2. Top-locus artifacts are mostly blocked or downgraded.
3. DR_MVP broad GO/Reactome candidates produce some V1.6b high-confidence signals after sensitivity completion.
4. However, many broad candidates are generic cell-cycle/senescence/mitochondrial/lipid modules.
5. A prespecified retinal-only DR-specific library produces no V1.6b high-confidence DR module at n_null=5000.
6. The best DR-specific library signals are only exploratory: oxidative stress/mitochondrial injury and ECM remodeling.

The central question is:

Why can V1.6b recover distributed synthetic signal but fail to produce strong DR-specific weak-signal module evidence in real DR_MVP?

## Code Areas To Audit

Prioritize:

- `ripple/modules/distributed.py`
- `scripts/run_v16_ripple_d_module_rescue.py`
- `scripts/run_v16_ripple_d_synthetic_validation.py`
- `scripts/run_v16_null_matching_sensitivity.py`
- `scripts/run_v16_null_matching_type1_calibration.py`
- `scripts/run_v16b_candidate_sensitivity_completion.py`
- `scripts/build_dr_specific_library_v1.py`
- `scripts/summarize_dr_specific_library_v1_v16b.py`
- `scripts/summarize_v16b_high8_specificity.py`

## Specific Questions

1. Does `T_V1.6` still behave like a pathway burden statistic rather than a distributed weak-signal module statistic?
2. Does using `max(capped gene score)` per locus discard true moderate polygenic signal within loci?
3. Are top1/top5 penalties, effective-loci penalties, leave-top-k gates, and top-conditioned nulls jointly over-conservative?
4. Does the matched-locus null overmatch biology by conditioning on degree/property/annotation/locus structure too strongly?
5. Is full-library BH-FDR the right multiplicity correction for a structured gene-set hierarchy, or should source-family/hierarchical FDR or max-null be used?
6. Are synthetic scenarios too idealized compared with real pathways that contain many passenger genes and sparse causal loci?
7. Does residualizing DR against T2D/BMI remove real DR biology as well as confounding?
8. Are binary gene-set memberships too crude, and should RIPPLE-D use weighted retinal/FVM/cell-type module membership?
9. Are pseudo-window and external LD-block sensitivity tests aligned with the observed module definition, or do they introduce an inconsistent gate?
10. Is there any implementation bug in empirical P values, plus-one formula, directionality, q-value denominator, null reuse, replacement, or annotation matching?

## Expected Output

Please return:

1. A ranked list of likely blockers.
2. For each blocker, state whether it is a mathematical design issue, statistical calibration issue, implementation bug, or data/phenotype limitation.
3. Point to exact code functions or scripts.
4. Suggest minimal diagnostic experiments to distinguish false-negative overcorrection from true lack of DR-specific signal.
5. Recommend V1.7 changes only if justified by the evidence.

Avoid overclaiming. The goal is to find the bottleneck, not to force a positive result.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.force:
        raise FileExistsError(f"{args.out_dir} exists and is not empty. Use --force.")
    tables_dir = args.out_dir / "tables"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    generated_at = now_utc()

    write_table(tables_dir / "dr_mvp_v16b_high8_specificity_compact.tsv", compact_high8_specificity())
    write_table(tables_dir / "dr_specific_library_v1_cross_context_compact.tsv", compact_dr_specific_library())
    write_table(tables_dir / "v16b_type1_calibration_compact.tsv", compact_type1())
    write_table(tables_dir / "v16b_synthetic_validation_compact.tsv", compact_synthetic())
    write_table(tables_dir / "v16b_sensitivity_completion_compact.tsv", compact_sensitivity_completion())
    write_table(tables_dir / "v16b_failure_mode_manifest.tsv", build_manifest(generated_at))
    write_table(tables_dir / "v16b_failure_mode_questions.tsv", build_questions())
    write_readme(args.out_dir / "README.md")
    write_prompt(args.out_dir / "AI_REVIEW_PROMPT.md")
    print(f"Wrote V1.6b failure-mode audit package to {args.out_dir}")


if __name__ == "__main__":
    main()
