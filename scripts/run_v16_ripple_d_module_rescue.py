#!/usr/bin/env python
"""Run RIPPLE-D V1.6 claim-readiness module analysis."""

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

from ripple.modules import (  # noqa: E402
    RippleDConfig,
    add_ripple_d_v16_claim_readiness,
    external_locus_audit_table,
    load_anchored_gene_set_library,
    locus_background_audit_table,
    prepare_locus_inputs,
    ripple_d_module_tests,
)

PRIVATE_ROOT = Path("/mnt/d/RIPPLE/RIPPLE_private")
DEFAULT_OUT_DIR = PRIVATE_ROOT / "30_analysis" / "tier4_v16_claim_readiness_hardening_v0_1"
DEFAULT_GENE_SET_FILE = (
    PRIVATE_ROOT
    / "20_processed_data"
    / "reference_pathways"
    / "anchored_broad_reactome_go_v1"
    / "tables"
    / "anchored_broad_reactome_go_gene_sets.tsv.gz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-file", type=Path, required=True)
    parser.add_argument("--trait", required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--gene-set-file", type=Path, default=DEFAULT_GENE_SET_FILE)
    parser.add_argument("--n-null", type=int, default=1000)
    parser.add_argument("--min-present", type=int, default=5)
    parser.add_argument("--score-cap", type=float, default=3.0)
    parser.add_argument("--locus-window-bp", type=int, default=500_000)
    parser.add_argument("--locus-id-column", type=str, default=None)
    parser.add_argument("--locus-definition-name", type=str, default=None)
    parser.add_argument("--locus-source", default="unspecified")
    parser.add_argument("--locus-source-version", default="unspecified")
    parser.add_argument("--genome-build", default="GRCh37")
    parser.add_argument("--ancestry", default="EUR")
    parser.add_argument("--construction-script", default="unspecified")
    parser.add_argument("--degree-bins", type=int, default=10)
    parser.add_argument("--property-bins", type=int, default=4)
    parser.add_argument("--annotation-sensitivity", action="store_true")
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_out_dir(path: Path, *, force: bool) -> None:
    if path.exists() and any(path.iterdir()) and not force:
        raise FileExistsError(f"{path} exists and is not empty. Use --force to overwrite.")
    path.mkdir(parents=True, exist_ok=True)
    (path / "tables").mkdir(parents=True, exist_ok=True)
    (path / "reports").mkdir(parents=True, exist_ok=True)


def write_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False)


def subset_columns(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return table.loc[:, [column for column in columns if column in table.columns]].copy()


def build_config(args: argparse.Namespace, *, annotation_matching_enabled: bool) -> RippleDConfig:
    return RippleDConfig(
        score_cap=args.score_cap,
        locus_window_bp=args.locus_window_bp,
        locus_id_column=args.locus_id_column,
        locus_definition_name=args.locus_definition_name,
        degree_bins=args.degree_bins,
        property_bins=args.property_bins,
        annotation_matching_enabled=annotation_matching_enabled,
    )


def render_report(trait: str, claim: pd.DataFrame, out_dir: Path) -> str:
    counts = claim["v16_claim_status"].value_counts(dropna=False)
    counts_v16b = (
        claim["v16b_claim_status"].value_counts(dropna=False)
        if "v16b_claim_status" in claim.columns
        else pd.Series(dtype=int)
    )
    lines = [
        f"# RIPPLE-D V1.6 claim-readiness report: {trait}",
        "",
        "V1.6 separates manuscript-ready candidates from high-confidence diagnostic and exploratory locus-distributed evidence.",
        "",
        "## Claim Status Counts: V1.6 Strict",
        "",
        counts.to_string(),
        "",
        "## Claim Status Counts: V1.6b Balanced Null-Matching",
        "",
        counts_v16b.to_string() if not counts_v16b.empty else "Not available.",
        "",
        "## Top Candidate Rows",
        "",
    ]
    top = claim.sort_values(
        ["v16_claim_status", "ripple_d_q_full_library", "module_specific_rank_q_full_library"],
        na_position="last",
    ).head(20)
    if top.empty:
        lines.append("No module rows available.")
    else:
        lines.append(
            top[
                [
                    "module_name",
                    "v16_claim_status",
                    "v16b_claim_status",
                    "v16_downgrade_reason",
                    "v16b_downgrade_reason",
                    "ripple_d_q_full_library",
                    "module_specific_rank_q_full_library",
                    "ripple_d_v16_empirical_p",
                    "n_effective_loci",
                    "top1_locus_contribution",
                    "top5_locus_contribution",
                ]
            ].to_string(index=False)
        )
    lines.extend(
        [
            "",
            "## Output Tables",
            "",
            f"- Claim readiness: `{out_dir / 'tables' / f'{trait}.v16_claim_readiness.tsv'}`",
            f"- Module tests: `{out_dir / 'tables' / f'{trait}.v16_ripple_d_module_tests.tsv'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    ensure_out_dir(args.out_dir, force=args.force)
    tables_dir = args.out_dir / "tables"
    reports_dir = args.out_dir / "reports"
    library = load_anchored_gene_set_library(args.gene_set_file)
    scores = pd.read_csv(args.score_file, sep="\t", compression="infer")

    config = build_config(args, annotation_matching_enabled=True)
    work, background = prepare_locus_inputs(scores, library, config)
    external_audit = external_locus_audit_table(
        work,
        locus_id_column=args.locus_id_column,
        locus_source=args.locus_source,
        locus_source_version=args.locus_source_version,
        genome_build=args.genome_build,
        ancestry=args.ancestry,
        construction_script=args.construction_script,
    )
    modules, locus_audit, nulls, summary = ripple_d_module_tests(
        scores,
        library,
        config=config,
        min_present=args.min_present,
        n_null=args.n_null,
        seed=args.seed,
        return_null_details=False,
        precomputed_work=work,
        precomputed_locus_background=background,
    )

    annotation_free_modules = None
    if args.annotation_sensitivity:
        free_config = build_config(args, annotation_matching_enabled=False)
        free_work, free_background = prepare_locus_inputs(scores, library, free_config)
        annotation_free_modules, _, _, _ = ripple_d_module_tests(
            scores,
            library,
            config=free_config,
            min_present=args.min_present,
            n_null=args.n_null,
            seed=args.seed + 9176,
            return_null_details=False,
            precomputed_work=free_work,
            precomputed_locus_background=free_background,
        )
        write_table(tables_dir / f"{args.trait}.v16_annotation_free_module_tests.tsv", annotation_free_modules)

    claim = add_ripple_d_v16_claim_readiness(
        modules,
        config,
        locus_audit=locus_audit,
        annotation_free_modules=annotation_free_modules,
        external_locus_audit=external_audit,
    )
    write_table(tables_dir / f"{args.trait}.v16_ripple_d_module_tests.tsv", modules)
    write_table(tables_dir / f"{args.trait}.v16_claim_readiness.tsv", claim)
    write_table(tables_dir / f"{args.trait}.v16_locus_contribution_audit.tsv", locus_audit)
    write_table(tables_dir / f"{args.trait}.v16_locus_background_audit.tsv", locus_background_audit_table(background))
    write_table(tables_dir / f"{args.trait}.v16_external_locus_audit.tsv", external_audit)
    write_table(
        tables_dir / f"{args.trait}.v16_null_quality_audit.tsv",
        subset_columns(
            claim,
            [
                "module_id",
                "module_name",
                "null_quality_strict_pass",
                "null_quality_pass",
                "null_quality_balanced_pass",
                "null_exact_match_rate",
                "null_global_fallback_rate",
                "null_reuse_fallback_rate",
                "null_with_replacement_rate",
                "min_match_pool_size",
                "median_match_pool_size",
                "null_loci_with_insufficient_gene_pool_rate",
            ],
        ),
    )
    write_table(
        tables_dir / f"{args.trait}.v16_top_tail_audit.tsv",
        subset_columns(
            claim,
            [
                "module_id",
                "module_name",
                "top_tail_pass",
                "fraction_loci_at_score_cap",
                "fraction_loci_with_uncapped_score_gt_3",
                "fraction_positive_signal_from_uncapped_gt_3",
                "n_loci_in_genome_top_1pct",
                "n_loci_in_genome_top_5pct",
                "moderate_locus_fraction",
            ],
        ),
    )
    write_table(
        tables_dir / f"{args.trait}.v16_leave_topk_audit.tsv",
        subset_columns(
            claim,
            [
                "module_id",
                "module_name",
                "leave_topk_pass",
                "leave_top1_locus_empirical_p",
                "leave_top3_locus_empirical_p",
                "leave_top5_locus_empirical_p",
                "top_locus_conditioned_leave_top1_p",
                "top_locus_conditioned_leave_top3_p",
            ],
        ),
    )
    write_table(
        tables_dir / f"{args.trait}.v16_multiplicity_table.tsv",
        subset_columns(
            claim,
            [
                "module_id",
                "module_name",
                "multiplicity_pass",
                "ripple_d_q_full_library",
                "locus_robust_q_full_library",
                "module_specific_rank_q_full_library",
                "positive_locus_q_full_library",
                "leave_top1_q_full_library",
            ],
        ),
    )
    write_table(
        tables_dir / f"{args.trait}.v16_redundancy_clusters.tsv",
        subset_columns(
            claim,
            [
                "module_id",
                "module_name",
                "module_overlap_cluster_id",
                "representative_module_in_cluster",
                "unique_locus_fraction",
                "redundancy_downgrade_reason",
                "max_gene_jaccard_to_higher_ranked_module",
                "max_locus_jaccard_to_higher_ranked_module",
            ],
        ),
    )
    write_table(
        tables_dir / f"{args.trait}.v16_annotation_sensitivity_summary.tsv",
        subset_columns(
            claim,
            [
                "module_id",
                "module_name",
                "annotation_sensitivity_pass",
                "annotation_sensitivity_status",
                "annotation_sensitivity_balanced_pass",
                "annotation_sensitivity_balanced_status",
                "annotation_free_v16_claim_status",
                "annotation_free_v16b_claim_status",
                "annotation_free_module_status",
            ],
        ),
    )
    if not nulls.empty:
        write_table(tables_dir / f"{args.trait}.v16_locus_nulls.tsv", nulls)

    run_summary = {
        **summary,
        "trait": args.trait,
        "score_file": str(args.score_file),
        "gene_set_file": str(args.gene_set_file),
        "out_dir": str(args.out_dir),
        "script_path": str(Path(__file__).resolve()),
        "seed": int(args.seed),
        "n_null": int(args.n_null),
        "annotation_sensitivity": bool(args.annotation_sensitivity),
        "created_utc": datetime.now(UTC).isoformat(),
        "v16_claim_status_counts": claim["v16_claim_status"].value_counts(dropna=False).to_dict(),
        "v16b_claim_status_counts": claim["v16b_claim_status"].value_counts(dropna=False).to_dict()
        if "v16b_claim_status" in claim.columns
        else {},
    }
    (reports_dir / f"{args.trait}.v16_ripple_d_summary.json").write_text(
        json.dumps(run_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report = render_report(args.trait, claim, args.out_dir)
    (reports_dir / f"{args.trait}.v16_ripple_d_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
