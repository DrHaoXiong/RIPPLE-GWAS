#!/usr/bin/env python
"""Run RIPPLE-D V1.6 null-matching sensitivity diagnostics.

This script is intentionally diagnostic. It varies the locus-null matching
granularity while keeping the same score table and anchored module library.
The goal is to identify whether V1.6 modules are blocked by biological signal
failure or by over-sparse exact matching bins.
"""

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
DEFAULT_OUT_DIR = (
    PRIVATE_ROOT
    / "30_analysis"
    / "tier4_v16_claim_readiness_hardening_v0_1"
    / "null_matching_sensitivity_v0_1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-file", type=Path, required=True)
    parser.add_argument("--gene-set-file", type=Path, required=True)
    parser.add_argument("--trait", required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-null", type=int, default=1000)
    parser.add_argument("--min-present", type=int, default=5)
    parser.add_argument("--score-cap", type=float, default=3.0)
    parser.add_argument("--locus-id-column", type=str, default=None)
    parser.add_argument("--locus-definition-name", type=str, default=None)
    parser.add_argument("--locus-source", default="unspecified")
    parser.add_argument("--locus-source-version", default="unspecified")
    parser.add_argument("--genome-build", default="GRCh37")
    parser.add_argument("--ancestry", default="EUR")
    parser.add_argument("--construction-script", default="unspecified")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help=(
            "Sensitivity config as name:degree_bins:property_bins:annotation_matching. "
            "annotation_matching must be on/off. May be repeated."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_config_specs(specs: list[str]) -> list[dict[str, object]]:
    if not specs:
        specs = [
            "default_10x4_ann_on:10:4:on",
            "default_10x4_ann_off:10:4:off",
            "coarse_5x3_ann_on:5:3:on",
            "coarse_5x2_ann_on:5:2:on",
            "coarse_5x2_ann_off:5:2:off",
        ]
    parsed: list[dict[str, object]] = []
    for spec in specs:
        parts = spec.split(":")
        if len(parts) != 4:
            raise ValueError(f"Invalid --config {spec!r}; expected name:degree_bins:property_bins:on/off")
        name, degree_bins, property_bins, annotation = parts
        annotation_lower = annotation.strip().lower()
        if annotation_lower not in {"on", "off", "true", "false", "1", "0"}:
            raise ValueError(f"Invalid annotation flag in --config {spec!r}")
        parsed.append(
            {
                "config_name": name,
                "degree_bins": int(degree_bins),
                "property_bins": int(property_bins),
                "annotation_matching_enabled": annotation_lower in {"on", "true", "1"},
            }
        )
    return parsed


def ensure_out_dir(path: Path, *, force: bool) -> None:
    if path.exists() and any(path.iterdir()) and not force:
        raise FileExistsError(f"{path} exists and is not empty. Use --force to overwrite.")
    path.mkdir(parents=True, exist_ok=True)
    (path / "tables").mkdir(parents=True, exist_ok=True)
    (path / "reports").mkdir(parents=True, exist_ok=True)


def write_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False)


def build_config(args: argparse.Namespace, spec: dict[str, object]) -> RippleDConfig:
    return RippleDConfig(
        score_cap=args.score_cap,
        locus_id_column=args.locus_id_column,
        locus_definition_name=args.locus_definition_name,
        degree_bins=int(spec["degree_bins"]),
        property_bins=int(spec["property_bins"]),
        annotation_matching_enabled=bool(spec["annotation_matching_enabled"]),
    )


def match_bin_summary(locus_background: pd.DataFrame, config_name: str) -> pd.DataFrame:
    sizes = locus_background.groupby("match_bin", observed=True).size()
    mapped_sizes = locus_background["match_bin"].map(sizes)
    return pd.DataFrame(
        [
            {
                "config_name": config_name,
                "n_loci": int(len(locus_background)),
                "n_match_bins": int(sizes.size),
                "median_match_bin_size": float(sizes.median()),
                "p10_match_bin_size": float(sizes.quantile(0.10)),
                "p25_match_bin_size": float(sizes.quantile(0.25)),
                "p75_match_bin_size": float(sizes.quantile(0.75)),
                "max_match_bin_size": int(sizes.max()),
                "fraction_bins_lt20": float((sizes < 20).mean()),
                "fraction_loci_in_bins_lt20": float(mapped_sizes.lt(20).mean()),
            }
        ]
    )


def summarize_claim_table(claim: pd.DataFrame, config_name: str) -> dict[str, object]:
    nonnegative = claim["v16_claim_status"].astype(str).isin(
        {
            "exploratory_locus_distributed_candidate",
            "multi_strong_locus_pathway_overlap",
            "top_locus_dominant",
            "raw_enrichment_only",
            "high_confidence_diagnostic_candidate",
            "manuscript_ready_distributed_candidate",
        }
    )
    distributed = claim["module_status"].astype(str).isin(
        {
            "distributed_weak_signal_module_candidate",
            "mixed_sparse_distributed_candidate",
            "moderate_locus_supported_module",
            "module_specific_rank_supported_module",
        }
    )
    row: dict[str, object] = {
        "config_name": config_name,
        "n_modules": int(len(claim)),
        "n_nonnegative": int(nonnegative.sum()),
        "n_distributed_evidence": int(distributed.sum()),
        "n_manuscript_ready": int(claim["v16_claim_status"].eq("manuscript_ready_distributed_candidate").sum()),
        "n_high_confidence": int(claim["v16_claim_status"].eq("high_confidence_diagnostic_candidate").sum()),
        "n_exploratory": int(claim["v16_claim_status"].eq("exploratory_locus_distributed_candidate").sum()),
        "n_multi_strong": int(claim["v16_claim_status"].eq("multi_strong_locus_pathway_overlap").sum()),
        "n_negative": int(claim["v16_claim_status"].eq("negative").sum()),
    }
    if "v16b_claim_status" in claim.columns:
        row.update(
            {
                "v16b_n_manuscript_ready": int(
                    claim["v16b_claim_status"].eq("manuscript_ready_distributed_candidate").sum()
                ),
                "v16b_n_high_confidence": int(
                    claim["v16b_claim_status"].eq("high_confidence_diagnostic_candidate").sum()
                ),
                "v16b_n_exploratory": int(
                    claim["v16b_claim_status"].eq("exploratory_locus_distributed_candidate").sum()
                ),
                "v16b_n_multi_strong": int(
                    claim["v16b_claim_status"].eq("multi_strong_locus_pathway_overlap").sum()
                ),
                "v16b_n_negative": int(claim["v16b_claim_status"].eq("negative").sum()),
            }
        )
    for gate in [
        "null_quality_pass",
        "null_quality_balanced_pass",
        "multiplicity_pass",
        "top_tail_pass",
        "leave_topk_pass",
        "redundancy_pass",
        "external_locus_sensitivity_pass",
    ]:
        if gate in claim.columns:
            row[f"{gate}_all_n"] = int(claim[gate].astype(bool).sum())
            row[f"{gate}_nonnegative_n"] = int(claim.loc[nonnegative, gate].astype(bool).sum())
    for metric in [
        "null_exact_match_rate",
        "min_match_pool_size",
        "median_match_pool_size",
        "null_with_replacement_rate",
        "null_loci_with_insufficient_gene_pool_rate",
        "ripple_d_q_full_library",
        "module_specific_rank_q_full_library",
        "ripple_d_v16_empirical_p",
        "module_specific_rank_empirical_p",
    ]:
        if metric in claim.columns:
            values = pd.to_numeric(claim.loc[nonnegative, metric], errors="coerce")
            row[f"{metric}_nonnegative_median"] = float(values.median())
            row[f"{metric}_nonnegative_min"] = float(values.min())
    return row


def main() -> None:
    args = parse_args()
    ensure_out_dir(args.out_dir, force=args.force)
    tables_dir = args.out_dir / "tables"
    reports_dir = args.out_dir / "reports"
    library = load_anchored_gene_set_library(args.gene_set_file)
    scores = pd.read_csv(args.score_file, sep="\t", compression="infer")
    specs = parse_config_specs(args.config)

    summary_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    match_rows: list[pd.DataFrame] = []
    for offset, spec in enumerate(specs):
        config_name = str(spec["config_name"])
        print(f"[{datetime.now(UTC).isoformat()}] running {config_name}", flush=True)
        config = build_config(args, spec)
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
        modules, locus_audit, _, run_summary = ripple_d_module_tests(
            scores,
            library,
            config=config,
            min_present=args.min_present,
            n_null=args.n_null,
            seed=args.seed + offset * 1009,
            return_null_details=False,
            precomputed_work=work,
            precomputed_locus_background=background,
        )
        claim = add_ripple_d_v16_claim_readiness(
            modules,
            config,
            locus_audit=locus_audit,
            external_locus_audit=external_audit,
        )
        claim.insert(0, "config_name", config_name)
        modules.insert(0, "config_name", config_name)
        config_dir = tables_dir / config_name
        write_table(config_dir / f"{args.trait}.{config_name}.v16_claim_readiness.tsv", claim)
        write_table(config_dir / f"{args.trait}.{config_name}.v16_module_tests.tsv", modules)
        write_table(config_dir / f"{args.trait}.{config_name}.v16_locus_background_audit.tsv", locus_background_audit_table(background))
        write_table(config_dir / f"{args.trait}.{config_name}.v16_external_locus_audit.tsv", external_audit)

        match_rows.append(match_bin_summary(background, config_name))
        row = {
            **summarize_claim_table(claim, config_name),
            "degree_bins": int(spec["degree_bins"]),
            "property_bins": int(spec["property_bins"]),
            "annotation_matching_enabled": bool(spec["annotation_matching_enabled"]),
            "n_null": int(args.n_null),
            "seed": int(args.seed + offset * 1009),
            "score_file": str(args.score_file),
            "gene_set_file": str(args.gene_set_file),
            "created_utc": datetime.now(UTC).isoformat(),
        }
        summary_rows.append(row)
        for status, count in claim["v16_claim_status"].value_counts(dropna=False).items():
            status_rows.append({"config_name": config_name, "policy": "v16_strict", "claim_status": status, "n": int(count)})
        if "v16b_claim_status" in claim.columns:
            for status, count in claim["v16b_claim_status"].value_counts(dropna=False).items():
                status_rows.append(
                    {"config_name": config_name, "policy": "v16b_balanced", "claim_status": status, "n": int(count)}
                )
        (reports_dir / f"{args.trait}.{config_name}.summary.json").write_text(
            json.dumps({**run_summary, **row}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(
            f"[{datetime.now(UTC).isoformat()}] finished {config_name}: "
            f"null_quality_nonnegative={row.get('null_quality_pass_nonnegative_n', 0)}/"
            f"{row['n_nonnegative']}, exploratory={row['n_exploratory']}",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows)
    status = pd.DataFrame(status_rows)
    match = pd.concat(match_rows, ignore_index=True) if match_rows else pd.DataFrame()
    write_table(tables_dir / f"{args.trait}.v16_null_matching_sensitivity_summary.tsv", summary)
    write_table(tables_dir / f"{args.trait}.v16_null_matching_sensitivity_status_counts.tsv", status)
    write_table(tables_dir / f"{args.trait}.v16_null_matching_sensitivity_match_bins.tsv", match)

    report_lines = [
        f"# RIPPLE-D V1.6 null-matching sensitivity: {args.trait}",
        "",
        f"n_null = {args.n_null}",
        "",
        "## Summary",
        "",
        summary.to_string(index=False),
        "",
        "## Status counts",
        "",
        status.pivot_table(index=["config_name", "policy"], columns="claim_status", values="n", fill_value=0).to_string(),
        "",
        "## Match-bin diagnostics",
        "",
        match.to_string(index=False),
        "",
    ]
    report = "\n".join(report_lines)
    (reports_dir / f"{args.trait}.v16_null_matching_sensitivity_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
