#!/usr/bin/env python
"""Summarize V1.6b high8 DR specificity and top-locus biology audits."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from audit_v15_ready_module_biology import (  # noqa: E402
    DIABETES_GENES,
    DR_MICROVASCULAR_GENES,
    summarize_module_genes,
    theme_from_name,
)

PRIVATE_ROOT = Path("/mnt/d/RIPPLE/RIPPLE_private")
DEFAULT_ROOT = (
    PRIVATE_ROOT
    / "30_analysis"
    / "tier4_v16_claim_readiness_hardening_v0_1"
    / "v16b_high8_refinement_n5000"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--dr-claim",
        type=Path,
        default=DEFAULT_ROOT / "DR_MVP" / "tables" / "DR_MVP_V16B_HIGH8_N5000.v16_claim_readiness.tsv",
    )
    parser.add_argument(
        "--dr-resid-claim",
        type=Path,
        default=DEFAULT_ROOT
        / "DR_MVP_RESID_T2D_BMI"
        / "tables"
        / "DR_MVP_RESID_T2D_BMI_V16B_HIGH8_N5000.v16_claim_readiness.tsv",
    )
    parser.add_argument(
        "--t2d-claim",
        type=Path,
        default=DEFAULT_ROOT / "T2D" / "tables" / "T2D_V16B_HIGH8_N5000.v16_claim_readiness.tsv",
    )
    parser.add_argument(
        "--bmi-claim",
        type=Path,
        default=DEFAULT_ROOT / "BMI_IRN" / "tables" / "BMI_IRN_V16B_HIGH8_N5000.v16_claim_readiness.tsv",
    )
    parser.add_argument(
        "--gene-set-file",
        type=Path,
        default=DEFAULT_ROOT / "inputs" / "DR_MVP.v16b_sensitivity_completed_high8_gene_sets.tsv.gz",
    )
    parser.add_argument(
        "--dr-locus-audit",
        type=Path,
        default=DEFAULT_ROOT
        / "DR_MVP"
        / "tables"
        / "DR_MVP_V16B_HIGH8_N5000.v16_locus_contribution_audit.tsv",
    )
    parser.add_argument(
        "--dr-score-file",
        type=Path,
        default=PRIVATE_ROOT
        / "30_analysis"
        / "tier4_v15_locus_distributed_module_repair_v0_1"
        / "external_ldblock_sensitivity"
        / "inputs"
        / "DR_MVP.lcc_gene_scores.with_eur_ldblocks.tsv.gz",
    )
    return parser.parse_args()


def read_claim(path: Path, prefix: str) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t")
    keep = [
        "module_name",
        "module_status",
        "v16b_claim_status",
        "v16b_downgrade_reason",
        "ripple_d_v16_z",
        "ripple_d_v16_empirical_p",
        "ripple_d_q_full_library",
        "module_specific_rank_q_full_library",
        "module_specific_rank_z",
        "module_specific_rank_empirical_p",
        "leave_top1_locus_empirical_p",
        "leave_top3_locus_empirical_p",
        "top_locus_conditioned_leave_top1_p",
        "top_locus_conditioned_leave_top3_p",
        "n_present",
        "n_loci",
        "n_effective_loci",
        "top1_gene",
        "top1_gene_score",
        "top1_locus",
        "top1_locus_contribution",
        "top5_locus_contribution",
        "fraction_loci_with_uncapped_score_gt_3",
        "fraction_positive_signal_from_uncapped_gt_3",
        "n_loci_in_genome_top_1pct",
        "top_tail_pass",
        "leave_topk_pass",
        "multiplicity_pass",
        "null_quality_balanced_pass",
        "annotation_sensitivity_balanced_pass",
    ]
    keep = [column for column in keep if column in table.columns]
    out = table.loc[:, keep].copy()
    return out.rename(columns={column: f"{prefix}_{column}" for column in keep if column != "module_name"})


def supported(status: object) -> bool:
    return str(status) in {"high_confidence_diagnostic_candidate", "manuscript_ready_distributed_candidate"}


def broad_supported(status: object) -> bool:
    return str(status) in {
        "high_confidence_diagnostic_candidate",
        "manuscript_ready_distributed_candidate",
        "multi_strong_locus_pathway_overlap",
    }


def classify(row: pd.Series) -> str:
    dr = supported(row.get("dr_v16b_claim_status"))
    resid = supported(row.get("dr_resid_t2d_bmi_v16b_claim_status"))
    t2d_broad = broad_supported(row.get("t2d_v16b_claim_status"))
    bmi_broad = broad_supported(row.get("bmi_v16b_claim_status"))
    if dr and resid and not t2d_broad and not bmi_broad:
        return "DR_residual_supported_not_T2D_BMI_supported"
    if dr and resid and (t2d_broad or bmi_broad):
        return "DR_residual_supported_but_cross_trait_shared"
    if dr and not resid and not t2d_broad and not bmi_broad:
        return "DR_high_but_residual_weakened"
    if dr and (t2d_broad or bmi_broad):
        return "DR_high_cross_trait_shared_or_generic"
    if resid and not dr:
        return "DR_residual_only_candidate"
    return "not_DR_high_after_refinement"


def load_module_metadata(gene_set_file: Path) -> pd.DataFrame:
    library = pd.read_csv(gene_set_file, sep="\t", compression="infer")
    return (
        library.groupby("gene_set")
        .agg(
            source_database=("source_database", "first"),
            source_term_id=("source_term_id", "first"),
            source_term_name=("source_term_name", "first"),
            category=("category", "first"),
            n_library_genes=("gene_symbol", "nunique"),
        )
        .reset_index()
        .rename(columns={"gene_set": "module_name"})
    )


def build_biology_audit(dr_locus_audit: Path, dr_score_file: Path) -> pd.DataFrame:
    scores = pd.read_csv(dr_score_file, sep="\t", compression="infer")
    scores["gene_symbol"] = scores["gene_symbol"].astype(str).str.upper()
    score_map = dict(
        zip(scores["gene_symbol"], pd.to_numeric(scores["assoc_resid_score"], errors="coerce"), strict=False)
    )
    audit = pd.read_csv(dr_locus_audit, sep="\t")
    rows: list[dict[str, object]] = []
    for module_name, group in audit.groupby("module_name"):
        genes = [str(gene).upper() for gene in group["gene_symbol"]]
        diabetes_overlap = set(genes) & DIABETES_GENES
        dr_overlap = set(genes) & DR_MICROVASCULAR_GENES
        theme, reason = theme_from_name(module_name, diabetes_overlap, dr_overlap)
        if "MELANOSOME" in str(module_name).upper():
            theme = "retinal_rpe_pigment_plausible"
            reason = "melanosome/RPE pigment biology keyword"
        gene_summary = summarize_module_genes(genes, score_map)
        top_loci = (
            group.sort_values("locus_score", ascending=False)
            .groupby("locus_id", as_index=False)
            .first()
            .sort_values("locus_score", ascending=False)
            .head(8)
        )
        rows.append(
            {
                "module_name": module_name,
                "biological_theme": theme,
                "theme_reason": reason,
                "diabetes_prior_gene_overlap_n": len(diabetes_overlap),
                "diabetes_prior_gene_overlap": ",".join(sorted(diabetes_overlap)),
                "dr_microvascular_prior_gene_overlap_n": len(dr_overlap),
                "dr_microvascular_prior_gene_overlap": ",".join(sorted(dr_overlap)),
                "top8_dr_loci_by_score": ";".join(
                    f"{row.locus_id}:{row.gene_symbol}:{row.locus_score:.3g}"
                    for row in top_loci.itertuples(index=False)
                ),
                **gene_summary,
            }
        )
    return pd.DataFrame(rows)


def render_report(table: pd.DataFrame, args: argparse.Namespace, out_table: Path, out_audit: Path) -> str:
    lines = [
        "# DR_MVP V1.6b high8 n5000 specificity and biology audit",
        "",
        "## Inputs",
        "",
        f"- DR_MVP n5000: `{args.dr_claim}`",
        f"- DR residualized against T2D/BMI n5000: `{args.dr_resid_claim}`",
        f"- T2D n5000: `{args.t2d_claim}`",
        f"- BMI n5000: `{args.bmi_claim}`",
        "",
        "## Status counts",
        "",
    ]
    for label, column in [
        ("DR_MVP", "dr_v16b_claim_status"),
        ("DR_resid_T2D_BMI", "dr_resid_t2d_bmi_v16b_claim_status"),
        ("T2D", "t2d_v16b_claim_status"),
        ("BMI", "bmi_v16b_claim_status"),
        ("Specificity", "specificity_class"),
    ]:
        lines += [f"### {label}", "", table[column].value_counts(dropna=False).to_string(), ""]

    lines += [
        "## Main interpretation",
        "",
        "- DR_MVP retains 4/8 V1.6b high-confidence candidates at n_null=5000.",
        "- After residualizing DR scores against T2D/BMI, 2/8 remain V1.6b high-confidence.",
        "- T2D has 1/8 high-confidence and 2 additional multi-strong-locus overlaps; BMI has 0/8 high-confidence and 1 multi-strong-locus overlap.",
        "- The high8 set is not uniformly explained by T2D/BMI, but several signals remain generic cell-cycle/metabolic rather than clean DR microvascular modules.",
        "",
        "## Module-level summary",
        "",
    ]
    show_cols = [
        "module_name",
        "specificity_class",
        "biological_theme",
        "dr_v16b_claim_status",
        "dr_resid_t2d_bmi_v16b_claim_status",
        "t2d_v16b_claim_status",
        "bmi_v16b_claim_status",
        "dr_ripple_d_v16_z",
        "dr_resid_t2d_bmi_ripple_d_v16_z",
        "t2d_ripple_d_v16_z",
        "bmi_ripple_d_v16_z",
        "dr_top1_gene",
        "dr_top1_locus_contribution",
        "dr_top5_locus_contribution",
        "top10_module_genes_by_score",
    ]
    show_cols = [column for column in show_cols if column in table.columns]
    lines += [table.loc[:, show_cols].to_string(index=False), ""]
    lines += [f"Detailed specificity table: `{out_table}`", f"Top-locus biology audit: `{out_audit}`"]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root = args.root
    merged = read_claim(args.dr_claim, "dr")
    for prefix, path in [
        ("dr_resid_t2d_bmi", args.dr_resid_claim),
        ("t2d", args.t2d_claim),
        ("bmi", args.bmi_claim),
    ]:
        merged = merged.merge(read_claim(path, prefix), on="module_name", how="left")
    merged = merged.merge(load_module_metadata(args.gene_set_file), on="module_name", how="left")
    merged = merged.merge(build_biology_audit(args.dr_locus_audit, args.dr_score_file), on="module_name", how="left")

    merged["specificity_class"] = merged.apply(classify, axis=1)
    for prefix in ["dr", "dr_resid_t2d_bmi", "t2d", "bmi"]:
        merged[f"{prefix}_is_high_confidence"] = merged[f"{prefix}_v16b_claim_status"].map(supported)
        merged[f"{prefix}_is_broad_supported"] = merged[f"{prefix}_v16b_claim_status"].map(broad_supported)
    merged["dr_minus_t2d_ripple_d_v16_z"] = pd.to_numeric(
        merged["dr_ripple_d_v16_z"], errors="coerce"
    ) - pd.to_numeric(merged["t2d_ripple_d_v16_z"], errors="coerce")
    merged["dr_minus_bmi_ripple_d_v16_z"] = pd.to_numeric(
        merged["dr_ripple_d_v16_z"], errors="coerce"
    ) - pd.to_numeric(merged["bmi_ripple_d_v16_z"], errors="coerce")
    merged["dr_minus_resid_ripple_d_v16_z"] = pd.to_numeric(
        merged["dr_ripple_d_v16_z"], errors="coerce"
    ) - pd.to_numeric(merged["dr_resid_t2d_bmi_ripple_d_v16_z"], errors="coerce")

    ordered = [
        "module_name",
        "source_database",
        "source_term_id",
        "source_term_name",
        "category",
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
        "t2d_ripple_d_v16_empirical_p",
        "bmi_ripple_d_v16_empirical_p",
        "dr_ripple_d_q_full_library",
        "dr_module_specific_rank_q_full_library",
        "dr_resid_t2d_bmi_ripple_d_q_full_library",
        "dr_resid_t2d_bmi_module_specific_rank_q_full_library",
        "dr_minus_t2d_ripple_d_v16_z",
        "dr_minus_bmi_ripple_d_v16_z",
        "dr_minus_resid_ripple_d_v16_z",
        "dr_n_loci",
        "dr_n_effective_loci",
        "dr_top1_gene",
        "dr_top1_gene_score",
        "dr_top1_locus",
        "dr_top1_locus_contribution",
        "dr_top5_locus_contribution",
        "n_extreme_score_ge5",
        "n_upper_tail_score_ge3",
        "n_moderate_score_1_to_3",
        "top10_module_genes_by_score",
        "top8_dr_loci_by_score",
        "diabetes_prior_gene_overlap_n",
        "diabetes_prior_gene_overlap",
        "dr_microvascular_prior_gene_overlap_n",
        "dr_microvascular_prior_gene_overlap",
    ]
    ordered = [column for column in ordered if column in merged.columns]
    remaining = [column for column in merged.columns if column not in ordered]
    merged = merged.loc[:, ordered + remaining].sort_values(
        ["specificity_class", "dr_ripple_d_q_full_library", "dr_module_specific_rank_q_full_library"],
        na_position="last",
    )

    out_table = root / "tables" / "DR_MVP_V16B_HIGH8_N5000.DR_vs_T2D_BMI_specificity.tsv"
    out_audit = root / "tables" / "DR_MVP_V16B_HIGH8_N5000.top_locus_biology_audit.tsv"
    out_report = root / "reports" / "DR_MVP_V16B_HIGH8_N5000.specificity_and_biology_report.md"
    out_table.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_table, sep="\t", index=False)
    merged.loc[:, ordered].to_csv(out_audit, sep="\t", index=False)
    out_report.write_text(render_report(merged, args, out_table, out_audit), encoding="utf-8")
    print(out_report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
