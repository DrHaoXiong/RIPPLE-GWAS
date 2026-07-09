#!/usr/bin/env python
"""Summarize DR-specific library v1 V1.6b cross-context results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PRIVATE_ROOT = Path("/mnt/d/RIPPLE/RIPPLE_private")
DEFAULT_ROOT = PRIVATE_ROOT / "30_analysis" / "dr_specific_library_v1_v16b_n5000"
DEFAULT_REGISTRY = (
    PRIVATE_ROOT
    / "20_processed_data"
    / "reference_pathways"
    / "dr_specific_library_v1"
    / "tables"
    / "dr_specific_library_v1.retinal_only.registry.tsv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    return parser.parse_args()


def claim_path(root: Path, context: str, trait: str) -> Path:
    return root / context / "tables" / f"{trait}.v16_claim_readiness.tsv"


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
        "top_tail_pass",
        "leave_topk_pass",
        "multiplicity_pass",
        "null_quality_balanced_pass",
        "annotation_sensitivity_balanced_pass",
    ]
    keep = [column for column in keep if column in table.columns]
    out = table.loc[:, keep].copy()
    return out.rename(columns={column: f"{prefix}_{column}" for column in keep if column != "module_name"})


def high(status: object) -> bool:
    return str(status) in {"high_confidence_diagnostic_candidate", "manuscript_ready_distributed_candidate"}


def suggestive(status: object) -> bool:
    return str(status) in {
        "high_confidence_diagnostic_candidate",
        "manuscript_ready_distributed_candidate",
        "exploratory_locus_distributed_candidate",
    }


def broad(status: object) -> bool:
    return str(status) in {
        "high_confidence_diagnostic_candidate",
        "manuscript_ready_distributed_candidate",
        "exploratory_locus_distributed_candidate",
        "multi_strong_locus_pathway_overlap",
        "raw_enrichment_only",
    }


def classify(row: pd.Series) -> str:
    dr_high = high(row.get("dr_v16b_claim_status"))
    resid_high = high(row.get("dr_resid_t2d_bmi_v16b_claim_status"))
    dr_suggestive = suggestive(row.get("dr_v16b_claim_status"))
    resid_suggestive = suggestive(row.get("dr_resid_t2d_bmi_v16b_claim_status"))
    t2d_broad = broad(row.get("t2d_v16b_claim_status"))
    bmi_broad = broad(row.get("bmi_v16b_claim_status"))
    if dr_high and resid_high and not t2d_broad and not bmi_broad:
        return "DR_specific_high_confidence_candidate"
    if dr_suggestive and resid_suggestive and not t2d_broad and not bmi_broad:
        return "DR_residual_suggestive_not_T2D_BMI"
    if dr_suggestive and not t2d_broad and not bmi_broad:
        return "DR_only_suggestive"
    if resid_suggestive and not t2d_broad and not bmi_broad:
        return "DR_residual_only_suggestive"
    if dr_suggestive and (t2d_broad or bmi_broad):
        return "DR_signal_cross_trait_shared_or_generic"
    return "negative_or_non_specific"


def render_report(summary: pd.DataFrame, root: Path, out_table: Path) -> str:
    lines = [
        "# DR-specific library v1 V1.6b n5000 summary",
        "",
        "This report summarizes the retinal-only DR-specific fixed hypothesis library across DR_MVP, DR residualized against T2D/BMI, T2D and BMI.",
        "",
        "## Claim Status Counts",
        "",
    ]
    for label, column in [
        ("DR_MVP", "dr_v16b_claim_status"),
        ("DR_MVP_residualized_T2D_BMI", "dr_resid_t2d_bmi_v16b_claim_status"),
        ("T2D", "t2d_v16b_claim_status"),
        ("BMI", "bmi_v16b_claim_status"),
        ("Specificity class", "specificity_class"),
    ]:
        lines += [f"### {label}", "", summary[column].value_counts(dropna=False).to_string(), ""]
    show_cols = [
        "module_name",
        "specificity_class",
        "category",
        "panel_role",
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
        "dr_top1_gene",
        "dr_n_effective_loci",
        "dr_top1_locus_contribution",
        "dr_top5_locus_contribution",
    ]
    show_cols = [column for column in show_cols if column in summary.columns]
    lines += [
        "## Module-Level Summary",
        "",
        summary.loc[:, show_cols].to_string(index=False),
        "",
        "## Interpretation",
        "",
        "- No retinal-only DR-specific module reached V1.6b high-confidence in DR_MVP at n_null=5000.",
        "- `DR_OXIDATIVE_STRESS_MITOCHONDRIAL_INJURY` was the strongest DR-context signal: exploratory in both DR_MVP and DR residualized against T2D/BMI, while negative in T2D and BMI.",
        "- `DR_BASEMENT_MEMBRANE_ECM_REMODELING` was residualized-DR exploratory but raw DR was only raw-enrichment, so it remains supportive-only.",
        "- This result does not strengthen a DR-specific module-discovery claim; it instead indicates that broad GO-derived signals were stronger than the hand-curated retinal pathobiology panel in the current MVP DR score layer.",
        "",
        f"Detailed table: `{out_table}`",
        f"Result root: `{root}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    paths = {
        "dr": claim_path(args.root, "DR_MVP", "DR_MVP_DR_SPECIFIC_V1_N5000"),
        "dr_resid_t2d_bmi": claim_path(
            args.root,
            "DR_MVP_RESID_T2D_BMI",
            "DR_MVP_RESID_T2D_BMI_DR_SPECIFIC_V1_N5000",
        ),
        "t2d": claim_path(args.root, "T2D", "T2D_DR_SPECIFIC_V1_N5000"),
        "bmi": claim_path(args.root, "BMI_IRN", "BMI_IRN_DR_SPECIFIC_V1_N5000"),
    }
    merged = read_claim(paths["dr"], "dr")
    for prefix in ["dr_resid_t2d_bmi", "t2d", "bmi"]:
        merged = merged.merge(read_claim(paths[prefix], prefix), on="module_name", how="left")
    registry = pd.read_csv(args.registry, sep="\t")
    meta_columns = [
        "gene_set",
        "category",
        "panel_role",
        "description",
        "n_query_genes",
    ]
    merged = merged.merge(
        registry.loc[:, [column for column in meta_columns if column in registry.columns]].rename(
            columns={"gene_set": "module_name"}
        ),
        on="module_name",
        how="left",
    )
    merged["specificity_class"] = merged.apply(classify, axis=1)
    for comparator in ["t2d", "bmi", "dr_resid_t2d_bmi"]:
        merged[f"dr_minus_{comparator}_ripple_d_v16_z"] = pd.to_numeric(
            merged["dr_ripple_d_v16_z"], errors="coerce"
        ) - pd.to_numeric(merged[f"{comparator}_ripple_d_v16_z"], errors="coerce")
    ordered = [
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
        "t2d_ripple_d_v16_empirical_p",
        "bmi_ripple_d_v16_empirical_p",
        "dr_ripple_d_q_full_library",
        "dr_resid_t2d_bmi_ripple_d_q_full_library",
        "t2d_ripple_d_q_full_library",
        "bmi_ripple_d_q_full_library",
        "dr_module_specific_rank_q_full_library",
        "dr_resid_t2d_bmi_module_specific_rank_q_full_library",
        "t2d_module_specific_rank_q_full_library",
        "bmi_module_specific_rank_q_full_library",
        "dr_minus_t2d_ripple_d_v16_z",
        "dr_minus_bmi_ripple_d_v16_z",
        "dr_minus_dr_resid_t2d_bmi_ripple_d_v16_z",
        "dr_top1_gene",
        "dr_top1_gene_score",
        "dr_n_loci",
        "dr_n_effective_loci",
        "dr_top1_locus_contribution",
        "dr_top5_locus_contribution",
        "dr_v16b_downgrade_reason",
        "dr_resid_t2d_bmi_v16b_downgrade_reason",
    ]
    ordered = [column for column in ordered if column in merged.columns]
    remaining = [column for column in merged.columns if column not in ordered]
    merged = merged.loc[:, ordered + remaining].sort_values(
        ["specificity_class", "dr_ripple_d_v16_empirical_p"],
        ascending=[True, True],
        na_position="last",
    )
    out_table = args.root / "tables" / "dr_specific_library_v1_v16b_cross_context_summary.tsv"
    out_report = args.root / "reports" / "dr_specific_library_v1_v16b_cross_context_report.md"
    out_table.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_table, sep="\t", index=False)
    out_report.write_text(render_report(merged, args.root, out_table), encoding="utf-8")
    print(out_report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
