#!/usr/bin/env python
"""Summarize RIPPLE-D V1.5 cross-trait generalization module scans."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


STATUS_ORDER = {
    "distributed_weak_signal_module_candidate": 0,
    "mixed_sparse_distributed_candidate": 1,
    "moderate_locus_supported_module": 2,
    "module_specific_rank_supported_module": 3,
    "top_locus_dominant_module": 4,
    "sparse_locus_pathway_overlap": 5,
    "raw_gene_set_enrichment_only": 6,
    "null_degraded_unresolved": 7,
    "negative": 8,
    "not_tested_low_overlap": 9,
}

EXPECTED_KEYWORDS = {
    "DR_MVP": [
        "TGF",
        "SMAD",
        "MATRIX",
        "COLLAGEN",
        "ENDOTHELIAL",
        "ANGIO",
        "VASCULAR",
        "HYPOXIA",
        "INFLAM",
        "COMPLEMENT",
        "RETINA",
    ],
    "T2D": ["INSULIN", "GLUCOSE", "PANCRE", "BETA", "ADIPO", "SECRETION", "METABOL", "PPAR"],
    "BMI_IRN": ["NEURON", "SYNAP", "HYPOTHAL", "ADIPO", "LEPTIN", "APPETITE", "FEEDING", "ENERGY"],
    "SCZ": ["SYNAP", "NEURON", "CALCIUM", "CHANNEL", "DENDR", "AXON", "POSTSYNAPTIC", "NEURODEVELOP"],
    "HEIGHT_IRN": ["SKELET", "BONE", "CARTILAGE", "CHONDRO", "GROWTH", "ECM", "COLLAGEN"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-tables", nargs="+", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--out-top-modules", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=25)
    return parser.parse_args()


def infer_trait(path: Path, table: pd.DataFrame) -> str:
    if "trait" in table.columns and table["trait"].notna().any():
        return str(table["trait"].dropna().iloc[0])
    name = path.name
    match = re.match(r"(.+?)\.w\d+bp\.v14c_ripple_d_module_tests\.tsv$", name)
    if match:
        return match.group(1)
    return path.stem


def read_module_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t")
    trait = infer_trait(path, table)
    table = table.copy()
    table["trait"] = trait
    table["source_table"] = str(path)
    table["_status_order"] = table["module_status"].map(STATUS_ORDER).fillna(99).astype(int)
    for col in [
        "ripple_d_empirical_p",
        "locus_robust_empirical_p",
        "module_specific_rank_empirical_p",
        "leave_top1_locus_empirical_p",
        "ripple_d_z",
        "n_effective_loci",
        "top1_locus_contribution",
        "top5_locus_contribution",
        "n_present",
        "n_loci",
    ]:
        if col in table.columns:
            table[col] = pd.to_numeric(table[col], errors="coerce")
    return table


def keyword_hit(trait: str, module_name: object) -> bool:
    text = str(module_name).upper()
    return any(keyword in text for keyword in EXPECTED_KEYWORDS.get(trait, []))


def summarize_trait(group: pd.DataFrame, trait: str) -> dict[str, object]:
    tested = group.loc[group["module_status"].ne("not_tested_low_overlap")].copy()
    distributed = tested["module_status"].eq("distributed_weak_signal_module_candidate")
    mixed = tested["module_status"].eq("mixed_sparse_distributed_candidate")
    raw_only = tested["module_status"].eq("raw_gene_set_enrichment_only")
    top_locus = tested["module_status"].eq("top_locus_dominant_module")
    expected_hits = tested.loc[distributed & tested["module_name"].map(lambda value: keyword_hit(trait, value))]
    top = tested.sort_values(
        ["_status_order", "ripple_d_empirical_p", "module_specific_rank_empirical_p", "locus_robust_empirical_p"],
        na_position="last",
    ).head(5)
    return {
        "trait": trait,
        "n_modules_total": int(len(group)),
        "n_modules_tested": int(len(tested)),
        "n_distributed_candidates": int(distributed.sum()),
        "n_mixed_sparse_distributed": int(mixed.sum()),
        "n_raw_gene_set_enrichment_only": int(raw_only.sum()),
        "n_top_locus_dominant": int(top_locus.sum()),
        "distributed_candidate_fraction": float(distributed.sum() / len(tested)) if len(tested) else np.nan,
        "top_locus_dominant_fraction": float(top_locus.sum() / len(tested)) if len(tested) else np.nan,
        "expected_keyword_distributed_hits": int(len(expected_hits)),
        "top5_modules": ";".join(top["module_name"].astype(str).tolist()),
        "median_top1_locus_contribution_distributed": float(
            tested.loc[distributed, "top1_locus_contribution"].median()
        )
        if distributed.any()
        else np.nan,
        "median_n_effective_loci_distributed": float(tested.loc[distributed, "n_effective_loci"].median())
        if distributed.any()
        else np.nan,
        "source_tables": "|".join(sorted(set(group["source_table"].astype(str)))),
    }


def top_module_table(all_modules: pd.DataFrame, top_n: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, group in all_modules.groupby("trait", observed=True):
        tested = group.loc[group["module_status"].ne("not_tested_low_overlap")].copy()
        tested["expected_keyword_hit"] = tested.apply(
            lambda row: keyword_hit(str(row["trait"]), row["module_name"]),
            axis=1,
        )
        ordered = tested.sort_values(
            ["_status_order", "ripple_d_empirical_p", "module_specific_rank_empirical_p", "locus_robust_empirical_p"],
            na_position="last",
        ).head(top_n)
        rows.append(ordered)
    if not rows:
        return pd.DataFrame()
    keep_cols = [
        "trait",
        "module_id",
        "module_name",
        "module_status",
        "expected_keyword_hit",
        "n_present",
        "n_loci",
        "n_effective_loci",
        "top1_gene",
        "top1_gene_score",
        "top1_locus",
        "top1_locus_contribution",
        "top5_locus_contribution",
        "ripple_d_z",
        "ripple_d_empirical_p",
        "module_specific_rank_z",
        "module_specific_rank_empirical_p",
        "leave_top1_locus_empirical_p",
        "locus_definition",
        "source_table",
    ]
    out = pd.concat(rows, ignore_index=True)
    return out.loc[:, [col for col in keep_cols if col in out.columns]]


def render_report(summary: pd.DataFrame, top_modules: pd.DataFrame) -> str:
    lines = [
        "# RIPPLE-D V1.5 cross-trait generalization summary",
        "",
        "This report summarizes fixed-library RIPPLE-D module scans using the same external LD-block "
        "locus definition across traits. It is a generalization screen, not a manuscript-ready "
        "trait-specific biological interpretation by itself.",
        "",
        "## Trait-level counts",
        "",
        summary.to_string(index=False),
        "",
        "## Top modules by trait",
        "",
    ]
    if top_modules.empty:
        lines.append("No top modules were available.")
    else:
        for trait, group in top_modules.groupby("trait", observed=True):
            lines.extend(
                [
                    f"### {trait}",
                    "",
                    group[
                        [
                            "module_name",
                            "module_status",
                            "expected_keyword_hit",
                            "ripple_d_z",
                            "ripple_d_empirical_p",
                            "module_specific_rank_empirical_p",
                            "top1_gene",
                            "n_effective_loci",
                            "top1_locus_contribution",
                        ]
                    ]
                    .head(10)
                    .to_string(index=False),
                    "",
                ]
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    modules = [read_module_table(path) for path in args.module_tables]
    all_modules = pd.concat(modules, ignore_index=True)
    summary = pd.DataFrame(
        [summarize_trait(group, str(trait)) for trait, group in all_modules.groupby("trait", observed=True)]
    ).sort_values("trait")
    top = top_module_table(all_modules, args.top_n)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_top_modules.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_summary, sep="\t", index=False)
    top.to_csv(args.out_top_modules, sep="\t", index=False)
    report = render_report(summary, top)
    args.out_report.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
