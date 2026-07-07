#!/usr/bin/env python
"""Summarize RIPPLE-D V1.5 module claim readiness across sensitivity runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


STRONG_STATUS = "distributed_weak_signal_module_candidate"
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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-tables", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-stable-windows", type=int, default=2)
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t")
    table["source_table"] = str(path)
    if "analysis_window_label" not in table.columns:
        label = "unknown_window"
        name = path.name
        for part in name.split("."):
            if part.startswith("w") and part.endswith("bp"):
                label = part
                break
        table["analysis_window_label"] = label
    return table


def status_for_group(group: pd.DataFrame, min_stable_windows: int) -> pd.Series:
    strong = group.loc[group["module_status"].eq(STRONG_STATUS)].copy()
    windows = sorted(str(value) for value in strong.get("analysis_window_label", pd.Series(dtype=str)).dropna().unique())
    locus_defs = sorted(str(value) for value in group.get("locus_definition", pd.Series(dtype=str)).dropna().unique())
    annotation_values = sorted(str(value) for value in group.get("annotation_matching_enabled", pd.Series(dtype=str)).dropna().unique())
    degraded = group.get("null_gene_count_match_degraded", pd.Series(False, index=group.index)).astype(str).str.lower().isin(
        ["true", "1"]
    )
    module_rank_p = pd.to_numeric(group.get("module_specific_rank_empirical_p", pd.Series([np.nan])), errors="coerce")
    locus_p = pd.to_numeric(group.get("locus_robust_empirical_p", pd.Series([np.nan])), errors="coerce")
    ripple_p = pd.to_numeric(group.get("ripple_d_empirical_p", pd.Series([np.nan])), errors="coerce")
    replacement_rate = pd.to_numeric(group.get("null_with_replacement_rate", pd.Series([np.nan])), errors="coerce")

    pseudo_status = "stable" if len(windows) >= min_stable_windows else ("single_window_only" if len(windows) == 1 else "failed")
    ld_status = (
        "passed"
        if any("external_locus_column" in str(value) for value in locus_defs) and not strong.empty
        else "not_tested"
    )
    annotation_status = (
        "passed"
        if {"True", "False"}.issubset(set(annotation_values)) and not strong.empty
        else ("not_tested" if len(annotation_values) <= 1 else "mixed")
    )
    rank_interp = "module_specific" if (module_rank_p.min() < 0.05) else "not_supported"

    downgrade: list[str] = []
    if strong.empty:
        downgrade.append("no_strong_v15_status")
    if pseudo_status != "stable":
        downgrade.append(f"pseudo_locus_window_stability_{pseudo_status}")
    if ld_status != "passed":
        downgrade.append("ld_block_or_clumped_locus_sensitivity_not_passed")
    if degraded.any():
        downgrade.append("null_gene_count_match_degraded")
    if rank_interp != "module_specific":
        downgrade.append("module_specific_rank_not_supported")

    manuscript_ready = not downgrade
    best = strong.iloc[0] if not strong.empty else group.iloc[0]
    return pd.Series(
        {
            "module_id": best.get("module_id", ""),
            "module_name": best.get("module_name", group.name),
            "best_status": STRONG_STATUS if not strong.empty else str(best.get("module_status", "")),
            "n_runs": int(len(group)),
            "n_strong_runs": int(len(strong)),
            "strong_windows": ",".join(windows),
            "locus_definitions": "|".join(locus_defs),
            "annotation_matching_values": ",".join(annotation_values),
            "pseudo_locus_window_stability_status": pseudo_status,
            "ld_block_locus_sensitivity_status": ld_status,
            "annotation_matching_sensitivity_status": annotation_status,
            "rank_evidence_interpretation": rank_interp,
            "manuscript_claim_ready": bool(manuscript_ready),
            "downgrade_reason": "none" if manuscript_ready else ";".join(downgrade),
            "min_locus_robust_empirical_p": float(locus_p.min()),
            "min_ripple_d_empirical_p": float(ripple_p.min()),
            "min_module_specific_rank_empirical_p": float(module_rank_p.min()),
            "max_null_with_replacement_rate": float(replacement_rate.max()),
            "source_tables": "|".join(sorted(set(group["source_table"].astype(str)))),
        }
    )


def parse_gene_set(value: object) -> set[str]:
    if pd.isna(value) or not str(value):
        return set()
    return {gene.strip().upper() for gene in str(value).split(",") if gene.strip()}


def add_overlap_redundancy(summary: pd.DataFrame, combined: pd.DataFrame) -> pd.DataFrame:
    work = combined.copy()
    work["_status_order"] = work["module_status"].map(STATUS_ORDER).fillna(99)
    for col in ["locus_robust_empirical_p", "ripple_d_empirical_p", "module_specific_rank_empirical_p"]:
        work[col] = pd.to_numeric(work.get(col), errors="coerce")
    best = (
        work.sort_values(["_status_order", "locus_robust_empirical_p", "ripple_d_empirical_p", "module_specific_rank_empirical_p"])
        .drop_duplicates("module_name", keep="first")
        .reset_index(drop=True)
    )
    gene_sets = {str(row.module_name): parse_gene_set(row.present_genes) for row in best.itertuples(index=False)}
    locus_sets = {
        str(row.module_name): {item for item in str(getattr(row, "top1_locus", "")).split(",") if item}
        for row in best.itertuples(index=False)
    }
    cluster_id: dict[str, str] = {}
    representative: dict[str, str] = {}
    max_jaccard: dict[str, float] = {}
    unique_locus_fraction: dict[str, float] = {}
    clusters: list[set[str]] = []
    reps: list[str] = []
    for name in best["module_name"].astype(str):
        genes = gene_sets.get(name, set())
        best_j = 0.0
        best_rep = name
        assigned = False
        for idx, rep in enumerate(reps):
            rep_genes = gene_sets.get(rep, set())
            union = genes | rep_genes
            jaccard = len(genes & rep_genes) / len(union) if union else 0.0
            if jaccard > best_j:
                best_j = jaccard
                best_rep = rep
            if jaccard >= 0.50:
                clusters[idx].add(name)
                cluster_id[name] = f"OC{idx + 1:04d}"
                representative[name] = rep
                assigned = True
                break
        if not assigned:
            reps.append(name)
            clusters.append({name})
            cluster_id[name] = f"OC{len(clusters):04d}"
            representative[name] = name
        max_jaccard[name] = best_j
        loci = locus_sets.get(name, set())
        rep_loci = locus_sets.get(best_rep, set())
        unique_locus_fraction[name] = len(loci - rep_loci) / len(loci) if loci else float("nan")
    out = summary.copy()
    out["module_overlap_cluster_id"] = out["module_name"].astype(str).map(cluster_id).fillna("")
    out["representative_module_in_cluster"] = out["module_name"].astype(str).map(representative).fillna("")
    out["max_jaccard_to_higher_ranked_module"] = out["module_name"].astype(str).map(max_jaccard).fillna(0.0)
    out["unique_locus_fraction"] = out["module_name"].astype(str).map(unique_locus_fraction)
    out["redundancy_downgrade_reason"] = np.where(
        (out["representative_module_in_cluster"].ne(out["module_name"])) & (out["max_jaccard_to_higher_ranked_module"] >= 0.50),
        "overlaps_higher_ranked_module",
        "none",
    )
    return out


def main() -> None:
    args = parse_args()
    tables = [read_table(path) for path in args.module_tables]
    combined = pd.concat(tables, ignore_index=True)
    group_key = "module_name" if "module_name" in combined.columns else "module_id"
    summary = (
        combined.groupby(group_key, dropna=False)
        .apply(status_for_group, min_stable_windows=args.min_stable_windows)
        .reset_index(drop=True)
    )
    summary = add_overlap_redundancy(summary, combined)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
