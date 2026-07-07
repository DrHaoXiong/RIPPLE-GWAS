#!/usr/bin/env python
"""Strengthen anchored module nulls with gene-property and annotation matching."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ripple.nulls.score_permutation import assign_degree_bins  # noqa: E402

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
ANCHOR_ROOT = ANALYSIS_ROOT / "tier4_v12_anchored_broad_reactome_go_cross_trait_n500"
DEFAULT_OUT_DIR = ANALYSIS_ROOT / "review_driven_revision_v1" / "anchored_strengthened_nulls"
DEFAULT_SUPPLEMENT_DIR = MANUSCRIPT_ROOT / "supplementary_files"
DEFAULT_LIBRARY = ANCHOR_ROOT / "DR_MVP" / "tables" / "DR_MVP.v12_anchored_module_library.tsv.gz"
DEFAULT_TOP_MODULES = ANCHOR_ROOT / "tables" / "cross_trait_top_modules.tsv"


ANALYSIS_DIRS = {
    "DR_MVP": ANALYSIS_ROOT / "dr_mvp_string_final5000",
    "T2D": ANALYSIS_ROOT / "t2d_analysis_ready",
    "HEIGHT_IRN": ANALYSIS_ROOT / "height_irn_analysis_ready",
    "BMI_IRN": ANALYSIS_ROOT / "bmi_irn_analysis_ready",
    "SCZ": ANALYSIS_ROOT / "scz_no_mhc_string_final5000",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-modules", type=Path, default=DEFAULT_TOP_MODULES)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--supplement-dir", type=Path, default=DEFAULT_SUPPLEMENT_DIR)
    parser.add_argument("--top-rank", type=int, default=5)
    parser.add_argument("--include-status", action="append", default=["anchored_familywise_supported"])
    parser.add_argument("--n-null", type=int, default=5000)
    parser.add_argument("--degree-bins", type=int, default=10)
    parser.add_argument("--property-bins", type=int, default=5)
    parser.add_argument("--annotation-bins", type=int, default=5)
    parser.add_argument("--jaccard-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--copy-to-supplement", action="store_true", default=True)
    return parser.parse_args()


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t", compression="infer")


def write_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False)


def lcc_scores_path(trait: str) -> Path:
    return ANALYSIS_DIRS[trait] / "tables" / f"{trait}.lcc_gene_scores.1000G_LD.residualized.tsv.gz"


def parse_genes(value: Any) -> set[str]:
    return {gene.strip().upper() for gene in str(value).split(",") if gene.strip()}


def annotation_counts(library: pd.DataFrame, background: set[str]) -> dict[str, int]:
    counts = {gene: 0 for gene in background}
    for genes in library["query_genes"].map(parse_genes):
        for gene in genes & background:
            counts[gene] += 1
    return counts


def quantile_bins(values: pd.Series, n_bins: int) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").fillna(values.median())
    if values.nunique() <= 1:
        return pd.Series(np.zeros(len(values), dtype=int), index=values.index)
    ranked = values.rank(method="first")
    return pd.qcut(ranked, q=min(n_bins, values.nunique()), labels=False, duplicates="drop").astype(int)


def prepare_background(scores: pd.DataFrame, library: pd.DataFrame, *, args: argparse.Namespace) -> pd.DataFrame:
    work = scores.copy()
    work["gene_symbol"] = work["gene_symbol"].astype(str).str.upper()
    work = work.drop_duplicates("gene_symbol", keep="first").reset_index(drop=True)
    background = set(work["gene_symbol"])
    counts = annotation_counts(library, background)
    work["annotation_count"] = work["gene_symbol"].map(counts).fillna(0).astype(int)
    work["degree_bin"] = assign_degree_bins(work["graph_degree"], n_bins=args.degree_bins).astype(int)
    work["length_bin"] = quantile_bins(work["gene_length"], args.property_bins)
    work["snp_count_bin"] = quantile_bins(work["n_mapped_snps"], args.property_bins)
    work["annotation_bin"] = quantile_bins(work["annotation_count"], args.annotation_bins)
    work["strict_match_key"] = list(
        zip(
            work["degree_bin"],
            work["length_bin"],
            work["snp_count_bin"],
            work["annotation_bin"],
            strict=True,
        )
    )
    work["degree_annotation_key"] = list(zip(work["degree_bin"], work["annotation_bin"], strict=True))
    return work


def sqrt_n_mean(values: np.ndarray) -> float:
    return float(np.mean(values) * np.sqrt(len(values))) if len(values) else float("nan")


def empirical_upper(null_values: np.ndarray, observed: float) -> float:
    finite = np.asarray(null_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float((1 + np.count_nonzero(finite >= observed)) / (1 + len(finite))) if len(finite) else float("nan")


def z_score(null_values: np.ndarray, observed: float) -> float:
    finite = np.asarray(null_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 2:
        return float("nan")
    sd = float(np.std(finite, ddof=1))
    return float((observed - float(np.mean(finite))) / sd) if sd > 0 else float("nan")


def build_key_index(work: pd.DataFrame, key_col: str) -> dict[Any, np.ndarray]:
    return {
        key: group.index.to_numpy(dtype=int)
        for key, group in work.groupby(key_col, sort=False, observed=True)
    }


def sample_matched_indices(
    module: pd.DataFrame,
    work: pd.DataFrame,
    *,
    key_col: str,
    key_index: dict[Any, np.ndarray],
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    sampled: list[int] = []
    fallback_count = 0
    for _, row in module.iterrows():
        candidates = key_index.get(row[key_col])
        if candidates is None or len(candidates) == 0:
            candidates = work.index.to_numpy(dtype=int)
            fallback_count += 1
        sampled.append(int(rng.choice(candidates)))
    return np.asarray(sampled, dtype=int), fallback_count


def strengthened_null_for_module(
    module_row: pd.Series,
    work: pd.DataFrame,
    *,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> dict[str, Any]:
    genes = parse_genes(module_row["present_genes"])
    module = work.loc[work["gene_symbol"].isin(genes)].copy()
    observed = sqrt_n_mean(module["assoc_resid_score"].to_numpy(dtype=float))
    key_indices = {
        "strict_match_key": build_key_index(work, "strict_match_key"),
        "degree_annotation_key": build_key_index(work, "degree_annotation_key"),
        "degree_bin": build_key_index(work, "degree_bin"),
    }
    rows: list[dict[str, Any]] = []
    for null_type, key_col in [
        ("degree_length_snp_annotation_matched", "strict_match_key"),
        ("degree_annotation_matched", "degree_annotation_key"),
        ("degree_only_matched", "degree_bin"),
    ]:
        values = np.empty(args.n_null, dtype=float)
        fallback_counts = np.empty(args.n_null, dtype=int)
        for idx in range(args.n_null):
            sampled, fallback_count = sample_matched_indices(
                module,
                work,
                key_col=key_col,
                key_index=key_indices[key_col],
                rng=rng,
            )
            values[idx] = sqrt_n_mean(work.loc[sampled, "assoc_resid_score"].to_numpy(dtype=float))
            fallback_counts[idx] = fallback_count
        rows.append(
            {
                "trait": module_row["trait"],
                "analysis_id": module_row["analysis_id"],
                "module_id": module_row["module_id"],
                "module_name": module_row["module_name"],
                "module_status_original": module_row["module_status"],
                "null_type": null_type,
                "statistic_name": "sqrt_n_mean_residualized_score",
                "statistic_direction": "greater_is_more_extreme",
                "observed_value": observed,
                "null_mean": float(np.mean(values)),
                "null_sd": float(np.std(values, ddof=1)),
                "z": z_score(values, observed),
                "empirical_p": empirical_upper(values, observed),
                "n_null": int(args.n_null),
                "mean_fallback_genes_per_replicate": float(np.mean(fallback_counts)),
                "max_fallback_genes_per_replicate": int(np.max(fallback_counts)),
                "n_present": int(len(module)),
                "present_genes": ",".join(sorted(genes)),
            }
        )
    return rows


def select_modules(top_modules: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    status_keep = top_modules["module_status"].isin(set(args.include_status))
    rank_keep = pd.to_numeric(top_modules["top_rank_within_trait"], errors="coerce").le(args.top_rank)
    return top_modules.loc[status_keep | rank_keep].drop_duplicates(["trait", "module_name"]).reset_index(drop=True)


def redundancy_clusters(modules: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    graph = nx.Graph()
    genes = {idx: parse_genes(row.present_genes) for idx, row in modules.iterrows()}
    for idx, row in modules.iterrows():
        graph.add_node(idx, trait=row.trait, module_name=row.module_name)
    for i, genes_i in genes.items():
        for j, genes_j in genes.items():
            if j <= i:
                continue
            if modules.loc[i, "trait"] != modules.loc[j, "trait"]:
                continue
            jaccard = len(genes_i & genes_j) / len(genes_i | genes_j) if genes_i or genes_j else 0.0
            if jaccard >= threshold:
                graph.add_edge(i, j, jaccard=jaccard)
    rows = []
    for cluster_id, component in enumerate(nx.connected_components(graph), start=1):
        sub = modules.loc[list(component)].copy()
        sub = sub.sort_values(["library_familywise_p", "degree_matched_empirical_p"], ascending=True)
        rows.append(
            {
                "trait": sub.iloc[0]["trait"],
                "cluster_id": f"GORED_{cluster_id:04d}",
                "n_modules": int(len(sub)),
                "representative_module": sub.iloc[0]["module_name"],
                "representative_familywise_p": sub.iloc[0].get("library_familywise_p", np.nan),
                "member_modules": "|".join(sub["module_name"].astype(str)),
                "jaccard_threshold": float(threshold),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    library = read_tsv(args.library)
    top_modules = select_modules(read_tsv(args.top_modules), args)
    all_rows: list[dict[str, Any]] = []
    for trait, group in top_modules.groupby("trait", sort=False):
        if trait not in ANALYSIS_DIRS or not lcc_scores_path(trait).exists():
            continue
        work = prepare_background(read_tsv(lcc_scores_path(trait)), library, args=args)
        for _, row in group.iterrows():
            all_rows.extend(strengthened_null_for_module(row, work, args=args, rng=rng))
    summary = pd.DataFrame(all_rows)
    summary["script_path"] = str(Path(__file__).resolve())
    summary["seed"] = int(args.seed)
    summary["timestamp"] = datetime.now(UTC).isoformat()
    out_path = args.out_dir / "anchored_strengthened_null_summary.tsv"
    write_table(out_path, summary)
    clusters = redundancy_clusters(top_modules, threshold=args.jaccard_threshold)
    cluster_path = args.out_dir / "anchored_module_redundancy_clusters.tsv"
    write_table(cluster_path, clusters)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "script_path": str(Path(__file__).resolve()),
        "n_modules_tested": int(top_modules.shape[0]),
        "n_null": int(args.n_null),
        "matching": "degree + gene_length + mapped_snp_count + annotation_count",
        "summary": str(out_path),
        "redundancy_clusters": str(cluster_path),
    }
    (args.out_dir / "anchored_strengthened_null_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    if args.copy_to_supplement:
        write_table(args.supplement_dir / "anchored_strengthened_null_summary.tsv", summary)
        write_table(args.supplement_dir / "anchored_module_redundancy_clusters.tsv", clusters)
    print(f"Wrote anchored strengthened nulls to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
