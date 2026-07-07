#!/usr/bin/env python
"""Run Type I calibration for RIPPLE V1.2 anchored module evidence.

This script calibrates the fixed-library anchored module layer without
rerunning LD scoring. It uses existing V1.2 anchored outputs to define the
tested module library and generates outer null score vectors by permuting
residualized gene scores within graph-degree bins.

The main target is the analysis-level false-positive rate for:
    1. any module passing max-over-library familywise calibration;
    2. any such module also passing degree-matched node-set calibration.

This remains a diagnostic calibration for V1.2 anchored evidence. It does not
upgrade anchored modules to de novo topology-specific discoveries.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ripple.modules.anchored import empirical_upper  # noqa: E402
from ripple.nulls.score_permutation import (  # noqa: E402
    assign_degree_bins,
    degree_stratified_permuted_scores,
)


PRIVATE_ROOT = Path("/path/to/ripple_private_workspace")
DEFAULT_ANCHORED_ROOT = (
    PRIVATE_ROOT
    / "30_analysis"
    / "tier4_v12_anchored_broad_reactome_go_cross_trait_n500"
)
DEFAULT_OUT_DIR = (
    PRIVATE_ROOT
    / "30_analysis"
    / "tier4_v12_anchored_broad_reactome_go_type1_robustness_v1"
    / "type1"
)
THIS_SCRIPT = Path(__file__).resolve()


@dataclass(frozen=True)
class AnchoredRun:
    analysis_id: str
    trait: str
    result_dir: Path
    summary_path: Path
    module_table_path: Path
    score_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchored-root", type=Path, default=DEFAULT_ANCHORED_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--outer-n", type=int, default=200)
    parser.add_argument("--calibration-null-n", type=int, default=500)
    parser.add_argument("--degree-matched-null-n", type=int, default=500)
    parser.add_argument("--degree-bins", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--max-candidates-per-outer", type=int, default=50)
    parser.add_argument(
        "--subset-module-sources",
        default="Gene Ontology,Reactome",
        help=(
            "Comma-separated module_source values to calibrate in addition to the full library. "
            "Use an empty string to disable subset-family calibration."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_out_dir(path: Path, *, force: bool) -> None:
    if path.exists() and any(path.iterdir()) and not force:
        raise FileExistsError(f"{path} exists and is not empty. Use --force to overwrite.")
    (path / "tables").mkdir(parents=True, exist_ok=True)
    (path / "reports").mkdir(parents=True, exist_ok=True)


def write_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False, compression="infer")


def discover_runs(root: Path) -> list[AnchoredRun]:
    runs: list[AnchoredRun] = []
    for result_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        reports = result_dir / "reports"
        tables = result_dir / "tables"
        summary_paths = sorted(reports.glob("*.v12_anchored_module_summary.json"))
        module_paths = sorted(tables.glob("*.v12_anchored_module_tests.tsv"))
        if len(summary_paths) != 1 or len(module_paths) != 1:
            continue
        summary = json.loads(summary_paths[0].read_text(encoding="utf-8"))
        score_path = Path(str(summary.get("score_path", "")))
        if not score_path.exists():
            raise FileNotFoundError(f"Missing score_path for {result_dir}: {score_path}")
        runs.append(
            AnchoredRun(
                analysis_id=result_dir.name,
                trait=str(summary.get("trait", result_dir.name)),
                result_dir=result_dir,
                summary_path=summary_paths[0],
                module_table_path=module_paths[0],
                score_path=score_path,
            )
        )
    if not runs:
        raise FileNotFoundError(f"No anchored V1.2 result directories found under {root}")
    return runs


def split_genes(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [part.strip().upper() for part in str(value).split(",") if part.strip()]


def prepare_scores(path: Path) -> pd.DataFrame:
    scores = pd.read_csv(path, sep="\t", compression="infer")
    required = {"gene_symbol", "assoc_resid_score", "graph_degree"}
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"{path} missing required score columns: {missing}")
    scores = scores.copy()
    scores["gene_symbol"] = scores["gene_symbol"].astype(str).str.upper()
    scores["assoc_resid_score"] = pd.to_numeric(scores["assoc_resid_score"], errors="raise").astype(float)
    scores["graph_degree"] = pd.to_numeric(scores["graph_degree"], errors="raise").astype(int)
    scores = scores.drop_duplicates("gene_symbol", keep="first").reset_index(drop=True)
    if scores.empty:
        raise ValueError(f"No score rows in {path}")
    return scores


def prepare_tested_modules(path: Path) -> pd.DataFrame:
    modules = pd.read_csv(path, sep="\t")
    tested = modules.loc[modules["module_status"].ne("not_tested_low_overlap")].copy()
    if tested.empty:
        raise ValueError(f"No tested anchored modules in {path}")
    tested["present_gene_list"] = tested["present_genes"].map(split_genes)
    tested = tested[tested["present_gene_list"].map(len).ge(1)].reset_index(drop=True)
    return tested


def membership_matrix(
    scores: pd.DataFrame,
    modules: pd.DataFrame,
) -> tuple[sparse.csr_matrix, list[np.ndarray]]:
    gene_to_idx = {
        str(gene): idx for idx, gene in enumerate(scores["gene_symbol"].astype(str).to_numpy())
    }
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    module_indices: list[np.ndarray] = []
    for module_idx, genes in enumerate(modules["present_gene_list"]):
        indices = np.asarray([gene_to_idx[gene] for gene in genes if gene in gene_to_idx], dtype=int)
        if indices.size == 0:
            raise ValueError(f"Module {module_idx} has no score-overlapping genes after filtering.")
        module_indices.append(indices)
        rows.extend(int(idx) for idx in indices)
        cols.extend([module_idx] * int(indices.size))
        data.extend([1.0 / np.sqrt(float(indices.size))] * int(indices.size))
    matrix = sparse.csr_matrix(
        (np.asarray(data, dtype=float), (np.asarray(rows), np.asarray(cols))),
        shape=(len(scores), len(modules)),
    )
    return matrix, module_indices


def familywise_p_from_sorted_max(sorted_max_null: np.ndarray, observed_stats: np.ndarray) -> np.ndarray:
    null = np.asarray(sorted_max_null, dtype=float)
    stats = np.asarray(observed_stats, dtype=float)
    positions = np.searchsorted(null, stats, side="left")
    exceedances = null.size - positions
    return (1.0 + exceedances.astype(float)) / (1.0 + float(null.size))


def parse_subset_sources(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def calibration_families(modules: pd.DataFrame, subset_sources: tuple[str, ...]) -> dict[str, np.ndarray]:
    families: dict[str, np.ndarray] = {"all_modules": np.arange(len(modules), dtype=int)}
    for source in subset_sources:
        indices = np.flatnonzero(modules["module_source"].astype(str).eq(source).to_numpy())
        if indices.size:
            label = "module_source:" + source.replace(" ", "_")
            families[label] = indices.astype(int)
    return families


def degree_profiles(module_indices: list[np.ndarray], bins: np.ndarray) -> list[dict[int, int]]:
    profiles: list[dict[int, int]] = []
    for indices in module_indices:
        profile: dict[int, int] = {}
        for bin_id in bins[indices]:
            key = int(bin_id)
            profile[key] = profile.get(key, 0) + 1
        profiles.append(profile)
    return profiles


def degree_matched_stat_null(
    values: np.ndarray,
    bin_to_indices: dict[int, np.ndarray],
    profile: dict[int, int],
    *,
    n_null: int,
    rng: np.random.Generator,
) -> np.ndarray:
    null = np.empty(n_null, dtype=float)
    for replicate in range(n_null):
        sampled: list[int] = []
        for bin_id, count in sorted(profile.items()):
            candidates = bin_to_indices[int(bin_id)]
            replace = int(count) > len(candidates)
            sampled.extend(int(idx) for idx in rng.choice(candidates, size=int(count), replace=replace))
        sample = np.asarray(sampled, dtype=int)
        null[replicate] = float(np.sum(values[sample]) / np.sqrt(float(sample.size)))
    return null


def wilson_ci(false_positive_count: int, n_outer: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    if n_outer <= 0:
        return float("nan"), float("nan")
    phat = false_positive_count / n_outer
    denom = 1.0 + z * z / n_outer
    center = (phat + z * z / (2.0 * n_outer)) / denom
    half = z * np.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n_outer)) / n_outer) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def summarize_fpr(
    rows: pd.DataFrame,
    *,
    trait: str,
    analysis_id: str,
    calibration_family: str,
    n_modules_in_family: int,
    flag_col: str,
    alpha: float,
    outer_n: int,
    calibration_null_n: int,
) -> dict[str, object]:
    count = int(rows[flag_col].sum())
    fpr = float(count / outer_n) if outer_n else float("nan")
    low, high = wilson_ci(count, outer_n)
    return {
        "trait": trait,
        "analysis_id": analysis_id,
        "calibration_family": calibration_family,
        "n_modules_in_family": int(n_modules_in_family),
        "calibration_target": flag_col,
        "alpha": float(alpha),
        "fpr": fpr,
        "false_positive_count": count,
        "n_outer": int(outer_n),
        "calibration_null_n": int(calibration_null_n),
        "binomial_95ci_low": low,
        "binomial_95ci_high": high,
        "mc_se": float(np.sqrt(fpr * (1.0 - fpr) / outer_n)) if outer_n else float("nan"),
    }


def calibrate_run(run: AnchoredRun, args: argparse.Namespace, *, run_index: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = prepare_scores(run.score_path)
    modules = prepare_tested_modules(run.module_table_path)
    membership, module_indices = membership_matrix(scores, modules)
    families = calibration_families(modules, parse_subset_sources(args.subset_module_sources))
    total_null = int(args.outer_n + args.calibration_null_n)
    seed = int(args.seed + run_index * 100_003)
    score_null = degree_stratified_permuted_scores(
        scores,
        score_col="assoc_resid_score",
        degree_col="graph_degree",
        n_replicates=total_null,
        seed=seed,
        n_bins=args.degree_bins,
    )
    stats = np.asarray(score_null @ membership, dtype=float)
    family_nulls: dict[str, dict[str, object]] = {}
    for family_label, family_indices in families.items():
        max_stats = np.max(stats[:, family_indices], axis=1)
        calibration_max = np.sort(max_stats[args.outer_n :])
        family_nulls[family_label] = {
            "indices": family_indices,
            "max_stats": max_stats,
            "calibration_max": calibration_max,
            "max_null_mean": float(np.mean(calibration_max)),
            "max_null_sd": float(np.std(calibration_max, ddof=1))
            if calibration_max.size >= 2
            else float("nan"),
        }
    bins = assign_degree_bins(scores["graph_degree"], n_bins=args.degree_bins).to_numpy(dtype=int)
    bin_to_indices = {int(bin_id): np.flatnonzero(bins == bin_id) for bin_id in sorted(np.unique(bins))}
    profiles = degree_profiles(module_indices, bins)

    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed + 7777)
    for outer_idx in range(args.outer_n):
        observed_stats = stats[outer_idx, :]
        for family_label, family_info in family_nulls.items():
            family_indices = np.asarray(family_info["indices"], dtype=int)
            calibration_max = np.asarray(family_info["calibration_max"], dtype=float)
            max_stats = np.asarray(family_info["max_stats"], dtype=float)
            max_null_mean = float(family_info["max_null_mean"])
            max_null_sd = float(family_info["max_null_sd"])
            family_observed = observed_stats[family_indices]
            family_p = familywise_p_from_sorted_max(calibration_max, family_observed)
            candidate_local = np.flatnonzero(family_p <= args.alpha)
            if candidate_local.size > args.max_candidates_per_outer:
                order = np.argsort(family_observed[candidate_local])[::-1]
                candidate_local = candidate_local[order[: args.max_candidates_per_outer]]
            candidate_indices = family_indices[candidate_local]
            degree_pass = False
            best_degree_p = float("nan")
            best_degree_module = ""
            best_local_idx = int(np.argmax(family_observed))
            best_idx = int(family_indices[best_local_idx])
            for module_idx in candidate_indices:
                degree_null = degree_matched_stat_null(
                    score_null[outer_idx, :],
                    bin_to_indices,
                    profiles[int(module_idx)],
                    n_null=args.degree_matched_null_n,
                    rng=rng,
                )
                degree_p = empirical_upper(degree_null, float(observed_stats[int(module_idx)]))
                if not np.isfinite(best_degree_p) or degree_p < best_degree_p:
                    best_degree_p = float(degree_p)
                    best_degree_module = str(modules.iloc[int(module_idx)]["module_name"])
                if degree_p <= args.alpha:
                    degree_pass = True
            family_p_for_max = float(family_p[best_local_idx])
            max_z = (
                float((max_stats[outer_idx] - max_null_mean) / max_null_sd)
                if np.isfinite(max_null_sd) and max_null_sd > 0
                else float("nan")
            )
            rows.append(
                {
                    "trait": run.trait,
                    "analysis_id": run.analysis_id,
                    "calibration_family": family_label,
                    "n_modules_in_family": int(family_indices.size),
                    "outer_replicate": int(outer_idx),
                    "best_module_id": str(modules.iloc[best_idx]["module_id"]),
                    "best_module_name": str(modules.iloc[best_idx]["module_name"]),
                    "best_module_category": str(modules.iloc[best_idx].get("module_category", "")),
                    "best_module_source": str(modules.iloc[best_idx].get("module_source", "")),
                    "max_library_stat": float(max_stats[outer_idx]),
                    "max_library_null_mean": max_null_mean,
                    "max_library_null_sd": max_null_sd,
                    "max_library_z": max_z,
                    "best_library_familywise_p": family_p_for_max,
                    "n_familywise_candidate_modules": int(candidate_indices.size),
                    "best_degree_matched_p_among_familywise_candidates": best_degree_p,
                    "best_degree_module_among_familywise_candidates": best_degree_module,
                    "any_familywise_positive": bool(candidate_indices.size > 0),
                    "any_familywise_and_degree_positive": bool(degree_pass),
                    "outer_null_type": "degree_stratified_score_permutation",
                    "calibration_null_type": "degree_stratified_score_permutation_library_max",
                    "statistic_name": "max_sqrt_n_mean_residualized_score",
                    "statistic_direction": "greater_is_more_extreme",
                    "alpha": float(args.alpha),
                    "n_outer": int(args.outer_n),
                    "calibration_null_n": int(args.calibration_null_n),
                    "degree_matched_null_n": int(args.degree_matched_null_n),
                    "degree_bins": int(args.degree_bins),
                    "seed": int(seed),
                    "source_result_path": str(run.result_dir),
                    "script_path": str(THIS_SCRIPT),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

    outer = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    for family_label, family_info in family_nulls.items():
        family_outer = outer.loc[outer["calibration_family"].eq(family_label)]
        n_modules_in_family = int(np.asarray(family_info["indices"]).size)
        for flag_col in ["any_familywise_positive", "any_familywise_and_degree_positive"]:
            summary_rows.append(
                summarize_fpr(
                    family_outer,
                    trait=run.trait,
                    analysis_id=run.analysis_id,
                    calibration_family=family_label,
                    n_modules_in_family=n_modules_in_family,
                    flag_col=flag_col,
                    alpha=args.alpha,
                    outer_n=args.outer_n,
                    calibration_null_n=args.calibration_null_n,
                )
            )
    summary = pd.DataFrame(summary_rows)
    summary["degree_matched_null_n"] = int(args.degree_matched_null_n)
    summary["degree_bins"] = int(args.degree_bins)
    summary["source_result_path"] = str(run.result_dir)
    summary["script_path"] = str(THIS_SCRIPT)
    summary["seed"] = int(seed)
    summary["timestamp"] = datetime.now(UTC).isoformat()
    return outer, summary


def render_report(summary: pd.DataFrame, args: argparse.Namespace) -> str:
    lines = [
        "# RIPPLE V1.2 Anchored Module Type I Calibration",
        "",
        f"Created: {datetime.now(UTC).isoformat()}",
        "",
        f"Outer null replicates per analysis: {int(args.outer_n):,}",
        f"Calibration null replicates per analysis: {int(args.calibration_null_n):,}",
        f"Degree-matched nulls for familywise candidates: {int(args.degree_matched_null_n):,}",
        "",
        "This calibration uses degree-stratified score permutations over the fixed anchored library. "
        "It estimates the false-positive rate for anchored familywise evidence and for the stricter "
        "familywise-plus-degree gate.",
        "",
        "| Trait | Analysis | Family | Target | FPR | FP / n | 95% CI | MC SE |",
        "|---|---|---|---|---:|---:|---|---:|",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"| {row['trait']} | {row['analysis_id']} | {row['calibration_family']} | "
            f"{row['calibration_target']} | "
            f"{float(row['fpr']):.4f} | {int(row['false_positive_count'])}/{int(row['n_outer'])} | "
            f"[{float(row['binomial_95ci_low']):.4f}, {float(row['binomial_95ci_high']):.4f}] | "
            f"{float(row['mc_se']):.4f} |"
        )
    pooled = (
        summary.groupby(["calibration_family", "calibration_target"], as_index=False)
        .agg(false_positive_count=("false_positive_count", "sum"), n_outer=("n_outer", "sum"))
    )
    lines.extend(
        [
            "",
            "## Pooled Summary",
            "",
            "| Family | Target | FPR | FP / n | 95% CI |",
            "|---|---|---:|---:|---|",
        ]
    )
    for row in pooled.to_dict(orient="records"):
        count = int(row["false_positive_count"])
        n_outer = int(row["n_outer"])
        fpr = count / n_outer if n_outer else float("nan")
        low, high = wilson_ci(count, n_outer)
        lines.append(
            f"| {row['calibration_family']} | {row['calibration_target']} | "
            f"{fpr:.4f} | {count}/{n_outer} | [{low:.4f}, {high:.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is anchored-layer Type I calibration, not whole-pipeline Type I calibration from SNP-level nulls.",
            "- A well-calibrated fixed-library familywise gate should be close to the nominal alpha within Monte Carlo error.",
            "- The familywise-plus-degree gate is expected to be equal or more conservative.",
            "- These results can support V1.2 anchored candidate-module evidence only after claim policy review.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ensure_out_dir(args.out_dir, force=args.force)
    runs = discover_runs(args.anchored_root)
    outer_tables: list[pd.DataFrame] = []
    summary_tables: list[pd.DataFrame] = []
    for run_index, run in enumerate(runs):
        print(f"Calibrating {run.analysis_id}", flush=True)
        outer, summary = calibrate_run(run, args, run_index=run_index)
        write_table(args.out_dir / "tables" / f"{run.analysis_id}.anchored_type1_outer.tsv.gz", outer)
        outer_tables.append(outer)
        summary_tables.append(summary)
    all_outer = pd.concat(outer_tables, ignore_index=True)
    all_summary = pd.concat(summary_tables, ignore_index=True)
    write_table(args.out_dir / "tables" / "anchored_type1_outer.all_traits.tsv.gz", all_outer)
    write_table(args.out_dir / "tables" / "anchored_type1_summary.tsv", all_summary)
    report = render_report(all_summary, args)
    (args.out_dir / "reports" / "anchored_type1_calibration_report.md").write_text(
        report + "\n",
        encoding="utf-8",
    )
    print(f"Wrote anchored Type I calibration to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
