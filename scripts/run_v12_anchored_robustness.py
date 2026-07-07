#!/usr/bin/env python
"""Summarize robustness of RIPPLE V1.2 broad anchored module diagnostics."""

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


PRIVATE_ROOT = Path("/path/to/ripple_private_workspace")
ANALYSIS_ROOT = PRIVATE_ROOT / "30_analysis"
DEFAULT_N500_ROOT = ANALYSIS_ROOT / "tier4_v12_anchored_broad_reactome_go_cross_trait_n500"
DEFAULT_N50_ROOT = ANALYSIS_ROOT / "tier4_v12_anchored_broad_reactome_go_cross_trait_smoke"
DEFAULT_OUT_DIR = (
    ANALYSIS_ROOT
    / "tier4_v12_anchored_broad_reactome_go_type1_robustness_v1"
    / "robustness"
)
DEFAULT_TYPE1_SUMMARY = (
    ANALYSIS_ROOT
    / "tier4_v12_anchored_broad_reactome_go_type1_robustness_v1"
    / "type1_outer1000"
    / "tables"
    / "anchored_type1_summary.tsv"
)
THIS_SCRIPT = Path(__file__).resolve()


@dataclass(frozen=True)
class AnchoredRun:
    analysis_id: str
    trait: str
    result_dir: Path
    summary_path: Path
    module_table_path: Path
    null_table_path: Path
    score_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n500-root", type=Path, default=DEFAULT_N500_ROOT)
    parser.add_argument("--n50-root", type=Path, default=DEFAULT_N50_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--type1-summary", type=Path, default=DEFAULT_TYPE1_SUMMARY)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--perturb-replicates", type=int, default=200)
    parser.add_argument("--perturb-noise-scale", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260714)
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


def discover_runs(root: Path) -> dict[str, AnchoredRun]:
    runs: dict[str, AnchoredRun] = {}
    for result_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        reports = result_dir / "reports"
        tables = result_dir / "tables"
        summary_paths = sorted(reports.glob("*.v12_anchored_module_summary.json"))
        module_paths = sorted(tables.glob("*.v12_anchored_module_tests.tsv"))
        null_paths = sorted(tables.glob("*.v12_anchored_module_nulls.tsv.gz"))
        if len(summary_paths) != 1 or len(module_paths) != 1 or len(null_paths) != 1:
            continue
        summary = json.loads(summary_paths[0].read_text(encoding="utf-8"))
        score_path = Path(str(summary.get("score_path", "")))
        if not score_path.exists():
            raise FileNotFoundError(f"Missing score_path for {result_dir}: {score_path}")
        runs[result_dir.name] = AnchoredRun(
            analysis_id=result_dir.name,
            trait=str(summary.get("trait", result_dir.name)),
            result_dir=result_dir,
            summary_path=summary_paths[0],
            module_table_path=module_paths[0],
            null_table_path=null_paths[0],
            score_path=score_path,
        )
    if not runs:
        raise FileNotFoundError(f"No anchored runs found under {root}")
    return runs


def split_genes(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [part.strip().upper() for part in str(value).split(",") if part.strip()]


def prepare_modules(path: Path) -> pd.DataFrame:
    modules = pd.read_csv(path, sep="\t")
    tested = modules.loc[modules["module_status"].ne("not_tested_low_overlap")].copy()
    tested["present_gene_list"] = tested["present_genes"].map(split_genes)
    tested = tested[tested["present_gene_list"].map(len).ge(1)].reset_index(drop=True)
    return tested


def prepare_scores(path: Path) -> pd.DataFrame:
    scores = pd.read_csv(path, sep="\t", compression="infer")
    required = {"gene_symbol", "assoc_resid_score"}
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    scores = scores.copy()
    scores["gene_symbol"] = scores["gene_symbol"].astype(str).str.upper()
    scores["assoc_resid_score"] = pd.to_numeric(scores["assoc_resid_score"], errors="raise").astype(float)
    scores = scores.drop_duplicates("gene_symbol", keep="first").reset_index(drop=True)
    return scores


def membership_matrix(scores: pd.DataFrame, modules: pd.DataFrame) -> sparse.csr_matrix:
    gene_to_idx = {str(gene): idx for idx, gene in enumerate(scores["gene_symbol"].astype(str).to_numpy())}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for module_idx, genes in enumerate(modules["present_gene_list"]):
        indices = np.asarray([gene_to_idx[gene] for gene in genes if gene in gene_to_idx], dtype=int)
        if indices.size == 0:
            raise ValueError(f"Module {module_idx} has no score-overlapping genes.")
        rows.extend(int(idx) for idx in indices)
        cols.extend([module_idx] * int(indices.size))
        data.extend([1.0 / np.sqrt(float(indices.size))] * int(indices.size))
    return sparse.csr_matrix(
        (np.asarray(data, dtype=float), (np.asarray(rows), np.asarray(cols))),
        shape=(len(scores), len(modules)),
    )


def empirical_upper_sorted(sorted_null: np.ndarray, observed: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(sorted_null, observed, side="left")
    exceedances = sorted_null.size - positions
    return (1.0 + exceedances.astype(float)) / (1.0 + float(sorted_null.size))


def top_set(table: pd.DataFrame, n: int) -> set[str]:
    tested = table.loc[table["module_status"].ne("not_tested_low_overlap")].copy()
    tested = tested.sort_values(
        ["library_familywise_p", "degree_matched_empirical_p", "observed_value"],
        ascending=[True, True, False],
    )
    return set(tested.head(n)["module_name"].astype(str))


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return float("nan")
    return float(len(left & right) / len(left | right))


def null_scale_stability(n500_runs: dict[str, AnchoredRun], n50_runs: dict[str, AnchoredRun], top_n: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for analysis_id, run500 in sorted(n500_runs.items()):
        if analysis_id not in n50_runs:
            continue
        run50 = n50_runs[analysis_id]
        m500 = pd.read_csv(run500.module_table_path, sep="\t")
        m50 = pd.read_csv(run50.module_table_path, sep="\t")
        tested500 = m500.loc[m500["module_status"].ne("not_tested_low_overlap")].copy()
        tested50 = m50.loc[m50["module_status"].ne("not_tested_low_overlap")].copy()
        tested500 = tested500.sort_values(
            ["library_familywise_p", "degree_matched_empirical_p", "observed_value"],
            ascending=[True, True, False],
        )
        tested50 = tested50.sort_values(
            ["library_familywise_p", "degree_matched_empirical_p", "observed_value"],
            ascending=[True, True, False],
        )
        best500 = tested500.iloc[0]
        best50 = tested50.iloc[0]
        rows.append(
            {
                "trait": run500.trait,
                "analysis_id": analysis_id,
                "best_module_n500": best500["module_name"],
                "best_status_n500": best500["module_status"],
                "best_familywise_p_n500": float(best500["library_familywise_p"]),
                "best_module_n50": best50["module_name"],
                "best_status_n50": best50["module_status"],
                "best_familywise_p_n50": float(best50["library_familywise_p"]),
                "best_module_same": bool(str(best500["module_name"]) == str(best50["module_name"])),
                "top10_jaccard_n50_n500": jaccard(top_set(m500, 10), top_set(m50, 10)),
                f"top{top_n}_jaccard_n50_n500": jaccard(top_set(m500, top_n), top_set(m50, top_n)),
                "source_result_path_n500": str(run500.result_dir),
                "source_result_path_n50": str(run50.result_dir),
                "script_path": str(THIS_SCRIPT),
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )
    return pd.DataFrame(rows)


def group_familywise(run: AnchoredRun) -> pd.DataFrame:
    modules = prepare_modules(run.module_table_path)
    nulls = pd.read_csv(run.null_table_path, sep="\t", compression="infer")
    fixed = nulls.loc[
        nulls["null_type"].eq("degree_stratified_score_permutation_fixed_module"),
        ["module_id", "replicate", "statistic_value"],
    ].copy()
    if fixed.empty:
        return pd.DataFrame()
    matrix = fixed.pivot(index="replicate", columns="module_id", values="statistic_value")
    rows: list[dict[str, object]] = []
    for group_type, column in [("module_source", "module_source"), ("module_category", "module_category")]:
        for group_value, group in modules.groupby(column, observed=True):
            module_ids = [mid for mid in group["module_id"].astype(str) if mid in matrix.columns]
            if not module_ids:
                continue
            sorted_max = np.sort(matrix[module_ids].max(axis=1).to_numpy(dtype=float))
            observed = group["observed_value"].to_numpy(dtype=float)
            group_p = empirical_upper_sorted(sorted_max, observed)
            group_out = group.copy()
            group_out["trait"] = run.trait
            group_out["analysis_id"] = run.analysis_id
            group_out["group_type"] = group_type
            group_out["group_value"] = str(group_value)
            group_out["group_familywise_p"] = group_p
            group_out["n_modules_in_group"] = int(len(module_ids))
            rows.extend(group_out.to_dict(orient="records"))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    keep = [
        "trait",
        "analysis_id",
        "module_id",
        "module_name",
        "group_type",
        "group_value",
        "n_modules_in_group",
        "module_category",
        "module_source",
        "n_present",
        "observed_value",
        "degree_matched_z",
        "degree_matched_empirical_p",
        "library_familywise_p",
        "group_familywise_p",
        "module_status",
    ]
    out = out[keep].sort_values(
        ["trait", "group_type", "group_familywise_p", "library_familywise_p", "observed_value"],
        ascending=[True, True, True, True, False],
    )
    out["source_result_path"] = str(run.result_dir)
    out["script_path"] = str(THIS_SCRIPT)
    out["timestamp"] = datetime.now(UTC).isoformat()
    return out


def go_reactome_confirmation(group_out: pd.DataFrame, type1_summary_path: Path) -> pd.DataFrame:
    if group_out.empty:
        return pd.DataFrame()
    source_rows = group_out.loc[
        group_out["group_type"].eq("module_source")
        & group_out["group_value"].isin(["Gene Ontology", "Reactome"])
    ].copy()
    if source_rows.empty:
        return pd.DataFrame()
    top_rows = (
        source_rows.sort_values(
            ["trait", "analysis_id", "group_value", "group_familywise_p", "library_familywise_p", "observed_value"],
            ascending=[True, True, True, True, True, False],
        )
        .groupby(["trait", "analysis_id", "group_value"], observed=True)
        .head(1)
        .copy()
    )
    top_rows["calibration_family"] = "module_source:" + top_rows["group_value"].str.replace(" ", "_", regex=False)
    type1 = pd.DataFrame()
    if type1_summary_path.exists():
        type1 = pd.read_csv(type1_summary_path, sep="\t")
        type1 = type1.loc[
            type1["calibration_target"].eq("any_familywise_positive")
            & type1["calibration_family"].isin(["module_source:Gene_Ontology", "module_source:Reactome"])
        ].copy()
        type1 = type1[
            [
                "analysis_id",
                "calibration_family",
                "fpr",
                "false_positive_count",
                "n_outer",
                "binomial_95ci_low",
                "binomial_95ci_high",
                "mc_se",
            ]
        ].rename(
            columns={
                "fpr": "type1_fpr_outer1000",
                "false_positive_count": "type1_false_positive_count",
                "n_outer": "type1_n_outer",
                "binomial_95ci_low": "type1_95ci_low",
                "binomial_95ci_high": "type1_95ci_high",
                "mc_se": "type1_mc_se",
            }
        )
    out = top_rows.merge(type1, on=["analysis_id", "calibration_family"], how="left")
    keep = [
        "trait",
        "analysis_id",
        "group_value",
        "module_name",
        "module_category",
        "module_source",
        "n_present",
        "observed_value",
        "degree_matched_z",
        "degree_matched_empirical_p",
        "group_familywise_p",
        "library_familywise_p",
        "module_status",
        "type1_fpr_outer1000",
        "type1_false_positive_count",
        "type1_n_outer",
        "type1_95ci_low",
        "type1_95ci_high",
        "type1_mc_se",
        "source_result_path",
        "script_path",
        "timestamp",
    ]
    available = [column for column in keep if column in out.columns]
    return out[available].sort_values(["trait", "group_value"]).reset_index(drop=True)


def perturbation_robustness(run: AnchoredRun, args: argparse.Namespace, *, run_index: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    modules = prepare_modules(run.module_table_path)
    scores = prepare_scores(run.score_path)
    membership = membership_matrix(scores, modules)
    values = scores["assoc_resid_score"].to_numpy(dtype=float)
    observed_stats = np.asarray(values @ membership, dtype=float)
    observed_order = np.argsort(observed_stats)[::-1]
    observed_top10 = set(modules.iloc[observed_order[:10]]["module_name"].astype(str))
    observed_best = str(modules.iloc[int(observed_order[0])]["module_name"])
    nulls = pd.read_csv(run.null_table_path, sep="\t", compression="infer")
    family_max = nulls.loc[
        nulls["null_type"].eq("degree_stratified_score_permutation_library_max"),
        "statistic_value",
    ].to_numpy(dtype=float)
    sorted_family_max = np.sort(family_max[np.isfinite(family_max)])
    if sorted_family_max.size == 0:
        raise ValueError(f"No familywise max null rows in {run.null_table_path}")
    rng = np.random.default_rng(int(args.seed + run_index * 100_003))
    noise_sd = float(np.std(values, ddof=1) * args.perturb_noise_scale)
    rows: list[dict[str, object]] = []
    for replicate in range(args.perturb_replicates):
        perturbed = values + rng.normal(0.0, noise_sd, size=values.size)
        stats = np.asarray(perturbed @ membership, dtype=float)
        family_p = empirical_upper_sorted(sorted_family_max, stats)
        order = np.argsort(stats)[::-1]
        best_idx = int(order[0])
        top10 = set(modules.iloc[order[:10]]["module_name"].astype(str))
        rows.append(
            {
                "trait": run.trait,
                "analysis_id": run.analysis_id,
                "perturb_replicate": int(replicate),
                "perturb_noise_scale": float(args.perturb_noise_scale),
                "perturb_noise_sd": noise_sd,
                "best_module_name": str(modules.iloc[best_idx]["module_name"]),
                "best_module_category": str(modules.iloc[best_idx].get("module_category", "")),
                "best_module_source": str(modules.iloc[best_idx].get("module_source", "")),
                "best_module_same_as_observed": bool(str(modules.iloc[best_idx]["module_name"]) == observed_best),
                "top10_jaccard_with_observed": jaccard(observed_top10, top10),
                "best_observed_value": float(stats[best_idx]),
                "best_library_familywise_p": float(family_p[best_idx]),
                "any_familywise_positive": bool(np.any(family_p <= args.alpha)),
                "n_familywise_positive": int(np.count_nonzero(family_p <= args.alpha)),
                "source_result_path": str(run.result_dir),
                "script_path": str(THIS_SCRIPT),
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )
    detail = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "trait": run.trait,
                "analysis_id": run.analysis_id,
                "observed_best_module": observed_best,
                "perturb_replicates": int(args.perturb_replicates),
                "perturb_noise_scale": float(args.perturb_noise_scale),
                "best_module_recurrence_rate": float(detail["best_module_same_as_observed"].mean()),
                "mean_top10_jaccard": float(detail["top10_jaccard_with_observed"].mean()),
                "median_top10_jaccard": float(detail["top10_jaccard_with_observed"].median()),
                "familywise_positive_rate": float(detail["any_familywise_positive"].mean()),
                "median_best_familywise_p": float(detail["best_library_familywise_p"].median()),
                "source_result_path": str(run.result_dir),
                "script_path": str(THIS_SCRIPT),
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ]
    )
    return detail, summary


def render_report(
    stability: pd.DataFrame,
    perturb_summary: pd.DataFrame,
    group_top: pd.DataFrame,
    go_reactome: pd.DataFrame,
    args: argparse.Namespace,
) -> str:
    lines = [
        "# RIPPLE V1.2 Anchored Module Robustness",
        "",
        f"Created: {datetime.now(UTC).isoformat()}",
        "",
        "This report summarizes robustness diagnostics for the broad Reactome/GO anchored layer.",
        "",
        "## Null-Scale Stability",
        "",
        "| Trait | Best n500 | Best n50 | Same best | Top10 Jaccard | Best family P n500 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in stability.to_dict(orient="records"):
        lines.append(
            f"| {row['trait']} | {row['best_module_n500']} | {row['best_module_n50']} | "
            f"{row['best_module_same']} | {float(row['top10_jaccard_n50_n500']):.3f} | "
            f"{float(row['best_familywise_p_n500']):.4g} |"
        )
    lines.extend(
        [
            "",
            "## Score Perturbation Robustness",
            "",
            f"Perturbation noise scale: {float(args.perturb_noise_scale):.3f} x residual-score SD.",
            "",
            "| Trait | Observed best | Best recurrence | Mean top10 Jaccard | Familywise positive rate | Median best family P |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in perturb_summary.to_dict(orient="records"):
        lines.append(
            f"| {row['trait']} | {row['observed_best_module']} | "
            f"{float(row['best_module_recurrence_rate']):.3f} | {float(row['mean_top10_jaccard']):.3f} | "
            f"{float(row['familywise_positive_rate']):.3f} | {float(row['median_best_familywise_p']):.4g} |"
        )
    lines.extend(
        [
            "",
            "## Top Group-Familywise Rows",
            "",
            "| Trait | Group | Module | Group family P | Library family P | Status |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for row in group_top.head(20).to_dict(orient="records"):
        lines.append(
            f"| {row['trait']} | {row['group_type']}={row['group_value']} | {row['module_name']} | "
            f"{float(row['group_familywise_p']):.4g} | {float(row['library_familywise_p']):.4g} | "
            f"{row['module_status']} |"
        )
    if not go_reactome.empty:
        lines.extend(
            [
                "",
                "## GO-Only / Reactome-Only Confirmation",
                "",
                "| Trait | Source | Best module | Source-family P | Full-library P | Type I FPR |",
                "|---|---|---|---:|---:|---:|",
            ]
        )
        for row in go_reactome.to_dict(orient="records"):
            type1_fpr = row.get("type1_fpr_outer1000", float("nan"))
            lines.append(
                f"| {row['trait']} | {row['group_value']} | {row['module_name']} | "
                f"{float(row['group_familywise_p']):.4g} | {float(row['library_familywise_p']):.4g} | "
                f"{float(type1_fpr):.4f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- n50-to-n500 stability checks whether smoke-scale top modules survive a larger null run.",
            "- Group-familywise P values test whether signals are driven by one source/category or remain visible within that subset.",
            "- Score perturbation assesses descriptive stability; it does not upgrade a module claim tier by itself.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ensure_out_dir(args.out_dir, force=args.force)
    runs500 = discover_runs(args.n500_root)
    runs50 = discover_runs(args.n50_root) if args.n50_root.exists() else {}
    stability = null_scale_stability(runs500, runs50, args.top_n) if runs50 else pd.DataFrame()
    group_tables: list[pd.DataFrame] = []
    perturb_details: list[pd.DataFrame] = []
    perturb_summaries: list[pd.DataFrame] = []
    for run_index, run in enumerate(runs500.values()):
        print(f"Robustness diagnostics for {run.analysis_id}", flush=True)
        group = group_familywise(run)
        if not group.empty:
            group_tables.append(group)
        detail, summary = perturbation_robustness(run, args, run_index=run_index)
        perturb_details.append(detail)
        perturb_summaries.append(summary)
    group_out = pd.concat(group_tables, ignore_index=True) if group_tables else pd.DataFrame()
    detail_out = pd.concat(perturb_details, ignore_index=True)
    summary_out = pd.concat(perturb_summaries, ignore_index=True)
    if not stability.empty:
        write_table(args.out_dir / "tables" / "anchored_null_scale_stability.tsv", stability)
    write_table(args.out_dir / "tables" / "anchored_group_familywise.tsv.gz", group_out)
    go_reactome = go_reactome_confirmation(group_out, args.type1_summary)
    if not go_reactome.empty:
        write_table(args.out_dir / "tables" / "anchored_go_reactome_confirmation.tsv", go_reactome)
    write_table(args.out_dir / "tables" / "anchored_score_perturbation_detail.tsv.gz", detail_out)
    write_table(args.out_dir / "tables" / "anchored_score_perturbation_summary.tsv", summary_out)
    group_top = group_out.sort_values(
        ["group_familywise_p", "library_familywise_p", "observed_value"],
        ascending=[True, True, False],
    )
    report = render_report(stability, summary_out, group_top, go_reactome, args)
    (args.out_dir / "reports" / "anchored_robustness_report.md").write_text(
        report + "\n",
        encoding="utf-8",
    )
    print(f"Wrote anchored robustness diagnostics to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
