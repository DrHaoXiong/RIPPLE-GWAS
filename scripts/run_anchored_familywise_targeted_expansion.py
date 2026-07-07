#!/usr/bin/env python
"""Targeted high-null expansion for anchored module familywise evidence.

The original V1.2 anchored cross-trait run used n=500 library-max nulls. This
script recomputes the max-over-library score-permutation null at higher scale
and reports expanded familywise P values for manuscript-top, familywise-positive
and borderline anchored modules. It avoids writing per-module null matrices.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

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
DEFAULT_OUT_DIR = ANALYSIS_ROOT / "review_driven_revision_v1" / "anchored_familywise_targeted_expansion"
DEFAULT_SUPPLEMENT_DIR = MANUSCRIPT_ROOT / "supplementary_files"

TRAIT_TO_RUN_DIR = {
    "BMI_IRN": ANCHOR_ROOT / "BMI_IRN",
    "DR_MVP": ANCHOR_ROOT / "DR_MVP",
    "HEIGHT_IRN": ANCHOR_ROOT / "HEIGHT_IRN",
    "SCZ": ANCHOR_ROOT / "SCZ_NO_MHC",
    "T2D": ANCHOR_ROOT / "T2D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchored-root", type=Path, default=ANCHOR_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--supplement-dir", type=Path, default=DEFAULT_SUPPLEMENT_DIR)
    parser.add_argument("--traits", nargs="+", default=list(TRAIT_TO_RUN_DIR))
    parser.add_argument("--n-null", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--degree-bins", type=int, default=10)
    parser.add_argument("--borderline-p", type=float, default=0.10)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--copy-to-supplement", action="store_true", default=True)
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def write_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False, compression="infer")


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t", compression="infer")


def discover_run_dir(trait: str, anchored_root: Path) -> Path:
    if trait in TRAIT_TO_RUN_DIR:
        path = TRAIT_TO_RUN_DIR[trait]
        if anchored_root != ANCHOR_ROOT:
            path = anchored_root / path.name
        if path.exists():
            return path
    candidates = sorted(path for path in anchored_root.iterdir() if path.is_dir() and path.name.upper().startswith(trait.upper()))
    if not candidates:
        raise FileNotFoundError(f"No anchored run directory found for {trait} under {anchored_root}")
    return candidates[0]


def load_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    tables = run_dir / "tables"
    reports = run_dir / "reports"
    module_path = next(tables.glob("*.v12_anchored_module_tests.tsv"))
    summary_path = next(reports.glob("*.v12_anchored_module_summary.json"))
    modules = read_table(module_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    score_path = Path(summary["score_path"])
    scores = read_table(score_path)
    return modules, scores, summary


def selected_modules(modules: pd.DataFrame, *, top_n: int, borderline_p: float) -> pd.DataFrame:
    tested = modules.loc[modules["module_status"].astype(str).ne("not_tested_low_overlap")].copy()
    tested["library_familywise_p"] = pd.to_numeric(tested["library_familywise_p"], errors="coerce")
    tested["anchored_module_rank"] = pd.to_numeric(tested["anchored_module_rank"], errors="coerce")
    selected = tested.loc[
        tested["module_status"].astype(str).eq("anchored_familywise_supported")
        | tested["library_familywise_p"].le(float(borderline_p))
        | tested["anchored_module_rank"].le(int(top_n))
    ].copy()
    return selected.sort_values(["anchored_module_rank", "library_familywise_p"]).drop_duplicates("module_id")


def module_matrix(
    modules: pd.DataFrame,
    genes: np.ndarray,
) -> tuple[sparse.csr_matrix, list[str]]:
    gene_to_idx = {str(gene).upper(): idx for idx, gene in enumerate(genes)}
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    module_ids: list[str] = []
    for col, item in enumerate(modules.to_dict(orient="records")):
        present = [gene for gene in str(item.get("present_genes", "")).split(",") if gene]
        idx = [gene_to_idx[gene.upper()] for gene in present if gene.upper() in gene_to_idx]
        if not idx:
            continue
        module_ids.append(str(item["module_id"]))
        weight = 1.0 / np.sqrt(len(idx))
        rows.extend(idx)
        cols.extend([len(module_ids) - 1] * len(idx))
        vals.extend([weight] * len(idx))
    matrix = sparse.csr_matrix((vals, (rows, cols)), shape=(len(genes), len(module_ids)))
    return matrix, module_ids


def permute_batch(
    values: np.ndarray,
    bins: np.ndarray,
    *,
    rng: np.random.Generator,
    batch_size: int,
) -> np.ndarray:
    out = np.empty((batch_size, len(values)), dtype=float)
    unique_bins = np.unique(bins)
    for row in range(batch_size):
        permuted = values.copy()
        for bin_id in unique_bins:
            idx = np.flatnonzero(bins == bin_id)
            if idx.size > 1:
                permuted[idx] = values[rng.permutation(idx)]
        out[row, :] = permuted
    return out


def empirical_upper_from_sorted(sorted_null: np.ndarray, observed: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(sorted_null, observed, side="left")
    exceed = sorted_null.size - idx
    return (1 + exceed.astype(float)) / (1 + sorted_null.size)


def stable_seed_offset(text: str, modulo: int = 100_000) -> int:
    """Deterministic seed offset independent of Python's randomized hash seed."""

    return sum((idx + 1) * ord(char) for idx, char in enumerate(text)) % modulo


def run_trait(args: argparse.Namespace, trait: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_dir = discover_run_dir(trait, args.anchored_root)
    modules, scores, summary = load_run(run_dir)
    tested = modules.loc[modules["module_status"].astype(str).ne("not_tested_low_overlap")].copy()
    selected = selected_modules(tested, top_n=args.top_n, borderline_p=args.borderline_p)
    if selected.empty:
        return pd.DataFrame(), pd.DataFrame()

    scores = scores.drop_duplicates("gene_symbol").copy()
    scores["gene_symbol"] = scores["gene_symbol"].astype(str).str.upper()
    values = pd.to_numeric(scores["assoc_resid_score"], errors="raise").to_numpy(dtype=float)
    genes = scores["gene_symbol"].to_numpy(dtype=object)
    if "graph_degree" not in scores.columns:
        raise ValueError(f"graph_degree missing from {summary['score_path']}")
    bins = assign_degree_bins(pd.to_numeric(scores["graph_degree"], errors="raise"), n_bins=args.degree_bins).to_numpy(dtype=int)

    all_matrix, all_ids = module_matrix(tested, genes)
    selected_matrix, selected_ids = module_matrix(selected, genes)
    if all_matrix.shape[1] == 0 or selected_matrix.shape[1] == 0:
        return pd.DataFrame(), pd.DataFrame()

    selected_lookup = selected.set_index("module_id")
    selected_observed = np.asarray(values @ selected_matrix).ravel()
    rng = np.random.default_rng(args.seed + stable_seed_offset(trait))
    family_max = np.empty(args.n_null, dtype=float)
    selected_null = np.empty((args.n_null, len(selected_ids)), dtype=float)
    cursor = 0
    while cursor < args.n_null:
        current = min(args.batch_size, args.n_null - cursor)
        batch = permute_batch(values, bins, rng=rng, batch_size=current)
        all_stats = batch @ all_matrix
        selected_stats = batch @ selected_matrix
        family_max[cursor : cursor + current] = np.asarray(all_stats.max(axis=1)).ravel()
        selected_null[cursor : cursor + current, :] = np.asarray(selected_stats)
        cursor += current

    sorted_family = np.sort(family_max)
    expanded_family_p = empirical_upper_from_sorted(sorted_family, selected_observed)
    rows: list[dict[str, Any]] = []
    for idx, module_id in enumerate(selected_ids):
        source = selected_lookup.loc[module_id]
        fixed_null = selected_null[:, idx]
        fixed_mean = float(np.mean(fixed_null))
        fixed_sd = float(np.std(fixed_null, ddof=1))
        observed = float(selected_observed[idx])
        fixed_z = float((observed - fixed_mean) / fixed_sd) if fixed_sd > 0 else float("nan")
        fixed_p = float((1 + np.count_nonzero(fixed_null >= observed)) / (1 + len(fixed_null)))
        family_p = float(expanded_family_p[idx])
        rows.append(
            {
                "trait": trait if trait != "SCZ" else "SCZ",
                "analysis_id": summary.get("analysis_id", trait),
                "module_id": module_id,
                "module_name": source.get("module_name", ""),
                "module_source": source.get("module_source", ""),
                "module_category": source.get("module_category", ""),
                "annotation_source_type": source.get("annotation_source_type", ""),
                "n_present": source.get("n_present", ""),
                "observed_value": observed,
                "prior_library_familywise_p_n500": source.get("library_familywise_p", ""),
                "expanded_library_familywise_p": family_p,
                "expanded_fixed_score_permutation_p": fixed_p,
                "expanded_fixed_score_permutation_z": fixed_z,
                "expanded_n_null": int(args.n_null),
                "mc_se_familywise_p": float(np.sqrt(family_p * (1.0 - family_p) / (args.n_null + 1))),
                "expanded_status": "expanded_familywise_supported" if family_p <= 0.05 else "expanded_familywise_negative",
                "original_module_status": source.get("module_status", ""),
                "selection_rule": f"top_n<={args.top_n};prior_familywise_p<={args.borderline_p};or_original_familywise_supported",
                "statistic_name": "sqrt_n_mean_residualized_score",
                "statistic_direction": "greater_is_more_extreme",
                "null_type": "degree_stratified_score_permutation_library_max_targeted_expansion",
                "source_result_path": str(args.out_dir / "anchored_familywise_targeted_expansion_summary.tsv"),
                "script_path": str(Path(__file__).resolve()),
                "seed": int(args.seed),
                "timestamp": now_utc(),
            }
        )
    null_rows = pd.DataFrame(
        {
            "trait": trait,
            "analysis_id": summary.get("analysis_id", trait),
            "replicate": np.arange(args.n_null),
            "null_type": "degree_stratified_score_permutation_library_max",
            "statistic_name": "max_sqrt_n_mean_residualized_score",
            "statistic_value": family_max,
        }
    )
    return pd.DataFrame(rows), null_rows


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[pd.DataFrame] = []
    nulls: list[pd.DataFrame] = []
    for trait in args.traits:
        summary, null = run_trait(args, trait)
        if not summary.empty:
            summaries.append(summary)
        if not null.empty:
            nulls.append(null)
    summary_all = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    null_all = pd.concat(nulls, ignore_index=True) if nulls else pd.DataFrame()
    summary_path = args.out_dir / "anchored_familywise_targeted_expansion_summary.tsv"
    null_path = args.out_dir / "anchored_familywise_targeted_expansion_family_max_null.tsv.gz"
    write_table(summary_path, summary_all)
    write_table(null_path, null_all)
    manifest = {
        "created_utc": now_utc(),
        "script_path": str(Path(__file__).resolve()),
        "anchored_root": str(args.anchored_root),
        "traits": args.traits,
        "n_null": int(args.n_null),
        "summary": str(summary_path),
        "family_max_null": str(null_path),
        "scope": "targeted selected-module report with full-library max null recalculated for each trait",
    }
    (args.out_dir / "anchored_familywise_targeted_expansion_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    if args.copy_to_supplement:
        write_table(args.supplement_dir / "anchored_familywise_targeted_expansion_summary.tsv", summary_all)
    print(f"Wrote anchored targeted familywise expansion to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
