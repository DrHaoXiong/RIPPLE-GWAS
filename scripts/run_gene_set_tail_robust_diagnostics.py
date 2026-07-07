#!/usr/bin/env python
"""Run tail-robust diagnostics for RIPPLE gene-set/module enrichment.

This script is a claim-gating layer for anchored modules, pathway sets and
cell-type marker sets. It does not change gene-level RIPPLE scores.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

PRIVATE_ROOT = Path("/path/to/ripple_private_workspace")
ANALYSIS_ROOT = PRIVATE_ROOT / "30_analysis"
DEFAULT_OUT_DIR = ANALYSIS_ROOT / "gene_set_tail_robust_diagnostics_v0_1"

TRAITS = {
    "DR_MVP": {
        "trait": "DR_MVP",
        "analysis_dir": "dr_mvp_string_final5000",
        "score_file": "DR_MVP.lcc_gene_scores.1000G_LD.residualized.tsv.gz",
        "role": "primary_dr",
    },
    "DR_MVP_NO_MHC_NO_APOE": {
        "trait": "DR_MVP_NO_MHC_NO_APOE",
        "analysis_dir": "dr_mvp_no_mhc_no_apoe_final5000",
        "score_file": "DR_MVP_NO_MHC_NO_APOE.lcc_gene_scores.1000G_LD.residualized.tsv.gz",
        "role": "primary_dr_sensitivity",
    },
    "DR_UKB_CAI_2026": {
        "trait": "DR_UKB_CAI_2026",
        "analysis_dir": "dr_ukb_cai_2026_analysis_ready",
        "score_file": "DR_UKB_CAI_2026.lcc_gene_scores.1000G_LD.residualized.tsv.gz",
        "role": "independent_dr_sensitivity",
    },
    "T2D": {
        "trait": "T2D",
        "analysis_dir": "t2d_analysis_ready",
        "score_file": "T2D.lcc_gene_scores.1000G_LD.residualized.tsv.gz",
        "role": "diabetic_liability_comparator",
    },
    "BMI_IRN": {
        "trait": "BMI_IRN",
        "analysis_dir": "bmi_irn_analysis_ready",
        "score_file": "BMI_IRN.lcc_gene_scores.1000G_LD.residualized.tsv.gz",
        "role": "metabolic_comparator",
    },
    "HEIGHT_IRN": {
        "trait": "HEIGHT_IRN",
        "analysis_dir": "height_irn_analysis_ready",
        "score_file": "HEIGHT_IRN.lcc_gene_scores.1000G_LD.residualized.tsv.gz",
        "role": "non_dr_anthropometric_comparator",
    },
    "SCZ": {
        "trait": "SCZ",
        "analysis_dir": "scz_no_mhc_string_final5000",
        "score_file": "SCZ.lcc_gene_scores.1000G_LD.residualized.tsv.gz",
        "role": "non_ocular_polygenic_comparator",
    },
}

STATISTICS = (
    "mean_score",
    "trimmed_mean_10pct",
    "winsorized_mean_p99",
    "winsorized_mean_p995",
    "rank_mean",
    "leave_top_1_mean",
    "leave_top_5_mean",
    "leave_top_10_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene-set-file", type=Path, required=True)
    parser.add_argument("--analysis-label", required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--traits", nargs="*", default=list(TRAITS))
    parser.add_argument("--n-null", type=int, default=5000)
    parser.add_argument("--degree-bins", type=int, default=10)
    parser.add_argument("--min-overlap", type=int, default=5)
    parser.add_argument("--trim-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--include-low-information", action="store_true")
    parser.add_argument("--include-special-region", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def stable_offset(text: str, modulo: int = 100_000) -> int:
    return sum((idx + 1) * ord(char) for idx, char in enumerate(text)) % modulo


def write_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False)


def load_gene_sets(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = pd.read_csv(path, sep="\t")
    required = {"gene_set", "gene_symbol"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    table = table.copy()
    table["gene_set"] = table["gene_set"].astype(str)
    table["gene_symbol"] = table["gene_symbol"].astype(str).str.upper()
    meta_candidates = [
        "dataset_id",
        "tissue_context",
        "condition_scope",
        "cell_type",
        "module_source",
        "annotation_source_type",
        "module_category",
        "panel_role",
    ]
    meta_cols = ["gene_set"] + [column for column in meta_candidates if column in table.columns]
    summary = (
        table.groupby(meta_cols, dropna=False, observed=True)
        .agg(n_query_genes=("gene_symbol", "nunique"))
        .reset_index()
    )
    return table, summary


def load_scores(config: dict[str, str], args: argparse.Namespace) -> tuple[pd.DataFrame, Path]:
    path = ANALYSIS_ROOT / config["analysis_dir"] / "tables" / config["score_file"]
    if not path.exists():
        raise FileNotFoundError(path)
    scores = pd.read_csv(path, sep="\t")
    required = {"gene_symbol", "assoc_resid_score", "graph_degree"}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    scores = scores.copy()
    scores["gene_symbol"] = scores["gene_symbol"].astype(str).str.upper()
    scores["assoc_resid_score"] = pd.to_numeric(scores["assoc_resid_score"], errors="coerce")
    scores["graph_degree"] = pd.to_numeric(scores["graph_degree"], errors="coerce").fillna(0.0)
    for column in ["is_low_information", "is_special_region"]:
        if column not in scores.columns:
            scores[column] = False
        scores[column] = scores[column].fillna(False).astype(bool)
    scores = scores.dropna(subset=["assoc_resid_score"]).drop_duplicates("gene_symbol")
    if not args.include_low_information:
        scores = scores.loc[~scores["is_low_information"]].copy()
    if not args.include_special_region:
        scores = scores.loc[~scores["is_special_region"]].copy()
    scores = add_degree_bins(scores, args.degree_bins)
    scores = add_rank_percentile(scores)
    return scores.reset_index(drop=True), path


def add_degree_bins(scores: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    scores = scores.copy()
    ranks = scores["graph_degree"].rank(method="first")
    scores["degree_bin"] = pd.qcut(ranks, q=min(n_bins, len(scores)), labels=False, duplicates="drop").astype(int)
    return scores


def add_rank_percentile(scores: pd.DataFrame) -> pd.DataFrame:
    scores = scores.copy()
    scores["score_rank_percentile"] = scores["assoc_resid_score"].rank(method="average", pct=True)
    return scores


def trim_values(values: np.ndarray, fraction: float) -> np.ndarray:
    values = np.sort(np.asarray(values, dtype=float))
    n = values.shape[0]
    trim_n = int(np.floor(n * fraction))
    if trim_n == 0 or n <= 2 * trim_n:
        return values
    return values[trim_n : n - trim_n]


def stat_value(
    values: np.ndarray,
    rank_values: np.ndarray,
    statistic: str,
    *,
    trim_fraction: float,
    cap_p99: float,
    cap_p995: float,
) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    if statistic == "mean_score":
        return float(np.mean(values))
    if statistic == "trimmed_mean_10pct":
        return float(np.mean(trim_values(values, trim_fraction)))
    if statistic == "winsorized_mean_p99":
        return float(np.mean(np.minimum(values, cap_p99)))
    if statistic == "winsorized_mean_p995":
        return float(np.mean(np.minimum(values, cap_p995)))
    if statistic == "rank_mean":
        return float(np.mean(rank_values))
    if statistic.startswith("leave_top_") and statistic.endswith("_mean"):
        k = int(statistic.removeprefix("leave_top_").removesuffix("_mean"))
        sorted_values = np.sort(values)[::-1]
        if sorted_values.size <= k:
            return float("nan")
        return float(np.mean(sorted_values[k:]))
    raise ValueError(f"Unknown statistic: {statistic}")


def empirical_p(observed: float, null_values: np.ndarray) -> float:
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]
    if not np.isfinite(observed) or null_values.size == 0:
        return float("nan")
    return float((1 + np.sum(null_values >= observed)) / (1 + null_values.size))


def degree_matched_null_statistics(
    universe: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    n_null: int,
    rng: np.random.Generator,
    trim_fraction: float,
    cap_p99: float,
    cap_p995: float,
) -> dict[str, np.ndarray]:
    bins = selected["degree_bin"].value_counts().to_dict()
    score_blocks: list[np.ndarray] = []
    rank_blocks: list[np.ndarray] = []
    pools = {
        int(bin_id): (
            group["assoc_resid_score"].to_numpy(dtype=float),
            group["score_rank_percentile"].to_numpy(dtype=float),
        )
        for bin_id, group in universe.groupby("degree_bin", observed=True)
    }
    for bin_id, count in bins.items():
        score_pool, rank_pool = pools[int(bin_id)]
        sampled_positions = rng.integers(0, score_pool.shape[0], size=(n_null, int(count)))
        score_blocks.append(score_pool[sampled_positions])
        rank_blocks.append(rank_pool[sampled_positions])
    values = np.concatenate(score_blocks, axis=1)
    ranks = np.concatenate(rank_blocks, axis=1)
    sorted_ascending = np.sort(values, axis=1)
    sorted_descending = sorted_ascending[:, ::-1]
    n_genes = values.shape[1]
    trim_n = int(np.floor(n_genes * trim_fraction))
    if trim_n > 0 and n_genes > 2 * trim_n:
        trimmed = sorted_ascending[:, trim_n : n_genes - trim_n]
    else:
        trimmed = values
    nulls: dict[str, np.ndarray] = {
        "mean_score": values.mean(axis=1),
        "trimmed_mean_10pct": trimmed.mean(axis=1),
        "winsorized_mean_p99": np.minimum(values, cap_p99).mean(axis=1),
        "winsorized_mean_p995": np.minimum(values, cap_p995).mean(axis=1),
        "rank_mean": ranks.mean(axis=1),
    }
    for k in [1, 5, 10]:
        statistic = f"leave_top_{k}_mean"
        if n_genes <= k:
            nulls[statistic] = np.full(n_null, np.nan)
        else:
            nulls[statistic] = sorted_descending[:, k:].mean(axis=1)
    return nulls


def positive_contribution(values: np.ndarray, top_n: int) -> float:
    positive = np.asarray(values, dtype=float)
    positive = positive[positive > 0]
    denominator = float(positive.sum())
    if denominator <= 0:
        return float("nan")
    numerator = float(np.sort(positive)[::-1][:top_n].sum())
    return numerator / denominator


def run_trait(
    trait_key: str,
    gene_sets: pd.DataFrame,
    gene_set_summary: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = TRAITS[trait_key]
    universe, score_path = load_scores(config, args)
    score_by_gene = universe.set_index("gene_symbol", drop=False)
    cap_p99 = float(universe["assoc_resid_score"].quantile(0.99))
    cap_p995 = float(universe["assoc_resid_score"].quantile(0.995))
    stat_rows: list[dict[str, object]] = []
    tail_rows: list[dict[str, object]] = []
    for row in gene_set_summary.to_dict(orient="records"):
        gene_set = str(row["gene_set"])
        genes = set(gene_sets.loc[gene_sets["gene_set"].eq(gene_set), "gene_symbol"])
        present_genes = sorted(genes.intersection(set(score_by_gene.index)))
        base = {
            "trait": config["trait"],
            "analysis_id": trait_key,
            "trait_role": config["role"],
            "analysis_label": args.analysis_label,
            "gene_set": gene_set,
            "n_query_genes": int(row.get("n_query_genes", len(genes))),
            "n_present_genes": len(present_genes),
            "score_path": str(score_path),
            "gene_set_file": str(args.gene_set_file),
        }
        for key, value in row.items():
            if key not in base:
                base[key] = value
        if len(present_genes) < args.min_overlap:
            tail_rows.append({**base, **empty_tail("low_gene_overlap")})
            for statistic in STATISTICS:
                stat_rows.append({**base, **empty_stat(statistic, args.n_null, "low_gene_overlap")})
            continue
        selected = score_by_gene.loc[present_genes].copy()
        rng = np.random.default_rng(args.seed + stable_offset(f"{trait_key}:{gene_set}"))
        nulls = degree_matched_null_statistics(
            universe,
            selected,
            n_null=args.n_null,
            rng=rng,
            trim_fraction=args.trim_fraction,
            cap_p99=cap_p99,
            cap_p995=cap_p995,
        )
        values = selected["assoc_resid_score"].to_numpy(dtype=float)
        rank_values = selected["score_rank_percentile"].to_numpy(dtype=float)
        top = selected.sort_values("assoc_resid_score", ascending=False)
        tail_rows.append(
            {
                **base,
                "max_gene_score": float(np.max(values)),
                "max_gene_symbol": str(top.iloc[0]["gene_symbol"]),
                "top1_positive_score_contribution": positive_contribution(values, 1),
                "top5_positive_score_contribution": positive_contribution(values, 5),
                "top10_positive_score_contribution": positive_contribution(values, 10),
                "top5_gene_symbols": ",".join(top["gene_symbol"].astype(str).head(5)),
                "top5_gene_scores": ",".join(f"{value:.5g}" for value in top["assoc_resid_score"].head(5)),
                "tail_diagnostic_status": tail_status(values),
                "exclusion_or_na_reason": "none",
            }
        )
        for statistic in STATISTICS:
            observed = stat_value(
                values,
                rank_values,
                statistic,
                trim_fraction=args.trim_fraction,
                cap_p99=cap_p99,
                cap_p995=cap_p995,
            )
            null_values = nulls[statistic]
            finite_null = null_values[np.isfinite(null_values)]
            null_mean = float(np.mean(finite_null)) if finite_null.size else float("nan")
            null_sd = float(np.std(finite_null, ddof=1)) if finite_null.size > 1 else float("nan")
            z = (observed - null_mean) / null_sd if null_sd > 0 else float("nan")
            stat_rows.append(
                {
                    **base,
                    "statistic_name": statistic,
                    "observed_value": observed,
                    "null_mean": null_mean,
                    "null_sd": null_sd,
                    "z": z,
                    "empirical_p": empirical_p(observed, null_values),
                    "n_null": int(args.n_null),
                    "statistic_direction": "greater_is_more_extreme",
                    "exclusion_or_na_reason": "none",
                    "script_path": str(Path(__file__).resolve()),
                    "seed": int(args.seed + stable_offset(f"{trait_key}:{gene_set}")),
                    "timestamp": now_utc(),
                }
            )
    stat_table = add_fdr(pd.DataFrame(stat_rows))
    claim_table = build_claim_table(stat_table, pd.DataFrame(tail_rows))
    return stat_table, claim_table


def empty_tail(reason: str) -> dict[str, object]:
    return {
        "max_gene_score": float("nan"),
        "max_gene_symbol": "",
        "top1_positive_score_contribution": float("nan"),
        "top5_positive_score_contribution": float("nan"),
        "top10_positive_score_contribution": float("nan"),
        "top5_gene_symbols": "",
        "top5_gene_scores": "",
        "tail_diagnostic_status": "not_tested",
        "exclusion_or_na_reason": reason,
    }


def empty_stat(statistic: str, n_null: int, reason: str) -> dict[str, object]:
    return {
        "statistic_name": statistic,
        "observed_value": float("nan"),
        "null_mean": float("nan"),
        "null_sd": float("nan"),
        "z": float("nan"),
        "empirical_p": float("nan"),
        "empirical_p_fdr_bh": float("nan"),
        "n_null": int(n_null),
        "statistic_direction": "greater_is_more_extreme",
        "exclusion_or_na_reason": reason,
        "script_path": str(Path(__file__).resolve()),
        "seed": "",
        "timestamp": now_utc(),
    }


def tail_status(values: np.ndarray) -> str:
    top1 = positive_contribution(values, 1)
    top5 = positive_contribution(values, 5)
    if np.isfinite(top1) and top1 >= 0.50:
        return "top1_dominated"
    if np.isfinite(top5) and top5 >= 0.80:
        return "top5_dominated"
    if np.isfinite(top5) and top5 >= 0.50:
        return "moderate_tail_concentration"
    return "not_tail_dominated"


def add_fdr(stat_table: pd.DataFrame) -> pd.DataFrame:
    stat_table = stat_table.copy()
    stat_table["empirical_p_fdr_bh"] = np.nan
    for (_, statistic), index in stat_table.groupby(["trait", "statistic_name"], observed=True).groups.items():
        tested = stat_table.loc[index]
        mask = tested["empirical_p"].notna().to_numpy()
        tested_index = tested.index[mask]
        p_values = tested.loc[tested_index, "empirical_p"].to_numpy(dtype=float)
        stat_table.loc[tested_index, "empirical_p_fdr_bh"] = bh_fdr(p_values)
    return stat_table


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    n = p_values.shape[0]
    if n == 0:
        return p_values
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def positive_row(row: pd.Series, *, require_fdr: bool = True, z_threshold: float = 2.5) -> bool:
    z = pd.to_numeric(pd.Series([row.get("z")]), errors="coerce").iloc[0]
    p = pd.to_numeric(pd.Series([row.get("empirical_p")]), errors="coerce").iloc[0]
    q = pd.to_numeric(pd.Series([row.get("empirical_p_fdr_bh")]), errors="coerce").iloc[0]
    if not np.isfinite(z) or not np.isfinite(p):
        return False
    if z < z_threshold or p > 0.05:
        return False
    return bool(np.isfinite(q) and q <= 0.10) if require_fdr else True


def build_claim_table(stat_table: pd.DataFrame, tail_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    meta_cols = [
        "trait",
        "analysis_id",
        "trait_role",
        "analysis_label",
        "gene_set",
        "n_query_genes",
        "n_present_genes",
    ]
    for key, group in stat_table.groupby(meta_cols, dropna=False, observed=True):
        base = dict(zip(meta_cols, key, strict=True))
        by_stat = {str(row["statistic_name"]): pd.Series(row) for row in group.to_dict(orient="records")}
        raw = by_stat.get("mean_score")
        winsor = by_stat.get("winsorized_mean_p99")
        trimmed = by_stat.get("trimmed_mean_10pct")
        rank = by_stat.get("rank_mean")
        leave5 = by_stat.get("leave_top_5_mean")
        raw_positive = positive_row(raw) if raw is not None else False
        robust_positive = (
            positive_row(winsor)
            and positive_row(leave5, require_fdr=False, z_threshold=2.0)
            and (
                positive_row(trimmed, require_fdr=False, z_threshold=2.0)
                or positive_row(rank, require_fdr=False, z_threshold=2.0)
            )
        )
        if raw_positive and robust_positive:
            claim_status = "calibrated_contextual_or_module_support"
        elif raw_positive and not robust_positive:
            claim_status = "outlier_driven_supportive_only"
        else:
            claim_status = "negative"
        tail = tail_table
        for column, value in base.items():
            tail = tail.loc[tail[column].astype(str).eq(str(value))]
        tail_row = tail.iloc[0].to_dict() if not tail.empty else {}
        rows.append(
            {
                **base,
                "raw_mean_z": value_from(raw, "z"),
                "raw_mean_empirical_p": value_from(raw, "empirical_p"),
                "raw_mean_fdr_bh": value_from(raw, "empirical_p_fdr_bh"),
                "winsorized_p99_z": value_from(winsor, "z"),
                "winsorized_p99_empirical_p": value_from(winsor, "empirical_p"),
                "trimmed_mean_z": value_from(trimmed, "z"),
                "trimmed_mean_empirical_p": value_from(trimmed, "empirical_p"),
                "rank_mean_z": value_from(rank, "z"),
                "rank_mean_empirical_p": value_from(rank, "empirical_p"),
                "leave_top5_z": value_from(leave5, "z"),
                "leave_top5_empirical_p": value_from(leave5, "empirical_p"),
                "raw_positive": str(raw_positive).lower(),
                "robust_positive": str(robust_positive).lower(),
                "tail_robust_claim_status": claim_status,
                "tail_diagnostic_status": tail_row.get("tail_diagnostic_status", ""),
                "max_gene_score": tail_row.get("max_gene_score", np.nan),
                "max_gene_symbol": tail_row.get("max_gene_symbol", ""),
                "top1_positive_score_contribution": tail_row.get("top1_positive_score_contribution", np.nan),
                "top5_positive_score_contribution": tail_row.get("top5_positive_score_contribution", np.nan),
                "top5_gene_symbols": tail_row.get("top5_gene_symbols", ""),
            }
        )
    return pd.DataFrame(rows)


def value_from(row: pd.Series | None, column: str) -> object:
    if row is None:
        return np.nan
    return row.get(column, np.nan)


def render_report(claims: pd.DataFrame, args: argparse.Namespace) -> str:
    lines = [
        f"# Tail-robust gene-set diagnostics: {args.analysis_label}",
        "",
        f"Created: {now_utc()}",
        "",
        "Claim rule:",
        "- raw positive + robust positive = calibrated_contextual_or_module_support",
        "- raw positive + robust negative = outlier_driven_supportive_only",
        "- raw negative = negative",
        "",
        "## Trait summary",
        "",
        "| Trait | Calibrated | Outlier-driven | Negative | Best raw-positive gene set |",
        "|---|---:|---:|---:|---|",
    ]
    for trait, group in claims.groupby("trait", observed=True):
        calibrated = int(group["tail_robust_claim_status"].eq("calibrated_contextual_or_module_support").sum())
        outlier = int(group["tail_robust_claim_status"].eq("outlier_driven_supportive_only").sum())
        negative = int(group["tail_robust_claim_status"].eq("negative").sum())
        raw_positive = group.loc[group["raw_positive"].astype(str).eq("true")].sort_values("raw_mean_empirical_p")
        best = raw_positive.iloc[0]["gene_set"] if not raw_positive.empty else ""
        lines.append(f"| {trait} | {calibrated} | {outlier} | {negative} | {best} |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_root = args.out_dir / args.analysis_label
    for subdir in ["tables", "reports"]:
        (output_root / subdir).mkdir(parents=True, exist_ok=True)
    gene_sets, gene_set_summary = load_gene_sets(args.gene_set_file)
    unknown = [trait for trait in args.traits if trait not in TRAITS]
    if unknown:
        raise ValueError(f"Unknown traits: {unknown}")
    stat_tables = []
    claim_tables = []
    for trait_key in args.traits:
        stat_table, claim_table = run_trait(trait_key, gene_sets, gene_set_summary, args)
        stat_tables.append(stat_table)
        claim_tables.append(claim_table)
    stats = pd.concat(stat_tables, ignore_index=True)
    claims = pd.concat(claim_tables, ignore_index=True)
    write_table(output_root / "tables" / f"{args.analysis_label}.tail_robust_statistics.tsv", stats)
    write_table(output_root / "tables" / f"{args.analysis_label}.tail_robust_claims.tsv", claims)
    report = render_report(claims, args)
    (output_root / "reports" / f"{args.analysis_label}.tail_robust_report.md").write_text(
        report + "\n",
        encoding="utf-8",
    )
    manifest = {
        "created_utc": now_utc(),
        "script_path": str(Path(__file__).resolve()),
        "analysis_label": args.analysis_label,
        "gene_set_file": str(args.gene_set_file),
        "out_dir": str(output_root),
        "traits": args.traits,
        "n_null": int(args.n_null),
        "degree_bins": int(args.degree_bins),
        "min_overlap": int(args.min_overlap),
        "trim_fraction": float(args.trim_fraction),
        "include_low_information": bool(args.include_low_information),
        "include_special_region": bool(args.include_special_region),
        "statistics": list(STATISTICS),
        "null_sampling": "degree_bin_matched_with_replacement",
        "claim_rule": "raw positive + robust positive = calibrated support; raw positive + robust negative = outlier-driven supportive only; raw negative = negative",
    }
    (output_root / "reports" / f"{args.analysis_label}.tail_robust_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote tail-robust diagnostics to {output_root}", flush=True)


if __name__ == "__main__":
    main()
