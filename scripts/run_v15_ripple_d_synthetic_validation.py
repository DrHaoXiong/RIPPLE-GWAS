#!/usr/bin/env python
"""Synthetic validation for RIPPLE-D V1.5 distributed weak-signal module gates.

This script is intentionally separate from real-trait analysis. It constructs
controlled external-locus synthetic score tables and verifies that the V1.5
RIPPLE-D module layer rejects top-locus artifacts while retaining sensitivity
to distributed multi-locus weak/moderate signal.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ripple.modules import RippleDConfig, ripple_d_module_tests  # noqa: E402
from ripple.modules.anchored import AnchoredModuleLibrary  # noqa: E402

PRIVATE_ROOT = Path("/mnt/d/RIPPLE/RIPPLE_private")
DEFAULT_OUT_DIR = (
    PRIVATE_ROOT
    / "30_analysis"
    / "tier4_v15_locus_distributed_module_repair_v0_1"
    / "synthetic_generalization_validation"
)


@dataclass(frozen=True)
class SyntheticScenario:
    name: str
    expected_behavior: str
    module_loci: int
    effect_kind: str


SCENARIOS = [
    SyntheticScenario("pure_null", "not_distributed", 20, "none"),
    SyntheticScenario("single_top_locus_artifact", "not_distributed", 20, "single_top_locus"),
    SyntheticScenario("sparse_top_locus_mixture", "not_distributed_or_mixed_only", 20, "three_top_loci"),
    SyntheticScenario("distributed_8_locus_moderate", "distributed_positive", 20, "eight_moderate"),
    SyntheticScenario("distributed_15_locus_weak", "distributed_positive", 24, "fifteen_weak"),
    SyntheticScenario("mixed_top_locus_plus_distributed", "distributed_or_mixed_positive", 24, "top_plus_distributed"),
    SyntheticScenario("distributed_signal_with_nonmodule_extreme", "distributed_positive", 20, "module_plus_background_top"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-null", type=int, default=500)
    parser.add_argument("--n-outer-null", type=int, default=100)
    parser.add_argument("--outer-null-n-null", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_out_dir(path: Path, *, force: bool) -> None:
    if path.exists() and any(path.iterdir()) and not force:
        raise FileExistsError(f"{path} exists and is not empty. Use --force to overwrite.")
    path.mkdir(parents=True, exist_ok=True)
    (path / "tables").mkdir(parents=True, exist_ok=True)
    (path / "reports").mkdir(parents=True, exist_ok=True)


def make_base_scores(
    rng: np.random.Generator,
    *,
    n_loci: int = 480,
    genes_per_locus: int = 2,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for locus_idx in range(n_loci):
        chrom = (locus_idx % 22) + 1
        rank_on_chrom = locus_idx // 22
        locus_start = rank_on_chrom * 1_500_000 + 100_000
        for gene_idx in range(genes_per_locus):
            gene_start = locus_start + gene_idx * 45_000
            rows.append(
                {
                    "gene_symbol": f"L{locus_idx:04d}_G{gene_idx}",
                    "assoc_resid_score": float(rng.normal(0.0, 0.55)),
                    "chrom": chrom,
                    "gene_start": gene_start,
                    "gene_end": gene_start + 25_000,
                    "graph_degree": int(rng.integers(1, 80)),
                    "gene_length": 25_000,
                    "n_mapped_snps": int(rng.integers(5, 90)),
                    "local_ld_score": float(rng.uniform(1.0, 10.0)),
                    "synthetic_ld_block_id": f"SYNLD_{locus_idx:04d}",
                }
            )
    return pd.DataFrame(rows)


def module_genes_from_loci(locus_indices: range | list[int], *, gene_idx: int = 0) -> set[str]:
    return {f"L{int(locus_idx):04d}_G{gene_idx}" for locus_idx in locus_indices}


def set_gene_score(scores: pd.DataFrame, gene: str, value: float) -> None:
    scores.loc[scores["gene_symbol"].eq(gene), "assoc_resid_score"] = float(value)


def scenario_scores(
    scenario: SyntheticScenario,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, AnchoredModuleLibrary]:
    scores = make_base_scores(rng)
    oracle_loci = list(range(scenario.module_loci))
    oracle = module_genes_from_loci(oracle_loci)
    decoy = module_genes_from_loci(range(220, 250))
    broad_decoy = module_genes_from_loci(range(300, 360))

    if scenario.effect_kind == "single_top_locus":
        set_gene_score(scores, "L0000_G0", 15.0)
    elif scenario.effect_kind == "three_top_loci":
        for locus_idx, value in [(0, 12.0), (1, 9.0), (2, 7.0)]:
            set_gene_score(scores, f"L{locus_idx:04d}_G0", value)
    elif scenario.effect_kind == "eight_moderate":
        for locus_idx in range(8):
            set_gene_score(scores, f"L{locus_idx:04d}_G0", 2.35)
    elif scenario.effect_kind == "fifteen_weak":
        for locus_idx in range(15):
            set_gene_score(scores, f"L{locus_idx:04d}_G0", 1.85)
    elif scenario.effect_kind == "top_plus_distributed":
        set_gene_score(scores, "L0000_G0", 11.0)
        for locus_idx in range(1, 13):
            set_gene_score(scores, f"L{locus_idx:04d}_G0", 1.80)
    elif scenario.effect_kind == "module_plus_background_top":
        for locus_idx in range(10):
            set_gene_score(scores, f"L{locus_idx:04d}_G0", 2.10)
        set_gene_score(scores, "L0400_G0", 20.0)

    library = AnchoredModuleLibrary(
        gene_sets={
            "oracle_module": oracle,
            "null_decoy_module": decoy,
            "broad_null_decoy_module": broad_decoy,
        },
        module_source={
            "oracle_module": "synthetic_oracle",
            "null_decoy_module": "synthetic_decoy",
            "broad_null_decoy_module": "synthetic_decoy",
        },
        annotation_source_type={
            "oracle_module": "internal_support",
            "null_decoy_module": "internal_support",
            "broad_null_decoy_module": "internal_support",
        },
        module_category={
            "oracle_module": scenario.name,
            "null_decoy_module": "negative_control",
            "broad_null_decoy_module": "negative_control",
        },
    )
    return scores, library


def behavior_passed(status: str, expected: str) -> bool:
    if expected == "not_distributed":
        return status != "distributed_weak_signal_module_candidate"
    if expected == "not_distributed_or_mixed_only":
        return status not in {
            "distributed_weak_signal_module_candidate",
            "moderate_locus_supported_module",
            "module_specific_rank_supported_module",
        }
    if expected == "distributed_positive":
        return status == "distributed_weak_signal_module_candidate"
    if expected == "distributed_or_mixed_positive":
        return status in {
            "distributed_weak_signal_module_candidate",
            "mixed_sparse_distributed_candidate",
        }
    raise ValueError(f"Unknown expected behavior: {expected}")


def run_scenario(
    scenario: SyntheticScenario,
    *,
    seed: int,
    n_null: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rng = np.random.default_rng(seed)
    scores, library = scenario_scores(scenario, rng)
    config = RippleDConfig(
        score_cap=3.0,
        locus_id_column="synthetic_ld_block_id",
        locus_definition_name="synthetic_external_ld_blocks",
        degree_bins=8,
        property_bins=4,
    )
    modules, locus_audit, _, summary = ripple_d_module_tests(
        scores,
        library,
        config=config,
        min_present=5,
        n_null=n_null,
        seed=seed,
        return_null_details=False,
    )
    modules = modules.copy()
    modules["scenario"] = scenario.name
    modules["expected_behavior"] = scenario.expected_behavior
    modules["scenario_seed"] = seed
    modules["scenario_n_null"] = int(n_null)
    oracle_status = str(modules.loc[modules["module_name"].eq("oracle_module"), "module_status"].iloc[0])
    summary = {
        **summary,
        "scenario": scenario.name,
        "expected_behavior": scenario.expected_behavior,
        "oracle_status": oracle_status,
        "behavior_passed": behavior_passed(oracle_status, scenario.expected_behavior),
        "seed": seed,
        "n_null": int(n_null),
    }
    locus_audit = locus_audit.copy()
    locus_audit["scenario"] = scenario.name
    return modules, summary


def run_outer_nulls(
    *,
    n_outer: int,
    n_null: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scenario = SyntheticScenario("outer_null", "not_distributed", 20, "none")
    for idx in range(n_outer):
        scenario_seed = seed + idx * 1009
        modules, summary = run_scenario(scenario, seed=scenario_seed, n_null=n_null)
        oracle = modules.loc[modules["module_name"].eq("oracle_module")].iloc[0]
        rows.append(
            {
                "outer_idx": idx,
                "seed": scenario_seed,
                "module_status": oracle["module_status"],
                "ripple_d_empirical_p": oracle["ripple_d_empirical_p"],
                "locus_robust_empirical_p": oracle["locus_robust_empirical_p"],
                "module_specific_rank_empirical_p": oracle["module_specific_rank_empirical_p"],
                "leave_top1_locus_empirical_p": oracle["leave_top1_locus_empirical_p"],
                "false_distributed_positive": bool(
                    oracle["module_status"] == "distributed_weak_signal_module_candidate"
                ),
                "summary": json.dumps(summary, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def render_report(scenario_summary: pd.DataFrame, outer_nulls: pd.DataFrame, out_dir: Path) -> str:
    false_positive_count = int(outer_nulls["false_distributed_positive"].sum()) if not outer_nulls.empty else 0
    n_outer = int(len(outer_nulls))
    fpr = false_positive_count / n_outer if n_outer else float("nan")
    lines = [
        "# RIPPLE-D V1.5 synthetic validation",
        "",
        "This validation uses synthetic external-locus score tables. It tests whether RIPPLE-D rejects "
        "single-top-locus artifacts while detecting distributed multi-locus weak/moderate signal.",
        "",
        "## Scenario outcomes",
        "",
        scenario_summary[
            [
                "scenario",
                "expected_behavior",
                "oracle_status",
                "behavior_passed",
                "n_effective_loci",
                "top1_locus_contribution",
                "top5_locus_contribution",
                "ripple_d_empirical_p",
                "module_specific_rank_empirical_p",
                "leave_top1_locus_empirical_p",
            ]
        ].to_string(index=False),
        "",
        "## Outer null",
        "",
        f"- Outer null replicates: {n_outer}",
        f"- False distributed positives: {false_positive_count}",
        f"- Empirical false-positive fraction: {fpr:.4f}" if n_outer else "- Empirical false-positive fraction: NA",
        "",
        "## Output files",
        "",
        f"- Scenario summary: `{out_dir / 'tables' / 'v15_synthetic_validation_summary.tsv'}`",
        f"- Module-level details: `{out_dir / 'tables' / 'v15_synthetic_module_tests.tsv'}`",
        f"- Outer null details: `{out_dir / 'tables' / 'v15_synthetic_outer_null.tsv'}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    ensure_out_dir(args.out_dir, force=args.force)
    tables_dir = args.out_dir / "tables"
    reports_dir = args.out_dir / "reports"

    module_tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for idx, scenario in enumerate(SCENARIOS):
        scenario_seed = int(args.seed + idx * 10_007)
        modules, summary = run_scenario(scenario, seed=scenario_seed, n_null=args.n_null)
        oracle = modules.loc[modules["module_name"].eq("oracle_module")].iloc[0].to_dict()
        summary_rows.append({**summary, **{key: oracle.get(key) for key in oracle}})
        module_tables.append(modules)

    all_modules = pd.concat(module_tables, ignore_index=True)
    scenario_summary = pd.DataFrame(summary_rows)
    outer_nulls = run_outer_nulls(
        n_outer=args.n_outer_null,
        n_null=args.outer_null_n_null,
        seed=args.seed + 500_000,
    )

    all_modules.to_csv(tables_dir / "v15_synthetic_module_tests.tsv", sep="\t", index=False)
    scenario_summary.to_csv(tables_dir / "v15_synthetic_validation_summary.tsv", sep="\t", index=False)
    outer_nulls.to_csv(tables_dir / "v15_synthetic_outer_null.tsv", sep="\t", index=False)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "script_path": str(Path(__file__).resolve()),
        "out_dir": str(args.out_dir),
        "n_null": int(args.n_null),
        "n_outer_null": int(args.n_outer_null),
        "outer_null_n_null": int(args.outer_null_n_null),
        "seed": int(args.seed),
    }
    (reports_dir / "v15_synthetic_validation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report = render_report(scenario_summary, outer_nulls, args.out_dir)
    (reports_dir / "v15_synthetic_validation_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
