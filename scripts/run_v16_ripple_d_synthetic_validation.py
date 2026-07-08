#!/usr/bin/env python
"""Synthetic RC validation for RIPPLE-D V1.6 claim-readiness gates."""

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
    / "tier4_v16_claim_readiness_hardening_v0_1"
    / "synthetic_rc_validation"
)


@dataclass(frozen=True)
class Scenario:
    name: str
    effect_kind: str
    expected_min_tier: str
    use_external_locus: bool = True
    module_loci: int = 24
    module_size: int = 24


SCENARIOS = [
    Scenario("pure_null", "none", "negative"),
    Scenario("single_top_locus_artifact", "single_top", "not_manuscript"),
    Scenario("five_strong_capped_loci", "five_strong", "not_manuscript"),
    Scenario("eight_strong_capped_loci", "eight_strong", "not_manuscript"),
    Scenario("top5_dominant_artifact", "top5_dominant", "not_manuscript"),
    Scenario("distributed_8_locus_moderate", "eight_moderate", "high_confidence"),
    Scenario("distributed_15_locus_weak", "fifteen_weak", "high_confidence"),
    Scenario("many_weak_loci_below_score_1", "twenty_subthreshold", "exploratory_or_high_confidence"),
    Scenario("broad_go_like_null_module", "broad_null", "negative", module_loci=80, module_size=80),
    Scenario("retinal_sparse_annotated_module", "sparse_retinal", "exploratory_or_high_confidence", module_loci=10, module_size=10),
    Scenario("pseudo_locus_split_merge_artifact", "eight_moderate", "diagnostic_only", use_external_locus=False),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-null", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_out_dir(path: Path, *, force: bool) -> None:
    if path.exists() and any(path.iterdir()) and not force:
        raise FileExistsError(f"{path} exists and is not empty. Use --force to overwrite.")
    path.mkdir(parents=True, exist_ok=True)
    (path / "tables").mkdir(parents=True, exist_ok=True)
    (path / "reports").mkdir(parents=True, exist_ok=True)


def make_scores(rng: np.random.Generator, *, n_loci: int = 520, genes_per_locus: int = 2) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for locus_idx in range(n_loci):
        chrom = (locus_idx % 22) + 1
        rank_on_chrom = locus_idx // 22
        locus_start = rank_on_chrom * 1_700_000 + 100_000
        for gene_idx in range(genes_per_locus):
            start = locus_start + gene_idx * 35_000
            rows.append(
                {
                    "gene_symbol": f"L{locus_idx:04d}_G{gene_idx}",
                    "assoc_resid_score": float(rng.normal(0.0, 0.55)),
                    "chrom": chrom,
                    "gene_start": start,
                    "gene_end": start + 20_000,
                    "graph_degree": int(rng.integers(1, 30)),
                    "gene_length": 20_000,
                    "n_mapped_snps": int(rng.integers(5, 60)),
                    "local_ld_score": float(rng.uniform(1.0, 5.0)),
                    "synthetic_ld_block_id": f"SYNLD_{locus_idx:04d}",
                }
            )
    return pd.DataFrame(rows)


def module_genes(n_loci: int) -> set[str]:
    return {f"L{idx:04d}_G0" for idx in range(n_loci)}


def set_score(scores: pd.DataFrame, locus_idx: int, value: float) -> None:
    scores.loc[scores["gene_symbol"].eq(f"L{locus_idx:04d}_G0"), "assoc_resid_score"] = float(value)


def apply_effect(scores: pd.DataFrame, scenario: Scenario) -> None:
    if scenario.effect_kind == "none":
        return
    if scenario.effect_kind == "single_top":
        set_score(scores, 0, 18.0)
    elif scenario.effect_kind == "five_strong":
        for idx in range(5):
            set_score(scores, idx, 7.0)
    elif scenario.effect_kind == "eight_strong":
        for idx in range(8):
            set_score(scores, idx, 6.0)
    elif scenario.effect_kind == "top5_dominant":
        for idx, value in enumerate([8.0, 7.5, 7.0, 6.5, 6.0, 0.4, 0.3, 0.2]):
            set_score(scores, idx, value)
    elif scenario.effect_kind == "eight_moderate":
        for idx in range(8):
            set_score(scores, idx, 2.35)
    elif scenario.effect_kind == "fifteen_weak":
        for idx in range(15):
            set_score(scores, idx, 1.75)
    elif scenario.effect_kind == "twenty_subthreshold":
        for idx in range(20):
            set_score(scores, idx, 0.85)
    elif scenario.effect_kind == "sparse_retinal":
        for idx in [0, 2, 4, 6, 8]:
            set_score(scores, idx, 1.85)
    elif scenario.effect_kind == "broad_null":
        return
    else:
        raise ValueError(f"Unknown effect kind: {scenario.effect_kind}")


def build_library(scenario: Scenario) -> AnchoredModuleLibrary:
    oracle = module_genes(scenario.module_size)
    decoy = {f"L{idx:04d}_G0" for idx in range(250, 250 + scenario.module_size)}
    return AnchoredModuleLibrary(
        gene_sets={"oracle_module": oracle, "decoy_module": decoy},
        module_source={"oracle_module": "synthetic_v16", "decoy_module": "synthetic_v16"},
        annotation_source_type={"oracle_module": "internal_support", "decoy_module": "internal_support"},
        module_category={"oracle_module": scenario.name, "decoy_module": "negative_control"},
    )


def tier_passed(status: str, expected: str) -> bool:
    if expected == "negative":
        return status in {"negative", "exploratory_locus_distributed_candidate"}
    if expected == "not_manuscript":
        return status != "manuscript_ready_distributed_candidate"
    if expected == "high_confidence":
        return status in {
            "high_confidence_diagnostic_candidate",
            "manuscript_ready_distributed_candidate",
        }
    if expected == "exploratory_or_high_confidence":
        return status in {
            "exploratory_locus_distributed_candidate",
            "high_confidence_diagnostic_candidate",
            "manuscript_ready_distributed_candidate",
        }
    if expected == "diagnostic_only":
        return status != "manuscript_ready_distributed_candidate"
    raise ValueError(expected)


def run_scenario(scenario: Scenario, *, n_null: int, seed: int) -> tuple[pd.DataFrame, dict[str, object]]:
    rng = np.random.default_rng(seed)
    scores = make_scores(rng)
    apply_effect(scores, scenario)
    config = RippleDConfig(
        locus_id_column="synthetic_ld_block_id" if scenario.use_external_locus else None,
        locus_definition_name="synthetic_external_ld_blocks" if scenario.use_external_locus else None,
        locus_window_bp=500_000,
        degree_bins=1,
        property_bins=1,
        annotation_matching_enabled=False,
    )
    modules, locus_audit, _, summary = ripple_d_module_tests(
        scores,
        build_library(scenario),
        config=config,
        min_present=5,
        n_null=n_null,
        seed=seed,
        return_null_details=False,
    )
    modules = modules.copy()
    modules["scenario"] = scenario.name
    modules["expected_min_tier"] = scenario.expected_min_tier
    modules["scenario_seed"] = int(seed)
    modules["scenario_n_null"] = int(n_null)
    modules["use_external_locus"] = bool(scenario.use_external_locus)
    oracle = modules.loc[modules["module_name"].eq("oracle_module")].iloc[0]
    summary = {
        **summary,
        "scenario": scenario.name,
        "expected_min_tier": scenario.expected_min_tier,
        "oracle_v16_claim_status": str(oracle["v16_claim_status"]),
        "behavior_passed": tier_passed(str(oracle["v16_claim_status"]), scenario.expected_min_tier),
        "seed": int(seed),
        "n_null": int(n_null),
        "n_locus_audit_rows": int(len(locus_audit)),
    }
    return modules, summary


def render_report(summary: pd.DataFrame, out_dir: Path) -> str:
    failed = summary.loc[~summary["behavior_passed"]]
    lines = [
        "# RIPPLE-D V1.6 synthetic RC validation",
        "",
        "This smoke-scale validation checks whether V1.6 claim-readiness gates downgrade artifact scenarios while retaining diagnostic support for distributed signal scenarios.",
        "",
        "## Scenario Summary",
        "",
        summary[
            [
                "scenario",
                "expected_min_tier",
                "oracle_v16_claim_status",
                "behavior_passed",
                "n_null",
            ]
        ].to_string(index=False),
        "",
        f"Failed expectations: {len(failed)}",
        "",
        f"Detailed table: `{out_dir / 'tables' / 'v16_synthetic_validation.tsv'}`",
        f"Module details: `{out_dir / 'tables' / 'v16_synthetic_module_tests.tsv'}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    ensure_out_dir(args.out_dir, force=args.force)
    modules_all: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for idx, scenario in enumerate(SCENARIOS):
        modules, summary = run_scenario(scenario, n_null=args.n_null, seed=args.seed + idx * 1009)
        modules_all.append(modules)
        oracle = modules.loc[modules["module_name"].eq("oracle_module")].iloc[0].to_dict()
        summaries.append({**summary, **oracle})
    module_table = pd.concat(modules_all, ignore_index=True)
    summary_table = pd.DataFrame(summaries)
    tables_dir = args.out_dir / "tables"
    reports_dir = args.out_dir / "reports"
    module_table.to_csv(tables_dir / "v16_synthetic_module_tests.tsv", sep="\t", index=False)
    summary_table.to_csv(tables_dir / "v16_synthetic_validation.tsv", sep="\t", index=False)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "script_path": str(Path(__file__).resolve()),
        "out_dir": str(args.out_dir),
        "n_null": int(args.n_null),
        "seed": int(args.seed),
    }
    (reports_dir / "v16_synthetic_validation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report = render_report(summary_table, args.out_dir)
    (reports_dir / "v16_synthetic_validation_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
