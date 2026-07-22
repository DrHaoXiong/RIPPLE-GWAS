#!/usr/bin/env python
"""Run RIPPLE-D V1.8 fixed-module synthetic tests on a real score background.

This RC runner is deliberately fixed-module only. It is the Phase 2 gate for
the experimental V1.8 profile mixture and cannot generate real-trait claims.
Outer score-vector permutation, signal injection, matched-locus null sampling,
and V1.8 fitting are all rerun per replicate.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V171_RUNNER = PROJECT_ROOT / "scripts" / "run_v171_end_to_end_synthetic_power.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ripple.experimental.v18_mixture import adaptive_locus_module_test_v18  # noqa: E402
from ripple.experimental.v18a_raw_tail import adaptive_locus_module_test_v18a  # noqa: E402
from ripple.experimental.v18a_joint import adaptive_locus_module_test_v18a_joint  # noqa: E402


_WORKER_RUNNER = None
_WORKER_STATE = None


@dataclass(frozen=True)
class V18Scenario:
    """Pre-specified V1.8 synthetic score components for one fixed module."""

    name: str
    module_size: int
    n_weak_loci: int
    weak_effect: float
    n_strong_loci: int = 0
    strong_effect: float = 0.0
    scenario_family: str = "distributed_power"


V18_SCENARIOS: dict[str, V18Scenario] = {
    "pure_null": V18Scenario("pure_null", 30, 0, 0.0, scenario_family="negative_control"),
    "single_strong": V18Scenario("single_strong", 30, 0, 0.0, 1, 6.0, "strong_artifact"),
    "five_strong": V18Scenario("five_strong", 30, 0, 0.0, 5, 4.0, "strong_artifact"),
    "eight_strong": V18Scenario("eight_strong", 30, 0, 0.0, 8, 4.0, "strong_artifact"),
    "top5_dominant": V18Scenario("top5_dominant", 30, 5, 0.5, 5, 4.0, "strong_artifact"),
    "weak_8_of_30_e1": V18Scenario("weak_8_of_30_e1", 30, 8, 1.0),
    "weak_8_of_30_e1p5": V18Scenario("weak_8_of_30_e1p5", 30, 8, 1.5),
    "weak_8_of_30_e2": V18Scenario("weak_8_of_30_e2", 30, 8, 2.0),
    "weak_15_of_30_e0p5": V18Scenario("weak_15_of_30_e0p5", 30, 15, 0.5),
    "weak_15_of_30_e0p8": V18Scenario("weak_15_of_30_e0p8", 30, 15, 0.8),
    "weak_15_of_30_e1": V18Scenario("weak_15_of_30_e1", 30, 15, 1.0),
    "weak_15_of_30_e1p5": V18Scenario("weak_15_of_30_e1p5", 30, 15, 1.5),
    "weak_15_of_30_e2": V18Scenario("weak_15_of_30_e2", 30, 15, 2.0),
    "weak_25_of_50_e0p5": V18Scenario("weak_25_of_50_e0p5", 50, 25, 0.5),
    "weak_25_of_50_e0p8": V18Scenario("weak_25_of_50_e0p8", 50, 25, 0.8),
    "mixed_1strong_15weak": V18Scenario("mixed_1strong_15weak", 30, 15, 0.8, 1, 4.0, "mixed"),
    "broad_passenger_15_of_100": V18Scenario("broad_passenger_15_of_100", 100, 15, 0.8, scenario_family="passenger"),
}


def _expand_scenarios(requested: list[str]) -> tuple[V18Scenario, ...]:
    names = requested or list(V18_SCENARIOS)
    if "core" in names:
        names = ["pure_null", "single_strong", "five_strong", "eight_strong", "weak_8_of_30_e1p5", "weak_15_of_30_e1", "weak_15_of_30_e2", "mixed_1strong_15weak", "broad_passenger_15_of_100"]
    unknown = sorted(set(names) - set(V18_SCENARIOS))
    if unknown:
        raise ValueError(f"Unknown V1.8 scenarios: {unknown}")
    return tuple(V18_SCENARIOS[name] for name in names)


def _load_v171_runner():
    spec = importlib.util.spec_from_file_location("v171_runner_for_v18", V171_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {V171_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _seed(global_seed: int, replicate: int, scenario: str, stage: str) -> int:
    import hashlib

    text = f"{global_seed}|{replicate}|{scenario}|{stage}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little", signed=False)


def _choose_target(context, library, spec: V18Scenario) -> str:
    work = context.work.copy()
    work["gene_key"] = work["gene_symbol"].astype(str).str.upper()
    locus_by_gene = dict(zip(work["gene_key"], work["locus_id"].astype(str), strict=False))
    required = spec.n_weak_loci + spec.n_strong_loci
    candidates: list[tuple[int, int, str]] = []
    for name, genes in library.gene_sets.items():
        present = {str(gene).upper() for gene in genes} & set(locus_by_gene)
        n_loci = len({locus_by_gene[gene] for gene in present})
        if len(present) >= max(5, required) and n_loci >= required:
            candidates.append((abs(len(present) - spec.module_size), -n_loci, str(name)))
    if not candidates:
        raise ValueError(f"no registered target can host {spec.name}")
    return min(candidates)[2]


def _inject_v18_signal(runner, state, context, target: str, spec: V18Scenario, seed: int):
    if spec.n_weak_loci + spec.n_strong_loci == 0:
        return context, (), ()
    members = {str(gene).upper() for gene in state.library.gene_sets[target]}
    selected = context.work.loc[context.work["gene_symbol"].astype(str).str.upper().isin(members)].copy()
    selected["gene_key"] = selected["gene_symbol"].astype(str).str.upper()
    representatives = selected.sort_values(["locus_id", "gene_key"], kind="mergesort").drop_duplicates("locus_id")
    order = np.random.default_rng(seed).permutation(len(representatives))
    chosen = representatives.iloc[order[: spec.n_weak_loci + spec.n_strong_loci]].copy()
    effects = np.concatenate([
        np.full(spec.n_strong_loci, spec.strong_effect),
        np.full(spec.n_weak_loci, spec.weak_effect),
    ])
    updates = {str(row.gene_key): float(row.assoc_resid_score) + float(effect) for row, effect in zip(chosen.itertuples(index=False), effects, strict=True)}
    updated = runner.context_with_updated_association_scores(context, updates, state.config.to_adaptive_locus_config())
    return updated, tuple(chosen["gene_key"].astype(str)), tuple(chosen["locus_id"].astype(str))


def _run_one(runner, state, spec: V18Scenario, replicate: int, n_null: int, global_seed: int, procedure: str) -> dict[str, object]:
    scenario_name = str(spec.name)
    outer_rng = np.random.default_rng(_seed(global_seed, replicate, scenario_name, "outer"))
    updates = runner.exact_locus_vector_permutation_updates(state.exact_locus_plan, outer_rng)
    base = runner.context_with_updated_association_scores(
        state.base_context, updates, state.config.to_adaptive_locus_config()
    )
    target = _choose_target(base, state.library, spec)
    context, signal_genes, signal_loci = _inject_v18_signal(
        runner, state, base, target, spec, _seed(global_seed, replicate, scenario_name, "signal")
    )
    test_function = {
        "v18": adaptive_locus_module_test_v18,
        "v18a_raw_tail": adaptive_locus_module_test_v18a,
        "v18a": adaptive_locus_module_test_v18a_joint,
    }[procedure]
    row, _ = test_function(
        context,
        state.module_memberships[target],
        state.config.to_adaptive_locus_config(),
        n_null=n_null,
        rng=np.random.default_rng(_seed(global_seed, replicate, scenario_name, "inner")),
    )
    return {
        "analysis_id": "tier4_v18a_joint_raw_tail_profile_v0_1" if procedure == "v18a" else ("tier4_v18a_raw_tail_conditioned_v0_1" if procedure == "v18a_raw_tail" else "tier4_v18_profile_mixture_v0_1"),
        "procedure": procedure,
        "scenario": scenario_name,
        "scenario_family": str(spec.scenario_family),
        "replicate": replicate,
        "weak_effect": float(spec.weak_effect),
        "strong_effect": float(spec.strong_effect),
        "n_weak_loci": int(spec.n_weak_loci),
        "n_strong_loci": int(spec.n_strong_loci),
        "target_module": target,
        "signal_genes": ",".join(signal_genes),
        "signal_loci": ",".join(signal_loci),
        "n_null": n_null,
        "outer_seed": _seed(global_seed, replicate, scenario_name, "outer"),
        "inner_seed": _seed(global_seed, replicate, scenario_name, "inner"),
        **row,
    }


def _initialize_worker(values: dict[str, object]) -> None:
    """Load the immutable registered score context once per process."""

    global _WORKER_RUNNER, _WORKER_STATE
    _WORKER_RUNNER = _load_v171_runner()
    _WORKER_STATE = _WORKER_RUNNER._load_worker_state(values)


def _run_task(task: tuple[object, int, int, int, str]) -> dict[str, object]:
    spec, replicate, n_null, seed, procedure = task
    if _WORKER_RUNNER is None or _WORKER_STATE is None:
        raise RuntimeError("V1.8 worker state was not initialized")
    return _run_one(_WORKER_RUNNER, _WORKER_STATE, spec, replicate, n_null, seed, procedure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gene-set-file", type=Path, required=True)
    parser.add_argument("--library-manifests", type=Path, required=True)
    parser.add_argument("--library-id", required=True)
    parser.add_argument("--score-template", type=Path, required=True)
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument("--n-replicates", type=int, default=5)
    parser.add_argument("--n-null", type=int, default=100)
    parser.add_argument("--procedure", choices=("v18", "v18a", "v18a_raw_tail"), default="v18")
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_replicates < 1 or args.n_null < 20:
        raise ValueError("n_replicates must be positive and n_null must be at least 20")
    state_values: dict[str, object] = {
        "config_path": args.config,
        "analysis_mode": "registered-library",
        "gene_set_file": args.gene_set_file,
        "library_manifests": args.library_manifests,
        "library_id": args.library_id,
        "score_template": args.score_template,
    }
    runner = _load_v171_runner()
    scenarios = _expand_scenarios(args.scenario)
    tasks = [
        (spec, replicate, args.n_null, args.seed, args.procedure)
        for spec in scenarios
        for replicate in range(args.n_replicates)
    ]
    started = time.perf_counter()
    if args.n_workers == 1:
        state = runner._load_worker_state(state_values)
        rows = [_run_one(runner, state, *task) for task in tasks]
    else:
        with ProcessPoolExecutor(
            max_workers=args.n_workers, initializer=_initialize_worker, initargs=(state_values,)
        ) as executor:
            rows = list(executor.map(_run_task, tasks))
    runtime_seconds = time.perf_counter() - started
    results = pd.DataFrame(rows)
    p_column = {"v18": "v18_profile_lrt_weak_given_strong_empirical_p", "v18a_raw_tail": "v18a_profile_lrt_weak_given_raw_tail_empirical_p", "v18a": "v18a_joint_profile_lrt_weak_given_strong_empirical_p"}[args.procedure]
    expected_column = {"v18": "v18_expected_weak_loci", "v18a_raw_tail": "v18a_expected_weak_loci", "v18a": "v18a_joint_expected_weak_loci"}[args.procedure]
    convergence_column = {"v18": "v18_fit_converged", "v18a_raw_tail": "v18a_fit_converged", "v18a": "v18a_joint_fit_converged"}[args.procedure]
    results["nominal_weak_pass"] = results[p_column].lt(0.05)
    summary = results.groupby("scenario", observed=True).agg(
        n_replicates=("replicate", "size"),
        nominal_weak_pass_rate=("nominal_weak_pass", "mean"),
        median_weak_p=(p_column, "median"),
        median_expected_weak_loci=(expected_column, "median"),
        fit_convergence_rate=(convergence_column, "mean"),
    ).reset_index()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out_dir / "v18_fixed_module_power.tsv", sep="\t", index=False)
    summary.to_csv(args.out_dir / "v18_fixed_module_power_summary.tsv", sep="\t", index=False)
    (args.out_dir / "v18_run_manifest.json").write_text(json.dumps({
        "analysis_id": "tier4_v18a_joint_raw_tail_profile_v0_1" if args.procedure == "v18a" else ("tier4_v18a_raw_tail_conditioned_v0_1" if args.procedure == "v18a_raw_tail" else "tier4_v18_profile_mixture_v0_1"), "experimental_only": True,
        "procedure": args.procedure,
        "n_null": args.n_null, "n_replicates": args.n_replicates, "seed": args.seed,
        "n_workers": args.n_workers, "runtime_seconds": runtime_seconds,
        "config": str(args.config), "score_template": str(args.score_template),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
