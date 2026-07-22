#!/usr/bin/env python
"""End-to-end synthetic power for the V1.7.2 adaptive-tail 5x2 procedure.

This runner uses the production adaptive module statistic, independent
screen/confirmation routing, complete-family BH correction, and D1 nomination
gates. Its defaults are deliberately smoke-scale; release-scale runs must set
the replicate and Monte Carlo counts explicitly.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Callable, Iterable, Mapping, NamedTuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ripple.modules.adaptive import (  # noqa: E402
    AdaptiveLocusContext,
    adaptive_library_fingerprint,
    adaptive_locus_module_test,
    context_with_updated_association_scores,
    prepare_adaptive_locus_context,
)
from ripple.modules.anchored import (  # noqa: E402
    AnchoredModuleLibrary,
    load_anchored_gene_set_library,
)
from ripple.modules.distributed import external_locus_audit_table  # noqa: E402
from ripple.modules.v17_config import RippleDV17Config  # noqa: E402
from ripple.modules.v17_contracts import (  # noqa: E402
    LibraryManifest,
    load_library_manifests,
    sha256_file,
)
from ripple.modules.v17_multiplicity import independent_screen_confirm_bh  # noqa: E402
from ripple.modules.v17_nomination import (  # noqa: E402
    NOMINATION_STATES,
    derive_leave_topk_pass,
    nominate_v17,
)
from ripple.modules.v17_tail import (  # noqa: E402
    replace_with_tail_refinement,
    tail_refinement_targets,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT / "ripple" / "config" / "v171_candidate_primary_5x2_annotation_off.json"
)
CONTROL_SCENARIOS = ("pure_null", "single_top_artifact")
CORE_SCENARIOS = (*CONTROL_SCENARIOS, "weak_15_of_30", "passenger_rich_8_of_80", "effect_grid")
WEAK_15_OF_30_EFFECTS = (0.5, 0.8, 1.0)
PASSENGER_RICH_8_OF_80_EFFECTS = (0.75, 1.0, 1.25)
DEFAULT_EFFECT_GRID = (0.5, 1.0, 1.5, 2.0, 3.0)
GRID_LOCI = (5, 8, 15)
PROTOCOL = "registered_library__exact_locus_vector_null__independent_screen_confirm_bh__complete_d1_gates_v3"
TAIL_PROTOCOL = "registered_library__exact_locus_vector_null__independent_adaptive_tail_confirm_bh__complete_d1_gates_v4"


def procedure_protocol(confirm_tail_null: int) -> str:
    return TAIL_PROTOCOL if confirm_tail_null > 0 else PROTOCOL


def finite_int_or_default(value: object, default: int = 0) -> int:
    """Convert finite numeric metadata while preserving an explicit NA sentinel."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return int(default)
    return int(numeric) if math.isfinite(numeric) else int(default)


def inference_source_tree_hash() -> str:
    """Hash the runner and reusable module implementation used by checkpoints."""

    paths = [
        Path(__file__).resolve(),
        *sorted((PROJECT_ROOT / "ripple" / "modules").glob("*.py")),
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class ExactLocusVectorPlan(NamedTuple):
    locus_ids_by_gene_count: tuple[tuple[str, ...], ...]
    recipient_genes: dict[str, tuple[str, ...]]
    score_vectors: dict[str, np.ndarray]


@dataclass(frozen=True)
class SyntheticWorkerState:
    """Read-only process-local inputs shared across replicate-grouped tasks."""

    config: RippleDV17Config
    analysis_mode: str
    library: AnchoredModuleLibrary | None = None
    manifest: LibraryManifest | None = None
    base_context: AdaptiveLocusContext | None = None
    exact_locus_plan: ExactLocusVectorPlan | None = None
    module_memberships: Mapping[str, frozenset[str]] | None = None
    external_locus_pass: bool | None = None
    family_fingerprint: str | None = None


_WORKER_STATE: SyntheticWorkerState | None = None


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    module_size: int
    n_signal_loci: int
    effect: float
    scenario_family: str


def _effect_label(effect: float) -> str:
    return format(float(effect), "g").replace("-", "m").replace(".", "p")


def expand_scenarios(
    requested: Iterable[str], effect_grid: Iterable[float]
) -> tuple[ScenarioSpec, ...]:
    """Expand controls and every predeclared synthetic power surface."""

    names = list(requested)
    if not names or "core" in names:
        names = [*CORE_SCENARIOS, *[name for name in names if name != "core"]]
    if "all" in names:
        names = list(CORE_SCENARIOS)
    fixed = {
        "pure_null": ScenarioSpec("pure_null", 30, 0, 0.0, "negative_control"),
        "single_top_artifact": ScenarioSpec("single_top_artifact", 30, 1, 6.0, "negative_control"),
    }
    generated: dict[str, ScenarioSpec] = {}
    for effect in WEAK_15_OF_30_EFFECTS:
        spec = ScenarioSpec(
            f"weak_15_of_30_effect_{_effect_label(effect)}",
            30,
            15,
            effect,
            "weak_15_of_30",
        )
        generated[spec.name] = spec
    for effect in PASSENGER_RICH_8_OF_80_EFFECTS:
        spec = ScenarioSpec(
            f"passenger_rich_8_of_80_effect_{_effect_label(effect)}",
            80,
            8,
            effect,
            "passenger_rich_8_of_80",
        )
        generated[spec.name] = spec
    for n_signal in GRID_LOCI:
        for effect in effect_grid:
            spec = ScenarioSpec(
                f"grid_{n_signal}_of_30_effect_{_effect_label(effect)}",
                30,
                n_signal,
                float(effect),
                "effect_grid",
            )
            generated[spec.name] = spec
    expanded: list[ScenarioSpec] = []
    seen: set[str] = set()

    def add(spec: ScenarioSpec) -> None:
        if spec.name in seen:
            return
        expanded.append(spec)
        seen.add(spec.name)

    for name in names:
        if name == "weak_15_of_30":
            for effect in WEAK_15_OF_30_EFFECTS:
                add(
                    ScenarioSpec(
                        f"weak_15_of_30_effect_{_effect_label(effect)}",
                        30,
                        15,
                        effect,
                        "weak_15_of_30",
                    )
                )
            continue
        if name == "passenger_rich_8_of_80":
            for effect in PASSENGER_RICH_8_OF_80_EFFECTS:
                add(
                    ScenarioSpec(
                        f"passenger_rich_8_of_80_effect_{_effect_label(effect)}",
                        80,
                        8,
                        effect,
                        "passenger_rich_8_of_80",
                    )
                )
            continue
        if name == "effect_grid":
            for n_signal in GRID_LOCI:
                for effect in effect_grid:
                    grid_name = f"grid_{n_signal}_of_30_effect_{_effect_label(effect)}"
                    add(ScenarioSpec(grid_name, 30, n_signal, float(effect), "effect_grid"))
            continue
        if name in generated:
            add(generated[name])
            continue
        if name not in fixed:
            raise ValueError(f"Unknown scenario: {name}")
        add(fixed[name])
    if not expanded:
        raise ValueError("At least one scenario is required")
    return tuple(expanded)


def _seed(global_seed: int, replicate: int, phase: str, module_name: str = "all") -> int:
    payload = f"{global_seed}:{replicate}:{phase}:{module_name}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def synthetic_scores(n_background_loci: int, paired_seed: int) -> pd.DataFrame:
    """Create one-gene external loci with paired technical covariates and scores."""

    if n_background_loci < 100:
        raise ValueError("n_background_loci must be at least 100")
    rng = np.random.default_rng(paired_seed)
    index = np.arange(n_background_loci)
    return pd.DataFrame(
        {
            "gene_symbol": [f"G{idx:05d}" for idx in index],
            "assoc_resid_score": rng.normal(0.0, 1.0, n_background_loci),
            "chrom": (index % 20 + 1).astype(str),
            "gene_start": index * 100_000 + 10_000,
            "gene_end": index * 100_000 + 20_000,
            "graph_degree": rng.lognormal(2.0, 0.5, n_background_loci),
            "gene_length": rng.lognormal(10.0, 0.3, n_background_loci),
            "n_mapped_snps": rng.integers(5, 80, n_background_loci),
            "local_ld_score": rng.lognormal(1.5, 0.3, n_background_loci),
            "eur_ld_block_id": [f"SYNLD_{idx:05d}" for idx in index],
        }
    )


def select_registered_manifest(manifest_path: Path, library_id: str) -> LibraryManifest:
    """Load one complete broad-discovery registration by stable ID."""

    matches = [
        item for item in load_library_manifests(manifest_path) if item.library_id == library_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one registered library_id={library_id!r}")
    manifest = matches[0]
    if manifest.role != "broad_discovery":
        raise ValueError("registered synthetic power requires a broad_discovery library")
    if not manifest.scope.scope_complete:
        raise ValueError("registered synthetic power requires a complete correction scope")
    return manifest


def load_registered_inputs(
    *,
    gene_set_file: Path,
    manifest_path: Path,
    library_id: str,
    score_template: Path,
    config: RippleDV17Config,
) -> tuple[AnchoredModuleLibrary, LibraryManifest, pd.DataFrame]:
    """Validate the actual registered library and external-locus score template."""

    library = load_anchored_gene_set_library(gene_set_file)
    manifest = select_registered_manifest(manifest_path, library_id)
    fingerprint = adaptive_library_fingerprint(library)
    if fingerprint != manifest.scope.registered_library_fingerprint:
        raise ValueError("gene-set library fingerprint differs from registered manifest")
    scores = pd.read_csv(score_template, sep="\t", compression="infer")
    required = {
        "gene_symbol",
        "assoc_resid_score",
        config.external_locus.locus_id_column,
    }
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"score template is missing required columns: {missing}")
    if scores[config.external_locus.locus_id_column].isna().any():
        raise ValueError("score template contains missing external locus IDs")
    return library, manifest, scores


def _validate_primary_config(config: RippleDV17Config) -> None:
    if (
        config.matching.annotation_matching_enabled
        or config.matching.degree_bins != 5
        or config.matching.property_bins != 2
    ):
        raise ValueError("synthetic power requires the annotation-off 5x2 configuration")


def _load_worker_state(values: Mapping[str, object]) -> SyntheticWorkerState:
    """Load immutable inputs once for a serial process or pool worker."""

    config = RippleDV17Config.load_json(Path(values["config_path"]))
    _validate_primary_config(config)
    analysis_mode = str(values["analysis_mode"])
    if analysis_mode == "random-decoy":
        return SyntheticWorkerState(config=config, analysis_mode=analysis_mode)
    if analysis_mode != "registered-library":
        raise ValueError(f"unknown analysis_mode: {analysis_mode}")
    required = ("gene_set_file", "library_manifests", "library_id", "score_template")
    if any(values.get(name) is None for name in required):
        raise ValueError("registered-library mode requires library and score inputs")
    library, manifest, base_scores = load_registered_inputs(
        gene_set_file=Path(values["gene_set_file"]),
        manifest_path=Path(values["library_manifests"]),
        library_id=str(values["library_id"]),
        score_template=Path(values["score_template"]),
        config=config,
    )
    if not library.gene_sets:
        raise ValueError("registered-library mode requires a non-empty registered scope")
    base_context = prepare_adaptive_locus_context(
        base_scores, library, config.to_adaptive_locus_config()
    )
    audit = external_locus_audit_table(
        base_context.work,
        locus_id_column=config.external_locus.locus_id_column,
        locus_source=config.external_locus.locus_source,
        locus_source_version=config.external_locus.locus_source_version,
        genome_build=config.external_locus.genome_build,
        ancestry=config.external_locus.ancestry,
        construction_script=str(Path(__file__).resolve()),
    )
    return SyntheticWorkerState(
        config=config,
        analysis_mode=analysis_mode,
        library=library,
        manifest=manifest,
        base_context=base_context,
        exact_locus_plan=prepare_exact_locus_vector_plan(base_context),
        module_memberships={
            str(name): frozenset(str(gene).upper() for gene in genes)
            for name, genes in library.gene_sets.items()
        },
        external_locus_pass=bool(audit.iloc[0]["external_locus_audit_pass"]),
        family_fingerprint=manifest.scope.registered_library_fingerprint,
    )


def _initialize_worker(values: Mapping[str, object]) -> None:
    """ProcessPool initializer; expensive registered inputs are loaded once."""

    global _WORKER_STATE
    _WORKER_STATE = _load_worker_state(values)


def prepare_exact_locus_vector_plan(
    context: AdaptiveLocusContext,
) -> ExactLocusVectorPlan:
    """Freeze complete score vectors within exact eligible-gene-count strata."""

    groups = {
        str(locus_id): group.sort_values(
            ["gene_start", "gene_end", "gene_symbol"], kind="mergesort"
        )
        for locus_id, group in context.work.groupby("locus_id", observed=True)
    }
    strata: dict[int, list[str]] = {}
    recipient_genes: dict[str, tuple[str, ...]] = {}
    score_vectors: dict[str, np.ndarray] = {}
    for locus_id, group in groups.items():
        strata.setdefault(len(group), []).append(locus_id)
        recipient_genes[locus_id] = tuple(group["gene_symbol"].astype(str).str.upper().tolist())
        score_vectors[locus_id] = pd.to_numeric(
            group["assoc_resid_score"], errors="raise"
        ).to_numpy(dtype=float)
    return ExactLocusVectorPlan(
        locus_ids_by_gene_count=tuple(
            tuple(sorted(locus_ids)) for _, locus_ids in sorted(strata.items())
        ),
        recipient_genes=recipient_genes,
        score_vectors=score_vectors,
    )


def exact_locus_vector_permutation_updates(
    plan: ExactLocusVectorPlan, rng: np.random.Generator
) -> dict[str, float]:
    """Move intact score vectors only between loci with equal gene counts."""

    updates: dict[str, float] = {}
    for locus_ids in plan.locus_ids_by_gene_count:
        donors = list(rng.permutation(locus_ids)) if len(locus_ids) > 1 else list(locus_ids)
        for recipient_id, donor_id in zip(locus_ids, donors, strict=True):
            genes = plan.recipient_genes[recipient_id]
            scores = plan.score_vectors[str(donor_id)]
            if len(genes) != len(scores):
                raise RuntimeError("exact-gene-count locus permutation changed vector length")
            updates.update(dict(zip(genes, scores, strict=True)))
    return updates


def choose_registered_target(
    context: AdaptiveLocusContext,
    library: AnchoredModuleLibrary,
    spec: ScenarioSpec,
) -> str:
    """Choose a deterministic predeclared module that can host the scenario."""

    work = context.work.copy()
    work["gene_key"] = work["gene_symbol"].astype(str).str.upper()
    eligible = set(work["gene_key"])
    locus_by_gene = dict(zip(work["gene_key"], work["locus_id"].astype(str), strict=False))
    candidates: list[tuple[int, int, str]] = []
    for name, genes in library.gene_sets.items():
        present = sorted({str(gene).upper() for gene in genes} & eligible)
        n_loci = len({locus_by_gene[gene] for gene in present})
        required_present = max(5, spec.n_signal_loci)
        if len(present) >= required_present and n_loci >= spec.n_signal_loci:
            candidates.append((abs(len(present) - spec.module_size), -n_loci, str(name)))
    if not candidates:
        raise ValueError(f"no registered module can host {spec.n_signal_loci} distinct signal loci")
    return min(candidates)[2]


def inject_distinct_locus_signal(
    context: AdaptiveLocusContext,
    library: AnchoredModuleLibrary,
    module_name: str,
    spec: ScenarioSpec,
    *,
    signal_order_seed: int,
    config: RippleDV17Config,
) -> tuple[AdaptiveLocusContext, tuple[str, ...], tuple[str, ...]]:
    """Inject one gene per distinct target locus after outer-null permutation."""

    if spec.n_signal_loci == 0:
        return context, (), ()
    target_genes = {str(gene).upper() for gene in library.gene_sets[module_name]}
    selected = context.work.loc[
        context.work["gene_symbol"].astype(str).str.upper().isin(target_genes)
    ].copy()
    selected["gene_key"] = selected["gene_symbol"].astype(str).str.upper()
    representatives = (
        selected.sort_values(["locus_id", "gene_key"], kind="mergesort")
        .drop_duplicates("locus_id", keep="first")
        .loc[:, ["locus_id", "gene_key", "assoc_resid_score"]]
    )
    if len(representatives) < spec.n_signal_loci:
        raise ValueError("target module has too few distinct loci for injection")
    order = np.random.default_rng(signal_order_seed).permutation(len(representatives))
    chosen = representatives.iloc[order[: spec.n_signal_loci]]
    updates = {
        str(row.gene_key): float(row.assoc_resid_score) + spec.effect
        for row in chosen.itertuples(index=False)
    }
    updated = context_with_updated_association_scores(
        context, updates, config.to_adaptive_locus_config()
    )
    return (
        updated,
        tuple(chosen["gene_key"].astype(str)),
        tuple(chosen["locus_id"].astype(str)),
    )


def build_library(
    scores: pd.DataFrame, *, target_size: int, family_size: int, library_seed: int
) -> tuple[AnchoredModuleLibrary, str]:
    """Build one target and a complete synthetic family of non-target modules."""

    if family_size < 2:
        raise ValueError("family_size must be at least 2")
    genes = scores["gene_symbol"].astype(str).tolist()
    if target_size >= len(genes):
        raise ValueError("target module must be smaller than the background")
    target_name = "TARGET"
    gene_sets: dict[str, set[str]] = {target_name: set(genes[:target_size])}
    decoy_pool = np.asarray(genes[target_size:], dtype=object)
    if len(decoy_pool) < target_size:
        raise ValueError("background is too small for target-sized decoys")
    rng = np.random.default_rng(library_seed + target_size)
    for index in range(family_size - 1):
        gene_sets[f"DECOY_{index + 1:04d}"] = set(
            rng.choice(decoy_pool, size=target_size, replace=False).tolist()
        )
    names = gene_sets.keys()
    library = AnchoredModuleLibrary(
        gene_sets=gene_sets,
        module_source={name: "v171_synthetic_full_library" for name in names},
        annotation_source_type={name: "synthetic" for name in names},
        module_category={name: "synthetic_power" for name in names},
    )
    return library, target_name


def inject_scenario(
    scores: pd.DataFrame, spec: ScenarioSpec, *, signal_order_seed: int
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Add the scenario effect to a paired ordering of target loci."""

    out = scores.copy()
    target_genes = out.loc[: spec.module_size - 1, "gene_symbol"].astype(str).to_numpy()
    order = np.random.default_rng(signal_order_seed).permutation(target_genes)
    signal_genes = tuple(str(gene) for gene in order[: spec.n_signal_loci])
    if signal_genes:
        selected = out["gene_symbol"].isin(signal_genes)
        out.loc[selected, "assoc_resid_score"] += spec.effect
    return out, signal_genes


def _run_phase(
    *,
    context: object,
    library: AnchoredModuleLibrary,
    config: RippleDV17Config,
    modules: Iterable[str],
    replicate: int,
    phase: str,
    n_null: int,
    global_seed: int,
    module_memberships: Mapping[str, frozenset[str]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    adaptive_config = config.to_adaptive_locus_config()
    for module_name in modules:
        module_seed = _seed(global_seed, replicate, phase, module_name)
        row, _ = adaptive_locus_module_test(
            context,
            (
                module_memberships[module_name]
                if module_memberships is not None
                else {str(gene).upper() for gene in library.gene_sets[module_name]}
            ),
            adaptive_config,
            n_null=n_null,
            rng=np.random.default_rng(module_seed),
        )
        rows.append({"module_name": module_name, f"{phase}_seed": module_seed, **row})
    return pd.DataFrame(rows)


def finalize_two_stage(
    screen: pd.DataFrame,
    confirm: pd.DataFrame,
    *,
    n_family: int,
    screen_null: int,
    confirm_null: int,
    q_max: float,
    null_policy: object,
    external_locus_pass: bool,
    leave_top1_supportive_p_max: float,
    discovery_scope_complete: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply the same complete-family BH and D1 routing as the production runner."""

    if "confirm_p" in confirm:
        confirm = confirm.copy()
    elif "v17_adaptive_omnibus_empirical_p" in confirm:
        confirm = confirm.rename(columns={"v17_adaptive_omnibus_empirical_p": "confirm_p"})
    elif confirm.empty:
        confirm = pd.DataFrame(columns=["module_name", "confirm_p"])
    table = screen.merge(confirm, on="module_name", how="left", suffixes=("", "_confirm"))
    table, multiplicity = independent_screen_confirm_bh(
        table,
        screen_p_column="screen_p",
        confirm_p_column="confirm_p",
        declared_family_size=n_family,
        q_max=q_max,
    )
    table["v17_complete_family_bh_q"] = table["v17_two_stage_bh_q"]
    table["v17_adaptive_omnibus_bh_q"] = table["v17_two_stage_bh_q"]
    table["tested"] = table.get("test_status", pd.Series("tested", index=table.index)).eq("tested")
    confirmation_n = pd.to_numeric(table.get("confirm_n_null"), errors="coerce")
    if not isinstance(confirmation_n, pd.Series):
        confirmation_n = pd.Series(confirm_null, index=table.index, dtype=float)
    table["n_null"] = np.where(table["screen_selected"], confirmation_n, screen_null)
    table["empirical_resolution_pass"] = table["screen_selected"] & (
        1.0 / (confirmation_n + 1.0) <= q_max / n_family
    )

    def numeric(name: str) -> pd.Series:
        confirm_name = f"{name}_confirm"
        source = confirm_name if confirm_name in table else name
        return pd.to_numeric(table.get(source), errors="coerce")

    table["null_quality_pass"] = (
        numeric("null_exact_match_rate").ge(null_policy.exact_match_rate_min)
        & numeric("null_global_fallback_rate").le(null_policy.global_fallback_rate_max)
        & numeric("null_reuse_fallback_rate").le(null_policy.reuse_fallback_rate_max)
        & numeric("within_locus_replacement_rate").le(null_policy.within_locus_replacement_rate_max)
    ).fillna(False)
    table["leave_topk_pass"] = derive_leave_topk_pass(
        table,
        supportive_p_max=leave_top1_supportive_p_max,
    )
    for name in ("n_loci", "n_effective_loci", "top1_locus_contribution", "top_tail_pass"):
        confirm_name = f"{name}_confirm"
        if confirm_name in table:
            table[name] = table[confirm_name].where(table["screen_selected"], table[name])
    table["external_locus_audit_pass"] = external_locus_pass
    table["library_role"] = "broad_discovery"
    table["selection_stage"] = "none"
    table["selected_from_same_trait"] = False
    table["q_value_valid_for_discovery"] = discovery_scope_complete
    table["multi_strong_locus_pathway_overlap"] = ~table["top_tail_pass"].fillna(False).astype(bool)
    table["hypothesis_prioritized_pattern"] = (
        table["screen_selected"] & ~table["v17_two_stage_pass"]
    )
    return nominate_v17(table, q_max=q_max), multiplicity


def module_audit_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Retain target and positive calls only, with one row per module."""

    bh = table["v17_two_stage_pass"].fillna(False).astype(bool)
    d1 = table["v17_nomination_state"].eq("D1_full_library_fdr_candidate")
    target = table["is_target"].fillna(False).astype(bool)
    audit = table.loc[target | bh | d1].copy()
    audit["module_audit_reason"] = [
        ";".join(
            reason
            for reason, passed in (
                ("target", bool(target.loc[index])),
                ("bh_positive", bool(bh.loc[index])),
                ("d1_positive", bool(d1.loc[index])),
            )
            if passed
        )
        for index in audit.index
    ]
    return audit.drop_duplicates("module_name", keep="first")


def affected_module_names(library: AnchoredModuleLibrary, signal_genes: Iterable[str]) -> set[str]:
    """Return every registered module whose tested genes include injected signal."""

    injected = {str(gene).upper() for gene in signal_genes}
    if not injected:
        return set()
    return {
        str(name)
        for name, genes in library.gene_sets.items()
        if injected & {str(gene).upper() for gene in genes}
    }


def _wilson_ci(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z**2 / (4.0 * total**2))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def run_replicate(
    spec: ScenarioSpec,
    replicate: int,
    *,
    config_path: Path = DEFAULT_CONFIG,
    global_seed: int,
    family_size: int,
    n_background_loci: int,
    screen_null: int,
    confirm_null: int,
    screen_p_max: float,
    confirm_tail_null: int = 0,
    confirm_tail_trigger_exceedances: int = 10,
    analysis_mode: str = "random-decoy",
    gene_set_file: Path | None = None,
    library_manifests: Path | None = None,
    library_id: str | None = None,
    score_template: Path | None = None,
    _worker_state: SyntheticWorkerState | None = None,
    _paired_context: AdaptiveLocusContext | None = None,
    _shared_base_scores: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Run one scenario replicate through screen, confirmation, BH, and D1."""

    if _worker_state is None:
        _worker_state = _load_worker_state(
            {
                "config_path": config_path,
                "analysis_mode": analysis_mode,
                "gene_set_file": gene_set_file,
                "library_manifests": library_manifests,
                "library_id": library_id,
                "score_template": score_template,
            }
        )
    config = _worker_state.config
    paired_seed = _seed(global_seed, replicate, "paired_background")
    registered_mode = analysis_mode == "registered-library"
    if registered_mode:
        library = _worker_state.library
        manifest = _worker_state.manifest
        base_context = _worker_state.base_context
        plan = _worker_state.exact_locus_plan
        if library is None or manifest is None or base_context is None or plan is None:
            raise RuntimeError("registered worker state is incomplete")
        target_name = choose_registered_target(base_context, library, spec)
        if _paired_context is None:
            updates = exact_locus_vector_permutation_updates(
                plan, np.random.default_rng(paired_seed)
            )
            _paired_context = context_with_updated_association_scores(
                base_context, updates, config.to_adaptive_locus_config()
            )
        context, signal_genes, signal_loci = inject_distinct_locus_signal(
            _paired_context,
            library,
            target_name,
            spec,
            signal_order_seed=_seed(global_seed, replicate, "paired_signal_order"),
            config=config,
        )
        discovery_scope_complete = True
        family_fingerprint = str(_worker_state.family_fingerprint)
    elif analysis_mode == "random-decoy":
        base_scores = (
            _shared_base_scores
            if _shared_base_scores is not None
            else synthetic_scores(n_background_loci, paired_seed)
        )
        scores, signal_genes = inject_scenario(
            base_scores,
            spec,
            signal_order_seed=_seed(global_seed, replicate, "paired_signal_order"),
        )
        signal_loci = tuple(signal_genes)
        library, target_name = build_library(
            scores,
            target_size=spec.module_size,
            family_size=family_size,
            library_seed=global_seed,
        )
        context = prepare_adaptive_locus_context(scores, library, config.to_adaptive_locus_config())
        discovery_scope_complete = False
        family_fingerprint = adaptive_library_fingerprint(library)
    else:
        raise ValueError(f"unknown analysis_mode: {analysis_mode}")
    if registered_mode:
        external_locus_pass = bool(_worker_state.external_locus_pass)
    else:
        audit = external_locus_audit_table(
            context.work,
            locus_id_column=config.external_locus.locus_id_column,
            locus_source="synthetic",
            locus_source_version="v171_e2e_power_v1",
            genome_build="synthetic",
            ancestry="synthetic",
            construction_script=str(Path(__file__).resolve()),
        )
        external_locus_pass = bool(audit.iloc[0]["external_locus_audit_pass"])
    modules = list(library.gene_sets)
    screen = _run_phase(
        context=context,
        library=library,
        config=config,
        modules=modules,
        replicate=replicate,
        phase="screen",
        n_null=screen_null,
        global_seed=global_seed,
        module_memberships=(_worker_state.module_memberships if registered_mode else None),
    ).rename(columns={"v17_adaptive_omnibus_empirical_p": "screen_p"})
    # Family membership is preregistered, not inferred from post-hoc testability.
    screen["multiplicity_eligible"] = True
    screen["screen_p"] = pd.to_numeric(screen["screen_p"], errors="coerce")
    screen.loc[~screen["screen_p"].between(0.0, 1.0), "screen_p"] = 1.0
    screen["screen_selected"] = screen["test_status"].eq("tested") & screen["screen_p"].le(
        screen_p_max
    )
    selected = screen.loc[screen["screen_selected"], "module_name"].astype(str).tolist()
    confirm = _run_phase(
        context=context,
        library=library,
        config=config,
        modules=selected,
        replicate=replicate,
        phase="confirm",
        n_null=confirm_null,
        global_seed=global_seed,
        module_memberships=(_worker_state.module_memberships if registered_mode else None),
    )
    if "v17_adaptive_omnibus_empirical_p" in confirm:
        confirm = confirm.rename(columns={"v17_adaptive_omnibus_empirical_p": "confirm_p"})
    elif confirm.empty:
        confirm = pd.DataFrame(columns=["module_name", "confirm_p"])
    tail_targets: list[str] = []
    tail = pd.DataFrame(columns=["module_name", "confirm_p"])
    if confirm_tail_null > 0:
        tail_targets = tail_refinement_targets(
            confirm,
            n_null=confirm_null,
            max_exceedances=confirm_tail_trigger_exceedances,
        )
        if tail_targets:
            tail = _run_phase(
                context=context,
                library=library,
                config=config,
                modules=tail_targets,
                replicate=replicate,
                phase="confirm_tail",
                n_null=confirm_tail_null,
                global_seed=global_seed,
                module_memberships=(_worker_state.module_memberships if registered_mode else None),
            ).rename(columns={"v17_adaptive_omnibus_empirical_p": "confirm_p"})
    confirm = replace_with_tail_refinement(
        confirm,
        tail,
        initial_n_null=confirm_null,
        refined_n_null=confirm_tail_null,
    )
    table, multiplicity = finalize_two_stage(
        screen,
        confirm,
        n_family=len(modules),
        screen_null=screen_null,
        confirm_null=confirm_null,
        q_max=config.q_policy.q_max,
        null_policy=config.null_policy,
        external_locus_pass=external_locus_pass,
        leave_top1_supportive_p_max=config.leave_top1_supportive_p_max,
        discovery_scope_complete=discovery_scope_complete,
    )
    if not registered_mode:
        table.loc[
            table["v17_nomination_state"].eq("D1_full_library_fdr_candidate"),
            "v17_nomination_state",
        ] = "D0_negative_or_not_tested"
    table["scenario"] = spec.name
    table["replicate"] = replicate
    table["is_target"] = table["module_name"].eq(target_name)
    affected_modules = affected_module_names(library, signal_genes)
    table["is_affected_by_injection"] = table["module_name"].isin(affected_modules)
    table["paired_seed"] = paired_seed
    protocol = procedure_protocol(confirm_tail_null)
    table["protocol"] = protocol
    bh = table["v17_two_stage_pass"].astype(bool)
    d1 = table["v17_nomination_state"].eq("D1_full_library_fdr_candidate")
    target = table.loc[table["is_target"]].iloc[0]
    non_target = ~table["is_target"]
    true_null = ~table["is_affected_by_injection"].astype(bool)
    n_bh = int(bh.sum())
    n_d1 = int(d1.sum())
    n_non_target_bh = int((bh & non_target).sum())
    n_non_target_d1 = int((d1 & non_target).sum())
    n_false_bh = int((bh & true_null).sum())
    n_false_d1 = int((d1 & true_null).sum())
    nomination_counts = table["v17_nomination_state"].value_counts()
    target_gate_columns = (
        "v17_n_loci_pass",
        "v17_n_effective_loci_pass",
        "v17_top1_locus_contribution_pass",
        "v17_top_tail_pass",
        "v17_external_locus_audit_pass",
        "v17_null_quality_pass",
        "v17_leave_topk_pass",
        "v17_empirical_resolution_pass",
        "v17_weak_eligibility_pass",
    )
    target_gates = {f"target_{name}": bool(target.get(name, False)) for name in target_gate_columns}
    target_gate_failures = (
        ";".join(
            name.removeprefix("v17_").removesuffix("_pass")
            for name in target_gate_columns
            if not bool(target.get(name, False))
        )
        or "none"
    )
    result = {
        **asdict(spec),
        "replicate": replicate,
        "paired_seed": paired_seed,
        "signal_genes": ",".join(signal_genes),
        "signal_loci": ",".join(signal_loci),
        "target_module": target_name,
        "analysis_mode": analysis_mode,
        "family_size": len(modules),
        "library_fingerprint": family_fingerprint,
        "config_hash": config.canonical_hash,
        "protocol": protocol,
        "screen_null": screen_null,
        "confirm_null": confirm_null,
        "confirm_tail_null": confirm_tail_null,
        "confirm_tail_trigger_exceedances": confirm_tail_trigger_exceedances,
        "n_confirm_tail_refined": len(tail_targets),
        "screen_p_max": screen_p_max,
        "n_screen_selected": int(table["screen_selected"].sum()),
        "target_screen_inclusion": bool(target["screen_selected"]),
        "target_confirm_tail_refined": bool(target.get("confirm_tail_refined", False)),
        "target_confirm_n_null": finite_int_or_default(
            target.get("confirm_n_null"), default=0
        ),
        "target_bh_recovery": bool(target["v17_two_stage_pass"]),
        "target_d1_recovery": bool(
            target["v17_nomination_state"] == "D1_full_library_fdr_candidate"
        ),
        "target_screen_to_bh_attrition": bool(
            target["screen_selected"] and not target["v17_two_stage_pass"]
        ),
        "target_bh_to_d1_attrition": bool(
            target["v17_two_stage_pass"]
            and target["v17_nomination_state"] != "D1_full_library_fdr_candidate"
        ),
        "n_bh_calls": n_bh,
        "n_d1_calls": n_d1,
        "family_any_bh": bool(n_bh),
        "family_any_d1": bool(n_d1),
        "n_false_bh_calls": n_false_bh,
        "n_false_d1_calls": n_false_d1,
        "n_non_target_bh_calls": n_non_target_bh,
        "n_non_target_d1_calls": n_non_target_d1,
        "off_target_bh_call": bool(n_non_target_bh),
        "off_target_d1_call": bool(n_non_target_d1),
        "true_null_bh_fwer": bool(n_false_bh),
        "true_null_d1_fwer": bool(n_false_d1),
        "bh_fdp": n_false_bh / n_bh if n_bh else 0.0,
        "d1_fdp": n_false_d1 / n_d1 if n_d1 else 0.0,
        "target_nomination_state": str(target["v17_nomination_state"]),
        "target_nomination_downgrade_reason": str(
            target.get("v17_nomination_downgrade_reason", "none")
        ),
        "target_weak_gate_failures": target_gate_failures,
        **target_gates,
        **{
            f"n_nomination_{state}": int(nomination_counts.get(state, 0))
            for state in NOMINATION_STATES
        },
        "external_locus_audit_pass": external_locus_pass,
        "n_final_bh_tests": multiplicity["n_final_bh_tests"],
    }
    audit_rows = module_audit_rows(table)
    return {"replicate": result, "modules": audit_rows.to_dict(orient="records")}


def run_replicate_group(
    specs: Iterable[ScenarioSpec],
    replicate: int,
    *,
    _worker_state: SyntheticWorkerState | None = None,
    _payload_callback: Callable[[dict[str, object]], None] | None = None,
    **values: object,
) -> list[dict[str, object]]:
    """Run all requested scenarios on one shared replicate background."""

    specs = tuple(specs)
    if not specs:
        return []
    state = _worker_state or _load_worker_state(values)
    if str(values["analysis_mode"]) != state.analysis_mode:
        raise ValueError("worker state analysis mode differs from task analysis mode")
    paired_seed = _seed(int(values["global_seed"]), replicate, "paired_background")
    paired_context: AdaptiveLocusContext | None = None
    shared_base_scores: pd.DataFrame | None = None
    if state.analysis_mode == "registered-library":
        if state.base_context is None or state.exact_locus_plan is None:
            raise RuntimeError("registered worker state is incomplete")
        updates = exact_locus_vector_permutation_updates(
            state.exact_locus_plan, np.random.default_rng(paired_seed)
        )
        paired_context = context_with_updated_association_scores(
            state.base_context,
            updates,
            state.config.to_adaptive_locus_config(),
        )
    else:
        shared_base_scores = synthetic_scores(int(values["n_background_loci"]), paired_seed)
    payloads: list[dict[str, object]] = []
    for spec in specs:
        payload = run_replicate(
            spec,
            replicate,
            _worker_state=state,
            _paired_context=paired_context,
            _shared_base_scores=shared_base_scores,
            **values,
        )
        if _payload_callback is not None:
            _payload_callback(payload)
        payloads.append(payload)
    return payloads


def summarize(replicates: pd.DataFrame) -> pd.DataFrame:
    """Summarize target power, gate attrition, and non-target error rates."""

    rows: list[dict[str, object]] = []
    for scenario, group in replicates.groupby("name", sort=False, observed=True):
        screen = group["target_screen_inclusion"].astype(bool)
        bh = group["target_bh_recovery"].astype(bool)
        d1 = group["target_d1_recovery"].astype(bool)
        rate_series = {
            "target_screen_inclusion": screen,
            "bh_recovery": bh,
            "d1_power": d1,
            "family_any_bh": group["family_any_bh"].astype(bool),
            "family_any_d1": group["family_any_d1"].astype(bool),
            "off_target_bh_call": group["off_target_bh_call"].astype(bool),
            "off_target_d1_call": group["off_target_d1_call"].astype(bool),
            "true_null_bh_fwer": group["true_null_bh_fwer"].astype(bool),
            "true_null_d1_fwer": group["true_null_d1_fwer"].astype(bool),
        }
        row: dict[str, object] = {
            "scenario": scenario,
            "n_replicates": len(group),
            "screen_to_bh_gate_attrition": float((screen & ~bh).mean()),
            "bh_to_d1_gate_attrition": float((bh & ~d1).mean()),
            "total_target_gate_attrition": float((screen & ~d1).mean()),
            "mean_bh_fdp": float(pd.to_numeric(group["bh_fdp"]).mean()),
            "mean_d1_fdp": float(pd.to_numeric(group["d1_fdp"]).mean()),
            "protocol": str(group["protocol"].iloc[0]),
        }
        for name, values in rate_series.items():
            successes = int(values.sum())
            low, high = _wilson_ci(successes, len(values))
            row[name] = float(values.mean())
            row[f"{name}_count"] = successes
            row[f"{name}_wilson_95ci_low"] = low
            row[f"{name}_wilson_95ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def gate_attrition_table(replicates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario, group in replicates.groupby("name", sort=False, observed=True):
        stage_values = (
            ("screen", group["target_screen_inclusion"].astype(bool)),
            ("BH", group["target_bh_recovery"].astype(bool)),
            ("D1", group["target_d1_recovery"].astype(bool)),
        )
        previous = len(group)
        for stage, values in stage_values:
            retained = int(values.sum())
            rows.append(
                {
                    "scenario": scenario,
                    "stage": stage,
                    "n_entering": previous,
                    "n_retained": retained,
                    "n_attrited": previous - retained,
                    "retention_rate_all_replicates": retained / len(group),
                }
            )
            previous = retained
    return pd.DataFrame(rows)


def _run_fingerprint(args: argparse.Namespace, scenarios: tuple[ScenarioSpec, ...]) -> str:
    mode = getattr(args, "analysis_mode", "random-decoy")
    confirm_tail_null = int(getattr(args, "confirm_tail_null", 0))
    confirm_tail_trigger = int(getattr(args, "confirm_tail_trigger_exceedances", 10))
    payload = {
        "config_hash": RippleDV17Config.load_json(args.config).canonical_hash,
        "scenarios": [asdict(spec) for spec in scenarios],
        "seed": args.seed,
        "family_size": args.family_size,
        "n_background_loci": args.n_background_loci,
        "screen_null": args.screen_null,
        "confirm_null": args.confirm_null,
        "confirm_tail_null": confirm_tail_null,
        "confirm_tail_trigger_exceedances": confirm_tail_trigger,
        "screen_p_max": args.screen_p_max,
        "protocol": procedure_protocol(confirm_tail_null),
        "analysis_mode": mode,
        "inference_source_tree_hash": inference_source_tree_hash(),
    }
    if mode == "registered-library":
        library, manifest, _ = load_registered_inputs(
            gene_set_file=args.gene_set_file,
            manifest_path=args.library_manifests,
            library_id=args.library_id,
            score_template=args.score_template,
            config=RippleDV17Config.load_json(args.config),
        )
        payload.update(
            {
                "library_id": manifest.library_id,
                "library_fingerprint": adaptive_library_fingerprint(library),
                "correction_scope_id": manifest.scope.correction_scope_id,
                "score_template_sha256": sha256_file(args.score_template),
            }
        )
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_manifest(
    args: argparse.Namespace,
    scenarios: tuple[ScenarioSpec, ...],
    run_fingerprint: str,
) -> dict[str, object]:
    """Describe inference scope without equating random decoys to the registered library."""

    mode = getattr(args, "analysis_mode", "random-decoy")
    confirm_tail_null = int(getattr(args, "confirm_tail_null", 0))
    family_size = int(args.family_size)
    registered = mode == "registered-library"
    library_fingerprint: str | None = None
    correction_scope_id: str | None = None
    if registered:
        library, registration, _ = load_registered_inputs(
            gene_set_file=args.gene_set_file,
            manifest_path=args.library_manifests,
            library_id=args.library_id,
            score_template=args.score_template,
            config=RippleDV17Config.load_json(args.config),
        )
        family_size = len(library.gene_sets)
        library_fingerprint = adaptive_library_fingerprint(library)
        correction_scope_id = registration.scope.correction_scope_id
    return {
        "run_fingerprint": run_fingerprint,
        "inference_source_tree_hash": inference_source_tree_hash(),
        "config": str(args.config.resolve()),
        "config_hash": RippleDV17Config.load_json(args.config).canonical_hash,
        "annotation_matching_enabled": False,
        "matching_resolution": "5x2",
        "family_size": family_size,
        "registered_full_family_size": family_size if registered else None,
        "registered_full_family_achieved": registered,
        "registered_family_size_equivalent_achieved": False,
        "family_structure": (
            "actual_registered_gene_set_library"
            if registered
            else "synthetic_size_matched_random_decoys"
        ),
        "family_size_shortfall": 0 if registered else None,
        "analysis_scope": (
            "registered_complete_scope" if registered else "random_decoy_diagnostic"
        ),
        "diagnostic_only": not registered,
        "library_id": getattr(args, "library_id", None),
        "library_fingerprint": library_fingerprint,
        "correction_scope_id": correction_scope_id,
        "score_template_sha256": (sha256_file(args.score_template) if registered else None),
        "scenario_names": [spec.name for spec in scenarios],
        "paired_seed_scheme": "sha256_replicate_background_and_signal_order_v1",
        "screen_confirm_seed_scheme": "sha256_replicate_phase_module_v1",
        "checkpoint_mode": "atomic_per_scenario_replicate",
        "module_output_scope": ("target_plus_deduplicated_bh_or_d1_positive_call_audit_rows"),
        "full_family_rows_persisted": False,
        "full_family_aggregate_location": "replicate_table",
        "protocol": procedure_protocol(confirm_tail_null),
        "smoke_scale_defaults": True,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }


def _checkpoint_path(out_dir: Path, scenario: str, replicate: int) -> Path:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", scenario)
    return out_dir / "checkpoints" / f"{slug}.replicate_{replicate:04d}.json"


def _write_checkpoint(path: Path, payload: Mapping[str, object], fingerprint: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"run_fingerprint": fingerprint, **payload}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_checkpoint(path: Path, fingerprint: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.pop("run_fingerprint", None) != fingerprint:
        raise ValueError(f"Checkpoint inference fingerprint mismatch: {path}")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--analysis-mode",
        choices=("random-decoy", "registered-library"),
        default="random-decoy",
    )
    parser.add_argument("--gene-set-file", type=Path)
    parser.add_argument("--library-manifests", type=Path)
    parser.add_argument("--library-id")
    parser.add_argument("--score-template", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="Repeat for named scenarios; use core, effect_grid, or all.",
    )
    parser.add_argument("--effect-grid", type=float, nargs="+", default=list(DEFAULT_EFFECT_GRID))
    parser.add_argument("--n-replicates", type=int, default=2)
    parser.add_argument("--family-size", type=int, default=8)
    parser.add_argument("--n-background-loci", type=int, default=600)
    parser.add_argument("--screen-null", type=int, default=49)
    parser.add_argument("--confirm-null", type=int, default=99)
    parser.add_argument("--confirm-tail-null", type=int, default=0)
    parser.add_argument("--confirm-tail-trigger-exceedances", type=int, default=10)
    parser.add_argument("--screen-p-max", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    if (
        min(
            args.n_replicates, args.family_size, args.screen_null, args.confirm_null, args.n_workers
        )
        < 1
    ):
        raise ValueError("replicate, family, null, and worker counts must be positive")
    if args.family_size < 2:
        raise ValueError("--family-size must be at least 2")
    if args.n_background_loci < 100:
        raise ValueError("--n-background-loci must be at least 100")
    if not 0.0 < args.screen_p_max <= 1.0:
        raise ValueError("--screen-p-max must be in (0, 1]")
    confirm_tail_null = int(getattr(args, "confirm_tail_null", 0))
    confirm_tail_trigger = int(getattr(args, "confirm_tail_trigger_exceedances", 10))
    if confirm_tail_null < 0 or confirm_tail_trigger < 0:
        raise ValueError("tail refinement nulls and trigger must be nonnegative")
    if confirm_tail_null > 0 and confirm_tail_null <= args.confirm_null:
        raise ValueError("--confirm-tail-null must exceed --confirm-null")
    config = RippleDV17Config.load_json(args.config)
    if (
        config.matching.annotation_matching_enabled
        or config.matching.degree_bins != 5
        or config.matching.property_bins != 2
    ):
        raise ValueError("--config must specify annotation-off 5x2 matching")
    if args.analysis_mode == "registered-library":
        required = {
            "--gene-set-file": args.gene_set_file,
            "--library-manifests": args.library_manifests,
            "--library-id": args.library_id,
            "--score-template": args.score_template,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"registered-library mode requires: {', '.join(missing)}")
        for name in ("gene_set_file", "library_manifests", "score_template"):
            path = Path(getattr(args, name))
            if not path.is_file():
                raise FileNotFoundError(path)
        library, _, _ = load_registered_inputs(
            gene_set_file=args.gene_set_file,
            manifest_path=args.library_manifests,
            library_id=args.library_id,
            score_template=args.score_template,
            config=config,
        )
        args.family_size = len(library.gene_sets)
        minimum_confirm_null = int(np.ceil(len(library.gene_sets) / config.q_policy.q_max)) - 1
        if args.confirm_null < minimum_confirm_null:
            raise ValueError(
                "registered-library confirmation nulls cannot resolve complete-family "
                f"BH: require at least {minimum_confirm_null}"
            )


def _task_group(
    payload: tuple[
        tuple[ScenarioSpec, ...],
        int,
        dict[str, object],
        Path,
        str,
    ],
) -> list[dict[str, object]]:
    specs, replicate, values, out_dir, fingerprint = payload
    if _WORKER_STATE is None:
        raise RuntimeError("synthetic worker was not initialized")

    def checkpoint(result: dict[str, object]) -> None:
        row = result["replicate"]
        path = _checkpoint_path(out_dir, str(row["name"]), int(row["replicate"]))
        _write_checkpoint(path, result, fingerprint)

    return run_replicate_group(
        specs,
        replicate,
        _worker_state=_WORKER_STATE,
        _payload_callback=checkpoint,
        **values,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_args(args)
    scenarios = expand_scenarios(args.scenario or ["core"], args.effect_grid)
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not (args.resume or args.force):
        raise FileExistsError(f"{args.out_dir} exists and is not empty; use --resume or --force")
    tables_dir = args.out_dir / "tables"
    reports_dir = args.out_dir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _run_fingerprint(args, scenarios)
    rows: list[dict[str, object]] = []
    module_rows: list[dict[str, object]] = []
    pending_by_replicate: dict[int, list[ScenarioSpec]] = {}
    values = {
        "config_path": args.config,
        "global_seed": args.seed,
        "family_size": args.family_size,
        "n_background_loci": args.n_background_loci,
        "screen_null": args.screen_null,
        "confirm_null": args.confirm_null,
        "confirm_tail_null": args.confirm_tail_null,
        "confirm_tail_trigger_exceedances": args.confirm_tail_trigger_exceedances,
        "screen_p_max": args.screen_p_max,
        "analysis_mode": args.analysis_mode,
        "gene_set_file": args.gene_set_file,
        "library_manifests": args.library_manifests,
        "library_id": args.library_id,
        "score_template": args.score_template,
    }
    for replicate in range(args.n_replicates):
        for spec in scenarios:
            checkpoint = _checkpoint_path(args.out_dir, spec.name, replicate)
            if args.resume and checkpoint.is_file():
                payload = _read_checkpoint(checkpoint, fingerprint)
                rows.append(payload["replicate"])
                module_rows.extend(payload["modules"])
            else:
                pending_by_replicate.setdefault(replicate, []).append(spec)

    def record(completed_group: Iterable[dict[str, object]]) -> None:
        for payload in completed_group:
            row = payload["replicate"]
            rows.append(row)
            module_rows.extend(payload["modules"])
            print(
                f"[{datetime.now(UTC).isoformat()}] scenario={row['name']} "
                f"replicate={int(row['replicate']) + 1}/{args.n_replicates} "
                f"screen={row['target_screen_inclusion']} BH={row['target_bh_recovery']} "
                f"D1={row['target_d1_recovery']}",
                flush=True,
            )

    pending = [
        (tuple(specs), replicate, values, args.out_dir, fingerprint)
        for replicate, specs in sorted(pending_by_replicate.items())
    ]
    if args.n_workers == 1:
        _initialize_worker(values)
        for task in pending:
            record(_task_group(task))
    elif pending:
        with ProcessPoolExecutor(
            max_workers=args.n_workers,
            initializer=_initialize_worker,
            initargs=(values,),
        ) as executor:
            futures = [executor.submit(_task_group, task) for task in pending]
            for future in as_completed(futures):
                record(future.result())
    replicates = pd.DataFrame(rows).sort_values(["name", "replicate"], kind="mergesort")
    modules = pd.DataFrame(module_rows).sort_values(
        ["scenario", "replicate", "module_name"], kind="mergesort"
    )
    summary = summarize(replicates)
    attrition = gate_attrition_table(replicates)
    replicates.to_csv(tables_dir / "v171_e2e_synthetic_power_replicates.tsv", sep="\t", index=False)
    modules.to_csv(tables_dir / "v171_e2e_synthetic_power_modules.tsv", sep="\t", index=False)
    summary.to_csv(tables_dir / "v171_e2e_synthetic_power_summary.tsv", sep="\t", index=False)
    attrition.to_csv(
        tables_dir / "v171_e2e_synthetic_power_gate_attrition.tsv", sep="\t", index=False
    )
    manifest = build_manifest(args, scenarios, fingerprint)
    (args.out_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = "\n".join(
        [
            "# V1.7.1 annotation-off 5x2 end-to-end synthetic power",
            "",
            summary.to_string(index=False),
            "",
            f"Protocol: {PROTOCOL}",
            "Defaults are smoke-scale and are not a formal power experiment.",
        ]
    )
    (reports_dir / "v171_e2e_synthetic_power.md").write_text(report + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
