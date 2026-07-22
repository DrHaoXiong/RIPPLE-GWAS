"""Experimental RIPPLE-D V1.8 matched-locus profile-mixture inference.

This module is intentionally outside the frozen V1.7 source tree. V1.8 tests
whether a fixed module retains a weak/moderate locus component after a distinct
strong/outlier component is allowed. It is an experimental RC only.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from ripple.experimental.v175 import conditional_rank_normal_scores
from ripple.modules.adaptive import (
    AdaptiveLocusConfig,
    AdaptiveLocusContext,
    _collapse_observed_module,
    _prepare_matched_module_pools,
    _sample_matched_module,
)
from ripple.modules.anchored import empirical_upper


WEAK_MEANS: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00, 1.50)
STRONG_MEANS: tuple[float, ...] = (2.00, 3.00, 4.00)
V18_STAT_VERSION = "v18_profile_mixture_fixed_grid_v1"


@dataclass(frozen=True)
class V18MixtureConfig:
    """Immutable profile-mixture settings frozen for the V1.8 RC."""

    weak_means: tuple[float, ...] = WEAK_MEANS
    strong_means: tuple[float, ...] = STRONG_MEANS
    tolerance: float = 1e-8
    max_iter: int = 500


@dataclass(frozen=True)
class MixtureFit:
    """Deterministic EM fit of null, weak and optional strong components."""

    log_likelihood: float
    null_weight: float
    weak_weight: float
    strong_weight: float
    weak_responsibility: np.ndarray
    strong_responsibility: np.ndarray
    null_responsibility: np.ndarray
    converged: bool
    iterations: int


@dataclass(frozen=True)
class V18ProfileResult:
    """Observed and null profile-LRT results for one matched-locus module."""

    observed: dict[str, float | str | bool]
    null_statistics: dict[str, np.ndarray]
    observed_z: np.ndarray
    null_z: np.ndarray
    observed_posteriors: dict[str, np.ndarray]


def _component_log_densities(values: np.ndarray, means: tuple[float, ...]) -> np.ndarray:
    shifts = np.asarray(means, dtype=float)
    log_terms = norm.logpdf(values[:, None] - shifts[None, :])
    return logsumexp(log_terms, axis=1) - np.log(shifts.size)


def _fit_fixed_mixture_batch(
    values: np.ndarray, *, allow_weak: bool, config: V18MixtureConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Batch EM over rows, preserving a separate fit for every null row."""

    z = np.asarray(values, dtype=float)
    if z.ndim != 2 or z.shape[1] == 0 or not np.isfinite(z).all():
        raise ValueError("mixture fitting requires a finite nonempty locus vector")
    null_density = norm.logpdf(z)
    strong_density = logsumexp(
        norm.logpdf(z[:, :, None] - np.asarray(config.strong_means)[None, None, :]), axis=2
    ) - np.log(len(config.strong_means))
    if allow_weak:
        weak_density = logsumexp(
            norm.logpdf(z[:, :, None] - np.asarray(config.weak_means)[None, None, :]), axis=2
        ) - np.log(len(config.weak_means))
        log_density = np.stack([null_density, weak_density, strong_density], axis=2)
        weights = np.tile(np.array([0.90, 0.05, 0.05], dtype=float), (z.shape[0], 1))
    else:
        log_density = np.stack([null_density, strong_density], axis=2)
        weights = np.tile(np.array([0.90, 0.10], dtype=float), (z.shape[0], 1))
    def em_update(current: np.ndarray) -> np.ndarray:
        weighted = log_density + np.log(np.clip(current, 1e-300, None))[:, None, :]
        log_norm = logsumexp(weighted, axis=2)
        return np.exp(weighted - log_norm[:, :, None]).mean(axis=1)

    def likelihood(current: np.ndarray) -> np.ndarray:
        weighted = log_density + np.log(np.clip(current, 1e-300, None))[:, None, :]
        return logsumexp(weighted, axis=2).sum(axis=1)

    # SQUAREM accelerates the deterministic EM fixed point, without changing
    # the model, objective, initialization, or final EM update. This matters
    # for the many nearly-null rows required for empirical calibration.
    previous = likelihood(weights)
    converged = np.zeros(z.shape[0], dtype=bool)
    iterations = np.zeros(z.shape[0], dtype=int)
    for iteration in range(1, config.max_iter + 1):
        theta1 = em_update(weights)
        theta2 = em_update(theta1)
        r = theta1 - weights
        v = theta2 - theta1 - r
        alpha = -np.sqrt(
            np.divide(
                np.square(r).sum(axis=1),
                np.square(v).sum(axis=1),
                out=np.ones(weights.shape[0]),
                where=np.square(v).sum(axis=1) > 1e-30,
            )
        )
        alpha = np.clip(alpha, -10.0, -1.0)
        accelerated = weights - 2.0 * alpha[:, None] * r + np.square(alpha)[:, None] * v
        accelerated = np.clip(accelerated, 1e-12, None)
        accelerated /= accelerated.sum(axis=1, keepdims=True)
        candidate = em_update(accelerated)
        candidate_ll = likelihood(candidate)
        theta2_ll = likelihood(theta2)
        use_accelerated = candidate_ll >= theta2_ll
        updated = np.where(use_accelerated[:, None], candidate, theta2)
        likelihood_updated = np.where(use_accelerated, candidate_ll, theta2_ll)
        if np.any(likelihood_updated + 1e-10 < previous):
            raise RuntimeError("EM likelihood decreased beyond numerical tolerance")
        likelihood_stable = np.abs(likelihood_updated - previous) <= config.tolerance * (1.0 + np.abs(previous))
        weights_stable = np.max(np.abs(updated - weights), axis=1) <= config.tolerance
        newly_converged = (iteration > 1) & likelihood_stable & weights_stable
        already_converged = converged.copy()
        just_finished = newly_converged & ~already_converged
        iterations[just_finished] = iteration
        converged |= newly_converged
        weights = np.where(already_converged[:, None], weights, updated)
        previous = likelihood_updated
        if converged.all():
            break
    iterations[~converged] = config.max_iter
    weighted = log_density + np.log(np.clip(weights, 1e-300, None))[:, None, :]
    log_norm = logsumexp(weighted, axis=2)
    responsibilities = np.exp(weighted - log_norm[:, :, None])
    return log_norm.sum(axis=1), weights, responsibilities, converged, iterations


def _fit_fixed_mixture(values: np.ndarray, *, allow_weak: bool, config: V18MixtureConfig) -> MixtureFit:
    """Fit one locus vector using the batch engine used for all null rows."""

    z = np.asarray(values, dtype=float).reshape(1, -1)
    likelihood, weights, responsibilities, converged, iterations = _fit_fixed_mixture_batch(z, allow_weak=allow_weak, config=config)
    if allow_weak:
        weak = responsibilities[0, :, 1]
        strong = responsibilities[0, :, 2]
        null = responsibilities[0, :, 0]
        return MixtureFit(float(likelihood[0]), float(weights[0, 0]), float(weights[0, 1]), float(weights[0, 2]), weak, strong, null, bool(converged[0]), int(iterations[0]))
    return MixtureFit(float(likelihood[0]), float(weights[0, 0]), 0.0, float(weights[0, 1]), np.zeros(z.shape[1]), responsibilities[0, :, 1], responsibilities[0, :, 0], bool(converged[0]), int(iterations[0]))


def fit_v18_profile_lrt(values: np.ndarray, config: V18MixtureConfig | None = None) -> tuple[dict[str, float | bool], MixtureFit, MixtureFit]:
    """Fit H0/H1 and return weak-given-strong plus total-enrichment LRTs."""

    frozen = config or V18MixtureConfig()
    h0 = _fit_fixed_mixture(values, allow_weak=False, config=frozen)
    h1 = _fit_fixed_mixture(values, allow_weak=True, config=frozen)
    weak_lrt = max(0.0, 2.0 * (h1.log_likelihood - h0.log_likelihood))
    baseline_ll = float(norm.logpdf(np.asarray(values, dtype=float)).sum())
    any_lrt = max(0.0, 2.0 * (h1.log_likelihood - baseline_ll))
    weak = h1.weak_responsibility
    denominator = float(np.square(weak).sum())
    effective = float(np.square(weak.sum()) / denominator) if weak.sum() > 1e-8 and denominator > 0 else 0.0
    mean_effect = float(np.dot(weak, np.asarray(values, dtype=float)) / weak.sum()) if weak.sum() > 0 else float("nan")
    return {
        "v18_h0_log_likelihood": h0.log_likelihood,
        "v18_h1_log_likelihood": h1.log_likelihood,
        "v18_profile_lrt_weak_given_strong": weak_lrt,
        "v18_profile_lrt_any": any_lrt,
        "v18_pi_weak_hat": h1.weak_weight,
        "v18_pi_strong_hat": h1.strong_weight,
        "v18_expected_weak_loci": float(weak.sum()),
        "v18_expected_strong_loci": float(h1.strong_responsibility.sum()),
        "v18_effective_weak_loci": effective,
        "v18_mean_weak_effect_hat": mean_effect,
        "v18_fit_converged": bool(h0.converged and h1.converged),
        "v18_h0_iterations": float(h0.iterations),
        "v18_h1_iterations": float(h1.iterations),
    }, h0, h1


def profile_lrt_from_matched_matrix(observed: np.ndarray, null_matrix: np.ndarray, config: V18MixtureConfig | None = None) -> V18ProfileResult:
    """Fit observed and every matched-null row after conditional rank mapping."""

    frozen = config or V18MixtureConfig()
    observed_z, null_z = conditional_rank_normal_scores(observed, null_matrix)
    all_z = np.vstack([observed_z, null_z])
    h0_ll, h0_weights, h0_resp, h0_converged, h0_iterations = _fit_fixed_mixture_batch(all_z, allow_weak=False, config=frozen)
    h1_ll, h1_weights, h1_resp, h1_converged, h1_iterations = _fit_fixed_mixture_batch(all_z, allow_weak=True, config=frozen)
    weak_lrt = np.maximum(0.0, 2.0 * (h1_ll - h0_ll))
    any_lrt = np.maximum(0.0, 2.0 * (h1_ll - norm.logpdf(all_z).sum(axis=1)))
    weak_resp = h1_resp[:, :, 1]
    strong_resp = h1_resp[:, :, 2]
    weak_sum = weak_resp.sum(axis=1)
    effective = np.divide(
        np.square(weak_sum), np.square(weak_resp).sum(axis=1), out=np.zeros_like(weak_sum),
        where=(weak_sum > 1e-8) & (np.square(weak_resp).sum(axis=1) > 0),
    )
    result: dict[str, float | str | bool] = {
        "v18_h0_log_likelihood": float(h0_ll[0]), "v18_h1_log_likelihood": float(h1_ll[0]),
        "v18_profile_lrt_weak_given_strong": float(weak_lrt[0]), "v18_profile_lrt_any": float(any_lrt[0]),
        "v18_h0_pi_strong_hat": float(h0_weights[0, 1]),
        "v18_pi_weak_hat": float(h1_weights[0, 1]), "v18_pi_strong_hat": float(h1_weights[0, 2]),
        "v18_expected_weak_loci": float(weak_sum[0]), "v18_expected_strong_loci": float(strong_resp[0].sum()),
        "v18_effective_weak_loci": float(effective[0]),
        "v18_mean_weak_effect_hat": float(np.dot(weak_resp[0], all_z[0]) / weak_sum[0]) if weak_sum[0] > 0 else float("nan"),
        "v18_fit_converged": bool(h0_converged[0] and h1_converged[0]),
        "v18_h0_iterations": float(h0_iterations[0]), "v18_h1_iterations": float(h1_iterations[0]),
        "v18_n_loci_posterior_weak_gt_0p5": float(np.count_nonzero(weak_resp[0] > 0.5)),
        "v18_n_loci_posterior_strong_gt_0p5": float(np.count_nonzero(strong_resp[0] > 0.5)),
    }
    result.update({
        "v18_profile_lrt_weak_given_strong_empirical_p": empirical_upper(weak_lrt[1:], weak_lrt[0]),
        "v18_profile_lrt_any_empirical_p": empirical_upper(any_lrt[1:], any_lrt[0]),
        "v18_statistic_direction": "greater_is_more_extreme",
        "ripple_d_stat_version": V18_STAT_VERSION,
    })
    leave_values: dict[int, np.ndarray] = {}
    for k in (1, 3):
        statistic = np.full(all_z.shape[0], np.nan)
        if all_z.shape[1] > k + 1:
            # Likelihood is invariant to locus order. Sorting once removes each
            # row's top-k loci while preserving null-row reselection exactly.
            remaining = np.sort(all_z, axis=1)[:, : all_z.shape[1] - k]
            leave_h0_ll, _, _, _, _ = _fit_fixed_mixture_batch(remaining, allow_weak=False, config=frozen)
            leave_h1_ll, _, _, _, _ = _fit_fixed_mixture_batch(remaining, allow_weak=True, config=frozen)
            statistic = np.maximum(0.0, 2.0 * (leave_h1_ll - leave_h0_ll))
        leave_values[k] = statistic
        result[f"v18_leave_top{k}_weak_lrt_empirical_p"] = empirical_upper(statistic[1:], statistic[0])
    return V18ProfileResult(result, {
        "v18_profile_lrt_weak_given_strong": weak_lrt[1:],
        "v18_profile_lrt_any": any_lrt[1:],
        "v18_leave_top1_weak_lrt": leave_values[1][1:],
        "v18_leave_top3_weak_lrt": leave_values[3][1:],
    }, observed_z, null_z, {
        "null": h1_resp[0, :, 0],
        "weak": h1_resp[0, :, 1],
        "strong": h1_resp[0, :, 2],
    })


def observed_locus_posterior_table(locus_ids: list[str] | tuple[str, ...], result: V18ProfileResult):
    """Return per-locus posterior evidence for the V1.8 audit table."""

    import pandas as pd

    if len(locus_ids) != result.observed_z.size:
        raise ValueError("locus_ids must align to the observed locus vector")
    return pd.DataFrame({
        "locus_id": [str(value) for value in locus_ids],
        "conditional_z": result.observed_z,
        "posterior_null": result.observed_posteriors["null"],
        "posterior_weak": result.observed_posteriors["weak"],
        "posterior_strong": result.observed_posteriors["strong"],
        "ripple_d_stat_version": V18_STAT_VERSION,
    })


def adaptive_locus_module_test_v18(
    context: AdaptiveLocusContext,
    genes: set[str] | frozenset[str],
    config: AdaptiveLocusConfig,
    *,
    n_null: int,
    rng: np.random.Generator,
    mixture_config: V18MixtureConfig | None = None,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Run a fixed module through the V1.7 matched sampler and V1.8 profile LRT.

    The sampler is deliberately shared with V1.7. The only experimental change
    is the row-wise profile-mixture statistic applied after matched scores have
    been drawn, so observed and null rows retain identical locus matching.
    """

    capped, _, _, observed_indices, observed_counts, selected = _collapse_observed_module(context, set(genes))
    if len(observed_indices) < 2:
        return {"test_status": "not_tested_fewer_than_two_loci"}, {}
    prepared = _prepare_matched_module_pools(context, observed_indices, observed_counts)
    null_matrix = np.empty((int(n_null), len(observed_indices)), dtype=float)
    audit_keys = (
        "null_exact_match_rate", "null_degree_fallback_rate", "null_global_fallback_rate",
        "null_reuse_fallback_rate", "min_match_pool_size", "median_match_pool_size",
        "within_locus_replacement_rate",
    )
    audits: dict[str, list[float]] = {key: [] for key in audit_keys}
    for replicate in range(int(n_null)):
        null_capped, _, _, audit = _sample_matched_module(
            context, observed_indices, observed_counts, rng, prepared=prepared
        )
        null_matrix[replicate] = null_capped
        for key in audit_keys:
            audits[key].append(float(audit[key]))
    result = profile_lrt_from_matched_matrix(capped, null_matrix, mixture_config)
    row: dict[str, object] = {
        "test_status": "tested",
        "n_present": int(selected["gene_symbol"].nunique()),
        "n_loci": int(len(observed_indices)),
        **result.observed,
    }
    for key, values in audits.items():
        row[key] = float(np.mean(values))
    return row, result.null_statistics


def classify_v18_module(row: dict[str, float | bool], *, weak_q: float, null_quality_pass: bool, external_locus_pass: bool, empirical_resolution_pass: bool) -> str:
    """Apply frozen experimental V1.8 labels without trait-specific biology."""

    weak_p = float(row["v18_profile_lrt_weak_given_strong_empirical_p"])
    any_p = float(row["v18_profile_lrt_any_empirical_p"])
    distributed = float(row["v18_expected_weak_loci"]) >= 5.0 and float(row["v18_effective_weak_loci"]) >= 5.0
    technical = null_quality_pass and external_locus_pass and empirical_resolution_pass and bool(row["v18_fit_converged"])
    if weak_q <= 0.10 and distributed and technical:
        return "v18_distributed_mixture_candidate" if float(row["v18_expected_strong_loci"]) < 1.0 else "v18_mixed_strong_and_distributed_candidate"
    if weak_p <= 0.05 and distributed:
        return "v18_nominal_diagnostic"
    if weak_p <= 0.05:
        return "v18_sparse_mixture_diagnostic"
    if any_p <= 0.05:
        return "v18_strong_locus_enrichment_only"
    return "negative"
