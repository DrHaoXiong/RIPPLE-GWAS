"""RIPPLE-D V1.8a raw-tail-conditioned weak-mixture diagnostic.

V1.8 showed that a capped locus score cannot distinguish an outlier from a
moderate signal once both reach the cap. V1.8a therefore uses uncapped raw
locus maxima solely to select strong/outlier loci, then tests the remaining
capped conditional-rank scores for a weak mixture. The selection is repeated
for observed and every matched-null row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from ripple.experimental.v175 import conditional_rank_normal_scores
from ripple.experimental.v18_mixture import V18MixtureConfig, V18ProfileResult
from ripple.modules.adaptive import (
    AdaptiveLocusConfig,
    AdaptiveLocusContext,
    _collapse_observed_module,
    _prepare_matched_module_pools,
    _sample_matched_module,
)
from ripple.modules.anchored import empirical_upper


V18A_STAT_VERSION = "v18a_raw_tail_conditioned_weak_mixture_v1"


@dataclass(frozen=True)
class V18ARawTailConfig:
    """Frozen raw-tail conditioning rule for the V1.8a repair RC."""

    raw_tail_z_threshold: float = 2.5
    min_weak_eligible_loci: int = 2
    mixture: V18MixtureConfig = V18MixtureConfig()


def _weak_log_density(values: np.ndarray, means: tuple[float, ...]) -> np.ndarray:
    shifts = np.asarray(means, dtype=float)
    return logsumexp(
        norm.logpdf(values[:, :, None] - shifts[None, None, :]), axis=2
    ) - np.log(shifts.size)


def _fit_weak_only_batch(values: np.ndarray, eligible: np.ndarray, config: V18MixtureConfig):
    """Fit null/weak weights by accelerated EM while excluding raw-tail loci."""

    z = np.asarray(values, dtype=float)
    mask = np.asarray(eligible, dtype=bool)
    if z.ndim != 2 or mask.shape != z.shape or not np.isfinite(z).all():
        raise ValueError("values and eligible mask must be finite aligned matrices")
    counts = mask.sum(axis=1)
    if np.any(counts < 2):
        raise ValueError("each row requires at least two weak-eligible loci")
    density = np.stack([norm.logpdf(z), _weak_log_density(z, config.weak_means)], axis=2)
    weights = np.tile(np.array([0.90, 0.10]), (z.shape[0], 1))

    def em_update(current: np.ndarray) -> np.ndarray:
        weighted = density + np.log(np.clip(current, 1e-300, None))[:, None, :]
        normed = logsumexp(weighted, axis=2)
        responsibility = np.exp(weighted - normed[:, :, None])
        responsibility[~mask] = 0.0
        return responsibility.sum(axis=1) / counts[:, None]

    def likelihood(current: np.ndarray) -> np.ndarray:
        weighted = density + np.log(np.clip(current, 1e-300, None))[:, None, :]
        return (logsumexp(weighted, axis=2) * mask).sum(axis=1)

    previous = likelihood(weights)
    converged = np.zeros(z.shape[0], dtype=bool)
    iterations = np.zeros(z.shape[0], dtype=int)
    for iteration in range(1, config.max_iter + 1):
        theta1 = em_update(weights)
        theta2 = em_update(theta1)
        r = theta1 - weights
        v = theta2 - theta1 - r
        alpha = -np.sqrt(np.divide(np.square(r).sum(axis=1), np.square(v).sum(axis=1), out=np.ones(z.shape[0]), where=np.square(v).sum(axis=1) > 1e-30))
        alpha = np.clip(alpha, -10.0, -1.0)
        accelerated = weights - 2.0 * alpha[:, None] * r + np.square(alpha)[:, None] * v
        accelerated = np.clip(accelerated, 1e-12, None)
        accelerated /= accelerated.sum(axis=1, keepdims=True)
        candidate = em_update(accelerated)
        candidate_ll = likelihood(candidate)
        theta2_ll = likelihood(theta2)
        use_accelerated = candidate_ll >= theta2_ll
        updated = np.where(use_accelerated[:, None], candidate, theta2)
        updated_ll = np.where(use_accelerated, candidate_ll, theta2_ll)
        if np.any(updated_ll + 1e-10 < previous):
            raise RuntimeError("V1.8a EM likelihood decreased")
        likelihood_stable = np.abs(updated_ll - previous) <= config.tolerance * (1.0 + np.abs(previous))
        weights_stable = np.max(np.abs(updated - weights), axis=1) <= config.tolerance
        newly_converged = (iteration > 1) & likelihood_stable & weights_stable
        already_converged = converged.copy()
        iterations[newly_converged & ~already_converged] = iteration
        converged |= newly_converged
        weights = np.where(already_converged[:, None], weights, updated)
        previous = updated_ll
        if converged.all():
            break
    iterations[~converged] = config.max_iter
    weighted = density + np.log(np.clip(weights, 1e-300, None))[:, None, :]
    log_norm = logsumexp(weighted, axis=2)
    responsibility = np.exp(weighted - log_norm[:, :, None])
    responsibility[~mask] = 0.0
    return log_norm, weights, responsibility, converged, iterations


def raw_tail_conditioned_profile(
    observed_capped: np.ndarray,
    observed_raw: np.ndarray,
    null_capped: np.ndarray,
    null_raw: np.ndarray,
    config: V18ARawTailConfig | None = None,
) -> V18ProfileResult:
    """Test a weak mixture after raw-tail strong loci are conditioned out."""

    frozen = config or V18ARawTailConfig()
    observed_z, null_z = conditional_rank_normal_scores(observed_capped, null_capped)
    all_z = np.vstack([observed_z, null_z])
    all_raw = np.vstack([np.asarray(observed_raw, dtype=float), np.asarray(null_raw, dtype=float)])
    if all_raw.shape != all_z.shape or not np.isfinite(all_raw).all():
        raise ValueError("raw and capped matrices must have identical finite shape")
    # Use the uncapped score's matched conditional rank rather than an absolute
    # raw threshold, which would delete moderate signal on a shifted trait scale.
    observed_raw_z, null_raw_z = conditional_rank_normal_scores(observed_raw, null_raw)
    all_raw_z = np.vstack([observed_raw_z, null_raw_z])
    tail_mask = all_raw_z > frozen.raw_tail_z_threshold
    eligible = ~tail_mask
    log_norm, weights, responsibility, converged, iterations = _fit_weak_only_batch(all_z, eligible, frozen.mixture)
    null_ll = (norm.logpdf(all_z) * eligible).sum(axis=1)
    statistic = np.maximum(0.0, 2.0 * (log_norm.sum(axis=1) - null_ll))
    weak_resp = responsibility[:, :, 1]
    weak_sum = weak_resp.sum(axis=1)
    weak_sq = np.square(weak_resp).sum(axis=1)
    effective = np.divide(np.square(weak_sum), weak_sq, out=np.zeros_like(weak_sum), where=(weak_sum > 1e-8) & (weak_sq > 0))
    result: dict[str, float | str | bool] = {
        "v18a_raw_tail_z_threshold": frozen.raw_tail_z_threshold,
        "v18a_raw_tail_rule": "raw_conditional_z_greater_than_threshold",
        "v18a_n_raw_tail_loci": float(tail_mask[0].sum()),
        "v18a_fraction_raw_tail_loci": float(tail_mask[0].mean()),
        "v18a_n_weak_eligible_loci": float(eligible[0].sum()),
        "v18a_profile_lrt_weak_given_raw_tail": float(statistic[0]),
        "v18a_pi_weak_hat": float(weights[0, 1]),
        "v18a_expected_weak_loci": float(weak_sum[0]),
        "v18a_effective_weak_loci": float(effective[0]),
        "v18a_fit_converged": bool(converged[0]),
        "v18a_fit_iterations": float(iterations[0]),
        "v18a_profile_lrt_weak_given_raw_tail_empirical_p": empirical_upper(statistic[1:], statistic[0]),
        "v18a_statistic_direction": "greater_is_more_extreme",
        "ripple_d_stat_version": V18A_STAT_VERSION,
    }
    return V18ProfileResult(result, {"v18a_profile_lrt_weak_given_raw_tail": statistic[1:]}, observed_z, null_z, {"null": responsibility[0, :, 0], "weak": weak_resp[0], "strong": tail_mask[0].astype(float)})


def adaptive_locus_module_test_v18a(
    context: AdaptiveLocusContext,
    genes: set[str] | frozenset[str],
    config: AdaptiveLocusConfig,
    *,
    n_null: int,
    rng: np.random.Generator,
    raw_tail_config: V18ARawTailConfig | None = None,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Run V1.8a with the unchanged V1.7 matched-locus sampler."""

    capped, _, raw, observed_indices, observed_counts, selected = _collapse_observed_module(context, set(genes))
    if len(observed_indices) < 2:
        return {"test_status": "not_tested_fewer_than_two_loci"}, {}
    prepared = _prepare_matched_module_pools(context, observed_indices, observed_counts)
    null_capped = np.empty((int(n_null), len(observed_indices)), dtype=float)
    null_raw = np.empty_like(null_capped)
    audit_keys = ("null_exact_match_rate", "null_degree_fallback_rate", "null_global_fallback_rate", "null_reuse_fallback_rate", "min_match_pool_size", "median_match_pool_size", "within_locus_replacement_rate")
    audit_sums = {key: 0.0 for key in audit_keys}
    for replicate in range(int(n_null)):
        capped_row, _, raw_row, audit = _sample_matched_module(context, observed_indices, observed_counts, rng, prepared=prepared)
        null_capped[replicate] = capped_row
        null_raw[replicate] = raw_row
        for key in audit_keys:
            audit_sums[key] += float(audit[key])
    result = raw_tail_conditioned_profile(capped, raw, null_capped, null_raw, raw_tail_config)
    row: dict[str, object] = {"test_status": "tested", "n_present": int(selected["gene_symbol"].nunique()), "n_loci": int(len(observed_indices)), **result.observed}
    for key, value in audit_sums.items():
        row[key] = value / int(n_null)
    return row, result.null_statistics
