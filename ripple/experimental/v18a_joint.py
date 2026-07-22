"""V1.8a joint capped-score/raw-tail profile mixture.

The capped score provides the weak/moderate channel. An uncapped raw-tail
indicator is a separate soft strong/outlier channel, rather than a hard locus
exclusion. This avoids both V1.8 cap-induced strong-to-weak misclassification
and the loss of moderate signal caused by V1.8a hard tail removal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from ripple.experimental.v175 import conditional_rank_normal_scores
from ripple.experimental.v18_mixture import V18MixtureConfig, V18ProfileResult
from ripple.modules.adaptive import AdaptiveLocusConfig, AdaptiveLocusContext, _collapse_observed_module, _prepare_matched_module_pools, _sample_matched_module
from ripple.modules.anchored import empirical_upper


V18A_JOINT_STAT_VERSION = "v18a_joint_capped_score_raw_tail_profile_v1"


@dataclass(frozen=True)
class V18AJointConfig:
    raw_tail_threshold: float = 3.0
    strong_tail_probability: float = 0.80
    null_tail_probability_floor: float = 0.01
    mixture: V18MixtureConfig = V18MixtureConfig()


def _bernoulli_log_probability(indicator: np.ndarray, probability: np.ndarray | float) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), 1e-8, 1.0 - 1e-8)
    return np.where(indicator, np.log(p), np.log1p(-p))


def _weak_log_density(z: np.ndarray, means: tuple[float, ...]) -> np.ndarray:
    shifts = np.asarray(means, dtype=float)
    return logsumexp(norm.logpdf(z[:, :, None] - shifts[None, None, :]), axis=2) - np.log(shifts.size)


def _fit_weights(log_density: np.ndarray, initial: np.ndarray, config: V18MixtureConfig):
    """Deterministic batch EM for a fixed, finite component family."""

    n_rows = log_density.shape[0]
    initial_array = np.asarray(initial, dtype=float)
    weights = np.tile(initial_array, (n_rows, 1)) if initial_array.ndim == 1 else initial_array.copy()
    if weights.shape != (n_rows, log_density.shape[2]):
        raise ValueError("initial mixture weights do not align to the component matrix")
    previous = np.full(n_rows, -np.inf)
    converged = np.zeros(n_rows, dtype=bool)
    iterations = np.zeros(n_rows, dtype=int)
    for iteration in range(1, config.max_iter + 1):
        weighted = log_density + np.log(np.clip(weights, 1e-300, None))[:, None, :]
        log_norm = logsumexp(weighted, axis=2)
        responsibilities = np.exp(weighted - log_norm[:, :, None])
        updated = responsibilities.mean(axis=1)
        likelihood = log_norm.sum(axis=1)
        if np.any(likelihood + 1e-10 < previous):
            raise RuntimeError("joint-mixture EM likelihood decreased")
        likelihood_stable = np.abs(likelihood - previous) <= config.tolerance * (1.0 + np.abs(previous))
        # Boundary mixtures can retain slowly changing weights after the
        # profile likelihood has stabilized. The frozen tolerance applies to
        # the likelihood objective; requiring a weight-delta criterion here
        # would falsely label valid boundary fits as failed.
        newly_converged = (iteration > 1) & likelihood_stable
        already_converged = converged.copy()
        iterations[newly_converged & ~already_converged] = iteration
        converged |= newly_converged
        weights = np.where(already_converged[:, None], weights, updated)
        previous = likelihood
        if converged.all():
            break
    iterations[~converged] = config.max_iter
    weighted = log_density + np.log(np.clip(weights, 1e-300, None))[:, None, :]
    log_norm = logsumexp(weighted, axis=2)
    responsibilities = np.exp(weighted - log_norm[:, :, None])
    return log_norm.sum(axis=1), weights, responsibilities, converged, iterations


def joint_raw_tail_profile(observed_capped: np.ndarray, observed_raw: np.ndarray, null_capped: np.ndarray, null_raw: np.ndarray, config: V18AJointConfig | None = None) -> V18ProfileResult:
    """Profile H0 strong+null versus H1 strong+weak+null on matched matrices."""

    frozen = config or V18AJointConfig()
    observed_z, null_z = conditional_rank_normal_scores(observed_capped, null_capped)
    z = np.vstack([observed_z, null_z])
    raw = np.vstack([np.asarray(observed_raw, dtype=float), np.asarray(null_raw, dtype=float)])
    if raw.shape != z.shape or not np.isfinite(raw).all():
        raise ValueError("raw and capped matrices must be aligned and finite")
    tail = raw > frozen.raw_tail_threshold
    null_tail_probability = np.clip(null_raw > frozen.raw_tail_threshold, frozen.null_tail_probability_floor, 1.0 - frozen.null_tail_probability_floor).mean(axis=0)
    null_tail_probability = np.clip(null_tail_probability, frozen.null_tail_probability_floor, 1.0 - frozen.null_tail_probability_floor)
    log_null = norm.logpdf(z) + _bernoulli_log_probability(tail, null_tail_probability[None, :])
    log_weak = _weak_log_density(z, frozen.mixture.weak_means) + _bernoulli_log_probability(tail, null_tail_probability[None, :])
    log_strong = norm.logpdf(z) + _bernoulli_log_probability(tail, frozen.strong_tail_probability)
    h0_ll, h0_w, h0_r, h0_ok, h0_iter = _fit_weights(np.stack([log_null, log_strong], axis=2), np.array([0.90, 0.10]), frozen.mixture)
    h1_initial = np.column_stack([h0_w[:, 0] * 0.99, np.full(z.shape[0], 0.01), h0_w[:, 1] * 0.99])
    h1_ll, h1_w, h1_r, h1_ok, h1_iter = _fit_weights(np.stack([log_null, log_weak, log_strong], axis=2), h1_initial, frozen.mixture)
    use_h0 = h1_ll < h0_ll
    h1_ll = np.maximum(h1_ll, h0_ll)
    h1_w[use_h0] = np.column_stack([h0_w[use_h0, 0], np.zeros(use_h0.sum()), h0_w[use_h0, 1]])
    h1_r[use_h0] = np.stack([h0_r[use_h0, :, 0], np.zeros((use_h0.sum(), z.shape[1])), h0_r[use_h0, :, 1]], axis=2)
    statistic = np.maximum(0.0, 2.0 * (h1_ll - h0_ll))
    weak = h1_r[:, :, 1]
    strong = h1_r[:, :, 2]
    weak_sum = weak.sum(axis=1)
    weak_sq = np.square(weak).sum(axis=1)
    effective = np.divide(np.square(weak_sum), weak_sq, out=np.zeros_like(weak_sum), where=(weak_sum > 1e-8) & (weak_sq > 0))
    result: dict[str, float | str | bool] = {
        "v18a_joint_raw_tail_threshold": frozen.raw_tail_threshold,
        "v18a_joint_n_raw_tail_loci": float(tail[0].sum()),
        "v18a_joint_profile_lrt_weak_given_strong": float(statistic[0]),
        "v18a_joint_pi_weak_hat": float(h1_w[0, 1]),
        "v18a_joint_pi_strong_hat": float(h1_w[0, 2]),
        "v18a_joint_expected_weak_loci": float(weak_sum[0]),
        "v18a_joint_expected_strong_loci": float(strong[0].sum()),
        "v18a_joint_effective_weak_loci": float(effective[0]),
        "v18a_joint_fit_converged": bool(h0_ok[0] and h1_ok[0]),
        "v18a_joint_h0_iterations": float(h0_iter[0]),
        "v18a_joint_h1_iterations": float(h1_iter[0]),
        "v18a_joint_profile_lrt_weak_given_strong_empirical_p": empirical_upper(statistic[1:], statistic[0]),
        "v18a_joint_statistic_direction": "greater_is_more_extreme",
        "ripple_d_stat_version": V18A_JOINT_STAT_VERSION,
    }
    return V18ProfileResult(result, {"v18a_joint_profile_lrt_weak_given_strong": statistic[1:]}, observed_z, null_z, {"null": h1_r[0, :, 0], "weak": weak[0], "strong": strong[0]})


def adaptive_locus_module_test_v18a_joint(context: AdaptiveLocusContext, genes: set[str] | frozenset[str], config: AdaptiveLocusConfig, *, n_null: int, rng: np.random.Generator, joint_config: V18AJointConfig | None = None) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Evaluate V1.8a joint mixture with the unchanged matched-locus sampler."""

    capped, _, raw, indices, counts, selected = _collapse_observed_module(context, set(genes))
    if len(indices) < 2:
        return {"test_status": "not_tested_fewer_than_two_loci"}, {}
    prepared = _prepare_matched_module_pools(context, indices, counts)
    null_capped = np.empty((int(n_null), len(indices)))
    null_raw = np.empty_like(null_capped)
    keys = ("null_exact_match_rate", "null_degree_fallback_rate", "null_global_fallback_rate", "null_reuse_fallback_rate", "min_match_pool_size", "median_match_pool_size", "within_locus_replacement_rate")
    sums = {key: 0.0 for key in keys}
    for replicate in range(int(n_null)):
        capped_row, _, raw_row, audit = _sample_matched_module(context, indices, counts, rng, prepared=prepared)
        null_capped[replicate] = capped_row
        null_raw[replicate] = raw_row
        for key in keys:
            sums[key] += float(audit[key])
    profile = joint_raw_tail_profile(capped, raw, null_capped, null_raw, joint_config)
    row: dict[str, object] = {"test_status": "tested", "n_present": int(selected["gene_symbol"].nunique()), "n_loci": int(len(indices)), **profile.observed}
    for key, value in sums.items():
        row[key] = value / int(n_null)
    return row, profile.null_statistics
