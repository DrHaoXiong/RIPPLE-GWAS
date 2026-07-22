import numpy as np
import pandas as pd
import pytest

import ripple.experimental.v18_mixture as v18
from ripple.modules.adaptive import AdaptiveLocusConfig, AdaptiveLocusContext
from ripple.modules.distributed import RippleDConfig


def test_profile_fit_is_deterministic_simplex_and_h1_contains_h0():
    values = np.array([0.0] * 15 + [1.0] * 10 + [3.0])
    first, h0, h1 = v18.fit_v18_profile_lrt(values)
    second, _, _ = v18.fit_v18_profile_lrt(values)
    assert first == second
    assert h1.null_weight + h1.weak_weight + h1.strong_weight == pytest.approx(1.0)
    assert h0.null_weight + h0.strong_weight == pytest.approx(1.0)
    assert first["v18_h1_log_likelihood"] >= first["v18_h0_log_likelihood"] - 1e-9
    assert first["v18_fit_converged"]


def test_single_strong_locus_is_assigned_more_strong_than_weak_responsibility():
    _, _, h1 = v18.fit_v18_profile_lrt(np.array([0.0] * 29 + [5.0]))
    assert h1.strong_responsibility[-1] > h1.weak_responsibility[-1]


def test_matched_profile_lrt_detects_distributed_shift_not_single_strong_artifact():
    rng = np.random.default_rng(712)
    nulls = rng.normal(size=(200, 30))
    distributed = np.zeros(30)
    distributed[:15] = 1.25
    single = np.zeros(30)
    single[0] = 6.0
    distributed_result = v18.profile_lrt_from_matched_matrix(distributed, nulls)
    single_result = v18.profile_lrt_from_matched_matrix(single, nulls)
    assert distributed_result.observed["v18_profile_lrt_weak_given_strong_empirical_p"] < 0.05
    assert single_result.observed["v18_profile_lrt_weak_given_strong_empirical_p"] > 0.05


def test_eight_strong_loci_do_not_create_weak_given_strong_evidence():
    rng = np.random.default_rng(721)
    nulls = rng.normal(size=(200, 30))
    eight_strong = np.zeros(30)
    eight_strong[:8] = 6.0
    result = v18.profile_lrt_from_matched_matrix(eight_strong, nulls)
    assert result.observed["v18_profile_lrt_weak_given_strong_empirical_p"] > 0.05
    assert result.observed["v18_expected_strong_loci"] > result.observed["v18_expected_weak_loci"]


def test_rank_transform_handles_ties_and_leave_statistics_use_plus_one_p():
    result = v18.profile_lrt_from_matched_matrix(np.zeros(5), np.zeros((20, 5)))
    assert np.isfinite(result.observed_z).all()
    assert result.observed["v18_profile_lrt_weak_given_strong_empirical_p"] == pytest.approx(1.0)
    assert result.observed["v18_leave_top1_weak_lrt_empirical_p"] == pytest.approx(1.0)


def test_posterior_audit_is_order_aligned_and_sums_to_one():
    rng = np.random.default_rng(12)
    result = v18.profile_lrt_from_matched_matrix(np.array([1.0, 0.0, 0.5]), rng.normal(size=(20, 3)))
    audit = v18.observed_locus_posterior_table(["L3", "L1", "L2"], result)
    assert audit["locus_id"].tolist() == ["L3", "L1", "L2"]
    assert np.allclose(audit[["posterior_null", "posterior_weak", "posterior_strong"]].sum(axis=1), 1.0)


def test_classification_requires_q_and_technical_requirements():
    row = {"v18_profile_lrt_weak_given_strong_empirical_p": 0.01, "v18_profile_lrt_any_empirical_p": 0.01, "v18_expected_weak_loci": 6.0, "v18_effective_weak_loci": 5.5, "v18_expected_strong_loci": 0.2, "v18_fit_converged": True}
    assert v18.classify_v18_module(row, weak_q=0.08, null_quality_pass=True, external_locus_pass=True, empirical_resolution_pass=True) == "v18_distributed_mixture_candidate"
    assert v18.classify_v18_module(row, weak_q=0.08, null_quality_pass=False, external_locus_pass=True, empirical_resolution_pass=True) == "v18_nominal_diagnostic"


def test_adaptive_wrapper_reuses_matched_null_draws(monkeypatch: pytest.MonkeyPatch):
    genes = ["A", "B", "C"]
    work = pd.DataFrame({"gene_symbol": genes, "locus_id": ["L0", "L1", "L2"], "assoc_resid_score": [1.0, 1.0, 1.0], "ripple_d_capped_score": [1.0, 1.0, 1.0], "ripple_d_huber_score": [1.0, 1.0, 1.0]})
    context = AdaptiveLocusContext(work=work, background=work, locus_id_to_index={"L0": 0, "L1": 1, "L2": 2}, match_bin_arr=np.array(["M", "M", "M"], dtype=object), degree_bin_arr=np.zeros(3, dtype=int), locus_gene_count_arr=np.ones(3, dtype=int), all_indices=np.arange(3), match_pools_idx={"M": np.arange(3)}, degree_pools_idx={0: np.arange(3)}, locus_gene_values={index: np.array([[0.0]]) for index in range(3)})
    draws = iter(np.random.default_rng(44).normal(size=(20, 3)))

    def fake_sample(*args: object, **kwargs: object):
        values = next(draws)
        audit = {"null_exact_match_rate": 1.0, "null_degree_fallback_rate": 0.0, "null_global_fallback_rate": 0.0, "null_reuse_fallback_rate": 0.0, "min_match_pool_size": 20.0, "median_match_pool_size": 20.0, "within_locus_replacement_rate": 0.0}
        return values, values, values, audit

    monkeypatch.setattr(v18, "_sample_matched_module", fake_sample)
    row, nulls = v18.adaptive_locus_module_test_v18(context, set(genes), AdaptiveLocusConfig(ripple_d=RippleDConfig(locus_id_column="locus_id")), n_null=20, rng=np.random.default_rng(1))
    assert row["test_status"] == "tested"
    assert len(nulls["v18_profile_lrt_weak_given_strong"]) == 20
