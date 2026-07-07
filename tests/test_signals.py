import numpy as np
import pytest
from scipy import stats

from ripple.defaults import LD_SHRINKAGE_SENSITIVITY
from ripple.signals.signed import shrink_ld, signed_ld_burden
from ripple.signals.unsigned import (
    clip_p_value,
    default_weight_matrix,
    mixture_eigenvalues,
    normal_score_from_p_value,
    normal_scores_from_p_values,
    quadratic_association,
    quadratic_p_value,
    quadratic_p_values,
    quadratic_statistic,
)


def test_shrink_ld_matches_formula():
    ld = np.array([[1.0, 0.4], [0.4, 1.0]])
    observed = shrink_ld(ld, shrinkage=0.10)
    expected = 0.90 * ld + 0.10 * np.eye(2)
    np.testing.assert_allclose(observed, expected)


def test_signed_ld_burden_single_snp_is_signed_z():
    result = signed_ld_burden([2.25], [1.0], np.eye(1))
    assert result.status == "available"
    assert result.score == pytest.approx(2.25)
    assert result.denominator_variance == pytest.approx(1.0)


def test_signed_ld_burden_has_unit_variance_under_matching_ld_null():
    rng = np.random.default_rng(123)
    ld = np.array(
        [
            [1.0, 0.35, 0.10],
            [0.35, 1.0, 0.25],
            [0.10, 0.25, 1.0],
        ]
    )
    weights = np.array([0.5, 1.0, -0.25])

    for shrinkage in LD_SHRINKAGE_SENSITIVITY:
        r_star = shrink_ld(ld, shrinkage=shrinkage)
        samples = rng.multivariate_normal(np.zeros(3), r_star, size=30_000)
        denom = signed_ld_burden(samples[0], weights, ld, shrinkage=shrinkage).denominator
        scores = samples @ weights / denom
        assert np.var(scores) == pytest.approx(1.0, abs=0.04)


def test_signed_ld_burden_flags_zero_weight_instability():
    result = signed_ld_burden([1.0, 2.0], [0.0, 0.0], np.eye(2))
    assert result.status == "unstable"
    assert np.isnan(result.score)


def test_default_weight_matrix_is_squared_diagonal():
    observed = default_weight_matrix([1.0, -2.0, 0.5])
    expected = np.diag([1.0, 4.0, 0.25])
    np.testing.assert_allclose(observed, expected)


def test_quadratic_statistic_matches_z_w_z():
    z = np.array([1.0, 2.0])
    w = np.diag([1.0, 4.0])
    assert quadratic_statistic(z, w) == pytest.approx(17.0)


def test_mixture_eigenvalues_identity_ld_are_weight_diagonal_values():
    lambdas = mixture_eigenvalues(np.eye(2), np.diag([1.0, 4.0]), shrinkage=0.05)
    np.testing.assert_allclose(lambdas, [4.0, 1.0])


def test_quadratic_association_single_snp_matches_chi_square():
    z = [2.0]
    result = quadratic_association(z, np.eye(1), weights=[1.0], method="liu")
    expected_p = stats.chi2.sf(4.0, df=1)
    assert result.statistic == pytest.approx(4.0)
    assert result.assoc_p_g == pytest.approx(expected_p)
    assert result.assoc_minuslog10p_g == pytest.approx(-np.log10(expected_p))
    assert result.assoc_normal_score_g == pytest.approx(stats.norm.isf(expected_p))


def test_quadratic_association_identity_ld_matches_standard_chi_square_with_liu():
    z = np.array([1.0, 1.5, -0.5])
    result = quadratic_association(z, np.eye(3), weights=np.ones(3), method="liu")
    q = float(np.sum(z**2))
    expected_p = stats.chi2.sf(q, df=3)
    assert result.statistic == pytest.approx(q)
    assert result.assoc_p_g == pytest.approx(expected_p)


def test_quadratic_saddlepoint_is_close_to_chi_square_for_identity_ld():
    q = 5.0
    p = quadratic_p_value(q, [1.0, 1.0, 1.0], method="saddlepoint")
    assert p == pytest.approx(stats.chi2.sf(q, df=3), abs=0.02)


def test_vectorized_quadratic_p_values_match_scalar_methods():
    statistics = np.array([0.5, 1.5, 4.0, 8.0])
    lambdas = np.array([1.0, 0.5, 0.25])

    for method in ("liu", "satterthwaite", "saddlepoint"):
        observed = quadratic_p_values(statistics, lambdas, method=method)
        expected = np.array([quadratic_p_value(q, lambdas, method=method) for q in statistics])
        np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)


def test_p_value_clipping_and_normal_score_are_finite():
    clipped_low, was_low = clip_p_value(0.0, epsilon=1e-6)
    clipped_high, was_high = clip_p_value(1.0, epsilon=1e-6)
    assert clipped_low == pytest.approx(1e-6)
    assert clipped_high == pytest.approx(1 - 1e-6)
    assert was_low
    assert was_high

    score, clipped, was_clipped = normal_score_from_p_value(0.0, epsilon=1e-6)
    assert np.isfinite(score)
    assert clipped == pytest.approx(1e-6)
    assert was_clipped


def test_vectorized_normal_scores_match_scalar_transform():
    p_values = np.array([0.0, 1e-4, 0.5, 1.0])

    scores, clipped, was_clipped = normal_scores_from_p_values(p_values, epsilon=1e-6)
    expected = [normal_score_from_p_value(float(p), epsilon=1e-6) for p in p_values]

    np.testing.assert_allclose(scores, [row[0] for row in expected])
    np.testing.assert_allclose(clipped, [row[1] for row in expected])
    np.testing.assert_array_equal(was_clipped, [row[2] for row in expected])


def test_p_value_clipping_keeps_upper_bound_finite_with_tiny_epsilon():
    score, clipped, was_clipped = normal_score_from_p_value(1.0, epsilon=1e-300)

    assert was_clipped
    assert clipped == pytest.approx(1 - 1e-16)
    assert np.isfinite(score)


def test_vectorized_p_value_clipping_uses_practical_upper_bound_with_tiny_epsilon():
    scores, clipped, was_clipped = normal_scores_from_p_values([0.0, 1.0], epsilon=1e-300)

    assert clipped[0] == pytest.approx(1e-300)
    assert clipped[1] == pytest.approx(1 - 1e-16)
    assert np.all(np.isfinite(scores))
    np.testing.assert_array_equal(was_clipped, [True, True])


def test_davies_method_is_explicitly_not_implemented():
    with pytest.raises(NotImplementedError):
        quadratic_p_value(1.0, [1.0], method="davies")


def test_negative_weight_matrix_is_rejected():
    with pytest.raises(ValueError, match="positive semidefinite"):
        quadratic_statistic([1.0, 2.0], np.diag([1.0, -1.0]))
