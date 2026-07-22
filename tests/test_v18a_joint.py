import numpy as np

from ripple.experimental.v18a_joint import joint_raw_tail_profile


def test_joint_profile_assigns_eight_raw_tail_loci_to_strong_not_weak():
    rng = np.random.default_rng(1811)
    null = rng.normal(size=(300, 30))
    capped = np.minimum(null, 3.0)
    observed_capped = np.zeros(30)
    observed_capped[:8] = 3.0
    observed_raw = observed_capped.copy()
    observed_raw[:8] = 6.0
    result = joint_raw_tail_profile(observed_capped, observed_raw, capped, null)
    assert result.observed["v18a_joint_profile_lrt_weak_given_strong_empirical_p"] > 0.05
    assert result.observed["v18a_joint_expected_strong_loci"] > result.observed["v18a_joint_expected_weak_loci"]


def test_joint_profile_retains_distributed_moderate_signal():
    rng = np.random.default_rng(1812)
    null = rng.normal(size=(300, 30))
    observed = np.zeros(30)
    observed[:15] = 1.25
    result = joint_raw_tail_profile(observed, observed, np.minimum(null, 3.0), null)
    assert result.observed["v18a_joint_profile_lrt_weak_given_strong_empirical_p"] < 0.05
    assert result.observed["v18a_joint_expected_weak_loci"] >= 5.0
