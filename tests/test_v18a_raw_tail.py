import numpy as np

from ripple.experimental.v18a_raw_tail import raw_tail_conditioned_profile


def test_raw_tail_conditioning_removes_eight_strong_artifact_from_weak_test():
    rng = np.random.default_rng(1801)
    null_capped = np.minimum(rng.normal(size=(300, 30)), 3.0)
    null_raw = null_capped.copy()
    observed_capped = np.zeros(30)
    observed_capped[:8] = 3.0
    observed_raw = observed_capped.copy()
    observed_raw[:8] = 6.0
    result = raw_tail_conditioned_profile(observed_capped, observed_raw, null_capped, null_raw)
    assert result.observed["v18a_n_raw_tail_loci"] == 8.0
    assert result.observed["v18a_profile_lrt_weak_given_raw_tail_empirical_p"] > 0.05


def test_raw_tail_conditioning_retains_distributed_moderate_signal():
    rng = np.random.default_rng(1802)
    null_capped = np.minimum(rng.normal(size=(300, 30)), 3.0)
    null_raw = null_capped.copy()
    observed = np.zeros(30)
    observed[:15] = 1.25
    result = raw_tail_conditioned_profile(observed, observed, null_capped, null_raw)
    assert result.observed["v18a_n_raw_tail_loci"] == 0.0
    assert result.observed["v18a_profile_lrt_weak_given_raw_tail_empirical_p"] < 0.05


def test_raw_tail_selection_is_repeated_for_every_null_row():
    rng = np.random.default_rng(1803)
    null_capped = np.minimum(rng.normal(size=(50, 6)), 3.0)
    null_raw = null_capped.copy()
    null_raw[0, 0] = 5.0
    result = raw_tail_conditioned_profile(np.zeros(6), np.zeros(6), null_capped, null_raw)
    assert result.null_z.shape == (50, 6)
    assert np.isfinite(result.null_statistics["v18a_profile_lrt_weak_given_raw_tail"]).all()
