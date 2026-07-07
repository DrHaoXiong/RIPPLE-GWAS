import numpy as np
import pandas as pd
import pytest

from ripple.defaults import DEGREE_RESIDUALIZATION_COVARIATE, PRIMARY_RESIDUALIZATION_COVARIATES
from ripple.signals.residualize import (
    append_residualized_score,
    covariate_matrix_from_frame,
    residualize_from_null,
    residualize_with_regression,
    validate_primary_covariates,
)
from ripple.signals.safety import (
    append_safety_columns,
    build_gene_safety_flags,
    overlapping_special_regions,
    summarize_clipping,
)


def test_build_gene_safety_flags_marks_one_snp_low_info_and_mhc():
    flags = build_gene_safety_flags(
        gene_id="GENE1",
        n_mapped_snps=1,
        m_eff=1.0,
        chrom="chr6",
        start=26_000_000,
        end=26_100_000,
        is_p_clipped=True,
    )

    assert flags.is_one_snp_gene
    assert flags.is_low_information
    assert not flags.is_high_snp_count
    assert flags.is_special_region
    assert flags.special_region_labels == ("MHC",)
    assert flags.is_p_clipped


def test_build_gene_safety_flags_marks_high_snp_count_and_apoe():
    flags = build_gene_safety_flags(
        gene_id="APOE_WINDOW",
        n_mapped_snps=6_000,
        m_eff=120.0,
        chrom=19,
        start=45_350_000,
        end=45_550_000,
    )

    assert flags.is_high_snp_count
    assert not flags.is_low_information
    assert flags.special_region_labels == ("APOE",)


def test_overlapping_special_regions_missing_coordinates_returns_empty():
    assert overlapping_special_regions(None, None, None) == ()
    assert overlapping_special_regions("1", 1_000, 2_000) == ()


def test_summarize_clipping_counts_flags():
    summary = summarize_clipping([True, False, True, False])
    assert summary.n_total == 4
    assert summary.n_clipped == 2
    assert summary.fraction_clipped == pytest.approx(0.5)


def test_append_safety_columns_adds_expected_flags():
    table = pd.DataFrame(
        {
            "gene_id": ["A", "B"],
            "n_mapped_snps": [1, 7_500],
            "m_eff": [1.0, 20.0],
            "chrom": ["6", "1"],
            "start": [25_500_000, 100],
            "end": [25_600_000, 200],
            "is_p_clipped": [False, True],
        }
    )

    out = append_safety_columns(table)
    assert list(out["is_low_information"]) == [True, False]
    assert list(out["is_high_snp_count"]) == [False, True]
    assert list(out["special_region_labels"]) == ["MHC", ""]


def _synthetic_covariates(n_genes=300):
    rng = np.random.default_rng(42)
    x1 = rng.normal(size=n_genes)
    x2 = rng.normal(size=n_genes)
    x3 = rng.normal(size=n_genes)
    covariates = np.column_stack([x1, x2, x3])
    beta = np.array([0.8, -0.5, 0.25])
    return rng, covariates, beta


def test_residualize_from_null_removes_technical_covariate_signal():
    rng, covariates, beta = _synthetic_covariates()
    null_scores = np.vstack(
        [covariates @ beta + rng.normal(scale=0.7, size=covariates.shape[0]) for _ in range(80)]
    )
    observed = covariates @ beta + rng.normal(scale=0.7, size=covariates.shape[0])
    names = PRIMARY_RESIDUALIZATION_COVARIATES[:3]

    before_corr = abs(np.corrcoef(observed, covariates[:, 0])[0, 1])
    result = residualize_from_null(observed, null_scores, covariates, names)
    after_corr = abs(np.corrcoef(result.residualized_score, covariates[:, 0])[0, 1])

    assert before_corr > 0.45
    assert after_corr < 0.10
    assert result.method == "null_estimated_linear_mean_per_gene_sigma"
    assert result.residualized_score.shape == observed.shape
    assert np.all(result.sigma0 > 0)


def test_residualize_with_regression_standardizes_observed_residuals():
    rng, covariates, beta = _synthetic_covariates(n_genes=200)
    observed = covariates @ beta + rng.normal(scale=0.5, size=covariates.shape[0])
    names = PRIMARY_RESIDUALIZATION_COVARIATES[:3]

    result = residualize_with_regression(observed, covariates, names)
    assert np.mean(result.residualized_score) == pytest.approx(0.0, abs=1e-12)
    assert np.std(result.residualized_score, ddof=1) == pytest.approx(1.0)
    assert result.method == "observed_regression_fallback"


def test_degree_covariate_rejected_by_default_and_allowed_for_sensitivity():
    names = (*PRIMARY_RESIDUALIZATION_COVARIATES[:2], DEGREE_RESIDUALIZATION_COVARIATE)
    with pytest.raises(ValueError, match="not allowed"):
        validate_primary_covariates(names)

    assert validate_primary_covariates(names, allow_degree=True) == names


def test_covariate_matrix_from_frame_and_append_residualized_score():
    rng, covariates, beta = _synthetic_covariates(n_genes=50)
    names = PRIMARY_RESIDUALIZATION_COVARIATES[:3]
    observed = covariates @ beta + rng.normal(scale=0.5, size=covariates.shape[0])
    table = pd.DataFrame(covariates, columns=names)
    table["assoc_normal_score_g"] = observed

    extracted = covariate_matrix_from_frame(table, names)
    np.testing.assert_allclose(extracted, covariates)

    out, result = append_residualized_score(table, covariate_names=names)
    assert "assoc_resid_score" in out.columns
    np.testing.assert_allclose(out["assoc_resid_score"], result.residualized_score)
