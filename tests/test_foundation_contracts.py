from ripple.contracts import RippleConfig
from ripple.defaults import (
    DEFAULT_LD_SHRINKAGE,
    PRIMARY_RESIDUALIZATION_COVARIATES,
    RANK_FRACTION_GRID,
    REQUIRED_PIPELINE_STAGES,
)
from ripple.nulls.pipeline_identity import validate_pipeline_identity
from ripple.pipeline import run_pipeline


def test_frozen_defaults():
    assert DEFAULT_LD_SHRINKAGE == 0.05
    assert RANK_FRACTION_GRID == (0.01, 0.02, 0.05, 0.10, 0.15, 0.20)
    assert "log_graph_degree" not in PRIMARY_RESIDUALIZATION_COVARIATES


def test_pipeline_identity_manifest_matches_for_observed_and_null():
    config = RippleConfig(trait="synthetic", graph_id="toy_ppi")
    observed = run_pipeline(config, mode="observed")
    null = run_pipeline(config, mode="null")
    assert observed.manifest.stages == REQUIRED_PIPELINE_STAGES
    validate_pipeline_identity(observed.manifest, null.manifest)
