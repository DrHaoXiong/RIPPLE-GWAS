import numpy as np
import pandas as pd
import pytest

from ripple.claims import FINAL_MANUSCRIPT_Z_THRESHOLD, SUPPORTIVE_Z_THRESHOLD
from ripple.manuscript import COMMON_RESULT_SCHEMA, ensure_columns, validate_vocabulary
from ripple.policy import (
    classify_z_claim,
    controlled_vocabulary,
    empirical_p_from_null,
    final_z_threshold,
    load_claim_policy,
    supportive_z_threshold,
)


def test_claim_policy_loads_thresholds_and_vocabularies():
    policy = load_claim_policy()

    assert final_z_threshold(policy) == 2.5
    assert supportive_z_threshold(policy) == 2.0
    assert FINAL_MANUSCRIPT_Z_THRESHOLD == 2.5
    assert SUPPORTIVE_Z_THRESHOLD == 2.0
    assert "final_positive" in controlled_vocabulary(policy, "claim_status")
    assert "graph_construction_related" in controlled_vocabulary(policy, "annotation_source_type")
    assert "module_layer_policy" in policy
    assert policy["module_layer_policy"]["default_thresholds"]["selection_calibrated_empirical_p"] == 0.05
    assert {
        "post_hoc_candidate_only",
        "fixed_module_supported",
        "selection_calibrated_module",
        "topology_specific_module",
        "no_local_module_support",
        "not_tested",
    }.issubset(set(policy["module_layer_policy"]["module_status_levels"]))
    assert "exploratory" in controlled_vocabulary(policy, "allowed_strength")


def test_z_claim_classification_uses_policy_thresholds():
    policy = load_claim_policy()

    assert classify_z_claim(2.6, policy) == "final_positive"
    assert classify_z_claim(2.2, policy) == "supportive"
    assert classify_z_claim(1.9, policy) == "negative"
    assert classify_z_claim(np.nan, policy) == "not_tested"


def test_empirical_p_uses_plus_one_correction_and_direction():
    null = np.array([0.1, 0.2, 0.3, 0.4])

    assert empirical_p_from_null(null, 0.25, direction="greater_is_more_extreme") == 3 / 5
    assert empirical_p_from_null(null, 0.25, direction="less_is_more_extreme") == 3 / 5
    assert empirical_p_from_null(null, 0.5, direction="greater_is_more_extreme") == 1 / 5


def test_manuscript_schema_and_vocab_validation():
    policy = load_claim_policy()
    table = ensure_columns(
        pd.DataFrame(
            [
                {
                    "trait": "DR",
                    "claim_status": "final_positive",
                    "statistic_direction": "greater_is_more_extreme",
                }
            ]
        ),
        COMMON_RESULT_SCHEMA,
    )
    assert list(table.columns[: len(COMMON_RESULT_SCHEMA)]) == list(COMMON_RESULT_SCHEMA)
    validate_vocabulary(
        table,
        {
            "claim_status": controlled_vocabulary(policy, "claim_status"),
            "statistic_direction": controlled_vocabulary(policy, "statistic_direction"),
        },
        table_name="test_table",
    )

    bad = table.copy()
    bad.loc[0, "claim_status"] = "old_positive"
    with pytest.raises(ValueError, match="invalid values"):
        validate_vocabulary(
            bad,
            {"claim_status": controlled_vocabulary(policy, "claim_status")},
            table_name="bad_table",
        )
