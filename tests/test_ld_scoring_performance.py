import numpy as np
import pytest

from scripts.run_height_ld_null_mvp import build_null_z_matrix_for_snps
from scripts.run_height_ld_null_mvp import payload_cache_is_compatible
from scripts.run_height_ld_null_mvp import read_ld_scoring_payload_cache
from scripts.run_height_ld_null_mvp import write_ld_scoring_payload_cache


def test_mapped_snp_null_matrix_samples_without_replacement():
    z = np.arange(10, dtype=float)

    null_z = build_null_z_matrix_for_snps(z, n_target_snps=4, n_null=5, seed=123)

    assert null_z.shape == (5, 4)
    for row in null_z:
        assert len(set(row.tolist())) == 4
        assert set(row.tolist()).issubset(set(z.tolist()))


def test_mapped_snp_null_matrix_rejects_too_many_targets():
    with pytest.raises(ValueError, match="cannot exceed"):
        build_null_z_matrix_for_snps(np.arange(3, dtype=float), n_target_snps=4, n_null=1, seed=123)


def test_ld_scoring_payload_cache_round_trip(tmp_path):
    path = tmp_path / "payload_cache.npz"
    payloads = [
        {
            "gene_id": "1",
            "gene_symbol": "GENE1",
            "chrom": "1",
            "gene_start": 10,
            "gene_end": 20,
            "snp_ids": ("rs1", "rs2"),
            "weights": np.array([1.0, 0.5]),
            "denominator_variance": 1.25,
            "lambdas": np.array([0.8, 0.2]),
            "m_eff": 1.5,
            "local_ld_score": 2.0,
            "ld_status": "computed",
            "ld_cache_path": "/tmp/1.ld.npz",
        }
    ]
    metadata = {
        "schema_version": 1,
        "mapping_signature": "map",
        "mapped_snp_ids_signature": "snps",
        "ld_cache_dirs_signature": "ld",
        "ld_shrinkage": 0.05,
        "has_identity_fallback": False,
    }

    write_ld_scoring_payload_cache(path, payloads, metadata)
    observed, observed_metadata = read_ld_scoring_payload_cache(path)

    assert observed_metadata == metadata
    assert observed[0]["gene_id"] == "1"
    assert observed[0]["snp_ids"] == ("rs1", "rs2")
    np.testing.assert_allclose(observed[0]["weights"], [1.0, 0.5])
    np.testing.assert_allclose(observed[0]["lambdas"], [0.8, 0.2])
    assert payload_cache_is_compatible(
        observed_metadata,
        mapping_signature="map",
        mapped_snp_ids_signature="snps",
        ld_dirs_signature="ld",
        shrinkage=0.05,
        allow_identity_fallback=False,
    )
