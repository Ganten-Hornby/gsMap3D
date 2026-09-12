import numpy as np
import pytest

from gsMap.spatial_ldsc.data.traits import prepare_snp_data_for_blocks


def test_prepare_snp_blocks_requires_at_least_two_blocks():
    with pytest.raises(ValueError, match="at least 2"):
        prepare_snp_data_for_blocks({"chisq": np.ones(10)}, n_blocks=1)


def test_prepare_snp_blocks_requires_one_snp_per_block():
    with pytest.raises(ValueError, match="one SNP per jackknife block"):
        prepare_snp_data_for_blocks({"chisq": np.ones(3)}, n_blocks=4)


def test_prepare_snp_blocks_truncates_to_equal_nonempty_blocks():
    result = prepare_snp_data_for_blocks(
        {
            "chisq": np.arange(11),
            "N": np.arange(11) + 1,
        },
        n_blocks=4,
    )

    assert result["block_size"] == 2
    assert result["n_snps_used"] == 8
    np.testing.assert_array_equal(result["chisq"], np.arange(8))
