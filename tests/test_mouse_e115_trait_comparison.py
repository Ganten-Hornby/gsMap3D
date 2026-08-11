import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parents[1] / "analysis" / "revision_graph_3d" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from compare_mouse_e115_supervised_trait_mapping import (  # noqa: E402
    align_region_scores,
    align_spot_scores,
    load_region_scores,
    load_spot_scores,
)


def test_spot_scores_use_neg_log10_p_and_align_by_exact_id(tmp_path) -> None:
    two_d_path = tmp_path / "two_d.csv.gz"
    three_d_path = tmp_path / "three_d.csv.gz"
    pd.DataFrame(
        {
            "spot": ["b", "a", "c"],
            "p": [0.1, 0.01, 1.0],
            "neg_log10_p": [1.0, 2.0, 0.0],
            "z": [1.0, 2.0, 0.0],
        }
    ).to_csv(two_d_path, index=False)
    pd.DataFrame(
        {
            "spot": ["a", "b", "c"],
            "p": [0.001, 0.1, 1.0],
            "neg_log10_p": [3.0, 1.0, 0.0],
            "z": [3.0, 1.0, 0.0],
        }
    ).to_csv(three_d_path, index=False)

    aligned = align_spot_scores(
        load_spot_scores(two_d_path), load_spot_scores(three_d_path)
    )

    assert aligned.attrs["score_kind"] == "-log10(p)"
    assert aligned["spot"].tolist() == ["b", "a", "c"]
    np.testing.assert_allclose(aligned["score_2d"], [1.0, 2.0, 0.0])
    np.testing.assert_allclose(aligned["score_3d"], [1.0, 3.0, 0.0])


def test_region_scores_rank_largest_cauchy_signal_first(tmp_path) -> None:
    two_d_path = tmp_path / "two_d_cauchy.csv"
    three_d_path = tmp_path / "three_d_cauchy.csv"
    base = {
        "trait": ["AF", "AF", "AF"],
        "annotation_name": ["heart_regions"] * 3,
        "annotation": ["RV", "LV", "OFT"],
    }
    pd.DataFrame({**base, "p_cauchy": [0.1, 0.001, 0.01]}).to_csv(
        two_d_path, index=False
    )
    pd.DataFrame({**base, "p_cauchy": [0.001, 0.01, 0.1]}).to_csv(
        three_d_path, index=False
    )

    aligned = align_region_scores(
        load_region_scores(two_d_path, "heart_regions"),
        load_region_scores(three_d_path, "heart_regions"),
    ).set_index("region")

    assert aligned.loc["LV", "rank_2d"] == 1
    assert aligned.loc["RV", "rank_3d"] == 1
    assert aligned.loc["OFT", "rank_2d"] == 2
