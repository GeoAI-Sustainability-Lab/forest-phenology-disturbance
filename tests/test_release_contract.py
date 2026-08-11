from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_feature_shards_are_complete_and_unique() -> None:
    indexes = []
    for path in sorted((ROOT / "data" / "features").glob("*.npz")):
        with np.load(path, allow_pickle=False) as source:
            indexes.append(source["global_index"])
    combined = np.concatenate(indexes)
    assert len(combined) == 516096
    assert len(np.unique(combined)) == 516096


def test_test_predictions_have_common_support() -> None:
    with np.load(
        ROOT / "data" / "evaluation" / "normal_test_predictions.npz", allow_pickle=False
    ) as source:
        truth = source["truth"]
        assert truth.shape == (147456, 6)
        for key in (
            "last_observation",
            "seasonal_median",
            "validation_selected_gru",
            "recent_season_gru",
            "phenology_calibrated_gru",
        ):
            assert source[key].shape == truth.shape
            assert np.isfinite(source[key]).all()
