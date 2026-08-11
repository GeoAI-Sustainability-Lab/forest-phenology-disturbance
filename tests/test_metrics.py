import numpy as np

from forest_phenology_disturbance.metrics import metric_values


def test_identical_spectra_have_zero_error() -> None:
    truth = np.asarray([[0.10, 0.12, 0.15, 0.40, 0.22, 0.18]], dtype=np.float32)
    values = metric_values(truth, truth.copy())
    assert values["mae"] == 0.0
    assert values["mape_percent"] == 0.0
    assert values["sam_degrees"] < 0.03


def test_mape_uses_reflectance_floor() -> None:
    truth = np.zeros((1, 6), dtype=np.float32)
    prediction = np.full((1, 6), 0.01, dtype=np.float32)
    values = metric_values(truth, prediction)
    assert np.isclose(values["mape_percent"], 50.0)
