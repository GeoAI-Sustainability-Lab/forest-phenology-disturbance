from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def load_feature_shards(folds: list[int] | None = None) -> dict[str, np.ndarray]:
    """Load feature shards and restore the stable global sample order."""

    selected_folds = list(range(5)) if folds is None else sorted(set(folds))
    parts: dict[str, list[np.ndarray]] = {}
    feature_names: np.ndarray | None = None
    for fold in selected_folds:
        path = DATA / "features" / f"temporal_samples_fold_{fold}.npz"
        with np.load(path, allow_pickle=False) as source:
            feature_names = source["feature_names"]
            for key in source.files:
                if key == "feature_names":
                    continue
                parts.setdefault(key, []).append(source[key])
    arrays = {key: np.concatenate(values, axis=0) for key, values in parts.items()}
    order = np.argsort(arrays["global_index"])
    arrays = {key: value[order] for key, value in arrays.items()}
    if feature_names is not None:
        arrays["feature_names"] = feature_names
    return arrays


def load_test_predictions() -> dict[str, np.ndarray]:
    path = DATA / "evaluation" / "normal_test_predictions.npz"
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def load_sample_groups() -> pd.DataFrame:
    return pd.read_csv(DATA / "evaluation" / "sample_groups.csv")


def load_event_catalog() -> pd.DataFrame:
    return pd.read_csv(DATA / "context" / "event_catalog.csv")
