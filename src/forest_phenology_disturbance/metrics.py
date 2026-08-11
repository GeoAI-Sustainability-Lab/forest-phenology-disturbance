from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

PREDICTION_METHODS = (
    "last_observation",
    "seasonal_median",
    "validation_selected_physical",
    "shared_residual_ssm",
    "forest_conditioned_ssm",
    "validation_selected_transformer",
    "presto_frozen",
    "validation_selected_gru",
    "recent_season_gru",
    "phenology_calibrated_gru",
)

METHOD_LABELS = {
    "last_observation": "Last observation",
    "seasonal_median": "Seasonal median",
    "validation_selected_physical": "Validation-selected physical phenology",
    "shared_residual_ssm": "Pooled residual SSM",
    "forest_conditioned_ssm": "SSM with forest-type correction",
    "validation_selected_transformer": "Causal Transformer",
    "presto_frozen": "Frozen Presto with residual head",
    "validation_selected_gru": "GRU",
    "recent_season_gru": "GRU with recent-season correction",
    "phenology_calibrated_gru": "GRU with forest-type phenology correction",
    "direct_pair": "Direct difference",
    "gru_h8": "GRU",
    "phenology_basis_gru": "GRU with forest-type phenology correction",
}


def metric_values(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Return the six-band metrics used by the experiment."""

    truth = np.asarray(truth, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    if truth.shape != prediction.shape or truth.ndim != 2 or truth.shape[1] != 6:
        raise ValueError(
            f"Expected matching [N, 6] arrays, got {truth.shape} and {prediction.shape}"
        )
    error = prediction - truth
    mae = float(np.mean(np.abs(error)))
    mape = float(np.mean(np.abs(error) / np.maximum(np.abs(truth), 0.02)) * 100.0)
    mse = float(np.mean(error**2))
    psnr = float(20.0 * math.log10(1.0 / max(math.sqrt(mse), 1e-8)))
    dot = np.sum(truth * prediction, axis=1)
    denominator = np.linalg.norm(truth, axis=1) * np.linalg.norm(prediction, axis=1)
    sam = float(
        np.degrees(np.mean(np.arccos(np.clip(dot / np.maximum(denominator, 1e-8), -1.0, 1.0))))
    )
    return {
        "mae": mae,
        "mape_percent": mape,
        "psnr_db": psnr,
        "sam_degrees": sam,
    }


def build_group_metrics(predictions: dict[str, np.ndarray], groups: pd.DataFrame) -> pd.DataFrame:
    """Compute metrics within every target group before any aggregation."""

    truth = predictions["truth"].astype(np.float32)
    group_ids = predictions["sample_group_id"].astype(np.int64)
    lookup = groups.set_index("sample_group_id").to_dict("index")
    rows: list[dict[str, Any]] = []
    for group_id in np.unique(group_ids):
        selected = group_ids == group_id
        metadata = lookup[int(group_id)]
        for method in PREDICTION_METHODS:
            rows.append(
                {
                    "sample_group_id": int(group_id),
                    **metadata,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "pixels": int(selected.sum()),
                    **metric_values(truth[selected], predictions[method][selected]),
                }
            )
    return pd.DataFrame(rows)


def aggregate_group_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    strata = [
        ("overall", frame),
        *[
            (f"class:{name}", frame[frame["forest_class"].eq(name)])
            for name in ("broadleaf", "conifer", "bamboo")
        ],
        *[(f"year:{year}", frame[frame["target_year"].eq(year)]) for year in (2024, 2025)],
    ]
    for stratum, subset in strata:
        for horizon in (0, 1, 2, 4):
            selected = subset if horizon == 0 else subset[subset["horizon_seasons"].eq(horizon)]
            for method, group in selected.groupby("method"):
                rows.append(
                    {
                        "stratum": stratum,
                        "horizon_seasons": "all" if horizon == 0 else str(horizon),
                        "method": method,
                        "groups": len(group),
                        "windows": int(group["window_id"].nunique()),
                        "spatial_blocks": int(group["spatial_block"].nunique()),
                        "mae": float(group["mae"].mean()),
                        "mape_percent": float(group["mape_percent"].mean()),
                        "psnr_db": float(group["psnr_db"].mean()),
                        "sam_degrees": float(group["sam_degrees"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def phenology_bootstrap(frame: pd.DataFrame, draws: int = 10_000) -> pd.DataFrame:
    """Paired spatial-block bootstrap for the retained phenology correction."""

    h1 = frame[frame["horizon_seasons"].eq(1)]
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(20260811)
    strata = [
        ("overall", h1),
        *[(name, h1[h1["forest_class"].eq(name)]) for name in ("broadleaf", "conifer", "bamboo")],
    ]
    candidate = "phenology_calibrated_gru"
    for stratum, subset in strata:
        for baseline in ("seasonal_median", "validation_selected_gru", "recent_season_gru"):
            for metric in ("mae", "sam_degrees"):
                pivot = subset[subset["method"].isin([baseline, candidate])].pivot(
                    index=["sample_group_id", "spatial_block"], columns="method", values=metric
                )
                pivot["gain"] = pivot[baseline] - pivot[candidate]
                blocks = pivot.groupby("spatial_block")["gain"].mean().to_numpy()
                sampled = blocks[rng.integers(0, len(blocks), size=(draws, len(blocks)))].mean(
                    axis=1
                )
                rows.append(
                    {
                        "stratum": stratum,
                        "candidate": candidate,
                        "baseline": baseline,
                        "metric": metric,
                        "gain": float(blocks.mean()),
                        "ci_low": float(np.quantile(sampled, 0.025)),
                        "ci_high": float(np.quantile(sampled, 0.975)),
                        "spatial_blocks": len(blocks),
                    }
                )
    return pd.DataFrame(rows)


def history_supported_events(
    candidate_metrics: pd.DataFrame,
    phenology_metrics: pd.DataFrame,
    event_catalog: pd.DataFrame,
) -> pd.DataFrame:
    event_ids = set(event_catalog["window_id"])
    base = candidate_metrics[
        candidate_metrics["evaluation_scope"].eq("history_supported")
        & candidate_metrics["auroc"].notna()
        & candidate_metrics["window_id"].isin(event_ids)
    ].copy()
    proposed = phenology_metrics[
        phenology_metrics["method"].eq("phenology_basis_gru")
        & phenology_metrics["auroc"].notna()
        & phenology_metrics["window_id"].isin(event_ids)
    ].copy()
    return pd.concat([base, proposed], ignore_index=True, sort=False)


def aggregate_events(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("method", as_index=False)
        .agg(
            events=("window_id", "nunique"),
            mean_auroc=("auroc", "mean"),
            mean_auprc=("auprc", "mean"),
            mean_topk_iou=("topk_iou", "mean"),
            mean_residual_ratio=("residual_ratio", "mean"),
        )
        .sort_values("mean_auroc", ascending=False)
    )
