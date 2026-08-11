from __future__ import annotations

import math

import numpy as np

EPOCH = np.datetime64("1970-01-01")


def _day_parts(days: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    safe = np.maximum(days, 0).astype(np.int64)
    dates = EPOCH + safe.astype("timedelta64[D]")
    month = dates.astype("datetime64[M]").astype(np.int64) % 12
    year_start = dates.astype("datetime64[Y]")
    doy = (dates - year_start).astype("timedelta64[D]").astype(np.float32) + 1.0
    return month.astype(np.int64), doy


def physical_prediction(
    arrays: dict[str, np.ndarray], indexes: np.ndarray, method: str, batch_size: int = 32_768
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, len(indexes), batch_size):
        selected = indexes[start : start + batch_size]
        features = arrays["features"][selected].astype(np.float32)
        history = features[..., :6]
        dates = arrays["dates"][selected].astype(np.int32)
        censored = arrays["history_censor"][selected].astype(bool)
        valid = (features[..., 18] > 0.5) & (dates >= 0) & ~censored
        fallback = arrays["baseline_seasonal"][selected].astype(np.float32)
        target_days = arrays["target_days"][selected].astype(np.int32)
        months, doy = _day_parts(dates)
        target_months, target_doy = _day_parts(target_days)
        age_years = np.maximum((target_days[:, None] - dates) / 365.2425, 0.0)
        if method == "seasonal_median":
            prediction = fallback
        elif method == "routed_recent_season":
            prediction = arrays["baseline_routed"][selected].astype(np.float32)
        else:
            if method.startswith("seasonal_ewma"):
                half_life = float(method.rsplit("h", 1)[1])
                same_quarter = (months // 3) == (target_months[:, None] // 3)
                weights = np.exp(-math.log(2.0) * age_years / half_life) * valid * same_quarter
            elif method.startswith("analog"):
                parts = method.split("_")
                sigma = float(parts[1][1:])
                half_life = float(parts[2][1:])
                delta = np.abs(doy - target_doy[:, None])
                distance = np.minimum(delta, 365.2425 - delta)
                season = np.exp(-0.5 * (distance / sigma) ** 2)
                recency = np.exp(-math.log(2.0) * age_years / half_life)
                weights = season * recency * valid
            else:
                raise ValueError(method)
            denominator = weights.sum(axis=1, keepdims=True)
            prediction = np.einsum("nt,ntb->nb", weights, history, optimize=True)
            prediction /= np.maximum(denominator, 1e-8)
            prediction[denominator[:, 0] <= 1e-8] = fallback[denominator[:, 0] <= 1e-8]
        outputs.append(np.clip(prediction, -0.05, 1.25).astype(np.float32))
    return np.concatenate(outputs)


def harmonic_prediction(
    arrays: dict[str, np.ndarray],
    indexes: np.ndarray,
    *,
    order: int,
    include_trend: bool,
    batch_size: int = 8192,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, len(indexes), batch_size):
        selected = indexes[start : start + batch_size]
        features = arrays["features"][selected].astype(np.float32)
        values = features[..., :6]
        dates = arrays["dates"][selected].astype(np.int32)
        censored = arrays["history_censor"][selected].astype(bool)
        valid = (features[..., 18] > 0.5) & (dates >= 0) & ~censored
        fallback = arrays["baseline_seasonal"][selected].astype(np.float32)
        target_days = arrays["target_days"][selected].astype(np.int32)
        _, doy = _day_parts(dates)
        _, target_doy = _day_parts(target_days)
        theta = 2.0 * np.pi * doy / 365.2425
        target_theta = 2.0 * np.pi * target_doy / 365.2425
        columns = [np.ones_like(theta)]
        target_columns = [np.ones_like(target_theta)]
        for harmonic in range(1, order + 1):
            columns.extend([np.sin(harmonic * theta), np.cos(harmonic * theta)])
            target_columns.extend(
                [np.sin(harmonic * target_theta), np.cos(harmonic * target_theta)]
            )
        if include_trend:
            columns.append((dates - target_days[:, None]) / 365.2425)
            target_columns.append(np.zeros_like(target_theta))
        design = np.stack(columns, axis=-1).astype(np.float32)
        target_design = np.stack(target_columns, axis=-1).astype(np.float32)
        age_years = np.maximum((target_days[:, None] - dates) / 365.2425, 0.0)
        weights = np.exp(-math.log(2.0) * age_years / 4.0).astype(np.float32) * valid
        weighted = design * weights[..., None]
        gram = np.einsum("ntk,ntl->nkl", weighted, design, optimize=True)
        rhs = np.einsum("ntk,ntb->nkb", weighted, values, optimize=True)
        parameter_count = design.shape[-1]
        regularization = np.full(parameter_count, 0.035, dtype=np.float32)
        regularization[0] = 1e-4
        if include_trend:
            regularization[-1] = 0.20
        scale = np.maximum(weights.sum(axis=1), 1.0)
        gram += (
            np.eye(parameter_count, dtype=np.float32)[None]
            * regularization[None, :, None]
            * scale[:, None, None]
        )
        prediction = fallback.copy()
        usable = valid.sum(axis=1) >= parameter_count + 1
        if usable.any():
            coefficients = np.linalg.solve(gram[usable], rhs[usable])
            prediction[usable] = np.einsum(
                "nk,nkb->nb", target_design[usable], coefficients, optimize=True
            )
        outputs.append(np.clip(prediction, -0.05, 1.25).astype(np.float32))
    return np.concatenate(outputs)
