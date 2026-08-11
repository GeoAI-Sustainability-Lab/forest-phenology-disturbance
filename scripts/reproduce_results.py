from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from forest_phenology_disturbance.data import (
    DATA,
    load_event_catalog,
    load_sample_groups,
    load_test_predictions,
)
from forest_phenology_disturbance.figures import make_all
from forest_phenology_disturbance.metrics import (
    METHOD_LABELS,
    aggregate_events,
    aggregate_group_metrics,
    build_group_metrics,
    history_supported_events,
    phenology_bootstrap,
)

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
METRICS = ROOT / "outputs" / "metrics"
EXPECTED = ROOT / "expected" / "results.json"


def table_normal_h1(summary: pd.DataFrame) -> pd.DataFrame:
    order = [
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
    ]
    selected = (
        summary[summary["stratum"].eq("overall") & summary["horizon_seasons"].astype(str).eq("1")]
        .set_index("method")
        .loc[order]
        .reset_index()
    )
    selected.insert(1, "method_label", selected["method"].map(METHOD_LABELS))
    return selected[
        [
            "method",
            "method_label",
            "groups",
            "spatial_blocks",
            "mae",
            "mape_percent",
            "psnr_db",
            "sam_degrees",
        ]
    ]


def table_forest_strata(summary: pd.DataFrame) -> pd.DataFrame:
    methods = [
        "seasonal_median",
        "validation_selected_gru",
        "recent_season_gru",
        "phenology_calibrated_gru",
    ]
    return summary[
        summary["stratum"].isin(["class:broadleaf", "class:conifer", "class:bamboo"])
        & summary["horizon_seasons"].astype(str).eq("1")
        & summary["method"].isin(methods)
    ].copy()


def collect_results(
    predictions: dict[str, np.ndarray],
    group_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    event_metrics: pd.DataFrame,
) -> dict[str, object]:
    h1 = summary[
        summary["stratum"].eq("overall") & summary["horizon_seasons"].astype(str).eq("1")
    ].set_index("method")
    normal_methods = [
        "seasonal_median",
        "validation_selected_gru",
        "recent_season_gru",
        "phenology_calibrated_gru",
    ]
    normal_h1 = {
        method: {
            metric: float(h1.loc[method, metric])
            for metric in ("mae", "mape_percent", "psnr_db", "sam_degrees")
        }
        for method in normal_methods
    }
    gain_rows = bootstrap[
        bootstrap["stratum"].eq("overall") & bootstrap["baseline"].eq("validation_selected_gru")
    ].set_index("metric")
    event_aggregate = aggregate_events(event_metrics).set_index("method")
    event_methods = ["direct_pair", "gru_h8", "phenology_basis_gru"]
    event_means = {
        method: {
            "auroc": float(event_aggregate.loc[method, "mean_auroc"]),
            "auprc": float(event_aggregate.loc[method, "mean_auprc"]),
            "topk_iou": float(event_aggregate.loc[method, "mean_topk_iou"]),
        }
        for method in event_methods
    }
    return {
        "counts": {
            "all_temporal_samples": 516096,
            "normal_test_pixels": len(predictions["truth"]),
            "normal_test_groups": int(group_metrics["sample_group_id"].nunique()),
            "spatial_blocks": int(group_metrics["spatial_block"].nunique()),
            "history_supported_events": int(event_metrics["window_id"].nunique()),
        },
        "normal_h1": normal_h1,
        "phenology_gain_over_gru": {
            "mae": float(gain_rows.loc["mae", "gain"]),
            "sam_degrees": float(gain_rows.loc["sam_degrees", "gain"]),
        },
        "event_means": event_means,
    }


def compare_nested(actual: object, expected: object, path: str = "root") -> list[str]:
    failures: list[str] = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path}: expected mapping"]
        for key, value in expected.items():
            if key not in actual:
                failures.append(f"{path}.{key}: missing")
            else:
                failures.extend(compare_nested(actual[key], value, f"{path}.{key}"))
    elif isinstance(expected, (int, float)):
        tolerance = 5e-7 if isinstance(expected, float) else 0
        if abs(float(actual) - float(expected)) > tolerance:
            failures.append(f"{path}: {actual} != {expected} (tol={tolerance})")
    elif actual != expected:
        failures.append(f"{path}: {actual!r} != {expected!r}")
    return failures


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    predictions = load_test_predictions()
    groups = load_sample_groups()
    group_metrics = build_group_metrics(predictions, groups)
    summary = aggregate_group_metrics(group_metrics)
    bootstrap = phenology_bootstrap(group_metrics)
    candidate_events = pd.read_csv(DATA / "evaluation" / "event_metrics_candidates.csv")
    phenology_events = pd.read_csv(DATA / "evaluation" / "event_metrics_phenology.csv")
    catalog = load_event_catalog()
    events = history_supported_events(candidate_events, phenology_events, catalog)
    event_aggregate = aggregate_events(events)

    group_metrics.to_csv(TABLES / "normal_group_metrics.csv", index=False)
    summary.to_csv(TABLES / "normal_summary_metrics.csv", index=False)
    bootstrap.to_csv(TABLES / "phenology_spatial_block_bootstrap.csv", index=False)
    table_normal_h1(summary).to_csv(TABLES / "table_normal_h1.csv", index=False)
    table_forest_strata(summary).to_csv(TABLES / "table_forest_strata.csv", index=False)
    events.to_csv(TABLES / "event_metrics_history_supported.csv", index=False)
    event_aggregate.to_csv(TABLES / "table_event_aggregate.csv", index=False)

    results = collect_results(predictions, group_metrics, summary, bootstrap, events)
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))
    failures = compare_nested(results, expected)
    audit = {"status": "PASS" if not failures else "FAIL", "failures": failures, "results": results}
    (METRICS / "reproduction_summary.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    if failures:
        print(json.dumps(audit, indent=2))
        raise SystemExit(1)
    make_all(summary, bootstrap, events, catalog)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
