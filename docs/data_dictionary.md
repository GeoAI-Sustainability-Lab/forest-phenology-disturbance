# Data dictionary

## Feature shards

`data/features/temporal_samples_fold_{0..4}.npz` contains extracted numerical
samples. A row is a pixel-target pair. Rows retain `global_index`, so shards can
be concatenated and restored to the original order.

| Array | Shape per row | Type | Meaning |
|---|---:|---|---|
| `features` | 16 x 22 | float16 | Pre-target optical, radar, local summaries, and support fields |
| `dates` | 16 | int32 | Days since 1970-01-01; -1 denotes padding |
| `history_censor` | 16 | uint8 | One when a history step is excluded after a known event |
| `targets` | 6 | float16 | Target Sentinel-2 B02, B03, B04, B08, B11, and B12 reflectance |
| `target_days` | scalar | int32 | Target date as days since 1970-01-01 |
| `target_years` | scalar | int16 | Calendar year |
| `horizon_seasons` | scalar | uint8 | Forecast horizon in three-month steps |
| `class_codes` | scalar | uint8 | 1 broadleaf, 2 conifer, 3 bamboo |
| `folds` | scalar | uint8 | Spatial fold from 0 to 4 |
| `window_indexes` | scalar | uint8 | Index of the 64 x 64 analysis window |
| `pixel_rows`, `pixel_columns` | scalar | uint8 | Local pixel position |
| `sample_group_ids` | scalar | int32 | Target-window-horizon group identifier |
| `baseline_seasonal` | 6 | float16 | Pixel-wise same-season median |
| `baseline_routed` | 6 | float16 | Quality-routed physical reference |

The 22 feature channels are stored in `feature_names` inside every shard. They
comprise six central Sentinel-2 bands, six 3 x 3 Sentinel-2 means, central and
local Sentinel-1 VV, VH, and VH-VV, plus four quality-support variables.

## Test predictions

`data/evaluation/normal_test_predictions.npz` contains 147,456 out-of-fold Test
pixels. `truth` and each prediction have six bands. Metadata arrays identify
group, fold, forest class, target year, horizon, window, and local pixel.

## Case arrays

Normal case files contain observed reflectance, masks, previous and seasonal
references, GRU forecasts, and phenology-calibrated forecasts. Event case files
contain valid target reflectance, forest and event masks, method predictions,
and residual maps. Grey pixels in rendered figures are invalid or outside the
declared evaluation support. They are not predicted disturbance.

## Tabular files

`sample_groups.csv` defines target windows and dates. `spatial_folds.csv`
defines the fixed block assignment. Phenology tables contain only
quality-screened normal-window summaries and Validation-selected coefficients.

