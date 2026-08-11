# Forest phenology disturbance forecasting with Sentinel-1/2

This repository contains code and derived point-level data for forecasting the
Sentinel-2 reflectance expected under undisturbed forest conditions. Candidate
disturbance evidence is the residual between a valid target observation and
that expected-normal forecast. Sentinel-1 history, Sentinel-2 history, local
3 x 3 summaries, acquisition timing, quality support, and forest type are used
under a common spatial and temporal split.

The repository contains no original satellite images and no original event
polygons. It provides model-ready feature tensors, fixed split identifiers,
frozen out-of-fold predictions, rasterized event labels, and selected
visualization windows.

## Evidence reproduced here

- 516,096 extracted pixel-time samples in five spatial-fold shards.
- 2019-2022 Training, 2023 Validation, and spatially held 2024-2025 Test data.
- 147,456 Test pixels, 1,152 target groups, 48 normal windows, and 43 blocks.
- Physical baselines, a pooled state-space model, a causal Transformer, a
  frozen Presto comparison, a GRU, and two forest-season corrections.
- Five history-supported retrospective event cases for pixel-ranking analysis.

The event experiment is retrospective. It evaluates pixel ranking on supported
event windows. It does not estimate fixed-threshold operational precision,
false alarms, or warning delay.

## Quick reproduction

```bash
git clone https://github.com/GeoAI-Sustainability-Lab/forest-phenology-disturbance.git
cd forest-phenology-disturbance
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[test]"
.venv\Scripts\python scripts\verify_release.py
.venv\Scripts\python scripts\reproduce_results.py
.venv\Scripts\python -m pytest -q
```

Linux or macOS:

```bash
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python scripts/verify_release.py
.venv/bin/python scripts/reproduce_results.py
.venv/bin/python -m pytest -q
```

`reproduce_results.py` recomputes pixel metrics, group-balanced summaries,
forest strata, paired spatial-block gains, and retrospective event summaries.
It writes the numerical tables to `outputs/tables`, metrics to
`outputs/metrics`, and eight figures to `outputs/figures`. The script then
checks the principal values against `expected/results.json`.

## Optional model refit

Install the training dependency and run a smoke test first.

```powershell
.venv\Scripts\python -m pip install -e ".[train]"
.venv\Scripts\python scripts\train_temporal_models.py --smoke
```

A full five-fold GRU refit is available with:

```powershell
.venv\Scripts\python scripts\train_temporal_models.py --model gru --history 8 --epochs 12 --seeds 4707 4721 4733
```

The canonical numerical reproduction uses the released out-of-fold
predictions. Neural refits can differ slightly across CUDA, PyTorch, and GPU
versions even when seeds are fixed. Presto weights and third-party source code
are not redistributed. Its common-Test prediction is included for numerical
comparison, while a new Presto embedding extraction requires the official
upstream implementation and weights.

## Repository structure

```text
data/features/       extracted 16-step model inputs, one shard per spatial fold
data/evaluation/     fixed predictions, metrics, splits, and phenology selections
data/cases/          extracted normal and event visualization windows
data/context/        WGS84 study points and reduced cartographic context
src/                 metrics, loaders, models, phenology, and figures
scripts/             verification, reproduction, and optional refit entry points
expected/            frozen numerical contract
tests/               unit and release-contract tests
```

The array schema and field definitions are documented in
[`docs/data_dictionary.md`](docs/data_dictionary.md). Reproduction design and
claim boundaries are documented in
[`docs/reproducibility.md`](docs/reproducibility.md).

## License and citation

Code is released under the MIT License. Derived data are released under the
terms described in [`DATA_LICENSE.md`](DATA_LICENSE.md). Source observations
retain their original provider terms. Citation metadata are available in
[`CITATION.cff`](CITATION.cff).

