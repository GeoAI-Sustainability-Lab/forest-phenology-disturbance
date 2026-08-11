# Reproducibility contract

## Fixed data split

Training uses 2019-2022 targets from non-held spatial folds. Validation uses
2023 targets from the same non-held folds. Test predictions use 2024-2025
targets from the held fold. Each Test pixel is predicted by a model that did
not receive its spatial fold during parameter fitting or Validation selection.

## Two reproduction levels

1. `scripts/reproduce_results.py` is the canonical release check. It starts
   from frozen out-of-fold predictions and recomputes metrics, tables, and
   figures. This route should match `expected/results.json` within the stated
   numerical tolerance.
2. `scripts/train_temporal_models.py` refits temporal candidates from the
   extracted feature shards. It is slower and can show minor GPU-dependent
   numerical variation. It never reads target-date evidence as an input.

## Support parity

Normal methods are compared on the same six-band Test pixels. Event residuals
are evaluated only where target-date Sentinel-2 is valid, the pixel belongs to
the forest mask, and pre-event history support exists. Event polygons are used
for evaluation, not as model inputs.

## Interpretation boundary

Normal reconstruction is tested on 43 spatial blocks and two unseen years.
Event evidence is a five-case retrospective ranking analysis. It should not be
read as fixed-threshold operational precision, false-alarm density, detection
delay, or validation outside Taiwan.

