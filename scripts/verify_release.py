from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    failures: list[str] = []
    inventory = json.loads((ROOT / "data" / "RELEASE_INVENTORY.json").read_text(encoding="utf-8"))
    if inventory["all_temporal_samples"] != 516096:
        failures.append("Temporal sample count differs from 516096")
    if inventory["normal_test_samples"] != 147456:
        failures.append("Normal Test count differs from 147456")
    if inventory["raw_imagery_included"] or inventory["original_event_polygons_included"]:
        failures.append("Release inventory indicates prohibited source data")

    shards = sorted((ROOT / "data" / "features").glob("temporal_samples_fold_*.npz"))
    global_indexes = []
    for path in shards:
        if path.stat().st_size >= 100_000_000:
            failures.append(f"GitHub file limit exceeded: {path.name}")
        with np.load(path, allow_pickle=False) as source:
            global_indexes.append(source["global_index"])
            if source["features"].shape[1:] != (16, 22):
                failures.append(f"Feature shape mismatch in {path.name}")
    indexes = np.concatenate(global_indexes)
    if len(indexes) != 516096 or len(np.unique(indexes)) != 516096:
        failures.append("Feature shards do not contain 516096 unique global indexes")

    with np.load(
        ROOT / "data" / "evaluation" / "normal_test_predictions.npz", allow_pickle=False
    ) as source:
        if source["truth"].shape != (147456, 6):
            failures.append("Test truth shape is not [147456, 6]")
        required = {
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
        }
        missing = sorted(required - set(source.files))
        if missing:
            failures.append(f"Missing Test prediction methods: {missing}")

    prohibited_extensions = {".tif", ".tiff", ".jp2", ".safe", ".shp", ".dbf", ".shx"}
    prohibited_files = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in prohibited_extensions
    ]
    if prohibited_files:
        failures.append(f"Source image or vector files present: {prohibited_files}")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for pattern in (r"Geo-spatial Information Science", r"submission", r"target journal"):
        if re.search(pattern, readme, flags=re.IGNORECASE):
            failures.append(f"README contains journal-specific wording: {pattern}")
    local_path_hits = []
    for path in ROOT.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        if re.search(r"[A-Za-z]:\\Users\\", text):
            local_path_hits.append(path.relative_to(ROOT).as_posix())
    if local_path_hits:
        failures.append(f"Local absolute paths found in documentation: {local_path_hits}")

    manifest = ROOT / "MANIFEST.sha256"
    if not manifest.exists():
        failures.append("MANIFEST.sha256 is missing")
    else:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            expected, relative = line.split("  ", 1)
            path = ROOT / relative
            if not path.exists() or digest(path) != expected:
                failures.append(f"Hash mismatch: {relative}")

    report = {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "feature_shards": len(shards),
        "feature_samples": len(indexes),
        "largest_file_bytes": max(
            path.stat().st_size for path in ROOT.rglob("*") if path.is_file()
        ),
        "raw_imagery_present": bool(prohibited_files),
    }
    output = ROOT / "outputs" / "metrics" / "release_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
