from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "outputs",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.sha256":
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in relative.parts):
            continue
        rows.append(f"{digest(path)}  {relative.as_posix()}")
    (ROOT / "MANIFEST.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} hashes")


if __name__ == "__main__":
    main()
