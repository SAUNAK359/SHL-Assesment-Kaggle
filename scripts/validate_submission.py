"""Validate the final SHL submission file without third-party dependencies."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def validate(path: Path, expected_rows: int = 216) -> None:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["filename", "label"]:
            raise ValueError(f"Expected columns ['filename', 'label']; found {reader.fieldnames}")
        rows = list(reader)

    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows; found {len(rows)}")

    filenames = [row["filename"] for row in rows]
    if any(not name for name in filenames):
        raise ValueError("Submission contains an empty filename")
    if len(set(filenames)) != len(filenames):
        raise ValueError("Submission contains duplicate filenames")

    labels = []
    for row in rows:
        try:
            value = float(row["label"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Non-numeric label for {row['filename']!r}") from exc
        if not math.isfinite(value) or not 0.0 <= value <= 5.0:
            raise ValueError(f"Label outside [0, 5] for {row['filename']!r}: {value}")
        labels.append(value)

    print(f"OK: {path} | rows={len(rows)} | labels=[{min(labels):.6f}, {max(labels):.6f}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--expected-rows", type=int, default=216)
    args = parser.parse_args()
    validate(args.submission, args.expected_rows)


if __name__ == "__main__":
    main()

