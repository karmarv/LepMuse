from __future__ import annotations

import csv
import threading
from pathlib import Path
from typing import Any, Iterable

from .types import MeasurementResult


MOTHRA_COLUMNS = [
    "image_id",
    "left_wing (mm)",
    "right_wing (mm)",
    "left_wing_center (mm)",
    "right_wing_center (mm)",
    "wing_span (mm)",
    "wing_shoulder (mm)",
]

EXTRA_COLUMNS = [
    "image_path",
    "view",
    "specimen_id",
    "stage",
    "segmenter",
    "pixels_per_mm",
    "status",
    "error",
]

DISTANCE_COLUMNS = {
    "left_wing (mm)": "dist_l",
    "right_wing (mm)": "dist_r",
    "left_wing_center (mm)": "dist_l_center",
    "right_wing_center (mm)": "dist_r_center",
    "wing_span (mm)": "dist_span",
    "wing_shoulder (mm)": "dist_shoulder",
}


_STATUS_ORDER = {"failed": 0, "invalid": 1, "ok": 2}


def write_results_csv(results: Iterable[MeasurementResult], path: str | Path) -> None:
    rows = [to_row(result) for result in results]
    # Sort anomalies (failed → invalid → ok) to the top for immediate visibility.
    rows.sort(key=lambda r: _STATUS_ORDER.get(r.get("status", "ok"), 99))
    fieldnames = _fieldnames(rows)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        # QUOTE_NONNUMERIC quotes every str field; keeps numeric columns unquoted.
        # This prevents multiline tracebacks in 'error' from breaking CSV readers.
        writer = csv.DictWriter(
            csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC,
        )
        writer.writeheader()
        writer.writerows(rows)


def to_row(result: MeasurementResult) -> dict[str, Any]:
    row: dict[str, Any] = {
        "image_id": result.image_id,
        "image_path": str(result.image_path),
        "view": result.view or "",
        "specimen_id": result.specimen_id or "",
        "stage": result.stage,
        "segmenter": result.segmenter,
        "pixels_per_mm": result.pixels_per_mm,
        "status": result.status,
        # Flatten multiline tracebacks to a single line so the error column
        # stays readable as a single cell in any CSV viewer.
        "error": _flatten(result.error),
    }
    for column, key in DISTANCE_COLUMNS.items():
        row[column] = result.measurements_mm.get(key)
    for key, value in result.metadata.items():
        row[f"metadata_{key}"] = value
    return row


def _flatten(text: str | None) -> str:
    if not text:
        return ""
    return " | ".join(line.strip() for line in text.splitlines() if line.strip())


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    base = [*MOTHRA_COLUMNS, *EXTRA_COLUMNS]
    extras = sorted({key for row in rows for key in row if key not in base})
    return [*base, *extras]


# ── Incremental (live) CSV writing ───────────────────────────────────────────

_INCREMENTAL_FIELDNAMES = [*MOTHRA_COLUMNS, *EXTRA_COLUMNS]


def _row_key(row: dict) -> str:
    """Unique key for a result row.

    Primary: full image_path.
    Fallback: composite of image_id + view + specimen_id, used only when
    image_path is absent.
    """
    image_path = row.get("image_path", "").strip()
    if image_path:
        return image_path
    return "\x00".join([
        row.get("image_id", ""),
        row.get("view", ""),
        row.get("specimen_id", ""),
    ])


def load_completed_paths(path: str | Path) -> set[str]:
    """Return row keys (image_path or composite fallback) whose status is 'ok'.

    Returns an empty set if the file does not exist or cannot be parsed.
    """
    path = Path(path)
    if not path.exists():
        return set()
    try:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            return {_row_key(row) for row in reader if row.get("status") == "ok"}
    except Exception:
        return set()


def init_incremental_csv(path: str | Path, append: bool = False) -> None:
    """Prepare the live CSV for writing.

    If *append* is False (fresh run), creates the file and writes the header.
    If *append* is True (incremental run), the file already exists with a
    valid header — do nothing.
    """
    if append:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        csv.DictWriter(
            f, fieldnames=_INCREMENTAL_FIELDNAMES, quoting=csv.QUOTE_NONNUMERIC,
        ).writeheader()


def append_result_csv(
    result: MeasurementResult, path: str | Path, lock: threading.Lock
) -> None:
    """Append one result row to the live CSV under *lock* (thread-safe)."""
    row = to_row(result)
    with lock:
        with Path(path).open("a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=_INCREMENTAL_FIELDNAMES,
                quoting=csv.QUOTE_NONNUMERIC,
                extrasaction="ignore",  # metadata_* cols are written in finalize_csv
            )
            writer.writerow(row)


def finalize_csv(path: str | Path) -> None:
    """Deduplicate, then sort the live CSV in-place: failed → invalid → ok.

    When a failed/invalid image is retried in an incremental run its old row
    and new row both exist in the file.  We keep the best result per image_id
    (ok beats invalid beats failed), then sort anomalies to the top.
    Uses QUOTE_ALL since all values are strings after a CSV round-trip.
    """
    path = Path(path)
    if not path.exists():
        return
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames) if reader.fieldnames else list(_INCREMENTAL_FIELDNAMES)
        rows = list(reader)

    # Deduplicate: keep the best status row per unique image_path (or composite
    # fallback of image_id + view + specimen_id when image_path is absent).
    best: dict[str, dict] = {}
    for row in rows:
        key = _row_key(row)
        existing = best.get(key)
        if existing is None:
            best[key] = row
        else:
            if _STATUS_ORDER.get(row.get("status", "ok"), 99) > _STATUS_ORDER.get(existing.get("status", "ok"), 99):
                best[key] = row

    deduped = sorted(best.values(), key=lambda r: _STATUS_ORDER.get(r.get("status", "ok"), 99))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(deduped)
