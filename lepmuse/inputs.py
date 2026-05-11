from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from .types import ImageRecord

SUPPORTED_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tiff", ".tif")
SUPPORTED_TEXT_EXT = (".txt", ".text")


def read_records(input_name: str | Path) -> list[ImageRecord]:
    path = Path(input_name)
    if path.is_file() and path.suffix.lower() == ".csv":
        return read_manifest(path)
    if path.is_file() and path.suffix.lower() in SUPPORTED_TEXT_EXT:
        return _records_from_paths(_read_text_paths(path))
    if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXT:
        return _records_from_paths([path])
    if path.is_dir():
        return _records_from_paths(_read_image_dir(path))
    raise FileNotFoundError(f"Input path not found or unsupported: {input_name}")


def read_manifest(path: str | Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    with Path(path).open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            return []
        path_column = _first_existing(reader.fieldnames, ["image_path", "image_paths", "path"])
        for row in reader:
            image_path = Path(row[path_column])
            metadata = {key: value for key, value in row.items() if key not in {path_column, "image_id", "view", "specimen_id"}}
            records.append(
                ImageRecord(
                    image_id=row.get("image_id") or image_path.name,
                    path=image_path,
                    view=row.get("view") or None,
                    specimen_id=row.get("specimen_id") or row.get("voucher") or None,
                    metadata=metadata,
                )
            )
    return records


def _first_existing(fieldnames: Iterable[str], candidates: list[str]) -> str:
    available = set(fieldnames)
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise ValueError(f"Manifest must include one of: {', '.join(candidates)}")


def _read_text_paths(path: Path) -> list[Path]:
    paths = []
    for line in path.read_text().splitlines():
        item = line.strip()
        if item and item != "image_paths":
            paths.append(Path(item))
    return paths


def _read_image_dir(path: Path) -> list[Path]:
    return sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_EXT)


def _records_from_paths(paths: list[Path]) -> list[ImageRecord]:
    return [ImageRecord(image_id=path.name, path=path) for path in sorted(set(paths))]
