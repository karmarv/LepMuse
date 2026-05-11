from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    path: Path
    view: str | None = None
    specimen_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SegmentationResult:
    lepidopteran_mask: np.ndarray
    ruler_mask: np.ndarray
    tag_mask: np.ndarray | None = None
    model_name: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MeasurementResult:
    image_id: str
    image_path: Path
    measurements_mm: Mapping[str, float] = field(default_factory=dict)
    measurements_px: Mapping[str, float] = field(default_factory=dict)
    points: Mapping[str, Any] = field(default_factory=dict)
    pixels_per_mm: float | None = None
    stage: str = "measurements"
    segmenter: str = "unknown"
    view: str | None = None
    specimen_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    status: str = "ok"
    error: str | None = None


class Segmenter(Protocol):
    name: str

    def segment(self, image_rgb: np.ndarray, record: ImageRecord) -> SegmentationResult:
        ...
