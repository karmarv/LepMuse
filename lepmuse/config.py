from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PipelineConfig:
    input: str = "input_images"
    output_folder: str = "outputs"
    path_csv: str = "outputs/results.csv"
    stage: str = "measurements"
    segmenter: str = "unet"
    weights: str = "segmentation/unet/models/battus100_segm_c4_resnet18_b8_e50_s1200x800.pkl"
    plot: bool = False
    detailed_plot: bool = False
    auto_rotate: bool = False
    dpi: int = 300
    cache: bool = False
    continue_on_error: bool = True
    write_failures: bool = True
    min_pixels_per_mm: float = 1.0
    max_pixels_per_mm: float = 500.0
    min_wing_mm: float = 5.0
    max_wing_mm: float = 120.0
    min_shoulder_mm: float = 1.0
    max_wing_asymmetry_mm: float = 30.0
    eval_config: str | None = None
    num_workers: int = 1


def load_config(path: str | Path | None) -> PipelineConfig:
    if path is None:
        return PipelineConfig()
    data = json.loads(Path(path).read_text())
    return PipelineConfig(**data)


def merge_config(config: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    updates: dict[str, Any] = {}
    for key in config.__dataclass_fields__:
        value = getattr(args, key, None)
        if value is not None:
            updates[key] = value
    return replace(config, **updates)
