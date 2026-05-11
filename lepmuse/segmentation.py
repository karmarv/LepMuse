from __future__ import annotations

from pathlib import Path

from .types import Segmenter


def build_segmenter(name: str, weights: str | Path | None = None) -> Segmenter:
    if name == "unet":
        from segmentation.unet.infer import UNetSegmenter

        return UNetSegmenter(weights=weights)
    raise ValueError(f"Unsupported segmenter: {name}")
