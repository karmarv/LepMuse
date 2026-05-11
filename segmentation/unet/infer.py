from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import scipy as sp
from fastai.vision.learner import load_learner
from skimage.measure import label, regionprops
from skimage.transform import rescale
from skimage.util import img_as_bool

from lepmuse.types import ImageRecord, SegmentationResult


DEFAULT_WEIGHTS = "segmentation/unet/models/battus100_segm_c4_resnet18_b8_e50_s1200x800.pkl"
SPECIMEN_OVERLAP_TOLERANCE_PX = 50


class UNetSegmenter:
    name = "unet"

    def __init__(self, weights: str | Path | None = None):
        import torch
        self.weights = Path(weights or DEFAULT_WEIGHTS)
        if not self.weights.exists():
            raise FileNotFoundError(f"UNet weights not found: {self.weights}")
        cuda_available = torch.cuda.is_available()
        device = "cuda" if cuda_available else "cpu"
        print(f"Loading UNet weights: {self.weights}  (device={device})")
        self.learner = load_learner(fname=self.weights, cpu=not cuda_available)

    def segment(self, image_rgb: np.ndarray, record: ImageRecord) -> SegmentationResult:
        print("Processing UNet segmentation...")
        _, _, classes = self.learner.predict(image_rgb)
        tags_bin, ruler_bin, lepidop_bin = masks_from_prediction(image_rgb, classes)
        lepidop_bin = largest_region(lepidop_bin)
        tags_bin, ruler_bin = remove_specimen_overlap(tags_bin, ruler_bin, lepidop_bin)
        return SegmentationResult(
            lepidopteran_mask=lepidop_bin,
            ruler_mask=ruler_bin,
            tag_mask=tags_bin,
            model_name=self.name,
            metadata={"weights": str(self.weights), "image_id": record.image_id},
        )


def masks_from_prediction(image_rgb: np.ndarray, classes: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction = np.asarray(classes)
    if prediction.shape[0] < 4:
        raise ValueError(f"Expected at least 4 segmentation classes, received shape {prediction.shape}")

    _, lepidop_bin, tags_bin, ruler_bin = prediction[:4]
    tags_bin = _fill_bool_mask(image_rgb, tags_bin)
    ruler_bin = _fill_bool_mask(image_rgb, ruler_bin)
    lepidop_bin = _fill_bool_mask(image_rgb, lepidop_bin)
    return tags_bin, ruler_bin, lepidop_bin


def _fill_bool_mask(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return sp.ndimage.binary_fill_holes(img_as_bool(_rescale_image(image_rgb, mask)))


def _rescale_image(image_refer: np.ndarray, image_to_rescale: np.ndarray) -> np.ndarray:
    scale_ratio = np.asarray(image_refer.shape[:2]) / np.asarray(image_to_rescale.shape)
    return rescale(image=image_to_rescale, scale=scale_ratio)


def largest_region(image_bin: np.ndarray) -> np.ndarray:
    props = regionprops(label(image_bin))
    if not props:
        raise ValueError("No lepidopteran region found in segmentation mask")
    largest = max(props, key=lambda region: region.area)
    output = np.copy(image_bin)
    output[label(output) != largest.label] = 0
    return img_as_bool(output)


def remove_specimen_overlap(
    tags_bin: np.ndarray,
    ruler_bin: np.ndarray,
    lepidop_bin: np.ndarray,
    tolerance: int = SPECIMEN_OVERLAP_TOLERANCE_PX,
) -> tuple[np.ndarray, np.ndarray]:
    if not lepidop_bin.any():
        return tags_bin, ruler_bin
    struct = sp.ndimage.generate_binary_structure(2, 1)
    dilated = sp.ndimage.binary_dilation(lepidop_bin, structure=struct, iterations=tolerance)
    return tags_bin & ~dilated, ruler_bin & ~dilated
