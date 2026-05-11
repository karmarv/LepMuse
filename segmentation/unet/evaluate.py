from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from skimage.io import imread

from lepmuse.types import ImageRecord


@dataclass(frozen=True)
class EvalConfig:
    weights: str = "segmentation/unet/models/battus100_segm_c4_resnet18_b8_e50_s1200x800.pkl"
    image_dir: str = "datasets/battus100/val_images/images"
    mask_dir: str = "datasets/battus100/val_images/labels"
    output_csv: str = "datasets/battus100/results/segmentation_eval.csv"


def load_config(path: str | None) -> EvalConfig:
    if path is None:
        return EvalConfig()
    return EvalConfig(**json.loads(Path(path).read_text()))


def foreground_accuracy(mask_ref: np.ndarray, pred_bin: np.ndarray, background_idx: int = 0) -> float:
    mask = mask_ref != background_idx
    if not np.any(mask):
        return 0.0
    return float((pred_bin[mask] >= mask_ref[mask]).mean())


def evaluate(config: EvalConfig) -> list[dict[str, object]]:
    from .infer import UNetSegmenter

    segmenter = UNetSegmenter(weights=config.weights)
    image_dir = Path(config.image_dir)
    mask_dir = Path(config.mask_dir)
    rows = []
    for image_path in sorted(image_dir.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            continue
        mask_path = mask_dir / f"{image_path.stem}.png"
        image_rgb = imread(image_path)
        try:
            segments = segmenter.segment(image_rgb, ImageRecord(image_id=image_path.name, path=image_path))
            pred_bin = segments.lepidopteran_mask + segments.ruler_mask
            if segments.tag_mask is not None:
                pred_bin = pred_bin + segments.tag_mask
            score = foreground_accuracy(imread(mask_path), pred_bin) if mask_path.exists() else None
            status = "ok"
            error = ""
        except Exception as exc:
            score = None
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        rows.append({"image_id": image_path.name, "mask_path": str(mask_path), "foreground_accuracy": score, "status": status, "error": error})
        print(image_path.name, status, score)
    return rows


def write_csv(rows: list[dict[str, object]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image_id", "mask_path", "foreground_accuracy", "status", "error"])
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate UNet segmentation masks.")
    parser.add_argument("--config")
    parser.add_argument("--weights")
    parser.add_argument("--image-dir")
    parser.add_argument("--mask-dir")
    parser.add_argument("--output-csv")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    updates = {key: value for key, value in vars(args).items() if key != "config" and value is not None}
    if updates:
        config = replace(config, **updates)
    write_csv(evaluate(config), config.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
