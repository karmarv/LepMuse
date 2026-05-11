from __future__ import annotations

import traceback
import os
import threading
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from skimage.io import imread
from skimage.measure import label, regionprops

from . import landmarks, measurements, preprocessing, scale
from .results import append_result_csv, init_incremental_csv, load_completed_ids

from .config import PipelineConfig
from .types import ImageRecord, MeasurementResult, Segmenter


STAGES = ("binarization", "ruler_detection", "measurements")


class PipelineRunner:
    def __init__(self, segmenter: Segmenter, config: PipelineConfig):
        self.segmenter = segmenter
        self.config = config

    def run(self, records: Iterable[ImageRecord]) -> list[MeasurementResult]:
        records = list(records)
        csv_path = self.config.path_csv

        completed = load_completed_ids(csv_path)
        if completed:
            pending = [r for r in records if r.image_id not in completed]
            skipped = len(records) - len(pending)
            print(
                f"Incremental run: {skipped} already done, {len(pending)} remaining"
                f" — delete {csv_path} for a full re-run"
            )
            records = pending
        if not records:
            print("All images already processed — nothing to do.")
            return []

        csv_lock = threading.Lock()
        init_incremental_csv(csv_path, append=bool(completed))
        if self.config.num_workers <= 1:
            return self._run_sequential(records, csv_path, csv_lock)
        return self._run_parallel(records, self.config.num_workers, csv_path, csv_lock)

    def _run_sequential(
        self, records: list, csv_path: str, csv_lock: threading.Lock
    ) -> list[MeasurementResult]:
        results = []
        for index, record in enumerate(records, start=1):
            print(f"\nImage {index}/{len(records)} : {record.image_id}")
            try:
                result = self.process(record)
            except Exception as exc:
                if not self.config.continue_on_error:
                    raise
                print(f"* Failed: {record.path}: {exc}")
                result = self.failure(record, exc) if self.config.write_failures else None
            if result is not None:
                append_result_csv(result, csv_path, csv_lock)
                results.append(result)
        return results

    def _run_parallel(
        self, records: list, num_workers: int, csv_path: str, csv_lock: threading.Lock
    ) -> list[MeasurementResult]:
        print(f"Running pipeline with {num_workers} parallel workers")
        _print_lock = threading.Lock()
        total = len(records)
        results: list[MeasurementResult | None] = [None] * total

        def _process(index: int, record: ImageRecord):
            with _print_lock:
                print(f"\nImage {index + 1}/{total} : {record.image_id}")
            try:
                result = self.process(record)
            except Exception as exc:
                if not self.config.continue_on_error:
                    raise
                with _print_lock:
                    print(f"* Failed: {record.path}: {exc}")
                result = self.failure(record, exc) if self.config.write_failures else None
            if result is not None:
                append_result_csv(result, csv_path, csv_lock)
                results[index] = result

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process, i, r): i for i, r in enumerate(records)}
            for future in as_completed(futures):
                future.result()  # re-raise any exception not caught above

        return [r for r in results if r is not None]

    def process(self, record: ImageRecord) -> MeasurementResult:
        if self.config.stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}; received {self.config.stage!r}")

        axes = self._create_axes()
        image_rgb = imread(record.path)
        if self.config.auto_rotate:
            image_rgb = preprocessing.auto_rotate(image_rgb, str(record.path))
        segments = self.segmenter.segment(image_rgb, record)
        self._validate_segments(segments.lepidopteran_mask, segments.ruler_mask)

        # Content-based orientation check: ruler must be below the specimen.
        # Some camera bodies write EXIF orientation=1 even when the image is
        # upside-down.  Rotating 180° is cheap and does not alter aspect ratio.
        if self._is_upside_down(segments):
            print("Content-based orientation: ruler right of specimen — rotating 180°")
            image_rgb = np.rot90(image_rgb, 2)
            segments = self.segmenter.segment(image_rgb, record)
            self._validate_segments(segments.lepidopteran_mask, segments.ruler_mask)

        self._plot_segments(segments, axes)

        if self.config.stage == "binarization":
            self._save_plot(record, axes)
            return self._result(record, segmenter=segments.model_name, stage="binarization")

        pixels_per_mm, _ = scale.main(image_rgb, segments.ruler_mask, axes)
        self._validate_scale(pixels_per_mm)
        if self.config.stage == "ruler_detection":
            self._save_plot(record, axes)
            return self._result(record, segmenter=segments.model_name, stage="ruler_detection", pixels_per_mm=pixels_per_mm)

        points = landmarks.main(segments.lepidopteran_mask, axes)
        measurements_px, measurements_mm = measurements.main(points, pixels_per_mm, axes)
        status, error = self._quality_status(measurements_mm, pixels_per_mm)
        self._save_plot(record, axes)
        return self._result(
            record,
            segmenter=segments.model_name,
            stage="measurements",
            pixels_per_mm=pixels_per_mm,
            measurements_px=measurements_px,
            measurements_mm=measurements_mm,
            points=points,
            status=status,
            error=error,
        )

    def failure(self, record: ImageRecord, exc: Exception) -> MeasurementResult:
        return self._result(
            record,
            segmenter=getattr(self.segmenter, "name", "unknown"),
            stage=self.config.stage,
            status="failed",
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=5)}",
        )

    def _result(
        self,
        record: ImageRecord,
        segmenter: str,
        stage: str,
        pixels_per_mm: float | None = None,
        measurements_px: dict | None = None,
        measurements_mm: dict | None = None,
        points: dict | None = None,
        status: str = "ok",
        error: str | None = None,
    ) -> MeasurementResult:
        return MeasurementResult(
            image_id=record.image_id,
            image_path=record.path,
            measurements_mm=measurements_mm or {},
            measurements_px=measurements_px or {},
            points=points or {},
            pixels_per_mm=pixels_per_mm,
            stage=stage,
            segmenter=segmenter,
            view=record.view,
            specimen_id=record.specimen_id,
            metadata=record.metadata,
            status=status,
            error=error,
        )

    @staticmethod
    def _plot_segments(segments, axes) -> None:
        """Populate the two segmentation panels in the detailed plot layout.

        axes[1] — Binarized lepidopteran mask.
        axes[6] — Tags + ruler detection: RGB overlay on the original image
                  with ruler pixels in red and tag pixels in cyan, plus a
                  dashed cyan line at the leftmost tag region edge.
        axes[3] — Image structure: cyan dashed line at the absolute column
                  of the first tag edge (same reference as mothra).
        """
        if axes[1] is not None:
            axes[1].imshow(segments.lepidopteran_mask)
            axes[1].set_title('Binarized lepidopteran')

        tag_mask = segments.tag_mask
        if axes[6] is not None:
            # Build an RGB overlay: light grey background, ruler in red, tags in cyan.
            h, w = segments.ruler_mask.shape
            overlay = np.full((h, w, 3), 220, dtype=np.uint8)
            overlay[segments.ruler_mask] = [220, 60, 60]
            if tag_mask is not None:
                overlay[tag_mask] = [0, 200, 220]
            axes[6].imshow(overlay)
            axes[6].set_title('Tags (cyan) + Ruler (red)')
            # Dashed line at first tag edge
            if tag_mask is not None and tag_mask.any():
                tag_regions = regionprops(label(tag_mask))
                if tag_regions:
                    first_tag_col = min(r.bbox[1] for r in tag_regions)
                    axes[6].axvline(x=first_tag_col, color='c', linestyle='dashed')

        # Mark first tag edge on the image-structure axes (axes[3]).
        if axes[3] is not None and tag_mask is not None and tag_mask.any():
            tag_regions_full = regionprops(label(tag_mask))
            if tag_regions_full:
                first_tag_edge = min(r.bbox[1] for r in tag_regions_full)
                axes[3].axvline(x=first_tag_edge, color='c', linestyle='dashed')

    @staticmethod
    def _is_upside_down(segments) -> bool:
        """Return True when the ruler centroid sits to the right of the specimen.

        In correctly oriented museum images the ruler is always on the left side
        of the frame.  When a camera writes an incorrect EXIF tag the stored
        pixels are rotated 180°, placing the ruler on the right.  A 180°
        rotation (np.rot90 k=2) restores the correct orientation.
        """
        ruler_cols = np.nonzero(segments.ruler_mask)[1]
        lepidop_cols = np.nonzero(segments.lepidopteran_mask)[1]
        if len(ruler_cols) == 0 or len(lepidop_cols) == 0:
            return False
        return float(ruler_cols.mean()) > float(lepidop_cols.mean())

    @staticmethod
    def _validate_segments(lepidopteran_mask: np.ndarray, ruler_mask: np.ndarray) -> None:
        if lepidopteran_mask.shape != ruler_mask.shape:
            raise ValueError(f"Mask shape mismatch: lepidopteran={lepidopteran_mask.shape}, ruler={ruler_mask.shape}")
        if not np.any(lepidopteran_mask):
            raise ValueError("Empty lepidopteran mask")
        if not np.any(ruler_mask):
            raise ValueError("Empty ruler mask")

    def _validate_scale(self, pixels_per_mm: float) -> None:
        if not np.isfinite(pixels_per_mm):
            raise ValueError(f"Non-finite ruler scale: {pixels_per_mm}")
        if pixels_per_mm < self.config.min_pixels_per_mm or pixels_per_mm > self.config.max_pixels_per_mm:
            raise ValueError(f"Ruler scale outside configured range: {pixels_per_mm}")

    def _quality_status(self, measurements_mm: dict, pixels_per_mm: float) -> tuple[str, str | None]:
        errors = []
        left = measurements_mm.get("dist_l")
        right = measurements_mm.get("dist_r")
        shoulder = measurements_mm.get("dist_shoulder")

        for label, value in [("left_wing", left), ("right_wing", right)]:
            if value is None or value < self.config.min_wing_mm or value > self.config.max_wing_mm:
                errors.append(f"{label} outside range: {value}")
        if left is not None and right is not None and abs(left - right) > self.config.max_wing_asymmetry_mm:
            errors.append(f"wing asymmetry too large: {abs(left - right):.2f} mm")
        if shoulder is None or shoulder < self.config.min_shoulder_mm:
            errors.append(f"shoulder width too small: {shoulder}")
        if not errors:
            return "ok", None
        return "invalid", "; ".join(errors)

    def _create_axes(self):
        if not (self.config.plot or self.config.detailed_plot):
            return [None] * 7
        from . import plots

        plot_level = 2 if self.config.detailed_plot else 1
        stage_idx = STAGES.index(self.config.stage)
        return plots.create_layout(stage_idx + 1, plot_level)

    def _save_plot(self, record: ImageRecord, axes) -> None:
        if not (self.config.plot or self.config.detailed_plot):
            return
        fig = next((ax.get_figure() for ax in axes if ax is not None), None)
        if fig is None:
            return
        output_path = Path(self.config.output_folder) / record.image_id
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dpi = int(1.5 * self.config.dpi) if self.config.detailed_plot else self.config.dpi
        fig.savefig(output_path, dpi=dpi)
        fig.clf()  # release axes memory; Figure() objects are not pyplot-managed
