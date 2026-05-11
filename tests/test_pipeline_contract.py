from pathlib import Path

import numpy as np

from lepmuse.config import PipelineConfig
from lepmuse.pipeline import PipelineRunner
from lepmuse.types import ImageRecord, SegmentationResult


class FakeSegmenter:
    name = "fake"

    def segment(self, image_rgb, record):
        mask = np.ones(image_rgb.shape[:2], dtype=bool)
        return SegmentationResult(lepidopteran_mask=mask, ruler_mask=mask, model_name=self.name)


def test_pipeline_marks_implausible_measurements_invalid(monkeypatch, tmp_path):
    image_path = tmp_path / "IMG_0001.JPG"
    image_path.write_text("not used")
    monkeypatch.setattr("lepmuse.pipeline.imread", lambda path: np.ones((10, 10, 3), dtype=np.uint8))
    monkeypatch.setattr("lepmuse.pipeline.scale.main", lambda image, mask, axes=None: (10.0, 0))
    monkeypatch.setattr("lepmuse.pipeline.landmarks.main", lambda mask, axes=None: {"ok": True})
    monkeypatch.setattr(
        "lepmuse.pipeline.measurements.main",
        lambda points, scale, axes=None: (
            {"dist_l": 1.0, "dist_r": 1.0, "dist_shoulder": 0.1},
            {"dist_l": 0.1, "dist_r": 0.1, "dist_shoulder": 0.01},
        ),
    )

    result = PipelineRunner(FakeSegmenter(), PipelineConfig()).process(ImageRecord("IMG_0001.JPG", image_path))

    assert result.status == "invalid"
    assert "left_wing outside range" in result.error


def test_pipeline_returns_failure_rows_when_configured(monkeypatch, tmp_path):
    image_path = tmp_path / "missing.JPG"
    monkeypatch.setattr("lepmuse.pipeline.imread", lambda path: (_ for _ in ()).throw(FileNotFoundError("missing")))

    results = PipelineRunner(FakeSegmenter(), PipelineConfig()).run([ImageRecord("missing.JPG", image_path)])

    assert len(results) == 1
    assert results[0].status == "failed"
    assert "FileNotFoundError" in results[0].error
