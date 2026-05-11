from pathlib import Path

from lepmuse.results import to_row
from lepmuse.types import MeasurementResult


def test_result_row_keeps_mothra_measurement_columns():
    result = MeasurementResult(
        image_id="IMG_0001.JPG",
        image_path=Path("IMG_0001.JPG"),
        measurements_mm={"dist_l": 1.2, "dist_r": 1.3, "dist_l_center": 2.0, "dist_r_center": 2.1, "dist_span": 4.0, "dist_shoulder": 0.5},
        pixels_per_mm=10.0,
        segmenter="unet",
    )

    row = to_row(result)

    assert row["left_wing (mm)"] == 1.2
    assert row["right_wing (mm)"] == 1.3
    assert row["pixels_per_mm"] == 10.0
    assert row["status"] == "ok"
