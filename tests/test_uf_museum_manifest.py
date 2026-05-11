from pathlib import Path

import pandas as pd

from datasets.uf_museum_battus.metadata import build_manifest, parse_image_numbers


def test_parse_image_numbers_handles_known_separators():
    assert parse_image_numbers("1169,1170") == ["1169", "1170"]
    assert parse_image_numbers("1,,2") == ["0001", "0002"]
    assert parse_image_numbers("IMG_9.JPG.IMG_10.JPG") == ["0009", "0010"]


def test_build_manifest_keeps_dorsal_ventral_metadata(tmp_path):
    image = tmp_path / "Battus philenor" / "Kailee Stover" / "20230410" / "IMG_1169.JPG"
    image.parent.mkdir(parents=True)
    image.write_text("x")
    ventral = image.parent / "IMG_1170.JPG"
    ventral.write_text("x")
    df = pd.DataFrame(
        [
            {
                "imag_numbers": "1169,1170",
                "Dorsal": 1169,
                "Ventral": 1170,
                "genus": "Battus",
                "species": "philenor",
                "image_folder": "Kailee Stover\\20230410",
                "voucher": "123",
                "sex": "male",
            }
        ]
    )

    manifest, missing = build_manifest(df, tmp_path)

    assert missing.empty
    assert list(manifest["view"]) == ["dorsal", "ventral"]
    assert set(manifest["specimen_id"]) == {"123"}


def test_build_manifest_prefers_dorsal_ventral_columns(tmp_path):
    for image_id in ["IMG_1169.JPG", "IMG_1170.JPG"]:
        image = tmp_path / "Battus philenor" / "Kailee Stover" / "20230410" / image_id
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_text("x")
    df = pd.DataFrame(
        [
            {
                "imag_numbers": "11,691,170",
                "Dorsal": 1169,
                "Ventral": 1170,
                "genus": "Battus",
                "species": "philenor",
                "image_folder": "Kailee Stover\\20230410",
            }
        ]
    )

    manifest, missing = build_manifest(df, tmp_path)

    assert missing.empty
    assert list(manifest["image_id"]) == ["IMG_1169.JPG", "IMG_1170.JPG"]
