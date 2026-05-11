from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ManifestConfig:
    metadata_path: str = "datasets/uf_museum_battus/metadata/231017_Battus_philenor_polydamas_FLMNH.xlsx"
    images_root: str = "/home/rahul/workspace/data/UF_museum_data_2023"
    output_manifest_csv: str = "datasets/uf_museum_battus/metadata/eval_image_manifest.csv"
    output_paths_csv: str = "datasets/uf_museum_battus/metadata/eval_image_paths.csv"
    output_missing_csv: str = "datasets/uf_museum_battus/metadata/eval_image_missing.csv"


def load_config(path: str | None) -> ManifestConfig:
    if path is None:
        return ManifestConfig()
    return ManifestConfig(**json.loads(Path(path).read_text()))


def read_metadata(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=0)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported metadata file type: {path}")


def build_manifest(metadata_df: pd.DataFrame, images_root: str | Path = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    missing = []
    images_root = Path(images_root)
    for row_index, row in metadata_df.iterrows():
        image_numbers = image_numbers_from_row(row)
        if len(image_numbers) < 2:
            missing.append({"metadata_row": row_index, "reason": "missing dorsal/ventral image numbers"})
            continue

        species_name = f"{safe_value(row.get('genus'))} {safe_value(row.get('species'))}".strip()
        folder = safe_value(row.get("image_folder")).replace("\\", "/")
        specimen_id = safe_value(row.get("voucher"))
        common = {
            "metadata_row": row_index,
            "specimen_id": specimen_id,
            "voucher": specimen_id,
            "family": safe_value(row.get("family")),
            "genus": safe_value(row.get("genus")),
            "species": safe_value(row.get("species")),
            "genus_species": safe_value(row.get("genus_species")) or species_name,
            "sex": safe_value(row.get("sex")),
            "country": safe_value(row.get("country")),
            "image_quality": safe_value(row.get("Image_quality")),
            "image_quality_reason": safe_value(row.get("Reason why")),
        }
        for view, image_number in zip(("dorsal", "ventral"), image_numbers[:2]):
            image_id = f"IMG_{image_number}.JPG"
            image_path = images_root / species_name / folder / image_id
            entry = {"image_id": image_id, "image_path": str(image_path), "view": view, **common}
            if image_path.exists():
                rows.append(entry)
            else:
                missing.append({**entry, "reason": "file not found"})
    return pd.DataFrame(rows), pd.DataFrame(missing)


def write_manifest(config: ManifestConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_df, missing_df = build_manifest(read_metadata(config.metadata_path), config.images_root)

    _csv_kwargs = {"index": False, "quoting": csv.QUOTE_ALL, "quotechar": '"'}

    manifest_path = Path(config.output_manifest_csv)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(manifest_path, **_csv_kwargs)

    paths_path = Path(config.output_paths_csv)
    paths_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df[["image_path"]].to_csv(paths_path, **_csv_kwargs)

    missing_path = Path(config.output_missing_csv)
    missing_path.parent.mkdir(parents=True, exist_ok=True)
    missing_df.to_csv(missing_path, **_csv_kwargs)

    return manifest_df, missing_df


def parse_image_numbers(value) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    matches = re.findall(r"(?:IMG_)?(\d+)(?:\.JPG)?", text, flags=re.IGNORECASE)
    return [normalize_image_number(match) for match in matches]


def image_numbers_from_row(row) -> list[str]:
    dorsal = row.get("Dorsal")
    ventral = row.get("Ventral")
    if not pd.isna(dorsal) and not pd.isna(ventral):
        try:
            return [normalize_image_number(str(int(dorsal))), normalize_image_number(str(int(ventral)))]
        except (TypeError, ValueError):
            return [normalize_image_number(str(dorsal)), normalize_image_number(str(ventral))]
    return parse_image_numbers(row.get("imag_numbers"))


def normalize_image_number(value: str) -> str:
    value = value.strip()
    if value.startswith("IMG_"):
        value = value[4:]
    if value.upper().endswith(".JPG"):
        value = value[:-4]
    return value.zfill(4)


def safe_value(value) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create UF Museum Battus image manifests from metadata.")
    parser.add_argument("--config", help="JSON config file. CLI arguments override config values.")
    parser.add_argument("--metadata-path", help="Path to Excel or CSV metadata file.")
    parser.add_argument("--images-root", help="Root directory where specimen images are stored.")
    parser.add_argument("--output-manifest-csv", help="Output path for the image manifest CSV.")
    parser.add_argument("--output-paths-csv", help="Output path for the image paths CSV file.")
    parser.add_argument("--output-missing-csv", help="Output path for the missing images CSV.")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    updates = {key: value for key, value in vars(args).items()
               if key != "config" and value is not None}
    if updates:
        config = replace(config, **updates)
    manifest_df, missing_df = write_manifest(config)
    print(f"Wrote {len(manifest_df)} rows to {config.output_manifest_csv}")
    print(f"Wrote {len(missing_df)} rows to {config.output_missing_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
