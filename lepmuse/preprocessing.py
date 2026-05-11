import numpy as np
from PIL import Image as PILImage, ImageOps
from skimage.util import img_as_ubyte

# EXIF orientation tag ID (TIFF/JPEG standard)
_EXIF_ORIENTATION_TAG = 274

# Orientation values that require no transformation
_ORIENTATION_NORMAL = {1, None}


def auto_rotate(image_rgb: np.ndarray, image_path: str) -> np.ndarray:
    """Apply EXIF orientation correction using PIL's exif_transpose.

    Handles all 8 EXIF orientation values (rotations and mirrors) reliably.
    Falls back gracefully when EXIF data is absent or unreadable.

    Parameters
    ----------
    image_rgb : 3D array
        RGB image already loaded from disk.
    image_path : str
        Path of the same image file, used to read EXIF metadata via PIL.

    Returns
    -------
    image_rgb : 3D array
        RGB image corrected for EXIF orientation.
    """
    try:
        pil_img = PILImage.open(image_path)
        orientation = _read_exif_orientation(pil_img)

        if orientation in _ORIENTATION_NORMAL:
            print("EXIF orientation: upright — no rotation needed")
            return img_as_ubyte(image_rgb)

        corrected = ImageOps.exif_transpose(pil_img)
        print(
            f"EXIF orientation tag {orientation}: "
            f"corrected {pil_img.size[0]}×{pil_img.size[1]} → "
            f"{corrected.size[0]}×{corrected.size[1]}"
        )
        return img_as_ubyte(np.asarray(corrected.convert("RGB")))

    except Exception as exc:
        print(f"EXIF orientation: could not apply ({type(exc).__name__}: {exc}) — skipping")
        return img_as_ubyte(image_rgb)


def _read_exif_orientation(pil_img: PILImage.Image) -> int | None:
    """Return the raw EXIF orientation tag value, or None if absent."""
    try:
        exif = pil_img._getexif()
        if exif:
            return exif.get(_EXIF_ORIENTATION_TAG)
    except Exception:
        pass
    return None
