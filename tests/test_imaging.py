from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from app.config import Settings
from app.errors import CorruptImage, FileTooLarge, InvalidImage, UnsupportedImageFormat
from app.imaging import decode_image, inspect_image, make_face_crop

from .conftest import RED, face_image, single


def test_accepts_supported_formats(settings: Settings) -> None:
    for fmt in ("JPEG", "PNG", "WEBP", "BMP", "TIFF"):
        info = inspect_image(single(RED, fmt=fmt), settings)
        assert info.format == fmt
        assert len(info.sha256) == 64


def test_rejects_non_image_with_image_extension(settings: Settings) -> None:
    with pytest.raises(UnsupportedImageFormat):
        inspect_image(b"<?php echo 1; ?>" * 100, settings)


def test_rejects_disallowed_but_valid_format(settings: Settings) -> None:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buf, format="GIF")
    with pytest.raises(UnsupportedImageFormat) as exc:
        inspect_image(buf.getvalue(), settings)
    assert "GIF" in exc.value.message


def test_truncated_image_is_corrupt(settings: Settings) -> None:
    data = single(RED)
    with pytest.raises(CorruptImage):
        decode_image(data[: len(data) // 2], settings)


def test_empty_file(settings: Settings) -> None:
    with pytest.raises(InvalidImage) as exc:
        inspect_image(b"", settings)
    assert exc.value.code == "empty_file"


def test_byte_limit(settings: Settings) -> None:
    small = settings.model_copy(update={"max_upload_bytes": 2048})
    with pytest.raises(FileTooLarge):
        inspect_image(single(RED, size=(600, 600), textured=True), small)


def test_pixel_and_dimension_limits(settings: Settings) -> None:
    tight = settings.model_copy(update={"max_pixels": 10_000, "max_dimension": 5000})
    with pytest.raises(InvalidImage) as exc:
        inspect_image(single(RED), tight)  # 320x320 = 102400 px
    assert exc.value.code == "image_too_large"
    tiny = settings.model_copy(update={"min_dimension": 400})
    with pytest.raises(InvalidImage) as exc2:
        inspect_image(single(RED), tiny)
    assert exc2.value.code == "image_too_small"


def test_exif_orientation_is_applied(settings: Settings) -> None:
    img = Image.new("RGB", (200, 100), (255, 0, 0))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW when displayed
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    decoded = decode_image(buf.getvalue(), settings)
    assert (decoded.info.width, decoded.info.height) == (100, 200)
    assert decoded.bgr.shape == (200, 100, 3)


def test_alpha_is_flattened_and_output_is_bgr(settings: Settings) -> None:
    img = Image.new("RGBA", (64, 64), (255, 0, 0, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    decoded = decode_image(buf.getvalue(), settings)
    assert decoded.bgr.dtype == np.uint8
    assert tuple(decoded.bgr[10, 10]) == (0, 0, 255)  # BGR


def test_face_crop_is_small_jpeg(settings: Settings) -> None:
    decoded = decode_image(face_image([(RED, 100, 100, 100)]), settings)
    crop = make_face_crop(decoded.bgr, (100, 100, 100, 100), 96)
    out = Image.open(io.BytesIO(crop))
    assert out.format == "JPEG"
    assert max(out.size) <= 96
