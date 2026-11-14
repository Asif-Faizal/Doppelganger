from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import Settings
from .errors import CorruptImage, FileTooLarge, InvalidImage, UnsupportedImageFormat

FORMAT_EXTENSIONS = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "BMP": ".bmp",
    "TIFF": ".tiff",
}

FORMAT_MEDIA_TYPES = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "BMP": "image/bmp",
    "TIFF": "image/tiff",
}


@dataclass(frozen=True)
class ImageInfo:
    format: str
    width: int
    height: int
    size_bytes: int
    sha256: str


@dataclass
class DecodedImage:
    bgr: np.ndarray
    info: ImageInfo


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inspect_image(data: bytes, settings: Settings) -> ImageInfo:
    """Validate by decoded content (not filename or declared type), without a full decode."""
    if not data:
        raise InvalidImage("The uploaded file is empty.", code="empty_file")
    if len(data) > settings.max_upload_bytes:
        raise FileTooLarge(
            f"File exceeds the {settings.max_upload_bytes} byte limit.",
            details={"max_bytes": settings.max_upload_bytes},
        )
    try:
        with Image.open(io.BytesIO(data)) as img:
            fmt = (img.format or "").upper()
            width, height = img.size
    except UnidentifiedImageError as exc:
        raise UnsupportedImageFormat(
            "The file is not a recognised image.",
            details={"allowed_formats": sorted(settings.formats)},
        ) from exc
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise CorruptImage("The image file is corrupt or truncated.") from exc

    if fmt not in settings.formats:
        raise UnsupportedImageFormat(
            f"Image format '{fmt or 'unknown'}' is not allowed.",
            details={"allowed_formats": sorted(settings.formats)},
        )
    if min(width, height) < settings.min_dimension:
        raise InvalidImage(
            "Image is too small.",
            code="image_too_small",
            details={"min_dimension": settings.min_dimension},
        )
    if max(width, height) > settings.max_dimension or width * height > settings.max_pixels:
        raise InvalidImage(
            "Image dimensions exceed the configured limits.",
            code="image_too_large",
            details={
                "max_dimension": settings.max_dimension,
                "max_pixels": settings.max_pixels,
            },
        )
    return ImageInfo(
        format=fmt, width=width, height=height, size_bytes=len(data), sha256=sha256_hex(data)
    )


def decode_image(data: bytes, settings: Settings) -> DecodedImage:
    """Safe decode: limits checked first, EXIF orientation applied, converted to 8-bit BGR."""
    info = inspect_image(data, settings)
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.seek(0)
            img.load()
            frame: Image.Image = ImageOps.exif_transpose(img)
            if frame.mode in ("RGBA", "LA") or (frame.mode == "P" and "transparency" in frame.info):
                rgba = frame.convert("RGBA")
                background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                frame = Image.alpha_composite(background, rgba)
            rgb = np.asarray(frame.convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError, SyntaxError, EOFError, Image.DecompressionBombError) as exc:
        raise CorruptImage("The image file is corrupt or truncated.") from exc
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    height, width = bgr.shape[:2]
    return DecodedImage(
        bgr=bgr,
        info=ImageInfo(
            format=info.format,
            width=width,
            height=height,
            size_bytes=info.size_bytes,
            sha256=info.sha256,
        ),
    )


def encode_jpeg(bgr: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise CorruptImage("Could not encode image.")
    return bytes(buf.tobytes())


def make_face_crop(
    bgr: np.ndarray, bbox: tuple[float, float, float, float], size: int, margin: float = 0.35
) -> bytes:
    """Square, margin-padded JPEG thumbnail of a face (kept so originals need not be re-decoded)."""
    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h / 2
    half = max(w, h) * (1 + margin) / 2
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(bgr.shape[1], cx + half)), int(min(bgr.shape[0], cy + half))
    if x1 <= x0 or y1 <= y0:
        x0, y0, x1, y1 = 0, 0, bgr.shape[1], bgr.shape[0]
    crop = bgr[y0:y1, x0:x1]
    scale = size / max(crop.shape[:2])
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    crop = cv2.resize(
        crop,
        (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))),
        interpolation=interp,
    )
    return encode_jpeg(crop, 82)
