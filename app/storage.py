from __future__ import annotations

import os
import re
import time
import uuid
from pathlib import Path

from .config import Settings
from .errors import NotFound

_NAME_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.[a-z0-9]{2,5}$"
)


def new_id() -> str:
    return str(uuid.uuid4())


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


class FileStore:
    """Local file storage. Names are always server-generated UUIDs; client filenames are never
    used as paths, and lookups accept only names matching a strict pattern."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        settings.ensure_dirs()

    @staticmethod
    def _safe(name: str) -> str:
        if not _NAME_RE.match(name):
            raise NotFound("File not found.")
        return name

    def save_original(self, image_id: str, ext: str, data: bytes) -> str:
        name = f"{image_id}{ext}"
        _atomic_write(self.settings.originals_dir / self._safe(name), data)
        return name

    def original_path(self, name: str) -> Path:
        return self.settings.originals_dir / self._safe(name)

    def save_crop(self, face_id: str, data: bytes) -> str:
        name = f"{face_id}.jpg"
        _atomic_write(self.settings.crops_dir / self._safe(name), data)
        return name

    def crop_path(self, name: str) -> Path:
        return self.settings.crops_dir / self._safe(name)

    def delete_original(self, name: str) -> None:
        self.original_path(name).unlink(missing_ok=True)

    def delete_crop(self, name: str) -> None:
        self.crop_path(name).unlink(missing_ok=True)

    def stage(self, data: bytes) -> Path:
        path = self.settings.staging_dir / f"{uuid.uuid4().hex}.upload"
        _atomic_write(path, data)
        return path

    def sweep_staging(self, older_than_hours: int | None = None) -> int:
        hours = (
            older_than_hours
            if older_than_hours is not None
            else self.settings.staging_retention_hours
        )
        cutoff = time.time() - hours * 3600
        removed = 0
        for p in self.settings.staging_dir.glob("*.upload"):
            try:
                if p.stat().st_mtime < cutoff:
                    p.unlink()
                    removed += 1
            except FileNotFoundError:
                pass
        return removed

    def list_originals(self) -> set[str]:
        return {p.name for p in self.settings.originals_dir.iterdir() if _NAME_RE.match(p.name)}

    def list_crops(self) -> set[str]:
        return {p.name for p in self.settings.crops_dir.iterdir() if _NAME_RE.match(p.name)}
