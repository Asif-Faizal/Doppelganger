from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ..errors import ModelUnavailable

MANIFEST_PATH = Path(__file__).resolve().parents[2] / "models" / "manifest.json"
_CHUNK = 1024 * 1024


@dataclass(frozen=True)
class ModelFile:
    key: str
    role: str
    name: str
    version: str
    filename: str
    sha256: str
    size: int
    url: str
    license: str


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, ModelFile]:
    raw = json.loads(path.read_text())
    out: dict[str, ModelFile] = {}
    for key, m in raw["models"].items():
        out[key] = ModelFile(
            key=key,
            role=m["role"],
            name=m["name"],
            version=m["version"],
            filename=m["filename"],
            sha256=m["sha256"],
            size=int(m["size"]),
            url=m["url"],
            license=m["license"],
        )
    return out


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def verify_model(models_dir: Path, model: ModelFile) -> Path:
    """Return the path of a present, integrity-checked model file or raise ModelUnavailable.

    Never downloads anything: provisioning is an explicit operator action (`models fetch`).
    """
    path = models_dir / model.filename
    if not path.is_file():
        raise ModelUnavailable(
            f"Model file '{model.filename}' is missing. Provision it with "
            "`python -m app.cli models fetch` (or copy it into the models directory) and restart.",
            code="model_file_missing",
        )
    if path.stat().st_size != model.size or file_sha256(path) != model.sha256:
        raise ModelUnavailable(
            f"Model file '{model.filename}' failed its integrity check (sha256/size mismatch). "
            "Remove it and provision it again from the pinned source.",
            code="model_checksum_mismatch",
        )
    return path


def fetch_model(models_dir: Path, model: ModelFile) -> Path:
    """Explicit operator-run download from the pinned URL; verifies sha256 before installing."""
    models_dir.mkdir(parents=True, exist_ok=True)
    dest = models_dir / model.filename
    with tempfile.NamedTemporaryFile(dir=models_dir, delete=False, suffix=".part") as tmp:
        tmp_path = Path(tmp.name)
        try:
            with urllib.request.urlopen(model.url, timeout=120) as resp:  # noqa: S310 (pinned https URL)
                shutil.copyfileobj(resp, tmp, length=_CHUNK)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    if tmp_path.stat().st_size != model.size or file_sha256(tmp_path) != model.sha256:
        tmp_path.unlink(missing_ok=True)
        raise ModelUnavailable(
            f"Downloaded '{model.filename}' does not match the pinned sha256; discarded.",
            code="model_checksum_mismatch",
        )
    tmp_path.replace(dest)
    return dest
