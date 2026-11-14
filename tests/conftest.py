from __future__ import annotations

import io
import os
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.container import Container, build_container
from app.main import create_app

ADMIN = "admin-token-for-tests-0123456789"
SEARCH = "search-token-for-tests-0123456789"

RED = (30, 30, 220)  # BGR
GREEN = (40, 200, 40)
BLUE = (220, 60, 40)
ORANGE = (30, 140, 240)


def face_image(
    faces: list[tuple[tuple[int, int, int], int, int, int]] | None = None,
    size: tuple[int, int] = (320, 320),
    fmt: str = "JPEG",
    textured: bool = True,
) -> bytes:
    """Synthetic 'photo': grey background with textured colour squares the fake detector finds.

    faces: list of (bgr_colour, x, y, side).
    """
    img = np.full((size[1], size[0], 3), 128, np.uint8)
    for colour, x, y, side in faces or []:
        patch = np.empty((side, side, 3), np.uint8)
        patch[:] = colour
        if textured:
            yy, xx = np.mgrid[0:side, 0:side]
            checker = ((yy // 2 + xx // 2) % 2).astype(bool)
            patch[checker] = np.clip(np.array(colour, np.int16) + 18, 0, 255).astype(np.uint8)
            patch[~checker] = np.clip(np.array(colour, np.int16) - 18, 0, 255).astype(np.uint8)
        y1, x1 = min(y + side, size[1]), min(x + side, size[0])
        img[y:y1, x:x1] = patch[: y1 - y, : x1 - x]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format=fmt, **({"quality": 95} if fmt == "JPEG" else {}))
    return buf.getvalue()


def single(
    colour: tuple[int, int, int], side: int = 100, x: int = 100, y: int = 100, **kw: object
) -> bytes:
    return face_image([(colour, x, y, side)], **kw)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's DOPPEL_* variables must never leak into the tests."""
    for key in [k for k in os.environ if k.startswith("DOPPEL_")]:
        monkeypatch.delenv(key)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        pipeline="fake",
        admin_token=ADMIN,
        search_token=SEARCH,
        rate_limit_per_minute=0,
        search_rate_limit_per_minute=0,
        index_sync_interval_s=0.0,
        inference_instances=2,
        worker_poll_s=0.05,
        import_root=tmp_path / "imports",
        min_face_px=24,
    )


@pytest.fixture
def container(settings: Settings) -> Iterator[Container]:
    c = build_container(settings)
    yield c
    c.close()


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    app = create_app(settings)
    with TestClient(app) as tc:
        yield tc


@pytest.fixture
def admin() -> dict[str, str]:
    return {"Authorization": f"Bearer {ADMIN}"}


@pytest.fixture
def searcher() -> dict[str, str]:
    return {"Authorization": f"Bearer {SEARCH}"}
