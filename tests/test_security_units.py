from __future__ import annotations

import time
from pathlib import Path

import pytest

from app.config import Settings
from app.errors import NotFound, RateLimited
from app.imaging import sha256_hex
from app.ingest import clean_display_name
from app.security import RateLimiter, UrlSigner
from app.storage import FileStore


def test_rate_limiter_refills_over_time() -> None:
    rl = RateLimiter(60)  # 1 token / second
    for _ in range(60):
        rl.check("k")
    with pytest.raises(RateLimited) as exc:
        rl.check("k")
    assert int(exc.value.headers["Retry-After"]) >= 1
    rl.check("other")  # separate bucket
    time.sleep(1.1)
    rl.check("k")


def test_rate_limiter_zero_disables() -> None:
    rl = RateLimiter(0)
    for _ in range(1000):
        rl.check("k")


def test_url_signer_persists_key_with_restrictive_mode(tmp_path: Path) -> None:
    s = Settings(data_dir=tmp_path, pipeline="fake", auth_disabled=True)
    a = UrlSigner(s)
    exp, sig = a.sign("image", "id1")
    b = UrlSigner(s)  # same persisted key -> signature still valid after restart
    b.verify("image", "id1", exp, sig)
    assert (tmp_path / "url_signing.key").stat().st_mode & 0o777 == 0o600
    with pytest.raises(NotFound):
        b.verify("thumb", "id1", exp, sig)  # kind is part of the signature
    with pytest.raises(NotFound):
        b.verify("image", "id1", exp + 1, sig)


def test_filestore_rejects_path_like_names(tmp_path: Path) -> None:
    store = FileStore(Settings(data_dir=tmp_path, pipeline="fake", auth_disabled=True))
    for bad in (
        "../x.jpg",
        "/etc/passwd",
        "a.jpg",
        "11111111-1111-4111-8111-111111111111/../x.jpg",
        "..\\x",
    ):
        with pytest.raises(NotFound):
            store.original_path(bad)
        with pytest.raises(NotFound):
            store.crop_path(bad)


def test_untrusted_filenames_are_only_labels() -> None:
    assert clean_display_name("../../etc/passwd") == "passwd"
    assert clean_display_name("C:\\Users\\x\\me.jpg") == "me.jpg"
    assert clean_display_name("a\x00b\nc.jpg") == "abc.jpg"
    assert clean_display_name("") is None
    assert len(clean_display_name("x" * 1000) or "") == 255


def test_staged_uploads_are_swept(tmp_path: Path) -> None:
    store = FileStore(Settings(data_dir=tmp_path, pipeline="fake", auth_disabled=True))
    p = store.stage(b"selfie-or-upload")
    assert p.parent == store.settings.staging_dir and p.stat().st_mode & 0o777 == 0o600
    assert store.sweep_staging(older_than_hours=1) == 0
    import os

    os.utime(p, (time.time() - 7200, time.time() - 7200))
    assert store.sweep_staging(older_than_hours=1) == 1 and not p.exists()


def test_search_never_persists_the_selfie(client, admin) -> None:  # type: ignore[no-untyped-def]
    from .conftest import single

    r = client.post(
        "/v1/faces/search",
        files={"file": ("q.jpg", single((10, 200, 10)), "image/jpeg")},
        headers=admin,
    )
    assert r.status_code == 200
    root = Path(client.app.state.container.settings.data_dir)
    files = [p for p in root.rglob("*") if p.is_file()]
    assert not [p for p in files if p.suffix in (".jpg", ".upload", ".png")]
    assert sha256_hex(single((10, 200, 10))) not in " ".join(str(p) for p in files)
