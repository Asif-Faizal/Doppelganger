from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

from .conftest import BLUE, GREEN, ORANGE, RED, face_image, single

Headers = dict[str, str]


def upload(client: TestClient, admin: Headers, items: list[tuple[str, bytes]]) -> str:
    files = [("files", (name, data, "image/jpeg")) for name, data in items]
    r = client.post("/v1/gallery/import", files=files, headers=admin)
    assert r.status_code == 202, r.text
    return str(r.json()["job_id"])


def drain(client: TestClient) -> None:
    client.app.state.container.worker.run_until_empty()  # type: ignore[attr-defined]


def seed(client: TestClient, admin: Headers) -> dict[str, str]:
    """Gallery: A=RED, B=(GREEN+BLUE faces), C=ORANGE, D=near-RED. Returns names -> job."""
    job = upload(
        client,
        admin,
        [
            ("a.jpg", single(RED)),
            ("b.jpg", face_image([(GREEN, 10, 10, 90), (BLUE, 150, 150, 90)])),
            ("c.jpg", single(ORANGE)),
            ("d.jpg", single((40, 40, 205))),
        ],
    )
    drain(client)
    return {"job": job}


def search(client: TestClient, headers: Headers, data: bytes, **form: Any) -> Any:
    return client.post(
        "/v1/faces/search",
        files={"file": ("q.jpg", data, "image/jpeg")},
        data={k: str(v) for k, v in form.items()},
        headers=headers,
    )


# --- health / readiness ---------------------------------------------------------------------


def test_health_and_ready(client: TestClient) -> None:
    h = client.get("/health").json()
    assert h["status"] == "healthy" and h["face_service"] == "ready"
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["checks"]["database"] == "ok" and body["checks"]["schema"] == "ok"
    assert body["checks"]["models"]["status"] == "ok"
    assert body["checks"]["index"]["status"] == "ok"
    assert client.get("/").json()["message"].startswith("Face Recognition API")


def test_not_ready_and_clear_errors_when_model_files_are_missing(tmp_path: Path) -> None:
    s = Settings(
        data_dir=tmp_path / "d",
        models_dir=tmp_path / "empty-models",
        pipeline="sface",
        require_models_at_startup=False,
        admin_token="x" * 20,
    )
    with TestClient(create_app(s)) as c:
        r = c.get("/ready")
        assert r.status_code == 503
        assert r.json()["checks"]["models"] == {
            "status": "unavailable",
            "reason": "model_file_missing",
        }
        assert c.get("/health").json()["face_service"] == "unavailable"  # liveness still answers
        resp = c.post(
            "/v1/faces/search",
            files={"file": ("q.jpg", single(RED), "image/jpeg")},
            headers={"Authorization": "Bearer " + "x" * 20},
        )
        assert resp.status_code == 503
        assert resp.json()["error"]["code"] == "model_file_missing"
        assert "tmp" not in resp.text  # no paths


def test_startup_fails_fast_by_default_when_model_missing(tmp_path: Path) -> None:
    from app.errors import ModelUnavailable

    s = Settings(
        data_dir=tmp_path / "d",
        models_dir=tmp_path / "nope",
        pipeline="sface",
        admin_token="x" * 20,
    )
    with pytest.raises(ModelUnavailable), TestClient(create_app(s)):
        pass


def test_startup_requires_a_token_unless_auth_disabled(tmp_path: Path) -> None:
    s = Settings(data_dir=tmp_path / "d", pipeline="fake")
    with pytest.raises(RuntimeError, match="DOPPEL_ADMIN_TOKEN"), TestClient(create_app(s)):
        pass


# --- import + jobs --------------------------------------------------------------------------


def test_import_creates_job_with_progress_and_counts(client: TestClient, admin: Headers) -> None:
    job_id = upload(client, admin, [("a.jpg", single(RED)), ("broken.jpg", b"nope")])
    j = client.get(f"/v1/jobs/{job_id}", headers=admin).json()
    assert j["total"] == 2
    drain(client)
    j = client.get(f"/v1/jobs/{job_id}", headers=admin).json()
    assert j["status"] == "completed_with_errors"
    assert j["counts"]["succeeded"] == 1 and j["counts"]["failed"] == 1
    assert j["progress"] == 1.0 and j["faces_indexed"] == 1
    assert j["errors"][0]["code"] == "unsupported_image_format"
    assert client.get("/v1/jobs/not-a-uuid", headers=admin).status_code == 404
    assert (
        client.get("/v1/jobs/00000000-0000-4000-8000-000000000000", headers=admin).status_code
        == 404
    )


def test_import_dedupes_at_intake(client: TestClient, admin: Headers) -> None:
    upload(client, admin, [("a.jpg", single(RED))])
    drain(client)
    job_id = upload(client, admin, [("again.jpg", single(RED))])
    j = client.get(f"/v1/jobs/{job_id}", headers=admin).json()
    assert j["counts"]["duplicate"] == 1 and j["status"] == "completed"
    assert client.get("/stats", headers=admin).json()["total_faces_stored"] == 1


def test_import_requires_exactly_one_source(client: TestClient, admin: Headers) -> None:
    assert client.post("/v1/gallery/import", headers=admin).status_code == 400
    r = client.post(
        "/v1/gallery/import",
        files=[("files", ("a.jpg", single(RED), "image/jpeg"))],
        data={"directory": "x"},
        headers=admin,
    )
    assert r.status_code == 400


def test_directory_import_is_confined_to_import_root(
    client: TestClient, admin: Headers, settings: Settings
) -> None:
    root = settings.import_root
    assert root is not None
    (root / "album").mkdir(parents=True)
    (root / "album" / "a.jpg").write_bytes(single(RED))
    (root.parent / "secret").mkdir()
    ok = client.post("/v1/gallery/import", data={"directory": "album"}, headers=admin)
    assert ok.status_code == 202 and ok.json()["total"] == 1
    for bad in ("../secret", "/etc", "album/../../secret", "nope"):
        r = client.post("/v1/gallery/import", data={"directory": bad}, headers=admin)
        assert r.status_code == 400, bad
        assert r.json()["error"]["code"] == "invalid_directory"
        assert "secret" not in r.text.replace('"secret"', "")  # path is not echoed back


def test_directory_import_disabled_without_root(tmp_path: Path) -> None:
    s = Settings(data_dir=tmp_path / "d", pipeline="fake", admin_token="a" * 20, min_face_px=24)
    with TestClient(create_app(s)) as c:
        r = c.post(
            "/v1/gallery/import",
            data={"directory": "x"},
            headers={"Authorization": "Bearer " + "a" * 20},
        )
        assert r.status_code == 503 and r.json()["error"]["code"] == "directory_import_disabled"


# --- search ----------------------------------------------------------------------------------


def test_lookalike_ranks_all_and_never_forces_identity(client: TestClient, admin: Headers) -> None:
    seed(client, admin)
    r = search(client, admin, single(RED, side=90), dedupe_by_image="false")
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "LOOKALIKE" and body["status"] == "ok"
    assert "not identity" in body["notice"]
    m = body["matches"]
    assert [x["rank"] for x in m] == list(range(1, len(m) + 1))
    sims = [x["cosine_similarity"] for x in m]
    assert sims == sorted(sims, reverse=True)
    assert len(m) == 5  # every face is ranked in lookalike mode, including different "people"
    assert m[0]["cosine_similarity"] > 0.95
    assert all(x["same_person_candidate"] is None for x in m)
    assert all(x["calibrated_confidence"] is None for x in m)
    assert body["calibration"] is None
    top = m[0]
    assert set(top) >= {
        "image_id",
        "face_id",
        "rank",
        "cosine_similarity",
        "cosine_distance",
        "ranking_score",
        "bbox",
        "image_url",
        "thumbnail_url",
        "mode",
        "model_id",
    }
    assert top["cosine_distance"] == pytest.approx(1 - top["cosine_similarity"], abs=1e-5)
    assert "/tmp" not in r.text and "storage" not in r.text and "originals" not in r.text


def test_same_person_applies_threshold_and_can_return_nothing(
    client: TestClient, admin: Headers
) -> None:
    seed(client, admin)
    r = search(client, admin, single(RED, side=90), mode="SAME_PERSON", min_similarity=0.9)
    body = r.json()
    assert body["threshold"] == 0.9 and body["threshold_source"] == "request"
    assert 1 <= len(body["matches"]) <= 2
    assert all(x["same_person_candidate"] is True for x in body["matches"])
    assert all(x["cosine_similarity"] >= 0.9 for x in body["matches"])
    none = search(client, admin, single((5, 5, 5)), mode="same_person", min_similarity=0.999).json()
    assert none["status"] == "no_reliable_match" and none["matches"] == []
    assert "rather than forcing" in none["message"]
    default = search(client, admin, single(RED, side=90), mode="SAME_PERSON").json()
    assert default["threshold_source"] == "vendor_default_uncalibrated"


def test_pagination_and_dedupe(client: TestClient, admin: Headers) -> None:
    seed(client, admin)
    q = single(RED, side=90)
    p1 = search(client, admin, q, max_results=2, dedupe_by_image="false").json()
    assert len(p1["matches"]) == 2 and p1["next_offset"] == 2
    p2 = search(
        client, admin, q, max_results=2, offset=p1["next_offset"], dedupe_by_image="false"
    ).json()
    assert [m["rank"] for m in p2["matches"]] == [3, 4]
    p3 = search(client, admin, q, max_results=2, offset=4, dedupe_by_image="false").json()
    assert len(p3["matches"]) == 1 and p3["next_offset"] is None
    seen = [m["face_id"] for p in (p1, p2, p3) for m in p["matches"]]
    assert len(seen) == len(set(seen)) == 5
    deduped = search(client, admin, q, max_results=10).json()  # dedupe_by_image defaults to true
    images = [m["image_id"] for m in deduped["matches"]]
    assert len(images) == len(set(images)) == 4  # b.jpg's two faces collapse to its best one


def test_search_validation_errors(client: TestClient, admin: Headers) -> None:
    seed(client, admin)
    ok = single(RED, side=90)
    assert search(client, admin, ok, mode="FRIEND").json()["error"]["code"] == "invalid_mode"
    assert search(client, admin, ok, max_results=0).status_code == 400
    assert search(client, admin, ok, max_results=10_000).status_code == 400
    r = search(client, admin, ok, min_similarity=7)
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"
    assert search(client, admin, ok, face_index=5).json()["error"]["code"] == "invalid_face_index"
    r = client.post("/v1/faces/search", headers=admin)
    assert r.status_code == 422


def test_search_error_cases_have_structured_codes(client: TestClient, admin: Headers) -> None:
    seed(client, admin)
    cases = [
        (face_image([]), 422, "no_face_detected"),
        (face_image([(RED, 10, 10, 90), (BLUE, 180, 180, 85)]), 422, "multiple_faces_ambiguous"),
        (single(RED, textured=False), 422, "face_quality_insufficient"),
        (single(RED)[:150], 422, "corrupt_image"),
        (b"%PDF-1.4 not an image", 415, "unsupported_image_format"),
    ]
    for data, status, code in cases:
        r = search(client, admin, data)
        assert r.status_code == status, (code, r.text)
        body = r.json()
        assert body["error"]["code"] == code
        assert body["detail"] == body["error"]["message"]
        assert body["error"]["request_id"]
        assert "Traceback" not in r.text and ".py" not in r.text


def test_oversized_upload_is_rejected(
    client: TestClient, admin: Headers, settings: Settings
) -> None:
    big = b"\xff\xd8" + b"0" * (settings.max_upload_bytes + 10)
    r = search(client, admin, big)
    assert r.status_code == 413
    r2 = client.post(
        "/v1/faces/search",
        content=b"x" * (settings.max_upload_bytes * 3),
        headers={**admin, "Content-Type": "multipart/form-data; boundary=x"},
    )
    assert r2.status_code == 413 and r2.json()["error"]["code"] == "request_too_large"


def test_empty_gallery_is_reported(client: TestClient, admin: Headers) -> None:
    body = search(client, admin, single(RED, side=90)).json()
    assert body["status"] == "empty_gallery" and body["matches"] == []


def test_index_not_loaded_per_query_and_concurrent_searches(
    client: TestClient, admin: Headers
) -> None:
    seed(client, admin)

    def one(_: int) -> int:
        r = search(client, admin, single(RED, side=90), max_results=3)
        assert r.status_code == 200, r.text
        return len(r.json()["matches"])

    with ThreadPoolExecutor(8) as ex:
        assert list(ex.map(one, range(32))) == [3] * 32


# --- signed URLs + deletion -----------------------------------------------------------------


def test_signed_urls_serve_files_and_reject_tampering(client: TestClient, admin: Headers) -> None:
    seed(client, admin)
    top = search(client, admin, single(RED, side=90)).json()["matches"][0]
    img = client.get(top["image_url"])
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    assert img.headers["x-content-type-options"] == "nosniff"
    thumb = client.get(top["thumbnail_url"])
    assert thumb.status_code == 200 and thumb.content[:2] == b"\xff\xd8"
    tampered = re.sub(r"sig=[0-9a-f]{4}", "sig=0000", top["image_url"])
    assert client.get(tampered).status_code == 404
    other = re.sub(
        r"/images/[0-9a-f-]+/", "/images/00000000-0000-4000-8000-000000000000/", top["image_url"]
    )
    assert client.get(other).status_code == 404
    expired = re.sub(r"exp=\d+", f"exp={int(time.time()) - 10}", top["image_url"])
    assert client.get(expired).status_code == 404
    assert client.get("/v1/images/..%2f..%2fetc%2fpasswd/file?exp=1&sig=a").status_code == 404
    assert client.get(f"/v1/images/{top['image_id']}/file").status_code == 422  # unsigned


def test_delete_image_removes_files_faces_and_index_entries(
    client: TestClient, admin: Headers, settings: Settings
) -> None:
    seed(client, admin)
    ms = search(client, admin, single(RED, side=90), max_results=10).json()["matches"]
    victim = ms[0]
    r = client.delete(f"/v1/images/{victim['image_id']}", headers=admin)
    assert r.status_code == 200 and r.json() == {
        "image_id": victim["image_id"],
        "deleted": True,
        "faces_removed": 1,
    }
    after = search(client, admin, single(RED, side=90), max_results=10).json()["matches"]
    assert victim["face_id"] not in [m["face_id"] for m in after]
    assert len(after) == len(ms) - 1
    assert client.get(victim["image_url"]).status_code == 404
    assert client.delete(f"/v1/images/{victim['image_id']}", headers=admin).status_code == 404
    assert client.delete("/v1/images/not-a-uuid", headers=admin).status_code == 404
    names = {p.name for p in settings.originals_dir.iterdir()} | {
        p.name for p in settings.crops_dir.iterdir()
    }
    assert not any(victim["image_id"] in n for n in names)
    assert not any(victim["face_id"] in n for n in names)
    assert client.get("/v1/admin/index/check", headers=admin).json()["ok"] is True


# --- auth ------------------------------------------------------------------------------------


def test_auth_required_and_scoped(client: TestClient, admin: Headers, searcher: Headers) -> None:
    img = single(RED)
    assert search(client, {}, img).status_code == 401
    assert (
        search(client, {"Authorization": "Bearer wrong-token-wrong-token"}, img).status_code == 401
    )
    assert search(client, {"Authorization": "Basic abc"}, img).status_code == 401
    assert search(client, {}, img).headers["www-authenticate"] == "Bearer"
    files = [("files", ("a.jpg", img, "image/jpeg"))]
    assert client.post("/v1/gallery/import", files=files).status_code == 401
    assert client.post("/v1/gallery/import", files=files, headers=searcher).status_code == 403
    assert (
        client.delete(
            "/v1/images/00000000-0000-4000-8000-000000000000", headers=searcher
        ).status_code
        == 403
    )
    assert (
        client.get("/v1/jobs/00000000-0000-4000-8000-000000000000", headers=searcher).status_code
        == 403
    )
    assert client.get("/v1/admin/metrics", headers=searcher).status_code == 403
    assert client.get("/stats").status_code == 401
    assert client.get("/stats", headers=searcher).status_code == 200
    assert client.get("/health").status_code == 200 and client.get("/ready").status_code == 200


def test_search_scope_token_can_search(
    client: TestClient, admin: Headers, searcher: Headers
) -> None:
    seed(client, admin)
    assert search(client, searcher, single(RED, side=90)).status_code == 200


def test_auth_can_be_disabled_for_local_dev(tmp_path: Path) -> None:
    s = Settings(data_dir=tmp_path / "d", pipeline="fake", auth_disabled=True, min_face_px=24)
    with TestClient(create_app(s)) as c:
        assert c.get("/stats").status_code == 200


def test_short_tokens_are_refused() -> None:
    with pytest.raises(ValueError):
        Settings(admin_token="short")


def test_rate_limiting(tmp_path: Path) -> None:
    s = Settings(
        data_dir=tmp_path / "d",
        pipeline="fake",
        admin_token="a" * 20,
        min_face_px=24,
        search_rate_limit_per_minute=3,
        rate_limit_per_minute=1000,
    )
    h = {"Authorization": "Bearer " + "a" * 20}
    with TestClient(create_app(s)) as c:
        codes = [search(c, h, single(RED)).status_code for _ in range(6)]
        assert codes[:3] == [200, 200, 200] and 429 in codes[3:]
        r = search(c, h, single(RED))
        assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
        assert c.get("/health").status_code == 200


def test_cors_is_restricted(tmp_path: Path) -> None:
    s = Settings(
        data_dir=tmp_path / "d",
        pipeline="fake",
        auth_disabled=True,
        cors_origins="https://app.example",
    )
    with TestClient(create_app(s)) as c:
        ok = c.get("/health", headers={"Origin": "https://app.example"})
        assert ok.headers["access-control-allow-origin"] == "https://app.example"
        bad = c.get("/health", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in bad.headers
    s2 = Settings(data_dir=tmp_path / "d2", pipeline="fake", auth_disabled=True)
    with TestClient(create_app(s2)) as c2:
        assert (
            "access-control-allow-origin"
            not in c2.get("/health", headers={"Origin": "https://app.example"}).headers
        )


def test_error_responses_do_not_leak_internals(client: TestClient, admin: Headers) -> None:
    r = client.get("/no/such/route")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    r = client.post("/health")
    assert r.status_code == 405
    assert "Traceback" not in r.text


def test_metrics_endpoint_has_no_biometric_data(client: TestClient, admin: Headers) -> None:
    seed(client, admin)
    search(client, admin, single(RED, side=90))
    m = client.get("/v1/admin/metrics", headers=admin).json()
    assert m["counters"]["search_requests"] == 1
    assert m["latency"]["search_total"]["count"] == 1
    assert m["queue_depth"] == 0 and m["images"] == 4
    assert m["process"]["rss_mb"] > 0
    rebuilt = client.post("/v1/admin/index/rebuild", headers=admin).json()
    assert rebuilt["vectors"] == 5


# --- legacy compatibility --------------------------------------------------------------------


def test_legacy_upload_and_search_contract(client: TestClient, admin: Headers) -> None:
    files = [
        ("files", ("one.jpg", single(RED), "image/jpeg")),
        ("files", ("two.jpg", single(BLUE), "image/jpeg")),
    ]
    r = client.post("/upload-images", files=files, headers=admin)
    assert r.status_code == 200
    body = r.json()
    assert body["message"] == "Processed 2 images"
    first = body["results"][0]
    assert first["filename"] == "one.jpg" and first["status"] == "success"
    assert first["embedding_shape"] == [128] and isinstance(first["embedding_id"], int)
    assert "image_path" not in first and "/" not in first["image_id"]
    again = client.post("/upload-images", files=files[:1], headers=admin).json()["results"][0]
    assert again["status"] == "success" and again["duplicate"] is True

    s = client.post(
        "/search-face",
        files={"file": ("q.jpg", single(RED, side=90), "image/jpeg")},
        data={"threshold": "0.5", "max_results": "5"},
        headers=admin,
    )
    assert s.status_code == 200
    sb = s.json()
    assert {"message", "query_filename", "threshold", "total_faces_searched", "matches"} <= set(sb)
    assert sb["query_filename"] == "q.jpg" and sb["total_faces_searched"] == 2
    assert sb["matches"] and set(sb["matches"][0]) >= {"distance", "similarity_score", "image_url"}
    assert "image_path" not in s.text
    assert all(m["distance"] <= 0.5 for m in sb["matches"])
    stats = client.get("/stats", headers=admin).json()
    assert stats["total_faces_stored"] == 2
    assert "storage_path" not in stats and "database_path" not in stats


def test_legacy_error_semantics(client: TestClient, admin: Headers) -> None:
    assert (
        client.post(
            "/upload-images",
            files=[("files", ("a.jpg", single(RED), "image/jpeg"))] * 3,
            headers=admin,
        ).status_code
        == 400
    )
    r = client.post(
        "/upload-images",
        files=[("files", ("empty.jpg", face_image([]), "image/jpeg"))],
        headers=admin,
    ).json()
    assert r["results"][0]["status"] == "error" and "No face" in r["results"][0]["message"]
    nf = client.post(
        "/search-face", files={"file": ("q.jpg", face_image([]), "image/jpeg")}, headers=admin
    )
    assert nf.status_code == 400  # legacy reported undecodable/no-face as 400
    assert nf.json()["detail"]


# --- multi-face gallery photo ---------------------------------------------------------------


def test_every_face_in_a_gallery_photo_is_searchable(client: TestClient, admin: Headers) -> None:
    seed(client, admin)
    for colour in (GREEN, BLUE):
        top = search(client, admin, single(colour, side=90)).json()["matches"][0]
        assert top["cosine_similarity"] > 0.95
    b_match = search(client, admin, single(GREEN, side=90)).json()["matches"][0]
    b_match2 = search(client, admin, single(BLUE, side=90)).json()["matches"][0]
    assert b_match["image_id"] == b_match2["image_id"]
    assert b_match["face_id"] != b_match2["face_id"]
