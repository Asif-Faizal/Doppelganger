from __future__ import annotations

import threading
import time
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import select, update

from app.container import Container
from app.db import Face, ImageProcessing, JobItem, utcnow
from app.errors import CorruptImage, ProcessingFailed
from app.jobs import ItemSpec, Worker, import_directory_items
from app.ml.fake import build_fake_pipeline_parts
from app.ml.pipeline import FacePipeline, PipelinePool

from .conftest import BLUE, GREEN, ORANGE, RED, face_image, single


def run_job(c: Container, items: list[ItemSpec], kind: str = "import") -> dict:  # type: ignore[type-arg]
    assert c.pool and c.worker
    job_id = c.jobs.create_job(kind, c.pool.info.model_id, items)
    c.worker.run_until_empty()
    job = c.jobs.get_job(job_id)
    assert job is not None
    return job


def stage(c: Container, data: bytes, name: str = "p.jpg") -> ItemSpec:
    return ItemSpec(source_path=str(c.store.stage(data)), delete_source=True, display_name=name)


def test_every_eligible_face_is_indexed_with_model_id(container: Container) -> None:
    job = run_job(
        container,
        [
            stage(
                container,
                face_image([(RED, 10, 10, 80), (GREEN, 120, 10, 80), (BLUE, 10, 150, 80)]),
            )
        ],
    )
    assert job["status"] == "completed"
    assert job["counts"]["succeeded"] == 1
    assert job["faces_indexed"] == 3
    with container.db.read() as s:
        faces = s.scalars(select(Face)).all()
    assert len(faces) == 3
    assert {f.model_id for f in faces} == {container.pool.info.model_id}  # type: ignore[union-attr]
    assert all(len(f.embedding) == 128 * 4 for f in faces)
    container.index.sync(force=True)  # type: ignore[union-attr]
    assert container.index.count() == 3  # type: ignore[union-attr]
    # staged upload removed after success; originals kept under server-generated names
    assert not list(container.settings.staging_dir.glob("*.upload"))
    assert len(list(container.settings.originals_dir.iterdir())) == 1


def test_duplicate_files_are_skipped_by_hash(container: Container) -> None:
    data = single(RED)
    job = run_job(
        container, [stage(container, data, "a.jpg"), stage(container, data, "copy-of-a.jpg")]
    )
    counts = job["counts"]
    assert counts["succeeded"] + counts["duplicate"] == 2
    assert counts["duplicate"] == 1
    assert container.repo.count_images() == 1
    assert container.repo.count_faces() == 1
    again = run_job(container, [stage(container, data, "a-again.jpg")])
    assert again["counts"]["duplicate"] == 1
    assert container.repo.count_faces() == 1


def test_no_face_image_is_not_a_failure_and_is_not_reprocessed(container: Container) -> None:
    job = run_job(container, [stage(container, face_image([]))])
    assert job["status"] == "completed"
    assert job["counts"]["no_face"] == 1
    with container.db.read() as s:
        assert s.scalars(select(ImageProcessing.status)).all() == ["no_face"]


def test_corrupt_file_fails_permanently_without_blocking_others(container: Container) -> None:
    good = stage(container, single(RED))
    bad = stage(container, single(BLUE)[:200], "broken.jpg")
    junk = stage(container, b"not an image at all", "evil.jpg")
    job = run_job(container, [bad, good, junk])
    assert job["status"] == "completed_with_errors"
    assert job["counts"]["succeeded"] == 1
    assert job["counts"]["failed"] == 2
    codes = {e["code"] for e in job["errors"]}
    assert codes <= {"corrupt_image", "unsupported_image_format"}
    assert all(e["attempts"] == 1 for e in job["errors"])  # permanent errors are not retried
    assert container.repo.count_faces() == 1


def test_all_failed_job_status(container: Container) -> None:
    job = run_job(container, [stage(container, b"garbage")])
    assert job["status"] == "failed"


def test_transient_failure_is_retried_then_succeeds(
    container: Container, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert container.ingest is not None
    real = container.ingest.ingest_bytes
    calls = {"n": 0}

    def flaky(*a: object, **k: object):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] < 3:
            raise ProcessingFailed("boom")
        return real(*a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(container.ingest, "ingest_bytes", flaky)
    job = run_job(container, [stage(container, single(RED))])
    assert calls["n"] == 3
    assert job["status"] == "completed"
    assert container.repo.count_faces() == 1


def test_transient_failure_gives_up_after_max_attempts(
    container: Container, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert container.ingest is not None

    def always(*a: object, **k: object):  # type: ignore[no-untyped-def]
        raise RuntimeError("internal detail /secret/path must not leak")

    monkeypatch.setattr(container.ingest, "ingest_bytes", always)
    spec = stage(container, single(RED))
    job = run_job(container, [spec])
    assert job["status"] == "failed"
    assert job["errors"][0]["attempts"] == container.settings.job_max_attempts
    assert "secret" not in str(job)
    # the staged file is retained for retries while attempts remain, and cleaned once final
    assert not Path(spec.source_path or "").exists()


def test_job_recovers_after_worker_crash_via_lease_expiry(container: Container) -> None:
    job_id = container.jobs.create_job(
        "import", container.pool.info.model_id, [stage(container, single(RED))]
    )  # type: ignore[union-attr]
    claimed = container.jobs.claim_next()  # a worker claims it ... and "crashes"
    assert claimed is not None
    assert container.jobs.claim_next() is None  # still leased: nobody else may take it
    with container.db.write() as s:
        s.execute(update(JobItem).values(lease_expires_at=utcnow() - timedelta(seconds=1)))
    assert container.worker is not None
    container.worker.run_until_empty()  # restarted worker reclaims it
    job = container.jobs.get_job(job_id)
    assert job is not None and job["status"] == "completed"
    assert container.repo.count_faces() == 1


def test_item_crashing_repeatedly_is_failed_not_looped(container: Container) -> None:
    job_id = container.jobs.create_job(
        "import", container.pool.info.model_id, [stage(container, single(RED))]
    )  # type: ignore[union-attr]
    for _ in range(container.settings.job_max_attempts):
        assert container.jobs.claim_next() is not None
        with container.db.write() as s:
            s.execute(update(JobItem).values(lease_expires_at=utcnow() - timedelta(seconds=1)))
    assert container.jobs.claim_next() is None
    job = container.jobs.get_job(job_id)
    assert job is not None
    assert job["status"] == "failed"
    assert job["errors"][0]["code"] == "max_attempts_exceeded"


def test_ingestion_is_idempotent_for_same_image_and_model(container: Container) -> None:
    assert container.ingest is not None
    data = face_image([(RED, 10, 10, 80), (GREEN, 150, 150, 80)])
    first = container.ingest.ingest_bytes(data)
    again = container.ingest.ingest_bytes(data)
    forced = container.ingest.ingest_bytes(data, force=True)  # retry after partial failure
    assert first.status == "processed" and first.face_count == 2
    assert again.status == "duplicate" and again.image_id == first.image_id
    assert forced.face_count == 2
    assert container.repo.count_faces() == 2  # replaced, not appended
    assert container.index.check().ok  # type: ignore[union-attr]
    assert len(list(container.settings.crops_dir.iterdir())) == 2  # old crops cleaned up


def test_concurrent_ingest_of_same_bytes_creates_one_image(container: Container) -> None:
    assert container.ingest is not None
    data = single(ORANGE)
    results: list[str] = []

    def go() -> None:
        results.append(container.ingest.ingest_bytes(data).image_id)  # type: ignore[union-attr]

    threads = [threading.Thread(target=go) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(results)) == 1
    assert container.repo.count_images() == 1
    assert container.repo.count_faces() == 1


def test_cancel_marks_pending_items_skipped(container: Container) -> None:
    job_id = container.jobs.create_job(
        "import",
        container.pool.info.model_id,
        [stage(container, single(RED)), stage(container, single(BLUE))],
    )  # type: ignore[union-attr]
    assert container.jobs.cancel(job_id)
    job = container.jobs.get_job(job_id)
    assert job is not None
    assert job["status"] == "cancelled" and job["counts"]["skipped"] == 2


def test_background_workers_process_concurrently_and_stop_gracefully(container: Container) -> None:
    assert container.pool and container.ingest
    worker = Worker(
        container.settings, container.jobs, container.ingest, container.store, concurrency=3
    )
    colours = [(i * 20 % 255, 200 - i * 10, 40 + i * 9) for i in range(12)]
    items = [stage(container, single(c, side=60 + i), f"{i}.jpg") for i, c in enumerate(colours)]
    job_id = container.jobs.create_job("import", container.pool.info.model_id, items)
    worker.start()
    deadline = time.time() + 20
    while time.time() < deadline:
        job = container.jobs.get_job(job_id)
        assert job is not None
        if job["status"] in ("completed", "completed_with_errors"):
            break
        time.sleep(0.05)
    worker.stop()
    assert job["status"] == "completed" and job["counts"]["succeeded"] == 12
    assert container.settings.heartbeat_file.exists()


def test_reprocess_with_new_model_recomputes_and_keeps_models_separate(
    container: Container,
) -> None:
    assert container.ingest and container.worker and container.pool
    container.ingest.ingest_bytes(face_image([(RED, 10, 10, 80), (GREEN, 150, 150, 80)]))
    old_model = container.pool.info.model_id
    d, e, info = build_fake_pipeline_parts(dim=64)
    new_pool = PipelinePool([FacePipeline(d, e, info, container.settings)])
    from app.ingest import IngestService

    new_ingest = IngestService(
        container.settings, container.repo, container.store, new_pool, container.metrics
    )
    ids = container.repo.all_image_ids()
    for image_id in ids:
        out = new_ingest.reprocess_image(image_id)
        assert out.face_count == 2
    by_model = container.repo.faces_by_model()
    assert by_model == {old_model: 2, info.model_id: 2}
    with container.db.read() as s:
        for f in s.scalars(select(Face)):
            assert (
                len(f.embedding) == (128 if f.model_id == old_model else 64) * 4
            )  # recomputed, never reused


def test_reprocess_job_reports_missing_original(container: Container) -> None:
    assert container.ingest and container.pool
    out = container.ingest.ingest_bytes(single(RED))
    rec = container.repo.get_image(out.image_id)
    assert rec is not None
    container.store.delete_original(rec.storage_name)
    job = run_job(container, [ItemSpec(image_id=out.image_id)], kind="reprocess")
    assert job["status"] == "failed" or job["counts"]["failed"] == 1 or job["errors"]
    assert job["errors"][0]["code"] == "original_missing"


def test_import_directory_skips_symlinks_and_non_images(tmp_path: Path) -> None:
    root = tmp_path / "gallery"
    (root / "sub").mkdir(parents=True)
    (root / "a.jpg").write_bytes(b"x")
    (root / "notes.txt").write_text("hi")
    (root / "sub" / "b.PNG").write_bytes(b"x")
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"x")
    (root / "link.jpg").symlink_to(outside)
    flat = import_directory_items(root, recursive=False)
    deep = import_directory_items(root, recursive=True)
    assert [Path(i.source_path or "").name for i in flat] == ["a.jpg"]
    assert sorted(Path(i.source_path or "").name for i in deep) == ["a.jpg", "b.PNG"]


def test_corrupt_error_never_contains_raw_bytes(container: Container) -> None:
    assert container.ingest is not None
    with pytest.raises(CorruptImage) as exc:
        container.ingest.ingest_bytes(single(RED)[:300])
    assert "\\x" not in exc.value.message
