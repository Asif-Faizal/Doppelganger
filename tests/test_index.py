from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from app.container import Container
from app.db import Database, utcnow
from app.errors import IndexUnavailable
from app.index import FaissFlatIndex, IndexManager
from app.migrate import upgrade
from app.ml.base import ModelInfo
from app.repository import ImageRecord, NewFace, SqlGalleryRepository


def unit(rng: np.random.Generator, n: int, dim: int = 32) -> np.ndarray:
    v = rng.normal(size=(n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_flat_index_matches_brute_force_topk_ordering() -> None:
    rng = np.random.default_rng(0)
    data = unit(rng, 500)
    idx = FaissFlatIndex(32)
    ids = np.arange(1000, 1500, dtype=np.int64)
    idx.add(ids, data)
    q = unit(rng, 1)[0]
    got_ids, got_scores = idx.search(q, 10)
    brute = data @ q
    order = np.argsort(-brute)[:10]
    assert list(got_ids) == list(ids[order])
    assert np.allclose(got_scores, brute[order], atol=1e-5)
    assert all(got_scores[i] >= got_scores[i + 1] for i in range(9))


def test_flat_index_remove_and_dimension_validation() -> None:
    rng = np.random.default_rng(1)
    idx = FaissFlatIndex(32)
    idx.add(np.arange(10, dtype=np.int64), unit(rng, 10))
    assert idx.remove(np.array([3, 4], dtype=np.int64)) == 2
    assert idx.count() == 8
    assert 3 not in set(idx.all_ids())
    with pytest.raises(ValueError):
        idx.add(np.array([99], dtype=np.int64), unit(rng, 1, dim=16))
    assert idx.search(unit(rng, 1)[0], 0)[0].size == 0


def test_empty_index_search_returns_nothing() -> None:
    ids, scores = FaissFlatIndex(8).search(np.ones(8, dtype=np.float32), 5)
    assert ids.size == 0 and scores.size == 0


# --- manager + repository ----------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path: Path) -> SqlGalleryRepository:
    url = f"sqlite:///{tmp_path / 't.db'}"
    upgrade(url)
    return SqlGalleryRepository(Database(url))


def info(model_id: str = "m1", dim: int = 16) -> ModelInfo:
    return ModelInfo(model_id, "d", "1", "e", "1", dim, "cosine", "p")


def add_image(
    repo: SqlGalleryRepository, vecs: np.ndarray, model_id: str = "m1", name: str = "x"
) -> str:
    import uuid

    image_id = str(uuid.uuid4())
    repo.create_image(
        ImageRecord(
            image_id,
            uuid.uuid4().hex * 2,
            f"{image_id}.jpg",
            "JPEG",
            100,
            100,
            10,
            name,
            None,
            utcnow(),
        )
    )
    repo.store_faces(
        image_id,
        model_id,
        [NewFace(i, (1, 2, 3, 4), 0.9, {}, v) for i, v in enumerate(vecs)],
        {},
    )
    return image_id


def manager(
    repo: SqlGalleryRepository, tmp_path: Path, model_id: str = "m1", dim: int = 16
) -> IndexManager:
    m = IndexManager(
        repo, info(model_id, dim), tmp_path / "idx", sync_interval_s=0.0, persist_every=1
    )
    m.load()
    return m


def test_sync_follows_inserts_and_deletes(repo: SqlGalleryRepository, tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    mgr = manager(repo, tmp_path)
    assert mgr.count() == 0
    a = add_image(repo, unit(rng, 3, 16))
    add_image(repo, unit(rng, 2, 16))
    mgr.sync()
    assert mgr.count() == 5
    repo.delete_image(a)
    mgr.sync()
    assert mgr.count() == 2
    assert mgr.check().ok


def test_replacing_faces_for_same_image_is_reflected(
    repo: SqlGalleryRepository, tmp_path: Path
) -> None:
    rng = np.random.default_rng(3)
    mgr = manager(repo, tmp_path)
    image_id = add_image(repo, unit(rng, 2, 16))
    mgr.sync()
    repo.store_faces(image_id, "m1", [NewFace(0, (0, 0, 1, 1), 0.9, {}, unit(rng, 1, 16)[0])], {})
    mgr.sync()
    assert mgr.count() == 1
    assert mgr.check().ok


def test_snapshot_roundtrip_and_incremental_catch_up(
    repo: SqlGalleryRepository, tmp_path: Path
) -> None:
    rng = np.random.default_rng(4)
    add_image(repo, unit(rng, 4, 16))
    first = manager(repo, tmp_path)
    assert first.rebuilt_on_load  # nothing persisted yet
    first.persist()
    add_image(repo, unit(rng, 2, 16))  # written while "API is down"
    second = manager(repo, tmp_path)
    assert not second.rebuilt_on_load  # snapshot reused ...
    assert second.count() == 6  # ... and caught up from the DB
    assert second.check().ok


def test_corrupt_snapshot_triggers_rebuild(repo: SqlGalleryRepository, tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    add_image(repo, unit(rng, 4, 16))
    m = manager(repo, tmp_path)
    m.persist()
    (m.dir / "index.faiss").write_bytes(b"garbage")
    again = manager(repo, tmp_path)
    assert again.rebuilt_on_load
    assert again.count() == 4


def test_snapshot_for_other_model_is_never_loaded(
    repo: SqlGalleryRepository, tmp_path: Path
) -> None:
    rng = np.random.default_rng(6)
    add_image(repo, unit(rng, 3, 16), model_id="m1")
    add_image(repo, unit(rng, 5, 16), model_id="m2")
    m1 = manager(repo, tmp_path, "m1")
    assert m1.count() == 3  # m2 faces never mixed in
    m2 = manager(repo, tmp_path, "m2")
    assert m2.count() == 5
    # tamper: copy m1 snapshot over m2's location -> metadata mismatch -> rebuild, not reuse
    m1.persist()
    (m2.dir).mkdir(parents=True, exist_ok=True)
    (m2.dir / "index.faiss").write_bytes((m1.dir / "index.faiss").read_bytes())
    (m2.dir / "meta.json").write_bytes((m1.dir / "meta.json").read_bytes())
    m2b = manager(repo, tmp_path, "m2")
    assert m2b.rebuilt_on_load and m2b.count() == 5


def test_dimension_change_forces_rebuild_and_bad_blobs_are_reported(
    repo: SqlGalleryRepository, tmp_path: Path
) -> None:
    rng = np.random.default_rng(7)
    manager(repo, tmp_path).persist()  # empty snapshot, dim 16
    other = IndexManager(repo, info("m1", 8), tmp_path / "idx", sync_interval_s=0.0)
    other.load()
    assert other.rebuilt_on_load  # snapshot dim 16 != 8
    add_image(repo, unit(rng, 2, 16))  # 16-d vectors stored under a model declared as 8-d
    with pytest.raises(IndexUnavailable) as exc:
        other.sync(force=True)
    assert exc.value.code == "embedding_size_mismatch"


def test_consistency_check_detects_drift_and_rebuild_fixes_it(
    repo: SqlGalleryRepository, tmp_path: Path
) -> None:
    rng = np.random.default_rng(8)
    add_image(repo, unit(rng, 5, 16))
    mgr = manager(repo, tmp_path)
    ids = mgr._index.all_ids()
    mgr._index.remove(ids[:2])
    report = mgr.check()
    assert not report.ok and len(report.missing_in_index) == 2
    mgr.rebuild()
    assert mgr.check().ok


def test_search_through_manager_never_loads_gallery_per_query(
    repo: SqlGalleryRepository, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(9)
    vecs = unit(rng, 50, 16)
    add_image(repo, vecs)
    mgr = manager(repo, tmp_path)

    def boom(*a: object, **k: object) -> None:
        raise AssertionError("gallery was loaded during a query")

    monkeypatch.setattr(repo, "iter_embeddings", boom)
    ids, scores = mgr.search(vecs[7], 3)
    assert scores[0] == pytest.approx(1.0, abs=1e-5)
    assert len(ids) == 3


def test_concurrent_search_while_writing(repo: SqlGalleryRepository, tmp_path: Path) -> None:
    rng = np.random.default_rng(10)
    vecs = unit(rng, 200, 16)
    add_image(repo, vecs[:100])
    mgr = manager(repo, tmp_path)
    errors: list[BaseException] = []
    stop = threading.Event()

    def searcher() -> None:
        try:
            while not stop.is_set():
                ids, scores = mgr.search(vecs[0], 5)
                assert len(ids) == 5
                assert all(scores[i] >= scores[i + 1] - 1e-6 for i in range(len(scores) - 1))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=searcher) for _ in range(6)]
    for t in threads:
        t.start()
    for i in range(100, 200, 10):
        add_image(repo, vecs[i : i + 10])
        mgr.sync()
    stop.set()
    for t in threads:
        t.join()
    assert not errors
    assert mgr.count() == 200


def test_container_index_matches_repo(container: Container) -> None:
    assert container.index is not None
    assert container.index.check().ok
