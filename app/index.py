from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import faiss
import numpy as np

from .errors import IndexUnavailable
from .ml.base import ModelInfo
from .repository import SqlGalleryRepository

logger = logging.getLogger(__name__)

SNAPSHOT_FORMAT = 1


class VectorIndex(Protocol):
    dim: int

    def add(self, ids: np.ndarray, vectors: np.ndarray) -> None: ...
    def remove(self, ids: np.ndarray) -> int: ...
    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]: ...
    def count(self) -> int: ...
    def all_ids(self) -> np.ndarray: ...


class FaissFlatIndex:
    """Exact inner-product index (cosine on L2-normalised vectors) with stable integer ids.

    Exact search is the default because the benchmark (docs/BENCHMARKS.md) shows it comfortably
    meets interactive latency at the supported gallery sizes; approximate indexes would add recall
    loss and tombstone handling for no measured benefit.
    """

    def __init__(self, dim: int, index: Any | None = None) -> None:
        self.dim = dim
        self._index: Any = index or faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    @property
    def raw(self) -> Any:
        return self._index

    def add(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError("vector dimension mismatch")
        self._index.add_with_ids(vectors, np.ascontiguousarray(ids, dtype=np.int64))

    def remove(self, ids: np.ndarray) -> int:
        if len(ids) == 0:
            return 0
        return int(self._index.remove_ids(np.ascontiguousarray(ids, dtype=np.int64)))

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        n = self._index.ntotal
        if n == 0 or k <= 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        q = np.ascontiguousarray(np.asarray(query, dtype=np.float32).reshape(1, -1))
        scores, ids = self._index.search(q, min(k, n))
        mask = ids[0] >= 0
        return ids[0][mask], scores[0][mask]

    def count(self) -> int:
        return int(self._index.ntotal)

    def all_ids(self) -> np.ndarray:
        return np.asarray(faiss.vector_to_array(self._index.id_map), dtype=np.int64)


class RWLock:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._readers = 0
        self._writer = False

    @contextlib.contextmanager
    def read(self) -> Iterator[None]:
        with self._cond:
            while self._writer:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                if self._readers == 0:
                    self._cond.notify_all()

    @contextlib.contextmanager
    def write(self) -> Iterator[None]:
        with self._cond:
            while self._writer or self._readers:
                self._cond.wait()
            self._writer = True
        try:
            yield
        finally:
            with self._cond:
                self._writer = False
                self._cond.notify_all()


def slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", model_id)


@dataclass
class ConsistencyReport:
    model_id: str
    db_faces: int
    index_faces: int
    missing_in_index: list[int]
    extra_in_index: list[int]

    @property
    def ok(self) -> bool:
        return not self.missing_in_index and not self.extra_in_index

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "ok": self.ok,
            "db_faces": self.db_faces,
            "index_faces": self.index_faces,
            "missing_in_index": len(self.missing_in_index),
            "extra_in_index": len(self.extra_in_index),
        }


class IndexManager:
    """Persistent per-model vector index kept in sync with the relational DB (the source of truth).

    * One index per model id; the manager is bound to one ModelInfo so embeddings from different
      models/preprocessing can never be mixed.
    * Snapshots are derived data: any snapshot that fails validation is discarded and rebuilt.
    * Additions are replayed by face `seq`, removals by the `index_events` log, so the API process
      follows writes made by the worker process without loading the gallery per query.
    """

    def __init__(
        self,
        repo: SqlGalleryRepository,
        info: ModelInfo,
        index_dir: Path,
        *,
        sync_interval_s: float = 1.0,
        persist_every: int = 500,
    ) -> None:
        if info.metric != "cosine":
            raise ValueError("only cosine (inner product on normalised vectors) is supported")
        self.repo = repo
        self.info = info
        self.dir = index_dir / slug(info.model_id)
        self.sync_interval_s = sync_interval_s
        self.persist_every = persist_every
        self._lock = RWLock()
        self._sync_lock = threading.Lock()
        self._index = FaissFlatIndex(info.dim)
        self._last_seq = 0
        self._last_event = 0
        self._last_sync = 0.0
        self._dirty = 0
        self._loaded = False
        self.last_persisted_at: float | None = None
        self.rebuilt_on_load = False

    # --- persistence -----------------------------------------------------------------------

    @property
    def _index_path(self) -> Path:
        return self.dir / "index.faiss"

    @property
    def _meta_path(self) -> Path:
        return self.dir / "meta.json"

    @staticmethod
    def _sha(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            while chunk := fh.read(1 << 20):
                h.update(chunk)
        return h.hexdigest()

    def persist(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tag = uuid.uuid4().hex
        tmp_index = self.dir / f".index.{tag}.tmp"
        tmp_meta = self.dir / f".meta.{tag}.tmp"
        with self._lock.read():
            faiss.write_index(self._index.raw, str(tmp_index))
            meta = {
                "snapshot_format": SNAPSHOT_FORMAT,
                "model_id": self.info.model_id,
                "dim": self.info.dim,
                "metric": self.info.metric,
                "count": self._index.count(),
                "last_seq": self._last_seq,
                "last_event": self._last_event,
                "sha256": self._sha(tmp_index),
                "saved_at": time.time(),
            }
        tmp_meta.write_text(json.dumps(meta))
        os.replace(tmp_index, self._index_path)
        os.replace(tmp_meta, self._meta_path)
        self._dirty = 0
        self.last_persisted_at = meta["saved_at"]  # type: ignore[assignment]

    def _try_load_snapshot(self) -> bool:
        try:
            meta = json.loads(self._meta_path.read_text())
            if (
                meta.get("snapshot_format") != SNAPSHOT_FORMAT
                or meta.get("model_id") != self.info.model_id
                or meta.get("dim") != self.info.dim
                or meta.get("metric") != self.info.metric
            ):
                logger.warning("Index snapshot does not match the active model; rebuilding")
                return False
            if self._sha(self._index_path) != meta.get("sha256"):
                logger.warning("Index snapshot failed its checksum; rebuilding")
                return False
            raw = faiss.read_index(str(self._index_path))
            if raw.d != self.info.dim or raw.ntotal != meta.get("count"):
                return False
        except (OSError, ValueError, KeyError, RuntimeError):
            return False
        self._index = FaissFlatIndex(self.info.dim, raw)
        self._last_seq = int(meta["last_seq"])
        self._last_event = int(meta["last_event"])
        return True

    # --- lifecycle -------------------------------------------------------------------------

    def load(self) -> None:
        with self._sync_lock:
            if self._try_load_snapshot():
                self.rebuilt_on_load = False
            else:
                self.rebuilt_on_load = True
                self._rebuild_locked()
            self._loaded = True
        self.sync(force=True)

    def _rebuild_locked(self) -> None:
        event0 = self.repo.max_event_id(self.info.model_id)
        fresh = FaissFlatIndex(self.info.dim)
        last = 0
        for seqs, vecs in self.repo.iter_embeddings(self.info.model_id, self.info.dim):
            fresh.add(seqs, vecs)
            last = int(seqs[-1])
        with self._lock.write():
            self._index = fresh
            self._last_seq = last
            self._last_event = event0
        self.persist()
        logger.info("Rebuilt index for %s: %d vectors", self.info.model_id, fresh.count())

    def rebuild(self) -> int:
        with self._sync_lock:
            self._rebuild_locked()
            self._loaded = True
            self._last_sync = time.monotonic()
        self.sync(force=True)
        return self._index.count()

    def sync(self, force: bool = False) -> int:
        """Apply DB changes made since the last sync. Cheap when nothing changed."""
        if not self._loaded:
            raise IndexUnavailable("The vector index is not loaded.")
        now = time.monotonic()
        if not force and now - self._last_sync < self.sync_interval_s:
            return 0
        applied = 0
        with self._sync_lock:
            self._last_sync = now
            model = self.info.model_id
            if self.repo.max_face_seq(model) > self._last_seq:
                for seqs, vecs in self.repo.iter_embeddings(
                    model, self.info.dim, after_seq=self._last_seq
                ):
                    with self._lock.write():
                        self._index.add(seqs, vecs)
                        self._last_seq = int(seqs[-1])
                    applied += len(seqs)
            events = self.repo.events_after(model, self._last_event)
            if events:
                with self._lock.write():
                    self._index.remove(np.array([e[1] for e in events], dtype=np.int64))
                    self._last_event = events[-1][0]
                applied += len(events)
            self._dirty += applied
            should_persist = self._dirty >= self.persist_every
        if should_persist:
            self.persist()
        return applied

    # --- queries ---------------------------------------------------------------------------

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if not self._loaded:
            raise IndexUnavailable("The vector index is not loaded.")
        if query.shape != (self.info.dim,):
            raise ValueError("query dimension mismatch")
        self.sync()
        with self._lock.read():
            return self._index.search(query, k)

    def count(self) -> int:
        return self._index.count()

    def check(self) -> ConsistencyReport:
        self.sync(force=True)
        with self._lock.read():
            in_index = {int(i) for i in self._index.all_ids()}
        in_db = self.repo.face_seqs(self.info.model_id)
        return ConsistencyReport(
            model_id=self.info.model_id,
            db_faces=len(in_db),
            index_faces=len(in_index),
            missing_in_index=sorted(in_db - in_index),
            extra_in_index=sorted(in_index - in_db),
        )

    def health(self) -> dict[str, object]:
        return {
            "model_id": self.info.model_id,
            "loaded": self._loaded,
            "vectors": self._index.count(),
            "last_face_seq": self._last_seq,
            "last_event_id": self._last_event,
            "snapshot_age_s": None
            if self.last_persisted_at is None
            else round(time.time() - self.last_persisted_at, 1),
            "rebuilt_on_load": self.rebuilt_on_load,
        }
