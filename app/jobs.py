from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from .config import Settings
from .db import Database, Job, JobItem, retry_locked, utcnow
from .errors import PERMANENT_INGEST_CODES, ApiError, FileTooLarge, NotFound
from .ingest import IngestService, clean_display_name
from .storage import FileStore, new_id

logger = logging.getLogger(__name__)

TERMINAL_ITEM = ("done", "no_face", "duplicate", "skipped", "failed")


@dataclass(frozen=True)
class ItemSpec:
    source_path: str | None = None
    delete_source: bool = False
    display_name: str | None = None
    image_id: str | None = None
    # Already-resolved outcome (e.g. rejected before staging); stored as a finished item.
    preset_status: str | None = None
    preset_error_code: str | None = None
    preset_error_message: str | None = None


@dataclass(frozen=True)
class ClaimedItem:
    id: int
    job_id: str
    source_path: str | None
    delete_source: bool
    display_name: str | None
    image_id: str | None
    attempts: int


class JobService:
    def __init__(self, db: Database, settings: Settings) -> None:
        self.db = db
        self.settings = settings

    def create_job(self, kind: str, model_id: str, items: list[ItemSpec]) -> str:
        job_id = new_id()
        with self.db.write() as s:
            s.add(Job(id=job_id, kind=kind, status="queued", model_id=model_id, total=len(items)))
            for it in items:
                s.add(
                    JobItem(
                        job_id=job_id,
                        status=it.preset_status or "pending",
                        source_path=it.source_path,
                        delete_source=it.delete_source,
                        display_name=clean_display_name(it.display_name),
                        image_id=it.image_id,
                        error_code=it.preset_error_code,
                        error_message=(it.preset_error_message or "")[:255] or None,
                    )
                )
            s.flush()
            self._refresh_job(s, job_id)
        return job_id

    def get_job(self, job_id: str, max_errors: int = 20) -> dict[str, Any] | None:
        with self.db.read() as s:
            job = s.get(Job, job_id)
            if job is None:
                return None
            counts = dict(
                s.execute(
                    select(JobItem.status, func.count())
                    .where(JobItem.job_id == job_id)
                    .group_by(JobItem.status)
                ).all()
            )
            errors = s.execute(
                select(
                    JobItem.display_name,
                    JobItem.error_code,
                    JobItem.error_message,
                    JobItem.attempts,
                )
                .where(JobItem.job_id == job_id, JobItem.status == "failed")
                .order_by(JobItem.id)
                .limit(max_errors)
            ).all()
            faces = s.scalar(
                select(func.coalesce(func.sum(JobItem.face_count), 0)).where(
                    JobItem.job_id == job_id
                )
            )
            finished = sum(counts.get(k, 0) for k in TERMINAL_ITEM)
            return {
                "job_id": job.id,
                "kind": job.kind,
                "status": job.status,
                "model_id": job.model_id,
                "total": job.total,
                "progress": round(finished / job.total, 4) if job.total else 1.0,
                "counts": {
                    "pending": counts.get("pending", 0),
                    "running": counts.get("running", 0),
                    "succeeded": counts.get("done", 0),
                    "no_face": counts.get("no_face", 0),
                    "duplicate": counts.get("duplicate", 0),
                    "skipped": counts.get("skipped", 0),
                    "failed": counts.get("failed", 0),
                },
                "faces_indexed": int(faces or 0),
                "created_at": job.created_at,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
                "errors": [
                    {"name": e[0], "code": e[1], "message": e[2], "attempts": e[3]} for e in errors
                ],
            }

    def cancel(self, job_id: str) -> bool:
        with self.db.write() as s:
            job = s.get(Job, job_id)
            if job is None:
                return False
            job.cancel_requested = True
            s.execute(
                update(JobItem)
                .where(JobItem.job_id == job_id, JobItem.status == "pending")
                .values(status="skipped", error_code="cancelled", updated_at=utcnow())
            )
            s.flush()
            self._refresh_job(s, job_id)
            return True

    # --- worker side -----------------------------------------------------------------------

    def claim_next(self) -> ClaimedItem | None:
        now = utcnow()
        with self.db.write() as s:
            exhausted = s.scalars(
                select(JobItem).where(
                    JobItem.status == "running",
                    JobItem.lease_expires_at < now,
                    JobItem.attempts >= self.settings.job_max_attempts,
                )
            ).all()
            for it in exhausted:
                it.status = "failed"
                it.error_code = "max_attempts_exceeded"
                it.error_message = "Processing was interrupted repeatedly."
                it.updated_at = now
            if exhausted:
                s.flush()
                for job_id in {it.job_id for it in exhausted}:
                    self._refresh_job(s, job_id)

            item = s.scalars(
                select(JobItem)
                .where(
                    (JobItem.status == "pending")
                    | ((JobItem.status == "running") & (JobItem.lease_expires_at < now))
                )
                .order_by(JobItem.id)
                .limit(1)
                .with_for_update(skip_locked=True)
            ).first()
            if item is None:
                return None
            item.status = "running"
            item.attempts += 1
            item.lease_expires_at = now + timedelta(seconds=self.settings.job_lease_s)
            item.updated_at = now
            job = s.get(Job, item.job_id)
            if job is not None and job.status == "queued":
                job.status = "running"
                job.started_at = now
            return ClaimedItem(
                id=item.id,
                job_id=item.job_id,
                source_path=item.source_path,
                delete_source=item.delete_source,
                display_name=item.display_name,
                image_id=item.image_id,
                attempts=item.attempts,
            )

    def finish(
        self,
        item_id: int,
        status: str,
        *,
        face_count: int = 0,
        image_id: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        with self.db.write() as s:
            item = s.get(JobItem, item_id)
            if item is None:
                return
            item.status = status
            item.face_count = face_count
            item.image_id = image_id or item.image_id
            item.error_code = error_code
            item.error_message = (error_message or "")[:255] or None
            item.lease_expires_at = None
            item.updated_at = utcnow()
            s.flush()
            self._refresh_job(s, item.job_id)

    def retry_or_fail(self, item_id: int, error_code: str, message: str) -> bool:
        """Transient failure: requeue until attempts are exhausted. Returns True if requeued."""
        with self.db.write() as s:
            item = s.get(JobItem, item_id)
            if item is None:
                return False
            requeue = item.attempts < self.settings.job_max_attempts
            item.status = "pending" if requeue else "failed"
            item.error_code = error_code
            item.error_message = message[:255]
            item.lease_expires_at = None
            item.updated_at = utcnow()
            s.flush()
            self._refresh_job(s, item.job_id)
            return requeue

    def release(self, item_id: int) -> None:
        """Hand an interrupted item back without consuming an attempt."""
        with self.db.write() as s:
            item = s.get(JobItem, item_id)
            if item is not None and item.status == "running":
                item.status = "pending"
                item.attempts = max(0, item.attempts - 1)
                item.lease_expires_at = None

    def _refresh_job(self, s: Session, job_id: str) -> None:
        job = s.get(Job, job_id)
        if job is None:
            return
        counts = dict(
            s.execute(
                select(JobItem.status, func.count())
                .where(JobItem.job_id == job_id)
                .group_by(JobItem.status)
            ).all()
        )
        open_items = counts.get("pending", 0) + counts.get("running", 0)
        if open_items:
            if job.status in ("completed", "completed_with_errors", "failed"):
                job.status = "running"
                job.finished_at = None
            return
        failed = counts.get("failed", 0)
        succeeded = sum(counts.get(k, 0) for k in ("done", "no_face", "duplicate"))
        if job.cancel_requested:
            job.status = "cancelled"
        elif failed == 0:
            job.status = "completed"
        elif succeeded == 0:
            job.status = "failed"
        else:
            job.status = "completed_with_errors"
        job.finished_at = job.finished_at or utcnow()

    def queue_depth(self) -> int:
        with self.db.read() as s:
            return int(
                s.scalar(
                    select(func.count())
                    .select_from(JobItem)
                    .where(JobItem.status.in_(("pending", "running")))
                )
                or 0
            )

    def failed_items_total(self) -> int:
        with self.db.read() as s:
            return int(
                s.scalar(
                    select(func.count()).select_from(JobItem).where(JobItem.status == "failed")
                )
                or 0
            )

    def cleanup_staging_for_terminal(self) -> None:
        """Drop staged uploads whose item is already terminal (e.g. after a crash)."""
        with self.db.read() as s:
            rows = s.execute(
                select(JobItem.source_path).where(
                    JobItem.delete_source.is_(True),
                    JobItem.status.in_(TERMINAL_ITEM),
                    JobItem.source_path.is_not(None),
                )
            ).all()
        for (path,) in rows:
            if path:
                Path(path).unlink(missing_ok=True)


class Worker:
    """Background ingestion worker: N threads claim items from the persistent SQLite queue."""

    def __init__(
        self,
        settings: Settings,
        jobs: JobService,
        ingest: IngestService,
        store: FileStore,
        concurrency: int | None = None,
    ) -> None:
        self.settings = settings
        self.jobs = jobs
        self.ingest = ingest
        self.store = store
        self.concurrency = concurrency or settings.worker_concurrency
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    # --- single item -----------------------------------------------------------------------

    def handle(self, item: ClaimedItem) -> None:
        keep_source = False
        try:
            if item.source_path is not None:
                path = Path(item.source_path)
                try:
                    size = path.stat().st_size
                except FileNotFoundError:
                    raise NotFound("Source file is missing.", code="source_missing") from None
                if size > self.settings.max_upload_bytes:
                    raise FileTooLarge(
                        f"File exceeds the {self.settings.max_upload_bytes} byte limit."
                    )
                outcome = self.ingest.ingest_bytes(
                    path.read_bytes(), display_name=item.display_name
                )
            elif item.image_id is not None:
                outcome = self.ingest.reprocess_image(item.image_id)
            else:
                raise NotFound("Job item has no source.", code="no_source")
            status = {"processed": "done", "no_face": "no_face", "duplicate": "duplicate"}[
                outcome.status
            ]
            self.jobs.finish(
                item.id,
                status,
                face_count=0 if outcome.status == "duplicate" else outcome.face_count,
                image_id=outcome.image_id,
            )
        except ApiError as exc:
            permanent = exc.code in PERMANENT_INGEST_CODES or isinstance(exc, NotFound)
            if permanent:
                self.jobs.finish(item.id, "failed", error_code=exc.code, error_message=exc.message)
            else:
                keep_source = self.jobs.retry_or_fail(item.id, exc.code, exc.message)
        except Exception as exc:  # unexpected: treated as transient, retried with bounded attempts
            logger.error("Unexpected error handling job item (%s)", type(exc).__name__)
            keep_source = self.jobs.retry_or_fail(
                item.id, "processing_failed", "Processing failed."
            )
        finally:
            if item.delete_source and item.source_path and not keep_source:
                Path(item.source_path).unlink(missing_ok=True)

    # --- loops -----------------------------------------------------------------------------

    def run_until_empty(self) -> int:
        """Process the queue on the calling thread until nothing is claimable (CLI / tests)."""
        n = 0
        while not self._stop.is_set():
            item = retry_locked(self.jobs.claim_next)
            if item is None:
                return n
            self.handle(item)
            n += 1
        return n

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = retry_locked(self.jobs.claim_next)
            except Exception as exc:
                logger.error("Queue claim failed (%s)", type(exc).__name__)
                self._stop.wait(self.settings.worker_poll_s * 2)
                continue
            if item is None:
                self._stop.wait(self.settings.worker_poll_s)
                continue
            self.handle(item)

    def _heartbeat(self) -> None:
        path = self.settings.heartbeat_file
        while not self._stop.is_set():
            with contextlib.suppress(OSError):
                path.write_text(str(time.time()))
            self._stop.wait(5.0)

    def start(self) -> None:
        self._stop.clear()
        hb = threading.Thread(target=self._heartbeat, name="worker-heartbeat", daemon=True)
        hb.start()
        self._threads = [hb]
        for i in range(self.concurrency):
            t = threading.Thread(target=self._loop, name=f"worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self, timeout: float = 60.0) -> None:
        """Graceful: in-flight items finish; unclaimed/expired work is picked up after restart."""
        self._stop.set()
        deadline = time.monotonic() + timeout
        for t in self._threads:
            t.join(max(0.0, deadline - time.monotonic()))

    def join(self) -> None:
        while any(t.is_alive() for t in self._threads[1:]):
            time.sleep(0.5)


def heartbeat_age(settings: Settings) -> float | None:
    try:
        return time.time() - float(settings.heartbeat_file.read_text())
    except (OSError, ValueError):
        return None


def import_directory_items(
    root: Path,
    *,
    recursive: bool,
    limit: int | None = None,
    extensions: frozenset[str] | None = None,
) -> list[ItemSpec]:
    """Collect candidate files under root (symlinks are not followed)."""
    exts = extensions or frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})
    items: list[ItemSpec] = []
    walker = os.walk(root, followlinks=False) if recursive else [(str(root), [], os.listdir(root))]
    for dirpath, dirnames, filenames in walker:
        dirnames.sort()
        for name in sorted(filenames):
            p = Path(dirpath) / name
            if p.suffix.lower() in exts and p.is_file() and not p.is_symlink():
                items.append(ItemSpec(source_path=str(p.resolve()), display_name=name))
                if limit and len(items) >= limit:
                    return items
    return items
