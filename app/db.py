from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    create_engine,
    event,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


def utcnow() -> datetime:
    """Naive UTC timestamp (SQLite has no timezone type; everything is stored as UTC)."""
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Image(Base):
    __tablename__ = "images"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    storage_name: Mapped[str] = mapped_column(String(64))
    format: Mapped[str] = mapped_column(String(16))
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    size_bytes: Mapped[int] = mapped_column(Integer)
    source_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    legacy_ref: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ImageProcessing(Base):
    """Per-(image, model) processing state. Drives skip-already-processed and reprocessing."""

    __tablename__ = "image_processing"

    image_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("images.id", ondelete="CASCADE"), primary_key=True
    )
    model_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    status: Mapped[str] = mapped_column(String(16))  # processed | no_face
    face_count: Mapped[int] = mapped_column(Integer, default=0)
    rejected: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    processed_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Face(Base):
    __tablename__ = "faces"
    __table_args__ = (
        Index("ix_faces_model_seq", "model_id", "seq"),
        {"sqlite_autoincrement": True},
    )

    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    image_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("images.id", ondelete="CASCADE"), index=True
    )
    model_id: Mapped[str] = mapped_column(String(128))
    face_index: Mapped[int] = mapped_column(Integer)
    bbox_x: Mapped[float] = mapped_column(Float)
    bbox_y: Mapped[float] = mapped_column(Float)
    bbox_w: Mapped[float] = mapped_column(Float)
    bbox_h: Mapped[float] = mapped_column(Float)
    det_score: Mapped[float] = mapped_column(Float)
    quality: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    embedding: Mapped[bytes] = mapped_column(LargeBinary)
    crop_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class IndexEvent(Base):
    """Append-only log of index removals so every process can replay deletions."""

    __tablename__ = "index_events"
    __table_args__ = (
        Index("ix_index_events_model_id", "model_id", "id"),
        {"sqlite_autoincrement": True},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(String(128))
    face_seq: Mapped[int] = mapped_column(Integer)
    op: Mapped[str] = mapped_column(String(16), default="delete")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    kind: Mapped[str] = mapped_column(String(16))  # import | reprocess
    status: Mapped[str] = mapped_column(String(24), index=True)
    model_id: Mapped[str] = mapped_column(String(128))
    total: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)


class JobItem(Base):
    __tablename__ = "job_items"
    __table_args__ = (Index("ix_job_items_claim", "status", "lease_expires_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("jobs.id", ondelete="CASCADE"), index=True
    )
    # pending | running | done | no_face | duplicate | skipped | failed
    status: Mapped[str] = mapped_column(String(16), default="pending")
    source_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    delete_source: Mapped[bool] = mapped_column(Boolean, default=False)
    display_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    image_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    face_count: Mapped[int] = mapped_column(Integer, default=0)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(255), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


def _make_engine(url: str, *, immediate: bool) -> Engine:
    is_sqlite = url.startswith("sqlite")
    kwargs: dict[str, Any] = {}
    if is_sqlite:
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
    engine = create_engine(url, future=True, **kwargs)
    if is_sqlite:

        @event.listens_for(engine, "connect")
        def _on_connect(dbapi_conn: Any, _record: Any) -> None:
            dbapi_conn.isolation_level = None  # transaction control handled in "begin" below
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA busy_timeout=30000")
            cur.close()

        @event.listens_for(engine, "begin")
        def _on_begin(conn: Any) -> None:
            conn.exec_driver_sql("BEGIN IMMEDIATE" if immediate else "BEGIN")

    return engine


class Database:
    """Separate read (deferred) and write (BEGIN IMMEDIATE) engines.

    Taking SQLite's write lock at transaction start avoids read-then-write upgrade deadlocks
    when the API process and the worker process write concurrently, while readers (search) never
    wait on writers in WAL mode.
    """

    def __init__(self, url: str) -> None:
        self.url = url
        if url.startswith("sqlite:///"):
            Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        self.read_engine = _make_engine(url, immediate=False)
        self.write_engine = _make_engine(url, immediate=True)
        self._read = sessionmaker(self.read_engine, expire_on_commit=False)
        self._write = sessionmaker(self.write_engine, expire_on_commit=False)

    @contextlib.contextmanager
    def read(self) -> Iterator[Session]:
        with self._read() as session:
            yield session

    @contextlib.contextmanager
    def write(self) -> Iterator[Session]:
        with self._write() as session:
            try:
                yield session
                session.commit()
            except BaseException:
                session.rollback()
                raise

    def dispose(self) -> None:
        self.read_engine.dispose()
        self.write_engine.dispose()


def retry_locked[T](fn: Callable[[], T], attempts: int = 5, delay: float = 0.2) -> T:
    """Retry on transient SQLite 'database is locked' errors beyond busy_timeout."""
    for i in range(attempts):
        try:
            return fn()
        except OperationalError as exc:
            if "locked" not in str(exc).lower() or i == attempts - 1:
                raise
            time.sleep(delay * (i + 1))
    raise AssertionError("unreachable")
