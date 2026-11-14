from __future__ import annotations

import logging
from dataclasses import dataclass, field

import faiss
from sqlalchemy import text

from . import migrate
from .config import Settings
from .db import Database
from .errors import IndexUnavailable, ModelUnavailable
from .index import IndexManager
from .ingest import IngestService
from .jobs import JobService, Worker, heartbeat_age
from .metrics import Metrics
from .ml.pipeline import PipelineFactory, PipelinePool, default_pipeline_factory
from .repository import SqlGalleryRepository
from .search import SearchService
from .security import Authenticator, RateLimiter, UrlSigner
from .storage import FileStore

logger = logging.getLogger(__name__)


@dataclass
class Container:
    """Wires the application together. Everything is injected, so tests can swap the models."""

    settings: Settings
    db: Database
    repo: SqlGalleryRepository
    store: FileStore
    jobs: JobService
    metrics: Metrics
    auth: Authenticator
    signer: UrlSigner
    ip_limiter: RateLimiter
    default_limiter: RateLimiter
    heavy_limiter: RateLimiter
    pool: PipelinePool | None = None
    index: IndexManager | None = None
    ingest: IngestService | None = None
    search: SearchService | None = None
    worker: Worker | None = None
    model_error: ModelUnavailable | None = None
    extras: dict[str, object] = field(default_factory=dict)

    def require_ready(self) -> tuple[PipelinePool, IndexManager, IngestService, SearchService]:
        if self.pool is None or self.ingest is None or self.search is None:
            raise self.model_error or ModelUnavailable("Face models are not loaded.")
        if self.index is None:
            raise IndexUnavailable("The vector index is not loaded.")
        return self.pool, self.index, self.ingest, self.search

    def readiness(self) -> tuple[bool, dict[str, object]]:
        checks: dict[str, object] = {}
        try:
            with self.db.read() as s:
                s.execute(text("SELECT 1"))
            checks["database"] = "ok"
            rev = migrate.current_revision(self.settings.database_url or "")
            checks["schema"] = "ok" if rev == migrate.head_revision() else "outdated"
        except Exception as exc:
            logger.warning("Readiness: database check failed (%s)", type(exc).__name__)
            checks["database"] = "unavailable"
            checks["schema"] = "unknown"
        if self.pool is not None:
            checks["models"] = {"status": "ok", "model_id": self.pool.info.model_id}
        else:
            code = self.model_error.code if self.model_error else "model_unavailable"
            checks["models"] = {"status": "unavailable", "reason": code}
        if self.index is not None:
            try:
                self.index.sync()
                health = self.index.health()
                checks["index"] = {"status": "ok", **health}
            except Exception as exc:
                logger.warning("Readiness: index check failed (%s)", type(exc).__name__)
                checks["index"] = {"status": "unavailable"}
        else:
            checks["index"] = {"status": "unavailable"}
        age = heartbeat_age(self.settings)
        checks["worker"] = (
            {"status": "unknown"}
            if age is None
            else {"status": "ok" if age < 30 else "stale", "heartbeat_age_s": round(age, 1)}
        )
        ok = (
            checks["database"] == "ok"
            and checks["schema"] == "ok"
            and isinstance(checks["models"], dict)
            and checks["models"]["status"] == "ok"
            and isinstance(checks["index"], dict)
            and checks["index"]["status"] == "ok"
        )
        return bool(ok), checks

    def close(self) -> None:
        if self.worker is not None:
            self.worker.stop()
        if self.index is not None:
            try:
                self.index.persist()
            except Exception as exc:
                logger.error("Could not persist index on shutdown (%s)", type(exc).__name__)
        self.db.dispose()


def build_container(
    settings: Settings,
    pipeline_factory: PipelineFactory | None = None,
) -> Container:
    settings.ensure_dirs()
    faiss.omp_set_num_threads(settings.index_threads)
    if settings.auto_migrate:
        migrate.upgrade(settings.database_url or "")
    db = Database(settings.database_url or "")
    repo = SqlGalleryRepository(db)
    container = Container(
        settings=settings,
        db=db,
        repo=repo,
        store=FileStore(settings),
        jobs=JobService(db, settings),
        metrics=Metrics(),
        auth=Authenticator(settings),
        signer=UrlSigner(settings),
        ip_limiter=RateLimiter(settings.rate_limit_per_minute * 2),
        default_limiter=RateLimiter(settings.rate_limit_per_minute),
        heavy_limiter=RateLimiter(settings.search_rate_limit_per_minute),
    )
    factory = pipeline_factory or default_pipeline_factory
    try:
        container.pool = factory(settings)
    except ModelUnavailable as exc:
        container.model_error = exc
        logger.error("Face models unavailable: %s", exc.message)
        if settings.require_models_at_startup:
            db.dispose()
            raise
    if container.pool is not None:
        index = IndexManager(
            repo,
            container.pool.info,
            settings.index_dir,
            sync_interval_s=settings.index_sync_interval_s,
            persist_every=settings.index_persist_every,
        )
        index.load()
        container.index = index
        container.ingest = IngestService(
            settings, repo, container.store, container.pool, container.metrics
        )
        container.search = SearchService(settings, container.pool, index, repo, container.metrics)
        container.worker = Worker(settings, container.jobs, container.ingest, container.store)
    return container
