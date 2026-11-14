from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .api import legacy, v1
from .config import Settings, get_settings
from .container import Container, build_container
from .errors import ApiError
from .ml.pipeline import PipelineFactory

logger = logging.getLogger("doppelganger")

API_VERSION = "2.0.0"
SEARCH_PATHS = ("/v1/faces/search", "/search-face", "/upload-images")


class _BodyTooLarge(Exception):
    pass


class BodySizeLimitMiddleware:
    """Reject oversized request bodies early, including chunked uploads without Content-Length."""

    def __init__(self, app: ASGIApp, settings: Settings) -> None:
        self.app = app
        self.settings = settings

    def _limit(self, path: str) -> int:
        s = self.settings
        if path in SEARCH_PATHS:
            return s.max_upload_bytes * 2 + 1024 * 1024
        return s.max_request_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = self._limit(scope["path"])
        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > limit:
            await self._reject(scope, receive, send, limit)
            return
        received = 0
        started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise _BodyTooLarge
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLarge:
            if not started:
                await self._reject(scope, receive, send, limit)

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send, limit: int) -> None:
        body = JSONResponse(
            status_code=413,
            content={
                "detail": "Request body too large.",
                "error": {
                    "code": "request_too_large",
                    "message": "Request body too large.",
                    "details": {"max_bytes": limit},
                },
            },
        )
        await body(scope, receive, send)


def _error_json(
    request: Request,
    status: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    rid = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=status,
        content={
            "detail": message,
            "error": {
                "code": code,
                "message": message,
                "details": details or {},
                "request_id": rid,
            },
        },
        headers=headers,
    )


def create_app(
    settings: Settings | None = None, pipeline_factory: PipelineFactory | None = None
) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    for noisy in ("alembic", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container: Container = build_container(settings, pipeline_factory)
        app.state.container = container
        if settings.embedded_worker and container.worker is not None:
            container.worker.start()
        logger.info("Doppelganger %s started", API_VERSION)
        try:
            yield
        finally:
            container.close()

    app = FastAPI(
        title="Doppelganger Face Matching API",
        description=(
            "Self-hosted private-gallery face search. **LOOKALIKE** ranks visually similar faces "
            "(not identity). **SAME_PERSON** returns threshold-passing candidates (not proof of "
            "identity). All processing and storage is local."
        ),
        version=API_VERSION,
        lifespan=lifespan,
    )

    if settings.cors_origin_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origin_list,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Authorization", "Content-Type"],
            allow_credentials=False,
        )
    app.add_middleware(BodySizeLimitMiddleware, settings=settings)

    @app.middleware("http")
    async def request_context(request: Request, call_next: Any) -> Any:
        rid = request.headers.get("x-request-id", "")
        request.state.request_id = rid if rid.isalnum() and len(rid) <= 64 else uuid.uuid4().hex
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers.setdefault("Cache-Control", "no-store")
        logger.info(
            "%s %s -> %d (%.1f ms) rid=%s",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - t0) * 1000,
            request.state.request_id,
        )
        return response

    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
        status = exc.status_code
        if request.url.path in legacy.LEGACY_PATHS and status == 422:
            status = 400  # original API reported invalid-image problems as 400
        return _error_json(request, status, exc.code, exc.message, exc.details, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = [
            {
                "field": ".".join(str(p) for p in e.get("loc", ()) if p != "body"),
                "problem": e.get("msg", "invalid"),
            }
            for e in exc.errors()
        ]
        return _error_json(
            request, 422, "validation_error", "The request is invalid.", {"fields": fields}
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return _error_json(request, exc.status_code, code, str(exc.detail), None, exc.headers)

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "Unhandled error (%s) rid=%s",
            type(exc).__name__,
            getattr(request.state, "request_id", "-"),
        )
        return _error_json(request, 500, "internal_error", "An internal error occurred.")

    app.include_router(legacy.router)
    app.include_router(v1.router)
    return app


_app: FastAPI | None = None


def __getattr__(name: str) -> FastAPI:
    """Lazy module-level ``app`` so `uvicorn app.main:app` needs no import-time side effects."""
    global _app
    if name == "app":
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(name)
