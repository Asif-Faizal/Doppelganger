"""Backward-compatible routes of the original Doppelganger API.

Paths and top-level response keys are preserved. Differences that follow from fixing the old
design (documented in README "Legacy API compatibility"): internal file paths are replaced by
signed URLs, `threshold` is a cosine *distance* instead of a dlib euclidean distance, requests
need a bearer token unless DOPPEL_AUTH_DISABLED=true, and `confidence` labels were removed
because they were never calibrated.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool

from ..container import Container
from ..errors import ApiError, InvalidRequest
from ..imaging import decode_image
from ..search import SearchMode, SearchParams
from ..security import Principal, Scope
from .deps import Guard, get_container, image_url, read_upload, thumbnail_url

router = APIRouter(tags=["legacy"])
LEGACY_PATHS = frozenset({"/upload-images", "/search-face", "/stats", "/health", "/"})

C = Annotated[Container, Depends(get_container)]


@router.get("/", summary="Service banner")
def root() -> dict[str, str]:
    return {"message": "Face Recognition API is running!", "version": "2.0.0"}


@router.get("/health", summary="Liveness (process is up). Use /ready for dependency readiness.")
def health(c: C) -> dict[str, Any]:
    return {
        "status": "healthy",
        "timestamp": datetime.now(UTC).replace(tzinfo=None).isoformat(),
        "face_service": "ready" if c.pool is not None else "unavailable",
    }


@router.post("/upload-images", summary="Legacy: upload 1-2 images and index their faces")
async def upload_images(
    c: C,
    _p: Annotated[Principal, Depends(Guard(Scope.ADMIN, heavy=True))],
    files: Annotated[list[UploadFile], File()],
) -> dict[str, Any]:
    if not 1 <= len(files) <= 2:
        raise InvalidRequest("Please upload 1 or 2 images")
    _, _, ingest, _ = c.require_ready()
    results: list[dict[str, Any]] = []
    for f in files:
        name = f.filename or ""
        if not name:
            raise InvalidRequest("No filename provided")
        try:
            data = await read_upload(f, c.settings.max_upload_bytes)
            out = await run_in_threadpool(ingest.ingest_bytes, data, display_name=name)
        except ApiError as exc:
            results.append(
                {"filename": name, "status": "error", "code": exc.code, "message": exc.message}
            )
            continue
        if out.face_count == 0 and out.status != "duplicate":
            results.append(
                {
                    "filename": name,
                    "status": "error",
                    "code": "no_face_detected",
                    "message": "No face detected in image",
                    "image_id": out.image_id,
                    "rejected": out.rejected,
                }
            )
            continue
        results.append(
            {
                "filename": name,
                "status": "success",
                "duplicate": out.status == "duplicate",
                "image_id": out.image_id,
                "image_url": image_url(c, out.image_id),
                "embedding_id": out.face_seqs[0] if out.face_seqs else None,
                "embedding_shape": [c.pool.info.dim if c.pool else 0],
                "faces": [{"face_id": fid} for fid in out.face_ids],
                "face_count": out.face_count,
                "model_id": ingest.model_id,
            }
        )
    return {"message": f"Processed {len(files)} images", "results": results}


@router.post("/search-face", summary="Legacy: search for matching faces (SAME_PERSON semantics)")
async def search_face(
    c: C,
    _p: Annotated[Principal, Depends(Guard(Scope.SEARCH, heavy=True))],
    file: Annotated[UploadFile, File()],
    threshold: Annotated[float | None, Form(ge=0.0, le=2.0)] = None,
    max_results: Annotated[int, Form(ge=1)] = 10,
) -> dict[str, Any]:
    _, _, _, search = c.require_ready()
    if not file.filename:
        raise InvalidRequest("No filename provided")
    max_results = min(max_results, c.settings.max_results_cap)
    min_similarity = None if threshold is None else 1.0 - threshold
    params = SearchParams(
        mode=SearchMode.SAME_PERSON, max_results=max_results, min_similarity=min_similarity
    )
    data = await read_upload(file, c.settings.max_upload_bytes)
    decoded = await run_in_threadpool(decode_image, data, c.settings)
    out = await run_in_threadpool(search.search_image, decoded.bgr, params)
    c.metrics.inc("search_requests")
    matches = [
        {
            "image_id": m.hit.image_id,
            "face_id": m.hit.face_id,
            "image_url": image_url(c, m.hit.image_id),
            "thumbnail_url": thumbnail_url(c, m.hit.face_id) if m.hit.crop_name else None,
            "distance": round(m.cosine_distance, 4),
            "similarity_score": round(max(0.0, m.cosine_similarity), 4),
        }
        for m in out.matches
    ]
    return {
        "message": f"Found {len(matches)} matches" if matches else (out.message or "No matches"),
        "query_filename": file.filename,
        "threshold": None if out.threshold is None else round(1.0 - out.threshold, 4),
        "total_faces_searched": out.gallery_faces,
        "matches": matches,
        "mode": out.mode.value,
        "model_id": out.model_id,
        "notice": out.notice,
    }


@router.get("/stats", summary="Legacy: database statistics (no filesystem paths)")
def stats(c: C, _p: Annotated[Principal, Depends(Guard(Scope.SEARCH))]) -> dict[str, Any]:
    model_id = c.pool.info.model_id if c.pool else None
    return {
        "total_faces_stored": c.repo.count_faces(model_id) if model_id else 0,
        "total_images": c.repo.count_images(),
        "active_model_id": model_id,
        "faces_by_model": c.repo.faces_by_model(),
    }
