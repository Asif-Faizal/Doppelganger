from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, Query, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from ..container import Container
from ..errors import ApiError, InvalidRequest, NotFound, ServiceUnavailable
from ..imaging import FORMAT_MEDIA_TYPES, decode_image, inspect_image
from ..jobs import ItemSpec, import_directory_items
from ..search import SearchMode, SearchParams
from ..security import Principal, Scope
from .deps import Guard, get_container, read_upload, search_response
from .schemas import (
    DeleteResponse,
    ErrorResponse,
    ImportAccepted,
    JobResponse,
    ReadyResponse,
    SearchResponse,
)

router = APIRouter()

_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse} for code in (400, 401, 403, 404, 413, 415, 422, 429, 503)
}

SearchGuard = Depends(Guard(Scope.SEARCH, heavy=True))
AdminGuard = Depends(Guard(Scope.ADMIN))
AdminHeavyGuard = Depends(Guard(Scope.ADMIN, heavy=True))
C = Annotated[Container, Depends(get_container)]


def parse_mode(value: str) -> SearchMode:
    try:
        return SearchMode(value.strip().upper())
    except ValueError:
        raise InvalidRequest(
            "mode must be LOOKALIKE or SAME_PERSON.",
            code="invalid_mode",
            details={"allowed": [m.value for m in SearchMode]},
        ) from None


def check_uuid(value: str, what: str = "resource") -> str:
    if not _UUID.match(value):
        raise NotFound(f"{what} not found.")
    return value


@router.post(
    "/v1/faces/search",
    response_model=SearchResponse,
    tags=["search"],
    summary="Search the gallery with a selfie",
    description=(
        "Detects the face in the uploaded image and returns ranked gallery faces.\n\n"
        "* `LOOKALIKE` (default): visual-resemblance ranking; results may be different people.\n"
        "* `SAME_PERSON`: only candidates above a similarity threshold; may return none.\n\n"
        "Scores are raw model similarities, not probabilities. If the selfie contains several "
        "faces of similar size the request fails with `multiple_faces_ambiguous`; resubmit with "
        "`face_index`."
    ),
    responses=_ERRORS,
)
async def search_faces(
    c: C,
    file: Annotated[UploadFile, File(description="Selfie image (JPEG/PNG/WebP/BMP/TIFF).")],
    _p: Annotated[Principal, SearchGuard],
    mode: Annotated[str, Form()] = "LOOKALIKE",
    max_results: Annotated[int | None, Form()] = None,
    offset: Annotated[int, Form(ge=0, le=100_000)] = 0,
    min_similarity: Annotated[float | None, Form(ge=-1.0, le=1.0)] = None,
    face_index: Annotated[int | None, Form(ge=0, le=1000)] = None,
    dedupe_by_image: Annotated[bool, Form(description="Keep only the best face per photo.")] = True,
) -> SearchResponse:
    params = SearchParams(
        mode=parse_mode(mode),
        max_results=_limit(c, max_results),
        offset=offset,
        min_similarity=min_similarity,
        face_index=face_index,
        dedupe_by_image=dedupe_by_image,
    )
    return await _do_search(c, file, params)


def _limit(c: Container, value: int | None) -> int:
    if value is None:
        return c.settings.default_max_results
    if not 1 <= value <= c.settings.max_results_cap:
        raise InvalidRequest(
            f"max_results must be between 1 and {c.settings.max_results_cap}.",
            code="invalid_max_results",
        )
    return value


async def _do_search(c: Container, file: UploadFile, params: SearchParams) -> SearchResponse:
    _, _, _, search = c.require_ready()
    data = await read_upload(file, c.settings.max_upload_bytes)
    with c.metrics.timer("search_total"):
        decoded = await run_in_threadpool(decode_image, data, c.settings)
        out = await run_in_threadpool(search.search_image, decoded.bgr, params)
    c.metrics.inc("search_requests")
    return search_response(c, out)


@router.post(
    "/v1/gallery/import",
    status_code=202,
    response_model=ImportAccepted,
    tags=["gallery"],
    summary="Queue photos for background indexing",
    description=(
        "Either upload `files` (multipart) or give `directory`, a path relative to the server's "
        "configured import root. Returns immediately with a job id; poll `GET /v1/jobs/{job_id}`. "
        "Identical files (sha256) are skipped; every eligible face in each photo is indexed. "
        "Only import photos you are authorised to process."
    ),
    responses=_ERRORS,
)
async def import_gallery(
    c: C,
    response: Response,
    _p: Annotated[Principal, AdminHeavyGuard],
    files: Annotated[list[UploadFile] | None, File()] = None,
    directory: Annotated[str | None, Form()] = None,
    recursive: Annotated[bool, Form()] = False,
) -> ImportAccepted:
    pool, *_ = c.require_ready()
    if bool(files) == bool(directory):
        raise InvalidRequest("Provide exactly one of `files` or `directory`.")
    s = c.settings
    items: list[ItemSpec] = []
    rejected = 0
    if files:
        if len(files) > s.max_import_files:
            raise InvalidRequest(
                f"At most {s.max_import_files} files per request.", code="too_many_files"
            )
        for f in files:
            name = f.filename or "upload"
            try:
                data = await read_upload(f, s.max_upload_bytes)
                info = await run_in_threadpool(inspect_image, data, s)
                existing = await run_in_threadpool(c.repo.get_image_by_sha, info.sha256)
                done = (
                    await run_in_threadpool(
                        c.repo.processed_face_count, existing.id, pool.info.model_id
                    )
                    if existing
                    else None
                )
                if existing is not None and done is not None:
                    items.append(
                        ItemSpec(display_name=name, image_id=existing.id, preset_status="duplicate")
                    )
                    continue
                staged = await run_in_threadpool(c.store.stage, data)
                items.append(
                    ItemSpec(source_path=str(staged), delete_source=True, display_name=name)
                )
            except ApiError as exc:
                items.append(
                    ItemSpec(
                        display_name=name,
                        preset_status="failed",
                        preset_error_code=exc.code,
                        preset_error_message=exc.message,
                    )
                )
                rejected += 1
    else:
        assert directory is not None
        root = _import_root(c)
        target = _resolve_under(root, directory)
        items = await run_in_threadpool(
            import_directory_items, target, recursive=recursive, limit=100_000
        )
        if not items:
            raise InvalidRequest("No image files found in that directory.", code="empty_directory")
    job_id = await run_in_threadpool(c.jobs.create_job, "import", pool.info.model_id, items)
    c.metrics.inc("import_jobs")
    response.headers["Location"] = f"/v1/jobs/{job_id}"
    job = await run_in_threadpool(c.jobs.get_job, job_id)
    assert job is not None
    return ImportAccepted(
        job_id=job_id,
        status=job["status"],
        total=len(items),
        rejected_at_intake=rejected,
        status_url=f"/v1/jobs/{job_id}",
    )


def _import_root(c: Container) -> Path:
    if c.settings.import_root is None:
        raise ServiceUnavailable(
            "Directory import is disabled; set DOPPEL_IMPORT_ROOT or upload files.",
            code="directory_import_disabled",
        )
    return c.settings.import_root.resolve()


def _resolve_under(root: Path, relative: str) -> Path:
    if "\x00" in relative or Path(relative).is_absolute():
        raise InvalidRequest("directory must be a relative path.", code="invalid_directory")
    target = (root / relative).resolve()
    if not target.is_relative_to(root) or not target.is_dir():
        raise InvalidRequest(
            "directory is not inside the import root or does not exist.", code="invalid_directory"
        )
    return target


@router.get(
    "/v1/jobs/{job_id}",
    response_model=JobResponse,
    tags=["gallery"],
    summary="Job status, progress and failure counts",
    responses=_ERRORS,
)
def get_job(job_id: str, c: C, _p: Annotated[Principal, AdminGuard]) -> JobResponse:
    job = c.jobs.get_job(check_uuid(job_id, "Job"))
    if job is None:
        raise NotFound("Job not found.")
    return JobResponse(**job)


@router.post(
    "/v1/jobs/{job_id}/cancel",
    response_model=JobResponse,
    tags=["gallery"],
    summary="Cancel the not-yet-started items of a job",
    responses=_ERRORS,
)
def cancel_job(job_id: str, c: C, _p: Annotated[Principal, AdminGuard]) -> JobResponse:
    if not c.jobs.cancel(check_uuid(job_id, "Job")):
        raise NotFound("Job not found.")
    job = c.jobs.get_job(job_id)
    assert job is not None
    return JobResponse(**job)


@router.delete(
    "/v1/images/{image_id}",
    response_model=DeleteResponse,
    tags=["gallery"],
    summary="Delete an image, its crops, embeddings and index entries",
    responses=_ERRORS,
)
def delete_image(image_id: str, c: C, _p: Annotated[Principal, AdminGuard]) -> DeleteResponse:
    check_uuid(image_id, "Image")
    result = c.repo.delete_image(image_id)
    if result is None:
        raise NotFound("Image not found.")
    record, crops, n_faces = result
    c.store.delete_original(record.storage_name)
    for name in crops:
        c.store.delete_crop(name)
    if c.index is not None:
        c.index.sync(force=True)
    c.metrics.inc("images_deleted")
    return DeleteResponse(image_id=image_id, deleted=True, faces_removed=n_faces)


@router.get(
    "/v1/images/{image_id}/file",
    tags=["files"],
    summary="Original image (signed URL)",
    responses={**_ERRORS, 200: {"content": {"image/*": {}}}},
)
def get_image_file(
    image_id: str,
    c: C,
    exp: Annotated[int, Query()],
    sig: Annotated[str, Query(max_length=128)],
    request: Request,
) -> FileResponse:
    c.ip_limiter.check(request.client.host if request.client else "unknown")
    check_uuid(image_id, "Image")
    c.signer.verify("image", image_id, exp, sig)
    record = c.repo.get_image(image_id)
    if record is None:
        raise NotFound("Image not found.")
    path = c.store.original_path(record.storage_name)
    if not path.is_file():
        raise NotFound("Image not found.")
    return FileResponse(
        path,
        media_type=FORMAT_MEDIA_TYPES.get(record.format, "application/octet-stream"),
        headers={"Cache-Control": "private, max-age=300", "Content-Disposition": "inline"},
    )


@router.get(
    "/v1/faces/{face_id}/thumbnail",
    tags=["files"],
    summary="Face crop (signed URL)",
    responses={**_ERRORS, 200: {"content": {"image/jpeg": {}}}},
)
def get_face_thumbnail(
    face_id: str,
    c: C,
    exp: Annotated[int, Query()],
    sig: Annotated[str, Query(max_length=128)],
    request: Request,
) -> FileResponse:
    c.ip_limiter.check(request.client.host if request.client else "unknown")
    check_uuid(face_id, "Face")
    c.signer.verify("thumb", face_id, exp, sig)
    face = c.repo.get_face(face_id)
    if face is None or not face.crop_name:
        raise NotFound("Face not found.")
    path = c.store.crop_path(face.crop_name)
    if not path.is_file():
        raise NotFound("Face not found.")
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=300", "Content-Disposition": "inline"},
    )


@router.get(
    "/ready",
    response_model=ReadyResponse,
    tags=["ops"],
    summary="Readiness: database, schema, models and index",
    responses={503: {"model": ReadyResponse}},
)
def ready(c: C, response: Response) -> ReadyResponse:
    ok, checks = c.readiness()
    if not ok:
        response.status_code = 503
    return ReadyResponse(status="ready" if ok else "not_ready", checks=checks)


@router.get("/v1/admin/metrics", tags=["ops"], summary="Operational metrics (no biometric data)")
def metrics(c: C, _p: Annotated[Principal, AdminGuard]) -> dict[str, Any]:
    snap = c.metrics.snapshot()
    snap["queue_depth"] = c.jobs.queue_depth()
    snap["failed_job_items_total"] = c.jobs.failed_items_total()
    snap["faces_by_model"] = c.repo.faces_by_model()
    snap["images"] = c.repo.count_images()
    snap["index"] = c.index.health() if c.index else None
    snap["time"] = time.time()
    return snap


@router.get("/v1/admin/index/check", tags=["ops"], summary="Compare the index with the database")
def index_check(c: C, _p: Annotated[Principal, AdminGuard]) -> dict[str, object]:
    _, index, _, _ = c.require_ready()
    return index.check().to_dict()


@router.post("/v1/admin/index/rebuild", tags=["ops"], summary="Rebuild the index from the database")
def index_rebuild(c: C, _p: Annotated[Principal, AdminHeavyGuard]) -> dict[str, object]:
    _, index, _, _ = c.require_ready()
    n = index.rebuild()
    return {"rebuilt": True, "vectors": n}
