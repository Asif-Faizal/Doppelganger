from __future__ import annotations

from fastapi import Request, UploadFile

from ..container import Container
from ..errors import FileTooLarge, InvalidRequest
from ..search import SearchOutcome
from ..security import Principal, Scope
from .schemas import BBox, CalibrationInfo, MatchOut, QueryFace, SearchResponse

SCORE_SEMANTICS = {
    "cosine_similarity": "Raw model similarity (cosine of L2-normalised embeddings). "
    "Not a probability.",
    "cosine_distance": "1 - cosine_similarity.",
    "ranking_score": "Used only to order results; equals cosine_similarity.",
    "calibrated_confidence": "Estimated P(same person) from a calibration fitted on labelled data; "
    "null unless a calibration file for this model is installed.",
}


def get_container(request: Request) -> Container:
    c: Container = request.app.state.container
    return c


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


class Guard:
    """Rate-limit (per IP, then per principal) and authorise a request."""

    def __init__(self, scope: Scope, *, heavy: bool = False) -> None:
        self.scope = scope
        self.heavy = heavy

    def __call__(self, request: Request) -> Principal:
        c = get_container(request)
        c.ip_limiter.check(client_ip(request))
        principal = c.auth.authenticate(request, self.scope)
        limiter = c.heavy_limiter if self.heavy else c.default_limiter
        limiter.check(principal.id if principal.id != "anonymous" else client_ip(request))
        return principal


async def read_upload(file: UploadFile, limit: int) -> bytes:
    """Read an upload in chunks, refusing anything above the configured size."""
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(1024 * 1024):
        total += len(chunk)
        if total > limit:
            raise FileTooLarge(
                f"File exceeds the {limit} byte limit.", details={"max_bytes": limit}
            )
        chunks.append(chunk)
    if total == 0:
        raise InvalidRequest("The uploaded file is empty.", code="empty_file")
    return b"".join(chunks)


def image_url(c: Container, image_id: str) -> str:
    exp, sig = c.signer.sign("image", image_id)
    return f"{c.settings.public_base_url}/v1/images/{image_id}/file?exp={exp}&sig={sig}"


def thumbnail_url(c: Container, face_id: str) -> str:
    exp, sig = c.signer.sign("thumb", face_id)
    return f"{c.settings.public_base_url}/v1/faces/{face_id}/thumbnail?exp={exp}&sig={sig}"


def search_response(c: Container, out: SearchOutcome) -> SearchResponse:
    x, y, w, h = out.query_bbox
    return SearchResponse(
        mode=out.mode.value,
        model_id=out.model_id,
        status=out.status,
        message=out.message,
        notice=out.notice,
        score_semantics=SCORE_SEMANTICS,
        threshold=out.threshold,
        threshold_source=out.threshold_source,
        query_face=QueryFace(
            bbox=BBox(x=x, y=y, width=w, height=h),
            detection_score=out.query_quality.get("detection_score", 0.0),
            quality=out.query_quality,
            faces_detected=out.detected_faces,
        ),
        gallery_faces=out.gallery_faces,
        matches=[
            MatchOut(
                rank=m.rank,
                image_id=m.hit.image_id,
                face_id=m.hit.face_id,
                cosine_similarity=round(m.cosine_similarity, 6),
                cosine_distance=round(m.cosine_distance, 6),
                ranking_score=round(m.cosine_similarity, 6),
                calibrated_confidence=None
                if m.calibrated_confidence is None
                else round(m.calibrated_confidence, 6),
                same_person_candidate=m.passes_same_person_threshold,
                bbox=BBox(
                    x=m.hit.bbox[0], y=m.hit.bbox[1], width=m.hit.bbox[2], height=m.hit.bbox[3]
                ),
                image_width=m.hit.image_width,
                image_height=m.hit.image_height,
                image_url=image_url(c, m.hit.image_id),
                thumbnail_url=thumbnail_url(c, m.hit.face_id) if m.hit.crop_name else None,
                mode=out.mode.value,
                model_id=out.model_id,
            )
            for m in out.matches
        ],
        next_offset=out.next_offset,
        calibration=None
        if out.calibration is None
        else CalibrationInfo(dataset=out.calibration.dataset, n_pairs=out.calibration.n_pairs),
    )
