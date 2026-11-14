from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class BBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None


class ErrorResponse(BaseModel):
    detail: str
    error: ErrorBody


class MatchOut(BaseModel):
    rank: int = Field(description="1-based position in the ranking (best first).")
    image_id: str
    face_id: str
    cosine_similarity: float = Field(
        description="Raw cosine similarity between L2-normalised embeddings (-1..1). "
        "Model-specific; NOT a probability."
    )
    cosine_distance: float = Field(description="1 - cosine_similarity.")
    ranking_score: float = Field(
        description="Monotone ordering score (equals cosine_similarity); only for ranking."
    )
    calibrated_confidence: float | None = Field(
        default=None,
        description="Calibrated P(same person) from a calibration file produced on a labelled "
        "dataset; null when no calibration is installed.",
    )
    same_person_candidate: bool | None = Field(
        default=None, description="SAME_PERSON mode only: similarity passed the threshold."
    )
    bbox: BBox
    image_width: int
    image_height: int
    image_url: str = Field(description="Signed, expiring URL for the original image.")
    thumbnail_url: str | None = Field(description="Signed, expiring URL for the face crop.")
    mode: str
    model_id: str


class QueryFace(BaseModel):
    bbox: BBox
    detection_score: float
    quality: dict[str, float]
    faces_detected: int


class CalibrationInfo(BaseModel):
    dataset: str
    n_pairs: int


class SearchResponse(BaseModel):
    mode: Literal["LOOKALIKE", "SAME_PERSON"]
    model_id: str
    status: Literal["ok", "no_reliable_match", "empty_gallery"]
    message: str | None = None
    notice: str
    score_semantics: dict[str, str]
    threshold: float | None = Field(
        default=None, description="Minimum cosine similarity applied (null = none)."
    )
    threshold_source: str | None = None
    query_face: QueryFace
    gallery_faces: int
    matches: list[MatchOut]
    next_offset: int | None = Field(
        default=None, description="Pass as `offset` to fetch the next page; null when exhausted."
    )
    calibration: CalibrationInfo | None = None


class JobCounts(BaseModel):
    pending: int
    running: int
    succeeded: int
    no_face: int
    duplicate: int
    skipped: int
    failed: int


class JobError(BaseModel):
    name: str | None
    code: str | None
    message: str | None
    attempts: int


class JobResponse(BaseModel):
    job_id: str
    kind: str
    status: str
    model_id: str
    total: int
    progress: float
    counts: JobCounts
    faces_indexed: int
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    errors: list[JobError]


class ImportAccepted(BaseModel):
    job_id: str
    status: str
    total: int
    rejected_at_intake: int
    status_url: str


class DeleteResponse(BaseModel):
    image_id: str
    deleted: bool
    faces_removed: int


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, Any]
