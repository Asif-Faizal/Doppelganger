from __future__ import annotations

from typing import Any


class ApiError(Exception):
    """Structured, client-safe error. Messages must never contain paths, SQL or stack traces."""

    status_code = 400
    code = "invalid_request"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        self.details = details or {}
        self.headers = headers or {}


class InvalidRequest(ApiError):
    status_code = 400
    code = "invalid_request"


class FileTooLarge(ApiError):
    status_code = 413
    code = "file_too_large"


class UnsupportedImageFormat(ApiError):
    status_code = 415
    code = "unsupported_image_format"


class CorruptImage(ApiError):
    status_code = 422
    code = "corrupt_image"


class InvalidImage(ApiError):
    status_code = 422
    code = "invalid_image"


class NoFaceDetected(ApiError):
    status_code = 422
    code = "no_face_detected"


class FaceQualityInsufficient(ApiError):
    status_code = 422
    code = "face_quality_insufficient"


class MultipleFacesAmbiguous(ApiError):
    status_code = 422
    code = "multiple_faces_ambiguous"


class Unauthorized(ApiError):
    status_code = 401
    code = "unauthorized"


class Forbidden(ApiError):
    status_code = 403
    code = "forbidden"


class NotFound(ApiError):
    status_code = 404
    code = "not_found"


class RateLimited(ApiError):
    status_code = 429
    code = "rate_limited"


class ServiceUnavailable(ApiError):
    status_code = 503
    code = "service_unavailable"


class ModelUnavailable(ServiceUnavailable):
    code = "model_unavailable"


class IndexUnavailable(ServiceUnavailable):
    code = "index_unavailable"


class ProcessingFailed(ApiError):
    status_code = 500
    code = "processing_failed"


PERMANENT_INGEST_CODES = frozenset(
    {
        "corrupt_image",
        "invalid_image",
        "unsupported_image_format",
        "file_too_large",
    }
)
