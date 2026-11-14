from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from sqlalchemy.exc import IntegrityError

from .config import Settings
from .db import utcnow
from .errors import ApiError, NotFound, ProcessingFailed
from .imaging import FORMAT_EXTENSIONS, DecodedImage, decode_image, inspect_image, make_face_crop
from .metrics import Metrics
from .ml.pipeline import PipelinePool
from .repository import ImageRecord, NewFace, SqlGalleryRepository
from .storage import FileStore, new_id

logger = logging.getLogger(__name__)

_CTRL = re.compile(r"[\x00-\x1f\x7f]")


def clean_display_name(name: str | None) -> str | None:
    """Untrusted client filename -> harmless label. Never used as a storage destination."""
    if not name:
        return None
    base = re.split(r"[\\/]", name)[-1]
    base = _CTRL.sub("", base).strip()
    return base[:255] or None


@dataclass
class IngestOutcome:
    status: str  # processed | no_face | duplicate
    image_id: str
    face_count: int
    created: bool
    face_ids: list[str] = field(default_factory=list)
    face_seqs: list[int] = field(default_factory=list)
    rejected: dict[str, int] = field(default_factory=dict)


class IngestService:
    def __init__(
        self,
        settings: Settings,
        repo: SqlGalleryRepository,
        store: FileStore,
        pool: PipelinePool,
        metrics: Metrics,
    ) -> None:
        self.settings = settings
        self.repo = repo
        self.store = store
        self.pool = pool
        self.metrics = metrics
        self.model_id = pool.info.model_id

    def ingest_bytes(
        self,
        data: bytes,
        *,
        display_name: str | None = None,
        legacy_ref: str | None = None,
        force: bool = False,
    ) -> IngestOutcome:
        """Validate, de-duplicate by sha256, store the original, index every eligible face.

        Idempotent: re-ingesting the same bytes for the same model is a no-op ('duplicate')."""
        info = inspect_image(data, self.settings)
        existing = self.repo.get_image_by_sha(info.sha256)
        if existing is not None:
            done = self.repo.processed_face_count(existing.id, self.model_id)
            if done is not None and not force:
                self.metrics.inc("ingest_duplicate")
                return IngestOutcome("duplicate", existing.id, done, created=False)
            return self._process(existing, decode_image(data, self.settings), created=False)

        decoded = decode_image(data, self.settings)
        image_id = new_id()
        storage_name = self.store.save_original(image_id, FORMAT_EXTENSIONS[info.format], data)
        record = ImageRecord(
            id=image_id,
            sha256=info.sha256,
            storage_name=storage_name,
            format=info.format,
            width=decoded.info.width,
            height=decoded.info.height,
            size_bytes=info.size_bytes,
            source_name=clean_display_name(display_name),
            legacy_ref=legacy_ref,
            created_at=utcnow(),
        )
        try:
            self.repo.create_image(record)
        except IntegrityError:
            self.store.delete_original(storage_name)
            winner = self.repo.get_image_by_sha(info.sha256)
            if winner is None:
                raise ProcessingFailed("Could not record the image.") from None
            done = self.repo.processed_face_count(winner.id, self.model_id)
            if done is not None:
                return IngestOutcome("duplicate", winner.id, done, created=False)
            return self._process(winner, decoded, created=False)
        return self._process(record, decoded, created=True)

    def reprocess_image(self, image_id: str) -> IngestOutcome:
        """Recompute faces for the active model from the stored original (never reuses vectors)."""
        record = self.repo.get_image(image_id)
        if record is None:
            raise NotFound("Image not found.")
        try:
            data = self.store.original_path(record.storage_name).read_bytes()
        except FileNotFoundError:
            raise ProcessingFailed(
                "The stored original is missing.", code="original_missing"
            ) from None
        return self._process(record, decode_image(data, self.settings), created=False)

    def _process(
        self, record: ImageRecord, decoded: DecodedImage, *, created: bool
    ) -> IngestOutcome:
        try:
            with self.metrics.timer("ingest_inference"), self.pool.acquire() as pipe:
                analysis = pipe.analyze_gallery_image(decoded.bgr)
            new_faces: list[NewFace] = []
            saved_crops: list[str] = []
            for cand in analysis.eligible:
                assert cand.embedding is not None
                face_id = new_id()
                crop = make_face_crop(decoded.bgr, cand.face.bbox, self.settings.crop_size)
                crop_name = self.store.save_crop(face_id, crop)
                saved_crops.append(crop_name)
                new_faces.append(
                    NewFace(
                        id=face_id,
                        face_index=cand.index,
                        bbox=cand.face.bbox,
                        det_score=cand.face.score,
                        quality=cand.quality,
                        embedding=cand.embedding,
                        crop_name=crop_name,
                    )
                )
            try:
                result = self.repo.store_faces(
                    record.id, self.model_id, new_faces, analysis.rejected_counts
                )
            except BaseException:
                for name in saved_crops:
                    self.store.delete_crop(name)
                raise
            for name in result.removed_crops:
                self.store.delete_crop(name)
        except ApiError:
            raise
        except IntegrityError:
            raise NotFound("Image was deleted during processing.") from None
        except Exception as exc:
            logger.error("Image processing failed (%s)", type(exc).__name__)
            self.metrics.inc("ingest_failed")
            raise ProcessingFailed("Image processing failed.") from exc

        self.metrics.inc("ingest_faces", len(new_faces))
        self.metrics.inc("ingest_images")
        return IngestOutcome(
            status="processed" if new_faces else "no_face",
            image_id=record.id,
            face_count=len(new_faces),
            created=created,
            face_ids=[f.id for f in new_faces if f.id],
            face_seqs=result.face_seqs,
            rejected=analysis.rejected_counts,
        )
