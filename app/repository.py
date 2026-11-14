from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

import numpy as np
from sqlalchemy import delete, func, select

from .db import Database, Face, Image, ImageProcessing, IndexEvent, utcnow
from .errors import IndexUnavailable
from .storage import new_id


@dataclass(frozen=True)
class NewFace:
    face_index: int
    bbox: tuple[float, float, float, float]
    det_score: float
    quality: dict[str, Any]
    embedding: np.ndarray
    crop_name: str | None = None
    id: str | None = None


@dataclass(frozen=True)
class FaceHit:
    seq: int
    face_id: str
    image_id: str
    model_id: str
    face_index: int
    bbox: tuple[float, float, float, float]
    det_score: float
    crop_name: str | None
    image_width: int
    image_height: int
    source_name: str | None
    created_at: datetime


@dataclass(frozen=True)
class ImageRecord:
    id: str
    sha256: str
    storage_name: str
    format: str
    width: int
    height: int
    size_bytes: int
    source_name: str | None
    legacy_ref: str | None
    created_at: datetime


@dataclass(frozen=True)
class StoreResult:
    face_seqs: list[int]
    removed_crops: list[str]


class GalleryRepository(Protocol):
    def get_image_by_sha(self, sha256: str) -> ImageRecord | None: ...
    def get_image(self, image_id: str) -> ImageRecord | None: ...
    def create_image(self, record: ImageRecord) -> ImageRecord: ...
    def processed_face_count(self, image_id: str, model_id: str) -> int | None: ...
    def store_faces(
        self,
        image_id: str,
        model_id: str,
        faces: Sequence[NewFace],
        rejected: dict[str, int],
    ) -> StoreResult: ...
    def delete_image(self, image_id: str) -> tuple[ImageRecord, list[str], int] | None: ...
    def fetch_hits(self, seqs: Sequence[int], model_id: str) -> dict[int, FaceHit]: ...
    def get_face(self, face_id: str) -> FaceHit | None: ...


def _record(img: Image) -> ImageRecord:
    return ImageRecord(
        id=img.id,
        sha256=img.sha256,
        storage_name=img.storage_name,
        format=img.format,
        width=img.width,
        height=img.height,
        size_bytes=img.size_bytes,
        source_name=img.source_name,
        legacy_ref=img.legacy_ref,
        created_at=img.created_at,
    )


def _hit(face: Face, img: Image) -> FaceHit:
    return FaceHit(
        seq=face.seq,
        face_id=face.id,
        image_id=face.image_id,
        model_id=face.model_id,
        face_index=face.face_index,
        bbox=(face.bbox_x, face.bbox_y, face.bbox_w, face.bbox_h),
        det_score=face.det_score,
        crop_name=face.crop_name,
        image_width=img.width,
        image_height=img.height,
        source_name=img.source_name,
        created_at=face.created_at,
    )


class SqlGalleryRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def get_image_by_sha(self, sha256: str) -> ImageRecord | None:
        with self.db.read() as s:
            img = s.scalar(select(Image).where(Image.sha256 == sha256))
            return _record(img) if img else None

    def get_image(self, image_id: str) -> ImageRecord | None:
        with self.db.read() as s:
            img = s.get(Image, image_id)
            return _record(img) if img else None

    def create_image(self, record: ImageRecord) -> ImageRecord:
        with self.db.write() as s:
            s.add(
                Image(
                    id=record.id,
                    sha256=record.sha256,
                    storage_name=record.storage_name,
                    format=record.format,
                    width=record.width,
                    height=record.height,
                    size_bytes=record.size_bytes,
                    source_name=record.source_name,
                    legacy_ref=record.legacy_ref,
                    created_at=record.created_at,
                )
            )
        return record

    def processed_face_count(self, image_id: str, model_id: str) -> int | None:
        """Face count if (image, model) was already processed, else None."""
        with self.db.read() as s:
            row = s.get(ImageProcessing, (image_id, model_id))
            return None if row is None else int(row.face_count)

    def store_faces(
        self,
        image_id: str,
        model_id: str,
        faces: Sequence[NewFace],
        rejected: dict[str, int],
    ) -> StoreResult:
        """Atomically replace the faces of (image, model); idempotent for retries."""
        removed_crops: list[str] = []
        seqs: list[int] = []
        with self.db.write() as s:
            old = s.execute(
                select(Face.seq, Face.crop_name).where(
                    Face.image_id == image_id, Face.model_id == model_id
                )
            ).all()
            if old:
                s.execute(delete(Face).where(Face.image_id == image_id, Face.model_id == model_id))
                s.add_all(
                    IndexEvent(model_id=model_id, face_seq=seq, op="delete") for seq, _ in old
                )
                removed_crops = [c for _, c in old if c]
            rows = []
            for f in faces:
                row = Face(
                    id=f.id or new_id(),
                    image_id=image_id,
                    model_id=model_id,
                    face_index=f.face_index,
                    bbox_x=f.bbox[0],
                    bbox_y=f.bbox[1],
                    bbox_w=f.bbox[2],
                    bbox_h=f.bbox[3],
                    det_score=f.det_score,
                    quality=f.quality,
                    embedding=np.asarray(f.embedding, dtype=np.float32).tobytes(),
                    crop_name=f.crop_name,
                )
                rows.append(row)
            s.add_all(rows)
            s.flush()
            seqs = [r.seq for r in rows]
            s.merge(
                ImageProcessing(
                    image_id=image_id,
                    model_id=model_id,
                    status="processed" if faces else "no_face",
                    face_count=len(faces),
                    rejected=rejected or None,
                    processed_at=utcnow(),
                )
            )
        return StoreResult(face_seqs=seqs, removed_crops=removed_crops)

    def delete_image(self, image_id: str) -> tuple[ImageRecord, list[str], int] | None:
        with self.db.write() as s:
            img = s.get(Image, image_id)
            if img is None:
                return None
            rec = _record(img)
            faces = s.execute(
                select(Face.seq, Face.model_id, Face.crop_name).where(Face.image_id == image_id)
            ).all()
            s.add_all(IndexEvent(model_id=m, face_seq=seq, op="delete") for seq, m, _ in faces)
            s.execute(delete(Face).where(Face.image_id == image_id))
            s.execute(delete(Image).where(Image.id == image_id))
            return rec, [c for _, _, c in faces if c], len(faces)

    def fetch_hits(self, seqs: Sequence[int], model_id: str) -> dict[int, FaceHit]:
        if not seqs:
            return {}
        with self.db.read() as s:
            rows = s.execute(
                select(Face, Image)
                .join(Image, Image.id == Face.image_id)
                .where(Face.seq.in_(list(seqs)), Face.model_id == model_id)
            ).all()
            return {f.seq: _hit(f, i) for f, i in rows}

    def get_face(self, face_id: str) -> FaceHit | None:
        with self.db.read() as s:
            row = s.execute(
                select(Face, Image).join(Image, Image.id == Face.image_id).where(Face.id == face_id)
            ).first()
            return _hit(*row) if row else None

    def first_face_of_image(self, image_id: str, model_id: str) -> FaceHit | None:
        with self.db.read() as s:
            row = s.execute(
                select(Face, Image)
                .join(Image, Image.id == Face.image_id)
                .where(Face.image_id == image_id, Face.model_id == model_id)
                .order_by(Face.face_index)
                .limit(1)
            ).first()
            return _hit(*row) if row else None

    # --- index support -------------------------------------------------------------------

    def iter_embeddings(
        self, model_id: str, dim: int, after_seq: int = 0, batch_size: int = 5000
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        last = after_seq
        while True:
            with self.db.read() as s:
                rows = s.execute(
                    select(Face.seq, Face.embedding)
                    .where(Face.model_id == model_id, Face.seq > last)
                    .order_by(Face.seq)
                    .limit(batch_size)
                ).all()
            if not rows:
                return
            seqs = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
            vecs = np.empty((len(rows), dim), dtype=np.float32)
            for i, r in enumerate(rows):
                if len(r[1]) != dim * 4:
                    raise IndexUnavailable(
                        f"Stored embedding for model '{model_id}' has an unexpected size "
                        f"(expected dimension {dim}). Run `check` and reprocess affected images.",
                        code="embedding_size_mismatch",
                    )
                vecs[i] = np.frombuffer(r[1], dtype=np.float32)
            last = int(seqs[-1])
            yield seqs, vecs

    def max_face_seq(self, model_id: str) -> int:
        with self.db.read() as s:
            return int(s.scalar(select(func.max(Face.seq)).where(Face.model_id == model_id)) or 0)

    def max_event_id(self, model_id: str) -> int:
        with self.db.read() as s:
            return int(
                s.scalar(select(func.max(IndexEvent.id)).where(IndexEvent.model_id == model_id))
                or 0
            )

    def events_after(self, model_id: str, after_id: int) -> list[tuple[int, int]]:
        with self.db.read() as s:
            rows = s.execute(
                select(IndexEvent.id, IndexEvent.face_seq)
                .where(IndexEvent.model_id == model_id, IndexEvent.id > after_id)
                .order_by(IndexEvent.id)
            ).all()
            return [(r[0], r[1]) for r in rows]

    def face_seqs(self, model_id: str) -> set[int]:
        with self.db.read() as s:
            return set(s.scalars(select(Face.seq).where(Face.model_id == model_id)).all())

    def count_faces(self, model_id: str | None = None) -> int:
        with self.db.read() as s:
            q = select(func.count()).select_from(Face)
            if model_id:
                q = q.where(Face.model_id == model_id)
            return int(s.scalar(q) or 0)

    def count_images(self) -> int:
        with self.db.read() as s:
            return int(s.scalar(select(func.count()).select_from(Image)) or 0)

    def faces_by_model(self) -> dict[str, int]:
        with self.db.read() as s:
            rows = s.execute(select(Face.model_id, func.count()).group_by(Face.model_id)).all()
            return {r[0]: int(r[1]) for r in rows}

    def all_image_ids(self) -> list[str]:
        with self.db.read() as s:
            return list(s.scalars(select(Image.id).order_by(Image.created_at)).all())

    def storage_names(self) -> tuple[set[str], set[str]]:
        with self.db.read() as s:
            originals = set(s.scalars(select(Image.storage_name)).all())
            crops = {c for c in s.scalars(select(Face.crop_name)).all() if c}
            return originals, crops

    def purge_model(self, model_id: str) -> list[str]:
        """Remove all faces of a (retired) model; returns crop names to delete from disk."""
        with self.db.write() as s:
            rows = s.execute(
                select(Face.seq, Face.crop_name).where(Face.model_id == model_id)
            ).all()
            s.execute(delete(Face).where(Face.model_id == model_id))
            s.execute(delete(ImageProcessing).where(ImageProcessing.model_id == model_id))
            s.execute(delete(IndexEvent).where(IndexEvent.model_id == model_id))
            return [c for _, c in rows if c]
