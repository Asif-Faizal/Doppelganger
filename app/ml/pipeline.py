from __future__ import annotations

import contextlib
import logging
import queue
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import Settings
from ..errors import (
    FaceQualityInsufficient,
    InvalidRequest,
    ModelUnavailable,
    MultipleFacesAmbiguous,
    NoFaceDetected,
)
from .base import DetectedFace, FaceDetector, FaceEmbedder, ModelInfo
from .fake import build_fake_pipeline_parts
from .opencv_models import build_sface_pipeline_parts

logger = logging.getLogger(__name__)

QUALITY_MESSAGES = {
    "low_detection_confidence": "Face detection confidence is too low.",
    "face_too_small": "The face is too small; use a closer or higher-resolution photo.",
    "face_blurry": "The face is too blurry; use a sharper photo.",
    "face_cut_off": "The face is cut off by the image border.",
}


@dataclass
class FaceCandidate:
    index: int
    face: DetectedFace
    quality: dict[str, float]
    reasons: list[str]
    embedding: np.ndarray | None = field(default=None, repr=False)

    @property
    def eligible(self) -> bool:
        return not self.reasons


@dataclass
class ImageAnalysis:
    candidates: list[FaceCandidate]

    @property
    def eligible(self) -> list[FaceCandidate]:
        return [c for c in self.candidates if c.eligible]

    @property
    def rejected_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self.candidates:
            for r in c.reasons:
                counts[r] = counts.get(r, 0) + 1
        return counts


class FacePipeline:
    """Detector + quality gate + embedder bound to one ModelInfo."""

    def __init__(
        self,
        detector: FaceDetector,
        embedder: FaceEmbedder,
        info: ModelInfo,
        settings: Settings,
    ) -> None:
        if embedder.dim != info.dim:
            raise ValueError("embedder dimension does not match model info")
        self.detector = detector
        self.embedder = embedder
        self.info = info
        self.settings = settings

    def quality(self, bgr: np.ndarray, face: DetectedFace) -> tuple[dict[str, float], list[str]]:
        s = self.settings
        h, w = bgr.shape[:2]
        x, y, fw, fh = face.bbox
        reasons: list[str] = []

        ix0, iy0 = max(0.0, x), max(0.0, y)
        ix1, iy1 = min(float(w), x + fw), min(float(h), y + fh)
        inside = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        overflow = 1.0 - inside / max(fw * fh, 1e-6)

        sharp = 0.0
        if inside > 0:
            crop = cv2.cvtColor(bgr[int(iy0) : int(iy1), int(ix0) : int(ix1)], cv2.COLOR_BGR2GRAY)
            if crop.size:
                crop = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)
                sharp = float(cv2.Laplacian(crop, cv2.CV_64F).var())

        roll = 0.0
        if face.landmarks is not None:
            re, le = face.landmarks[0], face.landmarks[1]
            roll = float(np.degrees(np.arctan2(le[1] - re[1], le[0] - re[0])))

        quality = {
            "detection_score": round(face.score, 4),
            "min_side_px": round(min(fw, fh), 1),
            "sharpness": round(sharp, 2),
            "edge_overflow": round(overflow, 3),
            "roll_deg": round(roll, 1),
        }
        if face.score < s.min_detection_score:
            reasons.append("low_detection_confidence")
        if min(fw, fh) < s.min_face_px:
            reasons.append("face_too_small")
        if sharp < s.min_sharpness:
            reasons.append("face_blurry")
        if overflow > s.max_edge_overflow:
            reasons.append("face_cut_off")
        return quality, reasons

    def detect(self, bgr: np.ndarray) -> ImageAnalysis:
        faces = self.detector.detect(bgr)[: self.settings.max_faces_per_image]
        candidates = []
        for i, face in enumerate(faces):
            quality, reasons = self.quality(bgr, face)
            candidates.append(FaceCandidate(index=i, face=face, quality=quality, reasons=reasons))
        return ImageAnalysis(candidates)

    def embed(self, bgr: np.ndarray, candidate: FaceCandidate) -> np.ndarray:
        vec = self.embedder.embed(bgr, candidate.face)
        if vec.shape != (self.info.dim,) or not np.all(np.isfinite(vec)):
            raise ValueError("embedder returned an invalid vector")
        candidate.embedding = vec.astype(np.float32, copy=False)
        return candidate.embedding

    def analyze_gallery_image(self, bgr: np.ndarray) -> ImageAnalysis:
        """Every eligible face is embedded (gallery ingestion indexes all of them)."""
        analysis = self.detect(bgr)
        for c in analysis.eligible:
            self.embed(bgr, c)
        return analysis

    def select_query_face(
        self, bgr: np.ndarray, face_index: int | None = None
    ) -> tuple[FaceCandidate, ImageAnalysis]:
        """Pick the selfie face or raise a precise error; never silently picks the first of many."""
        analysis = self.detect(bgr)
        if not analysis.candidates:
            raise NoFaceDetected("No face was detected in the image.")
        real = [c for c in analysis.candidates if "low_detection_confidence" not in c.reasons]
        if face_index is not None:
            if not 0 <= face_index < len(analysis.candidates):
                raise InvalidRequest(
                    "face_index is out of range.",
                    code="invalid_face_index",
                    details={"detected_faces": len(analysis.candidates)},
                )
            chosen = analysis.candidates[face_index]
        else:
            if not real:
                raise FaceQualityInsufficient(
                    QUALITY_MESSAGES["low_detection_confidence"],
                    details={"reasons": ["low_detection_confidence"]},
                )
            if (
                len(real) > 1
                and real[1].face.area >= self.settings.ambiguity_ratio * real[0].face.area
            ):
                raise MultipleFacesAmbiguous(
                    "Several faces of similar size were found; crop the image to one face or "
                    "resubmit with face_index.",
                    details={
                        "faces": [
                            {"face_index": c.index, "bbox": [round(v, 1) for v in c.face.bbox]}
                            for c in real
                        ]
                    },
                )
            chosen = real[0]
        if chosen.reasons:
            raise FaceQualityInsufficient(
                " ".join(QUALITY_MESSAGES[r] for r in chosen.reasons),
                details={"reasons": chosen.reasons, "quality": chosen.quality},
            )
        self.embed(bgr, chosen)
        return chosen, analysis


class PipelinePool:
    """Fixed set of pipeline instances (OpenCV DNN objects are not thread-safe)."""

    def __init__(self, pipelines: list[FacePipeline]) -> None:
        if not pipelines:
            raise ValueError("pool needs at least one pipeline")
        self.info = pipelines[0].info
        self.size = len(pipelines)
        self._q: queue.Queue[FacePipeline] = queue.Queue()
        for p in pipelines:
            self._q.put(p)

    @contextlib.contextmanager
    def acquire(self, timeout: float = 120.0) -> Iterator[FacePipeline]:
        try:
            pipe = self._q.get(timeout=timeout)
        except queue.Empty as exc:
            raise ModelUnavailable("The inference pool is saturated; retry shortly.") from exc
        try:
            yield pipe
        finally:
            self._q.put(pipe)


PipelineFactory = Callable[[Settings], PipelinePool]


def default_pipeline_factory(settings: Settings) -> PipelinePool:
    pipelines = []
    detector: FaceDetector
    embedder: FaceEmbedder
    for _ in range(settings.inference_instances):
        if settings.pipeline == "fake":
            detector, embedder, info = build_fake_pipeline_parts()
        else:
            detector, embedder, info = build_sface_pipeline_parts(
                settings.models_dir,
                max_side=settings.detect_max_side,
                threads=settings.inference_threads,
            )
        pipelines.append(FacePipeline(detector, embedder, info, settings))
    logger.info(
        "Loaded %d pipeline instance(s) for model %s", len(pipelines), pipelines[0].info.model_id
    )
    return PipelinePool(pipelines)
