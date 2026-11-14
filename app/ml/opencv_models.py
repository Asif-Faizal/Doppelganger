from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..errors import ModelUnavailable
from .base import DetectedFace, Metric, ModelInfo, l2_normalize
from .manifest import ModelFile, load_manifest, verify_model

PREPROCESSING = "yunet-5pt-alignCrop112-sface-l2norm"
PIPELINE_REVISION = "p1"


class YuNetDetector:
    """OpenCV YuNet face detector (MIT). Returns bbox, 5 landmarks and a confidence score."""

    def __init__(
        self,
        model: ModelFile,
        path: Path,
        *,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        max_side: int = 1280,
        threads: int = 1,
    ) -> None:
        self.name = model.name
        self.version = model.version
        self.max_side = max_side
        cv2.setNumThreads(threads)
        try:
            self._net = cv2.FaceDetectorYN.create(
                str(path), "", (320, 320), score_threshold, nms_threshold, 5000
            )
        except cv2.error as exc:
            raise ModelUnavailable(
                "The face detector model could not be loaded.", code="model_load_failed"
            ) from exc

    def detect(self, bgr: np.ndarray) -> list[DetectedFace]:
        h, w = bgr.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        img = bgr
        if scale < 1.0:
            img = cv2.resize(
                bgr,
                (max(1, round(w * scale)), max(1, round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        self._net.setInputSize((img.shape[1], img.shape[0]))
        _, rows = self._net.detect(img)
        if rows is None:
            return []
        faces: list[DetectedFace] = []
        for row in rows:
            r = np.asarray(row, dtype=np.float32).copy()
            r[:14] /= scale  # back to source-image pixel coordinates
            landmarks = r[4:14].reshape(5, 2)
            faces.append(
                DetectedFace(
                    bbox=(float(r[0]), float(r[1]), float(r[2]), float(r[3])),
                    score=float(r[14]),
                    landmarks=landmarks,
                    raw=r,
                )
            )
        faces.sort(key=lambda f: f.area, reverse=True)
        return faces


class SFaceEmbedder:
    """OpenCV SFace embedder (Apache-2.0 wrapper; MobileFaceNet trained with SFace loss), 128-d."""

    dim = 128
    metric: Metric = "cosine"

    def __init__(self, model: ModelFile, path: Path) -> None:
        self.name = model.name
        self.version = model.version
        try:
            self._net = cv2.FaceRecognizerSF.create(str(path), "")
        except cv2.error as exc:
            raise ModelUnavailable(
                "The face embedding model could not be loaded.", code="model_load_failed"
            ) from exc

    def embed(self, bgr: np.ndarray, face: DetectedFace) -> np.ndarray:
        if face.raw is None:
            raise ValueError("SFace needs YuNet landmarks for alignment")
        aligned = self._net.alignCrop(bgr, face.raw)
        feat = self._net.feature(aligned)
        return l2_normalize(feat)


def build_sface_pipeline_parts(
    models_dir: Path, *, max_side: int, threads: int
) -> tuple[YuNetDetector, SFaceEmbedder, ModelInfo]:
    manifest = load_manifest()
    yunet, sface = manifest["yunet"], manifest["sface"]
    detector = YuNetDetector(
        yunet, verify_model(models_dir, yunet), max_side=max_side, threads=threads
    )
    embedder = SFaceEmbedder(sface, verify_model(models_dir, sface))
    info = ModelInfo(
        model_id=f"sface-{sface.version}.yunet-{yunet.version}.{PIPELINE_REVISION}",
        detector=yunet.name,
        detector_version=yunet.version,
        embedder=sface.name,
        embedder_version=sface.version,
        dim=SFaceEmbedder.dim,
        metric="cosine",
        preprocessing=PREPROCESSING,
    )
    return detector, embedder, info
