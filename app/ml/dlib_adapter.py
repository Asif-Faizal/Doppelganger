"""dlib baseline (the pipeline the original Doppelganger used) - evaluation/migration only.

Not available in production images: it needs the `face_recognition`/`dlib` C++ build and its
embeddings are euclidean (distance threshold ~0.6), so the cosine/inner-product index refuses it.
Install with `pip install -r requirements-dlib.txt` to run `scripts/evaluate.py --pipeline dlib`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import DetectedFace, Metric, ModelInfo


def _load() -> tuple[Any, str]:
    import dlib
    import face_recognition

    return face_recognition, str(dlib.__version__)


class DlibHogDetector:
    name = "dlib-hog"

    def __init__(self, upsample: int = 1) -> None:
        self._fr, self.version = _load()
        self.upsample = upsample

    def detect(self, bgr: np.ndarray) -> list[DetectedFace]:
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        locations = self._fr.face_locations(
            rgb, number_of_times_to_upsample=self.upsample, model="hog"
        )
        faces = []
        for top, right, bottom, left in locations:
            faces.append(
                DetectedFace(
                    bbox=(float(left), float(top), float(right - left), float(bottom - top)),
                    score=1.0,  # HOG+SVM exposes no calibrated confidence here
                    raw=np.array([top, right, bottom, left], dtype=np.int64),
                )
            )
        faces.sort(key=lambda f: f.area, reverse=True)
        return faces


class DlibEmbedder:
    name = "dlib_face_recognition_resnet_model_v1"
    version = "face_recognition_models-0.3.0"
    dim = 128
    metric: Metric = "l2"

    def __init__(self, num_jitters: int = 1) -> None:
        self._fr, _ = _load()
        self.num_jitters = num_jitters

    def embed(self, bgr: np.ndarray, face: DetectedFace) -> np.ndarray:
        assert face.raw is not None
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        top, right, bottom, left = (int(v) for v in face.raw)
        enc = self._fr.face_encodings(
            rgb, [(top, right, bottom, left)], num_jitters=self.num_jitters
        )
        if not enc:
            raise ValueError("dlib produced no encoding")
        return np.asarray(enc[0], dtype=np.float32)


def build_dlib_pipeline_parts() -> tuple[DlibHogDetector, DlibEmbedder, ModelInfo]:
    detector, embedder = DlibHogDetector(), DlibEmbedder()
    info = ModelInfo(
        model_id="dlib-resnet-v1.hog.p1",
        detector=detector.name,
        detector_version=detector.version,
        embedder=embedder.name,
        embedder_version=embedder.version,
        dim=128,
        metric="l2",
        preprocessing="dlib-hog-face_encodings-jitters1",
        production=False,
    )
    return detector, embedder, info
