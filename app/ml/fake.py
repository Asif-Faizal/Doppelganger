"""Deterministic stand-in models for CI and local development (no weights, no GPU).

The "detector" finds solid-colour rectangles on a grey background; the "embedder" maps the mean
colour of a rectangle through a fixed random smooth function. Same colour -> same vector, similar
colours -> nearby vectors. This exercises the plumbing only; it says nothing about accuracy.
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import DetectedFace, Metric, ModelInfo, l2_normalize

BACKGROUND = 128


class FakeDetector:
    name = "FakeColorBlobs"
    version = "1"

    def __init__(self, min_side: int = 8) -> None:
        self.min_side = min_side

    def detect(self, bgr: np.ndarray) -> list[DetectedFace]:
        diff = np.abs(bgr.astype(np.int16) - BACKGROUND).max(axis=2)
        raw_mask = (diff > 40).astype(np.uint8)
        mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        faces = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w >= self.min_side and h >= self.min_side:
                faces.append(
                    DetectedFace(bbox=(float(x), float(y), float(w), float(h)), score=0.99)
                )
        faces.sort(key=lambda f: f.area, reverse=True)
        return faces


class FakeEmbedder:
    name = "FakeColorEmbedder"
    version = "1"
    metric: Metric = "cosine"

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim
        rng = np.random.default_rng(1234)
        self._w = rng.normal(size=(dim, 3)).astype(np.float32)
        self._b = rng.uniform(0, 2 * np.pi, size=dim).astype(np.float32)

    def embed(self, bgr: np.ndarray, face: DetectedFace) -> np.ndarray:
        x, y, w, h = (int(v) for v in face.bbox)
        dx, dy = w // 4, h // 4
        patch = bgr[y + dy : y + h - dy, x + dx : x + w - dx]
        if patch.size == 0:
            patch = bgr[y : y + h, x : x + w]
        mean = patch.reshape(-1, 3).mean(axis=0) / 255.0
        return l2_normalize(np.sin(self._w @ (mean * 4.0).astype(np.float32) + self._b))


def build_fake_pipeline_parts(dim: int = 128) -> tuple[FakeDetector, FakeEmbedder, ModelInfo]:
    info = ModelInfo(
        model_id=f"fake-colorblobs.v1.d{dim}",
        detector=FakeDetector.name,
        detector_version=FakeDetector.version,
        embedder=FakeEmbedder.name,
        embedder_version=FakeEmbedder.version,
        dim=dim,
        metric="cosine",
        preprocessing="mean-colour-sin",
        production=False,
    )
    return FakeDetector(), FakeEmbedder(dim), info
