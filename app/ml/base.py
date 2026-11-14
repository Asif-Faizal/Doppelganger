from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np

Metric = Literal["cosine", "l2"]


@dataclass(frozen=True)
class DetectedFace:
    bbox: tuple[float, float, float, float]  # x, y, w, h in source-image pixels
    score: float
    landmarks: np.ndarray | None = (
        None  # (5, 2): right eye, left eye, nose, right mouth, left mouth
    )
    raw: np.ndarray | None = field(
        default=None, repr=False
    )  # detector-specific row for the embedder

    @property
    def area(self) -> float:
        return float(self.bbox[2] * self.bbox[3])


@dataclass(frozen=True)
class ModelInfo:
    """Identity of the full detection+alignment+embedding recipe. Embeddings are only comparable
    when this id is identical, so it is stored with every face and with every index."""

    model_id: str
    detector: str
    detector_version: str
    embedder: str
    embedder_version: str
    dim: int
    metric: Metric
    preprocessing: str
    production: bool = True


@runtime_checkable
class FaceDetector(Protocol):
    name: str
    version: str

    def detect(self, bgr: np.ndarray) -> list[DetectedFace]: ...


@runtime_checkable
class FaceEmbedder(Protocol):
    name: str
    version: str
    dim: int
    metric: Metric

    def embed(self, bgr: np.ndarray, face: DetectedFace) -> np.ndarray:
        """Return one float32 vector of length ``dim`` (L2-normalised when metric == 'cosine')."""
        ...


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(v))
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("degenerate embedding")
    return v / norm
