from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from .config import Settings
from .index import IndexManager
from .metrics import Metrics
from .ml.pipeline import PipelinePool
from .repository import FaceHit, SqlGalleryRepository

LOOKALIKE_NOTICE = (
    "LOOKALIKE ranks faces by visual similarity under this model. Resemblance is not identity: "
    "high-ranked results can be different people."
)
SAME_PERSON_NOTICE = (
    "SAME_PERSON returns candidates whose similarity passes a threshold. A candidate is not proof "
    "of identity; thresholds trade false matches against misses and must be calibrated per gallery."
)


class SearchMode(StrEnum):
    LOOKALIKE = "LOOKALIKE"
    SAME_PERSON = "SAME_PERSON"


@dataclass(frozen=True)
class Calibration:
    """Optional output of scripts/evaluate.py. Without it no calibrated confidence is reported."""

    model_id: str
    dataset: str
    a: float
    b: float
    same_person_threshold: float | None
    n_pairs: int

    @classmethod
    def load(cls, path: Path | None, model_id: str) -> Calibration | None:
        if path is None or not path.is_file():
            return None
        raw = json.loads(path.read_text())
        if raw.get("model_id") != model_id:
            return None
        return cls(
            model_id=raw["model_id"],
            dataset=str(raw.get("dataset", "unspecified")),
            a=float(raw["logistic"]["a"]),
            b=float(raw["logistic"]["b"]),
            same_person_threshold=raw.get("same_person_threshold"),
            n_pairs=int(raw.get("n_pairs", 0)),
        )

    def confidence(self, cosine: float) -> float:
        z = self.a * cosine + self.b
        return 1.0 / (1.0 + math.exp(-z)) if z > -700 else 0.0


@dataclass(frozen=True)
class SearchParams:
    mode: SearchMode = SearchMode.LOOKALIKE
    max_results: int = 10
    offset: int = 0
    min_similarity: float | None = None
    face_index: int | None = None
    dedupe_by_image: bool = True


@dataclass
class Match:
    rank: int
    hit: FaceHit
    cosine_similarity: float
    cosine_distance: float
    calibrated_confidence: float | None
    passes_same_person_threshold: bool | None


@dataclass
class SearchOutcome:
    mode: SearchMode
    model_id: str
    status: str  # ok | no_reliable_match | empty_gallery
    message: str | None
    matches: list[Match]
    threshold: float | None
    threshold_source: str | None
    query_bbox: tuple[float, float, float, float]
    query_quality: dict[str, float]
    detected_faces: int
    gallery_faces: int
    next_offset: int | None
    calibration: Calibration | None
    notice: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


class SearchService:
    MAX_FETCH = 5000

    def __init__(
        self,
        settings: Settings,
        pool: PipelinePool,
        index: IndexManager,
        repo: SqlGalleryRepository,
        metrics: Metrics,
    ) -> None:
        self.settings = settings
        self.pool = pool
        self.index = index
        self.repo = repo
        self.metrics = metrics
        self.calibration = Calibration.load(settings.calibration_file, pool.info.model_id)

    def default_same_person_threshold(self) -> tuple[float, str]:
        if self.calibration and self.calibration.same_person_threshold is not None:
            return self.calibration.same_person_threshold, "calibrated"
        return self.settings.same_person_min_similarity, "vendor_default_uncalibrated"

    def search_image(self, bgr: np.ndarray, params: SearchParams) -> SearchOutcome:
        with self.metrics.timer("search_inference"), self.pool.acquire() as pipe:
            chosen, analysis = pipe.select_query_face(bgr, params.face_index)
        assert chosen.embedding is not None
        return self.search_vector(
            chosen.embedding,
            params,
            query_bbox=chosen.face.bbox,
            query_quality=chosen.quality,
            detected_faces=len(analysis.candidates),
        )

    def search_vector(
        self,
        vector: np.ndarray,
        params: SearchParams,
        *,
        query_bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        query_quality: dict[str, float] | None = None,
        detected_faces: int = 1,
    ) -> SearchOutcome:
        threshold: float | None = None
        source: str | None = None
        if params.mode is SearchMode.SAME_PERSON:
            threshold, source = self.default_same_person_threshold()
        if params.min_similarity is not None:
            threshold, source = params.min_similarity, "request"

        self.index.sync()
        gallery = self.index.count()
        notice = LOOKALIKE_NOTICE if params.mode is SearchMode.LOOKALIKE else SAME_PERSON_NOTICE
        base: dict[str, Any] = {
            "mode": params.mode,
            "model_id": self.pool.info.model_id,
            "threshold": threshold,
            "threshold_source": source,
            "query_bbox": query_bbox,
            "query_quality": query_quality or {},
            "detected_faces": detected_faces,
            "gallery_faces": gallery,
            "calibration": self.calibration,
            "notice": notice,
        }
        if gallery == 0:
            return SearchOutcome(
                status="empty_gallery",
                message="The gallery has no indexed faces for the active model.",
                matches=[],
                next_offset=None,
                **base,
            )

        want = params.offset + params.max_results
        fetch = min(max(want * (4 if params.dedupe_by_image else 1), 16), self.MAX_FETCH)
        results: list[tuple[FaceHit, float]] = []
        exhausted = False
        with self.metrics.timer("search_index"):
            while True:
                ids, scores = self.index.search(vector, fetch)
                hits = self.repo.fetch_hits([int(i) for i in ids], self.pool.info.model_id)
                results = []
                seen_images: set[str] = set()
                for i, sc in zip(ids, scores, strict=True):
                    hit = hits.get(int(i))
                    if hit is None:  # deleted between index search and DB fetch
                        continue
                    cos = float(sc)
                    if threshold is not None and cos < threshold:
                        exhausted = True  # scores are sorted: everything after is lower
                        break
                    if params.dedupe_by_image:
                        if hit.image_id in seen_images:
                            continue
                        seen_images.add(hit.image_id)
                    results.append((hit, cos))
                if len(results) > want or fetch >= min(gallery, self.MAX_FETCH) or exhausted:
                    break
                fetch = min(fetch * 4, self.MAX_FETCH)

        page = results[params.offset : params.offset + params.max_results]
        has_more = len(results) > params.offset + params.max_results
        matches = [
            Match(
                rank=params.offset + i + 1,
                hit=hit,
                cosine_similarity=cos,
                cosine_distance=1.0 - cos,
                calibrated_confidence=self.calibration.confidence(cos)
                if self.calibration
                else None,
                passes_same_person_threshold=None
                if params.mode is SearchMode.LOOKALIKE
                else (threshold is not None and cos >= threshold),
            )
            for i, (hit, cos) in enumerate(page)
        ]
        status, message = "ok", None
        if not matches:
            if params.mode is SearchMode.SAME_PERSON or threshold is not None:
                status = "no_reliable_match"
                message = (
                    "No indexed face reached the similarity threshold. Nothing was returned "
                    "rather than forcing a low-confidence result."
                )
            else:
                status, message = "no_reliable_match", "No results in the requested range."
        return SearchOutcome(
            status=status,
            message=message,
            matches=matches,
            next_offset=params.offset + params.max_results if has_more else None,
            **base,
        )
