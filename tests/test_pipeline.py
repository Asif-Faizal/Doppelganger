from __future__ import annotations

import numpy as np
import pytest

from app.config import Settings
from app.errors import (
    FaceQualityInsufficient,
    InvalidRequest,
    MultipleFacesAmbiguous,
    NoFaceDetected,
)
from app.imaging import decode_image
from app.ml.base import DetectedFace
from app.ml.fake import FakeDetector, FakeEmbedder, build_fake_pipeline_parts
from app.ml.pipeline import FacePipeline, PipelinePool, default_pipeline_factory

from .conftest import BLUE, GREEN, RED, face_image, single


def make_pipeline(settings: Settings) -> FacePipeline:
    d, e, info = build_fake_pipeline_parts()
    return FacePipeline(d, e, info, settings)


def bgr(data: bytes, settings: Settings) -> np.ndarray:
    return decode_image(data, settings).bgr


def test_no_face(settings: Settings) -> None:
    with pytest.raises(NoFaceDetected):
        make_pipeline(settings).select_query_face(bgr(face_image([]), settings))


def test_single_face_is_selected_and_embedded(settings: Settings) -> None:
    chosen, analysis = make_pipeline(settings).select_query_face(bgr(single(RED), settings))
    assert len(analysis.candidates) == 1
    assert chosen.embedding is not None
    assert chosen.embedding.shape == (128,)
    assert abs(float(np.linalg.norm(chosen.embedding)) - 1.0) < 1e-5


def test_multiple_similar_faces_are_ambiguous_not_first(settings: Settings) -> None:
    data = face_image([(RED, 20, 20, 90), (BLUE, 180, 180, 85)])
    with pytest.raises(MultipleFacesAmbiguous) as exc:
        make_pipeline(settings).select_query_face(bgr(data, settings))
    assert len(exc.value.details["faces"]) == 2


def test_dominant_face_wins_over_small_background_face(settings: Settings) -> None:
    data = face_image([(RED, 20, 20, 150), (BLUE, 250, 250, 40)])
    chosen, _ = make_pipeline(settings).select_query_face(bgr(data, settings))
    assert chosen.face.bbox[2] > 100


def test_face_index_resolves_ambiguity(settings: Settings) -> None:
    data = face_image([(RED, 20, 20, 90), (BLUE, 180, 180, 85)])
    pipe = make_pipeline(settings)
    first, _ = pipe.select_query_face(bgr(data, settings), face_index=0)
    second, _ = pipe.select_query_face(bgr(data, settings), face_index=1)
    assert first.face.bbox != second.face.bbox
    with pytest.raises(InvalidRequest):
        pipe.select_query_face(bgr(data, settings), face_index=7)


def test_small_face_has_quality_feedback(settings: Settings) -> None:
    strict = settings.model_copy(update={"min_face_px": 60})
    with pytest.raises(FaceQualityInsufficient) as exc:
        make_pipeline(strict).select_query_face(bgr(single(RED, side=30), strict))
    assert "face_too_small" in exc.value.details["reasons"]
    assert "too small" in exc.value.message


def test_blurry_face_is_rejected(settings: Settings) -> None:
    with pytest.raises(FaceQualityInsufficient) as exc:
        make_pipeline(settings).select_query_face(bgr(single(RED, textured=False), settings))
    assert "face_blurry" in exc.value.details["reasons"]


def test_cut_off_face_is_flagged(settings: Settings) -> None:
    pipe = make_pipeline(settings)
    img = bgr(single(GREEN), settings)
    overflowing = DetectedFace(
        bbox=(280.0, 100.0, 100.0, 100.0), score=0.99
    )  # 80% outside the frame
    quality, reasons = pipe.quality(img, overflowing)
    assert "face_cut_off" in reasons
    assert quality["edge_overflow"] > 0.5


def test_gallery_analysis_embeds_every_eligible_face(settings: Settings) -> None:
    data = face_image(
        [(RED, 10, 10, 80), (GREEN, 120, 10, 80), (BLUE, 10, 150, 80), (RED, 180, 180, 20)]
    )
    analysis = make_pipeline(settings).analyze_gallery_image(bgr(data, settings))
    assert len(analysis.eligible) == 3
    assert analysis.rejected_counts.get("face_too_small") == 1
    assert all(c.embedding is not None for c in analysis.eligible)


def test_embedding_dimension_is_validated(settings: Settings) -> None:
    d, _, info = build_fake_pipeline_parts(128)
    with pytest.raises(ValueError):
        FacePipeline(d, FakeEmbedder(64), info, settings)


def test_embedder_returning_wrong_shape_is_rejected(settings: Settings) -> None:
    class Bad(FakeEmbedder):
        def embed(self, bgr: np.ndarray, face):  # type: ignore[no-untyped-def]
            return np.ones(10, dtype=np.float32)

    d, _, info = build_fake_pipeline_parts()
    pipe = FacePipeline(d, Bad(), info, settings)
    cand = pipe.detect(bgr(single(RED), settings)).candidates[0]
    with pytest.raises(ValueError):
        pipe.embed(bgr(single(RED), settings), cand)


def test_similar_colours_embed_closer_than_different_ones(settings: Settings) -> None:
    pipe = make_pipeline(settings)

    def vec(colour: tuple[int, int, int]) -> np.ndarray:
        c, _ = pipe.select_query_face(bgr(single(colour), settings))
        assert c.embedding is not None
        return c.embedding

    near = float(vec(RED) @ vec((40, 40, 210)))
    far = float(vec(RED) @ vec(BLUE))
    assert near > far
    assert float(vec(RED) @ vec(RED)) == pytest.approx(1.0, abs=1e-5)


def test_pool_serialises_access_and_default_factory_builds_fake(settings: Settings) -> None:
    pool = default_pipeline_factory(settings)
    assert isinstance(pool, PipelinePool)
    assert pool.size == settings.inference_instances
    with pool.acquire() as a, pool.acquire() as b:
        assert a is not b
    assert isinstance(FakeDetector().detect(bgr(single(RED), settings)), list)
