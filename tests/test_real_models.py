"""Offline integration tests with the real ONNX weights (run: `pytest -m models`).

They are skipped (not failed) when the weights are not provisioned. The sample photos are the five
images shipped in the repo root; this is a smoke test of the real inference path, NOT an accuracy
measurement (see docs/BENCHMARKS.md and scripts/evaluate.py for those).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.imaging import decode_image
from app.ml.manifest import load_manifest, verify_model
from app.ml.pipeline import default_pipeline_factory

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
SAMPLES = [ROOT / f"{i}.jpg" for i in range(1, 6)]

pytestmark = pytest.mark.models


@pytest.fixture(scope="module")
def pool():  # type: ignore[no-untyped-def]
    manifest = load_manifest()
    try:
        for m in manifest.values():
            verify_model(MODELS, m)
    except Exception as exc:
        pytest.skip(f"model files not provisioned: {exc}")
    return default_pipeline_factory(
        Settings(models_dir=MODELS, pipeline="sface", auth_disabled=True, inference_instances=1)
    )


@pytest.mark.skipif(not all(p.exists() for p in SAMPLES), reason="sample photos not present")
def test_real_pipeline_detects_embeds_and_separates_samples(pool) -> None:  # type: ignore[no-untyped-def]
    s = Settings(models_dir=MODELS, pipeline="sface", auth_disabled=True)
    vecs = []
    with pool.acquire() as pipe:
        assert pipe.info.dim == 128 and pipe.info.metric == "cosine"
        for p in SAMPLES:
            img = decode_image(p.read_bytes(), s)
            chosen, _ = pipe.select_query_face(img.bgr)
            assert chosen.embedding is not None and chosen.embedding.shape == (128,)
            assert abs(float(np.linalg.norm(chosen.embedding)) - 1.0) < 1e-4
            vecs.append(chosen.embedding)
    sims = np.array(vecs) @ np.array(vecs).T
    same = [sims[0, 1], sims[0, 2], sims[1, 2], sims[2, 3]]  # photos 1-4 show the same person
    diff = [sims[0, 4], sims[1, 4], sims[2, 4], sims[3, 4]]  # photo 5 is a different person
    assert min(same) > max(diff)


def test_real_model_ids_are_versioned(pool) -> None:  # type: ignore[no-untyped-def]
    assert pool.info.model_id == "sface-2021dec.yunet-2023mar.p1"
