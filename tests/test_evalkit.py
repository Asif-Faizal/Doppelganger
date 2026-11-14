from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import evalkit


def _clusters(n_id: int = 40, per: int = 4, noise: float = 0.35, dim: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((n_id, dim))
    emb = np.repeat(centres, per, axis=0) + rng.standard_normal((n_id * per, dim)) * noise
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    labels = np.repeat([f"p{i:03d}" for i in range(n_id)], per)
    return emb.astype(np.float32), labels


def test_similarity_cosine_and_l2_orderings_agree_on_normalised_vectors() -> None:
    emb, _ = _clusters()
    cos = evalkit.similarity(emb[:5], emb, "cosine")
    l2 = evalkit.similarity(emb[:5], emb, "l2")
    assert np.array_equal(np.argsort(-cos, axis=1), np.argsort(-l2, axis=1))
    assert np.allclose(np.diag(cos[:, :5]), 1.0, atol=1e-5)


def test_identity_split_is_disjoint_and_deterministic() -> None:
    _, labels = _clusters()
    a1, b1 = evalkit.split_identities(labels, seed=3)
    a2, b2 = evalkit.split_identities(labels, seed=3)
    assert (a1, b1) == (a2, b2)
    assert not a1 & b1 and a1 | b1 == set(labels.tolist())


def test_open_set_split_never_puts_gallery_image_among_probes() -> None:
    _, labels = _clusters()
    valid = np.ones(len(labels), dtype=bool)
    split = evalkit.build_split(labels, valid, set(labels.tolist()), seed=1)
    assert not set(split.gallery.tolist()) & set(split.mated.tolist())
    assert not set(split.gallery.tolist()) & set(split.non_mated.tolist())
    enrolled = set(labels[split.gallery].tolist())
    assert set(labels[split.mated].tolist()) <= enrolled
    assert not set(labels[split.non_mated].tolist()) & enrolled
    assert len(set(labels[split.gallery].tolist())) == len(split.gallery)  # one per identity


def test_failure_to_acquire_counts_as_a_miss() -> None:
    emb, labels = _clusters()
    valid = np.ones(len(labels), dtype=bool)
    valid[1::4] = False
    split = evalkit.build_split(labels, valid, set(labels.tolist()), seed=1)
    assert split.mated_fta > 0
    top = evalkit.closed_set_topk(emb, labels, split, "cosine", (1,))
    assert top["top1"] < top["top1_detected_only"]


def test_well_separated_clusters_identify_perfectly_and_rates_move_with_threshold() -> None:
    emb, labels = _clusters(noise=0.15)
    valid = np.ones(len(labels), dtype=bool)
    split = evalkit.build_split(labels, valid, set(labels.tolist()), seed=2, enrolled_fraction=0.6)
    assert evalkit.closed_set_topk(emb, labels, split, "cosine", (1, 5))["top1"] == 1.0
    ms, ok, ns = evalkit.top1_scores(emb, labels, split, "cosine")
    fnir, fpir = evalkit.fnir_fpir(ms, ok, ns, np.array([-1.0, 0.5, 2.0]), 0)
    assert fnir[0] == 0.0 and fpir[0] == 1.0  # accept everything
    assert fnir[2] == 1.0 and fpir[2] == 0.0  # accept nothing
    assert fnir[0] <= fnir[1] <= fnir[2] and fpir[0] >= fpir[1] >= fpir[2]


def test_threshold_for_fpir_meets_target_on_the_calibration_set() -> None:
    rng = np.random.default_rng(0)
    scores = rng.random(2000).astype(np.float32)
    for target in (0.1, 0.01, 0.001):
        t = evalkit.threshold_for_fpir(scores, target)
        assert (scores >= t).mean() <= target + 1e-12


def test_roc_summary_and_logistic_calibration_are_monotone_and_sane() -> None:
    rng = np.random.default_rng(0)
    gen = rng.normal(0.6, 0.1, 5000).astype(np.float32)
    imp = rng.normal(0.0, 0.1, 5000).astype(np.float32)
    roc = evalkit.roc_summary(gen, imp)
    assert roc["auc"] > 0.99 and roc["eer"] < 0.01
    a, b = evalkit.fit_logistic(gen, imp)
    assert a > 0
    p = lambda s: 1 / (1 + math.exp(-(a * s + b)))  # noqa: E731
    assert p(0.0) < 0.05 and p(0.6) > 0.95 and p(0.3) == pytest.approx(0.5, abs=0.05)


def test_pair_scores_use_only_requested_identities() -> None:
    emb, labels = _clusters()
    ids = set(labels.tolist()[:40])
    gen, imp = evalkit.pair_scores(
        emb, labels, np.ones(len(labels), bool), ids, "cosine", 0, n_impostor=500
    )
    assert len(imp) == 500 and len(gen) > 0
    assert gen.mean() > imp.mean()


def test_leave_one_out_retrieval_on_clean_clusters() -> None:
    emb, labels = _clusters(noise=0.15)
    out = evalkit.leave_one_out_retrieval(emb, labels, np.ones(len(labels), bool), "cosine")
    assert out["hit@1"] == 1.0 and out["precision@1"] == 1.0 and out["mAP"] > 0.95
    assert out["precision@10"] < 1.0  # only 3 other photos per person exist


def test_ranking_metrics_for_lookalike_judgments() -> None:
    judgments = {"q1": {"a": 3, "b": 2, "c": 0, "d": 1}, "q2": {"x": 0, "y": 3}}
    perfect = {"q1": ["a", "b", "d", "c"], "q2": ["y", "x"]}
    reversed_ = {"q1": ["c", "d", "b", "a"], "q2": ["x", "y"]}
    good = evalkit.ranking_metrics(perfect, judgments, ks=(1, 2))
    bad = evalkit.ranking_metrics(reversed_, judgments, ks=(1, 2))
    assert good["ndcg@2"] == pytest.approx(1.0) and good["mrr"] == 1.0
    assert good["recall@2"] == 1.0 and good["precision@1"] == 1.0
    assert bad["ndcg@2"] < 0.5 and bad["mrr"] < 0.5
    unjudged = evalkit.ranking_metrics({"q2": ["zzz", "y"]}, judgments, ks=(1,))
    assert unjudged["precision@1"] == 0.0  # unjudged results are graded 0
    assert evalkit.ranking_metrics({"nope": ["a"]}, judgments)["queries"] == 0.0


def test_latency_summary_percentiles() -> None:
    out = evalkit.latency_summary([0.001 * i for i in range(1, 101)])
    assert out["p50_ms"] == pytest.approx(50.5)
    assert out["p99_ms"] > out["p95_ms"] > out["p50_ms"]
    assert evalkit.latency_summary([]) == {}
