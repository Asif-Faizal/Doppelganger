"""Metric and protocol helpers for the offline evaluation harness (pure numpy, no model imports)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

FloatArray = np.ndarray[Any, np.dtype[np.floating[Any]]]
IntArray = np.ndarray[Any, np.dtype[np.integer[Any]]]


def similarity(probe: FloatArray, gallery: FloatArray, metric: str) -> FloatArray:
    """Higher = more similar. cosine for L2-normalised embeddings; negative euclidean otherwise."""
    p = probe.astype(np.float32)
    g = gallery.astype(np.float32)
    if metric == "cosine":
        pn = p / np.linalg.norm(p, axis=1, keepdims=True)
        gn = g / np.linalg.norm(g, axis=1, keepdims=True)
        return np.asarray(pn @ gn.T, dtype=np.float32)
    d2 = (p**2).sum(1)[:, None] + (g**2).sum(1)[None, :] - 2.0 * (p @ g.T)
    return np.asarray(-np.sqrt(np.maximum(d2, 0.0)), dtype=np.float32)


@dataclass(frozen=True)
class Split:
    gallery: IntArray  # indices into the embedding table (1 enrolment image per enrolled identity)
    mated: IntArray  # probes whose identity is enrolled
    non_mated: IntArray  # probes whose identity is not enrolled
    mated_fta: int  # mated probes where no usable face was acquired (counted as misses)
    non_mated_fta: int


def split_identities(
    labels: np.ndarray[Any, Any], seed: int, cal_fraction: float = 0.3
) -> tuple[set[str], set[str]]:
    """Identity-disjoint calibration/test split (held-out calibration, no identity leakage)."""
    ids = sorted(set(labels.tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_cal = round(len(ids) * cal_fraction)
    return set(ids[:n_cal]), set(ids[n_cal:])


def build_split(
    labels: np.ndarray[Any, Any],
    valid: np.ndarray[Any, Any],
    identities: set[str],
    seed: int,
    enrolled_fraction: float = 0.8,
) -> Split:
    """Open-set 1:N protocol inside one identity-disjoint partition.

    Identities with >= 2 images are enrolled with probability `enrolled_fraction` (one random image
    each is the gallery entry, the rest are mated probes). All other images (unenrolled multi-image
    identities, single-image identities) are non-mated probes.
    """
    rng = np.random.default_rng(seed)
    by_id: dict[str, list[int]] = {}
    for i, lab in enumerate(labels.tolist()):
        if lab in identities:
            by_id.setdefault(lab, []).append(i)
    gallery: list[int] = []
    mated: list[int] = []
    non_mated: list[int] = []
    mated_fta = non_mated_fta = 0
    for lab in sorted(by_id):
        idx = by_id[lab]
        usable = [i for i in idx if valid[i]]
        enrol = len(idx) >= 2 and len(usable) >= 1 and rng.random() < enrolled_fraction
        if enrol:
            g = int(usable[int(rng.integers(len(usable)))])
            gallery.append(g)
            for i in idx:
                if i == g:
                    continue
                if valid[i]:
                    mated.append(i)
                else:
                    mated_fta += 1
        else:
            for i in idx:
                if valid[i]:
                    non_mated.append(i)
                else:
                    non_mated_fta += 1
    return Split(
        np.array(gallery, dtype=np.int64),
        np.array(mated, dtype=np.int64),
        np.array(non_mated, dtype=np.int64),
        mated_fta,
        non_mated_fta,
    )


def closed_set_topk(
    emb: FloatArray, labels: np.ndarray[Any, Any], split: Split, metric: str, ks: tuple[int, ...]
) -> dict[str, float]:
    """Top-k identification accuracy over mated probes; failures to acquire count as misses."""
    sim = similarity(emb[split.mated], emb[split.gallery], metric)
    glabels = labels[split.gallery]
    order = np.argsort(-sim, axis=1)
    ranked = glabels[order]
    truth = labels[split.mated][:, None]
    out: dict[str, float] = {}
    total = len(split.mated) + split.mated_fta
    for k in ks:
        hit = (ranked[:, :k] == truth).any(axis=1).sum()
        out[f"top{k}"] = float(hit / total) if total else float("nan")
        out[f"top{k}_detected_only"] = (
            float(hit / len(split.mated)) if len(split.mated) else float("nan")
        )
    return out


def top1_scores(
    emb: FloatArray, labels: np.ndarray[Any, Any], split: Split, metric: str
) -> tuple[FloatArray, np.ndarray[Any, Any], FloatArray]:
    """(mated top-1 score, mated top-1-is-correct, non-mated top-1 score)."""
    glabels = labels[split.gallery]
    sim_m = similarity(emb[split.mated], emb[split.gallery], metric)
    best = sim_m.argmax(axis=1)
    mated_score = sim_m[np.arange(len(best)), best]
    correct = glabels[best] == labels[split.mated]
    sim_n = similarity(emb[split.non_mated], emb[split.gallery], metric)
    return mated_score, correct, sim_n.max(axis=1)


def fnir_fpir(
    mated_score: FloatArray,
    correct: np.ndarray[Any, Any],
    non_mated_score: FloatArray,
    thresholds: FloatArray,
    mated_fta: int,
) -> tuple[FloatArray, FloatArray]:
    """FNIR(t): mated searches without the true mate returned at rank 1 with score >= t.
    FPIR(t): non-mated searches returning any candidate with score >= t."""
    total_m = len(mated_score) + mated_fta
    fnir = np.empty(len(thresholds))
    fpir = np.empty(len(thresholds))
    for i, t in enumerate(thresholds):
        hit = int(((mated_score >= t) & correct).sum())
        fnir[i] = 1.0 - hit / total_m if total_m else float("nan")
        fpir[i] = float((non_mated_score >= t).mean()) if len(non_mated_score) else float("nan")
    return fnir, fpir


def threshold_for_fpir(non_mated_score: FloatArray, target: float) -> float:
    """Smallest threshold whose empirical FPIR on this (calibration) set is <= target."""
    if len(non_mated_score) == 0:
        return float("nan")
    srt = np.sort(non_mated_score)[::-1]
    k = int(np.floor(target * len(srt)))  # allowed false positives
    if k >= len(srt):
        return float(srt[-1])
    return float(np.nextafter(srt[k], np.inf))


def pair_scores(
    emb: FloatArray,
    labels: np.ndarray[Any, Any],
    valid: np.ndarray[Any, Any],
    identities: set[str],
    metric: str,
    seed: int,
    n_impostor: int = 200_000,
    max_genuine: int = 200_000,
) -> tuple[FloatArray, FloatArray]:
    rng = np.random.default_rng(seed)
    idx = [i for i in range(len(labels)) if valid[i] and labels[i] in identities]
    by_id: dict[str, list[int]] = {}
    for i in idx:
        by_id.setdefault(labels[i], []).append(i)
    gen_pairs: list[tuple[int, int]] = []
    for members in by_id.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                gen_pairs.append((members[a], members[b]))
    if len(gen_pairs) > max_genuine:
        sel = rng.choice(len(gen_pairs), max_genuine, replace=False)
        gen_pairs = [gen_pairs[int(s)] for s in sel]
    arr = np.array(idx)
    imp_pairs: list[tuple[int, int]] = []
    while len(imp_pairs) < n_impostor and len(arr) > 1:
        a = arr[rng.integers(len(arr), size=n_impostor)]
        b = arr[rng.integers(len(arr), size=n_impostor)]
        keep = labels[a] != labels[b]
        imp_pairs.extend(zip(a[keep].tolist(), b[keep].tolist(), strict=True))
    imp_pairs = imp_pairs[:n_impostor]

    def score(pairs: list[tuple[int, int]]) -> FloatArray:
        if not pairs:
            return np.empty(0, dtype=np.float32)
        a = np.array([p[0] for p in pairs])
        b = np.array([p[1] for p in pairs])
        ea, eb = emb[a].astype(np.float32), emb[b].astype(np.float32)
        if metric == "cosine":
            ea = ea / np.linalg.norm(ea, axis=1, keepdims=True)
            eb = eb / np.linalg.norm(eb, axis=1, keepdims=True)
            return np.asarray((ea * eb).sum(1), dtype=np.float32)
        return np.asarray(-np.linalg.norm(ea - eb, axis=1), dtype=np.float32)

    return score(gen_pairs), score(imp_pairs)


def roc_summary(genuine: FloatArray, impostor: FloatArray) -> dict[str, float]:
    """Verification metrics: AUC, EER, TAR at fixed FAR (thresholds taken from the impostor set)."""
    if len(genuine) == 0 or len(impostor) == 0:
        return {}
    out: dict[str, float] = {}
    scores = np.concatenate([genuine, impostor])
    labels = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos, n_neg = len(genuine), len(impostor)
    out["auc"] = float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    ts = np.unique(np.quantile(scores, np.linspace(0, 1, 2001)))
    far = np.array([(impostor >= t).mean() for t in ts])
    frr = np.array([(genuine < t).mean() for t in ts])
    i = int(np.argmin(np.abs(far - frr)))
    out["eer"] = float((far[i] + frr[i]) / 2)
    out["eer_threshold"] = float(ts[i])
    for target in (1e-2, 1e-3):
        t = float(np.quantile(impostor, 1 - target))
        out[f"tar_at_far_{target:g}"] = float((genuine >= t).mean())
        out[f"threshold_at_far_{target:g}"] = t
    return out


def fit_logistic(
    genuine: FloatArray, impostor: FloatArray, iters: int = 100
) -> tuple[float, float]:
    """P(same | score) = sigmoid(a*score + b), class-balanced (i.e. assuming a 1:1 prior)."""
    x = np.concatenate([genuine, impostor]).astype(np.float64)
    y = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    w = np.where(y == 1, 0.5 / len(genuine), 0.5 / len(impostor))
    a, b = 1.0, 0.0
    for _ in range(iters):
        z = np.clip(a * x + b, -50, 50)
        p = 1.0 / (1.0 + np.exp(-z))
        g = np.array([(w * (p - y) * x).sum(), (w * (p - y)).sum()])
        s = w * p * (1 - p)
        h = np.array(
            [[(s * x * x).sum(), (s * x).sum()], [(s * x).sum(), s.sum()]]
        ) + 1e-9 * np.eye(2)
        step = np.linalg.solve(h, g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-10:
            break
    return float(a), float(b)


def leave_one_out_retrieval(
    emb: FloatArray,
    labels: np.ndarray[Any, Any],
    valid: np.ndarray[Any, Any],
    metric: str,
    ks: tuple[int, ...] = (1, 5, 10),
    max_probes: int = 3000,
    seed: int = 0,
    chunk: int = 512,
) -> dict[str, float]:
    """Photo-gallery retrieval: every usable image is a gallery photo; each probe (from an identity
    with >= 2 usable photos) is searched against all *other* photos. hit@k = a photo of the same
    person appears in the top-k; precision@k = share of the top-k that are the same person."""
    idx = np.flatnonzero(valid)
    lab = labels[idx]
    uniq, counts = np.unique(lab, return_counts=True)
    multi = set(uniq[counts >= 2].tolist())
    probes_local = np.array([i for i, lbl in enumerate(lab.tolist()) if lbl in multi])
    rng = np.random.default_rng(seed)
    if len(probes_local) > max_probes:
        probes_local = rng.choice(probes_local, max_probes, replace=False)
    e = emb[idx]
    hit = dict.fromkeys(ks, 0)
    prec = dict.fromkeys(ks, 0.0)
    ap_sum = 0.0
    for s in range(0, len(probes_local), chunk):
        part = probes_local[s : s + chunk]
        sim = similarity(e[part], e, metric)
        sim[np.arange(len(part)), part] = -np.inf
        full_order = np.argsort(-sim, axis=1)
        order = full_order[:, : max(ks)]
        for r, p in enumerate(part):
            same = lab[full_order[r]] == lab[p]
            same = same[:-1]  # drop the excluded self at the end (-inf)
            for k in ks:
                top = lab[order[r, :k]] == lab[p]
                hit[k] += int(top.any())
                prec[k] += float(top.mean())
            n_rel = int(same.sum())
            if n_rel:
                cum = np.cumsum(same)
                ap_sum += float((cum[same] / (np.flatnonzero(same) + 1)).sum() / n_rel)
    n = len(probes_local)
    out: dict[str, float] = {"probes": float(n), "gallery_photos": float(len(idx))}
    for k in ks:
        out[f"hit@{k}"] = hit[k] / n if n else float("nan")
        out[f"precision@{k}"] = prec[k] / n if n else float("nan")
    out["mAP"] = ap_sum / n if n else float("nan")
    return out


# --- LOOKALIKE: graded human judgments -----------------------------------------------------------


def ndcg_at_k(gains: list[float], ideal_gains: list[float], k: int) -> float:
    def dcg(g: list[float]) -> float:
        return float(sum((2**x - 1) / np.log2(i + 2) for i, x in enumerate(g[:k])))

    ideal = dcg(sorted(ideal_gains, reverse=True))
    return dcg(gains) / ideal if ideal > 0 else 0.0


def ranking_metrics(
    rankings: dict[str, list[str]],
    judgments: dict[str, dict[str, int]],
    ks: tuple[int, ...] = (1, 5, 10),
    relevant_grade: int = 2,
) -> dict[str, float]:
    """Mean nDCG@k, precision@k, recall@k and MRR over queries that have judgments.

    judgments[query][candidate] = grade (0 = not similar ... 3 = strong look-alike).
    Unjudged candidates count as grade 0 (state this in the report: it penalises unjudged results).
    """
    qs = [q for q in rankings if q in judgments]
    out: dict[str, float] = {"queries": float(len(qs))}
    for k in ks:
        nd, pk, rk = [], [], []
        for q in qs:
            j = judgments[q]
            ranked = rankings[q][:k]
            gains = [float(j.get(c, 0)) for c in ranked]
            nd.append(ndcg_at_k(gains, [float(v) for v in j.values()], k))
            rel_total = sum(1 for v in j.values() if v >= relevant_grade)
            rel_hit = sum(1 for c in ranked if j.get(c, 0) >= relevant_grade)
            pk.append(rel_hit / k)
            rk.append(rel_hit / rel_total if rel_total else 0.0)
        out[f"ndcg@{k}"] = float(np.mean(nd)) if nd else float("nan")
        out[f"precision@{k}"] = float(np.mean(pk)) if pk else float("nan")
        out[f"recall@{k}"] = float(np.mean(rk)) if rk else float("nan")
    rr = []
    for q in qs:
        pos = next(
            (i for i, c in enumerate(rankings[q]) if judgments[q].get(c, 0) >= relevant_grade), None
        )
        rr.append(0.0 if pos is None else 1.0 / (pos + 1))
    out["mrr"] = float(np.mean(rr)) if rr else float("nan")
    return out


def latency_summary(seconds: list[float]) -> dict[str, float]:
    if not seconds:
        return {}
    a = np.asarray(seconds) * 1000.0
    return {
        "n": float(len(a)),
        "mean_ms": float(a.mean()),
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "max_ms": float(a.max()),
    }
