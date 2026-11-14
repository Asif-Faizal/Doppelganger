#!/usr/bin/env python
"""Benchmarks: vector index (flat vs HNSW), service-level search (SQLite+FAISS), inference latency.

  python scripts/benchmark_index.py index     --sizes 10000 100000
  python scripts/benchmark_index.py service   --sizes 10000 100000
  python scripts/benchmark_index.py inference --images /path/to/images --count 200

Index and service benchmarks use synthetic clustered unit vectors: they measure speed, memory and
the recall of approximate search *relative to exact search on the same synthetic data*. They say
nothing about face-recognition accuracy (see scripts/evaluate.py for that). Results are printed as
JSON, together with the hardware, and are meant to be pasted into docs/BENCHMARKS.md verbatim.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import evalkit

DIM = 128


def hardware() -> dict[str, Any]:
    import faiss
    import psutil

    model = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "cpu": model,
        "logical_cores": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "gpu": "none",
        "os": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "faiss": faiss.__version__,
    }


def rss_mb() -> float:
    import psutil

    return round(psutil.Process().memory_info().rss / 2**20, 1)


def synthetic(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """n unit vectors grouped into 'identities' of 1-8 vectors (intra-identity cosine ~0.6-0.7)."""
    rng = np.random.default_rng(seed)
    sizes: list[int] = []
    while sum(sizes) < n:
        sizes.append(int(rng.integers(1, 9)))
    sizes[-1] -= sum(sizes) - n
    n_id = len(sizes)
    centres = rng.standard_normal((n_id, DIM)).astype(np.float32)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    owner = np.repeat(np.arange(n_id), sizes)
    noise = rng.standard_normal((n, DIM)).astype(np.float32) * (0.73 / np.sqrt(DIM))
    vecs = centres[owner] + noise
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs.astype(np.float32), owner


def queries_for(vecs: np.ndarray, count: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pick = rng.integers(len(vecs), size=count)
    q = vecs[pick] + rng.standard_normal((count, DIM)).astype(np.float32) * (0.5 / np.sqrt(DIM))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q.astype(np.float32)


def time_single(
    fn: Callable[[np.ndarray], Any], queries: np.ndarray, warmup: int = 20
) -> dict[str, float]:
    for q in queries[:warmup]:
        fn(q)
    secs = []
    for q in queries:
        t0 = time.perf_counter()
        fn(q)
        secs.append(time.perf_counter() - t0)
    s = evalkit.latency_summary(secs)
    s["qps_single_thread"] = round(1.0 / (sum(secs) / len(secs)), 1)
    return {k: round(v, 3) for k, v in s.items()}


def qps_threads(fn: Callable[[np.ndarray], Any], queries: np.ndarray, threads: int) -> float:
    per = len(queries) // threads
    barrier = threading.Barrier(threads + 1)

    def work(chunk: np.ndarray) -> None:
        barrier.wait()
        for q in chunk:
            fn(q)

    ts = [
        threading.Thread(target=work, args=(queries[i * per : (i + 1) * per],))
        for i in range(threads)
    ]
    for t in ts:
        t.start()
    barrier.wait()
    t0 = time.perf_counter()
    for t in ts:
        t.join()
    return round(per * threads / (time.perf_counter() - t0), 1)


# --- index benchmark -----------------------------------------------------------------------------


def _hnsw_runs(vecs: np.ndarray, qs: np.ndarray, exact_ids: list[Any]) -> list[dict[str, Any]]:
    import faiss

    runs: list[dict[str, Any]] = []
    m, efc = 32, 200
    t0 = time.perf_counter()
    h = faiss.IndexHNSWFlat(DIM, m, faiss.METRIC_INNER_PRODUCT)
    h.hnsw.efConstruction = efc
    h.add(vecs)
    build = time.perf_counter() - t0

    def hs(q: np.ndarray) -> Any:
        return h.search(q.reshape(1, -1), 10)

    for ef in (32, 64, 128):
        h.hnsw.efSearch = ef
        found = [set((hs(q)[1][0] + 1).tolist()) for q in qs]
        recall = float(
            np.mean([len(found[i] & set(exact_ids[i].tolist())) / 10 for i in range(len(qs))])
        )
        runs.append(
            {
                "M": m,
                "efConstruction": efc,
                "efSearch": ef,
                "build_s": round(build, 1),
                "recall_at_10_vs_exact": round(recall, 4),
                "latency": time_single(hs, qs),
            }
        )
    return runs


def bench_index(a: argparse.Namespace) -> dict[str, Any]:
    import faiss

    from app.index import FaissFlatIndex

    faiss.omp_set_num_threads(a.faiss_threads)
    results: list[dict[str, Any]] = []
    for n in a.sizes:
        vecs, _ = synthetic(n)
        qs = queries_for(vecs, a.queries)
        ids = np.arange(1, n + 1, dtype=np.int64)
        entry: dict[str, Any] = {"vectors": n, "dim": DIM, "faiss_omp_threads": a.faiss_threads}

        before = rss_mb()
        t0 = time.perf_counter()
        flat = FaissFlatIndex(DIM)
        flat.add(ids, vecs)
        entry["flat_build_s"] = round(time.perf_counter() - t0, 2)
        entry["flat_rss_delta_mb"] = round(rss_mb() - before, 1)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "f.faiss"
            faiss.write_index(flat.raw, str(p))
            entry["flat_snapshot_mb"] = round(p.stat().st_size / 2**20, 1)
            t0 = time.perf_counter()
            faiss.read_index(str(p))
            entry["flat_snapshot_load_s"] = round(time.perf_counter() - t0, 3)

        def search_flat(q: np.ndarray, index: FaissFlatIndex = flat) -> Any:
            return index.search(q, 10)

        entry["flat_k10"] = time_single(search_flat, qs)
        entry["flat_k10_qps_by_threads"] = {
            str(t): qps_threads(search_flat, qs, t) for t in a.threads
        }
        exact_ids = [flat.search(q, 10)[0] for q in qs]
        entry["hnsw_not_in_production"] = _hnsw_runs(vecs, qs, exact_ids)
        results.append(entry)
        print(json.dumps(entry, indent=2), flush=True)
    return {"benchmark": "index", "hardware": hardware(), "results": results}


# --- service benchmark ---------------------------------------------------------------------------


def _bulk_insert(db: Any, model_id: str, vecs: np.ndarray, owner: np.ndarray) -> None:
    import uuid
    from datetime import datetime

    from sqlalchemy import insert

    from app.db import Face, Image

    now = datetime(2026, 1, 1)
    n = len(vecs)
    img_ids = [str(uuid.uuid4()) for _ in range(n)]
    with db.write() as s:
        for lo in range(0, n, 10_000):
            hi = min(lo + 10_000, n)
            s.execute(
                insert(Image),
                [
                    {
                        "id": img_ids[i],
                        "sha256": f"{i:064x}",
                        "storage_name": f"{img_ids[i]}.jpg",
                        "format": "jpeg",
                        "width": 250,
                        "height": 250,
                        "size_bytes": 12_000,
                        "source_name": f"synthetic-{owner[i]}-{i}.jpg",
                        "created_at": now,
                    }
                    for i in range(lo, hi)
                ],
            )
            s.execute(
                insert(Face),
                [
                    {
                        "id": str(uuid.uuid4()),
                        "image_id": img_ids[i],
                        "model_id": model_id,
                        "face_index": 0,
                        "bbox_x": 60.0,
                        "bbox_y": 60.0,
                        "bbox_w": 120.0,
                        "bbox_h": 120.0,
                        "det_score": 0.99,
                        "quality": {},
                        "crop_name": None,
                        "embedding": vecs[i].tobytes(),
                        "created_at": now,
                    }
                    for i in range(lo, hi)
                ],
            )


def _searcher(svc: Any, params: Any) -> Callable[[np.ndarray], Any]:
    def run(q: np.ndarray) -> Any:
        return svc.search_vector(q, params)

    return run


def bench_service(a: argparse.Namespace) -> dict[str, Any]:
    import faiss

    from app import migrate
    from app.config import Settings
    from app.db import Database
    from app.index import IndexManager
    from app.metrics import Metrics
    from app.ml.base import ModelInfo
    from app.repository import SqlGalleryRepository
    from app.search import SearchMode, SearchParams, SearchService

    faiss.omp_set_num_threads(a.faiss_threads)
    results: list[dict[str, Any]] = []
    for n in a.sizes:
        work = Path(tempfile.mkdtemp(prefix="doppel-bench-", dir=a.workdir))
        try:
            settings = Settings(data_dir=work, pipeline="fake", _env_file=None)  # type: ignore[call-arg]
            settings.ensure_dirs()
            migrate.upgrade(settings.database_url or "")
            db = Database(settings.database_url or "")
            repo = SqlGalleryRepository(db)
            info = ModelInfo(
                model_id="bench-synthetic.d128",
                detector="synthetic",
                detector_version="0",
                embedder="synthetic",
                embedder_version="0",
                dim=DIM,
                metric="cosine",
                preprocessing="none",
                production=False,
            )
            vecs, owner = synthetic(n)
            t0 = time.perf_counter()
            _bulk_insert(db, info.model_id, vecs, owner)
            insert_s = time.perf_counter() - t0
            with sqlite3.connect(work / "doppelganger.db") as raw:
                raw.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            db_mb = round((work / "doppelganger.db").stat().st_size / 2**20, 1)

            before = rss_mb()
            idx = IndexManager(repo, info, settings.index_dir)
            t0 = time.perf_counter()
            idx.load()
            cold_rebuild_s = time.perf_counter() - t0
            idx.persist()
            snap_mb = round(
                sum(f.stat().st_size for f in settings.index_dir.rglob("*") if f.is_file()) / 2**20,
                1,
            )
            idx2 = IndexManager(repo, info, settings.index_dir)
            t0 = time.perf_counter()
            idx2.load()
            warm_load_s = time.perf_counter() - t0
            assert idx2.count() == n and not idx2.rebuilt_on_load
            rss_after = rss_mb()

            svc = SearchService(
                settings, cast(Any, SimpleNamespace(info=info)), idx2, repo, Metrics()
            )
            qs = queries_for(vecs, a.queries)
            entry: dict[str, Any] = {
                "faces": n,
                "sqlite_insert_s": round(insert_s, 1),
                "sqlite_mb": db_mb,
                "index_snapshot_mb": snap_mb,
                "cold_index_rebuild_from_sqlite_s": round(cold_rebuild_s, 2),
                "warm_index_snapshot_load_s": round(warm_load_s, 3),
                "rss_after_load_mb": rss_after,
                "rss_delta_mb": round(rss_after - before, 1),
                "faiss_omp_threads": a.faiss_threads,
            }
            for label, params in (
                ("lookalike_top10", SearchParams(mode=SearchMode.LOOKALIKE, max_results=10)),
                ("same_person_top10", SearchParams(mode=SearchMode.SAME_PERSON, max_results=10)),
            ):
                entry[label] = time_single(_searcher(svc, params), qs)
            lookalike = _searcher(svc, SearchParams(max_results=10))
            entry["lookalike_top10_qps_by_threads"] = {
                str(t): qps_threads(lookalike, qs, t) for t in a.threads
            }
            results.append(entry)
            print(json.dumps(entry, indent=2), flush=True)
            db.dispose()
        finally:
            shutil.rmtree(work, ignore_errors=True)
    return {"benchmark": "service", "hardware": hardware(), "results": results}


# --- inference benchmark -------------------------------------------------------------------------


def bench_inference(a: argparse.Namespace) -> dict[str, Any]:
    import cv2

    from app.config import Settings
    from app.ml.opencv_models import build_sface_pipeline_parts
    from app.ml.pipeline import FacePipeline

    exts = {".jpg", ".jpeg", ".png"}
    paths = sorted(p for p in Path(a.images).rglob("*") if p.suffix.lower() in exts)
    rng = np.random.default_rng(0)
    if len(paths) > a.count:
        paths = [paths[int(i)] for i in sorted(rng.choice(len(paths), a.count, replace=False))]
    images = [cv2.imread(str(p)) for p in paths]
    images = [im for im in images if im is not None]
    out: list[dict[str, Any]] = []
    for threads in a.threads:
        cv2.setNumThreads(threads)
        settings = Settings(models_dir=Path(a.models_dir), _env_file=None)  # type: ignore[call-arg]
        det, emb, info = build_sface_pipeline_parts(
            Path(a.models_dir), max_side=settings.detect_max_side, threads=threads
        )
        pipe = FacePipeline(det, emb, info, settings)
        for im in images[:10]:
            pipe.analyze_gallery_image(im)
        det_s, emb_s, tot_s = [], [], []
        for im in images:
            t0 = time.perf_counter()
            analysis = pipe.detect(im)
            t1 = time.perf_counter()
            for c in analysis.eligible[:1]:
                pipe.embed(im, c)
            t2 = time.perf_counter()
            det_s.append(t1 - t0)
            emb_s.append(t2 - t1)
            tot_s.append(t2 - t0)
        out.append(
            {
                "model_id": info.model_id,
                "opencv_threads": threads,
                "images": len(images),
                "image_size_example": list(images[0].shape[:2]),
                "detect": {k: round(v, 2) for k, v in evalkit.latency_summary(det_s).items()},
                "embed_one_face": {
                    k: round(v, 2) for k, v in evalkit.latency_summary(emb_s).items()
                },
                "detect_plus_embed": {
                    k: round(v, 2) for k, v in evalkit.latency_summary(tot_s).items()
                },
            }
        )
        print(json.dumps(out[-1], indent=2), flush=True)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return {
        "benchmark": "inference",
        "hardware": hardware(),
        "peak_rss_mb": round(peak, 1),
        "results": out,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("index", "service"):
        p = sub.add_parser(name)
        p.add_argument("--sizes", type=int, nargs="+", default=[10_000, 100_000])
        p.add_argument("--queries", type=int, default=1000)
        p.add_argument("--threads", type=int, nargs="+", default=[1, 4, 8])
        p.add_argument(
            "--faiss-threads", type=int, default=2, help="matches DOPPEL_INDEX_THREADS default"
        )
        p.add_argument("--workdir", default=None)
        p.add_argument("--out")
    p = sub.add_parser("inference")
    p.add_argument("--images", required=True)
    p.add_argument("--models-dir", default="models")
    p.add_argument("--count", type=int, default=200)
    p.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    p.add_argument("--out")
    args = ap.parse_args()
    fn = {"index": bench_index, "service": bench_service, "inference": bench_inference}[args.cmd]
    result = fn(args)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
