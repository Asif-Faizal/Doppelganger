#!/usr/bin/env python
"""End-to-end HTTP search latency against a running server (decode + detect + embed + search).

    DOPPEL_TOKEN=<token> python scripts/bench_http.py --url http://127.0.0.1:8000 \
        --images /data/selfies --count 300 --concurrency 1 2 4

Every request is a real multipart upload. Query images are sampled with a fixed seed. Prints JSON
(client-side wall-clock latency, so it includes loopback HTTP and JSON encoding).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalkit


def run(
    base: str, headers: dict[str, str], files: list[Path], mode: str, workers: int
) -> dict[str, Any]:
    lat: list[float] = []
    statuses: dict[str, int] = {}
    lock = threading.Lock()
    it = iter(files)

    def worker() -> None:
        sess = requests.Session()
        while True:
            with lock:
                path = next(it, None)
            if path is None:
                return
            data = path.read_bytes()
            t0 = time.perf_counter()
            r = sess.post(
                f"{base}/v1/faces/search",
                files={"file": (path.name, data, "image/jpeg")},
                data={"mode": mode, "max_results": 10},
                headers=headers,
                timeout=60,
            )
            dt = time.perf_counter() - t0
            with lock:
                lat.append(dt)
                key = str(r.status_code)
                statuses[key] = statuses.get(key, 0) + 1

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    summary = {k: round(v, 2) for k, v in evalkit.latency_summary(lat).items()}
    return {
        "mode": mode,
        "concurrency": workers,
        "requests": len(lat),
        "status_codes": statuses,
        "throughput_rps": round(len(lat) / wall, 1),
        "latency": summary,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--images", required=True)
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1])
    ap.add_argument("--mode", default="LOOKALIKE", choices=["LOOKALIKE", "SAME_PERSON"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args()

    token = os.environ.get("DOPPEL_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    base = args.url.rstrip("/")
    paths = sorted(Path(args.images).rglob("*.jpg"))
    random.Random(args.seed).shuffle(paths)
    files = paths[: args.count]
    ready = requests.get(f"{base}/ready", timeout=10).json()
    out: dict[str, Any] = {"ready": ready["status"], "index": ready["checks"].get("index")}
    out["runs"] = []
    for c in args.concurrency:
        out["runs"].append(run(base, headers, files, args.mode, c))
        print(json.dumps(out["runs"][-1]), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
