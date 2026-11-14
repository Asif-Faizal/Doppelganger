from __future__ import annotations

import contextlib
import threading
import time
from collections import defaultdict, deque
from collections.abc import Iterator

import numpy as np
import psutil


class Metrics:
    """In-process counters and latency windows. Contains no biometric data and sends nothing out."""

    def __init__(self, window: int = 2048) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = defaultdict(int)
        self._samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self.started_at = time.time()

    def inc(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] += value

    def observe(self, name: str, seconds: float) -> None:
        with self._lock:
            self._samples[name].append(seconds)

    @contextlib.contextmanager
    def timer(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, time.perf_counter() - t0)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            counters = dict(self._counters)
            latencies = {}
            for name, samples in self._samples.items():
                if samples:
                    arr = np.fromiter(samples, dtype=np.float64)
                    latencies[name] = {
                        "count": int(arr.size),
                        "p50_ms": round(float(np.percentile(arr, 50)) * 1000, 2),
                        "p95_ms": round(float(np.percentile(arr, 95)) * 1000, 2),
                        "p99_ms": round(float(np.percentile(arr, 99)) * 1000, 2),
                        "max_ms": round(float(arr.max()) * 1000, 2),
                    }
        proc = psutil.Process()
        return {
            "uptime_s": round(time.time() - self.started_at, 1),
            "counters": counters,
            "latency": latencies,
            "process": {
                "rss_mb": round(proc.memory_info().rss / 1e6, 1),
                "cpu_percent": proc.cpu_percent(interval=None),
                "threads": proc.num_threads(),
            },
        }
