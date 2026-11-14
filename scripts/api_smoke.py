#!/usr/bin/env python3
"""Smoke-test a running Doppelganger API (replaces the old manual test_api.py).

Usage:
    DOPPEL_TOKEN=<admin token> python scripts/api_smoke.py --url http://localhost:8000 \
        --gallery 1.jpg 3.jpg 4.jpg 5.jpg --selfie 2.jpg

Imports the gallery photos through POST /v1/gallery/import, waits for the job, searches the selfie
in both modes and prints the ranked results. Only use photos you are authorised to process.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--gallery", nargs="*", default=[])
    ap.add_argument("--selfie", required=True)
    args = ap.parse_args()
    token = os.environ.get("DOPPEL_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    base = args.url.rstrip("/")

    ready = requests.get(f"{base}/ready", timeout=10)
    print("ready:", ready.status_code, json.dumps(ready.json()["status"]))
    if ready.status_code != 200:
        print(json.dumps(ready.json(), indent=2))
        return 1

    if args.gallery:
        files = [
            ("files", (Path(p).name, Path(p).read_bytes(), "image/jpeg")) for p in args.gallery
        ]
        r = requests.post(f"{base}/v1/gallery/import", files=files, headers=headers, timeout=120)
        print("import:", r.status_code, r.text[:300])
        r.raise_for_status()
        job_id = r.json()["job_id"]
        while True:
            job = requests.get(f"{base}/v1/jobs/{job_id}", headers=headers, timeout=10).json()
            print(f"job {job['status']} progress={job['progress']:.0%} counts={job['counts']}")
            if job["status"] in ("completed", "completed_with_errors", "failed", "cancelled"):
                break
            time.sleep(1)

    for mode in ("LOOKALIKE", "SAME_PERSON"):
        with Path(args.selfie).open("rb") as fh:
            r = requests.post(
                f"{base}/v1/faces/search",
                files={"file": (Path(args.selfie).name, fh, "image/jpeg")},
                data={"mode": mode, "max_results": 5},
                headers=headers,
                timeout=60,
            )
        print(f"\n{mode}: HTTP {r.status_code}")
        body = r.json()
        if r.status_code != 200:
            print(json.dumps(body["error"], indent=2))
            continue
        print(body["notice"])
        print("status:", body["status"], "| threshold:", body["threshold"])
        for m in body["matches"]:
            print(
                f"  #{m['rank']} cos={m['cosine_similarity']:.3f} "
                f"dist={m['cosine_distance']:.3f} image={m['image_id'][:8]} face={m['face_id'][:8]}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
