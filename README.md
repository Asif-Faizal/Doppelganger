# Doppelganger

Self-hosted, CPU-only face search for a **private photo gallery you are authorised to process**.
Upload a selfie and get the most similar faces from your gallery.

* No paid or cloud services, no outbound network calls, no model downloads at startup or at search time.
* Two honest search modes: **`LOOKALIKE`** (default; "who looks most like this?", never claims
  identity) and **`SAME_PERSON`** (threshold-based candidates, never proof of identity).
* Vector index (FAISS) instead of load-everything search; background ingestion with retries,
  deduplication and re-processing; token auth, rate limits, signed file URLs, no path leaks.
* Models: OpenCV **YuNet** (detector, MIT) + **SFace** (embedder, Apache-2.0 per OpenCV Zoo; with
  unresolved upstream questions, see [docs/MODEL_LICENSES.md](docs/MODEL_LICENSES.md)).

> **Correction to earlier versions.** Earlier READMEs claimed FaceNet, MTCNN, 512-d embeddings and
> "99 %+ accuracy". None of that was true: the old code used dlib's HOG detector and dlib's ResNet
> (128-d, euclidean). This version documents what the code really does and reports **measured**
> results only ([docs/BENCHMARKS.md](docs/BENCHMARKS.md)).

Only use this on photos of people who have consented, or that you are otherwise legally entitled to
process. Face matching on biometric data is regulated in many jurisdictions. Resemblance is not
identity: never use a result as the sole basis for a decision about a person.

Documentation: [ARCHITECTURE](docs/ARCHITECTURE.md) · [DEPLOYMENT](docs/DEPLOYMENT.md) ·
[MODEL_LICENSES](docs/MODEL_LICENSES.md) · [BENCHMARKS](docs/BENCHMARKS.md) ·
[EVALUATION](docs/EVALUATION.md) · [MIGRATION_FROM_DLIB](docs/MIGRATION_FROM_DLIB.md)

## Quick start (Python 3.12, Linux/macOS)

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 1. Provision the models once (explicit download, sha256-verified). Air-gapped? copy the two
#    .onnx files into ./models/ and run only `verify`.
python -m app.cli models fetch
python -m app.cli models verify

# 2. Configure (a token is mandatory; tokens are never logged)
cp .env.example .env
echo "DOPPEL_ADMIN_TOKEN=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')" >> .env
set -a; . ./.env; set +a

# 3. Launch the API (applies DB migrations) and, in a second terminal, the worker
python run.py                        # http://127.0.0.1:8000  (interactive docs at /docs)
python -m app.cli worker             # or DOPPEL_EMBEDDED_WORKER=true to run it inside the API
```

Docker Compose (not exercised on the build machine, see the note in
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)):

```bash
cp .env.example .env && echo "DOPPEL_ADMIN_TOKEN=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')" >> .env
mkdir -p models photos               # copy the two .onnx files into ./models first
docker compose up -d --build         # services: api, worker (nothing else)
```

## Import a gallery and search a selfie

```bash
export TOKEN=$DOPPEL_ADMIN_TOKEN     # the same value the server was started with
BASE=http://127.0.0.1:8000

# a) import photos by upload (multipart) -> returns a job id immediately (202)
curl -s -X POST $BASE/v1/gallery/import -H "Authorization: Bearer $TOKEN" \
     -F files=@1.jpg -F files=@3.jpg -F files=@4.jpg -F files=@5.jpg
#    {"job_id":"4c462c7f-...","status":"queued","total":4,"rejected_at_intake":0,"status_url":"/v1/jobs/4c46..."}

# b) ...or import a whole directory (must live under DOPPEL_IMPORT_ROOT, symlinks are not followed)
curl -s -X POST $BASE/v1/gallery/import -H "Authorization: Bearer $TOKEN" -F directory=holiday-2025
python -m app.cli import /path/to/photos --recursive --wait      # operator shortcut

# c) follow the job
curl -s $BASE/v1/jobs/<job_id> -H "Authorization: Bearer $TOKEN"

# d) search with a selfie (LOOKALIKE is the default)
curl -s -X POST $BASE/v1/faces/search -H "Authorization: Bearer $TOKEN" \
     -F file=@2.jpg -F max_results=5

# e) only candidates that pass the SAME_PERSON threshold (may return none)
curl -s -X POST $BASE/v1/faces/search -H "Authorization: Bearer $TOKEN" \
     -F file=@2.jpg -F mode=SAME_PERSON

# f) command line, no server needed
python -m app.cli search 2.jpg --mode LOOKALIKE --max-results 5
```

An end-to-end script that does the same and prints the ranking:
`DOPPEL_TOKEN=$TOKEN python scripts/api_smoke.py --gallery 1.jpg 3.jpg 4.jpg 5.jpg --selfie 2.jpg`.

Other calls:

```bash
curl -s $BASE/ready                                                  # readiness (no auth)
curl -s $BASE/v1/admin/metrics -H "Authorization: Bearer $TOKEN"     # operational metrics
curl -s -X DELETE $BASE/v1/images/<image_id> -H "Authorization: Bearer $TOKEN"   # delete an image + its faces
curl -s -X POST $BASE/v1/jobs/<job_id>/cancel -H "Authorization: Bearer $TOKEN"
```

### What a search returns

Real response (shortened) from the five photos in this repository, `2.jpg` as the selfie, gallery
= `1.jpg 3.jpg 4.jpg 5.jpg`:

```json
{
  "mode": "LOOKALIKE",
  "model_id": "sface-2021dec.yunet-2023mar.p1",
  "status": "ok",
  "notice": "LOOKALIKE ranks faces by visual similarity under this model. Resemblance is not identity: high-ranked results can be different people.",
  "threshold": null,
  "gallery_faces": 4,
  "matches": [
    {
      "rank": 1,
      "image_id": "859c96a3-5de6-415e-8195-b177b501f5e7",
      "face_id": "9629efef-6813-4723-a589-ca18e152f735",
      "cosine_similarity": 0.735606,
      "cosine_distance": 0.264394,
      "ranking_score": 0.735606,
      "calibrated_confidence": null,
      "same_person_candidate": null,
      "bbox": {"x": 752.06, "y": 73.71, "width": 229.66, "height": 318.62},
      "image_url": "/v1/images/859c96a3-.../file?exp=1794668920&sig=fb29...",
      "thumbnail_url": "/v1/faces/9629efef-.../thumbnail?exp=1794668920&sig=611f..."
    }
  ],
  "next_offset": 3,
  "calibration": null
}
```

* `cosine_similarity` is the raw model score (higher = more similar, range −1..1). **It is not a
  probability** and its meaning is specific to this model.
* `calibrated_confidence` is `null` unless you install a calibration file fitted on labelled data
  (`DOPPEL_CALIBRATION_FILE`, produced by [`scripts/evaluate.py`](docs/EVALUATION.md)). Even then it
  is only valid for photos that resemble the calibration data.
* If there is nothing to report the response says so: `status` is `empty_gallery` or
  (`SAME_PERSON`) `no_reliable_match`, with `matches: []`.
* Image and thumbnail URLs are signed, expiring links; no filesystem path is ever returned.
* If the selfie contains several faces of similar size the call fails with
  `multiple_faces_ambiguous` and lists the faces; resend with `-F face_index=<n>`.

### The two modes

| | `LOOKALIKE` (default) | `SAME_PERSON` |
| --- | --- | --- |
| Use for | "Show me who looks like this" | "Is this person probably in the gallery?" |
| Threshold | none (unless you pass `min_similarity`) | calibrated threshold if installed, else the vendor default 0.363 (`threshold_source: vendor_default_uncalibrated`) |
| Result | always the nearest faces | only candidates above the threshold; possibly none |
| Claim | resemblance | a *candidate*, not proof |

**Threshold warning.** 0.363 is the vendor's *pairwise verification* threshold. In a gallery of
several hundred identities, 1:N search over many comparisons produces false candidates much more
often than a pairwise rate suggests (on LFW with ~690 enrolled people, 0.363 returned some candidate
for ~40 % of people who are *not* in the gallery; see [BENCHMARKS](docs/BENCHMARKS.md)). Calibrate on
your own data before using `SAME_PERSON` for anything that matters.

## API summary

| Route | Auth | Purpose |
| --- | --- | --- |
| `POST /v1/faces/search` | search or admin | selfie search (`mode`, `max_results`, `offset`, `min_similarity`, `face_index`, `dedupe_by_image`) |
| `POST /v1/gallery/import` | admin | queue uploaded files or a directory; `202` + job id |
| `GET /v1/jobs/{job_id}` · `POST /v1/jobs/{job_id}/cancel` | admin | job progress, counts, per-file error codes |
| `DELETE /v1/images/{image_id}` | admin | remove original, crops, embeddings, index entries |
| `GET /v1/images/{id}/file`, `GET /v1/faces/{id}/thumbnail` | signed URL | serve files |
| `GET /ready` · `GET /health` | none | readiness (db, schema, models, index, worker) · liveness |
| `GET /v1/admin/metrics`, `GET /v1/admin/index/check`, `POST /v1/admin/index/rebuild` | admin | operations |
| `POST /upload-images`, `POST /search-face`, `GET /stats`, `GET /health`, `GET /` | as above | **legacy routes, kept** |

Errors always look like `{"detail": "...", "error": {"code": "...", "message": "...", "details": {},
"request_id": "..."}}` and never contain exception text or paths. Common codes: `unauthorized`
(401), `file_too_large` (413), `unsupported_image_format` (415), `no_face_detected`,
`face_quality_insufficient`, `multiple_faces_ambiguous` (422), `rate_limited` (429),
`model_unavailable` / `index_unavailable` (503).

### Legacy API compatibility

| Legacy route | Still works | What changed (on purpose) |
| --- | --- | --- |
| `POST /upload-images` (1-2 files) | yes | Now requires the admin token; returns UUID `image_id`s, no paths; per-file `status`; duplicates (same sha256) are detected; every eligible face is indexed, not just the first; no more `str(exception)` in responses. |
| `POST /search-face` | yes | Now behaves as `SAME_PERSON`; `threshold` is a **cosine distance** (`1 − cosine similarity`) instead of dlib euclidean distance; `similarity_score` is the clamped cosine; `distance` is the cosine distance. Responses add `mode`, `model_id` and a `notice`. |
| `GET /stats` | yes | No filesystem paths; adds `active_model_id` and `faces_by_model`. Requires a token. |
| `GET /health`, `GET /` | yes | Unchanged shape; `/health` never touches the database. |

Old embeddings cannot be reused (different model, different metric): see
[docs/MIGRATION_FROM_DLIB.md](docs/MIGRATION_FROM_DLIB.md). `python init_db.py` no longer drops
tables; it only applies schema migrations to the new database (`data/doppelganger.db`). The legacy
`storage/embeddings.db` and `storage/images/` are never modified.

## Operations cheat sheet

```bash
python -m app.cli stats                        # counters, faces per model_id
python -m app.cli check [--fix]                # consistency of DB, index and files
python -m app.cli index rebuild                # rebuild the vector index from the database
python -m app.cli reprocess --all --wait       # recompute faces from stored originals
python -m app.cli backup /backups/doppel-$(date +%F)
python -m app.cli delete-image <image_id>      # deletion of one image and all derived data
python -m app.cli migrate-legacy               # re-embed the old dlib-era gallery
```

Backup, restore, recovery, retention and deletion procedures: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Calibration

```bash
python scripts/evaluate.py extract  --dataset /data/my_labelled_photos --out eval_out/clean.npz --workers 4
python scripts/evaluate.py identify --gallery eval_out/clean.npz --write-calibration data/calibration.json
export DOPPEL_CALIBRATION_FILE=data/calibration.json    # then restart the API
```

Layout of the dataset: `root/<person>/<photo>.jpg`. The protocol, the held-out split, the metrics
and the LOOKALIKE human-judgement procedure are described in [docs/EVALUATION.md](docs/EVALUATION.md).

## Development

```bash
pip install -r requirements-dev.txt
ruff format . && ruff check . && mypy
pytest                          # fast suite; uses a deterministic fake embedder, needs no models
pytest -m models                # real-model integration tests (needs ./models/*.onnx, offline)
python scripts/benchmark_index.py index --sizes 10000 100000     # see docs/BENCHMARKS.md
```

`DOPPEL_PIPELINE=fake` swaps in the fake models for development and CI. **Never use it with real
data**: it produces meaningless similarities.

## Troubleshooting

See the table in [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md#7-troubleshooting). The most common ones:
the service refuses to start without `DOPPEL_ADMIN_TOKEN` (or `DOPPEL_AUTH_DISABLED=true` for local
development), and refuses to start with missing or mismatching model files (`models fetch` /
`models verify`). `GET /ready` explains which dependency is not ready.

## Layout

```
app/            service (api/, ml/, index, ingest, jobs, search, security, cli)
migrations/     Alembic migrations (SQLite)
models/         manifest.json (pinned sha256); *.onnx are provisioned, not committed
scripts/        evaluate.py, benchmark_index.py, api_smoke.py
docs/           architecture, deployment, licences, benchmarks, evaluation, migration
tests/          pytest suite
Dockerfile, docker-compose.yml, .env.example, requirements*.txt
```

## License

This repository does not ship a `LICENSE` file for its own code; the owner should add one before
distributing it. The model weights are **not** part of this repository and carry their own terms:
read [docs/MODEL_LICENSES.md](docs/MODEL_LICENSES.md).
