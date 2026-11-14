# Architecture

Doppelganger is a self-hosted, CPU-first face search service for a **private gallery** you are
authorised to process. It makes no outbound network calls and uses no paid or cloud service.

```
                      ┌────────────────────────────────────────────┐
  selfie ────────────▶│ api (FastAPI, 1 process)                   │
  curl / client       │  auth ▸ rate limit ▸ size limit ▸ decode   │
                      │  detect+quality ▸ embed ▸ FAISS search     │──▶ ranked matches
                      │  hydrate hits from SQLite                  │
                      └──────────────┬───────────────▲─────────────┘
                                     │ jobs          │ index sync (new rows by seq,
                                     ▼               │ removals by index_events id)
  photos ──▶ staged file ──▶ ┌────────────────────┐  │
  (upload / directory)       │ worker (same image)│  │     ┌──────────────────────────┐
                             │ sha256 dedupe ▸    │──┼────▶│ data volume              │
                             │ store original ▸   │  │     │  doppelganger.db (WAL)   │
                             │ detect ▸ embed ▸   │  │     │  originals/ crops/       │
                             │ store_faces (tx)   │  └─────│  index/<model>/ snapshot │
                             └────────────────────┘        └──────────────────────────┘
```

## Components

| Piece | Module | Notes |
| --- | --- | --- |
| HTTP API | `app/main.py`, `app/api/` | `/v1/*` plus the legacy routes. One uvicorn process (the index and rate limiter are in-process). |
| Detector + embedder | `app/ml/` | YuNet + SFace through OpenCV DNN (`opencv_models.py`). `PipelinePool` holds N independent instances because OpenCV DNN objects are not thread-safe. |
| Quality gate | `app/ml/pipeline.py` | Rejects low-confidence, too small, blurry or border-clipped faces *before* they are indexed or searched. |
| Relational store | `app/db.py`, `migrations/` | SQLite in WAL mode, Alembic migrations. Source of truth. |
| Vector index | `app/index.py` | FAISS `IndexIDMap2(IndexFlatIP)`: exact inner product on L2-normalised vectors = cosine. One index per `model_id`. |
| Jobs / worker | `app/jobs.py`, `app/ingest.py` | SQLite-backed queue with leases, retries and idempotency. |
| Search | `app/search.py` | LOOKALIKE vs SAME_PERSON semantics, pagination, calibration. |
| Security | `app/security.py`, `app/api/deps.py` | Tokens, rate limits, signed URLs. |
| Files | `app/storage.py` | UUID-named originals and crops, atomic 0600 writes. |
| Operator CLI | `app/cli.py` | `python -m app.cli ...` for everything that should not be an HTTP route. |

## Model choice

* **Detector: YuNet 2023mar** (MIT) — small CNN, runs in a few milliseconds on CPU, returns a
  confidence and five landmarks, which the embedder needs for alignment.
* **Embedder: SFace 2021dec** (Apache-2.0 per the OpenCV Zoo) — 128-d, cosine similarity,
  5-landmark `alignCrop` preprocessing, about 39 MB ONNX, runs on CPU through OpenCV with no extra
  runtime.
* Selected because the weights have a documented, redistributable licence label, the runtime
  footprint is just `opencv-python-headless`, and CPU latency is low. Alternatives that were
  rejected, and the licensing uncertainties that remain, are in
  [MODEL_LICENSES.md](MODEL_LICENSES.md). Measured accuracy and latency are in
  [BENCHMARKS.md](BENCHMARKS.md).
* The model is identified by `model_id = sface-2021dec.yunet-2023mar.p1` (the last component is the
  project's preprocessing revision). Anything that changes embeddings changes the id, which makes
  stale vectors detectable and keeps different pipelines apart.

## Data model

* `images` — one row per unique file (sha256 unique). The client filename is a *label*
  (`source_name`); the stored name is a server-generated UUID. Dimensions/format come from the
  decoded content, never from the extension.
* `faces` — one row per detected, eligible face: `seq` (autoincrement integer, the FAISS id), UUID
  `id`, `image_id`, `model_id`, bbox, detection score, quality metrics, the float32 embedding blob,
  crop file name. Deleting an image cascades.
* `image_processing` — per `(image, model_id)` state (`processed`/`no_face`, face count, rejection
  reasons). This is what makes "skip already processed" and `reprocess` possible.
* `index_events` — append-only log of face removals so that every process can replay deletions.
* `jobs` / `job_items` — the queue: per-item status, attempts, lease expiry, error code.

The relational database is the **source of truth**; the FAISS index is a derived cache that can
always be rebuilt (`python -m app.cli index rebuild`) and checked against the database
(`python -m app.cli index check`).

## Vector index and sync

* Exact (flat) inner-product search. At 100 000 × 128-d vectors this is a few milliseconds per
  query on 2 threads (see [BENCHMARKS.md](BENCHMARKS.md)); approximate indexes (HNSW) were
  benchmarked but are not used because exact search is fast enough at this scale and removes recall
  risk.
* **Startup:** load the snapshot `index/<model>/index.faiss` if its `meta.json` (model id, dimension,
  vector count, sha256, last face `seq`, last event id) validates; otherwise discard it and rebuild
  from SQLite. A corrupt or stale snapshot can never produce wrong results silently.
* **Runtime:** the API and the worker are separate processes. Before each search the API calls
  `sync()`, which (rate-limited by `DOPPEL_INDEX_SYNC_INTERVAL_S`) adds faces with `seq` greater
  than the last seen and applies removals from `index_events`. The query itself never loads the
  gallery.
* **Shutdown:** the snapshot is persisted. After a crash the next start replays whatever happened
  after the snapshot.
* Embedding blobs whose size does not match the model dimension raise `IndexUnavailable
  (embedding_size_mismatch)` instead of being reinterpreted.

## Ingestion pipeline

1. `POST /v1/gallery/import` (multipart upload or a directory under `DOPPEL_IMPORT_ROOT`) validates
   intake limits, stages each file under a UUID name in `data/staging/`, creates a job with one
   `job_item` per file and returns `202` with the job id.
2. The worker claims items with a **lease**. A crashed worker's lease expires and the item is
   claimed again, up to `DOPPEL_JOB_MAX_ATTEMPTS`.
3. Per item: read bytes → sha256 → **dedupe** (same bytes already stored → `duplicate`, nothing
   re-embedded) → content-based validation and decode → store the original → detect → quality gate
   → embed every eligible face → `store_faces`, a single transaction that atomically replaces the
   `(image, model)` faces and records delete events. Re-running an item is therefore idempotent.
4. Errors are split into **permanent** (corrupt/unsupported/oversized image, no face — recorded
   once, not retried) and **transient** (database locked, inference pool saturated — requeued until
   `DOPPEL_JOB_MAX_ATTEMPTS` is reached, then marked `failed`; there is no delay between attempts).
   Only an error code and a short message are stored.
5. `GET /v1/jobs/{id}` reports status, counts, progress and per-file error codes (no paths).
   `python -m app.cli reprocess [--all]` queues a job over images already stored and
   `python -m app.cli index rebuild` rebuilds the index.

## Search semantics

| | `LOOKALIKE` (default) | `SAME_PERSON` |
| --- | --- | --- |
| Question answered | "Who looks most like this face?" | "Is this face probably in the gallery?" |
| Threshold | none unless `min_similarity` is passed | yes: calibrated if a calibration file is installed, else the vendor default (reported as `vendor_default_uncalibrated`) |
| Empty result | only when the gallery is empty | `no_reliable_match` when nothing passes |
| Claim | resemblance only, never identity | a *candidate*, never proof of identity |

Every response carries `mode`, `model_id`, the threshold and where it came from, a notice string,
and per-match `cosine_similarity`, `cosine_distance`, `ranking_score` (= cosine, for ordering only),
`calibrated_confidence` (null without a calibration file) and `same_person_candidate`. **Scores are
raw model similarities, not probabilities.**

The selfie goes through the same detector and quality gate as the gallery. If it contains several
faces of similar size the request fails with `multiple_faces_ambiguous` and the candidates' boxes so
the client can resubmit with `face_index`; the service never silently picks "the first face".
Results are de-duplicated to the best face per photo by default and are paginated with
`offset`/`next_offset`.

## Security model

* **Authn:** bearer tokens from the environment (`DOPPEL_ADMIN_TOKEN`; optional
  `DOPPEL_SEARCH_TOKEN` that can search but not import/delete/inspect). Minimum 16 characters,
  compared with `hmac.compare_digest`. The service refuses to start without a token unless
  `DOPPEL_AUTH_DISABLED=true`.
* **Abuse limits:** per-IP and per-principal token buckets (a stricter bucket for heavy routes),
  request-body size limit enforced while streaming, pixel/dimension limits applied from the image
  header before decoding.
* **Files:** UUID-only storage names, 0600 files, 0700 directories; client filenames are never used
  as paths; directory import is confined under `DOPPEL_IMPORT_ROOT` and does not follow symlinks.
* **Output:** originals and face crops are served only through HMAC-signed, expiring URLs; no
  filesystem path ever appears in a response. Errors use one structured shape with a request id and
  never include exception text.
* **Privacy:** selfies are processed in memory and never written to disk; logs and metrics contain
  no embeddings or images. Deleting an image removes the original, its crops, its faces and its
  index entries.
* **Not provided here:** TLS (terminate it at a reverse proxy, see [DEPLOYMENT.md](DEPLOYMENT.md)),
  multi-tenant isolation, user accounts, liveness/anti-spoofing checks.

## Deliberate non-goals and trade-offs

* **One API process.** The in-memory index and rate limiter are per process; scale up (more cores)
  rather than out. The worker is a separate process/container.
* **SQLite instead of Postgres/Qdrant.** Fewer moving parts to deploy offline; the design keeps the
  repository and index behind interfaces (`GalleryRepository`, `VectorIndex`) so they can be
  replaced, but no other backend is implemented or tested.
* **Flat index instead of ANN.** Chosen for exactness; revisit beyond a few million faces.
* **No GPU path.** OpenCV DNN runs on CPU here; a GPU would need a different embedder build.
