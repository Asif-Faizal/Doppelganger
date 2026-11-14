# Deployment guide

> **Verification status.** The Python service, CLI, worker, index and tests were run on Linux
> (Python 3.12, no Docker available on the build machine). The `Dockerfile` and
> `docker-compose.yml` were written against the same entry points but were **not built or run**
> during development. Treat the first `docker compose up --build` as part of your acceptance test;
> `docker compose config` and the health checks below tell you quickly whether anything needs
> adjusting.

Sizing guide (measured numbers in [BENCHMARKS.md](BENCHMARKS.md)): 2 vCPU / 2 GB RAM is enough for a
gallery of tens of thousands of faces. Each indexed face costs 512 B of vector in RAM plus the
SQLite row; each inference instance holds a copy of the ~39 MB SFace model.

## 1. Provision the models (once, explicit)

The service never downloads anything at startup or while searching. Models are placed on disk and
verified against the sha256 values in `models/manifest.json`.

Connected machine:

```bash
python -m app.cli models fetch      # downloads the two pinned files into ./models and verifies sha256
python -m app.cli models verify     # prints ok / the failure reason per file
```

Air-gapped target: copy the two `.onnx` files into `./models/` yourself (USB, internal mirror, ...)
and run `python -m app.cli models verify` on the target. A wrong or truncated file fails with
`model_checksum_mismatch`; a missing one with `model_file_missing`.

## 2. Docker Compose (recommended)

```bash
cp .env.example .env
python3 -c "import secrets; print('DOPPEL_ADMIN_TOKEN=' + secrets.token_urlsafe(32))" >> .env
mkdir -p models photos              # models/*.onnx from step 1; photos/ = optional import directory
docker compose up -d --build
docker compose ps                   # api: healthy (uses /ready), worker: healthy (heartbeat)
curl -s localhost:8000/ready | python3 -m json.tool
```

What you get:

* `api` and `worker`, the same image, running as UID 10001 with a read-only root filesystem, all
  capabilities dropped and `no-new-privileges`. Nothing else: no database container, no broker.
* Persistent named volume `doppel-data` (database, originals, face crops, index snapshots,
  URL-signing key). Models are bind-mounted read-only; `./photos` is mounted read-only at `/photos`
  for directory imports (`DOPPEL_IMPORT_ROOT=/photos`).
* The API listens on `127.0.0.1:8000` of the host only. Publish it deliberately (see TLS below).
* `restart: unless-stopped`; the worker starts only after the API is healthy (so only one process
  applies schema migrations).

Notes:

* Run **one** `api` container. The in-memory index and the rate limiter are per process.
* SQLite in WAL mode is shared by two containers through the volume: this works because both are on
  the same host and kernel. Do **not** put the data volume on NFS/SMB.
* If `DOPPEL_PHOTOS_DIR` does not exist Docker creates it as root; create it yourself first.

### Without Docker

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt           # for offline hosts: pip download on a connected machine, then
                                          #   pip install --no-index --find-links wheelhouse -r requirements.txt
python -m app.cli models verify
python -m app.cli db upgrade
python run.py &                           # API   (DOPPEL_HOST/DOPPEL_PORT)
python -m app.cli worker &                # worker
```

For small installs `DOPPEL_EMBEDDED_WORKER=true` runs the worker inside the API process and no
second process is needed. Use your init system (systemd, supervisord) to restart both.

## 3. Offline / air-gapped deployment

1. On a connected build machine: `docker compose build`, `docker save doppelganger:local | gzip >
   doppelganger.tar.gz`, and `python -m app.cli models fetch`.
2. Move `doppelganger.tar.gz`, `docker-compose.yml`, `.env.example` and `models/*.onnx` to the
   target.
3. On the target: `docker load < doppelganger.tar.gz`, then `docker compose up -d` (drop the
   `build:` key, or use `--no-build`).

The service itself makes no outbound connections and needs no DNS or internet access at runtime;
clients connect inbound only. You can confirm this on your own network by denying egress for the
containers and checking `/ready` and a search still work.

## 4. TLS and exposure

The service speaks plain HTTP and is meant for a **LAN or loopback**. Terminate TLS in a reverse
proxy and keep the container port bound to `127.0.0.1`. Example Caddyfile:

```
faces.example.lan {
    request_body { max_size 12MB }
    reverse_proxy 127.0.0.1:8000
}
```

Set `DOPPEL_PUBLIC_BASE_URL=https://faces.example.lan` so signed image URLs are absolute, and
`DOPPEL_CORS_ORIGINS=https://your-frontend.example` only if a browser app calls the API directly
(default: no CORS). Behind a proxy all clients share the proxy's IP for the per-IP rate limit; keep
the per-principal limits in mind or forward the real client address with uvicorn's
`--proxy-headers` if you adapt `run.py`.

Do **not** expose the service to the public internet: it has no user accounts, no CAPTCHA and
no per-tenant isolation.

## 5. Configuration

All settings are environment variables with the `DOPPEL_` prefix; the annotated list is
[`.env.example`](../.env.example). The ones that matter most:

| Variable | Meaning |
| --- | --- |
| `DOPPEL_ADMIN_TOKEN` | Required. Full access. Min 16 characters. |
| `DOPPEL_SEARCH_TOKEN` | Optional. May search only. |
| `DOPPEL_INFERENCE_INSTANCES` / `_THREADS` | Concurrent inferences / OpenCV threads each. Start with instances = cores/2. |
| `DOPPEL_SAME_PERSON_MIN_SIMILARITY` | Fallback threshold for `SAME_PERSON` when there is no calibration file. See the calibration warning in the README. |
| `DOPPEL_CALIBRATION_FILE` | Output of `scripts/evaluate.py identify --write-calibration`. |
| `DOPPEL_MAX_UPLOAD_BYTES`, `_MAX_PIXELS`, `_MAX_DIMENSION` | Intake limits. |
| `DOPPEL_RATE_LIMIT_PER_MINUTE`, `_SEARCH_RATE_LIMIT_PER_MINUTE` | Per-principal limits. |

## 6. Operations

### Health and monitoring

* `GET /health` — liveness (process is up, no secrets, no dependency checks).
* `GET /ready` — `200` only when the database schema is current, the models are loaded and
  verified, and the index is loaded; `503` with the failing check otherwise. Reports the worker
  heartbeat age.
* `GET /v1/admin/metrics` (admin token) — request counts, latency percentiles, queue depth, failed
  items, RSS. No biometric data.
* `python -m app.cli worker-health` — exit code 0 while the worker heartbeat is fresh.

### Backup

```bash
python -m app.cli backup /backups/doppel-$(date +%F)      # consistent copy of DB + originals + crops
# in Docker:
docker compose run --rm --no-deps -v /backups:/backups api python -m app.cli backup /backups/doppel-$(date +%F)
```

The SQLite database is copied with SQLite's online-backup API, so it is consistent while the
service is running. The vector index is **derived** and is not backed up. Backups contain original
photos and biometric templates: encrypt them and apply the same retention as the gallery. Keep the
`.env` (tokens) in your secret store, not in the backup.

### Restore

1. Stop `api` and `worker` (`docker compose down`).
2. Copy `doppelganger.db`, `originals/` and `crops/` from the backup into the data volume/directory
   (for Docker, a throwaway container with the volume mounted works) and fix ownership to UID 10001.
3. Delete `index/` from the data directory (or just start; a snapshot that does not match the
   database is discarded automatically).
4. Start the services, then `python -m app.cli check` and `python -m app.cli stats`.

### Index maintenance and recovery

| Situation | Action |
| --- | --- |
| Index snapshot missing/corrupt, or the API was killed | Nothing: the next start rebuilds from SQLite. |
| Suspect drift between DB and index | `python -m app.cli index check` (report only) then `python -m app.cli index rebuild`. |
| Orphan files or drift between DB, index and files | `python -m app.cli check` to report, `python -m app.cli check --fix` to remove orphan files and rebuild a drifted index. |
| Database file lost, originals intact | Restore the last backup of the DB; images added after it are re-imported with `python -m app.cli import <dir>` (sha256 dedupe skips what exists). Or rebuild from originals alone: new DB, `import data/originals`. |
| Worker crashed mid-import | Nothing: item leases expire and items are claimed again (up to `DOPPEL_JOB_MAX_ATTEMPTS`). |
| Changed detector/embedder/preprocessing (new `model_id`) | `python -m app.cli reprocess --all --wait`, then `purge-model <old-id> --yes`. |
| Stale uploads in `data/staging/` | `python -m app.cli sweep-staging`. |

### Retention and deletion

* Delete one image (original, crops, faces, index entries):
  `curl -X DELETE -H "Authorization: Bearer $DOPPEL_ADMIN_TOKEN" localhost:8000/v1/images/<image_id>`
  or `python -m app.cli delete-image <image_id>`. Deleting an unknown id returns `404`; the
  operation is safe to repeat.
* Delete everything: stop the services and remove the data volume (`docker compose down -v`).
  Remember backups.
* Retire a model's embeddings: `python -m app.cli purge-model <model_id> --yes`.
* Selfies used for search are never stored. Uploaded gallery files are staged under UUID names and
  removed after processing; leftovers are cleaned by `sweep-staging`.
* The service does not delete anything automatically on a timer. If you have a retention period,
  run your own scheduled job that lists images and calls the delete endpoint.

### Upgrading

* Application updates: `git pull`, `docker compose up -d --build`. Schema changes are Alembic
  migrations applied automatically by the API at startup (`DOPPEL_AUTO_MIGRATE`), and downgraded
  only by an explicit operator command. Take a backup first.
* **Dependencies** are pinned in `requirements.txt`. To upgrade: create a clean virtualenv, install
  the new versions of the packages listed in `requirements-dev.txt`/`requirements.txt`, run
  `ruff check`, `mypy`, `pytest` and `pytest -m models` (with the real models present), then
  `pip freeze` the runtime subset back into `requirements.txt`. Changing the OpenCV version can
  change numerical output slightly; re-run `scripts/evaluate.py` if you rely on a calibration.
* **Models:** a new model means a new `model_id` and a full `reprocess`; the old vectors keep
  working for the old id until you purge them.

## 7. Troubleshooting

| Symptom | Likely cause / fix |
| --- | --- |
| API exits at startup with `model_file_missing` / `model_checksum_mismatch` | Weights not provisioned or corrupted: step 1. For development without models set `DOPPEL_PIPELINE=fake` (never for real data). |
| API exits with an authentication configuration error | Set `DOPPEL_ADMIN_TOKEN` (min 16 chars) or `DOPPEL_AUTH_DISABLED=true` for local development only. |
| `/ready` returns 503 | Read `checks` in the body: `models`, `index`, `schema` or `database` tells you which. |
| `401 unauthorized` | Missing/incorrect `Authorization: Bearer <token>` header. |
| `429` | Rate limit; honour `Retry-After`. |
| `413 file_too_large` / `415 unsupported_image_format` | See limits above; formats are decided from file contents. |
| `422 no_face_detected` / `face_quality_insufficient` | Photo has no usable face (too small, blurry, cut off); the response lists the reasons. |
| `422 multiple_faces_ambiguous` | Several similar-size faces; resubmit with `face_index` from the response, or crop. |
| Import job stays `pending` | The worker is not running (`docker compose ps`, `worker-health`), or `DOPPEL_EMBEDDED_WORKER=false` with no worker started. |
| `database is locked` in worker logs | Transient; items are retried. Persistent: the data volume is on a network filesystem. |
| Search returns `empty_gallery` | No faces indexed for the *active* `model_id`: `python -m app.cli stats`, then `reprocess`. |
| Everything is `no_reliable_match` in `SAME_PERSON` | Threshold too strict for your photos; use `LOOKALIKE`, or calibrate (README, "Calibration"). |
