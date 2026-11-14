# Migrating from the original dlib-based Doppelganger

The first version of this project stored one 128-d vector per photo, produced by dlib's HOG
detector and `dlib_face_recognition_resnet_model_v1` (the README called it FaceNet; it never was),
in `storage/embeddings.db`, and compared vectors by loading every row and computing euclidean
distances with a 0.6 threshold.

## Why the old vectors are not reused

* **Different geometry.** dlib vectors are compared with euclidean distance (threshold ~0.6);
  SFace vectors are L2-normalised and compared with cosine similarity. A threshold, a score or a
  ranking from one model means nothing for the other.
* **Different vector spaces.** Even with the same dimension (both are 128-d) the coordinates of two
  models have no relationship. Mixing them in one index returns garbage without raising an error.
* **No provenance.** The old rows record neither the detector settings nor which face of a
  multi-face photo was embedded (the old code silently took `face_locations[0]`).

So the new service **never mixes models in one index**. Every stored face carries the `model_id`
of the pipeline that produced it, there is one index per `model_id`, and the active pipeline only
searches its own vectors. The old vectors are simply not used; the photos are re-embedded.

## What happens to your existing data

Nothing is deleted or overwritten:

* `storage/embeddings.db` (legacy database) is only ever opened **read-only** by the migration
  command and is otherwise ignored. The new service uses a different file: `data/doppelganger.db`.
* `storage/images/` is only read. Originals are copied (sha256-deduplicated) into
  `data/originals/` under random UUID names.
* `init_db.py` no longer drops tables. It now just applies the new schema migrations to the new
  database.

## Step by step

```bash
# 0. Provision the models once (explicit, checksum-verified) and configure the token
python -m app.cli models fetch
cp .env.example .env && $EDITOR .env          # set DOPPEL_ADMIN_TOKEN

# 1. Re-embed every image referenced by the legacy database (read-only access to the old DB).
#    The paths stored in the legacy DB are relative to --legacy-root.
python -m app.cli db upgrade
python -m app.cli migrate-legacy --legacy-db storage/embeddings.db --legacy-root .

# 2. Verify
python -m app.cli stats
python -m app.cli check
```

`migrate-legacy` prints one line per legacy row with `status` (`processed`, `duplicate`, `no_face`
or `failed`) and a `code` for failures such as `source_missing`. It exits with status 2 if any row
failed, and is idempotent (re-running skips images that are already stored).

Photos that contain several faces now index **every** eligible face (old behaviour: only the first
one). A photo with no usable face is stored but produces no vector, and the reason is recorded
(`no_face`, or a quality reason such as `face_too_small`).

## Re-processing later (for example after changing models)

```bash
python -m app.cli reprocess --all --wait   # recompute faces from stored originals for the active model
python -m app.cli purge-model <old-model-id> --yes   # only when you are sure you no longer need them
python -m app.cli stats                    # shows faces per model_id
```

Original images are stored, so switching embedders is a re-computation, not a re-upload.

## API differences for existing clients

The legacy routes still exist, but the semantics are now honest. See the README table; in short:
`/search-face` now behaves as `SAME_PERSON`, `threshold` is a **cosine distance** (`1 - cosine
similarity`) instead of a dlib euclidean distance, errors no longer include internal exception text
or filesystem paths, `/upload-images` no longer returns paths, all routes except `/`, `/health`, `/ready`
and the API docs require a bearer token, and validation errors are reported as `400` on legacy routes (as before)
and `422` on `/v1` routes.

## Running the old pipeline for comparison

```bash
python -m venv /opt/dlibenv && /opt/dlibenv/bin/pip install -r requirements-dlib.txt
/opt/dlibenv/bin/python scripts/evaluate.py extract --pipeline dlib --dataset <dir> --out eval_out/dlib.npz
```

The dlib adapter is evaluation-only: the index refuses non-cosine models, and the production image
does not contain dlib.
