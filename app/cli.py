"""Operator CLI: python -m app.cli <command>. All commands work offline against local data."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import signal
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import migrate
from .config import Settings, get_settings
from .container import Container, build_container
from .errors import ApiError, ModelUnavailable
from .imaging import decode_image
from .jobs import ItemSpec, heartbeat_age, import_directory_items
from .ml.manifest import MANIFEST_PATH, fetch_model, load_manifest, verify_model
from .search import SearchMode, SearchParams
from .storage import FileStore

logger = logging.getLogger("doppelganger.cli")


def _out(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _settings() -> Settings:
    return get_settings().model_copy(update={"auth_disabled": True})


def _container(settings: Settings | None = None) -> Container:
    return build_container(settings or _settings())


# --- commands --------------------------------------------------------------------------------


def cmd_db_upgrade(args: argparse.Namespace) -> int:
    s = _settings()
    s.ensure_dirs()
    migrate.upgrade(s.database_url or "")
    _out(
        {
            "revision": migrate.current_revision(s.database_url or ""),
            "head": migrate.head_revision(),
        }
    )
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    s = _settings()
    manifest = load_manifest(MANIFEST_PATH)
    rows = []
    rc = 0
    for key, m in manifest.items():
        status = "ok"
        try:
            if args.action == "fetch":
                try:
                    verify_model(s.models_dir, m)
                except ModelUnavailable:
                    fetch_model(s.models_dir, m)
            verify_model(s.models_dir, m)
        except ModelUnavailable as exc:
            status, rc = exc.code, 1
        rows.append(
            {
                "model": key,
                "file": m.filename,
                "sha256": m.sha256,
                "license": m.license,
                "status": status,
            }
        )
    _out(rows)
    return rc


def _wait_for_job(c: Container, job_id: str, inline: bool) -> dict[str, Any]:
    if inline:
        assert c.worker is not None
        c.worker.start()
    last = -1.0
    try:
        while True:
            job = c.jobs.get_job(job_id)
            assert job is not None
            if job["progress"] != last:
                last = job["progress"]
                print(f"progress {job['progress']:.0%} {job['counts']}", file=sys.stderr)
            if job["status"] in ("completed", "completed_with_errors", "failed", "cancelled"):
                return job
            time.sleep(0.5)
    finally:
        if inline and c.worker is not None:
            c.worker.stop()


def _enqueue_and_maybe_run(
    c: Container, kind: str, items: list[ItemSpec], args: argparse.Namespace
) -> int:
    assert c.pool is not None
    job_id = c.jobs.create_job(kind, c.pool.info.model_id, items)
    if not (args.inline or args.wait):
        _out(
            {
                "job_id": job_id,
                "total": len(items),
                "hint": "a worker will process it; see `job <id>`",
            }
        )
        return 0
    job = _wait_for_job(c, job_id, inline=args.inline)
    if c.index is not None:
        c.index.sync(force=True)
        c.index.persist()
    _out(job)
    return 0 if job["status"] in ("completed", "cancelled") else 2


def cmd_import(args: argparse.Namespace) -> int:
    root = Path(args.directory)
    if not root.is_dir():
        print("error: not a directory", file=sys.stderr)
        return 1
    c = _container()
    try:
        items = import_directory_items(root.resolve(), recursive=args.recursive)
        if not items:
            print("error: no image files found", file=sys.stderr)
            return 1
        return _enqueue_and_maybe_run(c, "import", items, args)
    finally:
        c.close()


def cmd_reprocess(args: argparse.Namespace) -> int:
    c = _container()
    try:
        assert c.pool is not None
        model_id = c.pool.info.model_id
        ids = c.repo.all_image_ids()
        done: set[str] = (
            set()
            if args.all
            else {i for i in ids if c.repo.processed_face_count(i, model_id) is not None}
        )
        todo = [ItemSpec(image_id=i) for i in ids if i not in done]
        if not todo:
            _out({"message": "nothing to reprocess for the active model", "images": len(ids)})
            return 0
        return _enqueue_and_maybe_run(c, "reprocess", todo, args)
    finally:
        c.close()


def cmd_worker(args: argparse.Namespace) -> int:
    s = get_settings().model_copy(update={"auth_disabled": True})
    if args.concurrency:
        s = s.model_copy(update={"worker_concurrency": args.concurrency})
    c = build_container(s)
    assert c.worker is not None
    c.store.sweep_staging()
    stop = {"flag": False}

    def _term(_sig: int, _frm: Any) -> None:
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    c.worker.start()
    logger.info(
        "Worker started (concurrency=%d, model=%s)",
        c.worker.concurrency,
        c.pool.info.model_id if c.pool else "-",
    )
    try:
        while not stop["flag"]:
            time.sleep(0.5)
    finally:
        logger.info("Worker stopping; finishing in-flight items")
        c.close()
    return 0


def cmd_worker_health(args: argparse.Namespace) -> int:
    age = heartbeat_age(_settings())
    return 0 if age is not None and age < 30 else 1


def cmd_job(args: argparse.Namespace) -> int:
    c = _container_noload()
    try:
        job = c.jobs.get_job(args.job_id)
        if job is None:
            print("error: job not found", file=sys.stderr)
            return 1
        _out(job)
        return 0
    finally:
        c.close()


def _container_noload() -> Container:
    s = _settings().model_copy(update={"require_models_at_startup": False})
    return build_container(s)


def cmd_index(args: argparse.Namespace) -> int:
    c = _container()
    try:
        assert c.index is not None
        if args.action == "rebuild":
            _out({"rebuilt": True, "vectors": c.index.rebuild()})
            return 0
        if args.action == "sync":
            _out({"applied": c.index.sync(force=True), **c.index.health()})
            return 0
        report = c.index.check()
        _out(report.to_dict())
        return 0 if report.ok else 1
    finally:
        c.close()


def cmd_check(args: argparse.Namespace) -> int:
    c = _container()
    try:
        assert c.index is not None and c.pool is not None
        s, rc = c.settings, 0
        report = c.index.check()
        originals_db, crops_db = c.repo.storage_names()
        originals_fs, crops_fs = c.store.list_originals(), c.store.list_crops()
        result: dict[str, Any] = {
            "schema_at_head": migrate.current_revision(s.database_url or "")
            == migrate.head_revision(),
            "index": report.to_dict(),
            "originals_missing_on_disk": sorted(originals_db - originals_fs),
            "originals_orphaned": sorted(originals_fs - originals_db),
            "crops_missing_on_disk": sorted(crops_db - crops_fs),
            "crops_orphaned": sorted(crops_fs - crops_db),
            "faces_by_model": c.repo.faces_by_model(),
            "active_model_id": c.pool.info.model_id,
        }
        bad_dim = _bad_dimension_faces(c)
        result["faces_with_wrong_embedding_size"] = bad_dim
        if args.fix:
            fixed = {
                "orphan_originals_removed": 0,
                "orphan_crops_removed": 0,
                "index_rebuilt": False,
            }
            for name in result["originals_orphaned"]:
                c.store.delete_original(name)
                fixed["orphan_originals_removed"] += 1
            for name in result["crops_orphaned"]:
                c.store.delete_crop(name)
                fixed["orphan_crops_removed"] += 1
            if not report.ok:
                c.index.rebuild()
                fixed["index_rebuilt"] = True
            result["fixed"] = fixed
        else:
            problems = (
                not report.ok
                or result["originals_missing_on_disk"]
                or result["originals_orphaned"]
                or result["crops_missing_on_disk"]
                or result["crops_orphaned"]
                or bad_dim
                or not result["schema_at_head"]
            )
            rc = 1 if problems else 0
        result["ok"] = rc == 0
        _out(result)
        return rc
    finally:
        c.close()


def _bad_dimension_faces(c: Container) -> int:
    assert c.pool is not None
    want = c.pool.info.dim * 4
    n = 0
    from sqlalchemy import func, select

    from .db import Face

    with c.db.read() as s:
        n = int(
            s.scalar(
                select(func.count())
                .select_from(Face)
                .where(Face.model_id == c.pool.info.model_id, func.length(Face.embedding) != want)
            )
            or 0
        )
    return n


def cmd_purge_model(args: argparse.Namespace) -> int:
    c = _container()
    try:
        assert c.pool is not None
        if args.model_id == c.pool.info.model_id:
            print("error: refusing to purge the active model", file=sys.stderr)
            return 1
        if not args.yes:
            print("error: pass --yes to confirm deleting that model's embeddings", file=sys.stderr)
            return 1
        crops = c.repo.purge_model(args.model_id)
        for name in crops:
            c.store.delete_crop(name)
        import re

        idx_dir = c.settings.index_dir / re.sub(r"[^A-Za-z0-9._-]", "_", args.model_id)
        if idx_dir.is_dir():
            shutil.rmtree(idx_dir)
        _out({"purged_model": args.model_id, "crops_removed": len(crops)})
        return 0
    finally:
        c.close()


def cmd_delete_image(args: argparse.Namespace) -> int:
    c = _container()
    try:
        res = c.repo.delete_image(args.image_id)
        if res is None:
            print("error: image not found", file=sys.stderr)
            return 1
        rec, crops, n = res
        c.store.delete_original(rec.storage_name)
        for name in crops:
            c.store.delete_crop(name)
        assert c.index is not None
        c.index.sync(force=True)
        c.index.persist()
        _out({"image_id": args.image_id, "faces_removed": n})
        return 0
    finally:
        c.close()


def cmd_migrate_legacy(args: argparse.Namespace) -> int:
    """Re-embed images referenced by the original SQLite DB with the new pipeline.

    The legacy database is opened read-only and never modified. Its dlib vectors are NOT reused.
    """
    legacy_db = Path(args.legacy_db)
    root = Path(args.legacy_root).resolve()
    if not legacy_db.is_file():
        print("error: legacy database not found", file=sys.stderr)
        return 1
    con = sqlite3.connect(f"file:{legacy_db.resolve()}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT id, image_path FROM face_embeddings ORDER BY id").fetchall()
    finally:
        con.close()
    c = _container()
    report: list[dict[str, Any]] = []
    try:
        assert c.ingest is not None
        ingest = c.ingest

        def one(row: tuple[int, str]) -> dict[str, Any]:
            legacy_id, rel = row
            path = (root / rel).resolve()
            entry: dict[str, Any] = {"legacy_id": legacy_id, "legacy_path": rel}
            if not path.is_relative_to(root) or not path.is_file():
                return {**entry, "status": "failed", "code": "source_missing"}
            try:
                out = ingest.ingest_bytes(
                    path.read_bytes(), display_name=path.name, legacy_ref=f"legacy:{legacy_id}"
                )
                return {
                    **entry,
                    "status": out.status,
                    "image_id": out.image_id,
                    "faces": out.face_count,
                }
            except ApiError as exc:
                return {**entry, "status": "failed", "code": exc.code}

        with ThreadPoolExecutor(max_workers=c.settings.worker_concurrency) as ex:
            report = list(ex.map(one, rows))
        assert c.index is not None
        c.index.sync(force=True)
        c.index.persist()
    finally:
        c.close()
    failed = sum(1 for r in report if r["status"] == "failed")
    _out(
        {
            "legacy_rows": len(rows),
            "failed": failed,
            "results": report,
            "note": "legacy vectors were not reused",
        }
    )
    return 0 if not failed else 2


def cmd_backup(args: argparse.Namespace) -> int:
    s = _settings()
    dest = Path(args.destination)
    dest.mkdir(parents=True, exist_ok=True)
    dest.chmod(0o700)
    src_url = s.database_url or ""
    if not src_url.startswith("sqlite:///"):
        print("error: backup supports the default SQLite database only", file=sys.stderr)
        return 1
    src_path = src_url.removeprefix("sqlite:///")
    src = sqlite3.connect(src_path)
    dst = sqlite3.connect(dest / "doppelganger.db")
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    counts = {"originals": 0, "crops": 0}
    for sub, key in (("originals", "originals"), ("crops", "crops")):
        (dest / sub).mkdir(exist_ok=True)
        for p in (s.data_dir / sub).glob("*"):
            target = dest / sub / p.name
            if p.is_file() and not (target.exists() and target.stat().st_size == p.stat().st_size):
                shutil.copy2(p, target)
            counts[key] += 1
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        **counts,
        "schema": migrate.current_revision(src_url),
    }
    (dest / "backup.json").write_text(json.dumps(manifest, indent=2))
    _out(manifest)
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    c = _container()
    try:
        _, _, _, search = c.require_ready()
        decoded = decode_image(Path(args.selfie).read_bytes(), c.settings)
        params = SearchParams(
            mode=SearchMode(args.mode),
            max_results=args.max_results,
            face_index=args.face_index,
            min_similarity=args.min_similarity,
        )
        out = search.search_image(decoded.bgr, params)
        _out(
            {
                "mode": out.mode.value,
                "status": out.status,
                "message": out.message,
                "notice": out.notice,
                "threshold": out.threshold,
                "gallery_faces": out.gallery_faces,
                "matches": [
                    {
                        "rank": m.rank,
                        "image_id": m.hit.image_id,
                        "face_id": m.hit.face_id,
                        "name": m.hit.source_name,
                        "cosine_similarity": round(m.cosine_similarity, 4),
                        "cosine_distance": round(m.cosine_distance, 4),
                        "bbox": [round(v, 1) for v in m.hit.bbox],
                    }
                    for m in out.matches
                ],
            }
        )
        return 0
    except ApiError as exc:
        _out({"error": {"code": exc.code, "message": exc.message, "details": exc.details}})
        return 1
    finally:
        c.close()


def cmd_stats(args: argparse.Namespace) -> int:
    c = _container_noload()
    try:
        _out(
            {
                "images": c.repo.count_images(),
                "faces_by_model": c.repo.faces_by_model(),
                "queue_depth": c.jobs.queue_depth(),
                "failed_job_items": c.jobs.failed_items_total(),
            }
        )
        return 0
    finally:
        c.close()


def cmd_sweep(args: argparse.Namespace) -> int:
    s = _settings()
    removed = FileStore(s).sweep_staging(args.older_than_hours)
    _out({"staging_files_removed": removed})
    return 0


# --- parser ----------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="doppelganger", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    db = sub.add_parser("db", help="database migrations")
    db.add_argument("action", choices=["upgrade"])
    db.set_defaults(fn=cmd_db_upgrade)

    m = sub.add_parser("models", help="verify or explicitly fetch pinned model weights")
    m.add_argument("action", choices=["verify", "fetch"])
    m.set_defaults(fn=cmd_models)

    for name, fn, helptext in (
        ("import", cmd_import, "queue every image under a directory"),
        ("reprocess", cmd_reprocess, "recompute faces from stored originals for the active model"),
    ):
        sp = sub.add_parser(name, help=helptext)
        if name == "import":
            sp.add_argument("directory")
            sp.add_argument("--recursive", "-r", action="store_true")
        else:
            sp.add_argument("--all", action="store_true", help="redo images already processed")
        sp.add_argument(
            "--inline", action="store_true", help="process in this process, no separate worker"
        )
        sp.add_argument("--wait", action="store_true", help="wait for a separate worker to finish")
        sp.set_defaults(fn=fn)

    w = sub.add_parser("worker", help="run the background ingestion worker")
    w.add_argument("--concurrency", type=int)
    w.set_defaults(fn=cmd_worker)
    sub.add_parser("worker-health", help="exit 0 if the worker heartbeat is fresh").set_defaults(
        fn=cmd_worker_health
    )

    j = sub.add_parser("job", help="show job status")
    j.add_argument("job_id")
    j.set_defaults(fn=cmd_job)

    ix = sub.add_parser("index", help="vector index maintenance")
    ix.add_argument("action", choices=["rebuild", "check", "sync"])
    ix.set_defaults(fn=cmd_index)

    ck = sub.add_parser("check", help="consistency check of DB, index and files")
    ck.add_argument(
        "--fix", action="store_true", help="remove orphan files and rebuild a drifted index"
    )
    ck.set_defaults(fn=cmd_check)

    pm = sub.add_parser("purge-model", help="delete embeddings of a retired model")
    pm.add_argument("model_id")
    pm.add_argument("--yes", action="store_true")
    pm.set_defaults(fn=cmd_purge_model)

    di = sub.add_parser(
        "delete-image", help="delete an image with its faces, crops and index entries"
    )
    di.add_argument("image_id")
    di.set_defaults(fn=cmd_delete_image)

    ml = sub.add_parser(
        "migrate-legacy", help="re-embed images from the original dlib-era database"
    )
    ml.add_argument("--legacy-db", default="storage/embeddings.db")
    ml.add_argument(
        "--legacy-root", default=".", help="directory the legacy image_path values are relative to"
    )
    ml.set_defaults(fn=cmd_migrate_legacy)

    bk = sub.add_parser("backup", help="consistent backup of DB, originals and crops")
    bk.add_argument("destination")
    bk.set_defaults(fn=cmd_backup)

    se = sub.add_parser("search", help="search a selfie from the command line")
    se.add_argument("selfie")
    se.add_argument("--mode", choices=[m.value for m in SearchMode], default="LOOKALIKE")
    se.add_argument("--max-results", type=int, default=10)
    se.add_argument("--face-index", type=int)
    se.add_argument("--min-similarity", type=float)
    se.set_defaults(fn=cmd_search)

    sub.add_parser("stats", help="gallery counters").set_defaults(fn=cmd_stats)

    sw = sub.add_parser("sweep-staging", help="remove stale staged uploads")
    sw.add_argument("--older-than-hours", type=int)
    sw.set_defaults(fn=cmd_sweep)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    for noisy in ("alembic",):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = build_parser().parse_args(argv)
    try:
        rc: int = args.fn(args)
        return rc
    except ModelUnavailable as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
