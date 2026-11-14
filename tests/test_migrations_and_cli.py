from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, inspect, text

from app import cli, migrate
from app.config import get_settings
from app.db import Base

from .conftest import BLUE, RED, face_image, single


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data = tmp_path / "data"
    monkeypatch.setenv("DOPPEL_DATA_DIR", str(data))
    monkeypatch.setenv("DOPPEL_PIPELINE", "fake")
    monkeypatch.setenv("DOPPEL_MIN_FACE_PX", "24")
    monkeypatch.setenv("DOPPEL_INDEX_SYNC_INTERVAL_S", "0")
    monkeypatch.setenv("DOPPEL_MODELS_DIR", str(tmp_path / "models"))
    get_settings.cache_clear()
    yield data  # type: ignore[misc]
    get_settings.cache_clear()


def run(*argv: str) -> int:
    return cli.main(list(argv))


# --- migrations ------------------------------------------------------------------------------


def test_migration_creates_schema_matching_models(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'm.db'}"
    migrate.upgrade(url)
    engine = create_engine(url)
    assert set(inspect(engine).get_table_names()) >= {
        "images",
        "faces",
        "image_processing",
        "index_events",
        "jobs",
        "job_items",
        "alembic_version",
    }
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diff == [], diff
    assert migrate.current_revision(url) == migrate.head_revision()
    with engine.connect() as conn:
        assert conn.execute(text("PRAGMA journal_mode")).scalar() in ("delete", "wal")


def test_migration_is_idempotent_and_preserves_rows(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'm.db'}"
    migrate.upgrade(url)
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO images (id, sha256, storage_name, format, width, height, size_bytes, created_at) "
                "VALUES ('i1', 'abc', 'i1.jpg', 'JPEG', 1, 1, 1, '2026-01-01')"
            )
        )
    migrate.upgrade(url)
    migrate.upgrade(url)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM images")).scalar() == 1


def test_downgrade_then_upgrade_roundtrip(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'm.db'}"
    migrate.upgrade(url)
    migrate.downgrade(url, "base")
    assert "faces" not in inspect(create_engine(url)).get_table_names()
    migrate.upgrade(url)
    assert "faces" in inspect(create_engine(url)).get_table_names()


def test_init_db_script_is_non_destructive(env: Path) -> None:
    assert run("db", "upgrade") == 0
    url = get_settings().database_url or ""
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO images (id, sha256, storage_name, format, width, height, size_bytes, created_at) "
                "VALUES ('keep', 'k', 'keep.jpg', 'JPEG', 1, 1, 1, '2026-01-01')"
            )
        )
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "init_db", Path(__file__).resolve().parents[1] / "init_db.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert cli.main(["db", "upgrade"]) == 0
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM images")).scalar() == 1


# --- CLI: import / check / backup / purge -----------------------------------------------------


def make_gallery(tmp_path: Path) -> Path:
    g = tmp_path / "gal"
    g.mkdir()
    (g / "a.jpg").write_bytes(single(RED))
    (g / "b.jpg").write_bytes(face_image([(BLUE, 10, 10, 90), (RED, 150, 150, 90)]))
    (g / "bad.jpg").write_bytes(b"junk")
    return g


def test_cli_import_inline_search_check_and_reimport(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    g = make_gallery(tmp_path)
    assert run("import", str(g), "--inline") == 2  # one corrupt file -> completed_with_errors
    out = capsys.readouterr().out
    job = json.loads(out[out.index("{") :])
    assert (
        job["counts"]["succeeded"] == 2
        and job["counts"]["failed"] == 1
        and job["faces_indexed"] == 3
    )

    assert run("import", str(g), "--inline") == 2
    out = capsys.readouterr().out
    job2 = json.loads(out[out.index("{") :])
    assert job2["counts"]["duplicate"] == 2  # already processed -> skipped, no new faces
    assert job2["faces_indexed"] == 0

    selfie = tmp_path / "selfie.jpg"
    selfie.write_bytes(single(RED, side=90))
    assert run("search", str(selfie), "--mode", "SAME_PERSON", "--min-similarity", "0.9") == 0
    res = json.loads(capsys.readouterr().out)
    assert res["matches"] and res["matches"][0]["cosine_similarity"] > 0.9

    assert run("check") == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert run("worker-health") == 0  # inline run wrote a fresh heartbeat
    (env / "worker.heartbeat").write_text("0")  # stale
    assert run("worker-health") == 1
    (env / "worker.heartbeat").unlink()
    assert run("worker-health") == 1


def test_cli_check_finds_and_fixes_orphans_and_index_drift(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    g = make_gallery(tmp_path)
    run("import", str(g), "--inline")
    capsys.readouterr()
    orphan = env / "originals" / "11111111-1111-4111-8111-111111111111.jpg"
    orphan.write_bytes(b"x")
    snapshot = next((env / "index").glob("*/index.faiss"))
    snapshot.write_bytes(b"corrupt")
    assert run("check") == 1
    report = json.loads(capsys.readouterr().out)
    assert report["originals_orphaned"] == ["11111111-1111-4111-8111-111111111111.jpg"]
    assert run("check", "--fix") == 0
    capsys.readouterr()
    assert not orphan.exists()
    assert run("check") == 0


def test_cli_reprocess_skips_done_and_all_recomputes(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run("import", str(make_gallery(tmp_path)), "--inline")
    capsys.readouterr()
    assert run("reprocess", "--inline") == 0
    assert "nothing to reprocess" in capsys.readouterr().out
    assert run("reprocess", "--inline", "--all") == 0
    out = capsys.readouterr().out
    job = json.loads(out[out.index("{") :])
    assert job["kind"] == "reprocess" and job["counts"]["succeeded"] == 2
    assert run("check") == 0


def test_cli_backup_is_consistent_and_restorable(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run("import", str(make_gallery(tmp_path)), "--inline")
    capsys.readouterr()
    dest = tmp_path / "bk"
    assert run("backup", str(dest)) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["originals"] == 2 and manifest["crops"] == 3
    con = sqlite3.connect(dest / "doppelganger.db")
    assert con.execute("SELECT count(*) FROM faces").fetchone()[0] == 3
    assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    con.close()
    assert len(list((dest / "originals").iterdir())) == 2


def test_cli_purge_model_refuses_active_and_removes_retired(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run("import", str(make_gallery(tmp_path)), "--inline")
    capsys.readouterr()
    con = sqlite3.connect(env / "doppelganger.db")
    con.execute(
        "INSERT INTO faces (id, image_id, model_id, face_index, bbox_x, bbox_y, bbox_w, bbox_h, det_score, embedding, created_at) "
        "SELECT id || 'r', image_id, 'retired-model', face_index, bbox_x, bbox_y, bbox_w, bbox_h, det_score, embedding, created_at FROM faces"
    )
    con.commit()
    con.close()
    active = "fake-colorblobs.v1.d128"
    assert run("purge-model", active, "--yes") == 1
    assert run("purge-model", "retired-model") == 1  # needs --yes
    assert run("purge-model", "retired-model", "--yes") == 0
    capsys.readouterr()
    assert run("stats") == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["faces_by_model"] == {active: 3}


def test_cli_models_verify_reports_missing_files(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run("models", "verify") == 1
    rows = json.loads(capsys.readouterr().out)
    assert {r["status"] for r in rows} == {"model_file_missing"}


def test_model_integrity_check_detects_tampering(tmp_path: Path) -> None:
    from app.errors import ModelUnavailable
    from app.ml.manifest import ModelFile, file_sha256, verify_model

    f = tmp_path / "m.onnx"
    f.write_bytes(b"hello")
    good = ModelFile(
        "m",
        "embedder",
        "M",
        "1",
        "m.onnx",
        hashlib.sha256(b"hello").hexdigest(),
        5,
        "https://x",
        "MIT",
    )
    assert verify_model(tmp_path, good) == f and file_sha256(f) == good.sha256
    f.write_bytes(b"hellO")
    with pytest.raises(ModelUnavailable) as exc:
        verify_model(tmp_path, good)
    assert exc.value.code == "model_checksum_mismatch"


def test_delete_image_cli(env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run("import", str(make_gallery(tmp_path)), "--inline")
    capsys.readouterr()
    import re

    first = sorted((env / "originals").iterdir())[0].name
    image_id = re.sub(r"\..*", "", first)
    assert run("delete-image", image_id) == 0
    assert run("delete-image", image_id) == 1
    assert run("check") == 0


# --- legacy migration ------------------------------------------------------------------------


def make_legacy(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "legacy"
    (root / "storage" / "images").mkdir(parents=True)
    img1 = root / "storage" / "images" / "aaaa.jpg"
    img1.write_bytes(single(RED))
    img2 = root / "storage" / "images" / "bbbb.jpg"
    img2.write_bytes(single(BLUE))
    db = root / "storage" / "embeddings.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE face_embeddings (id INTEGER NOT NULL, image_path VARCHAR, embedding TEXT, created_at DATETIME, PRIMARY KEY (id))"
    )
    vec = json.dumps([0.1] * 128)
    con.execute(
        "INSERT INTO face_embeddings VALUES (1, 'storage/images/aaaa.jpg', ?, '2025-07-08')", (vec,)
    )
    con.execute(
        "INSERT INTO face_embeddings VALUES (2, 'storage/images/bbbb.jpg', ?, '2025-07-08')", (vec,)
    )
    con.execute(
        "INSERT INTO face_embeddings VALUES (3, 'storage/images/missing.jpg', ?, '2025-07-08')",
        (vec,),
    )
    con.execute(
        "INSERT INTO face_embeddings VALUES (4, '../../etc/passwd', ?, '2025-07-08')", (vec,)
    )
    con.commit()
    con.close()
    return root, db


def test_legacy_migration_reembeds_and_never_modifies_legacy_data(
    env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, db = make_legacy(tmp_path)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
    rc = run("migrate-legacy", "--legacy-db", str(db), "--legacy-root", str(root))
    assert rc == 2  # two rows could not be migrated (missing file, path escape)
    report = json.loads(capsys.readouterr().out)
    by_id = {r["legacy_id"]: r for r in report["results"]}
    assert by_id[1]["status"] == "processed" and by_id[2]["status"] == "processed"
    assert by_id[3]["code"] == "source_missing" and by_id[4]["code"] == "source_missing"
    assert report["note"] == "legacy vectors were not reused"
    after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
    assert before == after  # legacy DB and images untouched

    new_db = sqlite3.connect(env / "doppelganger.db")
    rows = new_db.execute(
        "SELECT i.legacy_ref, f.model_id, length(f.embedding) FROM faces f JOIN images i ON i.id = f.image_id ORDER BY i.legacy_ref"
    ).fetchall()
    assert [r[0] for r in rows] == ["legacy:1", "legacy:2"]
    assert all(r[1] == "fake-colorblobs.v1.d128" and r[2] == 128 * 4 for r in rows)
    new_db.close()
    assert run("migrate-legacy", "--legacy-db", str(db), "--legacy-root", str(root)) == 2
    capsys.readouterr()
    again = sqlite3.connect(env / "doppelganger.db")
    assert again.execute("SELECT count(*) FROM faces").fetchone()[0] == 2  # idempotent
    again.close()
