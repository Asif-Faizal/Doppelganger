from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine

MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"


def _config(url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return cfg


def upgrade(url: str, revision: str = "head") -> None:
    command.upgrade(_config(url), revision)


def downgrade(url: str, revision: str) -> None:
    command.downgrade(_config(url), revision)


def head_revision() -> str:
    head = ScriptDirectory(str(MIGRATIONS_DIR)).get_current_head()
    assert head is not None
    return str(head)


def current_revision(url: str) -> str | None:
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            rev = MigrationContext.configure(conn).get_current_revision()
            return None if rev is None else str(rev)
    finally:
        engine.dispose()
