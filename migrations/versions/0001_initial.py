"""Initial schema: images, faces (with model id), processing state, index events, jobs.

Revision ID: 0001
Revises:
"""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "images",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("storage_name", sa.String(64), nullable=False),
        sa.Column("format", sa.String(16), nullable=False),
        sa.Column("width", sa.Integer, nullable=False),
        sa.Column("height", sa.Integer, nullable=False),
        sa.Column("size_bytes", sa.Integer, nullable=False),
        sa.Column("source_name", sa.String(255)),
        sa.Column("legacy_ref", sa.String(255)),
        sa.Column("created_at", sa.DateTime, nullable=False),
    )
    op.create_index("ix_images_sha256", "images", ["sha256"], unique=True)

    op.create_table(
        "image_processing",
        sa.Column(
            "image_id",
            sa.String(36),
            sa.ForeignKey("images.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("model_id", sa.String(128), primary_key=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("face_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("rejected", sa.JSON),
        sa.Column("processed_at", sa.DateTime, nullable=False),
    )

    op.create_table(
        "faces",
        sa.Column("seq", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column(
            "image_id",
            sa.String(36),
            sa.ForeignKey("images.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("model_id", sa.String(128), nullable=False),
        sa.Column("face_index", sa.Integer, nullable=False),
        sa.Column("bbox_x", sa.Float, nullable=False),
        sa.Column("bbox_y", sa.Float, nullable=False),
        sa.Column("bbox_w", sa.Float, nullable=False),
        sa.Column("bbox_h", sa.Float, nullable=False),
        sa.Column("det_score", sa.Float, nullable=False),
        sa.Column("quality", sa.JSON),
        sa.Column("embedding", sa.LargeBinary, nullable=False),
        sa.Column("crop_name", sa.String(64)),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sqlite_autoincrement=True,
    )
    op.create_index("ix_faces_id", "faces", ["id"], unique=True)
    op.create_index("ix_faces_image_id", "faces", ["image_id"])
    op.create_index("ix_faces_model_seq", "faces", ["model_id", "seq"])

    op.create_table(
        "index_events",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("model_id", sa.String(128), nullable=False),
        sa.Column("face_seq", sa.Integer, nullable=False),
        sa.Column("op", sa.String(16), nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sqlite_autoincrement=True,
    )
    op.create_index("ix_index_events_model_id", "index_events", ["model_id", "id"])

    op.create_table(
        "jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("model_id", sa.String(128), nullable=False),
        sa.Column("total", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("started_at", sa.DateTime),
        sa.Column("finished_at", sa.DateTime),
        sa.Column("cancel_requested", sa.Boolean, nullable=False, server_default=sa.false()),
    )
    op.create_index("ix_jobs_status", "jobs", ["status"])

    op.create_table(
        "job_items",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "job_id",
            sa.String(36),
            sa.ForeignKey("jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("source_path", sa.String(1024)),
        sa.Column("delete_source", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("display_name", sa.String(255)),
        sa.Column("image_id", sa.String(36)),
        sa.Column("attempts", sa.Integer, nullable=False, server_default="0"),
        sa.Column("lease_expires_at", sa.DateTime),
        sa.Column("face_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("error_code", sa.String(64)),
        sa.Column("error_message", sa.String(255)),
        sa.Column("updated_at", sa.DateTime, nullable=False),
    )
    op.create_index("ix_job_items_job_id", "job_items", ["job_id"])
    op.create_index("ix_job_items_claim", "job_items", ["status", "lease_expires_at"])


def downgrade() -> None:
    for table in ("job_items", "jobs", "index_events", "faces", "image_processing", "images"):
        op.drop_table(table)
