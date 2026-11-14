from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PipelineName = Literal["sface", "fake"]


class Settings(BaseSettings):
    """All runtime configuration. Every value can be set via a DOPPEL_* env var."""

    model_config = SettingsConfigDict(
        env_prefix="DOPPEL_", env_file=".env", extra="ignore", protected_namespaces=()
    )

    data_dir: Path = Path("data")
    database_url: str | None = None
    models_dir: Path = Path("models")
    import_root: Path | None = None
    auto_migrate: bool = True

    pipeline: PipelineName = "sface"
    require_models_at_startup: bool = True
    inference_instances: int = Field(default=2, ge=1, le=16)
    inference_threads: int = Field(default=1, ge=1, le=32)
    detect_max_side: int = Field(default=1280, ge=160, le=8192)

    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)
    max_request_bytes: int = Field(default=256 * 1024 * 1024, ge=1024)
    max_pixels: int = Field(default=40_000_000, ge=10_000)
    max_dimension: int = Field(default=10_000, ge=64)
    min_dimension: int = Field(default=32, ge=8)
    allowed_formats: str = "JPEG,PNG,WEBP,BMP,TIFF"
    max_import_files: int = Field(default=200, ge=1)

    min_face_px: int = Field(default=40, ge=8)
    min_detection_score: float = Field(default=0.80, ge=0.0, le=1.0)
    min_sharpness: float = Field(default=15.0, ge=0.0)
    max_edge_overflow: float = Field(default=0.25, ge=0.0, le=1.0)
    max_faces_per_image: int = Field(default=64, ge=1)
    ambiguity_ratio: float = Field(default=0.5, gt=0.0, le=1.0)
    crop_size: int = Field(default=160, ge=32, le=512)

    same_person_min_similarity: float = Field(default=0.363, ge=-1.0, le=1.0)
    default_max_results: int = Field(default=10, ge=1)
    max_results_cap: int = Field(default=100, ge=1)
    index_threads: int = Field(default=2, ge=1, le=64)
    embedded_worker: bool = False
    index_sync_interval_s: float = Field(default=1.0, ge=0.0)
    index_persist_every: int = Field(default=500, ge=1)
    calibration_file: Path | None = None

    admin_token: str | None = None
    search_token: str | None = None
    auth_disabled: bool = False
    rate_limit_per_minute: int = Field(default=120, ge=0)
    search_rate_limit_per_minute: int = Field(default=30, ge=0)
    cors_origins: str = ""
    url_signing_key: str | None = None
    url_ttl_s: int = Field(default=3600, ge=30)
    public_base_url: str = ""

    worker_concurrency: int = Field(default=2, ge=1, le=16)
    worker_poll_s: float = Field(default=1.0, gt=0.0)
    job_lease_s: int = Field(default=300, ge=10)
    job_max_attempts: int = Field(default=3, ge=1)
    staging_retention_hours: int = Field(default=24, ge=1)

    log_level: str = "INFO"

    @field_validator("admin_token", "search_token")
    @classmethod
    def _token_length(cls, v: str | None) -> str | None:
        if v is not None and v != "" and len(v) < 16:
            raise ValueError("API tokens must be at least 16 characters")
        return v or None

    @model_validator(mode="after")
    def _resolve(self) -> Settings:
        if self.database_url is None:
            self.database_url = f"sqlite:///{(self.data_dir / 'doppelganger.db').resolve()}"
        return self

    @property
    def formats(self) -> frozenset[str]:
        return frozenset(f.strip().upper() for f in self.allowed_formats.split(",") if f.strip())

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def originals_dir(self) -> Path:
        return self.data_dir / "originals"

    @property
    def crops_dir(self) -> Path:
        return self.data_dir / "crops"

    @property
    def staging_dir(self) -> Path:
        return self.data_dir / "staging"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def heartbeat_file(self) -> Path:
        return self.data_dir / "worker.heartbeat"

    def ensure_dirs(self) -> None:
        for d in (
            self.data_dir,
            self.originals_dir,
            self.crops_dir,
            self.staging_dir,
            self.index_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
            d.chmod(0o700)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
