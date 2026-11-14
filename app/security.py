from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from fastapi import Request

from .config import Settings
from .errors import Forbidden, NotFound, RateLimited, Unauthorized


class Scope(StrEnum):
    SEARCH = "search"
    ADMIN = "admin"


@dataclass(frozen=True)
class Principal:
    id: str
    scope: Scope

    def allows(self, required: Scope) -> bool:
        return self.scope is Scope.ADMIN or required is Scope.SEARCH


def _digest(token: str) -> bytes:
    return hashlib.sha256(token.encode()).digest()


class Authenticator:
    """Bearer-token auth from environment-provided secrets. No accounts, no external IdP."""

    def __init__(self, settings: Settings) -> None:
        self.disabled = settings.auth_disabled
        self._admin = _digest(settings.admin_token) if settings.admin_token else None
        self._search = _digest(settings.search_token) if settings.search_token else None
        if not self.disabled and self._admin is None:
            raise RuntimeError(
                'DOPPEL_ADMIN_TOKEN is not set. Generate one (python -c "import secrets; '
                'print(secrets.token_urlsafe(32))") or set DOPPEL_AUTH_DISABLED=true for local '
                "development only."
            )

    def authenticate(self, request: Request, required: Scope) -> Principal:
        if self.disabled:
            return Principal("anonymous", Scope.ADMIN)
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise Unauthorized(
                "A bearer token is required.", headers={"WWW-Authenticate": "Bearer"}
            )
        d = _digest(token.strip())
        principal: Principal | None = None
        if self._admin is not None and hmac.compare_digest(d, self._admin):
            principal = Principal("admin:" + d.hex()[:8], Scope.ADMIN)
        elif self._search is not None and hmac.compare_digest(d, self._search):
            principal = Principal("search:" + d.hex()[:8], Scope.SEARCH)
        if principal is None:
            raise Unauthorized("Invalid token.", headers={"WWW-Authenticate": "Bearer"})
        if not principal.allows(required):
            raise Forbidden("This token is not permitted to perform this action.")
        return principal


class RateLimiter:
    """In-memory token bucket per key. Per-process: run a single API process (see DEPLOYMENT.md)."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._lock = threading.Lock()
        self._buckets: dict[str, tuple[float, float]] = {}

    def check(self, key: str) -> None:
        if self.per_minute <= 0:
            return
        rate = self.per_minute / 60.0
        now = time.monotonic()
        with self._lock:
            tokens, ts = self._buckets.get(key, (float(self.per_minute), now))
            tokens = min(float(self.per_minute), tokens + (now - ts) * rate)
            if tokens < 1.0:
                self._buckets[key] = (tokens, now)
                retry = max(1, int((1.0 - tokens) / rate) + 1)
                raise RateLimited("Rate limit exceeded.", headers={"Retry-After": str(retry)})
            self._buckets[key] = (tokens - 1.0, now)
            if len(self._buckets) > 10_000:
                cutoff = now - 120
                self._buckets = {k: v for k, v in self._buckets.items() if v[1] > cutoff}


class UrlSigner:
    """HMAC-signed, expiring URLs so <img> tags can load photos without exposing storage paths."""

    def __init__(self, settings: Settings) -> None:
        self.ttl = settings.url_ttl_s
        key = settings.url_signing_key
        if key:
            self._key = key.encode()
        else:
            path = settings.data_dir / "url_signing.key"
            settings.data_dir.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "w") as fh:
                    fh.write(secrets.token_hex(32))
            self._key = Path(path).read_text().strip().encode()

    def _mac(self, kind: str, resource_id: str, exp: int) -> str:
        msg = f"{kind}:{resource_id}:{exp}".encode()
        return hmac.new(self._key, msg, hashlib.sha256).hexdigest()

    def sign(self, kind: str, resource_id: str) -> tuple[int, str]:
        exp = int(time.time()) + self.ttl
        return exp, self._mac(kind, resource_id, exp)

    def verify(self, kind: str, resource_id: str, exp: int, sig: str) -> None:
        if exp < time.time() or not hmac.compare_digest(self._mac(kind, resource_id, exp), sig):
            raise NotFound("File not found.")
