#!/usr/bin/env python3
"""Start the API with uvicorn.

Single process on purpose: the in-memory vector index and rate limiter are per process.
"""

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "app.main:app",
        host=os.environ.get("DOPPEL_HOST", "127.0.0.1"),
        port=int(os.environ.get("DOPPEL_PORT", "8000")),
        workers=1,
        log_level=os.environ.get("DOPPEL_LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    main()
