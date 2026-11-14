#!/usr/bin/env python3
"""Apply database migrations (non-destructive; kept for backward compatibility).

The original script dropped every table. This version only runs `alembic upgrade head` on the
new database (default data/doppelganger.db). It never touches the legacy storage/embeddings.db.
"""

import sys

from app.cli import main

if __name__ == "__main__":
    sys.exit(main(["db", "upgrade"]))
