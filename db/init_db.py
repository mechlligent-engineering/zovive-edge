"""Opens (creating if needed) the SQLite outbox database in WAL mode
and applies db/schema.sql. Every module that touches the DB calls
`get_connection()` rather than opening sqlite3 directly, so
WAL/foreign-keys/busy-timeout are set consistently everywhere.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import paths

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"

_local = threading.local()


def _configure(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.row_factory = sqlite3.Row


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Returns a connection, one per thread (SQLite connections aren't
    safe to share across threads). Safe to call every time you need
    the DB — it's cached per-thread, not reopened each call."""
    path = str(db_path) if db_path else str(paths.DB_PATH)
    cache_key = f"_conn_{path}"
    conn = getattr(_local, cache_key, None)
    if conn is not None:
        return conn

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=True)
    _configure(conn)
    setattr(_local, cache_key, conn)
    return conn


def init_db(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Create the schema if it doesn't exist yet. Idempotent (schema.sql
    uses IF NOT EXISTS everywhere)."""
    conn = get_connection(db_path)
    sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    conn.executescript(sql)
    conn.commit()
    return conn
