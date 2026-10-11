"""
Version bookkeeping for the knowledge pipeline.

``save_version`` used to do a plain ``INSERT INTO documents`` against a
table whose ``file_hash`` column was UNIQUE. Reverting a document to an
older version re-saves a hash that is already stored, and that INSERT
crashed with ``sqlite3.IntegrityError: UNIQUE constraint failed:
documents.file_hash``. Every save now runs ``ensure_documents_schema``
first, which upgrades a legacy database in place (and is a no-op on a
current one), so a revert saves a new *version row* carrying the old
hash -- exactly what a revert means.
"""

import importlib.util
import sqlite3
from pathlib import Path

DB = "metadata.db"

_DATABASE_MODULE = None


def _database_module():
    """
    The sibling ``database`` module, which owns the schema.

    Plain ``import database`` works when this directory is on
    ``sys.path`` (running the file directly, or the scheduler test).
    When this module is itself loaded by path -- the way the scheduler
    wires it -- that import fails, so it falls back to loading
    ``database.py`` by its file path. Never adds a directory to
    ``sys.path``.
    """

    global _DATABASE_MODULE

    if _DATABASE_MODULE is not None:
        return _DATABASE_MODULE

    try:
        import database  # noqa: S107 - sibling module

        _DATABASE_MODULE = database

        return database

    except ImportError:
        pass

    path = Path(__file__).resolve().parent / "database.py"

    spec = importlib.util.spec_from_file_location(
        "kp_database_by_version_manager", path
    )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    _DATABASE_MODULE = module

    return module


def get_next_version(filename, db_path=None):
    """
    The next version number for a filename.

    ``db_path`` is optional and defaults to the historical relative
    ``"metadata.db"``, so every existing caller keeps its behaviour.
    It exists because a relative path resolves against the current
    working directory: a caller running from a different directory --
    the scheduler, for instance -- would otherwise read and write a
    second, empty database and never find the real one.
    """

    conn = sqlite3.connect(db_path or DB)
    cur = conn.cursor()

    cur.execute(
        "SELECT MAX(version) FROM documents WHERE filename=?",
        (filename,)
    )

    result = cur.fetchone()[0]
    conn.close()

    return 1 if result is None else result + 1


def save_version(filename, file_hash, db_path=None):
    version = get_next_version(filename, db_path)

    conn = sqlite3.connect(db_path or DB)

    try:
        # Upgrades a legacy UNIQUE(file_hash) table in place; a no-op
        # when the schema is already current. Without this, saving a
        # hash that an earlier version already used -- a revert --
        # raised sqlite3.IntegrityError.
        _database_module().ensure_documents_schema(conn)

        cur = conn.cursor()

        # Deactivate previous version
        cur.execute(
            "UPDATE documents SET active=0 WHERE filename=?",
            (filename,)
        )

        # Insert new active version
        cur.execute("""
            INSERT INTO documents
            (filename, version, file_hash, status, active)
            VALUES (?, ?, ?, ?, 1)
        """, (filename, version, file_hash, "approved"))

        # Save history
        cur.execute("""
            INSERT INTO versions
            (filename, version, file_hash)
            VALUES (?, ?, ?)
        """, (filename, version, file_hash))

        conn.commit()

    finally:
        conn.close()

    print(f"✅ {filename} saved as Version {version}")
