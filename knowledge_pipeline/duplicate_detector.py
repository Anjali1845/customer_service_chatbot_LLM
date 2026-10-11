"""
Duplicate detection for incoming knowledge documents.

``register_document`` had the same latent defect ``save_version`` had:
a plain ``INSERT INTO documents`` against a table whose ``file_hash``
column was UNIQUE. Registering a document whose content (and therefore
hash) had been seen before -- a re-registration, or a revert --
crashed with ``sqlite3.IntegrityError``. It now runs the schema
upgrade first and is idempotent per (filename, file_hash): a hash that
is already registered is left exactly as it is instead of being
inserted twice.
"""

import hashlib
import importlib.util
import sqlite3
from pathlib import Path
import shutil
from version_manager import save_version

DB_PATH = "metadata.db"

APPROVED = Path("knowledge_pipeline/approved")
QUARANTINE = Path("knowledge_pipeline/quarantine")

_DATABASE_MODULE = None


def _database_module():
    """
    The sibling ``database`` module, loaded without touching sys.path.

    Same pattern as ``version_manager``: plain import when this
    directory is on ``sys.path`` (the normal script case), by-path load
    otherwise.
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
        "kp_database_by_duplicate_detector", path
    )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    _DATABASE_MODULE = module

    return module


def generate_hash(file_path):
    sha = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(4096):
            sha.update(chunk)

    return sha.hexdigest()


def is_duplicate(file_hash, db_path=None):
    """
    Whether a document with this content hash is already registered.

    ``db_path`` is optional and defaults to the historical relative
    ``DB_PATH``; passing an absolute path keeps the check pointed at
    the database the caller means, wherever it was launched from.
    """

    conn = sqlite3.connect(db_path or DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM documents WHERE file_hash=?",
        (file_hash,)
    )

    result = cursor.fetchone()
    conn.close()

    return result is not None


def register_document(filename, file_hash, db_path=None):
    """
    Record a document as approved.

    Idempotent per (filename, file_hash): registering the same content
    again is a no-op rather than a duplicate row, and no longer a
    ``sqlite3.IntegrityError`` on a legacy UNIQUE(file_hash) table.

    ``db_path`` is optional, like ``version_manager.save_version``'s,
    so a caller working from another directory registers into the
    database the documents actually live in.
    """

    conn = sqlite3.connect(db_path or DB_PATH)

    try:
        # Upgrade a legacy UNIQUE(file_hash) table in place; no-op when
        # the schema is current.
        _database_module().ensure_documents_schema(conn)

        cursor = conn.cursor()

        cursor.execute("""
            SELECT id FROM documents
            WHERE filename=? AND file_hash=?
        """, (filename, file_hash))

        already = cursor.fetchone()

        if already is None:
            cursor.execute("""
                INSERT INTO documents
                (filename, version, file_hash, status)
                VALUES (?, ?, ?, ?)
            """, (filename, 1, file_hash, "approved"))

        conn.commit()

    finally:
        conn.close()


def process_document(file_path):
    file_path = Path(file_path)

    file_hash = generate_hash(file_path)

    if is_duplicate(file_hash):
        destination = QUARANTINE / file_path.name
        shutil.copy(file_path, destination)

        print("❌ Duplicate detected")
        print(f"Moved to: {destination}")

    else:
        destination = APPROVED / file_path.name
        shutil.copy(file_path, destination)

        save_version(file_path.name, file_hash)
        print("✅ New document approved")
        print(f"Saved to: {destination}")


if __name__ == "__main__":
    sample = input("Enter document path: ")
    process_document(sample)
