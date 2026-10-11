"""
Task 3 metadata store.

The historical defect this module now guards against: ``documents`` was
created with ``file_hash TEXT UNIQUE NOT NULL``. A content hash is
unique per *content*, not per *row*: reverting a document to an older
version re-saves a hash that is already in the table, and the UNIQUE
constraint turned that legitimate revert into
``sqlite3.IntegrityError: UNIQUE constraint failed: documents.file_hash``.

The column is now plain ``NOT NULL`` with a *non-unique* index for
lookups, so any number of rows may share a hash. ``ensure_documents_schema``
upgrades an existing database in place (detecting the legacy UNIQUE
column via ``PRAGMA table_info`` / ``PRAGMA index_list`` and rebuilding
the table with its data), so an old ``metadata.db`` keeps working
instead of blowing up on the next revert.
"""

import sqlite3
from pathlib import Path
from typing import Optional, Union

DB_PATH = Path("metadata.db")

# Column order of the documents table. Shared by the CREATE and by the
# legacy-to-current migration so the two can never drift apart.
DOCUMENTS_COLUMNS = (
    "id",
    "filename",
    "version",
    "file_hash",
    "status",
    "uploaded_at",
    "active",
)

DOCUMENTS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        version INTEGER NOT NULL,
        file_hash TEXT NOT NULL,
        status TEXT NOT NULL,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        active INTEGER DEFAULT 1
    )
"""

VERSIONS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        version INTEGER,
        file_hash TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
"""

# Non-unique: a hash may legitimately appear in many rows (a revert
# re-saves it). Lookups stay fast without the crash.
HASH_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_documents_file_hash "
    "ON documents (file_hash)"
)

# The same table under its migration name, built from the canonical
# DDL above so the two shapes cannot drift apart.
MIGRATION_TABLE_SQL = DOCUMENTS_TABLE_SQL.replace(
    "IF NOT EXISTS documents", "documents_migration"
)


def get_connection(db_path: Optional[Union[str, Path]] = None):
    """Open the metadata database, or create its file if absent."""

    conn = sqlite3.connect(db_path if db_path is not None else DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _table_columns(cursor, table: str) -> list:
    """Column names of ``table`` per ``PRAGMA table_info`` (empty if absent)."""

    return [
        row[1]
        for row in cursor.execute(f'PRAGMA table_info("{table}")')
    ]


def _unique_hash_index(cursor) -> Optional[str]:
    """
    The name of a UNIQUE index over exactly ``documents.file_hash``,
    or None. That index is the legacy crash-on-revert constraint --
    whether it came from the column-level ``UNIQUE`` (origin ``u``) or a
    hand-written ``CREATE UNIQUE INDEX`` (origin ``c``).
    """

    columns = _table_columns(cursor, "documents")

    if not columns:
        # Table missing entirely: nothing to migrate.
        return None

    if "file_hash" not in columns:
        return None

    for index in cursor.execute(
        "PRAGMA index_list(documents)"
    ).fetchall():

        # Row shape: (seq, name, unique, origin, partial).
        if not index[2]:
            continue

        name = index[1].replace('"', '""')

        columns = [
            row[2]
            for row in cursor.execute(
                f'PRAGMA index_info("{name}")'
            ).fetchall()
        ]

        if columns == ["file_hash"]:
            return index[1]

    return None


def _rebuild_documents_table(cursor) -> None:
    """
    Rebuild ``documents`` without the UNIQUE hash constraint, keeping
    every row and id.

    SQLite cannot drop a column constraint in place, so the table is
    copied into a fresh one with the current shape and renamed. The
    caller has already confirmed the legacy constraint exists, and the
    whole operation runs inside one transaction so a failure leaves the
    original table untouched.
    """

    cursor.execute("DROP TABLE IF EXISTS documents_migration")

    cursor.execute(MIGRATION_TABLE_SQL)

    present = _table_columns(cursor, "documents")

    shared = [
        column for column in DOCUMENTS_COLUMNS if column in present
    ]

    cursor.execute(
        f"INSERT INTO documents_migration ({', '.join(shared)}) "
        f"SELECT {', '.join(shared)} FROM documents"
    )

    cursor.execute("DROP TABLE documents")

    cursor.execute(
        "ALTER TABLE documents_migration RENAME TO documents"
    )

    # AUTOINCREMENT bookkeeping: drop the stale row left under the
    # migration name; the renamed table re-seeds itself from the
    # preserved ids on the next insert. sqlite_sequence only exists
    # once an AUTOINCREMENT table has been written to.
    try:
        cursor.execute(
            "DELETE FROM sqlite_sequence WHERE name='documents_migration'"
        )
    except sqlite3.OperationalError:
        pass


def ensure_documents_schema(conn) -> bool:
    """
    Upgrade a legacy ``documents`` table (UNIQUE ``file_hash``) in place.

    Safe to call on every connection: when the schema is already
    current it only creates the lookup index and returns False. Returns
    True only when a legacy table was actually rebuilt.

    Does not create missing tables -- that is ``initialize_database``'s
    job -- so a save against a database that never had them fails
    exactly as it always did.
    """

    cursor = conn.cursor()
    migrated = False

    legacy_index = _unique_hash_index(cursor)

    if legacy_index is not None:
        _rebuild_documents_table(cursor)
        migrated = True

    # The lookup index, unique-free, on every path where the table
    # exists. It is deliberately not created for a database that has
    # no documents table: creating one there would mask a genuinely
    # broken database behind a different error.
    if _table_columns(cursor, "documents"):
        cursor.execute(HASH_INDEX_SQL)

    conn.commit()

    return migrated


def initialize_database(
    db_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Create the metadata database, or upgrade an existing one.

    On a database whose ``documents`` table still carries the legacy
    UNIQUE ``file_hash`` column the table is rebuilt in place with all
    its rows, so re-saving a previously-seen hash (a version revert)
    works instead of raising ``sqlite3.IntegrityError``.
    """

    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute(DOCUMENTS_TABLE_SQL)
    cursor.execute(VERSIONS_TABLE_SQL)

    migrated = ensure_documents_schema(conn)

    conn.close()

    if migrated:
        print("✅ metadata.db upgraded: file_hash UNIQUE constraint removed")
    else:
        print("✅ metadata.db created successfully")


if __name__ == "__main__":
    initialize_database()
