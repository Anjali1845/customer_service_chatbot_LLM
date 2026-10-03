import sqlite3

DB = "metadata.db"


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
    conn.close()

    print(f"✅ {filename} saved as Version {version}")