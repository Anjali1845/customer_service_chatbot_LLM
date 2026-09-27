import sqlite3

DB = "metadata.db"


def get_next_version(filename):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute(
        "SELECT MAX(version) FROM documents WHERE filename=?",
        (filename,)
    )

    result = cur.fetchone()[0]
    conn.close()

    return 1 if result is None else result + 1


def save_version(filename, file_hash):
    version = get_next_version(filename)

    conn = sqlite3.connect(DB)
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