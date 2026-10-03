import sqlite3

DB = "metadata.db"


def rollback(filename, target_version, db_path=None):
    """
    Reactivate a previous version of a document.

    ``db_path`` is optional and defaults to the historical relative
    ``"metadata.db"``, so existing callers are unaffected. It exists so
    that a caller working from another directory -- the scheduler --
    restores a version in the database the documents actually live in,
    rather than in a second empty one created by the relative path.
    """

    conn = sqlite3.connect(db_path or DB)
    cur = conn.cursor()

    # Deactivate all versions
    cur.execute(
        "UPDATE documents SET active=0 WHERE filename=?",
        (filename,)
    )

    # Activate target version
    cur.execute("""
        UPDATE documents
        SET active=1
        WHERE filename=? AND version=?
    """, (filename, target_version))

    conn.commit()
    conn.close()

    print(f"✅ Rolled back {filename} to Version {target_version}")


if __name__ == "__main__":
    file = input("Filename: ")
    version = int(input("Rollback to version: "))
    rollback(file, version)