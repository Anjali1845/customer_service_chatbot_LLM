import sqlite3

DB = "metadata.db"


def rollback(filename, target_version):
    conn = sqlite3.connect(DB)
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