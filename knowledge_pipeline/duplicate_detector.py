import hashlib
import sqlite3
from pathlib import Path
import shutil
from version_manager import save_version    

DB_PATH = "metadata.db"

APPROVED = Path("knowledge_pipeline/approved")
QUARANTINE = Path("knowledge_pipeline/quarantine")


def generate_hash(file_path):
    sha = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(4096):
            sha.update(chunk)

    return sha.hexdigest()


def is_duplicate(file_hash):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM documents WHERE file_hash=?",
        (file_hash,)
    )

    result = cursor.fetchone()
    conn.close()

    return result is not None


def register_document(filename, file_hash):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO documents
        (filename, version, file_hash, status)
        VALUES (?, ?, ?, ?)
    """, (filename, 1, file_hash, "approved"))

    conn.commit()
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