import sqlite3
from pathlib import Path

DB_PATH = Path("metadata.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():
    conn = get_connection()
    cursor = conn.cursor()

    # Main documents table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        version INTEGER NOT NULL,
        file_hash TEXT UNIQUE NOT NULL,
        status TEXT NOT NULL,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        active INTEGER DEFAULT 1
    )
    """)

    # Version history table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        version INTEGER,
        file_hash TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()
    print("✅ metadata.db created successfully")


if __name__ == "__main__":
    initialize_database()