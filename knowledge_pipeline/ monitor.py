import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path("metadata.db")


def log_pipeline_event(filename, status, latency, confidence, escalations):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            status TEXT,
            latency REAL,
            confidence REAL,
            escalations INTEGER,
            timestamp TEXT
        )
    """)

    cursor.execute("""
        INSERT INTO pipeline_logs
        (filename, status, latency, confidence, escalations, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        filename,
        status,
        latency,
        confidence,
        escalations,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()

    print("📊 Log recorded successfully!")


if __name__ == "__main__":
    log_pipeline_event(
        filename="faq.txt",
        status="Activated",
        latency=1.82,
        confidence=0.96,
        escalations=0
    )