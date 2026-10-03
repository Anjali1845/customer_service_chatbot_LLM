"""
TASK 3 - PIPELINE MONITORING

Records what a knowledge-base update run did, including how many
support escalations it caused.

Two things were wrong before:

* ``escalations`` was passed as a literal ``0`` by every caller, so the
  column could never say anything. It is now read from the support
  store's real escalation rows, and the hardcoded argument is gone.

* There were two copies of this file, ``monitor.py`` and
  ``" monitor.py"`` (a leading space), which disagreed about
  ``DB_PATH``. The leading-space copy pointed at
  ``knowledge_pipeline/metadata.db``, which does not exist, so it would
  have quietly created a second, empty log database. It has been
  deleted.

``DB_PATH`` is now anchored to the project root instead of the current
working directory, so the log always lands in the same database as the
documents it describes, whatever directory the caller runs from.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The project-root database that database.py, version_manager.py and
# rollback.py all write to.
DB_PATH = PROJECT_ROOT / "metadata.db"

# Where the conversation-side escalation store lives.
SUPPORT_DB_PATH = PROJECT_ROOT / "support.db"


def _as_utc(moment: Optional[datetime]) -> datetime:
    """Normalise to aware UTC, assuming naive input is already UTC."""

    if moment is None:

        return datetime.now(timezone.utc)

    if moment.tzinfo is None:

        return moment.replace(tzinfo=timezone.utc)

    return moment.astimezone(timezone.utc)


def escalation_count(
    db_path: Optional[Path] = None,
    since: Optional[datetime] = None,
) -> int:
    """
    How many escalations the support flow has actually raised.

    Reads the ``escalations`` table written by
    ``src/support_store.py``. Returns 0 when the store does not exist
    yet -- a knowledge-base update on a fresh install genuinely has no
    escalations -- but never pretends a missing store is a measured
    zero: ``log_pipeline_event`` says so on stdout.
    """

    path = Path(db_path or SUPPORT_DB_PATH)

    if not path.is_file():

        return 0

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row

    try:

        if since is not None:

            row = connection.execute(
                "SELECT COUNT(*) AS total FROM escalations "
                "WHERE escalated_at >= ?",
                (_as_utc(since).isoformat(),),
            ).fetchone()

        else:

            row = connection.execute(
                "SELECT COUNT(*) AS total FROM escalations"
            ).fetchone()

    except sqlite3.OperationalError:

        # No escalations table yet: an empty store, not an error.
        return 0

    finally:

        connection.close()

    return int(row["total"]) if row else 0


def log_pipeline_event(
    filename,
    status,
    latency,
    confidence,
    escalations=None,
    now: Optional[datetime] = None,
    support_db: Optional[Path] = None,
    db_path: Optional[Path] = None,
    since: Optional[datetime] = None,
):
    """
    Record one pipeline event.

    ``escalations`` is now optional. Leaving it out reads the real
    count from the support store, which is what the callers that used
    to hardcode ``0`` should do. Passing it explicitly still works, for
    a caller that has already counted them.

    ``now`` and ``db_path`` are injectable so the same simulated clock
    that drives the flow drives the log, and so a test can write to a
    temp database instead of the real one.
    """

    path = Path(db_path or DB_PATH)

    if escalations is None:

        escalations = escalation_count(support_db, since=since)

        if not Path(support_db or SUPPORT_DB_PATH).is_file():

            print(
                "⚠️ No support store found at "
                f"{Path(support_db or SUPPORT_DB_PATH)}; "
                "recording 0 escalations."
            )

    conn = sqlite3.connect(path)
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
        _as_utc(now).strftime("%Y-%m-%d %H:%M:%S"),
    ))

    conn.commit()
    conn.close()

    print("📊 Log recorded successfully!")


def recent_events(
    limit: int = 10,
    db_path: Optional[Path] = None,
):
    """The most recent pipeline events, newest first."""

    path = Path(db_path or DB_PATH)

    if not path.is_file():

        return []

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row

    try:

        rows = connection.execute(
            "SELECT filename, status, latency, confidence, "
            "escalations, timestamp FROM pipeline_logs "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()

    except sqlite3.OperationalError:

        return []

    finally:

        connection.close()

    return [dict(row) for row in rows]


def escalation_summary(
    db_path: Optional[Path] = None,
) -> dict:
    """
    Escalation counts by condition, from the support store.

    The monitoring view that the hardcoded 0 made impossible: which
    conditions are actually firing.
    """

    path = Path(db_path or SUPPORT_DB_PATH)

    if not path.is_file():

        return {}

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row

    try:

        rows = connection.execute(
            "SELECT condition_id, COUNT(*) AS total FROM escalations "
            "GROUP BY condition_id ORDER BY total DESC"
        ).fetchall()

    except sqlite3.OperationalError:

        return {}

    finally:

        connection.close()

    return {row["condition_id"]: int(row["total"]) for row in rows}


if __name__ == "__main__":

    print("📊 PIPELINE MONITOR")
    print("   log database:     ", DB_PATH)
    print("   support store:    ", SUPPORT_DB_PATH)
    print()

    print("Escalations recorded by the support flow:",
          escalation_count())

    summary = escalation_summary()

    if summary:

        print("\nBy condition:")

        for condition, total in summary.items():

            print(f"   {condition}: {total}")

    else:

        print("   (no escalations recorded yet)")

    events = recent_events(5)

    if events:

        print("\nRecent pipeline events:")

        for event in events:

            print(
                f"   {event['timestamp']}  {event['filename']}  "
                f"{event['status']}  escalations={event['escalations']}"
            )

    else:

        print("\n   (no pipeline events recorded yet)")
