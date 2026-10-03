"""
SUPPORT OPS - PERSISTENCE

SQLite storage for tickets, ticket links, escalations, sentiment
events, SLA policies, agents and routing decisions.

Conventions, kept deliberately close to
``knowledge_pipeline/database.py`` so the two subsystems look alike:

* ``sqlite3`` with ``row_factory = sqlite3.Row``
* ``CREATE TABLE IF NOT EXISTS`` on every initialise
* the database path is injectable, so tests use a temp file and never
  touch the default

One difference: each table carries a few indexed columns for filtering
plus a ``payload`` column holding the canonical JSON. The JSON is what
reads return. That keeps round-tripping exact -- nested structures and
ISO timestamps come back byte-identical, which is what makes a
simulated clock survive a save/load cycle.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from support_deps import as_utc


DEFAULT_DB_PATH = Path(
    __file__
).resolve().parent.parent / "support.db"


# ============================================================
# SERIALISATION
# ============================================================

def encode_value(value: Any) -> Any:
    """
    Convert a value into something sqlite3 can store.

    Datetimes become ISO strings so a simulated time round-trips to
    the second; anything structured becomes JSON.
    """

    if value is None or isinstance(value, (int, float, str)):

        return value

    if isinstance(value, bool):

        return int(value)

    if isinstance(value, datetime):

        return as_utc(value).isoformat()

    if isinstance(value, date):

        return value.isoformat()

    if is_dataclass(value) and not isinstance(value, type):

        return json.dumps(asdict(value), default=str, sort_keys=True)

    if isinstance(value, (list, tuple, dict)):

        return json.dumps(value, default=str, sort_keys=True)

    return str(value)


def to_payload(record: Any) -> str:
    """
    The canonical JSON for a record.

    Anything with ``to_dict()`` -- every support-ops dataclass has one
    -- is asked for it, so the stored form is the documented one
    rather than an ``asdict`` of whatever fields happen to exist.
    """

    if hasattr(record, "to_dict"):

        data = record.to_dict()

    elif is_dataclass(record) and not isinstance(record, type):

        data = asdict(record)

    elif isinstance(record, dict):

        data = record

    else:

        raise TypeError(
            f"Cannot store {type(record).__name__}: expected a "
            f"dataclass, a dict, or an object with to_dict()."
        )

    return json.dumps(data, default=str, sort_keys=True)


def from_payload(payload: Optional[str]) -> Optional[Dict[str, Any]]:

    if not payload:

        return None

    return json.loads(payload)


# ============================================================
# STORE
# ============================================================

class SupportStore:
    """
    All support-ops persistence.

    Every method opens and closes its own connection, mirroring
    ``knowledge_pipeline``. SQLite is fast enough at this scale and it
    keeps the object safe to share across Streamlit reruns.
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
    ):

        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH

    # connection
    def connect(self) -> sqlite3.Connection:

        connection = sqlite3.connect(str(self.db_path))

        connection.row_factory = sqlite3.Row

        return connection

    def initialize(self) -> None:
        """
        Create every table. Safe to call on each start-up.
        """

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        connection = self.connect()

        cursor = connection.cursor()

        # Filterable columns sit alongside the canonical payload.
        schema = {

            "support_tickets": """
                ticket_id TEXT PRIMARY KEY,
                session_id TEXT,
                status TEXT,
                priority TEXT,
                category TEXT,
                duplicate_of TEXT,
                group_id TEXT,
                created_at TEXT,
                payload TEXT NOT NULL
            """,

            "ticket_links": """
                link_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_ticket_id TEXT NOT NULL,
                target_ticket_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                score REAL,
                created_at TEXT,
                payload TEXT NOT NULL
            """,

            "escalations": """
                escalation_id TEXT PRIMARY KEY,
                session_id TEXT,
                condition_id TEXT,
                severity TEXT,
                queue TEXT,
                escalated_at TEXT,
                payload TEXT NOT NULL
            """,

            "sentiment_events": """
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                label TEXT,
                language TEXT,
                created_at TEXT,
                payload TEXT NOT NULL
            """,

            "sla_policies": """
                name TEXT PRIMARY KEY,
                updated_at TEXT,
                payload TEXT NOT NULL
            """,

            "agents": """
                agent_id TEXT PRIMARY KEY,
                team TEXT,
                available INTEGER,
                payload TEXT NOT NULL
            """,

            "routing_decisions": """
                decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT,
                agent_id TEXT,
                team TEXT,
                decided_at TEXT,
                payload TEXT NOT NULL
            """,
        }

        for table, columns in schema.items():

            cursor.execute(
                f"CREATE TABLE IF NOT EXISTS {table} ({columns})"
            )

        connection.commit()
        connection.close()

    # generic helpers
    def _write(
        self,
        sql: str,
        parameters: Sequence[Any],
    ) -> None:

        connection = self.connect()

        try:

            connection.execute(sql, tuple(parameters))

            connection.commit()

        finally:

            connection.close()

    def _read(
        self,
        sql: str,
        parameters: Sequence[Any] = (),
    ) -> List[Dict[str, Any]]:

        connection = self.connect()

        try:

            rows = connection.execute(sql, tuple(parameters)).fetchall()

        finally:

            connection.close()

        return [
            from_payload(row["payload"]) for row in rows
        ]

    # tickets
    def save_ticket(self, ticket: Any) -> str:
        """
        Insert or replace a ticket. Returns its id.
        """

        data = json.loads(to_payload(ticket))

        ticket_id = data["ticket_id"]

        self._write(
            """
            INSERT OR REPLACE INTO support_tickets (
                ticket_id, session_id, status, priority, category,
                duplicate_of, group_id, created_at, payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticket_id,
                data.get("session_id"),
                data.get("status"),
                data.get("priority"),
                data.get("category"),
                data.get("duplicate_of"),
                data.get("group_id"),
                encode_value(data.get("created_at")),
                to_payload(ticket),
            ),
        )

        return ticket_id

    def get_ticket(
        self, ticket_id: str
    ) -> Optional[Dict[str, Any]]:

        rows = self._read(
            "SELECT payload FROM support_tickets WHERE ticket_id = ?",
            (ticket_id,),
        )

        return rows[0] if rows else None

    def list_tickets(
        self,
        session_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:

        clauses: List[str] = []
        parameters: List[Any] = []

        if session_id:

            clauses.append("session_id = ?")
            parameters.append(session_id)

        if status:

            clauses.append("status = ?")
            parameters.append(status)

        sql = "SELECT payload FROM support_tickets"

        if clauses:

            sql += " WHERE " + " AND ".join(clauses)

        sql += " ORDER BY created_at ASC, ticket_id ASC"

        if limit:

            sql += " LIMIT ?"
            parameters.append(limit)

        return self._read(sql, parameters)

    def update_ticket(
        self, ticket_id: str, **fields: Any
    ) -> Optional[Dict[str, Any]]:
        """
        Merge fields into a stored ticket. Returns the new state, or
        None when the ticket does not exist.
        """

        ticket = self.get_ticket(ticket_id)

        if ticket is None:

            return None

        ticket.update(fields)

        self.save_ticket(ticket)

        return ticket

    # ticket links
    def save_ticket_link(
        self,
        source_ticket_id: str,
        target_ticket_id: str,
        relation: str,
        score: Optional[float] = None,
        now: Optional[datetime] = None,
    ) -> None:

        payload = {
            "source_ticket_id": source_ticket_id,
            "target_ticket_id": target_ticket_id,
            "relation": relation,
            "score": score,
            "created_at": as_utc(now).isoformat(),
        }

        self._write(
            """
            INSERT INTO ticket_links (
                source_ticket_id, target_ticket_id, relation,
                score, created_at, payload
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                source_ticket_id,
                target_ticket_id,
                relation,
                score,
                payload["created_at"],
                json.dumps(payload, sort_keys=True),
            ),
        )

    def list_ticket_links(
        self, ticket_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:

        if ticket_id:

            return self._read(
                """
                SELECT payload FROM ticket_links
                WHERE source_ticket_id = ? OR target_ticket_id = ?
                ORDER BY link_id ASC
                """,
                (ticket_id, ticket_id),
            )

        return self._read(
            "SELECT payload FROM ticket_links ORDER BY link_id ASC"
        )

    # escalations
    def save_escalation(self, record: Any) -> str:
        """
        Store an EscalationRecord (or an equivalent dict).
        """

        data = json.loads(to_payload(record))

        escalation_id = data["escalation_id"]

        self._write(
            """
            INSERT OR REPLACE INTO escalations (
                escalation_id, session_id, condition_id, severity,
                queue, escalated_at, payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                escalation_id,
                data.get("session_id"),
                data.get("condition_id"),
                data.get("severity"),
                data.get("queue"),
                encode_value(data.get("escalated_at")),
                to_payload(record),
            ),
        )

        return escalation_id

    def save_escalations(
        self, records: Sequence[Any]
    ) -> List[str]:
        """
        Store a whole EscalationResult's records at once.
        """

        return [
            self.save_escalation(record) for record in records
        ]

    def list_escalations(
        self,
        session_id: Optional[str] = None,
        condition_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:

        clauses: List[str] = []
        parameters: List[Any] = []

        if session_id:

            clauses.append("session_id = ?")
            parameters.append(session_id)

        if condition_id:

            clauses.append("condition_id = ?")
            parameters.append(condition_id)

        sql = "SELECT payload FROM escalations"

        if clauses:

            sql += " WHERE " + " AND ".join(clauses)

        sql += " ORDER BY escalated_at ASC, escalation_id ASC"

        if limit:

            sql += " LIMIT ?"
            parameters.append(limit)

        return self._read(sql, parameters)

    def count_escalations(
        self, session_id: Optional[str] = None
    ) -> int:
        """
        The real escalation count that
        ``knowledge_pipeline/monitor.py`` was hardcoding as 0.
        """

        connection = self.connect()

        try:

            if session_id:

                row = connection.execute(
                    "SELECT COUNT(*) AS total FROM escalations "
                    "WHERE session_id = ?",
                    (session_id,),
                ).fetchone()

            else:

                row = connection.execute(
                    "SELECT COUNT(*) AS total FROM escalations"
                ).fetchone()

        finally:

            connection.close()

        return int(row["total"])

    # sentiment events
    def save_sentiment_event(
        self,
        session_id: str,
        result: Any,
        now: Optional[datetime] = None,
    ) -> None:

        data = json.loads(to_payload(result))

        payload = dict(data)

        payload["session_id"] = session_id
        payload["created_at"] = as_utc(now).isoformat()

        self._write(
            """
            INSERT INTO sentiment_events (
                session_id, label, language, created_at, payload
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                session_id,
                data.get("label"),
                data.get("language"),
                payload["created_at"],
                json.dumps(payload, default=str, sort_keys=True),
            ),
        )

    def list_sentiment_events(
        self, session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:

        if session_id:

            return self._read(
                "SELECT payload FROM sentiment_events "
                "WHERE session_id = ? ORDER BY event_id ASC",
                (session_id,),
            )

        return self._read(
            "SELECT payload FROM sentiment_events ORDER BY event_id ASC"
        )

    # SLA policies
    def save_sla_policy(self, name: str, policy: Any) -> None:

        self._write(
            """
            INSERT OR REPLACE INTO sla_policies (
                name, updated_at, payload
            ) VALUES (?, ?, ?)
            """,
            (
                name,
                datetime.now().astimezone().isoformat(),
                to_payload(policy),
            ),
        )

    def get_sla_policy(
        self, name: str
    ) -> Optional[Dict[str, Any]]:

        rows = self._read(
            "SELECT payload FROM sla_policies WHERE name = ?",
            (name,),
        )

        return rows[0] if rows else None

    def list_sla_policies(self) -> List[Dict[str, Any]]:

        return self._read(
            "SELECT payload FROM sla_policies ORDER BY name ASC"
        )

    # agents
    def save_agent(self, agent: Any) -> str:

        data = json.loads(to_payload(agent))

        agent_id = data["agent_id"]

        self._write(
            """
            INSERT OR REPLACE INTO agents (
                agent_id, team, available, payload
            ) VALUES (?, ?, ?, ?)
            """,
            (
                agent_id,
                data.get("team"),
                int(bool(data.get("available", True))),
                to_payload(agent),
            ),
        )

        return agent_id

    def list_agents(
        self, team: Optional[str] = None
    ) -> List[Dict[str, Any]]:

        if team:

            return self._read(
                "SELECT payload FROM agents WHERE team = ? "
                "ORDER BY agent_id ASC",
                (team,),
            )

        return self._read(
            "SELECT payload FROM agents ORDER BY agent_id ASC"
        )

    # routing decisions
    def save_routing_decision(self, decision: Any) -> None:

        data = json.loads(to_payload(decision))

        payload = dict(data)

        payload.setdefault(
            "decided_at",
            as_utc(
                data.get("decided_at")
                if isinstance(data.get("decided_at"), datetime)
                else None
            ).isoformat(),
        )

        self._write(
            """
            INSERT INTO routing_decisions (
                ticket_id, agent_id, team, decided_at, payload
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                data.get("ticket_id"),
                data.get("agent_id"),
                data.get("team"),
                encode_value(payload["decided_at"]),
                json.dumps(payload, default=str, sort_keys=True),
            ),
        )

    def list_routing_decisions(
        self, ticket_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:

        if ticket_id:

            return self._read(
                "SELECT payload FROM routing_decisions "
                "WHERE ticket_id = ? ORDER BY decision_id ASC",
                (ticket_id,),
            )

        return self._read(
            "SELECT payload FROM routing_decisions "
            "ORDER BY decision_id ASC"
        )

    # maintenance
    def clear(self) -> None:
        """
        Empty every table. Test helper -- the schema stays.
        """

        connection = self.connect()

        try:

            for table in (
                "support_tickets",
                "ticket_links",
                "escalations",
                "sentiment_events",
                "sla_policies",
                "agents",
                "routing_decisions",
            ):

                connection.execute(f"DELETE FROM {table}")

            connection.commit()

        finally:

            connection.close()


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    import tempfile

    print("=" * 60)
    print("SUPPORT STORE SELF-TEST")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as folder:

        store = SupportStore(Path(folder) / "support.db")

        store.initialize()

        store.save_ticket(
            {
                "ticket_id": "TCK-1",
                "session_id": "sess-1",
                "status": "open",
                "priority": "P2",
                "category": "payment",
                "created_at": "2026-10-02T10:00:00+00:00",
            }
        )

        print("\nStored and read back:")
        print(" ", store.get_ticket("TCK-1"))

        assert store.get_ticket("TCK-1")["priority"] == "P2"

        assert store.list_tickets(session_id="sess-1")[0][
            "ticket_id"
        ] == "TCK-1"

        store.save_escalation(
            {
                "escalation_id": "ESC-03:sess-1:T",
                "session_id": "sess-1",
                "condition_id": "ESC-03",
                "severity": "high",
                "queue": "priority_support",
                "escalated_at": "2026-10-02T10:00:00+00:00",
            }
        )

        print("  escalations:", store.count_escalations())

        assert store.count_escalations() == 1

        print("\nRESULT: PASS")
