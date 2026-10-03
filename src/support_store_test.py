"""
SUPPORT STORE TEST SUITE

Covers:
- Round-trip fidelity for tickets, escalations, policies, agents and
  routing decisions
- Simulated datetimes preserved to the second
- Nested list/dict columns surviving JSON encoding
- Filtering by session, status, condition and team
- Merging updates without dropping untouched fields
- Ticket links readable from either end
- The escalation count that monitoring now reads
- Every test running against a temp database
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

from support_store import DEFAULT_DB_PATH, SupportStore
from task7_escalation import EscalationEngine, EscalationConfig
from task7_sentiment import analyse_message
from business_calendar import BusinessCalendarConfig
from task6_multilingual import ConversationMessage, ConversationSession


PASS = 0
FAIL = 0


def check(name, condition, details=""):
    global PASS, FAIL

    if condition:
        PASS += 1
        print(f"✅ {name}")
    else:
        FAIL += 1
        print(f"❌ {name}")
        if details:
            print(f"   {details}")


def utc(year, month, day, hour=0, minute=0, second=0):
    return datetime(
        year, month, day, hour, minute, second, tzinfo=timezone.utc
    )


print("=" * 75)
print("SUPPORT STORE TESTS")
print("=" * 75)

folder = tempfile.TemporaryDirectory()

db_path = Path(folder.name) / "support.db"

store = SupportStore(db_path)

store.initialize()

MOMENT = utc(2026, 10, 2, 10, 0, 0)

TICKET = {
    "ticket_id": "TCK-1001",
    "session_id": "sess-a",
    "status": "open",
    "priority": "P2",
    "priority_score": 0.71,
    "category": "payment",
    "severity": "high",
    "issue": "Charged twice for ORD-12345",
    "order_id": "ORD-12345",
    "amount": "1499.00",
    "sentiment_label": "frustrated",
    "sentiment_confidence": 0.78,
    "risk_topics": ["duplicate_payment"],
    "evidence": ["charged twice", "again and again"],
    "missing_fields": ["contact"],
    "created_at": "2026-10-02T10:00:00+00:00",
    "due_at": "2026-10-02T14:00:00+00:00",
    "warn_at": "2026-10-02T13:00:00+00:00",
    "breached": False,
    "masked_handoff": "customer: my card ******3210 was charged twice",
}


# ============================================================
# 1. SCHEMA AND INITIALISE
# ============================================================

print("\n[1] SCHEMA")

check("the database file is created", db_path.is_file())

store.initialize()

check("initialize is idempotent", db_path.is_file())

check(
    "the default path is not the temp path",
    Path(DEFAULT_DB_PATH) != db_path,
)

check(
    "the temp database stays inside the temp folder",
    str(db_path).startswith(folder.name),
)


# ============================================================
# 2. TICKETS
# ============================================================

print("\n[2] TICKETS")

saved_id = store.save_ticket(TICKET)

check("save_ticket returns the id", saved_id == "TCK-1001")

loaded = store.get_ticket("TCK-1001")

check("the ticket round-trips", loaded is not None)

check(
    "scalar fields survive",
    loaded["priority"] == "P2"
    and loaded["category"] == "payment"
    and loaded["order_id"] == "ORD-12345",
    str(loaded),
)

check(
    "the float survives",
    loaded["priority_score"] == 0.71,
    str(loaded.get("priority_score")),
)

check(
    "list fields survive as lists",
    loaded["risk_topics"] == ["duplicate_payment"]
    and loaded["evidence"] == ["charged twice", "again and again"]
    and loaded["missing_fields"] == ["contact"],
    str(loaded.get("evidence")),
)

check(
    "the simulated created_at survives to the second",
    loaded["created_at"] == "2026-10-02T10:00:00+00:00",
    str(loaded.get("created_at")),
)

check(
    "the derived SLA deadlines survive",
    loaded["due_at"] == "2026-10-02T14:00:00+00:00"
    and loaded["warn_at"] == "2026-10-02T13:00:00+00:00",
)

check(
    "the boolean survives as a boolean",
    loaded["breached"] is False,
    repr(loaded.get("breached")),
)

check("a missing ticket reads as None", store.get_ticket("TCK-NOPE") is None)

# A second ticket in a different session and status.
store.save_ticket(
    {
        "ticket_id": "TCK-1002",
        "session_id": "sess-b",
        "status": "awaiting_info",
        "priority": "P3",
        "category": "account",
        "created_at": "2026-10-02T11:00:00+00:00",
    }
)

check(
    "list_tickets returns both",
    len(store.list_tickets()) == 2,
)

check(
    "list_tickets filters by session",
    [row["ticket_id"] for row in store.list_tickets(session_id="sess-b")]
    == ["TCK-1002"],
)

check(
    "list_tickets filters by status",
    [row["ticket_id"] for row in store.list_tickets(status="open")]
    == ["TCK-1001"],
)

check(
    "list_tickets respects a limit",
    len(store.list_tickets(limit=1)) == 1,
)

check(
    "list_tickets orders by creation time",
    [row["ticket_id"] for row in store.list_tickets()]
    == ["TCK-1001", "TCK-1002"],
)

# ------------------------------------------------------------
# merging updates
# ------------------------------------------------------------
updated = store.update_ticket("TCK-1001", status="resolved", priority="P4")

check(
    "update_ticket merges the new fields",
    updated["status"] == "resolved" and updated["priority"] == "P4",
)

check(
    "update_ticket keeps untouched fields",
    updated["order_id"] == "ORD-12345"
    and updated["evidence"] == ["charged twice", "again and again"],
    str(updated),
)

check(
    "the merged state is persisted",
    store.get_ticket("TCK-1001")["status"] == "resolved",
)

check(
    "updating a missing ticket returns None",
    store.update_ticket("TCK-NOPE", status="open") is None,
)

check(
    "saving the same id replaces rather than duplicates",
    len(store.list_tickets()) == 2,
)


# ============================================================
# 3. TICKET LINKS
# ============================================================

print("\n[3] TICKET LINKS")

store.save_ticket_link(
    "TCK-1002", "TCK-1001", "duplicate", 0.92, MOMENT
)

links = store.list_ticket_links()

check("the link is stored", len(links) == 1)

check(
    "the link round-trips",
    links[0]["relation"] == "duplicate"
    and links[0]["score"] == 0.92
    and links[0]["created_at"] == "2026-10-02T10:00:00+00:00",
    str(links[0]),
)

check(
    "the link is found from the source end",
    len(store.list_ticket_links("TCK-1002")) == 1,
)

check(
    "the link is found from the target end",
    len(store.list_ticket_links("TCK-1001")) == 1,
)

check(
    "an unrelated ticket has no links",
    store.list_ticket_links("TCK-9999") == [],
)


# ============================================================
# 4. ESCALATIONS
# ============================================================

print("\n[4] ESCALATIONS")

# Build a real EscalationResult so the dataclass path is exercised,
# not just the dict path.
engine = EscalationEngine(
    EscalationConfig(
        calendar=BusinessCalendarConfig(timezone_name="UTC")
    )
)

text = "My card was charged twice and this is very bad."

session = ConversationSession(
    session_id="sess-esc",
    created_at=MOMENT,
    last_activity=MOMENT,
)

session.messages.append(
    ConversationMessage(
        role="user",
        content=text,
        timestamp=MOMENT,
        language="english",
    )
)

result = engine.evaluate(
    "sess-esc", text, analyse_message(text), session, MOMENT
)

check(
    "the escalation result has records",
    len(result.records) > 0,
    f"got {result.condition_ids}",
)

ids = store.save_escalations(result.records)

check("save_escalations returns ids", len(ids) == len(result.records))

check(
    "the escalation count matches",
    store.count_escalations() == len(result.records),
    f"got {store.count_escalations()}",
)

check(
    "the count filters by session",
    store.count_escalations("sess-esc") == len(result.records)
    and store.count_escalations("sess-nobody") == 0,
)

stored = store.list_escalations(session_id="sess-esc")

first_record = result.records[0]

check(
    "the escalation round-trips",
    stored[0]["escalation_id"] == first_record.escalation_id,
    str(stored[0].get("escalation_id")),
)

check(
    "the reason survives",
    stored[0]["reason"] == first_record.reason,
)

check(
    "the activated condition survives",
    stored[0]["condition_id"] == first_record.condition_id
    and stored[0]["condition"] == first_record.condition,
)

check(
    "the simulated escalated_at survives to the second",
    stored[0]["escalated_at"] == "2026-10-02T10:00:00+00:00",
    str(stored[0].get("escalated_at")),
)

check(
    "the masked summary survives",
    stored[0]["masked_summary"] == first_record.masked_summary,
)

check(
    "no raw card number reached the store",
    all(
        "4111111111111111" not in str(row)
        for row in stored
    ),
)

check(
    "list_escalations filters by condition",
    all(
        row["condition_id"] == "ESC-03"
        for row in store.list_escalations(condition_id="ESC-03")
    )
    and len(store.list_escalations(condition_id="ESC-99")) == 0,
)

check(
    "a scheduled_for of None round-trips as None",
    stored[0]["scheduled_for"] is None
    if first_record.scheduled_for is None
    else stored[0]["scheduled_for"] is not None,
)

# A dict-shaped escalation still works.
store.save_escalation(
    {
        "escalation_id": "ESC-10:sess-esc:T",
        "session_id": "sess-esc",
        "condition_id": "ESC-10",
        "severity": "medium",
        "queue": "priority_support",
        "escalated_at": "2026-10-02T10:05:00+00:00",
    }
)

check(
    "dict escalations are accepted",
    store.count_escalations() == len(result.records) + 1,
)


# ============================================================
# 5. SENTIMENT EVENTS
# ============================================================

print("\n[5] SENTIMENT EVENTS")

sentiment = analyse_message("I called again and again, still not resolved.")

store.save_sentiment_event("sess-a", sentiment, MOMENT)

events = store.list_sentiment_events("sess-a")

check("the sentiment event is stored", len(events) == 1)

check(
    "the label round-trips",
    events[0]["label"] == sentiment.label,
    f"got {events[0].get('label')}",
)

check(
    "the evidence list round-trips",
    events[0]["evidence"] == list(sentiment.evidence),
    str(events[0].get("evidence")),
)

check(
    "the event carries its session and time",
    events[0]["session_id"] == "sess-a"
    and events[0]["created_at"] == "2026-10-02T10:00:00+00:00",
)

check(
    "events filter by session",
    store.list_sentiment_events("sess-none") == []
    and len(store.list_sentiment_events()) == 1,
)


# ============================================================
# 6. SLA POLICIES
# ============================================================

print("\n[6] SLA POLICIES")

policy = {
    "first_response_minutes": 30,
    "resolution_minutes_by_priority": {
        "P1": 60, "P2": 240, "P3": 480, "P4": 1440,
    },
    "warn_fraction": 0.75,
    "exclude_weekends": True,
    "exclude_holidays": True,
}

store.save_sla_policy("default", policy)

loaded_policy = store.get_sla_policy("default")

check("the policy round-trips", loaded_policy is not None)

check(
    "the nested priority table survives",
    loaded_policy["resolution_minutes_by_priority"]
    == {"P1": 60, "P2": 240, "P3": 480, "P4": 1440},
    str(loaded_policy.get("resolution_minutes_by_priority")),
)

check(
    "the warn fraction survives",
    loaded_policy["warn_fraction"] == 0.75,
)

# A runtime change replaces the stored policy in place.
store.save_sla_policy(
    "default",
    dict(policy, resolution_minutes_by_priority={"P1": 30}, warn_fraction=0.5),
)

check(
    "a policy update replaces rather than duplicates",
    len(store.list_sla_policies()) == 1,
)

check(
    "the updated policy is read back",
    store.get_sla_policy("default")["warn_fraction"] == 0.5,
)

check(
    "a missing policy reads as None",
    store.get_sla_policy("nope") is None,
)


# ============================================================
# 7. AGENTS AND ROUTING
# ============================================================

print("\n[7] AGENTS AND ROUTING")

for agent in (
    {
        "agent_id": "AGT-1",
        "name": "Asha",
        "team": "billing",
        "skills": ["payment", "refund"],
        "languages": ["english", "hindi"],
        "available": True,
        "active_tickets": 2,
        "max_capacity": 5,
    },
    {
        "agent_id": "AGT-2",
        "name": "Ravi",
        "team": "security",
        "skills": ["account"],
        "languages": ["english"],
        "available": False,
        "active_tickets": 0,
        "max_capacity": 4,
    },
):

    store.save_agent(agent)

check("both agents are stored", len(store.list_agents()) == 2)

check(
    "the agent skill list round-trips",
    store.list_agents("billing")[0]["skills"] == ["payment", "refund"],
)

check(
    "the availability flag round-trips as a boolean",
    store.list_agents("security")[0]["available"] is False,
    repr(store.list_agents("security")[0].get("available")),
)

check(
    "agents filter by team",
    len(store.list_agents("billing")) == 1
    and store.list_agents("nope") == [],
)

store.save_routing_decision(
    {
        "ticket_id": "TCK-1001",
        "agent_id": "AGT-1",
        "team": "billing",
        "reason": "lowest load in team",
        "decided_at": "2026-10-02T10:00:00+00:00",
    }
)

decisions = store.list_routing_decisions("TCK-1001")

check("the routing decision is stored", len(decisions) == 1)

check(
    "the decision round-trips",
    decisions[0]["agent_id"] == "AGT-1"
    and decisions[0]["reason"] == "lowest load in team"
    and decisions[0]["decided_at"] == "2026-10-02T10:00:00+00:00",
    str(decisions[0]),
)

check(
    "decisions filter by ticket",
    store.list_routing_decisions("TCK-9999") == []
    and len(store.list_routing_decisions()) == 1,
)


# ============================================================
# 8. ISOLATION AND MAINTENANCE
# ============================================================

print("\n[8] ISOLATION AND MAINTENANCE")

second = SupportStore(Path(folder.name) / "other.db")

second.initialize()

second.save_ticket(
    {
        "ticket_id": "TCK-2000",
        "session_id": "sess-c",
        "status": "open",
        "created_at": "2026-10-02T12:00:00+00:00",
    }
)

check(
    "two stores do not share data",
    second.get_ticket("TCK-2000") is not None
    and store.get_ticket("TCK-2000") is None,
)

check(
    "the second store is its own file",
    len(store.list_tickets()) == 2
    and len(second.list_tickets()) == 1,
)

store.clear()

check(
    "clear empties every table",
    len(store.list_tickets()) == 0
    and store.count_escalations() == 0
    and store.list_agents() == []
    and store.list_sla_policies() == [],
)

check(
    "clear leaves the other store untouched",
    len(second.list_tickets()) == 1,
)

store.save_ticket(TICKET)

check(
    "the schema survives a clear",
    store.get_ticket("TCK-1001") is not None,
)

folder.cleanup()

check(
    "the temp database is gone",
    not db_path.exists(),
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 75)
print(f"PASSED: {PASS}")
print(f"FAILED: {FAIL}")
print("=" * 75)

raise SystemExit(0 if FAIL == 0 else 1)
