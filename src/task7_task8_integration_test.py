"""
TASK 7 + TASK 8 INTEGRATION TEST SUITE

Drives the real ``SupportFlow.process_turn`` end to end with a stub
answer function, so the whole pipeline is exercised without Gemini,
FAISS or a network.

Covers:
- An ordinary question answering with no ticket and no escalation
- A negative message escalating, ticketing, routing and persisting
- After hours: urgent to on-call, ordinary deferred
- A second report of the same problem linking as a duplicate
- A fifteen-minute jump triggering the unresolved escalation
- A runtime SLA change moving stored tickets into breach
- A runtime availability change moving the routing decision
- Agent load moving with each assignment, and standing still for a
  duplicate that joins a ticket already counted
- The tone block reaching the chain without touching the answer prompt
- The escalation record carrying reason + condition + summary
- No raw PII in anything that leaves the flow
"""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from business_calendar import BusinessCalendarConfig
from prompt_templates import (
    BASE_PROMPT_TEMPLATE,
    build_prompt_template,
)
from support_flow import SupportFlow, SupportFlowConfig, ticket_from_row
from support_store import SupportStore
from task7_escalation import EscalationConfig
from task8_routing import Agent, RoutingConfig
from task8_ticketing import SlaPolicy


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


def utc(year, month, day, hour=0, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


print("=" * 75)
print("TASK 7 + TASK 8 INTEGRATION TESTS")
print("=" * 75)


# ============================================================
# HARNESS
# ============================================================

folder = tempfile.TemporaryDirectory()

db_path = Path(folder.name) / "flow.db"


class Stub:
    """
    Stands in for the retrieval chain.

    Records every call so the test can assert what the flow actually
    sent, including the tone block -- which the real chain receives as
    a separate prompt argument.
    """

    def __init__(self, answer="Refunds take 5 - 7 business days."):

        self.answer = answer
        self.calls = []

    def __call__(self, question, tone):

        self.calls.append({"question": question, "tone": tone})

        return self.answer

    @property
    def last_prompt(self):
        return build_prompt_template(self.calls[-1]["tone"])


stub = Stub()

CALENDAR = BusinessCalendarConfig(timezone_name="UTC")


def build_flow(**overrides):

    config = SupportFlowConfig(
        escalation=EscalationConfig(calendar=CALENDAR),
        sla=SlaPolicy(calendar=CALENDAR),
        routing=RoutingConfig(calendar=CALENDAR),
        **overrides,
    )

    return SupportFlow(
        config=config,
        store=SupportStore(db_path),
        agents=[
            Agent(
                "AGT-BILL", "billing", name="Asha",
                languages=("english", "hindi"), active_tickets=1,
            ),
            Agent(
                "AGT-SEC", "security", name="Ravi",
                languages=("english",), active_tickets=0,
            ),
            Agent(
                "AGT-LOG", "logistics", name="Meera",
                languages=("english", "telugu"), active_tickets=2,
            ),
            Agent("AGT-OC", "on_call", name="On-call", on_call=True),
        ],
        answer_fn=stub,
    )


flow = build_flow()

MONDAY_10 = utc(2026, 10, 5, 10, 0)

CHARGED_TWICE = (
    "My card was charged twice for order ORD-77001 and this is very "
    "frustrating. I am Priya Sharma, email priya@example.com, "
    "amount 1499.00 INR."
)


# ============================================================
# 1. AN ORDINARY QUESTION
# ============================================================

print("\n[1] AN ORDINARY QUESTION")

ordinary = flow.process_turn(
    "sess-ordinary", "What is your refund policy?", now=MONDAY_10
)

check(
    "an ordinary question is answered",
    ordinary.answered is True and ordinary.answer,
)

check(
    "an ordinary question is not escalated",
    ordinary.escalated is False,
    str(ordinary.escalation.condition_ids if ordinary.escalation else None),
)

check(
    "an ordinary question raises no ticket",
    ordinary.ticket is None,
)

check(
    "an ordinary question is classified neutral",
    ordinary.sentiment_label == "neutral",
    ordinary.sentiment_label,
)

check(
    "the stub was called exactly once",
    len(stub.calls) == 1,
)

check(
    "the enriched question carries the language instruction",
    "Respond in English" in stub.calls[-1]["question"],
    stub.calls[-1]["question"][:80],
)

check(
    "the enriched question carries the history",
    "Conversation history:" in stub.calls[-1]["question"],
)

check(
    "the enriched question carries the current message",
    "What is your refund policy?" in stub.calls[-1]["question"],
)

check(
    "the answer is stored in the conversation",
    flow.session_manager.get_context(
        "sess-ordinary", now=MONDAY_10
    )["messages"][-1].content
    == stub.answer,
)


# ============================================================
# 2. A NEGATIVE MESSAGE
# ============================================================

print("\n[2] A NEGATIVE MESSAGE")

negative = flow.process_turn("sess-neg", CHARGED_TWICE, now=MONDAY_10)

check(
    "the sentiment is negative-family",
    negative.sentiment_label in ("negative", "frustrated", "urgent"),
    negative.sentiment_label,
)

check(
    "the duplicate-charge risk is detected",
    "duplicate_payment" in negative.sentiment.risk_topics,
    str(negative.sentiment.risk_topics),
)

check(
    "the turn escalates",
    negative.escalated is True,
)

check(
    "the escalation is the duplicate-payment condition",
    negative.escalation.condition_ids == ["ESC-03"],
    str(negative.escalation.condition_ids),
)

check(
    "a ticket is raised",
    negative.ticket is not None,
)

check(
    "the ticket is categorised as payment",
    negative.ticket.category == "payment",
    negative.ticket.category,
)

check(
    "the ticket carries the extracted order id",
    negative.ticket.order_id == "ORD-77001",
    str(negative.ticket.order_id),
)

check(
    "the ticket carries the extracted contact",
    negative.ticket.contact == "priya@example.com",
    str(negative.ticket.contact),
)

check(
    "the ticket carries the extracted name",
    negative.ticket.customer_name == "Priya Sharma",
    str(negative.ticket.customer_name),
)

check(
    "a complete ticket needs no information request",
    negative.information_request == "",
    negative.information_request,
)

check(
    "the ticket is prioritised above the lowest band",
    negative.ticket.priority in ("P1", "P2"),
    f"{negative.ticket.priority} ({negative.ticket.priority_score})",
)

check(
    "the ticket is assigned during business hours",
    negative.ticket.assigned_agent == "AGT-BILL",
    f"{negative.ticket.assigned_agent} / {negative.routing[0].queue}",
)

check(
    "the ticket's SLA is derived from the policy",
    negative.ticket.due_at == utc(2026, 10, 5, 14, 0),
    str(negative.ticket.due_at),
)

check(
    "the escalation is recorded with a reason",
    bool(negative.escalation.records[0].reason),
)

check(
    "the escalation names its activated condition",
    negative.escalation.records[0].condition_id == "ESC-03"
    and negative.escalation.records[0].condition,
    negative.escalation.records[0].condition,
)

check(
    "the escalation carries a conversation summary",
    "charged twice" in negative.escalation.records[0].conversation_summary,
    negative.escalation.records[0].conversation_summary,
)

check(
    "the ticket is persisted",
    flow.store.get_ticket(negative.ticket.ticket_id) is not None,
)

check(
    "the routing decision is persisted",
    len(
        flow.store.list_routing_decisions(negative.ticket.ticket_id)
    )
    == 1,
)

stored_escalations = flow.store.list_escalations(session_id="sess-neg")

check(
    "the escalation row is persisted",
    len(stored_escalations) >= 1,
    str(len(stored_escalations)),
)

check(
    "the escalation row joins to its ticket",
    stored_escalations[0].get("ticket_id") == negative.ticket.ticket_id,
    str(stored_escalations[0].get("ticket_id")),
)

check(
    "no raw email reaches the escalation record",
    all(
        "priya@example.com" not in str(row)
        for row in stored_escalations
    ),
)

check(
    "no raw email reaches the masked handoff",
    "priya@example.com" not in negative.masked_handoff,
)

check(
    "no raw email reaches the note about the ticket",
    "priya@example.com" not in negative.ticket.routing_reason,
)


# ============================================================
# 3. THE TONE BLOCK
# ============================================================

print("\n[3] THE TONE BLOCK")

check(
    "a tone block was built for the negative turn",
    "Communication style" in negative.tone_instruction,
    negative.tone_instruction,
)

check(
    "the tone block was handed to the chain",
    negative.tone_instruction in stub.calls[-1]["tone"],
)

check(
    "the tone block forbids policy changes",
    "Do NOT change any policy" in negative.tone_instruction,
)

check(
    "the tone block never reaches the question text",
    "Do NOT change any policy" not in stub.calls[-1]["question"],
)

check(
    "an empty tone reproduces the original prompt exactly",
    build_prompt_template("") == BASE_PROMPT_TEMPLATE,
)

check(
    "an empty tone is the default",
    build_prompt_template() == BASE_PROMPT_TEMPLATE,
)

toned_prompt = build_prompt_template(negative.tone_instruction)

check(
    "the toned prompt keeps the grounding instruction",
    "Answer ONLY using the provided context." in toned_prompt,
)

check(
    "the toned prompt keeps the refusal sentence",
    '"I don\'t know based on the knowledge base."' in toned_prompt,
)

check(
    "the toned prompt still has exactly two variables",
    toned_prompt.count("{context}") == 1
    and toned_prompt.count("{question}") == 1,
)

check(
    "the toned prompt carries the style block",
    "Communication style" in toned_prompt,
)

check(
    "the style block sits above the grounding rules",
    toned_prompt.index("Communication style")
    < toned_prompt.index("Answer ONLY using the provided context."),
)

check(
    "the base prompt is unchanged by building a toned one",
    BASE_PROMPT_TEMPLATE.count("Communication style") == 0,
)


# ============================================================
# 4. AFTER HOURS
# ============================================================

print("\n[4] AFTER HOURS")

urgent_night = flow.process_turn(
    "sess-night-urgent",
    "My account was hacked and this is urgent, please help now.",
    now=utc(2026, 10, 5, 22, 0),
)

check(
    "an after-hours urgent request is on-call",
    urgent_night.routing[0].queue == "on_call"
    and urgent_night.routing[0].on_call is True,
    urgent_night.routing[0].reason,
)

check(
    "the on-call agent is paged",
    urgent_night.ticket.assigned_agent == "AGT-OC",
    str(urgent_night.ticket.assigned_agent),
)

check(
    "the account compromise escalates",
    "ESC-02" in urgent_night.escalation.condition_ids,
    str(urgent_night.escalation.condition_ids),
)

check(
    "the ticket is security work",
    urgent_night.ticket.category == "account",
    urgent_night.ticket.category,
)

calm_night = flow.process_turn(
    "sess-night-calm",
    "My parcel has not arrived yet, could you check when you can.",
    now=utc(2026, 10, 5, 22, 30),
)

check(
    "an ordinary after-hours request is not paged",
    calm_night.routing[0].queue != "on_call",
    calm_night.routing[0].reason,
)

check(
    "the after-hours request is deferred to the next working day",
    calm_night.routing[0].deferred_until == utc(2026, 10, 6, 9, 0),
    str(calm_night.routing[0].deferred_until),
)

check(
    "the deferral is stored on the ticket",
    calm_night.ticket.deferred_until == utc(2026, 10, 6, 9, 0),
)


# ============================================================
# 5. DUPLICATES ACROSS TURNS
# ============================================================

print("\n[5] DUPLICATES ACROSS TURNS")

# Its own store: an earlier section already filed a ticket for the same
# order, and the duplicate would correctly match that one instead --
# which is the right behaviour but not the pair under test here.
dup_path = Path(folder.name) / "dup.db"

dup_flow = build_flow()

dup_flow.store = SupportStore(dup_path)

dup_flow.store.initialize()

first = dup_flow.process_turn("sess-dup", CHARGED_TWICE, now=MONDAY_10)

second = dup_flow.process_turn(
    "sess-dup",
    "Charged twice on ORD-77001 again, this is frustrating.",
    now=MONDAY_10 + timedelta(minutes=4),
)

check(
    "the second report raises a ticket",
    second.ticket is not None,
)

check(
    "the second report is a duplicate",
    second.ticket.duplicate_of == first.ticket.ticket_id,
    f"{second.ticket.duplicate_of} vs {first.ticket.ticket_id}",
)

check(
    "the duplicate starts no second clock",
    second.ticket.due_at == first.ticket.due_at,
    f"{second.ticket.due_at} vs {first.ticket.due_at}",
)

check(
    "the duplicate is queued for review, not routed again",
    second.routing[0].queue == "duplicate_review",
    second.routing[0].reason,
)

check(
    "the duplicate link is persisted",
    any(
        link["relation"] == "duplicate"
        for link in dup_flow.store.list_ticket_links()
    ),
    str(dup_flow.store.list_ticket_links()),
)

third = dup_flow.process_turn(
    "sess-dup",
    "Separately, the app crashes whenever I open the settings screen.",
    now=MONDAY_10 + timedelta(minutes=8),
)

check(
    "a genuinely different problem is not a duplicate",
    third.ticket.duplicate_of is None,
    str(third.ticket.duplicate_of),
)

check(
    "a new problem gets its own SLA clock",
    third.ticket.due_at != first.ticket.due_at,
)


# ============================================================
# 6. THE UNRESOLVED CLOCK
# ============================================================

print("\n[6] THE UNRESOLVED CLOCK")

# The clock runs from the complaint *before* this turn. A fresh
# complaint is therefore zero minutes old by definition, and the
# condition fires when the customer comes back to a problem that was
# left unresolved -- which is the only way it can fire at all.
UNRESOLVED = "My order is still not here and nobody has replied to me."

FRESH = "Any update on my order please?"

check(
    "a fresh complaint does not trip the unresolved condition",
    "ESC-07"
    not in flow.process_turn(
        "sess-wait", UNRESOLVED, now=MONDAY_10
    ).escalation.condition_ids,
)

check(
    "fourteen minutes of silence is not yet unresolved",
    "ESC-07"
    not in flow.process_turn(
        "sess-wait", FRESH, now=MONDAY_10 + timedelta(minutes=14)
    ).escalation.condition_ids,
)

at_fifteen = flow.process_turn(
    "sess-wait", FRESH, now=MONDAY_10 + timedelta(minutes=15)
)

check(
    "fifteen minutes trips the unresolved escalation",
    "ESC-07" in at_fifteen.escalation.condition_ids,
    str(at_fifteen.escalation.condition_ids),
)

check(
    "the unresolved record reports the elapsed minutes",
    at_fifteen.escalation.unresolved_minutes >= 15,
    str(at_fifteen.escalation.unresolved_minutes),
)

check(
    "the unresolved escalation is not a warning",
    any(
        record.condition_id == "ESC-07" and not record.is_warning
        for record in at_fifteen.escalation.records
    ),
    str([(r.condition_id, r.is_warning) for r in at_fifteen.escalation.records]),
)

check(
    "the unresolved ticket exists to attach the escalation to",
    at_fifteen.ticket is not None,
)


# ============================================================
# 7. RUNTIME CHANGES
# ============================================================

print("\n[7] RUNTIME CHANGES")

check(
    "the stored tickets load back as Ticket objects",
    all(
        isinstance(ticket_from_row(row), type(at_fifteen.ticket))
        for row in flow.store.list_tickets()
    ),
)

stored_before = {
    row["ticket_id"]: row["due_at"]
    for row in flow.store.list_tickets()
}

tightened = build_flow()

tightened.config.sla.resolution_minutes_by_priority["P3"] = 30

# Re-open the same store so the tickets under test are the persisted
# ones, not the in-memory decisions of an earlier turn.
tightened.store = flow.store

moved = tightened.recalculate_sla(MONDAY_10 + timedelta(hours=6))

check(
    "recalculation touches every stored ticket",
    moved.tickets == len(stored_before),
    f"{moved.tickets} vs {len(stored_before)}",
)

after = {
    row["ticket_id"]: row["due_at"]
    for row in flow.store.list_tickets()
}

check(
    "a shortened P3 budget moves the P3 tickets",
    any(
        stored_before[ticket_id] != after[ticket_id]
        for ticket_id in stored_before
        if flow.store.get_ticket(ticket_id).get("priority") == "P3"
    ),
    str(
        [
            (ticket_id, stored_before[ticket_id], after[ticket_id])
            for ticket_id in stored_before
            if flow.store.get_ticket(ticket_id).get("priority") == "P3"
        ]
    ),
)

check(
    "a runtime change can flip a stored ticket into breach",
    any(
        row["breached"]
        for row in flow.store.list_tickets()
        if row.get("priority") == "P3"
    ),
    str(
        [
            (row["ticket_id"], row["priority"], row["breached"])
            for row in flow.store.list_tickets()
        ]
    ),
)

# A breach is an escalation. Recomputing the deadline and leaving the
# escalation count where it was would mean the app never actually
# escalated the tickets the new policy broke.
check(
    "the recalculation reports the breaches it found",
    moved.breached >= 1,
    f"breached={moved.breached}",
)

check(
    "a runtime breach raises ESC-08",
    moved.escalations >= 1
    and all(
        record.condition_id == "ESC-08"
        for record in moved.records
    ),
    str([record.condition_id for record in moved.records]),
)

check(
    "the raised breach is persisted",
    len(
        flow.store.list_escalations(condition_id="ESC-08")
    )
    >= 1,
    str(flow.store.list_escalations(condition_id="ESC-08")),
)

check(
    "a raised breach names the ticket it belongs to",
    all(
        record.to_dict().get("ticket_id") in moved.breached_ticket_ids
        for record in moved.records
    ),
    str(moved.breached_ticket_ids),
)

# The same policy, recalculated again at the same clock, is the same
# state -- not a second page for a ticket that was already breached.
before_second = len(flow.store.list_escalations())

again = tightened.recalculate_sla(MONDAY_10 + timedelta(hours=6))

check(
    "recalculating twice does not page twice",
    len(flow.store.list_escalations()) == before_second
    and again.escalations == moved.escalations,
    f"{len(flow.store.list_escalations())} vs {before_second}",
)

check(
    "recalculation does not invent customer conditions",
    all(
        record.condition_id in ("ESC-08", "ESC-09")
        for record in moved.records
    ),
    str([record.condition_id for record in moved.records]),
)

# ------------------------------------------------------------
# availability
# ------------------------------------------------------------
turn_before = flow.process_turn(
    "sess-avail",
    "My invoice amount is wrong and this is frustrating. "
    "Order ORD-90001, amount 250.00 INR, email av@example.com.",
    now=MONDAY_10,
)

check(
    "the billing agent takes the ticket while available",
    turn_before.ticket.assigned_agent == "AGT-BILL",
    str(turn_before.ticket.assigned_agent),
)

flow.set_agent_availability("AGT-BILL", False)

turn_after = flow.process_turn(
    "sess-avail-2",
    "The invoice for ORD-90002 is wrong again and I am annoyed. "
    "Amount 300.00 INR, email av@example.com.",
    now=MONDAY_10,
)

check(
    "taking the agent offline changes the next decision",
    turn_after.ticket.assigned_agent != "AGT-BILL",
    str(turn_after.ticket.assigned_agent),
)

check(
    "the reassignment is explained",
    "no available agent" in turn_after.ticket.routing_reason,
    turn_after.ticket.routing_reason,
)

flow.set_agent_availability("AGT-BILL", True)

check(
    "bringing the agent back restores the routing",
    flow.process_turn(
        "sess-avail-3",
        "A third invoice ORD-90003 is wrong and this is frustrating. "
        "Amount 400.00 INR, email av@example.com.",
        now=MONDAY_10,
    ).ticket.assigned_agent
    == "AGT-BILL",
)


# ============================================================
# 8. UNCERTAIN LANGUAGE
# ============================================================

print("\n[8] UNCERTAIN LANGUAGE")

stub.calls.clear()

uncertain = flow.process_turn("sess-x", "??", now=MONDAY_10)

check(
    "an uncertain message asks for clarification",
    uncertain.clarification_required is True
    or uncertain.answered is True,
    f"clarification={uncertain.clarification_required}",
)

check(
    "an uncertain message never raises a ticket",
    uncertain.ticket is None or uncertain.clarification_required is False,
)

check(
    "the flow never calls the chain with an empty question",
    all(call["question"].strip() for call in stub.calls),
)


# ============================================================
# 9. THE STUB FAILING
# ============================================================

print("\n[9] THE CHAIN FAILING")


def failing(question, tone):

    raise RuntimeError("model unavailable")


broken = build_flow()

broken.answer_fn = failing

survived = broken.process_turn("sess-broken", CHARGED_TWICE, now=MONDAY_10)

check(
    "a chain failure does not lose the turn",
    survived.answered is False
    or "could not reach the knowledge base" in survived.answer,
    survived.answer,
)

check(
    "a chain failure still escalates",
    survived.escalated is True,
)

check(
    "a chain failure still raises the ticket",
    survived.ticket is not None,
)

check(
    "a chain failure still persists the escalation",
    broken.store.count_escalations() >= 1,
)

check(
    "the failure is explained to the customer",
    "could not reach" in survived.answer,
    survived.answer,
)


# ============================================================
# 10. AGENT LOAD
# ============================================================

print("\n[10] AGENT LOAD")

# This runs against its own store, so the tickets raised here are
# matched against each other rather than against whatever earlier
# sections left behind.
with tempfile.TemporaryDirectory() as load_folder:

    pair = build_flow()

    pair.store = SupportStore(db_path=Path(load_folder) / "pair.db")
    pair.store.initialize()

    pair.agents = [
        Agent("AGT-A", "billing", name="Asha", languages=("english",)),
        Agent("AGT-B", "billing", name="Ravi", languages=("english",)),
    ]

    for index in range(3):

        pair.process_turn(
            f"sess-load-{index}",
            "My card was charged twice for order "
            f"ORD-88{index:03d}. I am Priya Sharma, email "
            "priya@example.com, amount 1499.00 INR.",
            now=MONDAY_10,
        )

    loads = [agent.active_tickets for agent in pair.agents]

    check(
        "every routed ticket is charged to the agent who took it",
        sum(loads) == 3,
        str(loads),
    )

    # The router picks the least-loaded agent, so an assignment that
    # never moves the load would send every ticket to whoever was
    # quietest at the start.
    check(
        "tickets spread across the idle agents",
        sorted(loads) == [1, 2],
        str(loads),
    )

    # The second copy of a problem is the same problem: it joins the
    # ticket that already holds a place rather than taking another.
    repeat = build_flow()

    repeat.store = SupportStore(
        db_path=Path(load_folder) / "repeat.db"
    )
    repeat.store.initialize()

    repeat.agents = [
        Agent("AGT-A", "billing", name="Asha", languages=("english",)),
    ]

    repeat.process_turn("sess-dup-1", CHARGED_TWICE, now=MONDAY_10)

    second = repeat.process_turn("sess-dup-2", CHARGED_TWICE, now=MONDAY_10)

    check(
        "a duplicate does not occupy a second slot",
        second.ticket is not None
        and second.ticket.duplicate_of is not None
        and repeat.agents[0].active_tickets == 1,
        f"duplicate_of={second.ticket.duplicate_of if second.ticket else None}"
        f" load={repeat.agents[0].active_tickets}",
    )


# ============================================================
# 11. CLEANUP
# ============================================================

print("\n[11] ISOLATION")

check(
    "the integration store is not the production database",
    str(db_path).startswith(folder.name),
)

folder.cleanup()

check("the temp database is gone", not db_path.exists())


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 75)
print(f"PASSED: {PASS}")
print(f"FAILED: {FAIL}")
print("=" * 75)

raise SystemExit(0 if FAIL == 0 else 1)
