"""
TASK 8 TICKET LIFECYCLE REGRESSION SUITE

Every case here is one a code review named as a verified failure.
Each asserts the behaviour the fix is meant to produce, driven
through the real ``SupportFlow.process_turn`` with a stub answer
function -- no Gemini, no FAISS, no network.

Covers:
- an information request the customer answers completes the ticket
  it belongs to, instead of opening a second, differently-broken one
- two unrelated problems in one message become two differently
  categorised tickets
- an unrelated later problem does not inherit the first ticket's
  category
- a follow-up nudge on an open ticket does not raise a new ticket
"""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from business_calendar import BusinessCalendarConfig
from support_flow import SupportFlow, SupportFlowConfig
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
print("TASK 8 TICKET LIFECYCLE REGRESSION TESTS")
print("=" * 75)

folder = tempfile.TemporaryDirectory()

CALENDAR = BusinessCalendarConfig(timezone_name="UTC")

MONDAY_10 = utc(2026, 10, 5, 10, 0)


def build_flow(name):
    """A fresh flow on its own database, so cases cannot bleed."""

    return SupportFlow(
        config=SupportFlowConfig(
            escalation=EscalationConfig(calendar=CALENDAR),
            sla=SlaPolicy(calendar=CALENDAR),
            routing=RoutingConfig(calendar=CALENDAR),
        ),
        store=SupportStore(Path(folder.name) / f"{name}.db"),
        agents=[
            Agent(
                "AGT-BILL", "billing", name="Asha",
                languages=("english", "hindi"), active_tickets=0,
            ),
            Agent(
                "AGT-SEC", "security", name="Ravi",
                languages=("english",), active_tickets=0,
            ),
            Agent(
                "AGT-TECH", "tech", name="Dev",
                languages=("english",), active_tickets=0,
            ),
            Agent(
                "AGT-LOG", "logistics", name="Meera",
                languages=("english", "telugu"), active_tickets=0,
            ),
            Agent("AGT-OC", "on_call", name="On-call", on_call=True),
        ],
        answer_fn=lambda question, tone: "Refunds take 5-7 days.",
    )


def ticket_ids(flow):
    return [row["ticket_id"] for row in flow.store.list_tickets()]


# ============================================================
# 1. THE MISSING-INFORMATION LOOP
# ============================================================

print("\n[1] AN INFORMATION REQUEST THE CUSTOMER ANSWERS")

loop = build_flow("loop")

first = loop.process_turn(
    "sess-loop",
    "My card was charged twice on ORD-777 and this is frustrating.",
    now=MONDAY_10,
)

check(
    "the first turn raises a ticket",
    first.ticket is not None,
)

check(
    "the ticket is waiting for information",
    first.ticket is not None and first.ticket.status == "awaiting_info",
    first.ticket.status if first.ticket else "no ticket",
)

check(
    "an information request is produced",
    bool(first.information_request),
    first.information_request,
)

first_id = first.ticket.ticket_id if first.ticket else ""

second = loop.process_turn(
    "sess-loop",
    "I am Anil Rao, reach me at anil.rao@example.com, "
    "and the amount was 1499 INR.",
    now=MONDAY_10 + timedelta(minutes=5),
)

check(
    "the reply reaches the SAME ticket",
    second.ticket is not None
    and second.ticket.ticket_id == first_id,
    f"{second.ticket.ticket_id if second.ticket else None} "
    f"vs {first_id}",
)

check(
    "no second ticket is opened",
    len(ticket_ids(loop)) == 1,
    str(ticket_ids(loop)),
)

check(
    "the name is collected",
    second.ticket is not None
    and second.ticket.customer_name == "Anil Rao",
    str(second.ticket.customer_name if second.ticket else None),
)

check(
    "the contact is collected",
    second.ticket is not None
    and second.ticket.contact == "anil.rao@example.com",
    str(second.ticket.contact if second.ticket else None),
)

check(
    "the amount is collected",
    second.ticket is not None and bool(second.ticket.amount),
    str(second.ticket.amount if second.ticket else None),
)

check(
    "the ticket is no longer waiting",
    second.ticket is not None
    and second.ticket.status != "awaiting_info",
    second.ticket.status if second.ticket else "no ticket",
)

check(
    "the order id survived from the first turn",
    second.ticket is not None
    and second.ticket.order_id == "ORD-777",
    str(second.ticket.order_id if second.ticket else None),
)

check(
    "no information request remains",
    not second.information_request,
    second.information_request,
)


# ============================================================
# 2. TWO UNRELATED PROBLEMS IN ONE MESSAGE
# ============================================================

print("\n[2] TWO UNRELATED PROBLEMS IN ONE MESSAGE")

split = build_flow("split")

turn = split.process_turn(
    "sess-split",
    "My card was charged twice for ORD-777 and separately my "
    "account password reset email never arrives.",
    now=MONDAY_10,
)

check(
    "the message raises two tickets",
    len(turn.tickets) == 2,
    str([t.ticket_id for t in turn.tickets]),
)

categories = {t.category for t in turn.tickets}

check(
    "the two problems do not share one category",
    len(categories) > 1,
    str(sorted(categories)),
)

check(
    "only the charged-twice half is a payment ticket",
    sum(1 for t in turn.tickets if t.category == "payment") <= 1,
    str([(t.ticket_id, t.category) for t in turn.tickets]),
)


# ============================================================
# 3. A FOLLOW-UP NUDGE ON AN OPEN TICKET
# ============================================================

print("\n[3] A FOLLOW-UP NUDGE ON AN OPEN TICKET")

open_flow = build_flow("open")

opened = open_flow.process_turn(
    "sess-open",
    "My card was charged twice on ORD-888, I am Ravi Kumar, "
    "ravi@example.com, amount 2000 INR. This is frustrating.",
    now=MONDAY_10,
)

check(
    "the problem is ticketed and complete",
    opened.ticket is not None
    and opened.ticket.status != "awaiting_info",
    opened.ticket.status if opened.ticket else "no ticket",
)

opened_id = opened.ticket.ticket_id if opened.ticket else ""

nudge = open_flow.process_turn(
    "sess-open",
    "Still waiting, this is really frustrating.",
    now=MONDAY_10 + timedelta(minutes=6),
)

check(
    "the nudge does not open a second ticket",
    len(ticket_ids(open_flow)) == 1,
    str(ticket_ids(open_flow)),
)

check(
    "the nudge lands on the open ticket",
    nudge.ticket is not None and nudge.ticket.ticket_id == opened_id,
    f"{nudge.ticket.ticket_id if nudge.ticket else None} "
    f"vs {opened_id}",
)


# ============================================================
# 4. AN UNRELATED LATER PROBLEM STAYS ITS OWN PROBLEM
# ============================================================

print("\n[4] AN UNRELATED LATER PROBLEM")

later = build_flow("later")

pay = later.process_turn(
    "sess-later",
    "My card was charged twice on ORD-999, I am Sana Ali, "
    "sana@example.com, amount 750 INR. This is frustrating.",
    now=MONDAY_10,
)

check(
    "the first problem is a payment ticket",
    pay.ticket is not None and pay.ticket.category == "payment",
    pay.ticket.category if pay.ticket else "no ticket",
)

tech = later.process_turn(
    "sess-later",
    "Separately, the app keeps crashing with error E-4021 "
    "whenever I open the settings screen.",
    now=MONDAY_10 + timedelta(minutes=3),
)

check(
    "the second problem is a ticket of its own",
    tech.ticket is not None,
)

check(
    "it is not filed as a payment problem",
    tech.ticket is not None and tech.ticket.category != "payment",
    tech.ticket.category if tech.ticket else "no ticket",
)

check(
    "it is not a duplicate of the payment ticket",
    tech.ticket is not None and tech.ticket.duplicate_of is None,
    str(tech.ticket.duplicate_of if tech.ticket else None),
)


# ============================================================

print("\n" + "=" * 75)
print(f"PASSED: {PASS}")
print(f"FAILED: {FAIL}")
print("=" * 75)
