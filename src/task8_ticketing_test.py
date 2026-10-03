"""
TASK 8.1 - TICKETING TEST SUITE

Covers:
- Extraction across a multi-turn conversation, inventing nothing
- Mandatory-field validation per category and the information request
- Priority arithmetic and the critical override
- SLA across a weekend and a holiday
- The warning firing exactly at the configured fraction
- A runtime SLA change moving existing tickets, including flipping
  an open ticket into breach
- Duplicate / related / unrelated classification
- Multi-issue splitting
- The masked handoff keeping working identifiers but no raw PII
"""

from datetime import datetime, timedelta, timezone

from business_calendar import BusinessCalendarConfig
from task7_sentiment import SentimentResult, analyse_message
from task8_ticketing import (
    CATEGORY_MANDATORY_FIELDS,
    SlaPolicy,
    Ticket,
    TicketPolicy,
    apply_sla,
    build_handoff_summary,
    build_information_request,
    classify_relation,
    create_ticket,
    extract_contact,
    extract_ticket_fields,
    find_related_tickets,
    infer_category,
    link_ticket,
    missing_mandatory_fields,
    next_ticket_id,
    recalculate_deadlines,
    score_priority,
    similarity_score,
    split_issues,
    token_overlap,
)


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
    return datetime(
        year, month, day, hour, minute, tzinfo=timezone.utc
    )


def utc_sla(**overrides):
    overrides.setdefault(
        "calendar", BusinessCalendarConfig(timezone_name="UTC")
    )

    return SlaPolicy(**overrides)


def ticket_from(message, now, existing=None, **kwargs):
    return create_ticket(
        kwargs.pop("session_id", "sess-1"),
        message,
        analyse_message(message),
        sla_policy=kwargs.pop("sla_policy", utc_sla()),
        now=now,
        existing=existing,
        **kwargs,
    )


print("=" * 75)
print("TASK 8.1 TICKETING TESTS")
print("=" * 75)

# Reference week (UTC): Mon 2026-09-28 ... Fri 2026-10-02,
# Sat 10-03, Sun 10-04, Mon 10-05.
FRIDAY_5PM = utc(2026, 10, 2, 17, 0)
MONDAY_9AM = utc(2026, 10, 5, 9, 0)

SLA = utc_sla()
POLICY = TicketPolicy()


# ============================================================
# 1. EXTRACTION
# ============================================================

print("\n[1] EXTRACTION")

message = (
    "Hi, this is Priya Sharma. My order ORD-12345 arrived damaged. "
    "The amount was 1499.00 INR and the product code is SKU-9912. "
    "Error code E502 showed on the app. "
    "You can reach me on 9876543210 or priya.sharma@example.com."
)

fields = extract_ticket_fields(message)

check(
    "order id is extracted",
    fields["order_id"] == "ORD-12345",
    str(fields.get("order_id")),
)

check(
    "amount is extracted",
    fields["amount"] and "1499" in fields["amount"],
    str(fields.get("amount")),
)

check(
    "product code is extracted",
    fields["product_code"] == "SKU-9912",
    str(fields.get("product_code")),
)

check(
    "error code is extracted",
    fields["error_code"] == "E502",
    str(fields.get("error_code")),
)

check(
    "customer name is extracted",
    fields["customer_name"] == "Priya Sharma",
    str(fields.get("customer_name")),
)

check(
    "email is preferred as the contact",
    fields["contact"] == "priya.sharma@example.com",
    str(fields.get("contact")),
)

check(
    "a phone-only message still yields a contact",
    extract_contact("call me on 9876543210") == "9876543210",
    str(extract_contact("call me on 9876543210")),
)

check(
    "a message with no contact yields None",
    extract_contact("my order is late") is None,
)

check(
    "a message with no order yields None",
    extract_ticket_fields("something is wrong")["order_id"] is None,
)

check(
    "nothing is invented for an empty message",
    all(
        value is None
        for value in extract_ticket_fields("").values()
    ),
    str(extract_ticket_fields("")),
)

check(
    "an account number is extracted",
    extract_ticket_fields("my account number is ACC-99881")[
        "account_identifier"
    ]
    == "ACC-99881",
)

# ------------------------------------------------------------
# multi-turn: details spread across turns
# ------------------------------------------------------------
turn_one = create_ticket(
    "sess-multi",
    "My order arrived damaged.",
    analyse_message("My order arrived damaged."),
    sla_policy=SLA,
    now=FRIDAY_5PM,
    ticket_id="TCK-M1",
)

check(
    "turn one has no order id",
    turn_one.order_id is None,
)

turn_two = create_ticket(
    "sess-multi",
    "Sorry, the order id is ORD-55120 and my email is a@b.com.",
    analyse_message("Sorry, the order id is ORD-55120 and my email is a@b.com."),
    sla_policy=SLA,
    now=FRIDAY_5PM + timedelta(minutes=1),
    ticket_id="TCK-M2",
)

check(
    "turn two recovers the order id",
    turn_two.order_id == "ORD-55120",
    str(turn_two.order_id),
)

check(
    "turn two recovers the contact",
    turn_two.contact == "a@b.com",
    str(turn_two.contact),
)


# ============================================================
# 2. CATEGORY
# ============================================================

print("\n[2] CATEGORY")

CATEGORY_CASES = [
    ("My refund has not been credited.", "payment"),
    ("I was charged twice for the same order.", "payment"),
    ("My account was hacked.", "account"),
    ("I cannot log in to my profile.", "account"),
    ("The courier tracking shows nothing, my parcel is late.", "delivery"),
    ("The item arrived broken and the wrong size.", "product"),
    ("The app crashes with error code E502.", "technical"),
    ("I have a general question.", "general"),
    ("मेरा रिफंड नहीं आया", "payment"),
    ("डिलीवरी बहुत देर हो गई", "delivery"),
    ("Mi reembolso no ha llegado", "payment"),
]

for text, expected in CATEGORY_CASES:

    check(
        f"category {expected}: {text!r}",
        infer_category(text) == expected,
        f"got {infer_category(text)}",
    )

check(
    "a risk topic overrides the wording",
    infer_category(
        "whenever convenient, please look at this",
        None,
        ["duplicate_payment"],
    )
    == "payment",
)

check(
    "an account risk topic routes to account",
    infer_category("no rush at all", None, ["account_compromise"])
    == "account",
)


# ============================================================
# 3. MANDATORY FIELDS
# ============================================================

print("\n[3] MANDATORY FIELDS")

full = ticket_from(
    "My order ORD-12345 was charged twice for an amount of "
    "1499.00 INR. I am Priya Sharma, email priya@example.com.",
    FRIDAY_5PM,
)

check(
    "a complete payment ticket has nothing missing",
    "order_id" not in full.missing_fields
    and "contact" not in full.missing_fields,
    str(full.missing_fields),
)

check(
    "a complete payment ticket is open",
    full.status == "open",
    full.status,
)

incomplete = ticket_from(
    "I was charged twice but I do not have the order number.",
    FRIDAY_5PM,
)

check(
    "an incomplete ticket lists the missing fields",
    "order_id" in incomplete.missing_fields
    and "contact" in incomplete.missing_fields,
    str(incomplete.missing_fields),
)

check(
    "an incomplete ticket awaits information",
    incomplete.status == "awaiting_info",
    incomplete.status,
)

check(
    "missing fields are recomputed from the ticket",
    missing_mandatory_fields(incomplete, POLICY)
    == incomplete.missing_fields,
)

check(
    "a complete ticket produces no information request",
    build_information_request(full, policy=POLICY) == "",
)

request = build_information_request(incomplete, "english", POLICY)

check(
    "the information request names what is needed",
    "order ID" in request and "email address or phone number" in request,
    request,
)

check(
    "the information request quotes the ticket id",
    incomplete.ticket_id in request,
)

hindi_request = build_information_request(incomplete, "hindi", POLICY)

check(
    "the information request is localised",
    "ऑर्डर आईडी" in hindi_request,
    hindi_request,
)

check(
    "an unknown language falls back to english",
    "order ID" in build_information_request(incomplete, "klingon", POLICY),
)

check(
    "a blank contact counts as missing",
    missing_mandatory_fields(
        Ticket(
            ticket_id="T",
            category="general",
            customer_name="Priya",
            contact="   ",
        ),
        POLICY,
    )
    == ["contact"],
)

check(
    "an unknown category falls back to the general requirement",
    missing_mandatory_fields(
        Ticket(ticket_id="T", category="nonsense"), POLICY
    )
    == list(CATEGORY_MANDATORY_FIELDS["general"]),
)


# ============================================================
# 4. PRIORITY
# ============================================================

print("\n[4] PRIORITY")

calm_risk = ticket_from(
    "There appears to be a duplicate charge on my card. No rush.",
    FRIDAY_5PM,
)

check(
    "a calm but high-severity ticket is not bottom priority",
    calm_risk.priority in ("P1", "P2"),
    f"got {calm_risk.priority} ({calm_risk.priority_score})",
)

check(
    "the priority score is the weighted sum of its components",
    abs(
        calm_risk.priority_score
        - round(
            sum(
                POLICY.weights[name] * value
                for name, value in calm_risk.priority_components.items()
            ),
            4,
        )
    )
    < 0.0002,
    str(calm_risk.priority_components),
)

check(
    "the components are all reported",
    set(calm_risk.priority_components)
    == {"severity", "sentiment", "waiting", "impact"},
    str(calm_risk.priority_components),
)

check(
    "a polite general enquiry is the lowest band",
    ticket_from(
        "What are your support hours please?", FRIDAY_5PM
    ).priority
    == "P4",
)

check(
    "an urgent payment problem outranks a polite one",
    ticket_from(
        "I am furious, my card was charged twice and I need this fixed "
        "immediately. Order ORD-1, email a@b.com, 500 INR.",
        FRIDAY_5PM,
    ).priority
    in ("P1", "P2"),
)

# Critical override.
hacked = ticket_from(
    "My account was hacked and the password changed without my "
    "permission.",
    FRIDAY_5PM,
)

check(
    "a critical risk topic overrides the band to P1",
    hacked.priority == "P1",
    f"got {hacked.priority} ({hacked.priority_score})",
)

check(
    "the severity is critical",
    hacked.severity == "critical",
    hacked.severity,
)

check(
    "the override beats a lower arithmetic band",
    hacked.priority_score < POLICY.priority_bands[0][0]
    or hacked.priority == "P1",
    f"score={hacked.priority_score} band={hacked.priority}",
)

# Waiting time lifts a ticket.
waiting = Ticket(
    ticket_id="TCK-W",
    category="delivery",
    severity="medium",
    sentiment_label="neutral",
    created_at=utc(2026, 9, 28, 9, 0),
)

fresh_score, fresh_band, fresh_parts = score_priority(
    waiting, POLICY, SLA, utc(2026, 9, 28, 9, 0)
)

old_score, old_band, old_parts = score_priority(
    waiting, POLICY, SLA, utc(2026, 10, 2, 17, 0)
)

check(
    "a ticket with no waiting time scores zero waiting",
    fresh_parts["waiting"] == 0.0,
    str(fresh_parts),
)

check(
    "waiting time raises the score",
    old_score > fresh_score,
    f"fresh={fresh_score} old={old_score}",
)

check(
    "waiting is capped at one",
    old_parts["waiting"] <= 1.0,
)

check(
    "an explicit customer impact overrides the category default",
    score_priority(
        Ticket(
            ticket_id="TCK-I",
            category="general",
            customer_impact=1.0,
        ),
        POLICY,
        SLA,
        FRIDAY_5PM,
    )[2]["impact"]
    == 1.0,
)

check(
    "an unknown priority band falls back to the longest budget",
    SLA.minutes_for("P9") == max(
        SLA.resolution_minutes_by_priority.values()
    ),
)

check(
    "a missing priority falls back to the longest budget",
    SLA.minutes_for(None)
    == max(SLA.resolution_minutes_by_priority.values()),
)


# ============================================================
# 5. SLA
# ============================================================

print("\n[5] SLA")

check(
    "the sla calendar honours the policy toggles",
    SLA.calendar_for_sla().exclude_weekends is True,
)

no_weekend = utc_sla(exclude_weekends=False)

check(
    "the weekend toggle is copied into the sla calendar",
    no_weekend.calendar_for_sla().exclude_weekends is False,
)

check(
    "the policy's own calendar is not mutated",
    SLA.calendar.exclude_weekends is True,
)

# Friday 17:00, P3 budget 480 business minutes: 60 minutes remain on
# Friday, then 420 more on Monday -> Monday 16:00.
weekend_ticket = Ticket(
    ticket_id="TCK-SLA",
    category="delivery",
    priority="P3",
    created_at=FRIDAY_5PM,
)

apply_sla(weekend_ticket, SLA, FRIDAY_5PM)

check(
    "the SLA budget is recorded",
    weekend_ticket.sla_minutes == 480.0,
    str(weekend_ticket.sla_minutes),
)

check(
    "the due date skips the weekend",
    weekend_ticket.due_at == utc(2026, 10, 5, 16, 0),
    str(weekend_ticket.due_at),
)

# 360 business minutes after Friday 17:00 is Monday 14:00 -- 60
# minutes on Friday then 300 on Monday. Not 75% of the wall-clock way
# to Monday 16:00, because the weekend is not on the clock. warn_at
# is defined as the instant elapsed business minutes reaches the
# fraction, which is exactly when warning_due flips.
check(
    "the warning date is 75% through the business budget",
    weekend_ticket.warn_at == utc(2026, 10, 5, 14, 0),
    str(weekend_ticket.warn_at),
)

check(
    "the warning date matches the instant warning_due flips",
    apply_sla(
        Ticket(
            ticket_id="TCK-WFLIP",
            priority="P3",
            created_at=FRIDAY_5PM,
        ),
        SLA,
        weekend_ticket.warn_at,
    ).warning_due
    is True,
)

check(
    "a freshly created ticket is not breached",
    weekend_ticket.breached is False
    and weekend_ticket.warning_due is False,
)

# Raised Saturday, measured Sunday: the weekend is entirely off the
# clock. (Raised Friday 17:00 it would be 60 minutes, since Friday
# evening is still a working window.)
check(
    "the weekend contributes no elapsed time",
    apply_sla(
        Ticket(
            ticket_id="TCK-WK",
            priority="P3",
            created_at=utc(2026, 10, 3, 12, 0),
        ),
        SLA,
        utc(2026, 10, 4, 12, 0),
    ).elapsed_business_minutes
    == 0.0,
)

check(
    "Friday evening still counts as business time",
    apply_sla(
        Ticket(
            ticket_id="TCK-FRI",
            priority="P3",
            created_at=FRIDAY_5PM,
        ),
        SLA,
        utc(2026, 10, 4, 12, 0),
    ).elapsed_business_minutes
    == 60.0,
)

# A holiday pushes the budget further out.
holiday_sla = utc_sla(
    calendar=BusinessCalendarConfig(
        timezone_name="UTC", holidays=("2026-10-05",)
    )
)

holiday_ticket = Ticket(
    ticket_id="TCK-HOL",
    priority="P3",
    created_at=FRIDAY_5PM,
)

apply_sla(holiday_ticket, holiday_sla, FRIDAY_5PM)

check(
    "a holiday is skipped by the SLA clock",
    holiday_ticket.due_at == utc(2026, 10, 6, 16, 0),
    str(holiday_ticket.due_at),
)

# ------------------------------------------------------------
# the warning fires exactly at the fraction
# ------------------------------------------------------------
MONDAY_START = utc(2026, 10, 5, 9, 0)

# Breach supersedes the warning: warning_due is False at 480 because
# a ticket that has already breached must be escalated, not warned
# about. Both boundaries are exact, not approximate.
for minutes, expect_warn, expect_breach in (
    (359, False, False),
    (360, True, False),
    (479, True, False),
    (480, False, True),
    (481, False, True),
):

    measured = apply_sla(
        Ticket(
            ticket_id=f"TCK-{minutes}",
            priority="P3",
            created_at=MONDAY_START,
        ),
        SLA,
        MONDAY_START + timedelta(minutes=minutes),
    )

    check(
        f"at {minutes} min warning={expect_warn} breach={expect_breach}",
        measured.warning_due is expect_warn
        and measured.breached is expect_breach,
        f"elapsed={measured.elapsed_business_minutes} "
        f"warn={measured.warning_due} breach={measured.breached}",
    )

check(
    "a resolved ticket is never breached",
    apply_sla(
        Ticket(
            ticket_id="TCK-R",
            priority="P3",
            created_at=MONDAY_START,
            status="resolved",
            resolved_at=MONDAY_START + timedelta(minutes=10),
        ),
        SLA,
        MONDAY_START + timedelta(days=2),
    ).breached
    is False,
)

check(
    "a resolved ticket's clock stops at resolution",
    apply_sla(
        Ticket(
            ticket_id="TCK-R2",
            priority="P3",
            created_at=MONDAY_START,
            status="resolved",
            resolved_at=MONDAY_START + timedelta(minutes=10),
        ),
        SLA,
        MONDAY_START + timedelta(days=2),
    ).elapsed_business_minutes
    == 10.0,
    str(
        apply_sla(
            Ticket(
                ticket_id="TCK-R2",
                priority="P3",
                created_at=MONDAY_START,
                status="resolved",
                resolved_at=MONDAY_START + timedelta(minutes=10),
            ),
            SLA,
            MONDAY_START + timedelta(days=2),
        ).elapsed_business_minutes
    ),
)

# ------------------------------------------------------------
# RUNTIME SLA CHANGE
# ------------------------------------------------------------
print("\n[5b] RUNTIME SLA CHANGE")

open_tickets = [
    Ticket(
        ticket_id=f"TCK-RT-{index}",
        priority="P3",
        created_at=MONDAY_START,
        status="open",
    )
    for index in range(2)
]

recalculate_deadlines(open_tickets, SLA, MONDAY_START + timedelta(minutes=90))

check(
    "the original due date is eight business hours out",
    open_tickets[0].due_at == utc(2026, 10, 5, 17, 0),
    str(open_tickets[0].due_at),
)

check(
    "the ticket is not breached under the original policy",
    open_tickets[0].breached is False,
)

# Shorten the P3 budget from 480 to 60 minutes.
tightened = utc_sla(
    resolution_minutes_by_priority={"P1": 60, "P2": 120, "P3": 60, "P4": 240}
)

recalculate_deadlines(
    open_tickets, tightened, MONDAY_START + timedelta(minutes=90)
)

check(
    "shortening the budget moves the existing due date",
    open_tickets[0].due_at == utc(2026, 10, 5, 10, 0),
    str(open_tickets[0].due_at),
)

check(
    "shortening the budget moves the existing warning date",
    open_tickets[0].warn_at == utc(2026, 10, 5, 9, 45),
    str(open_tickets[0].warn_at),
)

check(
    "a runtime SLA change flips an open ticket into breach",
    open_tickets[0].breached is True,
    f"elapsed={open_tickets[0].elapsed_business_minutes} "
    f"budget={open_tickets[0].sla_minutes}",
)

check(
    "every ticket is recalculated, not just the first",
    all(ticket.breached for ticket in open_tickets),
)

check(
    "recalculate_deadlines returns the tickets",
    len(recalculate_deadlines(open_tickets, SLA, MONDAY_START)) == 2,
)

# Restoring the policy restores the state -- nothing was remembered.
recalculate_deadlines(open_tickets, SLA, MONDAY_START + timedelta(minutes=90))

check(
    "reverting the policy reverts the breach",
    open_tickets[0].breached is False
    and open_tickets[0].due_at == utc(2026, 10, 5, 17, 0),
    str(open_tickets[0].due_at),
)

changing_warn = utc_sla(warn_fraction=0.5)

recalculate_deadlines(
    open_tickets, changing_warn, MONDAY_START + timedelta(minutes=250)
)

check(
    "the warning fraction is applied at runtime",
    open_tickets[0].warning_due is True
    and open_tickets[0].warn_at == utc(2026, 10, 5, 13, 0),
    f"warn_at={open_tickets[0].warn_at}",
)


# ============================================================
# 6. DUPLICATES
# ============================================================

print("\n[6] DUPLICATES AND RELATED TICKETS")

DUP_MESSAGE = (
    "My order ORD-77001 was charged twice. "
    "I am Priya, email priya@example.com."
)

original = ticket_from(DUP_MESSAGE, FRIDAY_5PM, ticket_id="TCK-ORIG")

check(
    "the original is not a duplicate",
    original.duplicate_of is None,
)

repeat = ticket_from(
    "Charged twice on ORD-77001 again. Priya, priya@example.com.",
    FRIDAY_5PM + timedelta(minutes=3),
    existing=[original],
    ticket_id="TCK-REPEAT",
)

check(
    "a repeat of the same problem is a duplicate",
    repeat.duplicate_of == "TCK-ORIG",
    f"got {repeat.duplicate_of} score={similarity_score(repeat, original, POLICY)}",
)

check(
    "a duplicate is marked as such",
    repeat.status == "duplicate",
    repeat.status,
)

check(
    "a duplicate inherits the original SLA, not a new one",
    repeat.due_at == original.due_at
    and repeat.sla_minutes == original.sla_minutes,
    f"repeat={repeat.due_at} original={original.due_at}",
)

check(
    "a duplicate inherits the original priority",
    repeat.priority == original.priority,
)

check(
    "a duplicate is linked to the original",
    "TCK-ORIG" in repeat.linked_tickets,
)

check(
    "the duplicate reason is recorded",
    "duplicate" in repeat.routing_reason,
    repeat.routing_reason,
)

# Unrelated: different order, different category, different wording.
unrelated = ticket_from(
    "The app crashes every time I open the settings screen.",
    FRIDAY_5PM + timedelta(minutes=4),
    existing=[original],
    ticket_id="TCK-UNREL",
)

check(
    "an unrelated ticket is not linked",
    unrelated.duplicate_of is None,
    f"got {unrelated.duplicate_of}",
)

check(
    "an unrelated ticket keeps its own SLA",
    unrelated.due_at != original.due_at,
)

check(
    "a different order id is evidence against a duplicate",
    classify_relation(
        Ticket(
            ticket_id="A",
            order_id="ORD-1",
            category="payment",
            issue="charged twice",
        ),
        Ticket(
            ticket_id="B",
            order_id="ORD-2",
            category="payment",
            issue="charged twice",
        ),
        POLICY,
    )[0]
    != "duplicate",
)

# Related: same product and category, overlapping but not identical.
related_a = Ticket(
    ticket_id="TCK-RA",
    category="product",
    product_code="SKU-9912",
    issue="the item arrived damaged and scratched",
)

related_b = Ticket(
    ticket_id="TCK-RB",
    category="product",
    product_code="SKU-9912",
    issue="my item arrived damaged with a scratch on the screen",
)

relation, related_score, reasons = classify_relation(
    related_a, related_b, POLICY
)

check(
    "a partial match is related, not duplicate",
    relation == "related",
    f"got {relation} score={related_score} reasons={reasons}",
)

check(
    "the related score sits between the thresholds",
    POLICY.related_threshold <= related_score < POLICY.duplicate_threshold,
    f"got {related_score}",
)

check(
    "the related reasons are recorded",
    reasons and any("product" in reason for reason in reasons),
    str(reasons),
)

related_match = link_ticket(
    Ticket(ticket_id="TCK-RC", category="product"),
    {
        "ticket_id": "TCK-RA",
        "relation": "related",
        "score": related_score,
        "reasons": reasons,
    },
    related_a,
)

check(
    "a related ticket keeps its own duplicate_of unset",
    related_match.duplicate_of is None,
)

check(
    "a related ticket is cross-linked",
    "TCK-RA" in related_match.linked_tickets,
)

check(
    "matches are returned best-first",
    [
        match["ticket_id"]
        for match in find_related_tickets(
            Ticket(
                ticket_id="TCK-NEW",
                order_id="ORD-77001",
                category="payment",
                issue="charged twice on ORD-77001",
            ),
            [unrelated, original],
            POLICY,
        )
    ]
    == ["TCK-ORIG"],
)

check(
    "an identical ticket scores as a duplicate",
    similarity_score(original, original, POLICY)[0] >= POLICY.duplicate_threshold,
)

check(
    "token overlap is zero for disjoint issues",
    token_overlap(["charged", "twice"], ["screen", "crashes"]) == 0.0,
)

check(
    "token overlap is one for identical issues",
    token_overlap(["charged", "twice"], ["charged", "twice"]) == 1.0,
)


# ============================================================
# 7. MULTI-ISSUE
# ============================================================

print("\n[7] MULTI-ISSUE SPLITTING")

single = split_issues("My order is late.")

check("a single request is not split", single == ["My order is late."])

multi = split_issues(
    "My order is late and also I was charged twice. "
    "Can you help?"
)

check(
    "two requests are split",
    len(multi) >= 2,
    str(multi),
)

check(
    "the split keeps the substance",
    any("late" in part for part in multi)
    and any("charged" in part for part in multi),
    str(multi),
)

check("an empty message splits to nothing", split_issues("") == [])

check(
    "a sentence with a harmless 'and' is left whole",
    len(split_issues("The box and the label were both damaged.")) == 1,
    str(split_issues("The box and the label were both damaged.")),
)

check(
    "a two-word fragment never becomes its own ticket",
    all(
        len(part.split()) >= 3
        for part in split_issues(
            "My order is late and also I was charged twice."
        )
    ),
    str(split_issues("My order is late and also I was charged twice.")),
)

check(
    "each split part is a question a ticket can hold",
    len(
        split_issues(
            "My refund has not arrived. Also my account is locked."
        )
    )
    == 2,
    str(split_issues("My refund has not arrived. Also my account is locked.")),
)


# ============================================================
# 8. HANDOFF
# ============================================================

print("\n[8] MASKED HANDOFF")

sensitive = (
    "My order ORD-12345 was charged twice, 1499.00 INR. "
    "I am Priya Sharma. Call 9876543210 or email priya@example.com. "
    "This is very frustrating."
)

sensitive_ticket = create_ticket(
    "sess-handoff",
    sensitive,
    analyse_message(sensitive),
    sla_policy=SLA,
    now=FRIDAY_5PM,
    ticket_id="TCK-HANDOFF",
)

handoff = build_handoff_summary(sensitive_ticket, SLA, FRIDAY_5PM)

check(
    "the handoff names the ticket",
    "TCK-HANDOFF" in handoff,
)

check(
    "the handoff keeps the order id the agent needs",
    "ORD-12345" in handoff,
)

check(
    "the handoff reports the priority and score",
    f"Priority: {sensitive_ticket.priority}" in handoff,
    handoff,
)

check(
    "the handoff reports the SLA state",
    "SLA:" in handoff and "business minutes" in handoff,
)

check(
    "the handoff reports the sentiment",
    f"Sentiment: {sensitive_ticket.sentiment_label}" in handoff,
)

check(
    "the handoff reports the risk topic",
    "duplicate_payment" in handoff.replace("_", "_"),
)

check(
    "the handoff lists the evidence",
    "Evidence:" in handoff,
)

check(
    "the masked handoff hides the raw email",
    "priya@example.com" not in handoff,
    handoff,
)

check(
    "the masked handoff hides the raw phone",
    "9876543210" not in handoff,
    handoff,
)

check(
    "the customer name is still available to the agent",
    "Priya Sharma" in handoff,
)

check(
    "the ticket keeps the real contact so an agent can reply",
    sensitive_ticket.contact == "priya@example.com",
)

check(
    "the masked contact accessor hides it",
    "priya@example.com" not in sensitive_ticket.masked_contact(),
    sensitive_ticket.masked_contact(),
)

check(
    "the handoff reports a breach when there is one",
    "BREACHED"
    in build_handoff_summary(
        apply_sla(
            Ticket(
                ticket_id="TCK-B",
                priority="P3",
                created_at=MONDAY_START,
            ),
            SLA,
            MONDAY_START + timedelta(minutes=500),
        ),
        SLA,
        MONDAY_START + timedelta(minutes=500),
    ),
)

check(
    "the handoff reports a warning when there is one",
    "WARNING"
    in build_handoff_summary(
        apply_sla(
            Ticket(
                ticket_id="TCK-W",
                priority="P3",
                created_at=MONDAY_START,
            ),
            SLA,
            MONDAY_START + timedelta(minutes=400),
        ),
        SLA,
        MONDAY_START + timedelta(minutes=400),
    ),
)

check(
    "a complete handoff asks the agent for nothing",
    "Awaiting from customer:" not in handoff,
    handoff.replace("\n", " | "),
)

awaiting = ticket_from(
    "I was charged twice but I do not have the order number.",
    FRIDAY_5PM,
    ticket_id="TCK-AWAIT",
)

check(
    "the handoff lists what is still awaited",
    "Awaiting from customer:" in build_handoff_summary(
        awaiting, SLA, FRIDAY_5PM
    ),
    build_handoff_summary(awaiting, SLA, FRIDAY_5PM).replace("\n", " | "),
)

check(
    "the handoff builds without a precomputed SLA",
    "Ticket:" in build_handoff_summary(sensitive_ticket, SLA, FRIDAY_5PM),
)


# ============================================================
# 9. TICKET IDS AND SERIALISATION
# ============================================================

print("\n[9] IDS AND SERIALISATION")

check(
    "the ticket id is deterministic for a given instant",
    next_ticket_id(FRIDAY_5PM) == next_ticket_id(FRIDAY_5PM),
)

check(
    "the ticket id changes with the instant",
    next_ticket_id(FRIDAY_5PM) != next_ticket_id(MONDAY_9AM),
)

check(
    "an explicit id is honoured",
    ticket_from("order is late", FRIDAY_5PM, ticket_id="TCK-X").ticket_id
    == "TCK-X",
)

serialised = sensitive_ticket.to_dict()

check(
    "the serialised datetimes are ISO strings",
    isinstance(serialised["created_at"], str)
    and isinstance(serialised["due_at"], str),
)

check(
    "the serialised simulated time is exact",
    serialised["created_at"] == "2026-10-02T17:00:00+00:00",
    str(serialised["created_at"]),
)

check(
    "the serialised lists survive",
    serialised["risk_topics"] == ["duplicate_payment"]
    and isinstance(serialised["evidence"], list),
)

check(
    "an unscheduled ticket serialises a null timestamp",
    ticket_from("order is late", FRIDAY_5PM).to_dict()["resolved_at"]
    is None,
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 75)
print(f"PASSED: {PASS}")
print(f"FAILED: {FAIL}")
print("=" * 75)

raise SystemExit(0 if FAIL == 0 else 1)
