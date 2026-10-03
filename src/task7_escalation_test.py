"""
TASK 7.2 - ESCALATION ENGINE TEST SUITE

Covers:
- ESC-01 ... ESC-10, one condition apiece
- Calm high-risk phrasing escalating in all four languages
- After-hours routing: urgent -> on-call, non-urgent -> scheduled
- The unresolved clock at 14 / 15 / 16 minutes under a simulated time
- A resolved ticket never re-escalating
- Every record carrying reason + activated condition + masked summary
- PII never reaching the audit trail or the handoff
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from business_calendar import BusinessCalendarConfig, add_business_minutes
from task6_multilingual import ConversationMessage, ConversationSession
from task7_escalation import (
    CONDITION_CATALOG,
    EscalationConfig,
    EscalationEngine,
    build_conversation_summary,
    last_user_message,
    mask_summary,
    unresolved_minutes,
)
from task7_sentiment import (
    SentimentConfig,
    analyse_message,
    combine_with_history,
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


def session_with(
    texts,
    start=utc(2026, 10, 2, 10, 0),
    language="english",
    session_id="sess-test",
):
    """
    Build a session whose user turns are spaced one minute apart.
    """

    session = ConversationSession(
        session_id=session_id,
        created_at=start,
        last_activity=start,
    )

    moment = start

    for text in texts:

        session.messages.append(
            ConversationMessage(
                role="user",
                content=text,
                timestamp=moment,
                language=language,
            )
        )

        session.messages.append(
            ConversationMessage(
                role="assistant",
                content="Thanks for reaching out.",
                timestamp=moment,
                language=language,
            )
        )

        moment = moment + timedelta(minutes=1)

    session.last_activity = moment

    return session


def engine_for(now_config=None, **overrides):
    """
    An engine on a UTC calendar, so every assertion below reads in
    plain UTC rather than IST.
    """

    config = EscalationConfig(
        calendar=BusinessCalendarConfig(timezone_name="UTC"),
        **overrides,
    )

    return EscalationEngine(config)


def sentiment_of(text, language=None):
    return analyse_message(text, language)


@dataclass
class FakeTicket:
    """
    Stand-in for task8_ticketing.Ticket, read duck-typed by the
    engine so the two modules stay independent.
    """

    ticket_id: str = "TCK-1"
    priority: str = "P3"
    breached: bool = False
    warning_due: bool = False
    warn_fraction: float = 0.75


print("=" * 75)
print("TASK 7.2 ESCALATION ENGINE TESTS")
print("=" * 75)

# Reference week (UTC): Mon 2026-09-28 ... Fri 2026-10-02,
# Sat 10-03, Sun 10-04, Mon 10-05.
FRIDAY_10AM = utc(2026, 10, 2, 10, 0)
FRIDAY_8PM = utc(2026, 10, 2, 20, 0)


# ============================================================
# 1. EACH CONDITION
# ============================================================

print("\n[1] ESC-01 ... ESC-10")

# ------------------------------------------------------------
# ESC-01 -- repeated negative contact
# ------------------------------------------------------------
engine = engine_for()

first = sentiment_of("My order is late and this is bad.")
current = sentiment_of("The delivery is still not here.")

blended = combine_with_history(
    current, [first], engine.config.sentiment_config
)

result = engine.evaluate(
    "sess-01", "The delivery is still not here.", blended, None, FRIDAY_10AM
)

check(
    "ESC-01 repeated negative escalates",
    "ESC-01" in result.condition_ids,
    f"got {result.condition_ids}",
)

check(
    "ESC-01 goes to the priority queue",
    result.records[0].queue == "priority_support",
    f"got {result.records[0].queue}",
)

check(
    "a single negative message is not ESC-01",
    "ESC-01"
    not in engine.evaluate(
        "sess-01b", "bad", current, None, FRIDAY_10AM
    ).condition_ids,
)

# ------------------------------------------------------------
# ESC-02 -- account compromise
# ------------------------------------------------------------
for label, text, expected_severity in (
    (
        "unrecognised login",
        "Could you kindly help me understand an unrecognised login on my account?",
        "critical",
    ),
    ("fraudulent charge", "There is a fraudulent charge on my card.", "high"),
    ("data leak", "I think there has been a data leak of my details.", "critical"),
):

    result = engine.evaluate(
        f"sess-02-{label}",
        text,
        sentiment_of(text),
        None,
        FRIDAY_10AM,
    )

    check(
        f"ESC-02 {label} escalates in business hours",
        "ESC-02" in result.condition_ids,
        f"got {result.condition_ids}",
    )

    check(
        f"ESC-02 {label} severity is {expected_severity}",
        result.severity == expected_severity,
        f"got {result.severity}",
    )

check(
    "ESC-02 routes to on-call after hours",
    engine.evaluate(
        "sess-02-night",
        "my account was hacked",
        sentiment_of("my account was hacked"),
        None,
        FRIDAY_8PM,
    ).records[0].queue
    == "on_call",
)

# ------------------------------------------------------------
# ESC-03 -- duplicate payment
# ------------------------------------------------------------
text = "I have been charged twice for the same order."

result = engine.evaluate(
    "sess-03", text, sentiment_of(text), None, FRIDAY_10AM
)

check(
    "ESC-03 duplicate payment escalates",
    "ESC-03" in result.condition_ids,
    f"got {result.condition_ids}",
)

check(
    "ESC-03 routes to priority in business hours",
    result.records[0].queue == "priority_support",
    f"got {result.records[0].queue}",
)

# ------------------------------------------------------------
# ESC-04 -- legal threat
# ------------------------------------------------------------
text = "I will speak to my lawyer about this."

result = engine.evaluate(
    "sess-04", text, sentiment_of(text), None, FRIDAY_10AM
)

check(
    "ESC-04 legal threat escalates",
    "ESC-04" in result.condition_ids,
    f"got {result.condition_ids}",
)

check(
    "ESC-04 goes to on-call even during business hours",
    result.records[0].queue == "on_call",
    f"got {result.records[0].queue}",
)

# ------------------------------------------------------------
# ESC-05 -- after-hours urgent
# ------------------------------------------------------------
text = "This is urgent, I need it immediately."

result = engine.evaluate(
    "sess-05", text, sentiment_of(text), None, FRIDAY_8PM
)

check(
    "ESC-05 after-hours urgent escalates",
    "ESC-05" in result.condition_ids,
    f"got {result.condition_ids}",
)

check(
    "ESC-05 goes to the on-call queue",
    result.records[0].queue == "on_call",
    f"got {result.records[0].queue}",
)

check(
    "urgent during business hours is not ESC-05",
    "ESC-05"
    not in engine.evaluate(
        "sess-05-day", text, sentiment_of(text), None, FRIDAY_10AM
    ).condition_ids,
)

# ------------------------------------------------------------
# ESC-06 -- after-hours non-urgent negative
# ------------------------------------------------------------
text = "My order is late and this is bad."

result = engine.evaluate(
    "sess-06", text, sentiment_of(text), None, FRIDAY_8PM
)

check(
    "ESC-06 after-hours complaint escalates",
    "ESC-06" in result.condition_ids,
    f"got {result.condition_ids}",
)

check(
    "ESC-06 schedules for the next business start",
    result.records[0].scheduled_for == utc(2026, 10, 5, 9, 0),
    f"got {result.records[0].scheduled_for}",
)

check(
    "ESC-06 uses the scheduled queue",
    result.records[0].queue == "scheduled_next_working_day",
    f"got {result.records[0].queue}",
)

# ------------------------------------------------------------
# ESC-07 -- unresolved beyond the limit (see section 3 too)
# ------------------------------------------------------------
session = session_with(["My order is late and this is bad."])

neg = sentiment_of("My order is late and this is bad.")

result = engine.evaluate(
    "sess-07",
    "My order is late and this is bad.",
    neg,
    session,
    utc(2026, 10, 2, 10, 30),
)

check(
    "ESC-07 long-unresolved escalates",
    "ESC-07" in result.condition_ids,
    f"got {result.condition_ids}",
)

check(
    "ESC-07 requests an automatic ticket",
    any("ticket" in action for action in result.actions),
    f"got {result.actions}",
)

# ------------------------------------------------------------
# ESC-08 -- SLA breached
# ------------------------------------------------------------
result = engine.evaluate(
    "sess-08",
    "This is still not resolved.",
    sentiment_of("This is still not resolved."),
    None,
    FRIDAY_10AM,
    ticket=FakeTicket(priority="P3", breached=True),
)

check(
    "ESC-08 breached SLA escalates",
    "ESC-08" in result.condition_ids,
    f"got {result.condition_ids}",
)

check(
    "ESC-08 routes to on-call",
    result.records[0].queue == "on_call",
    f"got {result.records[0].queue}",
)

check(
    "ESC-08 bumps the priority one band",
    result.records[0].priority == "P2",
    f"got {result.records[0].priority}",
)

check(
    "ESC-08 severity is critical",
    result.severity == "critical",
    f"got {result.severity}",
)

check(
    "ESC-08 names the ticket it is about",
    result.records[0].ticket_id == "TCK-1",
    f"got {result.records[0].ticket_id}",
)

check(
    "the ticket id survives into the stored form",
    result.records[0].to_dict().get("ticket_id") == "TCK-1",
    str(result.records[0].to_dict().get("ticket_id")),
)

check(
    "a condition raised without a ticket names none",
    engine.evaluate(
        "sess-08b",
        "The delivery is still not here and this is bad.",
        sentiment_of("The delivery is still not here and this is bad."),
        None,
        FRIDAY_8PM,
    ).records[0].to_dict().get("ticket_id")
    is None,
)

# ------------------------------------------------------------
# ESC-09 -- SLA warning
# ------------------------------------------------------------
result = engine.evaluate(
    "sess-09",
    "Any update on my order?",
    sentiment_of("Any update on my order?"),
    None,
    FRIDAY_10AM,
    ticket=FakeTicket(priority="P3", warning_due=True),
)

check(
    "ESC-09 warns",
    "ESC-09" in result.condition_ids,
    f"got {result.condition_ids}",
)

check(
    "ESC-09 alone does not escalate",
    result.escalate is False,
    f"got {result.escalate}",
)

check(
    "ESC-09 is a warning record",
    result.records[0].is_warning is True,
)

check(
    "ESC-09 is reported as a warning",
    len(result.warnings) == 1,
)

check(
    "a warning still produces the audit record",
    len(result.records) == 1,
)

check(
    "ESC-09 names the ticket it is about",
    result.records[0].ticket_id == "TCK-1",
    f"got {result.records[0].ticket_id}",
)

# ------------------------------------------------------------
# ESC-10 -- explicit human request
# ------------------------------------------------------------
text = "I want to talk to a human"

result = engine.evaluate(
    "sess-10", text, sentiment_of(text), None, FRIDAY_10AM
)

check(
    "ESC-10 human request escalates",
    "ESC-10" in result.condition_ids,
    f"got {result.condition_ids}",
)

check(
    "ESC-10 goes to the priority queue",
    result.records[0].queue == "priority_support",
    f"got {result.records[0].queue}",
)

check(
    "every condition in the catalog has a unique id",
    len({value[0] for value in CONDITION_CATALOG.values()})
    == 10,
    str(sorted({value[0] for value in CONDITION_CATALOG.values()})),
)


# ============================================================
# 2. CALM HIGH RISK IN FOUR LANGUAGES
# ============================================================

print("\n[2] CALM HIGH-RISK PHRASING ESCALATES")

CALM = [
    ("english", "Could you kindly look into a duplicate charge on my card? No rush."),
    ("hindi", "कृपया देख लीजिए, डुप्लीकेट पेमेंट हो गया है"),
    ("telugu", "దయచేసి చూడండి, రెండు సార్లు కట్ అయింది"),
    ("spanish", "Disculpe, creo que hay un pago duplicado. Sin prisa."),
]

for language, text in CALM:

    sentiment = sentiment_of(text, language)

    result = engine.evaluate(
        f"sess-calm-{language}", text, sentiment, None, FRIDAY_10AM
    )

    check(
        f"calm phrasing is not negative ({language})",
        sentiment.label in ("positive", "neutral"),
        f"got {sentiment.label}",
    )

    check(
        f"calm phrasing still escalates ({language})",
        result.escalate is True,
        f"got conditions={result.condition_ids}",
    )

    check(
        f"calm phrasing escalates as ESC-03 ({language})",
        "ESC-03" in result.condition_ids,
        f"got {result.condition_ids}",
    )

    check(
        f"calm phrasing keeps high severity ({language})",
        result.severity == "high",
        f"got {result.severity}",
    )


# ============================================================
# 3. THE UNRESOLVED CLOCK
# ============================================================

print("\n[3] UNRESOLVED CLOCK AT 14 / 15 / 16 MINUTES")

config = EscalationConfig(
    calendar=BusinessCalendarConfig(timezone_name="UTC")
)

negative = sentiment_of("My order is late and this is bad.")

session = session_with(["My order is late and this is bad."])

raised_at = session.messages[0].timestamp

check(
    "the session's last user message is found",
    last_user_message(session.messages) is session.messages[0],
)

for minutes, expected_escalation in ((14, False), (15, True), (16, True)):

    moment = raised_at + timedelta(minutes=minutes)

    measured = unresolved_minutes(
        session.messages, negative, config, moment
    )

    check(
        f"unresolved at {minutes} min reads {minutes}",
        measured == float(minutes),
        f"got {measured}",
    )

    result = engine_for().evaluate(
        f"sess-clock-{minutes}",
        "My order is late and this is bad.",
        negative,
        session,
        moment,
    )

    check(
        f"ESC-07 at {minutes} min is {expected_escalation}",
        ("ESC-07" in result.condition_ids) is expected_escalation,
        f"got {result.condition_ids}",
    )

check(
    "before the customer speaks the clock is zero",
    unresolved_minutes([], negative, config, FRIDAY_10AM) == 0.0,
)

check(
    "a resolved ticket never escalates on time",
    unresolved_minutes(
        session.messages, negative, config,
        raised_at + timedelta(hours=3), status="resolved",
    )
    == 0.0,
)

check(
    "a closed ticket never escalates on time",
    unresolved_minutes(
        session.messages, negative, config,
        raised_at + timedelta(hours=3), status="closed",
    )
    == 0.0,
)

closed = engine_for().evaluate(
    "sess-closed",
    "My order is late and this is bad.",
    negative,
    session,
    raised_at + timedelta(hours=3),
    status="resolved",
)

check(
    "a resolved session raises no ESC-07",
    "ESC-07" not in closed.condition_ids,
    f"got {closed.condition_ids}",
)

positive = sentiment_of("Thank you, everything is perfect now.")

check(
    "a satisfied customer's clock does not run",
    unresolved_minutes(
        session.messages, positive, config,
        raised_at + timedelta(hours=3),
    )
    == 0.0,
)

# Business clock: Friday 17:30 -> Friday 19:00 is 90 wall minutes but
# only 30 business minutes.
business_config = EscalationConfig(
    calendar=BusinessCalendarConfig(timezone_name="UTC"),
    unresolved_clock="business",
)

wall_config = EscalationConfig(
    calendar=BusinessCalendarConfig(timezone_name="UTC"),
    unresolved_clock="wall",
)

late_session = session_with(
    ["This is bad and I am still waiting."],
    start=utc(2026, 10, 2, 17, 30),
)

late_sentiment = sentiment_of("This is bad and I am still waiting.")

check(
    "wall clock reads 90 minutes",
    unresolved_minutes(
        late_session.messages, late_sentiment, wall_config,
        utc(2026, 10, 2, 19, 0),
    )
    == 90.0,
)

check(
    "business clock reads 30 minutes",
    unresolved_minutes(
        late_session.messages, late_sentiment, business_config,
        utc(2026, 10, 2, 19, 0),
    )
    == 30.0,
    str(
        unresolved_minutes(
            late_session.messages, late_sentiment, business_config,
            utc(2026, 10, 2, 19, 0),
        )
    ),
)


# ============================================================
# 4. AUDIT TRAIL AND MASKING
# ============================================================

print("\n[4] AUDIT TRAIL AND MASKING")

text = (
    "My card 4111111111111111 was charged twice. "
    "Email me at john.doe@example.com or call +91 9876543210."
)

sentiment = sentiment_of(text)

session = session_with([text])

result = engine_for().evaluate(
    "sess-pii", text, sentiment, session, FRIDAY_10AM
)

check(
    "the PII message escalates",
    result.escalate is True,
    f"got {result.condition_ids}",
)

check(
    "every record carries a reason",
    all(record.reason for record in result.records),
)

check(
    "every record carries the activated condition",
    all(
        record.condition_id and record.condition
        for record in result.records
    ),
)

check(
    "every record carries a conversation summary",
    all(record.conversation_summary for record in result.records),
)

check(
    "every record carries a masked summary",
    all(record.masked_summary for record in result.records),
)

check(
    "every record names its queue",
    all(record.queue for record in result.records),
)

check(
    "every record is timestamped at the simulated now",
    all(record.escalated_at == FRIDAY_10AM for record in result.records),
)

for record in result.records:

    check(
        f"card number is masked [{record.condition_id}]",
        "4111111111111111" not in record.masked_summary
        and "4111111111111111" not in record.message_excerpt,
        record.masked_summary,
    )

    check(
        f"email is masked [{record.condition_id}]",
        "john.doe@example.com" not in record.masked_summary
        and "john.doe@example.com" not in record.message_excerpt,
        record.masked_summary,
    )

    check(
        f"phone is masked [{record.condition_id}]",
        "9876543210" not in record.masked_summary
        and "9876543210" not in record.message_excerpt,
        record.masked_summary,
    )

check(
    "the masked summary keeps the order id",
    "ORD-12345"
    in mask_summary("My order ORD-12345 is late. Email a@b.com"),
)

summary = build_conversation_summary(session.messages, engine_for().config)

check(
    "the summary carries both roles",
    summary.startswith("user:") and "assistant:" in summary,
    summary,
)

check(
    "an empty conversation summarises safely",
    build_conversation_summary([]) == "No conversation history.",
)

check(
    "the summary is truncated to the configured length",
    len(
        build_conversation_summary(
            session_with(["x" * 2000]).messages,
            EscalationConfig(summary_max_chars=50),
        )
    )
    <= 53,
)

check(
    "the record serialises to JSON-friendly values",
    isinstance(result.records[0].to_dict()["escalated_at"], str),
)


# ============================================================
# 5. DETERMINISM AND RUNTIME RECONFIGURATION
# ============================================================

print("\n[5] DETERMINISM AND RUNTIME RECONFIGURATION")

same = engine_for().evaluate(
    "sess-pii", text, sentiment, session, FRIDAY_10AM
)

check(
    "the same input produces the same escalation id",
    [record.escalation_id for record in same.records]
    == [record.escalation_id for record in result.records],
)

check(
    "the same input produces the same conditions",
    same.condition_ids == result.condition_ids,
)

# The unresolved limit is a runtime knob, not a constant.
lenient = engine_for(unresolved_minutes=120)

session_short = session_with(["This is bad and I am still waiting."])

result_lenient = lenient.evaluate(
    "sess-limit",
    "This is bad and I am still waiting.",
    sentiment_of("This is bad and I am still waiting."),
    session_short,
    utc(2026, 10, 2, 10, 30),
)

check(
    "raising the limit at runtime suppresses ESC-07",
    "ESC-07" not in result_lenient.condition_ids,
    f"got {result_lenient.condition_ids}",
)

strict = engine_for(unresolved_minutes=5)

result_strict = strict.evaluate(
    "sess-limit",
    "This is bad and I am still waiting.",
    sentiment_of("This is bad and I am still waiting."),
    session_short,
    utc(2026, 10, 2, 10, 30),
)

check(
    "lowering the limit at runtime activates ESC-07",
    "ESC-07" in result_strict.condition_ids,
    f"got {result_strict.condition_ids}",
)

# The calendar is a runtime knob too.
holiday_engine = EscalationEngine(
    EscalationConfig(
        calendar=BusinessCalendarConfig(
            timezone_name="UTC", holidays=("2026-10-05",)
        )
    )
)

scheduled = holiday_engine.evaluate(
    "sess-holiday",
    "My order is late and this is bad.",
    sentiment_of("My order is late and this is bad."),
    None,
    FRIDAY_8PM,
)

check(
    "a configured holiday pushes the schedule past Monday",
    scheduled.records[0].scheduled_for == utc(2026, 10, 6, 9, 0),
    f"got {scheduled.records[0].scheduled_for}",
)

check(
    "the business calendar is the single source of SLA time",
    add_business_minutes(
        FRIDAY_10AM,
        60,
        BusinessCalendarConfig(timezone_name="UTC"),
    )
    == utc(2026, 10, 2, 11, 0),
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 75)
print(f"PASSED: {PASS}")
print(f"FAILED: {FAIL}")
print("=" * 75)

raise SystemExit(0 if FAIL == 0 else 1)
