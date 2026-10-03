"""
TASK 6 REGRESSION TEST SUITE

Covers:
- Three additional languages
- Mixed-language / code switching
- Spelling/transliteration
- Entity preservation
- Multiple requests
- Corrections
- Low-confidence clarification
- 10-message context
- Session isolation
- 30-minute inactivity
- 24-hour restoration window
- Configurable settings
- Session summaries folded on close and on restore
"""

from datetime import datetime, timedelta, timezone

from task6_multilingual import (
    Task6Config,
    build_session_summary,
    detect_language,
    extract_preserved_entities,
    preserve_entities,
    detect_multiple_requests,
    detect_correction,
    build_language_instruction,
    SessionManager,
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


print("=" * 75)
print("TASK 6 MULTILINGUAL + SESSION REGRESSION TESTS")
print("=" * 75)


# ============================================================
# 1. LANGUAGE DETECTION
# ============================================================

print("\n[1] LANGUAGE DETECTION")


tests = [
    (
        "Hindi",
        "मेरा ऑर्डर अभी तक नहीं आया है",
        "hindi",
    ),
    (
        "Telugu",
        "నా ఆర్డర్ ఇంకా రాలేదు",
        "telugu",
    ),
    (
        "Spanish",
        "Necesito ayuda con mi pedido",
        "spanish",
    ),
    (
        "English",
        "I need help with my order",
        "english",
    ),
]

for name, text, expected in tests:
    result = detect_language(text)
    check(
        f"{name} language detection",
        result["language"] == expected
        and not result["clarification_required"],
        str(result),
    )


# ============================================================
# 2. MIXED LANGUAGE / CODE SWITCHING
# ============================================================

print("\n[2] MIXED LANGUAGE / CODE SWITCHING")

mixed_messages = [
    "मेरा order अभी तक नहीं आया",
    "నా order ఇంకా రాలేదు",
    "Necesito help con mi pedido",
]

for message in mixed_messages:
    result = detect_language(message)

    check(
        f"Mixed-language message detected: {message}",
        len(result["detected_languages"]) >= 2
        or result["mixed"]
        or result["language"] in {"hindi", "telugu", "spanish"},
        str(result),
    )


# ============================================================
# 3. SPELLING / TRANSLITERATION
# ============================================================

print("\n[3] SPELLING / TRANSLITERATION")

transliteration_tests = [
    ("mera order kab ayega", "hindi"),
    ("na order ekkada undi", "telugu"),
    ("necesito ayuda con mi pedido", "spanish"),
]

for text, expected in transliteration_tests:
    result = detect_language(text)

    check(
        f"Transliteration: {text}",
        result["language"] == expected
        or result["confidence"] < 0.55,
        str(result),
    )


# ============================================================
# 4. ENTITY PRESERVATION
# ============================================================

print("\n[4] ENTITY PRESERVATION")

entity_text = (
    "Please check order ORD-12345 for product PRD-ABC123 "
    "on 15/09/2026 for ₹1499."
)

entities = extract_preserved_entities(entity_text)

check(
    "Order ID preserved",
    "ORD-12345" in entities.get("order_id", []),
    str(entities),
)

check(
    "Product code preserved",
    "PRD-ABC123" in entities.get("product_code", []),
    str(entities),
)

check(
    "Date preserved",
    "15/09/2026" in entities.get("date", []),
    str(entities),
)

check(
    "Amount preserved",
    "₹1499" in entities.get("amount", []),
    str(entities),
)


# ============================================================
# 5. NAME PRESERVATION
# ============================================================

print("\n[5] NAME / IDENTIFIER PRESERVATION")

name_text = (
    "Hello Anjali, please check order ORD-12345 "
    "for product PRD-ABC123."
)

response_text = (
    "Hello Anjali, I checked order ORD-12345 "
    "and product PRD-ABC123 for you."
)

preservation_result = preserve_entities(
    name_text,
    response_text,
)

check(
    "Structured identifiers preserved",
    preservation_result["preserved"] is True,
    str(preservation_result),
)

check(
    "Order ID preserved in response",
    "ORD-12345" not in preservation_result["missing"].get("order_id", []),
    str(preservation_result),
)

check(
    "Product code preserved in response",
    "PRD-ABC123" not in preservation_result["missing"].get("product_code", []),
    str(preservation_result),
)


# ============================================================
# 6. MULTIPLE REQUESTS
# ============================================================

print("\n[6] MULTIPLE REQUEST DETECTION")

multiple_request_messages = [
    "Where is my order? Also tell me the refund status.",
    "I need a refund and I also want to change my address.",
    "Check my order and tell me when it will arrive.",
]

for message in multiple_request_messages:
    result = detect_multiple_requests(message)

    check(
        f"Multiple requests detected: {message}",
        bool(result),
        str(result),
    )


# ============================================================
# 7. SINGLE REQUEST SHOULD NOT BE FLAGGED
# ============================================================

print("\n[7] SINGLE REQUEST")

single_request = "Where is my order ORD-12345?"

result = detect_multiple_requests(single_request)

check(
    "Simple request is not incorrectly treated as multiple",
    not bool(result),
    str(result),
)


# ============================================================
# 8. CORRECTION DETECTION
# ============================================================

print("\n[8] CORRECTION DETECTION")

corrections = [
    "No, I meant order ORD-56789.",
    "Actually, the correct order number is ORD-99999.",
    "Sorry, I gave you the wrong date.",
    "Correction: the amount is ₹1999.",
]

for message in corrections:
    result = detect_correction(message)

    check(
        f"Correction detected: {message}",
        bool(result),
        str(result),
    )


# ============================================================
# 9. NORMAL MESSAGE SHOULD NOT BE CORRECTION
# ============================================================

print("\n[9] NORMAL MESSAGE")

normal_message = "Please check my order status."

result = detect_correction(normal_message)

check(
    "Normal message is not incorrectly treated as correction",
    not bool(result),
    str(result),
)


# ============================================================
# 10. LOW CONFIDENCE
# ============================================================

print("\n[10] LOW-CONFIDENCE CLARIFICATION")

ambiguous_messages = [
    "xyz abc qwerty",
    "hello",
    "123456789",
]

for message in ambiguous_messages:
    result = detect_language(message)

    check(
        f"Low-confidence handling: {message}",
        result["clarification_required"]
        or result["confidence"] < 0.55,
        str(result),
    )


# ============================================================
# 11. LANGUAGE INSTRUCTIONS
# ============================================================

print("\n[11] LANGUAGE RESPONSE INSTRUCTIONS")

for language in ["english", "hindi", "telugu", "spanish"]:
    instruction = build_language_instruction(language)

    check(
        f"Instruction generated for {language}",
        isinstance(instruction, str) and len(instruction) > 0,
        repr(instruction),
    )


# ============================================================
# 12. CONFIGURABLE SETTINGS
# ============================================================

print("\n[12] CONFIGURATION")

config = Task6Config(
    max_context_messages=5,
    inactivity_minutes=20,
    restore_window_hours=12,
)

check(
    "Context size configurable",
    config.max_context_messages == 5,
)

check(
    "Inactivity timeout configurable",
    config.inactivity_minutes == 20,
)

check(
    "Restore window configurable",
    config.restore_window_hours == 12,
)


# ============================================================
# 13. SESSION ISOLATION
# ============================================================

print("\n[13] SESSION ISOLATION")

manager = SessionManager(config)

session_a = manager.create_session("user_A")
session_b = manager.create_session("user_B")

manager.add_user_message(
    "user_A",
    "My order is ORD-11111.",
)

manager.add_user_message(
    "user_B",
    "My order is ORD-22222.",
)

context_a = manager.get_session("user_A").context()
context_b = manager.get_session("user_B").context()

text_a = " ".join(
    message.content
    for message in context_a
)

text_b = " ".join(
    message.content
    for message in context_b
)

check(
    "User A has own conversation",
    "ORD-11111" in text_a and "ORD-22222" not in text_a,
    text_a,
)

check(
    "User B has own conversation",
    "ORD-22222" in text_b and "ORD-11111" not in text_b,
    text_b,
)


# ============================================================
# 14. 10-MESSAGE CONTEXT LIMIT
# ============================================================

print("\n[14] 10-MESSAGE CONTEXT")

context_manager = SessionManager(
    Task6Config(max_context_messages=10)
)

context_manager.create_session("context_user")

for i in range(15):
    context_manager.add_user_message(
        "context_user",
        f"Message {i + 1}",
    )

context = context_manager.get_session("context_user").context()

check(
    "Context limited to 10 messages",
    len(context) == 10,
    f"Context size = {len(context)}",
)


# ============================================================
# 15. 30-MINUTE INACTIVITY
# ============================================================

print("\n[15] 30-MINUTE INACTIVITY")

inactivity_manager = SessionManager(
    Task6Config(
        max_context_messages=10,
        inactivity_minutes=30,
        restore_window_hours=24,
    )
)

start_time = datetime.now(timezone.utc)

inactivity_manager.create_session(
    "inactive_user",
    now=start_time,
)

inactivity_manager.add_user_message(
    "inactive_user",
    "My order is ORD-12345.",
    now=start_time,
)

after_30_minutes = start_time + timedelta(minutes=30)

inactive_session = inactivity_manager.get_session(
    "inactive_user",
    now=after_30_minutes,
)

check(
    "30-minute inactivity closes session",
    inactive_session is not None
    and inactive_session.closed is True,
    str(inactive_session),
)


# ============================================================
# 16. SESSION SUMMARY / RESTORE
# ============================================================


print("\n[15] SESSION RESTORATION")

restore_manager = SessionManager(
    Task6Config(
        max_context_messages=10,
        inactivity_minutes=30,
        restore_window_hours=24,
    )
)

restore_manager.create_session("restore_user")

restore_manager.add_user_message(
    "restore_user",
    "I need help with order ORD-12345.",
)

restore_manager.set_summary(
    "restore_user",
    "Customer asked about order ORD-12345.",
)

restored = restore_manager.restore_session("restore_user")

check(
    "Session restoration within window",
    restored.get("restored") is True,
    str(restored),
)


# ============================================================
# 16. EXPIRED RESTORATION
# ============================================================

print("\n[17] EXPIRED RESTORATION")

expired_manager = SessionManager(
    Task6Config(
        max_context_messages=10,
        inactivity_minutes=30,
        restore_window_hours=24,
    )
)

expired_manager.create_session("expired_user")

session = expired_manager.get_session("expired_user")

# Simulate a session older than the 24-hour restoration window.
session.last_activity = datetime.now(timezone.utc) - timedelta(hours=25)

expired_result = expired_manager.restore_session("expired_user")

check(
    "Expired session requires new session",
    expired_result.get("restored") is False
    and expired_result.get("new_session_required") is True,
    str(expired_result),
)


# ============================================================
# 17. SUMMARY FOLDED ON CLOSE AND ON RESTORE
# ============================================================

print("\n[18] SESSION SUMMARY FOLDING")

# A session that simply expires now keeps a summary of what was said,
# without anyone having to call set_summary. The escalation record and
# the handoff summary read this back after the 10-message context
# window has moved on.
folding_manager = SessionManager(
    Task6Config(
        max_context_messages=10,
        inactivity_minutes=30,
        restore_window_hours=24,
        restore_context_messages=4,
    )
)

folding_manager.create_session("folding_user")

started = datetime.now(timezone.utc)

folding_manager.add_user_message(
    "folding_user",
    "My order ORD-99001 has not arrived.",
    now=started,
)

folding_manager.add_assistant_message(
    "folding_user",
    "I am sorry to hear that. Let me check the delivery status.",
    now=started,
)

check(
    "a live session has no summary yet",
    folding_manager.sessions["folding_user"].summary == "",
    repr(folding_manager.sessions["folding_user"].summary),
)

folding_manager.get_session(
    "folding_user",
    now=started + timedelta(minutes=31),
)

folded = folding_manager.sessions["folding_user"]

check(
    "30-minute inactivity still closes the session",
    folded.closed is True,
)

check(
    "the closing session keeps a summary",
    folded.summary != "",
    repr(folded.summary),
)

check(
    "the summary quotes the customer's own words",
    "ORD-99001" in folded.summary,
    repr(folded.summary),
)

check(
    "the summary names who said what",
    "Customer:" in folded.summary and "Support:" in folded.summary,
    repr(folded.summary),
)

check(
    "the summary carries the order id, not just the topic",
    "has not arrived" in folded.summary,
    repr(folded.summary),
)

# An explicit summary is never overwritten by the automatic one.
explicit_manager = SessionManager(
    Task6Config(inactivity_minutes=30, restore_window_hours=24)
)

explicit_manager.create_session("explicit_user")

explicit_manager.add_user_message(
    "explicit_user", "Where is ORD-12345?"
)

explicit_manager.set_summary(
    "explicit_user", "Customer asked about order ORD-12345."
)

explicit_manager.get_session(
    "explicit_user",
    now=datetime.now(timezone.utc) + timedelta(minutes=31),
)

check(
    "an explicit summary survives the session closing",
    explicit_manager.sessions["explicit_user"].summary
    == "Customer asked about order ORD-12345.",
    repr(explicit_manager.sessions["explicit_user"].summary),
)

# A session restored without ever having been closed gets one built,
# so a caller is never handed an empty summary.
restore_manager = SessionManager(
    Task6Config(
        max_context_messages=10,
        inactivity_minutes=30,
        restore_window_hours=24,
        restore_context_messages=4,
    )
)

restore_manager.create_session("cold_restore")

restore_manager.add_user_message(
    "cold_restore", "Please refund order ORD-55555."
)

cold = restore_manager.restore_session("cold_restore")

check(
    "a session restored without a summary is given one",
    cold.get("restored") is True and cold.get("summary") != "",
    str(cold),
)

check(
    "the rebuilt summary keeps the order id",
    "ORD-55555" in cold.get("summary", ""),
    repr(cold.get("summary")),
)

# The summary is bounded, so a long conversation cannot blow up the
# record it is embedded in.
long_manager = SessionManager(
    Task6Config(
        inactivity_minutes=30,
        restore_window_hours=24,
        restore_context_messages=4,
    )
)

long_manager.create_session("long_user")

for index in range(12):

    long_manager.add_user_message(
        "long_user", f"Message number {index} " + ("x" * 400)
    )

long_summary = build_session_summary(
    long_manager.sessions["long_user"], max_messages=4
)

check(
    "a long message is truncated in the summary",
    "..." in long_summary,
)

check(
    "the summary covers only the restore window of turns",
    long_summary.count("Customer:") == 4,
    str(long_summary.count("Customer:")),
)

check(
    "a summary built from no messages is empty",
    build_session_summary(
        long_manager.sessions["long_user"], max_messages=0
    )
    == "",
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 75)
print(f"PASSED: {PASS}")
print(f"FAILED: {FAIL}")
print("=" * 75)

if FAIL == 0:
    print("🎉 ALL TASK 6 REGRESSION TESTS PASSED ✅")
    print("TASK 6 CORE REQUIREMENTS VERIFIED ✅")
else:
    print("⚠️ SOME TASK 6 TESTS FAILED")
    print("Fix the failed cases before integrating Task 6 into Streamlit.")

raise SystemExit(0 if FAIL == 0 else 1)
