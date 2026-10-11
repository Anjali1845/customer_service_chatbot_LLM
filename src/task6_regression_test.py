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
- Short follow-ups inheriting the session's established language
- Unsupported languages flagged explicitly, not guessed
- Labelled names extracted; emails and codes never yield a name
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
    extract_person_names,
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
# 19. SHORT FOLLOW-UPS INHERIT THE SESSION LANGUAGE
# ============================================================

print("\n[19] CONTEXT-AWARE FOLLOW-UPS")

followup_manager = SessionManager(
    Task6Config(max_context_messages=10)
)

followup_manager.add_user_message(
    "followup_user",
    "My order ORD-12345 has not arrived.",
)

# Each of these is the natural next message inside an established
# session. None of them carries a language of its own, so each must
# continue the session's language instead of falling to "unknown"
# and triggering the low-confidence clarification.

for followup in ["yes", "ORD-1234", "ok thanks"]:

    result = followup_manager.add_user_message(
        "followup_user",
        followup,
    )

    check(
        f"Short follow-up inherits the session language: {followup!r}",
        result["language"]["language"] == "english"
        and result["language"]["confidence"] >= 0.55
        and not result["clarification_required"],
        str(result["language"]),
    )

bare_id_result = followup_manager.add_user_message(
    "followup_user",
    "ORD-5678",
)

check(
    "a bare order id is a valid in-context reply, never low-confidence",
    not bare_id_result["clarification_required"]
    and bare_id_result["language"]["language"] == "english"
    and "ORD-5678" in bare_id_result["entities"]["order_id"],
    str(bare_id_result),
)

# A bare identifier is never a language problem even in a brand-new
# session: it is the answer to the assistant's own question, so it
# must not be rejected with a clarification.

fresh_id_manager = SessionManager(Task6Config())

fresh_id_result = fresh_id_manager.add_user_message(
    "fresh_id_user",
    "ORD-1234",
)

check(
    "a bare order id is never rejected, even with no session language",
    not fresh_id_result["clarification_required"]
    and "ORD-1234" in fresh_id_result["entities"]["order_id"],
    str(fresh_id_result),
)

# A Hindi-speaking customer's single-word "yes" continues their
# Hindi conversation instead of tripping the clarification.

hindi_followup_manager = SessionManager(Task6Config())

hindi_followup_manager.add_user_message(
    "hindi_followup_user",
    "मेरा ऑर्डर अभी तक नहीं आया है",
)

haan_result = hindi_followup_manager.add_user_message(
    "hindi_followup_user",
    "Haan",
)

check(
    "'Haan' continues a hindi session without clarification",
    haan_result["language"]["language"] == "hindi"
    and not haan_result["clarification_required"],
    str(haan_result["language"]),
)

check(
    "'Haan' is recognised as hindi on its own",
    detect_language("Haan")["language"] == "hindi",
    str(detect_language("Haan")),
)

# The inheritance must not swallow genuinely ambiguous first
# messages: with no session language to inherit, a bare "yes" is
# still unknown and still asks for clarification.

fresh_yes_manager = SessionManager(Task6Config())

fresh_yes_result = fresh_yes_manager.add_user_message(
    "fresh_yes_user",
    "yes",
)

check(
    "an ambiguous first message with no session still clarifies",
    fresh_yes_result["clarification_required"],
    str(fresh_yes_result["language"]),
)

# A message with neither words nor identifiers -- bare punctuation --
# has nothing to inherit, so it stays a clarification even inside
# an established session.

punctuation_result = followup_manager.add_user_message(
    "followup_user",
    "??",
)

check(
    "punctuation-only stays a clarification even in a session",
    punctuation_result["clarification_required"],
    str(punctuation_result["language"]),
)


# ============================================================
# 20. UNSUPPORTED LANGUAGES ARE FLAGGED, NOT GUESSED
# ============================================================

print("\n[20] UNSUPPORTED LANGUAGES")

french_message = (
    "Bonjour, je veux un remboursement pour ma commande"
)

french_result = detect_language(french_message)

check(
    "a french message is flagged unsupported, not answered as spanish",
    french_result["language"] == "unsupported"
    and french_result.get("unsupported") is True
    and french_result.get("unsupported_language") == "french"
    and french_result["clarification_required"],
    str(french_result),
)

german_result = detect_language(
    "Ich möchte eine Rückerstattung für meine Bestellung"
)

check(
    "a german message is flagged unsupported",
    german_result["language"] == "unsupported"
    and german_result.get("unsupported_language") == "german"
    and german_result["clarification_required"],
    str(german_result),
)

script_result = detect_language("我的订单还没有到")

check(
    "a message in an unsupported script is flagged",
    script_result["language"] == "unsupported"
    and script_result.get("unsupported") is True
    and script_result["clarification_required"],
    str(script_result),
)

# The unsupported signal survives inside an established session: it
# is never silently inherited past, because the customer has
# switched to a language the product cannot answer in.

unsupported_session_manager = SessionManager(Task6Config())

unsupported_session_manager.add_user_message(
    "unsupported_user",
    "My order ORD-12345 has not arrived.",
)

unsupported_result = unsupported_session_manager.add_user_message(
    "unsupported_user",
    french_message,
)

check(
    "an unsupported language still clarifies inside a session",
    unsupported_result["clarification_required"]
    and unsupported_result["language"]["language"] == "unsupported",
    str(unsupported_result["language"]),
)

# A supported message must never be flagged by mistake: the spanish
# sample is answered in spanish, not sent to clarification.

spanish_check_result = detect_language(
    "Necesito ayuda con mi pedido"
)

check(
    "a supported message is never flagged unsupported",
    spanish_check_result["language"] == "spanish"
    and not spanish_check_result.get("unsupported")
    and not spanish_check_result["clarification_required"],
    str(spanish_check_result),
)


# ============================================================
# 21. TRANSLITERATED TELUGU CONFIDENCE
# ============================================================

print("\n[21] TRANSLITERATED TELUGU CONFIDENCE")

telugu_result = detect_language("Naa order inka raaledu")

check(
    "romanised telugu is detected with sensible confidence",
    telugu_result["language"] == "telugu"
    and telugu_result["confidence"] >= 0.55
    and not telugu_result["clarification_required"],
    str(telugu_result),
)

check(
    "romanised telugu may be mixed but never a clarification",
    telugu_result["mixed"] is True
    and "telugu" in telugu_result["detected_languages"],
    str(telugu_result),
)


# ============================================================
# 22. LABELLED NAMES ARE EXTRACTED
# ============================================================

print("\n[22] LABELLED NAMES")

# Each of these is a customer giving their name in a form a ticket
# needs: without the extraction, the ticket asks the customer for a
# name they already gave.

labelled_name_tests = [
    ("Name Anil Rao anil@x.com", "Anil Rao"),
    ("my name is Priya Sharma", "Priya Sharma"),
    ("I am Ravi Kumar", "Ravi Kumar"),
    ("this is Sana Ali", "Sana Ali"),
    ("name: Anil Rao", "Anil Rao"),
]

for text, expected in labelled_name_tests:

    names = extract_person_names(text)

    check(
        f"Labelled name extracted: {text!r}",
        expected in names,
        f"expected {expected!r} in {names}",
    )

# The labelled name survives into the full entity extraction that
# feeds the ticket's customer_name.

labelled_entities = extract_preserved_entities(
    "Name Anil Rao anil@x.com"
)

check(
    "a labelled name reaches extract_preserved_entities",
    "Anil Rao" in labelled_entities.get("person_name", []),
    str(labelled_entities.get("person_name")),
)

# An email address must never yield a person name: not its local
# part, not the words that follow it.

for text, expected in [
    ("my name is Anil Anil.Rao@x.com", ["Anil"]),
    ("Email anil.rao@x.com", []),
    ("Email anil.rao@x.com Name Anil Rao", ["Anil Rao"]),
    ("contact priya.sharma@example.com", []),
]:

    names = extract_person_names(text)

    check(
        f"An email never yields a name: {text!r}",
        names == expected,
        f"expected {expected}, got {names}",
    )

# The words after an order id or a product code are never read as a
# name: a code fragment is not a person.

for text, expected in [
    ("this is ORD-12345 status pending", []),
    ("this is PRD-ABC123 broken", []),
    ("product SKU-9912 Name Anil Rao", ["Anil Rao"]),
]:

    names = extract_person_names(text)

    check(
        f"Words after an identifier are not a name: {text!r}",
        names == expected,
        f"expected {expected}, got {names}",
    )

# A bare capitalised full name with no label stays unextracted:
# the extractor is conservative, and an ordinary pair of
# capitalised words is not evidence of a person.

check(
    "a bare capitalised full name without a label stays conservative",
    extract_person_names("Anil Rao") == [],
    str(extract_person_names("Anil Rao")),
)

# A label that names a thing, not a person, is not a customer name.

check(
    "a product name is not a person name",
    extract_person_names("product name Premium Mixer is wrong")
    == [],
    str(extract_person_names("product name Premium Mixer is wrong")),
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
