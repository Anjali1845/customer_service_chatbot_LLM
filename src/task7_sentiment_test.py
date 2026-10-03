"""
TASK 7.1 - SENTIMENT ANALYSIS TEST SUITE

Covers:
- Six labels across four languages
- Negation and double negation
- Intensifier scaling
- Neutral fallback confidence
- Sarcasm in all four languages, keeping negative polarity
- Calm high-risk phrasing staying positive/neutral in tone
- Language detection agreement with Task 6
- History blend and repeated-negative promotion
- Explicit human requests
- Tone instruction carries the no-policy-change guard
"""

from datetime import datetime, timezone

from task6_multilingual import ConversationMessage
from task7_sentiment import (
    SentimentConfig,
    SentimentLabel,
    RiskTopic,
    analyse_conversation,
    analyse_message,
    build_tone_instruction,
    combine_with_history,
    detect_high_risk_topics,
    requests_human,
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
print("TASK 7.1 SENTIMENT ANALYSIS TESTS")
print("=" * 75)


# ============================================================
# 1. SIX LABELS X FOUR LANGUAGES
# ============================================================

print("\n[1] SIX LABELS ACROSS FOUR LANGUAGES")

MATRIX = [
    # English
    ("english", "positive", "Thank you, the refund worked perfectly and I am very happy."),
    ("english", "neutral", "What is the refund policy for my order?"),
    ("english", "negative", "This is terrible service, my order is broken and late."),
    ("english", "frustrated", "I called again and again, still not resolved."),
    ("english", "urgent", "This is urgent, I need it immediately."),
    ("english", "sarcastic", "Great, still not refunded. Thanks for nothing."),

    # Hindi
    ("hindi", "positive", "बहुत अच्छा, मेरा रिफंड मिल गया, धन्यवाद"),
    ("hindi", "neutral", "मेरे ऑर्डर का रिफंड कैसे मिलेगा?"),
    ("hindi", "negative", "यह बहुत खराब सेवा है, मेरा ऑर्डर खराब है"),
    ("hindi", "frustrated", "मैंने कितनी बार बताया, अभी तक नहीं हुआ"),
    ("hindi", "urgent", "यह तुरंत जरूरी है, जल्दी करें"),
    ("hindi", "sarcastic", "वाह बढ़िया, फिर से वही समस्या"),

    # Telugu
    ("telugu", "positive", "చాలా బాగుంది, ధన్యవాదాలు"),
    ("telugu", "neutral", "నా ఆర్డర్ రిఫండ్ ఎలా వస్తుంది?"),
    ("telugu", "negative", "ఇది చాలా చెడు సేవ"),
    ("telugu", "frustrated", "మళ్ళీ మళ్ళీ అడుగుతున్నా, ఇంకా రాలేదు"),
    ("telugu", "urgent", "ఇది అత్యవసరం, వెంటనే కావాలి"),
    ("telugu", "sarcastic", "చాలా బాగుంది, ఇంకా రాలేదు"),

    # Spanish
    ("spanish", "positive", "Muchas gracias, el reembolso funcionó perfecto"),
    ("spanish", "neutral", "¿Cuál es la política de reembolso?"),
    ("spanish", "negative", "El servicio es pésimo, mi pedido está dañado"),
    ("spanish", "frustrated", "He llamado otra vez, todavía no está resuelto"),
    ("spanish", "urgent", "Es urgente, lo necesito inmediatamente"),
    ("spanish", "sarcastic", "Qué genial, otra vez el mismo problema"),
]

for language, expected, text in MATRIX:

    result = analyse_message(text, language)

    check(
        f"{language}/{expected}",
        result.label == expected,
        f"got {result.label} conf={result.confidence} "
        f"evidence={result.evidence}",
    )

    check(
        f"{language}/{expected} has language {language}",
        result.language == language,
        f"got {result.language}",
    )


# ============================================================
# 2. NEGATION
# ============================================================

print("\n[2] NEGATION")

not_good = analyse_message("The service was not good at all.")

check(
    "'not good' is negative",
    not_good.label == SentimentLabel.NEGATIVE.value,
    f"got {not_good.label}",
)

check(
    "'not good' has negative polarity",
    not_good.polarity < 0,
    f"got {not_good.polarity}",
)

double = analyse_message("Not bad actually, quite good.")

check(
    "'not bad' plus 'good' is positive",
    double.label == SentimentLabel.POSITIVE.value,
    f"got {double.label}",
)

check(
    "'not bad' has positive polarity",
    double.polarity > 0,
    f"got {double.polarity}",
)

hindi_not = analyse_message("सेवा अच्छी नहीं है")

check(
    "Hindi 'अच्छी नहीं' is negative",
    hindi_not.label == SentimentLabel.NEGATIVE.value,
    f"got {hindi_not.label} evidence={hindi_not.evidence}",
)


# ============================================================
# 3. INTENSIFIERS
# ============================================================

print("\n[3] INTENSIFIERS")

plain = analyse_message("The service was bad.")
strong = analyse_message("The service was very bad.")

check(
    "intensifier raises intensity",
    strong.intensity > plain.intensity,
    f"plain={plain.intensity} strong={strong.intensity}",
)

check(
    "intensifier raises confidence",
    strong.confidence > plain.confidence,
    f"plain={plain.confidence} strong={strong.confidence}",
)

hindi_plain = analyse_message("सेवा खराब है")
hindi_strong = analyse_message("सेवा बहुत खराब है")

check(
    "Hindi intensifier raises intensity",
    hindi_strong.intensity > hindi_plain.intensity,
    f"plain={hindi_plain.intensity} strong={hindi_strong.intensity}",
)


# ============================================================
# 3b. FRUSTRATION VOCABULARY
#
# "frustrating" is the most common complaint word in support
# traffic. It was missing from the lexicon once and made
# "this is very frustrating" read as neutral, so it is asserted
# explicitly rather than left to the matrix above.
# ============================================================

print("\n[3b] FRUSTRATION VOCABULARY")

NEGATIVE_WORDS = (
    "this is very frustrating",
    "I am frustrated with this service",
    "the whole thing is so annoying",
    "I am fed up with waiting",
    "they were completely unhelpful",
    "this is absolutely appalling",
    "I am sick of chasing you",
)

for text in NEGATIVE_WORDS:

    result = analyse_message(text)

    check(
        f"negative: {text!r}",
        result.label in ("negative", "frustrated", "urgent")
        and result.polarity < 0,
        f"got {result.label} pol={result.polarity} ev={result.evidence}",
    )

for text in (
    "still no update from anyone",
    "I have been waiting for days",
):

    result = analyse_message(text)

    check(
        f"frustrated: {text!r}",
        result.label in ("frustrated", "negative"),
        f"got {result.label} ev={result.evidence}",
    )

for language, text in (
    ("hindi", "मैं बहुत निराश हूँ"),
    ("hindi", "सेवा बेकार है"),
    ("telugu", "నేను చాలా నిరాశగా ఉన్నాను"),
    ("spanish", "Estoy muy frustrado con el servicio"),
    ("spanish", "El servicio es muy molesto"),
):

    result = analyse_message(text, language)

    check(
        f"frustration vocabulary ({language}): {text!r}",
        result.label in ("negative", "frustrated"),
        f"got {result.label} ev={result.evidence}",
    )


# ============================================================
# 4. NEUTRAL FALLBACK
# ============================================================

print("\n[4] NEUTRAL FALLBACK")

config = SentimentConfig()

blank = analyse_message("Where is my invoice please")

check(
    "no markers yields neutral",
    blank.label == SentimentLabel.NEUTRAL.value,
    f"got {blank.label}",
)

check(
    "no markers yields the configured default confidence",
    blank.confidence == config.neutral_default_confidence,
    f"got {blank.confidence}",
)

check(
    "no markers yields zero polarity",
    blank.polarity == 0.0,
    f"got {blank.polarity}",
)

check(
    "neutral has no risk",
    blank.risk_severity == "none",
    f"got {blank.risk_severity}",
)


# ============================================================
# 5. SARCASM
# ============================================================

print("\n[5] SARCASM")

SARCASM_CASES = [
    ("english", "Great, still not refunded."),
    ("hindi", "वाह बढ़िया, फिर से वही समस्या"),
    ("telugu", "చాలా బాగుంది, ఇంకా రాలేదు"),
    ("spanish", "Qué genial, otra vez el mismo problema"),
]

for language, text in SARCASM_CASES:

    result = analyse_message(text, language)

    check(
        f"sarcasm detected ({language})",
        result.sarcasm is True
        and result.label == SentimentLabel.SARCASTIC.value,
        f"got label={result.label} sarcasm={result.sarcasm} "
        f"score={result.sarcasm_score}",
    )

    check(
        f"sarcasm keeps negative polarity ({language})",
        result.polarity < 0,
        f"got {result.polarity}",
    )

    check(
        f"sarcasm keeps the underlying label ({language})",
        result.underlying_label != SentimentLabel.SARCASTIC.value,
        f"got underlying={result.underlying_label}",
    )

check(
    "sarcastic message still counts as negative for escalation",
    analyse_message("Great, still not refunded.").label
    in ("negative", "frustrated", "sarcastic", "urgent"),
)

# Sincere praise must not be misread as sarcasm.
sincere = analyse_message(
    "Great, thank you so much for resolving this!"
)

check(
    "sincere thanks is not sarcastic",
    sincere.sarcasm is False
    and sincere.label == SentimentLabel.POSITIVE.value,
    f"got label={sincere.label} sarcasm={sincere.sarcasm}",
)

sure = analyse_message(
    "Sure, because you people never respond to anyone."
)

check(
    "'sure, because ... never' is sarcastic",
    sure.sarcasm is True,
    f"got label={sure.label} sarcasm={sure.sarcasm}",
)


# ============================================================
# 6. CALM HIGH RISK (INDEPENDENT OF POLARITY)
# ============================================================

print("\n[6] CALM HIGH-RISK PHRASING")

CALM_RISK = [
    (
        "duplicate_payment",
        "high",
        "I would appreciate it if you could look into what appears to "
        "be a duplicate charge on my card, whenever it is convenient.",
    ),
    (
        "duplicate_payment",
        "high",
        "Could you kindly check whether I have been charged twice? "
        "No urgency at all.",
    ),
    (
        "account_compromise",
        "critical",
        "Could you kindly help me understand an unrecognised login on "
        "my account? There is no rush.",
    ),
    (
        "account_compromise",
        "critical",
        "My account was hacked and the password changed without my "
        "permission. Please advise when you can.",
    ),
    (
        "legal_threat",
        "critical",
        "I wanted to politely ask whether I should involve my lawyer "
        "about this matter.",
    ),
]

for topic, severity, text in CALM_RISK:

    result = analyse_message(text)

    check(
        f"calm phrasing stays non-negative [{topic}]",
        result.label in ("positive", "neutral"),
        f"got {result.label}",
    )

    check(
        f"calm phrasing still flags {topic}",
        topic in result.risk_topics,
        f"got {result.risk_topics}",
    )

    check(
        f"calm phrasing reports {severity} severity [{topic}]",
        result.risk_severity == severity,
        f"got {result.risk_severity}",
    )


# ============================================================
# 7. RISK TOPICS ACROSS LANGUAGES
# ============================================================

print("\n[7] RISK TOPICS ACROSS LANGUAGES")

RISK_MATRIX = [
    ("duplicate_payment", "do baar paise kat gaye, duplicate payment"),
    ("duplicate_payment", "rendu sarlu cut ayindi"),
    ("duplicate_payment", "me cobraron dos veces, pago duplicado"),
    ("account_compromise", "mera account hack ho gaya"),
    ("account_compromise", "naa account hack ayindi"),
    ("account_compromise", "mi cuenta fue hackeada, acceso no autorizado"),
    ("legal_threat", "मैं वकील से बात करूंगा, कोर्ट जाऊंगा"),
    ("legal_threat", "voy a hablar con un abogado, demanda"),
    ("unauthorized_charge", "there is a fraudulent charge on my card"),
    ("data_breach", "I think there has been a data leak of my details"),
]

for expected_topic, text in RISK_MATRIX:

    matches = detect_high_risk_topics(text)

    topics = [match["topic"] for match in matches]

    check(
        f"risk {expected_topic}",
        expected_topic in topics,
        f"got {topics} for {text!r}",
    )

check(
    "disabled risk topics are not reported",
    detect_high_risk_topics(
        "there is a fraudulent charge",
        SentimentConfig(risk_topics_enabled=("legal_threat",)),
    )
    == [],
)

check(
    "risk severity ordering picks the worst",
    analyse_message(
        "My account was hacked and I have been charged twice."
    ).risk_severity
    == "critical",
)


# ============================================================
# 8. HISTORY BLEND
# ============================================================

print("\n[8] HISTORY BLEND")

config = SentimentConfig()

first = analyse_message("My order is late and this is bad.")
second = analyse_message("The delivery is still not here.")

check(
    "first negative turn alone is not repeated",
    first.repeated_negative is False,
)

blended = combine_with_history(second, [first], config)

check(
    "two negatives mark the conversation repeated",
    blended.repeated_negative is True,
    f"got {blended.repeated_negative}",
)

check(
    "repeated negativity promotes to frustrated",
    blended.label == SentimentLabel.FRUSTRATED.value,
    f"got {blended.label}",
)

blended_single = combine_with_history(second, [], config)

check(
    "a single turn is not repeated",
    blended_single.repeated_negative is False,
)

urgent_second = analyse_message("This is urgent, I need it immediately.")

urgent_blended = combine_with_history(urgent_second, [first], config)

check(
    "urgent is never downgraded by history",
    urgent_blended.label == SentimentLabel.URGENT.value,
    f"got {urgent_blended.label}",
)

risk_second = analyse_message(
    "Could you kindly check a duplicate charge on my card?"
)

risk_blended = combine_with_history(risk_second, [first, second], config)

check(
    "risk topic survives the history blend",
    "duplicate_payment" in risk_blended.risk_topics,
    f"got {risk_blended.risk_topics}",
)


# ============================================================
# 9. CONVERSATION ANALYSIS
# ============================================================

print("\n[9] CONVERSATION ANALYSIS")

now = datetime(2026, 10, 2, 10, 0, tzinfo=timezone.utc)

messages = [
    ConversationMessage(
        role="user",
        content="My order has not arrived.",
        timestamp=now,
        language="english",
    ),
    ConversationMessage(
        role="assistant",
        content="Let me check that for you.",
        timestamp=now,
        language="english",
    ),
    ConversationMessage(
        role="user",
        content="This is bad, still not here. How many times?",
        timestamp=now,
        language="english",
    ),
]

conversation = analyse_conversation(messages)

check(
    "conversation analysis sees history",
    conversation.repeated_negative is True,
    f"got {conversation.repeated_negative}",
)

check(
    "conversation analysis counts messages",
    conversation.message_count >= 2,
    f"got {conversation.message_count}",
)

check(
    "conversation analysis reports a label",
    conversation.label in [label.value for label in SentimentLabel],
    f"got {conversation.label}",
)

empty = analyse_conversation([])

check(
    "empty conversation is neutral",
    empty.label == SentimentLabel.NEUTRAL.value,
    f"got {empty.label}",
)


# ============================================================
# 10. HUMAN REQUESTS
# ============================================================

print("\n[10] HUMAN REQUESTS")

check("English human request", requests_human("I want to talk to a human"))
check("manager request", requests_human("Please connect me to your manager"))
check("Hindi human request", requests_human("मुझे मैनेजर से बात करनी है"))
check("Spanish human request", requests_human("Quiero hablar con un agente"))
check(
    "ordinary message is not a human request",
    not requests_human("What is your refund policy?"),
)


# ============================================================
# 11. TONE INSTRUCTION
# ============================================================

print("\n[11] TONE INSTRUCTION")

frustrated = analyse_message(
    "I called again and again, still not resolved."
)

tone = build_tone_instruction(frustrated, "english")

check(
    "tone instruction states the style",
    "Communication style" in tone,
)

check(
    "tone instruction keeps the response language",
    "Respond in English" in tone,
)

check(
    "tone instruction forbids policy changes",
    "Do NOT change any policy" in tone and "eligibility" in tone,
)

check(
    "tone instruction promises nothing extra",
    "Do not promise anything beyond the knowledge base" in tone,
)

sarcastic_tone = build_tone_instruction(
    analyse_message("Great, still not refunded."), "english"
)

check(
    "sarcastic tone never mirrors sarcasm",
    "Never mirror sarcasm" in sarcastic_tone,
    sarcastic_tone,
)

hindi_tone = build_tone_instruction(
    analyse_message("सेवा बहुत खराब है"), "hindi"
)

check(
    "tone instruction uses the display language name",
    "Respond in Hindi" in hindi_tone,
    hindi_tone,
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 75)
print(f"PASSED: {PASS}")
print(f"FAILED: {FAIL}")
print("=" * 75)

raise SystemExit(0 if FAIL == 0 else 1)
