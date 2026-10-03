"""
TASK 6 - MULTILINGUAL + CONTEXT + SESSION MANAGEMENT

Requirements covered by this core module:
- Three additional languages:
    Hindi
    Telugu
    Spanish
- Mixed-language / code switching
- Preserve names, order IDs, dates, product codes and amounts
- Maximum 10-message conversation context
- Session isolation
- Low-confidence language clarification
- Spelling / transliteration tolerance
- Multiple request detection
- Customer correction detection
- 30-minute inactivity handling
- Conversation restoration within 24 hours
- New session after 24 hours
- Configurable limits

No external language-detection dependency is required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import re
import uuid
from typing import Dict, List, Optional, Tuple


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Task6Config:

    supported_languages: Tuple[str, ...] = (
        "english",
        "hindi",
        "telugu",
        "spanish",
    )

    max_context_messages: int = 10

    inactivity_minutes: int = 30

    restore_window_hours: int = 24

    minimum_language_confidence: float = 0.55

    minimum_intent_confidence: float = 0.50

    # Tolerate customer spelling errors and transliteration.
    typo_tolerance: bool = True

    # How many recent messages are folded into a restored summary.
    restore_context_messages: int = 4


# ============================================================
# LANGUAGE DETECTION
# ============================================================

LANGUAGE_NAMES = {
    "english": "English",
    "hindi": "Hindi",
    "telugu": "Telugu",
    "spanish": "Spanish",
}


# Common words support both native scripts and
# transliterated customer messages.
LANGUAGE_KEYWORDS = {

    "english": {
        "the",
        "is",
        "are",
        "was",
        "were",
        "my",
        "order",
        "refund",
        "payment",
        "please",
        "help",
        "want",
        "need",
        "when",
        "why",
        "how",
        "and",
        "for",
        "where",
        "what",
        "status",
        "track",
        "tracking",
        "delivery",
        "delivered",
        "late",
        "arrive",
        "arrived",
        "money",
        "back",
        "cancel",
        "return",
        "charge",
        "charged",
        "wrong",
        "issue",
        "problem",
        "account",
        "damaged",
        "missing",
        "still",
        "received",
        "again",
        "address",
        "change",
    },

    "hindi": {
        "hai",
        "mera",
        "meri",
        "mere",
        "mujhe",
        "aap",
        "kya",
        "kaise",
        "kab",
        "kyun",
        "chahiye",
        "kripya",
        "paise",
        "nahi",
        "nahin",
        "tha",
        "thi",
        "karna",
    },

    "telugu": {
        "na",
        "naa",
        "naku",
        "naaku",
        "meeru",
        "emi",
        "ela",
        "eppudu",
        "enduku",
        "kavali",
        "ledu",
        "chesanu",
        "chesina",
        "undi",
        "unnadi",
        "dayachesi",
    },

    "spanish": {
        "el",
        "la",
        "los",
        "las",
        "mi",
        "mis",
        "es",
        "son",
        "que",
        "como",
        "cuando",
        "por",
        "para",
        "necesito",
        "quiero",
        "ayuda",
        "pedido",
        "reembolso",
        "pago",
        "gracias",
    },
}


def _normalize_text(text: str) -> str:

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def _tokenize(text: str) -> List[str]:

    return re.findall(
        r"[a-zA-ZÀ-ÿ]+",
        _normalize_text(text),
    )


def _as_utc(
    moment: Optional[datetime],
) -> datetime:
    """
    Normalise any datetime to timezone-aware UTC.

    Naive timestamps are assumed to already be UTC. This keeps
    simulated time travel and mixed naive/aware callers from
    raising TypeError during session expiry maths.
    """

    if moment is None:

        return datetime.now(timezone.utc)

    if moment.tzinfo is None:

        return moment.replace(tzinfo=timezone.utc)

    return moment.astimezone(timezone.utc)


def _edit_distance(
    first: str,
    second: str,
    max_distance: int,
) -> int:
    """
    Levenshtein distance with an early exit once the
    distance provably exceeds max_distance.
    """

    if abs(len(first) - len(second)) > max_distance:

        return max_distance + 1

    previous = list(range(len(second) + 1))

    for row, char_first in enumerate(first, start=1):

        current = [row]

        for column, char_second in enumerate(second, start=1):

            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1]
                    + (char_first != char_second),
                )
            )

        if min(current) > max_distance:

            return max_distance + 1

        previous = current

    return previous[-1]


def _fuzzy_keyword_match(
    token: str,
    keywords,
) -> bool:
    """
    Match a misspelled or transliterated token against a
    keyword set. Short tokens are never fuzzy matched so that
    common stop words do not bleed across languages.
    """

    if len(token) < 4:
        return False

    allowed = 2 if len(token) >= 8 else 1

    for keyword in keywords:

        if len(keyword) < 4:
            continue

        if abs(len(keyword) - len(token)) > allowed:
            continue

        if (
            _edit_distance(token, keyword, allowed)
            <= allowed
        ):

            return True

    return False


def _script_scores(text: str) -> Dict[str, int]:
    """
    Detect strong Unicode-script evidence.

    Devanagari -> Hindi
    Telugu Unicode block -> Telugu
    """

    scores = {
        "hindi": 0,
        "telugu": 0,
    }

    for char in text:

        code = ord(char)

        # Devanagari
        if 0x0900 <= code <= 0x097F:

            scores["hindi"] += 1

        # Telugu
        elif 0x0C00 <= code <= 0x0C7F:

            scores["telugu"] += 1

    return scores


def detect_language(
    text: str,
    config: Optional[Task6Config] = None,
) -> Dict:
    """
    Detect primary language and possible mixed-language content.

    Returns:
        language
        confidence
        mixed
        detected_languages
        clarification_required
    """

    config = config or Task6Config()

    if not text or not text.strip():

        return {
            "language": "unknown",
            "confidence": 0.0,
            "mixed": False,
            "detected_languages": [],
            "clarification_required": True,
            "reason": "Empty message.",
        }

    normalized = _normalize_text(text)

    tokens = _tokenize(text)

    scores = {
        language: 0.0
        for language in config.supported_languages
    }

    # Unicode script evidence.
    script_scores = _script_scores(text)

    scores["hindi"] += (
        script_scores["hindi"] * 2.0
    )

    scores["telugu"] += (
        script_scores["telugu"] * 2.0
    )

    # Keyword evidence.
    exact_matches = set()

    for language, keywords in LANGUAGE_KEYWORDS.items():

        if language not in scores:
            continue

        for token in tokens:

            if token in keywords:

                scores[language] += 1.0
                exact_matches.add(token)

    # Spelling-error / transliteration tolerance. Only tokens that
    # did not exactly match any language are fuzzy matched, so a
    # real keyword is never re-scored into another language.
    if config.typo_tolerance:

        for token in tokens:

            if token in exact_matches:
                continue

            for language, keywords in LANGUAGE_KEYWORDS.items():

                if language not in scores:
                    continue

                if _fuzzy_keyword_match(token, keywords):

                    scores[language] += 0.6
                    break

    # Spanish-specific characters.
    if re.search(
        r"[áéíóúñ¿¡]",
        normalized,
    ):

        scores["spanish"] += 1.5

    total = sum(scores.values())

    if total <= 0:

        return {
            "language": "unknown",
            "confidence": 0.0,
            "mixed": False,
            "detected_languages": [],
            "clarification_required": True,
            "reason": "Language confidence is too low.",
        }

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    primary_language, primary_score = ranked[0]

    confidence = primary_score / total

    detected_languages = [
        language
        for language, score in ranked
        if score > 0
    ]

    # Mixed / code-switched when a second language carries real
    # evidence, not one stray token in a long monolingual message.
    secondary_score = (
        ranked[1][1]
        if len(ranked) > 1
        else 0.0
    )

    mixed = (
        len(detected_languages) >= 2
        and secondary_score >= 1.0
        and secondary_score / primary_score >= 0.10
    )

    clarification_required = (
        confidence
        < config.minimum_language_confidence
    )

    return {
        "language": primary_language,
        "confidence": round(
            confidence,
            3,
        ),
        "mixed": mixed,
        "detected_languages": detected_languages,
        "clarification_required": clarification_required,
        "scores": {
            language: round(
                score,
                3,
            )
            for language, score in scores.items()
        },
    }


# ============================================================
# IMPORTANT ENTITY PRESERVATION
# ============================================================

ENTITY_PATTERNS = {

    "order_id": [
        r"\bORD[-_][A-Z0-9-]+\b",
        r"\bORDER[-_ ]?ID[:\s#-]*[A-Z0-9-]+\b",
        r"\border\s*(?:number|no\.?|#)\s*[:\s#-]*([A-Z0-9][A-Z0-9-]{3,})\b",
    ],

    "product_code": [
        r"\b(?!ORD[-_])(?!INR\b)[A-Z]{1,6}[-_][A-Z0-9]{2,12}\b",
    ],

    "date": [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
        r"\b\d{1,2}\s+"
        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
        r"[a-z]*\s+\d{2,4}\b",
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
        r"[a-z]*\s+\d{1,2},?\s+\d{2,4}\b",
    ],

    "amount": [
        r"(?:₹|INR|\$|€|£)\s?\d+(?:[,.]\d+)*",
        r"\b(?:rs\.?|rupees)\s?\d+(?:[,.]\d+)*",
        r"\b\d+(?:[,.]\d+)*\s?(?:rupees|inr)\b",
    ],
}


# Person names are capitalisation sensitive, so they cannot use
# the case-insensitive ENTITY_PATTERNS above.
NAME_PATTERNS = [
    r"\b(?:mr|mrs|ms|miss|dr|shri|smt)\.?\s+"
    r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)",

    # The cue words are matched case-insensitively -- a customer writes
    # "I am Priya Sharma", not "i am Priya Sharma" -- while the captured
    # name stays capitalisation-sensitive, which is the whole reason
    # this pattern cannot use the case-insensitive ENTITY_PATTERNS.
    r"\b(?i:my name is|i am|i'm|this is|name is|"
    r"mera naam|mera nam|mi nombre es|me llamo|"
    r"naa peru|na peru)\s+"
    r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)",

    r"^(?:hi|hello|hey|namaste|hola|namaskaram)\s*,?\s+"
    r"([A-Z][a-zA-Z]+)\b",
]


# Capitalised words that are never customer names.
NAME_STOPWORDS = {
    "i",
    "the",
    "my",
    "order",
    "hi",
    "hello",
    "hey",
    "please",
    "thanks",
    "thank",
    "mr",
    "mrs",
    "ms",
    "dr",
    "inr",
    "ord",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}


def extract_person_names(text: str) -> List[str]:
    """
    Extract customer names from common introduction patterns.

    Deliberately conservative: only names introduced by a
    greeting, an introduction phrase or an honorific are
    captured, so ordinary capitalised words are not mistaken
    for people.
    """

    names: List[str] = []

    for pattern in NAME_PATTERNS:

        for match in re.findall(pattern, text):

            candidate = match.strip()

            if not candidate:
                continue

            first_word = candidate.split()[0]

            if first_word.lower() in NAME_STOPWORDS:
                continue

            if candidate not in names:
                names.append(candidate)

    return names


def extract_preserved_entities(
    text: str,
) -> Dict[str, List[str]]:
    """
    Extract identifiers that must survive
    language switching or translation.
    """

    entities = {}

    for entity_type, patterns in ENTITY_PATTERNS.items():

        matches = []

        for pattern in patterns:

            matches.extend(
                re.findall(
                    pattern,
                    text,
                    flags=re.IGNORECASE,
                )
            )

        entities[entity_type] = list(
            dict.fromkeys(matches)
        )

    entities["person_name"] = extract_person_names(
        text
    )

    return entities


def preserve_entities(
    original_text: str,
    response_text: str,
) -> Dict:
    """
    Verify important entities from the customer
    message remain unchanged in the response.
    """

    entities = extract_preserved_entities(
        original_text
    )

    missing = {}

    for entity_type, values in entities.items():

        for value in values:

            if value.lower() not in response_text.lower():

                missing.setdefault(
                    entity_type,
                    [],
                ).append(value)

    return {
        "entities": entities,
        "missing": missing,
        "preserved": not bool(missing),
    }


# ============================================================
# CONVERSATION MESSAGE
# ============================================================

@dataclass
class ConversationMessage:

    role: str

    content: str

    timestamp: datetime

    # Stable per message, and the only field that stays unique when the
    # clock is simulated: a frozen `now` gives consecutive messages the
    # same timestamp, so anything that needs to identify one message
    # cannot use the timestamp to do it. The UI keys a turn's support
    # detail -- sentiment, escalation, ticket -- by this, so the detail
    # can be found again after the rerun that follows every turn.
    message_id: str = field(
        default_factory=lambda: uuid.uuid4().hex
    )

    language: str = "unknown"

    confidence: float = 0.0

    entities: Dict[str, List[str]] = field(
        default_factory=dict
    )


# ============================================================
# CONVERSATION SESSION
# ============================================================

@dataclass
class ConversationSession:

    session_id: str

    created_at: datetime

    last_activity: datetime

    messages: List[ConversationMessage] = field(
        default_factory=list
    )

    summary: str = ""

    closed: bool = False

    # Set when this session was rebuilt from a previous one.
    restored_from_previous: bool = False

    previous_summary: str = ""

    def add_message(
        self,
        message: ConversationMessage,
        max_messages: int = 10,
    ) -> None:

        self.messages.append(message)

        # Keep only the latest N messages.
        self.messages = self.messages[
            -max_messages:
        ]

        self.last_activity = message.timestamp

    def context(
        self,
    ) -> List[ConversationMessage]:

        return list(self.messages)

    def inactivity_seconds(
        self,
        now: datetime,
    ) -> float:

        return max(
            0.0,
            (
                _as_utc(now)
                - _as_utc(self.last_activity)
            ).total_seconds(),
        )

    def user_messages(
        self,
    ) -> List[ConversationMessage]:

        return [
            message
            for message in self.messages
            if message.role == "user"
        ]

    def languages_used(self) -> List[str]:
        """
        Languages seen in this conversation, in order of
        first appearance. Used to report language switching.
        """

        seen: List[str] = []

        for message in self.user_messages():

            language = message.language

            if language in ("unknown", "", None):
                continue

            if language not in seen:
                seen.append(language)

        return seen

    def dominant_language(self) -> Optional[str]:
        """
        Most frequently used language in this conversation.
        Used to resolve very short or ambiguous follow-ups.
        """

        counts: Dict[str, int] = {}

        for message in self.user_messages():

            language = message.language

            if language in ("unknown", "", None):
                continue

            counts[language] = counts.get(language, 0) + 1

        if not counts:
            return None

        return max(
            counts.items(),
            key=lambda item: item[1],
        )[0]

    def last_user_language(self) -> Optional[str]:

        for message in reversed(self.user_messages()):

            if message.language not in ("unknown", "", None):

                return message.language

        return None


# ============================================================
# SESSION SUMMARY
# ============================================================

SUMMARY_MESSAGE_CHARS = 120


def build_session_summary(
    session: ConversationSession,
    max_messages: Optional[int] = None,
) -> str:
    """
    Fold a session's recent turns into one line of carried context.

    Extractive, not generative: it quotes whoever spoke rather than
    paraphrasing, so nothing is invented and no model call is needed.
    That matters because this string outlives the conversation it
    describes -- once the 10-message context window has pushed the
    original turns out, the summary is the only record of them, and it
    is read back by the escalation record and the handoff summary.

    ``max_messages`` defaults to the configured restore window, so the
    summary and the restore path agree on how much is kept.
    """

    if max_messages is None:

        max_messages = 4

    if max_messages <= 0:

        return ""

    messages = session.messages[-max_messages:]

    if not messages:

        return ""

    parts = []

    for message in messages:

        speaker = (
            "Customer" if message.role == "user" else "Support"
        )

        text = " ".join(message.content.split())

        if len(text) > SUMMARY_MESSAGE_CHARS:

            text = (
                text[: SUMMARY_MESSAGE_CHARS - 3].rstrip() + "..."
            )

        parts.append(f"{speaker}: {text}")

    return " | ".join(parts)


# ============================================================
# SESSION MANAGER
# ============================================================

class SessionManager:

    def __init__(
        self,
        config: Optional[Task6Config] = None,
    ):

        self.config = (
            config
            or Task6Config()
        )

        self.sessions: Dict[
            str,
            ConversationSession,
        ] = {}

    def create_session(
        self,
        session_id: str,
        now: Optional[datetime] = None,
    ) -> ConversationSession:

        now = (
            now
            or datetime.now(timezone.utc)
        )

        session = ConversationSession(
            session_id=session_id,
            created_at=now,
            last_activity=now,
        )

        self.sessions[session_id] = session

        return session

    def get_session(
        self,
        session_id: str,
        now: Optional[datetime] = None,
    ) -> Optional[ConversationSession]:

        now = (
            now
            or datetime.now(timezone.utc)
        )

        session = self.sessions.get(
            session_id
        )

        if session is None:

            return None

        inactivity = (
            session.inactivity_seconds(
                now
            )
        )

        if inactivity >= (
            self.config.inactivity_minutes
            * 60
        ):

            # Fold the conversation into a summary as it closes. This
            # is the 30-minute expiry doing the folding, because it is
            # the only point at which the app learns a session ended:
            # without it a restored session carries no summary at all,
            # and the escalation record and handoff have nothing to
            # show for the turns that have since left the context
            # window. An explicit summary set by the caller wins.
            if not session.closed and not session.summary:

                session.summary = build_session_summary(
                    session,
                    self.config.restore_context_messages,
                )

            session.closed = True

        return session

    def get_or_create(
        self,
        session_id: str,
        now: Optional[datetime] = None,
    ) -> ConversationSession:

        now = (
            now
            or datetime.now(timezone.utc)
        )

        existing = self.get_session(
            session_id,
            now,
        )

        if (
            existing is not None
            and not existing.closed
        ):

            return existing

        return self.create_session(
            session_id,
            now,
        )

    def add_user_message(
        self,
        session_id: str,
        content: str,
        now: Optional[datetime] = None,
    ) -> Dict:

        now = (
            now
            or datetime.now(timezone.utc)
        )

        session = self.get_or_create(
            session_id,
            now,
        )

        detection = detect_language(
            content,
            self.config,
        )

        entities = extract_preserved_entities(
            content
        )

        message = ConversationMessage(
            role="user",
            content=content,
            timestamp=now,
            language=detection["language"],
            confidence=detection["confidence"],
            entities=entities,
        )

        session.add_message(
            message,
            self.config.max_context_messages,
        )

        return {
            "session_id": session_id,
            "language": detection,
            "entities": entities,
            "context_size": len(
                session.messages
            ),
            "clarification_required": (
                detection[
                    "clarification_required"
                ]
            ),
        }

    def add_assistant_message(
        self,
        session_id: str,
        content: str,
        now: Optional[datetime] = None,
    ) -> ConversationMessage:
        """
        Append an assistant reply and hand it back, so the caller can
        hold on to its ``message_id``.
        """

        now = (
            now
            or datetime.now(timezone.utc)
        )

        session = self.get_or_create(
            session_id,
            now,
        )

        detection = detect_language(
            content,
            self.config,
        )

        message = ConversationMessage(
            role="assistant",
            content=content,
            timestamp=now,
            language=detection["language"],
            confidence=detection["confidence"],
            entities=extract_preserved_entities(
                content
            ),
        )

        session.add_message(
            message,
            self.config.max_context_messages,
        )

        return message

    def get_context(
        self,
        session_id: str,
        now: Optional[datetime] = None,
    ) -> Dict:

        session = self.get_session(
            session_id,
            now,
        )

        if session is None:

            return {
                "session_id": session_id,
                "active": False,
                "messages": [],
                "summary": "",
            }

        if session.closed:

            return {
                "session_id": session_id,
                "active": False,
                "messages": [],
                "summary": session.summary,
            }

        return {
            "session_id": session_id,
            "active": True,
            "messages": session.context(),
            "summary": session.summary,
        }

    def set_summary(
        self,
        session_id: str,
        summary: str,
    ) -> None:

        session = self.sessions.get(
            session_id
        )

        if session is not None:

            session.summary = summary

    def restore_session(
        self,
        session_id: str,
        now: Optional[datetime] = None,
    ) -> Dict:

        now = (
            now
            or datetime.now(timezone.utc)
        )

        session = self.sessions.get(
            session_id
        )

        if session is None:

            return {
                "restored": False,
                "reason": "No previous session.",
                "new_session_required": True,
            }

        age = (
            now - session.last_activity
        ).total_seconds()

        if age <= (
            self.config.restore_window_hours
            * 3600
        ):

            session.closed = False

            # A session restored without a summary -- one that was
            # never closed by the inactivity check, or was created
            # before this existed -- gets one built now, so the caller
            # always has something to hand to an escalation record.
            if not session.summary:

                session.summary = build_session_summary(
                    session,
                    self.config.restore_context_messages,
                )

            return {
                "restored": True,
                "reason": (
                    "Previous session is "
                    "within restore window."
                ),
                "summary": session.summary,
                "context_size": len(
                    session.messages
                ),
            }

        return {
            "restored": False,
            "reason": (
                "Restore window expired."
            ),
            "summary": "",
            "new_session_required": True,
        }


# ============================================================
# REQUEST ANALYSIS
# ============================================================

def detect_multiple_requests(
    text: str,
) -> bool:
    """
    Detect obvious multiple customer requests.
    """

    separators = [
        " and ",
        " also ",
        " además ",
        " y ",
        " तथा ",
        " और ",
    ]

    normalized = _normalize_text(
        text
    )

    return any(
        separator in normalized
        for separator in separators
    )


def detect_correction(text: str) -> bool:
    normalized = _normalize_text(text)

    correction_patterns = [
        r"\bactually\b",
        r"\bcorrect(?:ion)?\b",
        r"\bi meant\b",
        r"\bi mean\b",
        r"\bno[,\s]+i meant\b",
        r"\bsorry[,\s]+i gave you the wrong\b",
        r"\bi gave you the wrong\b",
        r"\bwrong order\b",
        r"\bwrong number\b",
        r"\bwrong date\b",
        r"\bwrong amount\b",
        r"\bincorrect order\b",
        r"\bincorrect number\b",
        r"\bincorrect date\b",
        r"\bincorrect amount\b",
    ]

    return any(
        re.search(pattern, normalized)
        for pattern in correction_patterns
    )



def build_language_instruction(
    language: str,
) -> str:

    display_name = LANGUAGE_NAMES.get(
        language,
        "the customer's language",
    )

    return (
        f"Respond in {display_name}. "
        "Preserve customer names, order IDs, dates, "
        "product codes, and other important identifiers "
        "exactly as provided. Do not translate identifiers."
    )


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    config = Task6Config()

    print("=" * 75)
    print(
        "TASK 6 MULTILINGUAL + SESSION MODULE SELF TEST"
    )
    print("=" * 75)

    tests = [

        "My order ORD-12345 needs a refund.",

        "मेरा order ORD-12345 refund चाहिए।",

        "నా order ORD-12345 refund కావాలి.",

        "Necesito un reembolso para mi pedido ORD-12345.",
    ]

    for text in tests:

        result = detect_language(
            text,
            config,
        )

        print("\nText:", text)
        print("Detection:", result)

        assert result["language"] in {
            "english",
            "hindi",
            "telugu",
            "spanish",
        }

    # Entity preservation test.
    entity_text = (
        "Please check order ORD-12345 "
        "for ₹1499 on 15/09/2026."
    )

    entities = extract_preserved_entities(
        entity_text
    )

    print("\nEntities:", entities)

    assert (
        "ORD-12345"
        in entities["order_id"]
    )

    assert (
        "₹1499"
        in entities["amount"]
    )

    assert (
        "15/09/2026"
        in entities["date"]
    )

    # Session test.
    manager = SessionManager(
        config
    )

    now = datetime(
        2026,
        10,
        2,
        10,
        0,
        tzinfo=timezone.utc,
    )

    # Add 12 messages.
    for index in range(12):

        manager.add_user_message(
            "session-a",
            f"Message {index}",
            now=now + timedelta(
                minutes=index
            ),
        )

    context = manager.get_context(
        "session-a",
        now=now + timedelta(
            minutes=12
        ),
    )

    print(
        "\nContext size:",
        len(context["messages"])
    )

    assert len(
        context["messages"]
    ) == 10

    # Session isolation test.
    manager.add_user_message(
        "session-b",
        "This belongs to another customer.",
        now=now,
    )

    context_a = manager.get_context(
        "session-a",
        now=now + timedelta(
            minutes=12
        ),
    )

    context_b = manager.get_context(
        "session-b",
        now=now,
    )

    assert len(
        context_a["messages"]
    ) == 10

    assert len(
        context_b["messages"]
    ) == 1

    # Summary restoration.
    manager.set_summary(
        "session-a",
        "Customer asked about order ORD-12345.",
    )

    restored = manager.restore_session(
        "session-a",
        now=now + timedelta(
            hours=1
        ),
    )

    print(
        "\nRestoration:",
        restored
    )

    assert restored["restored"] is True

    assert (
        "ORD-12345"
        in restored["summary"]
    )

    # Expired restoration.
    expired = manager.restore_session(
        "session-a",
        now=now + timedelta(
            hours=25
        ),
    )

    print(
        "\nExpired restoration:",
        expired
    )

    assert expired["restored"] is False

    assert (
        expired["new_session_required"]
        is True
    )

    print(
        "\nALL TASK 6 CORE SELF TESTS PASSED ✅"
    )
