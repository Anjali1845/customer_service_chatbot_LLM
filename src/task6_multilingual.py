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
- Unsupported-language clarification (French, German, etc. are
  flagged explicitly, not guessed into a supported language)
- Short follow-ups inherit the session's established language
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
import unicodedata
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
        # Romanised everyday words. "haan" in particular is the
        # single-word "yes" a Hindi-speaking customer sends as a
        # whole follow-up message.
        "haan",
        "haanji",
        "theek",
        "achha",
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
        # Romanised everyday words, so that a message like
        # "Naa order inka raaledu" is recognised as Telugu-heavy
        # instead of coming out as an English/Telugu coin flip.
        "avunu",
        "sare",
        "inka",
        "raaledu",
        "raavu",
        "vachinda",
        "vachindi",
        "tarvata",
        "ippudu",
        "konchem",
        "chala",
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


# Languages the product deliberately does NOT support, but that a
# customer may still write in. Words here are chosen to be
# unambiguous for their language -- nothing in this table is also a
# word in a supported language's keyword set -- because a single
# borrowed foreign word must never flag a whole message.
UNSUPPORTED_LANGUAGE_KEYWORDS = {

    "french": {
        "bonjour",
        "salut",
        "merci",
        "oui",
        "non",
        "je",
        "tu",
        "il",
        "elle",
        "nous",
        "vous",
        "ils",
        "suis",
        "est",
        "sont",
        "pas",
        "besoin",
        "veux",
        "veut",
        "voudrais",
        "mon",
        "ma",
        "mes",
        "votre",
        "commande",
        "remboursement",
        "rembourser",
        "colis",
        "livraison",
        "livre",
        "recu",
        "probleme",
        "aide",
        "avec",
        "pour",
        "une",
        "et",
        "de",
        "du",
        "le",
        "les",
        "des",
        "annule",
        "jamais",
        "toujours",
        "encore",
    },

    "german": {
        "bitte",
        "danke",
        "hallo",
        "guten",
        "nicht",
        "mein",
        "meine",
        "bestellung",
        "bestellt",
        "brauche",
        "hilfe",
        "entschuldigung",
        "ich",
        "mich",
        "mir",
        "sie",
        "und",
        "fur",
        "ist",
        "habe",
        "haben",
        "sehr",
        "warum",
        "wann",
        "wie",
        "wo",
        "noch",
        "auch",
        "aber",
        "kaputt",
        "defekt",
        "ruckgabe",
        "ruckzahlung",
        "schicken",
        "wunsche",
        "ware",
        "leid",
    },

    "italian": {
        "buongiorno",
        "grazie",
        "ciao",
        "ordine",
        "rimborso",
        "bisogno",
        "aiuto",
        "favore",
        "sono",
        "per",
        "mia",
        "mio",
        "del",
        "della",
        "arrivato",
        "pagamento",
        "problema",
        "vorrei",
        "voglio",
        "questo",
        "questa",
        "scusi",
        "signore",
        "signora",
        "il",
        "lo",
        "non",
        "ho",
    },

    "portuguese": {
        "obrigado",
        "obrigada",
        "ola",
        "chegou",
        "meu",
        "minha",
        "meus",
        "minhas",
        "quero",
        "preciso",
        "ajuda",
        "conta",
        "cancelar",
        "dinheiro",
        "voltar",
        "favor",
        "atrasado",
        "nao",
    },
}

# Scripts no supported language is written in. One character of one
# of these blocks is decisive evidence on its own.
UNSUPPORTED_SCRIPT_RANGES = {
    "chinese": (0x4E00, 0x9FFF),
    "japanese": (0x3040, 0x30FF),
    "korean": (0xAC00, 0xD7AF),
    "arabic": (0x0600, 0x06FF),
    "hebrew": (0x0590, 0x05FF),
    "russian": (0x0400, 0x04FF),
    "greek": (0x0370, 0x03FF),
    "thai": (0x0E00, 0x0E7F),
}

# Values that may sit in a message's ``language`` field without
# being a language the conversation can continue in.
NON_LANGUAGE_VALUES = ("unknown", "", None, "unsupported")


def _is_real_language(language) -> bool:
    """
    True when ``language`` names an actual language, rather than a
    detection failure ("unknown") or an explicit unsupported-language
    signal ("unsupported").
    """

    return language not in NON_LANGUAGE_VALUES


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


def _fold_diacritics(token: str) -> str:
    """
    Reduce a token to its base letters: "reçu" -> "recu",
    "für" -> "fur", "não" -> "nao".
    """

    decomposed = unicodedata.normalize(
        "NFKD",
        token,
    )

    return "".join(
        char
        for char in decomposed
        if not unicodedata.combining(char)
    )


def _unsupported_language_scores(
    text: str,
    tokens: List[str],
    normalized: str,
) -> Dict[str, float]:
    """
    Score evidence that the message is in a language the product
    does not support (French, German, and so on).

    Only exact keyword matches count, and a language needs two
    distinct hits before it counts at all -- fuzzy matching against a
    language we cannot even answer in would guess far too often, and
    one borrowed foreign word must never flag a whole message. Tokens
    are folded to their base letters first, so both "reçu" and "recu"
    match, whichever keyboard the customer typed on.
    """

    scores: Dict[str, float] = {}

    def add(
        language: str,
        amount: float,
    ) -> None:

        scores[language] = (
            scores.get(language, 0.0) + amount
        )

    folded = {
        _fold_diacritics(token)
        for token in tokens
    }

    for language, keywords in (
        UNSUPPORTED_LANGUAGE_KEYWORDS.items()
    ):

        matched = folded & keywords

        if len(matched) >= 2:

            add(
                language,
                float(len(matched)),
            )

    # A script no supported language uses is decisive on its own;
    # the cap keeps one long Chinese message from swamping the
    # numbers without meaning anything more than one character does.
    for language, (start, end) in (
        UNSUPPORTED_SCRIPT_RANGES.items()
    ):

        count = sum(
            1
            for char in text
            if start <= ord(char) <= end
        )

        if count:

            add(
                language,
                min(count, 3) * 2.0,
            )

    # Diacritics that belong to French and never to a supported
    # language. These can push a short accented message over the
    # keyword threshold, but never reach it alone.
    if re.search(
        r"[èàùçêôâîûëï]",
        normalized,
    ):

        add("french", 1.5)

    if "é" in normalized:

        add("french", 1.0)

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
        scores

    When the message is confidently a language the product does not
    support (French, German, and so on, or any script no supported
    language is written in), the language is "unsupported" with
    ``clarification_required`` True so the caller can ask, instead
    of guessing the closest supported language. The result then also
    carries ``unsupported`` (True) and ``unsupported_language`` (the
    name of the detected language).
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

    # Spanish-specific characters. "é" is deliberately excluded: it is
    # at least as common in French and Italian, and crediting it to
    # Spanish is what made French support messages come out labelled
    # "spanish" with enough confidence to be answered.
    if (
        "spanish" in scores
        and re.search(
            r"[áíóúñ¿¡]",
            normalized,
        )
    ):

        scores["spanish"] += 1.5

    # Unsupported-language evidence. This must be decided before the
    # low-confidence fallback: a French message with no English,
    # Hindi, Telugu or Spanish words has zero supported evidence, and
    # would otherwise fall through to "unknown" instead of the
    # explicit unsupported signal the caller needs in order to
    # clarify rather than guess.
    unsupported_scores = _unsupported_language_scores(
        text,
        tokens,
        normalized,
    )

    best_supported = (
        max(scores.values())
        if scores
        else 0.0
    )

    best_unsupported_language = None
    best_unsupported_score = 0.0

    for language, score in unsupported_scores.items():

        if score > best_unsupported_score:

            best_unsupported_language = language
            best_unsupported_score = score

    # The unsupported evidence must be real (two distinct keyword
    # hits, or one character of a script no supported language uses)
    # and must outweigh the supported evidence, so a supported
    # message is never flagged because it borrowed a foreign word.
    if (
        best_unsupported_score >= 2.0
        and best_unsupported_score > best_supported
    ):

        total = (
            sum(scores.values())
            + sum(unsupported_scores.values())
        )

        confidence = (
            best_unsupported_score / total
            if total > 0
            else 1.0
        )

        return {
            "language": "unsupported",
            "confidence": round(
                confidence,
                3,
            ),
            "mixed": False,
            "detected_languages": [
                best_unsupported_language
            ],
            "clarification_required": True,
            "unsupported": True,
            "unsupported_language": (
                best_unsupported_language
            ),
            "reason": (
                "Message appears to be "
                f"{best_unsupported_language}, "
                "which is not supported."
            ),
            "scores": {
                language: round(
                    score,
                    3,
                )
                for language, score in scores.items()
            },
        }

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


# An email address is never a person, and neither is a structured
# code. Both are masked out of the text before any name pattern
# runs, so the local part of an email ("anil" in "anil@x.com") can
# never be captured as, or glued onto, a customer name, and the
# fragment of a code can never be read as one.
EMAIL_PATTERN = (
    r"[A-Za-z0-9._%+\-]+@"
    r"[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"
)

# All-caps letters-dash-digits codes: ORD-1234, PRD-ABC123, SKU-9912.
# Case-sensitive on purpose -- "Anil-Rao" is a hyphenated surname,
# not a product code, and must survive as a name.
STRUCTURED_CODE_PATTERN = (
    r"\b[A-Z]{1,6}[-_][A-Z0-9]{2,12}\b"
)


# Person names are capitalisation sensitive, so they cannot use
# the case-insensitive ENTITY_PATTERNS above.
NAME_PATTERNS = [
    r"\b(?:mr|mrs|ms|miss|dr|shri|smt)\.?\s+"
    r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)",

    # The cue words are matched case-insensitively -- a customer writes
    # "I am Priya Sharma", not "i am Priya Sharma" -- while the captured
    # name stays capitalisation-sensitive, which is the whole reason
    # this pattern cannot use the case-insensitive ENTITY_PATTERNS.
    # A bare "name" is a cue too, so the labelled forms a customer
    # actually sends -- "Name Anil Rao", "name: Anil Rao",
    # "Full name: Anil Rao" -- are recognised; the separator then
    # tolerates the label's colon, comma or dash. The lookbehinds keep
    # "product name Premium Mixer" and friends from being read as a
    # customer: those label a thing, not a person.
    r"(?<!product )(?<!item )(?<!brand )"
    r"\b(?i:my name is|name is|customer name|full name|"
    r"i am|i'm|this is|"
    r"mera naam|mera nam|mi nombre es|me llamo|"
    r"naa peru|na peru|name)\b"
    r"[\s:,\-]+"
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
    greeting, an introduction phrase, a label ("Name Anil Rao",
    "name: Anil Rao") or an honorific are captured, so ordinary
    capitalised words are not mistaken for people.

    Email addresses and structured codes (order ids, product
    codes) are masked out before any pattern runs: an email
    address must never yield a person name, and neither must the
    words around an identifier.
    """

    names: List[str] = []

    masked = re.sub(
        EMAIL_PATTERN,
        " ",
        text,
    )

    masked = re.sub(
        STRUCTURED_CODE_PATTERN,
        " ",
        masked,
    )

    for pattern in NAME_PATTERNS:

        for match in re.findall(pattern, masked):

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


def _is_identifier_only(text: str) -> bool:
    """
    True when the message carries an identifier or a number but no
    language signal at all: "ORD-1234", "123456789", "PRD-ABC123".

    Such a message is the natural answer to "what is your order
    ID?" -- it is never a language problem, so asking which language
    the customer prefers is always the wrong reply to it.
    """

    stripped = text.strip()

    if not stripped:

        return False

    remainder = stripped
    has_identifier = False

    for patterns in ENTITY_PATTERNS.values():

        for pattern in patterns:

            remainder, replaced = re.subn(
                pattern,
                " ",
                remainder,
                flags=re.IGNORECASE,
            )

            if replaced:

                has_identifier = True

    if re.search(r"\d", stripped):

        has_identifier = True

    if not has_identifier:

        return False

    # Whatever survives the entity patterns must contain no letters
    # in any script for the message to be identifier-only.
    return not re.search(
        r"[^\W\d_]",
        remainder,
        flags=re.UNICODE,
    )


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

            if not _is_real_language(language):
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

            if not _is_real_language(language):
                continue

            counts[language] = counts.get(language, 0) + 1

        if not counts:
            return None

        return max(
            counts.items(),
            key=lambda item: item[1],
        )[0]

    def last_user_language(self) -> Optional[str]:
        """
        The language of the most recent user message that was
        detected as a real language. Follow-ups too short to detect
        anything from inherit this, which is what makes "yes",
        "ORD-1234" or "Haan" continue the conversation instead of
        tripping the low-confidence clarification.
        """

        for message in reversed(self.user_messages()):

            if _is_real_language(message.language):

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

    def _resolve_followup_language(
        self,
        session: ConversationSession,
        content: str,
        detection: Dict,
    ) -> Dict:
        """
        Resolve a short or ambiguous follow-up against the language
        the session already carries.

        In a live conversation the replies "yes", "ok thanks",
        "Haan" or a bare "ORD-1234" carry no language signal of their
        own, but they are perfectly clear answers -- "ORD-1234" is
        exactly what the assistant asked for when it requested the
        order ID. Flagging them as unknown, and answering with the
        "please rephrase or tell me your preferred language"
        clarification, rejects the customer's answer to the
        assistant's own question, so they continue the conversation
        in the language already established by ``last_user_language``
        and ``dominant_language``.

        A message confidently in a language the product does not
        support is never inherited past: the customer switched to
        French, and that still needs to be clarified, not guessed.
        """

        if not detection.get("clarification_required"):

            return detection

        if (
            detection.get("unsupported")
            or detection.get("language") == "unsupported"
        ):

            return detection

        identifier_only = _is_identifier_only(content)

        tokens = _tokenize(content)

        # A message with neither words nor identifiers -- empty, or
        # bare punctuation -- has nothing to inherit from: it stays a
        # clarification no matter what the session already holds.
        if not tokens and not identifier_only:

            return detection

        # A long unrecognisable message keeps the clarification: only
        # a short follow-up or a bare identifier is plausibly a clear
        # answer that merely carries no language of its own.
        if not identifier_only and len(tokens) > 4:

            return detection

        session_language = (
            session.last_user_language()
            or session.dominant_language()
        )

        if session_language:

            resolved = dict(detection)

            resolved.update(
                {
                    "language": session_language,
                    "confidence": max(
                        detection.get(
                            "confidence",
                            0.0,
                        ),
                        0.75,
                    ),
                    "clarification_required": False,
                    "inherited_from_session": True,
                    "identifier_only": identifier_only,
                    "reason": (
                        "Short or ambiguous follow-up; continuing "
                        "in the session's established language."
                    ),
                }
            )

            return resolved

        if identifier_only:

            # No session language to inherit and no language signal
            # in the message either: an identifier-only message is
            # still a valid in-context reply, never a low-confidence
            # one, so it is answered in the default language.
            resolved = dict(detection)

            resolved.update(
                {
                    "language": "english",
                    "confidence": 0.75,
                    "clarification_required": False,
                    "inherited_from_session": False,
                    "identifier_only": True,
                    "reason": (
                        "Identifier-only message with no session "
                        "language; using the default language."
                    ),
                }
            )

            return resolved

        return detection

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

        # The session's existing language state resolves short
        # follow-ups and bare identifiers before anything is stored,
        # so the stored message and the returned detection agree.
        detection = self._resolve_followup_language(
            session,
            content,
            detection,
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
