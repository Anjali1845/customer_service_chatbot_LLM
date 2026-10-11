"""
TASK 8.1 - TICKETING

Turns an unresolved conversation into a structured, prioritised,
SLA-tracked ticket.

Design rules this module follows, and why:

* **Nothing is invented.** Extraction reuses the Task 5 extractors and
  the Task 6 entity patterns; a field that is not present stays None.
  A missing mandatory field becomes an information request, never a
  guess.

* **Deadlines are derived, never stored as truth.** ``due_at`` and
  ``warn_at`` are always recomputed from ``created_at`` plus the
  current SlaPolicy, so changing the policy at runtime moves existing
  tickets -- including flipping a comfortable ticket into breach.
  ``recalculate_deadlines`` is that recompute, and it is pure.

* **SLA time is business time.** Weekends and holidays are excluded
  through the shared business calendar, so a ticket raised on Friday
  evening is not "overdue" on Saturday morning.

* **Structured fields carry the customer's data; derived text does
  not.** The ticket keeps the real contact so an agent can reply, but
  every summary that leaves the ticket -- the handoff block, the log
  line -- is masked first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

from business_calendar import (
    BusinessCalendarConfig,
    add_business_minutes,
    elapsed_business_minutes,
    next_business_start,
)
from support_deps import (
    as_utc,
    extract_message_fields,
    extract_structured_data,
    mask_sensitive_text,
)
from task6_multilingual import (
    detect_multiple_requests,
    extract_preserved_entities,
)
from task7_sentiment import SentimentResult


# ============================================================
# CONSTANTS
# ============================================================

TICKET_CATEGORIES = (
    "payment",
    "account",
    "delivery",
    "product",
    "technical",
    "general",
)

TICKET_STATUSES = (
    "new",
    "awaiting_info",
    "open",
    "assigned",
    "in_progress",
    "resolved",
    "closed",
    "duplicate",
)

CLOSED_STATUSES = ("resolved", "closed")

PRIORITY_ORDER = ("P1", "P2", "P3", "P4")


# Severity carried by a sentiment label when no risk topic outranks it.
SEVERITY_FROM_LABEL: Dict[str, str] = {
    "urgent": "high",
    "frustrated": "medium",
    "sarcastic": "medium",
    "negative": "medium",
    "positive": "low",
    "neutral": "low",
}

SEVERITY_ORDER: Dict[str, int] = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


# Category keyword cues. English plus romanised and native variants,
# so a Hindi or Telugu message routes as reliably as an English one.
CATEGORY_KEYWORDS: Dict[str, Tuple[str, ...]] = {

    "payment": (
        "refund", "charge", "charged", "payment", "paid", "billing",
        "invoice", "money", "amount", "deducted", "debited",
        "paisa", "paise", "bhugtan", "paisa kat", "paise kat",
        "पैसे", "पैसा", "भुगतान", "रिफंड", "चार्ज",
        "డబ్బు", "చెల్లింపు", "రిఫండ్",
        "pago", "reembolso", "cobro", "factura", "dinero",
    ),

    "account": (
        "account", "login", "log in", "sign in", "password", "otp",
        "hacked", "locked", "profile", "credentials", "unauthorised",
        "unauthorized", "verification",
        "खाता", "अकाउंट", "पासवर्ड", "लॉगिन",
        "ఖాతా", "అకౌంట్", "పాస్వర్డ్",
        "cuenta", "contraseña", "sesión", "sesion",
    ),

    "delivery": (
        "delivery", "deliver", "shipped", "shipping", "courier",
        "tracking", "parcel", "package", "not arrived", "late",
        "dispatch", "transit",
        "डिलीवरी", "पहुंचा", "भेजा", "ऑर्डर नहीं आया",
        "డెలివరీ", "పంప", "రాలేదు",
        "entrega", "envío", "envio", "paquete", "llega",
    ),

    "product": (
        "product", "item", "damaged", "broken", "defective", "wrong item",
        "quality", "size", "colour", "color", "missing item",
        "उत्पाद", "सामान", "खराब", "टूटा",
        "ఉత్పత్తి", "వస్తువు", "పాడై",
        "producto", "artículo", "articulo", "dañado", "roto",
    ),

    "technical": (
        "error", "bug", "crash", "not working", "app", "website",
        "technical", "glitch", "failed", "failure", "down",
        "त्रुटि", "काम नहीं", "ऐप",
        "లోపం", "పని చేయడం", "యాప్",
        "error", "fallo", "aplicación", "aplicacion", "no funciona",
    ),
}


# How a genuine category tie is resolved. Ordered by the cost of
# misrouting: a billing or security problem sent to the general queue
# loses money and trust, a general enquiry sent to billing costs a
# reassignment.
CATEGORY_TIE_ORDER: Tuple[str, ...] = (
    "payment",
    "account",
    "delivery",
    "product",
    "technical",
    "general",
)


# Mandatory fields per category. A ticket cannot be routed or
# SLA-tracked until these are known.
CATEGORY_MANDATORY_FIELDS: Dict[str, Tuple[str, ...]] = {
    "payment": ("customer_name", "contact", "order_id", "amount"),
    "account": ("customer_name", "contact", "account_identifier"),
    "delivery": ("customer_name", "contact", "order_id"),
    "product": ("customer_name", "contact", "order_id", "product_code"),
    "technical": ("customer_name", "contact", "error_code"),
    "general": ("customer_name", "contact"),
}


EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)

PHONE_PATTERN = re.compile(
    r"(?:(?:\+|00)\d{1,3}[\s\-]?)?(?:\(?\d{3,5}\)?[\s\-]?)\d{3}[\s\-]?\d{3,4}\b"
)

STOPWORDS = frozenset(
    """
    a an and are as at be been but by can could did do does for from
    had has have how i if in is it its me my not of on or our please
    so than that the their them then there these they this to too was
    we were what when where which who why will with you your
    """.split()
)

# Separators used to split a message carrying more than one request.
ISSUE_SEPARATORS = (
    " and also ", " and ", " also ", " plus ", " as well as ",
    " además ", " ademas ", " y ", " तथा ", " और ", " మరియు ",
)


# ============================================================
# POLICIES
# ============================================================

@dataclass
class SlaPolicy:
    """
    The runtime-mutable SLA definition.

    Changing any of these fields and recomputing is the supported way
    to apply a new SLA rule; nothing caches a derived deadline.
    """

    first_response_minutes: float = 30

    resolution_minutes_by_priority: Dict[str, float] = field(
        default_factory=lambda: {
            "P1": 60,
            "P2": 240,
            "P3": 480,
            "P4": 1440,
        }
    )

    # Warn once this fraction of the budget is spent.
    warn_fraction: float = 0.75

    exclude_weekends: bool = True
    exclude_holidays: bool = True

    auto_escalate_on_breach: bool = True

    calendar: BusinessCalendarConfig = field(
        default_factory=BusinessCalendarConfig
    )

    def minutes_for(self, priority: Optional[str]) -> float:
        """
        The resolution budget for a priority band.

        An unknown or missing band falls back to the highest budget
        rather than to the lowest, so a config mistake cannot make
        every ticket look instantly overdue.
        """

        if not priority:

            return max(self.resolution_minutes_by_priority.values())

        if priority in self.resolution_minutes_by_priority:

            return self.resolution_minutes_by_priority[priority]

        return max(self.resolution_minutes_by_priority.values())

    def calendar_for_sla(self) -> BusinessCalendarConfig:
        """
        The calendar with this policy's weekend/holiday toggles
        applied. Returned as a copy so the policy's own calendar is
        not mutated by a caller.
        """

        return BusinessCalendarConfig(
            timezone_name=self.calendar.timezone_name,
            working_days=self.calendar.working_days,
            business_hours_start=self.calendar.business_hours_start,
            business_hours_end=self.calendar.business_hours_end,
            holidays=self.calendar.holidays,
            exclude_weekends=self.exclude_weekends,
            exclude_holidays=self.exclude_holidays,
            on_call_hours_start=self.calendar.on_call_hours_start,
            on_call_hours_end=self.calendar.on_call_hours_end,
        )


@dataclass
class TicketPolicy:
    """
    Everything about how a ticket is categorised, validated and
    prioritised. Mutable for the same reason as SlaPolicy.
    """

    categories: Tuple[str, ...] = TICKET_CATEGORIES

    mandatory_fields: Dict[str, Tuple[str, ...]] = field(
        default_factory=lambda: dict(CATEGORY_MANDATORY_FIELDS)
    )

    # Priority = severity*w + sentiment*w + waiting*w + impact*w.
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "severity": 0.40,
            "sentiment": 0.25,
            "waiting": 0.20,
            "impact": 0.15,
        }
    )

    severity_scores: Dict[str, float] = field(
        default_factory=lambda: {
            "critical": 1.0,
            "high": 0.8,
            "medium": 0.5,
            "low": 0.2,
            "none": 0.1,
        }
    )

    sentiment_scores: Dict[str, float] = field(
        default_factory=lambda: {
            "urgent": 1.0,
            "frustrated": 0.85,
            "sarcastic": 0.70,
            "negative": 0.60,
            "neutral": 0.20,
            "positive": 0.05,
        }
    )

    category_impact: Dict[str, float] = field(
        default_factory=lambda: {
            "account": 1.0,
            "payment": 0.9,
            "technical": 0.7,
            "delivery": 0.6,
            "product": 0.5,
            "general": 0.3,
        }
    )

    # Highest band whose threshold the score reaches.
    #
    # Calibrated to the range the weighted sum actually produces, not
    # to 0..1 in the abstract. A freshly created ticket has no waiting
    # time yet, so its ceiling is severity + sentiment + impact --
    # roughly 0.80 for a critical, urgent, account-level issue and
    # 0.18 for a polite general enquiry. Bands set against an
    # unreachable 0.85 would leave P1 and P2 permanently empty.
    # Waiting time then lifts a ticket through the bands as it ages,
    # which is what the brief asks for.
    priority_bands: Tuple[Tuple[float, str], ...] = (
        (0.68, "P1"),
        (0.48, "P2"),
        (0.30, "P3"),
        (0.00, "P4"),
    )

    # Any critical risk topic forces this band regardless of score.
    critical_override_priority: str = "P1"

    # Waiting is measured against this many business minutes.
    target_resolution_minutes: float = 1440

    # Duplicate / related detection.
    # A duplicate is the same problem on the same order, so the
    # arithmetic is: 0.50 for the shared order id, 0.10 for the shared
    # category, plus up to 0.40 for how alike the two descriptions
    # are. 0.70 therefore asks for a shared order plus a roughly
    # quarter-overlapping description -- which is what a customer
    # re-reporting the same fault actually produces. At 0.75 a real
    # duplicate written in different words ("charged twice on ORD-1
    # again" vs "my card was charged twice ... order ORD-1") scored
    # 0.746 and was filed as merely related.
    duplicate_threshold: float = 0.70
    related_threshold: float = 0.40
    duplicate_weight_order_id: float = 0.50
    duplicate_weight_product_code: float = 0.30
    duplicate_weight_contact: float = 0.20
    duplicate_weight_category: float = 0.10
    duplicate_weight_similarity: float = 0.40
    minimum_token_overlap: int = 2

    def priority_for_score(self, score: float) -> str:
        """
        The band a score falls into.
        """

        for threshold, priority in self.priority_bands:

            if score >= threshold:

                return priority

        return self.priority_bands[-1][1]


# ============================================================
# TICKET
# ============================================================

@dataclass
class Ticket:
    """
    One structured support ticket.
    """

    ticket_id: str

    session_id: str = ""
    category: str = "general"
    status: str = "new"

    priority: str = "P4"
    priority_score: float = 0.0
    priority_components: Dict[str, float] = field(default_factory=dict)
    severity: str = "low"

    issue: str = ""

    # Extracted facts. None means "the customer did not say", which is
    # what drives an information request.
    customer_name: Optional[str] = None
    contact: Optional[str] = None
    account_identifier: Optional[str] = None
    order_id: Optional[str] = None
    product_name: Optional[str] = None
    product_code: Optional[str] = None
    amount: Optional[str] = None
    error_code: Optional[str] = None
    reference_date: Optional[str] = None

    evidence: List[str] = field(default_factory=list)
    missing_fields: List[str] = field(default_factory=list)

    language: str = "english"
    sentiment_label: str = "neutral"
    sentiment_confidence: float = 0.0
    risk_topics: List[str] = field(default_factory=list)

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None

    # Derived from created_at + SlaPolicy. Never authoritative on
    # their own -- recalculate_deadlines() regenerates them.
    due_at: Optional[datetime] = None
    warn_at: Optional[datetime] = None
    sla_minutes: float = 0.0
    warn_fraction: float = 0.75
    breached: bool = False
    warning_due: bool = False
    elapsed_business_minutes: float = 0.0

    duplicate_of: Optional[str] = None
    group_id: Optional[str] = None
    linked_tickets: List[str] = field(default_factory=list)

    assigned_agent: Optional[str] = None
    team: Optional[str] = None
    routing_reason: str = ""
    deferred_until: Optional[datetime] = None

    # An explicit impact override, when a caller knows more than the
    # category does.
    customer_impact: Optional[float] = None

    masked_handoff: str = ""
    conversation_summary: str = ""

    @property
    def is_closed(self) -> bool:

        return self.status in CLOSED_STATUSES

    @property
    def is_duplicate(self) -> bool:

        return self.duplicate_of is not None

    def masked_contact(self) -> str:
        """
        The contact field, safe to print in a log or a handoff.
        """

        return mask_sensitive_text(self.contact or "") or "not provided"

    def to_dict(self) -> Dict[str, Any]:
        """
        JSON-friendly form. Datetimes become ISO strings so a
        simulated clock round-trips through SQLite exactly.
        """

        def stamp(value: Optional[datetime]) -> Optional[str]:

            return as_utc(value).isoformat() if value else None

        return {
            "ticket_id": self.ticket_id,
            "session_id": self.session_id,
            "category": self.category,
            "status": self.status,
            "priority": self.priority,
            "priority_score": self.priority_score,
            "priority_components": dict(self.priority_components),
            "severity": self.severity,
            "issue": self.issue,
            "customer_name": self.customer_name,
            "contact": self.contact,
            "account_identifier": self.account_identifier,
            "order_id": self.order_id,
            "product_name": self.product_name,
            "product_code": self.product_code,
            "amount": self.amount,
            "error_code": self.error_code,
            "reference_date": self.reference_date,
            "evidence": list(self.evidence),
            "missing_fields": list(self.missing_fields),
            "language": self.language,
            "sentiment_label": self.sentiment_label,
            "sentiment_confidence": self.sentiment_confidence,
            "risk_topics": list(self.risk_topics),
            "created_at": stamp(self.created_at),
            "updated_at": stamp(self.updated_at),
            "resolved_at": stamp(self.resolved_at),
            "due_at": stamp(self.due_at),
            "warn_at": stamp(self.warn_at),
            "sla_minutes": self.sla_minutes,
            "warn_fraction": self.warn_fraction,
            "breached": self.breached,
            "warning_due": self.warning_due,
            "elapsed_business_minutes": self.elapsed_business_minutes,
            "duplicate_of": self.duplicate_of,
            "group_id": self.group_id,
            "linked_tickets": list(self.linked_tickets),
            "assigned_agent": self.assigned_agent,
            "team": self.team,
            "routing_reason": self.routing_reason,
            "deferred_until": stamp(self.deferred_until),
            "customer_impact": self.customer_impact,
            "masked_handoff": self.masked_handoff,
            "conversation_summary": self.conversation_summary,
        }


# ============================================================
# EXTRACTION
# ============================================================

def extract_contact(text: str) -> Optional[str]:
    """
    Email or phone, whichever appears first. Returns None when the
    customer has not given one.
    """

    if not text:

        return None

    email = EMAIL_PATTERN.search(text)

    if email:

        return email.group(0)

    phone = PHONE_PATTERN.search(text)

    if phone:

        return phone.group(0).strip()

    return None


# The prefix must be a token of its own: without the lookahead, "the
# product code is SKU-9912" matches the "PROD" inside "product" and
# yields "PROD-UCT".
PRODUCT_CODE_PATTERN = re.compile(
    r"\b(SKU|PRD|PROD|ITEM)(?![A-Za-z_])[\s\-_]*"
    r"([A-Z0-9][A-Z0-9\-_]{2,})\b",
    re.IGNORECASE,
)


def extract_product_code(text: str) -> Optional[str]:
    """
    A product code, keeping its manufacturer prefix.

    The shared Task 5 extractor returns "9912" for "the product code
    is SKU-9912" but "SKU-9912" for "product code SKU-9912", so the
    same part is identified two different ways depending on the
    sentence. Task 6 requires product codes to survive intact, so the
    prefixed form is matched directly here and wins when present.
    """

    if not text:

        return None

    match = PRODUCT_CODE_PATTERN.search(text)

    if not match:

        return None

    return f"{match.group(1).upper()}-{match.group(2).upper()}"


def extract_account_identifier(text: str) -> Optional[str]:
    """
    An account or customer number, which none of the shared
    extractors cover.
    """

    if not text:

        return None

    # The connector between the label and the value is either
    # punctuation ("account: ACC-1") or a copula ("my account number
    # is ACC-1"), which is how customers actually write it.
    connector = r"\s*(?:is|are|was|=|:|-)?\s*"

    patterns = (
        r"\b(?:account|acct|a/c)\s*(?:id|number|no\.?|#)?"
        + connector
        + r"([A-Z0-9][A-Z0-9\-_]{3,})\b",
        r"\b(?:customer|user)\s*(?:id|number|no\.?|#)"
        + connector
        + r"([A-Z0-9][A-Z0-9\-_]{3,})\b",
        r"\b(?:खाता|अकाउंट)\s*(?:संख्या|नंबर)?" + connector
        + r"([A-Z0-9][A-Z0-9\-_]{3,})\b",
        r"\b(?:ఖాతా)\s*(?:నంబర్)?" + connector
        + r"([A-Z0-9][A-Z0-9\-_]{3,})\b",
        r"\b(?:cuenta)\s*(?:número|numero)?" + connector
        + r"([A-Z0-9][A-Z0-9\-_]{3,})\b",
    )

    for pattern in patterns:

        match = re.search(pattern, text, re.IGNORECASE)

        if match:

            return match.group(1)

    return None


def extract_ticket_fields(
    text: str,
) -> Dict[str, Optional[str]]:
    """
    Every structured field a ticket can carry.

    Three existing extractors are combined -- the Task 5 structured
    extractor, the Task 5 message extractor and the Task 6 entity
    patterns -- because each covers a slightly different phrasing.
    Nothing is inferred: a field that no extractor finds stays None.
    """

    text = text or ""

    structured = extract_structured_data(text) or {}

    message = extract_message_fields(text) or {}

    entities = extract_preserved_entities(text) or {}

    def first(name: str, *sources: Dict[str, Any]) -> Optional[str]:

        for source in sources:

            value = source.get(name)

            if value:

                return value

        values = entities.get(name) or []

        return values[0] if values else None

    return {
        "order_id": first("order_id", structured, message),
        "product_code": (
            extract_product_code(text)
            or first("product_code", structured, message)
        ),
        "product_name": first("product_name", structured),
        "amount": first("amount", structured, message),
        "error_code": first("error_code", structured, message),
        "reference_date": first("date", structured, message),
        "customer_name": first("person_name"),
        "account_identifier": extract_account_identifier(text),
        "contact": extract_contact(text),
    }


def infer_category(
    text: str,
    fields: Optional[Dict[str, Any]] = None,
    risk_topics: Optional[Sequence[str]] = None,
) -> str:
    """
    The ticket category.

    Risk topics win, because they are the most reliable signal: a
    duplicate charge is a payment issue whatever words surround it.
    Otherwise the category with the most keyword cues wins, and a
    tie falls back to "general" rather than to an arbitrary pick.
    """

    topics = set(risk_topics or ())

    if topics & {"duplicate_payment", "unauthorized_charge"}:

        return "payment"

    if topics & {"account_compromise", "data_breach"}:

        return "account"

    lowered = (text or "").lower()

    scores: Dict[str, int] = {}

    for category, keywords in CATEGORY_KEYWORDS.items():

        hits = sum(1 for keyword in keywords if keyword in lowered)

        if hits:

            scores[category] = hits

    if scores:

        best = max(scores.values())

        winners = sorted(
            category
            for category, value in scores.items()
            if value == best
        )

        if len(winners) == 1:

            return winners[0]

        # A tie is broken by the structured evidence on the ticket.
        fields = fields or {}

        if fields.get("order_id") or fields.get("amount"):

            return "payment"

        if fields.get("error_code"):

            return "technical"

        # Otherwise fall back to a fixed order rather than to
        # "general". A tie is common and legitimate -- "my refund has
        # not arrived" cues payment and delivery equally -- and
        # routing a real billing or security problem into the general
        # queue is by far the more expensive mistake.
        for category in CATEGORY_TIE_ORDER:

            if category in winners:

                return category

    return "general"


# ============================================================
# MANDATORY FIELDS
# ============================================================

def missing_mandatory_fields(
    ticket: Ticket,
    policy: Optional[TicketPolicy] = None,
) -> List[str]:
    """
    The required fields this ticket does not have yet.
    """

    policy = policy or TicketPolicy()

    required = policy.mandatory_fields.get(
        ticket.category, policy.mandatory_fields.get("general", ())
    )

    missing: List[str] = []

    for name in required:

        value = getattr(ticket, name, None)

        if value is None or (isinstance(value, str) and not value.strip()):

            missing.append(name)

    return missing


FIELD_PROMPTS: Dict[str, Dict[str, str]] = {

    "english": {
        "customer_name": "your full name",
        "contact": "an email address or phone number we can reach you on",
        "order_id": "your order ID (it looks like ORD-12345)",
        "amount": "the amount involved",
        "product_code": "the product code from the item label",
        "account_identifier": "your account or customer number",
        "error_code": "the error code shown on screen",
    },

    "hindi": {
        "customer_name": "आपका पूरा नाम",
        "contact": "आपका ईमेल पता या फ़ोन नंबर",
        "order_id": "आपकी ऑर्डर आईडी (जैसे ORD-12345)",
        "amount": "संबंधित राशि",
        "product_code": "उत्पाद का कोड",
        "account_identifier": "आपका खाता या ग्राहक नंबर",
        "error_code": "स्क्रीन पर दिखने वाला त्रुटि कोड",
    },

    "telugu": {
        "customer_name": "మీ పూర్తి పేరు",
        "contact": "మీ ఇమెయిల్ లేదా ఫోన్ నంబర్",
        "order_id": "మీ ఆర్డర్ ఐడీ (ఉదా. ORD-12345)",
        "amount": "సంబంధిత మొత్తం",
        "product_code": "ఉత్పత్తి కోడ్",
        "account_identifier": "మీ ఖాతా లేదా కస్టమర్ నంబర్",
        "error_code": "స్క్రీన్పై కనిపించే ఎర్రర్ కోడ్",
    },

    "spanish": {
        "customer_name": "su nombre completo",
        "contact": "un correo electrónico o teléfono de contacto",
        "order_id": "su número de pedido (por ejemplo ORD-12345)",
        "amount": "el importe afectado",
        "product_code": "el código del producto",
        "account_identifier": "su número de cuenta o de cliente",
        "error_code": "el código de error que aparece en pantalla",
    },
}

INFORMATION_REQUEST_HEADERS: Dict[str, str] = {
    "english": (
        "I have logged this for you, but I need a few details before "
        "I can take it further:"
    ),
    "hindi": (
        "मैंने यह दर्ज कर लिया है, लेकिन आगे बढ़ाने से पहले मुझे कुछ "
        "जानकारी चाहिए:"
    ),
    "telugu": (
        "నేను దీన్ని నమోదు చేసాను, కానీ ముందుకు వెళ్లడానికి కొన్ని "
        "వివరాలు అవసరం:"
    ),
    "spanish": (
        "He registrado su solicitud, pero necesito algunos datos "
        "antes de continuar:"
    ),
}


def build_information_request(
    ticket: Ticket,
    language: Optional[str] = None,
    policy: Optional[TicketPolicy] = None,
) -> str:
    """
    The message appended to the reply when a ticket is missing
    mandatory information.

    Returns an empty string when nothing is missing, so a caller can
    append unconditionally.
    """

    missing = ticket.missing_fields or missing_mandatory_fields(
        ticket, policy
    )

    if not missing:

        return ""

    language = language or ticket.language or "english"

    if language not in FIELD_PROMPTS:

        language = "english"

    prompts = FIELD_PROMPTS[language]

    header = INFORMATION_REQUEST_HEADERS.get(
        language, INFORMATION_REQUEST_HEADERS["english"]
    )

    lines = [
        f"- {prompts.get(name, name.replace('_', ' '))}"
        for name in missing
    ]

    return (
        f"{header}\n"
        + "\n".join(lines)
        + f"\n(Ticket {ticket.ticket_id})"
    )


# ============================================================
# SEVERITY AND PRIORITY
# ============================================================

def derive_severity(sentiment: Optional[SentimentResult]) -> str:
    """
    The worst of the label severity and the risk severity.

    Risk wins when it is higher, which is what makes a calmly-worded
    duplicate-charge complaint a high-severity ticket.
    """

    if sentiment is None:

        return "low"

    label_severity = SEVERITY_FROM_LABEL.get(
        sentiment.label, "medium"
    )

    if SEVERITY_ORDER.get(
        sentiment.risk_severity, 0
    ) > SEVERITY_ORDER.get(label_severity, 0):

        return sentiment.risk_severity

    return label_severity


def score_priority(
    ticket: Ticket,
    policy: Optional[TicketPolicy] = None,
    sla_policy: Optional[SlaPolicy] = None,
    now: Optional[datetime] = None,
) -> Tuple[float, str, Dict[str, float]]:
    """
    Priority = 0.40*severity + 0.25*sentiment + 0.20*waiting +
    0.15*impact.

    Waiting is business minutes since the ticket was created, against
    a configurable target, so a ticket raised before a weekend does
    not silently climb while nobody is working.

    A critical risk topic forces the configured top band regardless of
    the arithmetic -- an active account compromise is never P4 because
    the customer was polite.
    """

    policy = policy or TicketPolicy()

    sla_policy = sla_policy or SlaPolicy()

    components: Dict[str, float] = {}

    components["severity"] = policy.severity_scores.get(
        ticket.severity, 0.5
    )

    components["sentiment"] = policy.sentiment_scores.get(
        ticket.sentiment_label, 0.2
    )

    target = policy.target_resolution_minutes or 1440.0

    if ticket.created_at and sla_policy:

        elapsed = elapsed_business_minutes(
            ticket.created_at,
            as_utc(now),
            sla_policy.calendar_for_sla(),
        )

    else:

        elapsed = 0.0

    components["waiting"] = min(1.0, elapsed / target)

    if ticket.customer_impact is not None:

        components["impact"] = max(
            0.0, min(1.0, float(ticket.customer_impact))
        )

    else:

        components["impact"] = policy.category_impact.get(
            ticket.category, 0.3
        )

    score = sum(
        policy.weights.get(name, 0.0) * value
        for name, value in components.items()
    )

    score = round(min(1.0, max(0.0, score)), 4)

    priority = policy.priority_for_score(score)

    # A critical risk topic overrides the band.
    if "critical" == ticket.severity or any(
        topic in ("account_compromise", "legal_threat", "data_breach")
        for topic in ticket.risk_topics
    ):

        priority = policy.critical_override_priority

    return score, priority, components


# ============================================================
# SLA
# ============================================================

def apply_sla(
    ticket: Ticket,
    sla_policy: Optional[SlaPolicy] = None,
    now: Optional[datetime] = None,
) -> Ticket:
    """
    Recompute this ticket's deadlines and SLA state from
    ``created_at`` and the current policy, in place.

    Because everything is derived, calling this again after a policy
    change is all that "the SLA rules changed at runtime" requires.
    """

    sla_policy = sla_policy or SlaPolicy()

    calendar = sla_policy.calendar_for_sla()

    moment = as_utc(now)

    created = as_utc(ticket.created_at) if ticket.created_at else moment

    ticket.created_at = created

    budget = sla_policy.minutes_for(ticket.priority)

    ticket.sla_minutes = float(budget)

    ticket.warn_fraction = float(sla_policy.warn_fraction)

    ticket.due_at = add_business_minutes(created, budget, calendar)

    ticket.warn_at = add_business_minutes(
        created, budget * sla_policy.warn_fraction, calendar
    )

    # A closed ticket's clock stopped when it closed.
    measured_to = moment

    if ticket.is_closed and ticket.resolved_at:

        measured_to = as_utc(ticket.resolved_at)

    ticket.elapsed_business_minutes = round(
        elapsed_business_minutes(created, measured_to, calendar), 4
    )

    if ticket.is_closed:

        ticket.breached = False
        ticket.warning_due = False

    else:

        ticket.breached = (
            ticket.elapsed_business_minutes >= budget
        )

        ticket.warning_due = (
            not ticket.breached
            and ticket.elapsed_business_minutes
            >= budget * sla_policy.warn_fraction
        )

    return ticket


def recalculate_deadlines(
    tickets: Sequence[Ticket],
    sla_policy: Optional[SlaPolicy] = None,
    now: Optional[datetime] = None,
) -> List[Ticket]:
    """
    Apply a (possibly just-changed) SLA policy to existing tickets.

    This is the supported path for "runtime changes to SLA
    conditions": shorten a resolution budget and every open ticket's
    ``due_at`` moves, including flipping an open ticket into breach.
    Pure recompute -- nothing is remembered from the previous policy.
    """

    for ticket in tickets:

        apply_sla(ticket, sla_policy, now)

    return list(tickets)


# ============================================================
# DUPLICATES AND RELATED TICKETS
# ============================================================

def issue_signature(text: str) -> List[str]:
    """
    The meaningful words of an issue, for overlap comparison.
    """

    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())

    return [
        token
        for token in tokens
        if token not in STOPWORDS and len(token) > 1
    ]


def token_overlap(left: Sequence[str], right: Sequence[str]) -> float:
    """
    Jaccard overlap of two token sets.
    """

    first = set(left)
    second = set(right)

    if not first or not second:

        return 0.0

    return len(first & second) / len(first | second)


def similarity_score(
    left: Ticket,
    right: Ticket,
    policy: Optional[TicketPolicy] = None,
) -> Tuple[float, List[str]]:
    """
    How likely two tickets describe the same problem.

    Additive and capped at 1.0, with the reasons returned so a
    duplicate link can be justified in the audit trail. Identifiers
    are weighted far above wording, because two customers describing
    the same broken order in different words are still the same
    problem, and two customers with the same complaint about
    different orders are not.
    """

    policy = policy or TicketPolicy()

    score = 0.0

    reasons: List[str] = []

    if left.order_id and right.order_id:

        if left.order_id.upper() == right.order_id.upper():

            score += policy.duplicate_weight_order_id

            reasons.append(f"same order {left.order_id}")

        else:

            # Different, explicit orders are strong evidence of
            # different problems.
            score -= policy.duplicate_weight_order_id

            reasons.append("different order")

    if left.product_code and right.product_code:

        if left.product_code.upper() == right.product_code.upper():

            score += policy.duplicate_weight_product_code

            reasons.append(f"same product {left.product_code}")

    if left.contact and right.contact:

        if left.contact.strip().lower() == right.contact.strip().lower():

            score += policy.duplicate_weight_contact

            reasons.append("same contact")

    if left.category and left.category == right.category:

        score += policy.duplicate_weight_category

        reasons.append(f"same category {left.category}")

    left_tokens = issue_signature(left.issue)

    right_tokens = issue_signature(right.issue)

    overlap = token_overlap(left_tokens, right_tokens)

    shared = len(set(left_tokens) & set(right_tokens))

    if shared >= policy.minimum_token_overlap:

        score += policy.duplicate_weight_similarity * overlap

        reasons.append(f"{shared} shared issue terms")

    return round(max(0.0, min(1.0, score)), 4), reasons


def classify_relation(
    left: Ticket,
    right: Ticket,
    policy: Optional[TicketPolicy] = None,
) -> Tuple[str, float, List[str]]:
    """
    "duplicate", "related" or "unrelated", with the score and the
    reasons behind it.
    """

    policy = policy or TicketPolicy()

    score, reasons = similarity_score(left, right, policy)

    if score >= policy.duplicate_threshold:

        return "duplicate", score, reasons

    if score >= policy.related_threshold:

        return "related", score, reasons

    return "unrelated", score, reasons


def find_related_tickets(
    ticket: Ticket,
    existing: Sequence[Ticket],
    policy: Optional[TicketPolicy] = None,
) -> List[Dict[str, Any]]:
    """
    Every existing ticket that is a duplicate or is related, best
    match first.
    """

    policy = policy or TicketPolicy()

    matches: List[Dict[str, Any]] = []

    for candidate in existing:

        if candidate.ticket_id == ticket.ticket_id:

            continue

        if candidate.is_closed and candidate.status == "closed":

            continue

        relation, score, reasons = classify_relation(
            ticket, candidate, policy
        )

        if relation == "unrelated":

            continue

        matches.append(
            {
                "ticket_id": candidate.ticket_id,
                "relation": relation,
                "score": score,
                "reasons": reasons,
            }
        )

    matches.sort(key=lambda item: (-item["score"], item["ticket_id"]))

    return matches


def link_ticket(
    ticket: Ticket,
    match: Dict[str, Any],
    original: Optional[Ticket] = None,
) -> Ticket:
    """
    Apply a duplicate/related match to a ticket.

    A duplicate inherits the original's SLA: it is the same problem,
    so it must not start a second clock. A related ticket keeps its
    own SLA and is only cross-linked and grouped.
    """

    ticket.linked_tickets.append(match["ticket_id"])

    if match["relation"] == "duplicate":

        ticket.duplicate_of = match["ticket_id"]
        ticket.status = "duplicate"

        if original is not None:

            ticket.group_id = original.group_id or original.ticket_id
            ticket.due_at = original.due_at
            ticket.warn_at = original.warn_at
            ticket.sla_minutes = original.sla_minutes
            ticket.warn_fraction = original.warn_fraction
            ticket.priority = original.priority
            ticket.priority_score = original.priority_score

    else:

        ticket.group_id = ticket.group_id or match["ticket_id"]

        if original is not None and original.group_id:

            ticket.group_id = original.group_id

    return ticket


# ============================================================
# MULTI-ISSUE MESSAGES
# ============================================================

# A fragment shorter than this is not a standalone issue.
MIN_ISSUE_WORDS = 3


def split_issues(text: str) -> List[str]:
    """
    Split a message carrying more than one request.

    Only splits when the Task 6 detector says there is more than one
    request, so an ordinary sentence containing "and" is left alone.
    Returns the original text as a single element otherwise.
    """

    text = (text or "").strip()

    if not text:

        return []

    if not detect_multiple_requests(text):

        return [text]

    working = text

    for separator in ISSUE_SEPARATORS:

        working = re.sub(
            re.escape(separator),
            "\n",
            working,
            flags=re.IGNORECASE,
        )

    parts = [
        part.strip(" ,.;:-")
        for part in working.split("\n")
        if part.strip(" ,.;:-")
    ]

    if len(parts) < 2:

        return [text]

    # The Task 6 detector fires on any conjunction, so the split has
    # to be justified before it is taken.
    #
    # "the box and the label were both damaged" is one issue, not two,
    # and cutting it leaves fragments nobody can act on.
    if any(len(part.split()) < MIN_ISSUE_WORDS for part in parts):

        return [text]

    # "My card was charged twice and this is very frustrating" is also
    # one issue: the second half is how the customer feels about the
    # first, and it matches no category at all.
    #
    # So two parts describe two issues only when every one of them is
    # recognisably about something and they are about *different*
    # things -- the same test that decides whether two tickets are
    # unrelated. A part that reads as "general" is evidence of nothing,
    # so it never justifies a split.
    categories = [infer_category(part) for part in parts]

    if "general" in categories:

        return [text]

    if len(set(categories)) < 2:

        return [text]

    return parts


# ============================================================
# HANDOFF
# ============================================================

def build_handoff_summary(
    ticket: Ticket,
    sla_policy: Optional[SlaPolicy] = None,
    now: Optional[datetime] = None,
) -> str:
    """
    The masked block an agent receives.

    Everything a human needs to pick the ticket up without reading the
    raw transcript, and nothing that would leak personal data: the
    contact is masked, the transcript excerpt is masked, and the
    identifiers the agent needs to work the case (order id, product
    code, error code) are kept.
    """

    sla_policy = sla_policy or SlaPolicy()

    if ticket.due_at is None:

        apply_sla(ticket, sla_policy, now)

    lines = [
        f"Ticket: {ticket.ticket_id}",
        f"Session: {ticket.session_id}",
        f"Category: {ticket.category}",
        f"Status: {ticket.status}",
        f"Priority: {ticket.priority} "
        f"(score {ticket.priority_score:.2f})",
        f"Severity: {ticket.severity}",
        f"Created: {as_utc(ticket.created_at).isoformat()}",
        f"Language: {ticket.language}",
        f"Sentiment: {ticket.sentiment_label} "
        f"({ticket.sentiment_confidence:.2f})",
    ]

    if ticket.risk_topics:

        lines.append(
            "Risk: " + ", ".join(ticket.risk_topics)
        )

    sla_line = (
        f"SLA: {ticket.sla_minutes:.0f} business minutes, "
        f"due {as_utc(ticket.due_at).isoformat()}, "
        f"elapsed {ticket.elapsed_business_minutes:.0f}"
    )

    if ticket.breached:

        sla_line += " -- BREACHED"

    elif ticket.warning_due:

        sla_line += " -- WARNING"

    lines.append(sla_line)

    lines.append("")

    lines.append("Issue:")
    lines.append(mask_sensitive_text(ticket.issue or "(none recorded)"))

    identifiers = [
        ("Order", ticket.order_id),
        ("Product", ticket.product_name),
        ("Product code", ticket.product_code),
        ("Amount", ticket.amount),
        ("Error code", ticket.error_code),
        ("Account", ticket.account_identifier),
        ("Customer", ticket.customer_name),
        ("Contact", ticket.masked_contact()),
    ]

    present = [
        f"- {label}: {value}"
        for label, value in identifiers
        if value
    ]

    if present:

        lines.append("")
        lines.append("Details:")
        lines.extend(present)

    if ticket.missing_fields:

        lines.append("")
        lines.append(
            "Awaiting from customer: "
            + ", ".join(ticket.missing_fields)
        )

    if ticket.evidence:

        lines.append("")
        lines.append("Evidence:")
        lines.extend(
            f"- {mask_sensitive_text(item)}"
            for item in ticket.evidence
        )

    if ticket.conversation_summary:

        lines.append("")
        lines.append("Transcript excerpt:")
        lines.append(
            mask_sensitive_text(ticket.conversation_summary)
        )

    return "\n".join(lines)


# ============================================================
# CREATION
# ============================================================

def next_ticket_id(
    now: Optional[datetime] = None,
    taken: Sequence[str] = (),
) -> str:
    """
    A deterministic ticket id for a given instant, so tests can
    assert on it.

    The id is only second-accurate, and a real conversation can raise
    two tickets within one second (a multi-issue message, or a
    duplicate arriving hot on the original). ``taken`` carries the ids
    already in use so the caller still gets a stable, sortable id
    instead of an overwritten row.
    """

    base = "TCK-" + as_utc(now).strftime("%Y%m%d%H%M%S")

    if base not in set(taken):

        return base

    suffix = 2

    while f"{base}-{suffix}" in set(taken):

        suffix += 1

    return f"{base}-{suffix}"


def create_ticket(
    session_id: str,
    message: str,
    sentiment: Optional[SentimentResult] = None,
    policy: Optional[TicketPolicy] = None,
    sla_policy: Optional[SlaPolicy] = None,
    now: Optional[datetime] = None,
    ticket_id: Optional[str] = None,
    conversation_summary: str = "",
    language: Optional[str] = None,
    existing: Optional[Sequence[Ticket]] = None,
) -> Ticket:
    """
    Build a structured ticket from one customer message.

    Order of work matters: extract, categorise, validate, prioritise,
    then compute the SLA -- because the SLA budget depends on the
    priority band, which depends on the score.

    When ``existing`` is supplied, the new ticket is matched against
    it and linked as a duplicate or a related ticket. Duplicates
    inherit the original's deadlines and get no clock of their own.

    ``sentiment`` is the sentiment of *this issue's own words*, not of
    the whole conversation. That distinction matters: a risk topic
    raised earlier in the conversation -- or by another half of a
    multi-issue message -- must not categorise an unrelated problem.
    """

    policy = policy or TicketPolicy()

    sla_policy = sla_policy or SlaPolicy()

    moment = as_utc(now)

    fields = extract_ticket_fields(message)

    language = language or (
        sentiment.language if sentiment else "english"
    )

    ticket = Ticket(
        ticket_id=ticket_id or next_ticket_id(moment),
        session_id=session_id,
        issue=(message or "").strip(),
        language=language,
        created_at=moment,
        updated_at=moment,
        customer_name=fields.get("customer_name"),
        contact=fields.get("contact"),
        account_identifier=fields.get("account_identifier"),
        order_id=fields.get("order_id"),
        product_name=fields.get("product_name"),
        product_code=fields.get("product_code"),
        amount=fields.get("amount"),
        error_code=fields.get("error_code"),
        reference_date=fields.get("reference_date"),
        sentiment_label=sentiment.label if sentiment else "neutral",
        sentiment_confidence=(
            sentiment.confidence if sentiment else 0.0
        ),
        risk_topics=list(sentiment.risk_topics) if sentiment else [],
        evidence=list(sentiment.evidence) if sentiment else [],
        conversation_summary=conversation_summary,
    )

    ticket.category = infer_category(
        message, fields, ticket.risk_topics
    )

    ticket.severity = derive_severity(sentiment)

    ticket.missing_fields = missing_mandatory_fields(ticket, policy)

    ticket.status = (
        "awaiting_info" if ticket.missing_fields else "open"
    )

    score, priority, components = score_priority(
        ticket, policy, sla_policy, moment
    )

    ticket.priority_score = score
    ticket.priority = priority
    ticket.priority_components = components

    apply_sla(ticket, sla_policy, moment)

    # Duplicate / related matching.
    if existing:

        matches = find_related_tickets(ticket, existing, policy)

        if matches:

            best = matches[0]

            original = next(
                (
                    candidate
                    for candidate in existing
                    if candidate.ticket_id == best["ticket_id"]
                ),
                None,
            )

            link_ticket(ticket, best, original)

            ticket.routing_reason = (
                f"{best['relation']} of {best['ticket_id']}: "
                + "; ".join(best["reasons"])
            )

    ticket.masked_handoff = build_handoff_summary(
        ticket, sla_policy, moment
    )

    return ticket


# ============================================================
# TICKET LIFECYCLE -- CONTINUING AND COMPLETING
# ============================================================

def introduces_new_issue(
    message: str,
    ticket: Ticket,
    policy: Optional[TicketPolicy] = None,
) -> bool:
    """
    Does this message describe a *different* problem from the open
    ticket it arrived alongside?

    This is what stops every follow-up in a conversation becoming a
    fresh ticket. A message continues the open ticket when it names no
    new category, when it names the same category, or when it is about
    the same order -- an angry "still waiting" is the same problem
    again, not a second one. It only counts as new when it is
    recognisably about something else, so a genuine second problem in
    the same conversation still gets its own ticket.
    """

    policy = policy or TicketPolicy()

    fields = extract_ticket_fields(message)

    category = infer_category(message, fields, ())

    if category == "general":

        return False

    if category == ticket.category:

        return False

    incoming_order = (fields.get("order_id") or "").upper()

    if incoming_order and incoming_order == (
        ticket.order_id or ""
    ).upper():

        return False

    return True


def is_followup_without_issue(message: str) -> bool:
    """
    A message that carries no issue of its own -- a nudge, a thank
    you, a request for an update.

    It continues whatever is already open rather than opening anything
    new. A message that *does* name a problem is deliberately not this:
    a restated fault is left to the duplicate path, which files it as
    a duplicate on the review queue instead of quietly absorbing it.
    """

    fields = extract_ticket_fields(message)

    return infer_category(message, fields, ()) == "general"


def complete_ticket(
    ticket: Ticket,
    message: str,
    sentiment: Optional[SentimentResult] = None,
    policy: Optional[TicketPolicy] = None,
    sla_policy: Optional[SlaPolicy] = None,
    now: Optional[datetime] = None,
    conversation_summary: str = "",
) -> Tuple[Ticket, List[str]]:
    """
    Fold a follow-up reply into a ticket that is still waiting for
    information, rather than raising a second ticket for the same
    problem.

    Only fields the ticket is *missing* are written, so a reply can
    never silently overwrite an order id the customer already gave.
    Returns the ticket and the names of the fields this message
    completed, which is what lets the caller tell "the customer
    answered" from "the customer said something else".

    The clock is deliberately not reset: ``created_at`` is untouched,
    so completing a ticket at minute 20 still measures from the
    original report and the SLA conditions see the real waiting time.
    """

    policy = policy or TicketPolicy()

    sla_policy = sla_policy or SlaPolicy()

    moment = as_utc(now)

    fields = extract_ticket_fields(message)

    filled: List[str] = []

    for name in (
        "customer_name",
        "contact",
        "account_identifier",
        "order_id",
        "product_name",
        "product_code",
        "amount",
        "error_code",
        "reference_date",
    ):

        value = fields.get(name)

        if value and not getattr(ticket, name, None):

            setattr(ticket, name, value)

            filled.append(name)

    # The reply is part of the same problem, so it can also sharpen a
    # category that was only ever guessed at.
    if ticket.category == "general" and fields:

        ticket.category = infer_category(
            message, fields, ticket.risk_topics
        )

    ticket.missing_fields = missing_mandatory_fields(ticket, policy)

    if not ticket.missing_fields and ticket.status == "awaiting_info":

        ticket.status = "open"

    if conversation_summary:

        ticket.conversation_summary = conversation_summary

    ticket.updated_at = moment

    score, priority, components = score_priority(
        ticket, policy, sla_policy, moment
    )

    ticket.priority_score = score
    ticket.priority = priority
    ticket.priority_components = components

    apply_sla(ticket, sla_policy, moment)

    ticket.masked_handoff = build_handoff_summary(
        ticket, sla_policy, moment
    )

    return ticket, filled


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    from datetime import timezone

    print("=" * 60)
    print("TICKETING SELF-TEST")
    print("=" * 60)

    moment = datetime(2026, 10, 2, 10, 0, tzinfo=timezone.utc)

    sla = SlaPolicy(
        calendar=BusinessCalendarConfig(timezone_name="UTC")
    )

    message = (
        "My order ORD-12345 was charged twice, 1499.00 INR. "
        "Email me at john.doe@example.com, this is very frustrating."
    )

    sentiment = __import__(
        "task7_sentiment"
    ).analyse_message(message)

    ticket = create_ticket(
        "sess-1",
        message,
        sentiment,
        sla_policy=sla,
        now=moment,
        ticket_id="TCK-1",
    )

    print("\nTicket:")
    print("  category :", ticket.category)
    print("  priority :", ticket.priority, ticket.priority_score)
    print("  severity :", ticket.severity)
    print("  due_at   :", ticket.due_at)
    print("  missing  :", ticket.missing_fields)

    print("\nHandoff:")
    print(build_handoff_summary(ticket, sla, moment))

    assert ticket.order_id == "ORD-12345"
    assert ticket.duplicate_of is None

    # The same problem again, two minutes later.
    again = create_ticket(
        "sess-2",
        message,
        sentiment,
        sla_policy=sla,
        now=moment + timedelta(minutes=2),
        ticket_id="TCK-2",
        existing=[ticket],
    )

    print("\nSecond ticket duplicate_of:", again.duplicate_of)

    assert again.duplicate_of == "TCK-1"
    assert again.due_at == ticket.due_at

    print("\nRESULT: PASS")
