"""
TASK 7.2 - ESCALATION ENGINE

Turns a classified message into escalation decisions. Ten conditions
(ESC-01 ... ESC-10) cover the brief:

    ESC-01  repeated negative contact
    ESC-02  account compromise / unauthorised access / data breach
    ESC-03  duplicate payment
    ESC-04  legal threat
    ESC-05  urgent contact outside business hours
    ESC-06  non-urgent negative contact outside business hours
    ESC-07  unresolved longer than the configured limit
    ESC-08  SLA breached
    ESC-09  SLA at or past the warning fraction
    ESC-10  explicit request for a human

Every escalation is recorded with its reason, the condition that
activated it, the queue it went to, and a conversation summary that is
masked before it is stored, so no raw email, phone, card or account
number reaches the audit trail or the agent handoff.

Two properties matter more than the rest:

* **High-risk detection is independent of sentiment.** A calm,
  politely-worded message about a duplicate charge still escalates.
  Nothing here reads the sentiment label to decide a risk condition.

* **Time is injected, never sampled.** Every function takes
  ``now=None``; when it is None the real clock is read. Tests pass an
  explicit instant, so "unresolved for exactly 15 minutes" and
  "after hours" are deterministic assertions rather than flaky ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

from business_calendar import (
    BusinessCalendarConfig,
    elapsed_business_minutes,
    is_after_hours,
    next_business_start,
)
from support_deps import as_utc, mask_sensitive_text
from task6_multilingual import ConversationMessage, ConversationSession
from task7_sentiment import (
    NEGATIVE_LABELS,
    SentimentConfig,
    SentimentLabel,
    SentimentResult,
    analyse_message,
    requests_human,
)


# ============================================================
# CONDITION CATALOG
# ============================================================

# key -> (condition id, human-readable description)
#
# The description is written into the escalation record verbatim, so
# an agent reading the audit trail sees which rule fired and why.
CONDITION_CATALOG: Dict[str, Tuple[str, str]] = {

    "repeated_negative": (
        "ESC-01",
        "The customer has raised repeated negative contacts within "
        "the configured window.",
    ),

    "high_risk_account_compromise": (
        "ESC-02",
        "Possible account compromise: unauthorised access, an "
        "unrecognised login or a hijacked account.",
    ),

    "high_risk_unauthorized_charge": (
        "ESC-02",
        "Possible unauthorised or fraudulent charge on the "
        "customer's payment method.",
    ),

    "high_risk_data_breach": (
        "ESC-02",
        "Possible exposure or leak of the customer's personal data.",
    ),

    "high_risk_duplicate_payment": (
        "ESC-03",
        "Possible duplicate payment: the customer reports being "
        "charged more than once.",
    ),

    "high_risk_legal_threat": (
        "ESC-04",
        "The customer has raised legal action, a lawyer or a "
        "regulator.",
    ),

    "after_hours_urgent": (
        "ESC-05",
        "The customer marked this as urgent outside business hours.",
    ),

    "after_hours_negative": (
        "ESC-06",
        "A negative contact arrived outside business hours; the "
        "request is scheduled for the next working day.",
    ),

    "unresolved_too_long": (
        "ESC-07",
        "The conversation has been unresolved beyond the configured "
        "limit.",
    ),

    "sla_breached": (
        "ESC-08",
        "The SLA resolution target for this ticket has been "
        "breached.",
    ),

    "sla_warning": (
        "ESC-09",
        "The SLA is at or past its warning threshold but not yet "
        "breached.",
    ),

    "human_requested": (
        "ESC-10",
        "The customer explicitly asked for a human agent or a "
        "manager.",
    ),
}


# Risk topic -> condition key. The topic names come from
# task7_sentiment.RiskTopic; the values keep ESC-02/03/04 as the
# plan specifies, with the remaining two topics folded into the
# account-security family.
RISK_CONDITION_KEYS: Dict[str, str] = {
    "account_compromise": "high_risk_account_compromise",
    "unauthorized_charge": "high_risk_unauthorized_charge",
    "data_breach": "high_risk_data_breach",
    "duplicate_payment": "high_risk_duplicate_payment",
    "legal_threat": "high_risk_legal_threat",
}


# Severity a label carries when no risk topic outranks it.
LABEL_SEVERITY: Dict[str, str] = {
    SentimentLabel.URGENT.value: "high",
    SentimentLabel.FRUSTRATED.value: "medium",
    SentimentLabel.SARCASTIC.value: "medium",
    SentimentLabel.NEGATIVE.value: "medium",
    SentimentLabel.POSITIVE.value: "low",
    SentimentLabel.NEUTRAL.value: "low",
}


SEVERITY_ORDER: Dict[str, int] = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class EscalationConfig:
    """
    Every threshold an escalation decision depends on.

    Mutable by design: the hidden evaluation may change the
    unresolved limit, the after-hours behaviour or the queue names at
    runtime, and the next evaluate() call must honour the change.
    """

    # ESC-01
    repeated_negative_window: int = 4
    repeated_negative_count: int = 2

    # ESC-07
    unresolved_minutes: int = 15

    # "wall" counts real elapsed time; "business" counts only time
    # inside business hours, so a complaint raised on Friday evening
    # does not "age" over the weekend.
    unresolved_clock: str = "wall"

    # When True, the unresolved clock only runs for a most recent
    # customer turn that is negative. A satisfied customer waiting on
    # a delivery is not an unresolved complaint.
    unresolved_requires_negative: bool = True

    # ESC-02 / ESC-03 / ESC-04
    high_risk_topics: Tuple[str, ...] = (
        "account_compromise",
        "duplicate_payment",
        "legal_threat",
        "unauthorized_charge",
        "data_breach",
    )

    # A legal threat goes to the on-call queue at any hour, not just
    # outside business hours.
    legal_always_immediate: bool = True

    # ESC-10
    human_requests_escalate: bool = True

    # Queue routing
    priority_queue: str = "priority_support"
    on_call_queue: str = "on_call"
    scheduled_queue: str = "scheduled_next_working_day"

    # Queue names considered "already resolved" for ESC-07.
    resolved_statuses: Tuple[str, ...] = (
        "resolved", "closed", "cancelled",
    )

    # Turns folded into the conversation summary on a record.
    summary_turns: int = 4
    summary_max_chars: int = 600

    calendar: BusinessCalendarConfig = field(
        default_factory=BusinessCalendarConfig
    )
    sentiment_config: SentimentConfig = field(
        default_factory=SentimentConfig
    )


# ============================================================
# RECORDS
# ============================================================

@dataclass
class EscalationRecord:
    """
    One activated condition, with everything needed to justify it
    later without re-reading the conversation.
    """

    escalation_id: str
    session_id: str

    condition_id: str
    condition: str
    reason: str

    severity: str
    queue: str

    escalated_at: datetime
    scheduled_for: Optional[datetime] = None

    sentiment_label: str = "neutral"
    risk_topics: List[str] = field(default_factory=list)
    unresolved_minutes: float = 0.0

    priority: Optional[str] = None

    # The ticket this condition is about, when there is one. ESC-08 and
    # ESC-09 are raised *against* a ticket rather than producing one, so
    # they carry its id here; without it a breach stored by a
    # recalculation would name no ticket and be unactionable.
    ticket_id: Optional[str] = None

    conversation_summary: str = ""
    masked_summary: str = ""
    message_excerpt: str = ""

    # ESC-09 warns without escalating; the flag keeps it in the audit
    # trail without flipping the escalate decision.
    is_warning: bool = False

    def to_dict(self) -> Dict[str, object]:
        """
        JSON-friendly form, with datetimes as ISO strings so simulated
        times survive a round trip through SQLite exactly.

        The summary is emitted *masked*. This is the form that gets
        written to the database and shown to whoever picks the
        escalation up, and an escalation record travels further than
        the conversation it came from -- into a log, an on-call page, a
        dashboard. The unmasked transcript stays on the record itself,
        where the conversation already is, and never leaves as a dict.
        """

        return {
            "escalation_id": self.escalation_id,
            "session_id": self.session_id,
            "condition_id": self.condition_id,
            "condition": self.condition,
            "reason": self.reason,
            "severity": self.severity,
            "queue": self.queue,
            "escalated_at": as_utc(self.escalated_at).isoformat(),
            "scheduled_for": (
                as_utc(self.scheduled_for).isoformat()
                if self.scheduled_for
                else None
            ),
            "sentiment_label": self.sentiment_label,
            "risk_topics": list(self.risk_topics),
            "unresolved_minutes": self.unresolved_minutes,
            "priority": self.priority,
            "ticket_id": self.ticket_id,
            "conversation_summary": self.masked_summary,
            "masked_summary": self.masked_summary,
            "message_excerpt": self.message_excerpt,
            "is_warning": self.is_warning,
        }


@dataclass
class EscalationResult:
    """
    Everything one evaluate() call decided.
    """

    escalate: bool = False
    records: List[EscalationRecord] = field(default_factory=list)
    queues: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    unresolved_minutes: float = 0.0
    severity: str = "none"

    @property
    def warnings(self) -> List[EscalationRecord]:

        return [record for record in self.records if record.is_warning]

    @property
    def condition_ids(self) -> List[str]:

        return [record.condition_id for record in self.records]

    def to_dict(self) -> Dict[str, object]:

        return {
            "escalate": self.escalate,
            "queues": list(self.queues),
            "actions": list(self.actions),
            "unresolved_minutes": self.unresolved_minutes,
            "severity": self.severity,
            "records": [record.to_dict() for record in self.records],
        }


# ============================================================
# UNRESOLVED CLOCK
# ============================================================

def last_user_message(
    messages: Sequence[ConversationMessage],
) -> Optional[ConversationMessage]:
    """
    The most recent customer turn, or None when the customer has not
    spoken yet.
    """

    for message in reversed(list(messages)):

        if message.role == "user":

            return message

    return None


def unresolved_minutes(
    messages: Sequence[ConversationMessage],
    sentiment: Optional[SentimentResult] = None,
    config: Optional[EscalationConfig] = None,
    now: Optional[datetime] = None,
    status: Optional[str] = None,
) -> float:
    """
    How long the conversation has been unresolved.

    The clock starts at the **oldest complaint still outstanding**, not
    at the newest message. That distinction is the whole point of the
    condition: a customer who complained at 10:00 and is still chasing
    it at 10:20 has been waiting twenty minutes, and measuring from
    their 10:20 message would report zero and never escalate. A neutral
    "any update?" does not start a new clock, and neither does an
    assistant reply -- an acknowledgement is not an outcome.

    ``sentiment``, when supplied, is the analysis of the newest user
    message and is used for that message; every older one is analysed
    from its own text, because a caller's blended history sentiment
    describes the conversation rather than any one turn.

    Returns 0.0 when:

    * the customer has not spoken,
    * the ticket status is already resolved/closed/cancelled,
    * ``unresolved_requires_negative`` is set and no user turn was
      negative, or
    * the clock has not advanced.
    """

    config = config or EscalationConfig()

    if status and status.lower() in config.resolved_statuses:

        return 0.0

    spoken = [
        message for message in messages if message.role == "user"
    ]

    if not spoken:

        return 0.0

    newest = spoken[-1]

    def is_negative(message: ConversationMessage) -> bool:

        if message is newest and sentiment is not None:

            analysis = sentiment

        else:

            analysis = analyse_message(
                message.content,
                (
                    message.language
                    if message.language != "unknown"
                    else None
                ),
                config.sentiment_config,
                message.timestamp,
            )

        return (
            analysis.label in NEGATIVE_LABELS
            or analysis.polarity <= -0.3
        )

    if config.unresolved_requires_negative:

        outstanding = [
            message for message in spoken if is_negative(message)
        ]

        if not outstanding:

            return 0.0

        earliest = outstanding[0]

    else:

        earliest = spoken[0]

    moment = as_utc(now)

    started = as_utc(earliest.timestamp)

    if moment <= started:

        return 0.0

    if config.unresolved_clock == "business":

        return round(
            elapsed_business_minutes(
                started, moment, config.calendar
            ),
            4,
        )

    return round((moment - started).total_seconds() / 60.0, 4)


# ============================================================
# SUMMARY
# ============================================================

def build_conversation_summary(
    messages: Sequence[ConversationMessage],
    config: Optional[EscalationConfig] = None,
) -> str:
    """
    A deterministic transcript digest -- no model involved, so the
    summary in an escalation record is reproducible from the session.
    """

    config = config or EscalationConfig()

    turns = list(messages)[-config.summary_turns:]

    if not turns:

        return "No conversation history."

    lines = [
        f"{message.role}: {message.content.strip()}"
        for message in turns
        if message.content and message.content.strip()
    ]

    summary = "\n".join(lines)

    if len(summary) > config.summary_max_chars:

        summary = summary[: config.summary_max_chars].rstrip() + "..."

    return summary


def mask_summary(summary: str) -> str:
    """
    Mask the digest before it is stored or handed to an agent.
    """

    return mask_sensitive_text(summary or "")


# ============================================================
# ENGINE
# ============================================================

class EscalationEngine:
    """
    Evaluates the ten escalation conditions for one turn.

    Stateless apart from its config: every input arrives as an
    argument, so the same engine instance serves every session and a
    test can drive it with fabricated sessions and simulated clocks.
    """

    def __init__(
        self,
        config: Optional[EscalationConfig] = None,
    ):

        self.config = config or EscalationConfig()

    # helpers
    def _after_hours(self, now: Optional[datetime]) -> bool:

        return is_after_hours(now, self.config.calendar)

    def _make_record(
        self,
        session_id: str,
        condition_key: str,
        reason: str,
        severity: str,
        queue: str,
        sentiment: SentimentResult,
        now: Optional[datetime],
        unresolved: float,
        summary: str,
        message: str,
        scheduled_for: Optional[datetime] = None,
        priority: Optional[str] = None,
        ticket_id: Optional[str] = None,
        is_warning: bool = False,
    ) -> EscalationRecord:

        condition_id, condition = CONDITION_CATALOG[condition_key]

        moment = as_utc(now)

        return EscalationRecord(
            escalation_id=(
                f"{session_id}:{condition_id}:{moment.isoformat()}"
            ),
            session_id=session_id,
            condition_id=condition_id,
            condition=condition,
            reason=reason,
            severity=severity,
            queue=queue,
            escalated_at=moment,
            scheduled_for=scheduled_for,
            sentiment_label=sentiment.label,
            risk_topics=list(sentiment.risk_topics),
            unresolved_minutes=unresolved,
            priority=priority,
            ticket_id=ticket_id,
            conversation_summary=summary,
            masked_summary=mask_summary(summary),
            message_excerpt=mask_sensitive_text(message or ""),
            is_warning=is_warning,
        )

    # main entry point
    def evaluate(
        self,
        session_id: str,
        message: str,
        sentiment: SentimentResult,
        session: Optional[ConversationSession] = None,
        now: Optional[datetime] = None,
        status: Optional[str] = None,
        ticket: Optional[object] = None,
    ) -> EscalationResult:
        """
        Decide which conditions this turn activates.

        ``ticket`` is optional and read duck-typed: when it carries
        ``priority``, ``due_at``, ``warn_at`` or ``breached``, the SLA
        conditions (ESC-08/ESC-09) are evaluated too. Escalation
        therefore has no import dependency on the ticketing module.

        The full session is passed to ``unresolved_minutes``, including
        the turn being evaluated, which is safe precisely because that
        function anchors its clock at the *oldest* outstanding
        complaint rather than the newest message. Appending this turn
        therefore cannot reset the wait to zero, and a customer chasing
        a complaint they raised twenty minutes ago still escalates.
        """

        config = self.config

        moment = as_utc(now)

        messages = list(session.messages) if session else []

        after_hours = self._after_hours(moment)

        unresolved = unresolved_minutes(
            messages, sentiment, config, moment, status
        )

        summary = build_conversation_summary(messages, config)

        records: List[EscalationRecord] = []

        actions: List[str] = []

        def add(
            condition_key: str,
            reason: str,
            severity: str,
            queue: str,
            **extra,
        ) -> None:

            records.append(
                self._make_record(
                    session_id=session_id,
                    condition_key=condition_key,
                    reason=reason,
                    severity=severity,
                    queue=queue,
                    sentiment=sentiment,
                    now=moment,
                    unresolved=unresolved,
                    summary=summary,
                    message=message,
                    **extra,
                )
            )

        # ESC-02 / ESC-03 / ESC-04 -- high risk
        #
        # Deliberately first, and deliberately blind to the label.
        risk_fired = False

        for topic in sentiment.risk_topics:

            if topic not in config.high_risk_topics:

                continue

            condition_key = RISK_CONDITION_KEYS.get(topic)

            if condition_key is None:

                continue

            risk_fired = True

            is_legal = topic == "legal_threat"

            to_on_call = is_legal or after_hours

            if is_legal and config.legal_always_immediate:

                to_on_call = True

            add(
                condition_key,
                reason=(
                    f"High-risk topic detected: {topic.replace('_', ' ')} "
                    f"(severity {sentiment.risk_severity}). Escalated "
                    f"regardless of the message tone, which was "
                    f"'{sentiment.label}'."
                ),
                severity=sentiment.risk_severity
                if sentiment.risk_severity != "none"
                else "high",
                queue=(
                    config.on_call_queue if to_on_call
                    else config.priority_queue
                ),
            )

            actions.append(
                f"Route to {'on-call' if to_on_call else 'priority'} "
                f"queue for {topic.replace('_', ' ')}."
            )

        # ESC-01 -- repeated negative contact
        if sentiment.repeated_negative:

            add(
                "repeated_negative",
                reason=(
                    f"The customer has contacted negatively at least "
                    f"{config.repeated_negative_count} times within the "
                    f"last {config.repeated_negative_window} turns."
                ),
                severity="high",
                queue=config.priority_queue,
            )

            actions.append(
                "Assign a senior agent: this is a repeat contact."
            )

        # ESC-05 / ESC-06 -- outside business hours
        if after_hours and not risk_fired:

            if sentiment.label == SentimentLabel.URGENT.value:

                add(
                    "after_hours_urgent",
                    reason=(
                        "The message is urgent and arrived outside "
                        "business hours."
                    ),
                    severity="high",
                    queue=config.on_call_queue,
                )

                actions.append(
                    "Page the on-call agent immediately."
                )

            elif (
                sentiment.label in NEGATIVE_LABELS
                or sentiment.polarity <= -0.3
            ):

                scheduled = next_business_start(
                    moment, config.calendar
                )

                add(
                    "after_hours_negative",
                    reason=(
                        "A negative contact arrived outside business "
                        "hours and is not urgent, so it is scheduled "
                        "for the next working day."
                    ),
                    severity="medium",
                    queue=config.scheduled_queue,
                    scheduled_for=scheduled,
                )

                actions.append(
                    "Schedule a follow-up for the next working day "
                    f"({as_utc(scheduled).isoformat()})."
                )

        # ESC-10 -- explicit request for a human
        if config.human_requests_escalate and requests_human(message or ""):

            add(
                "human_requested",
                reason=(
                    "The customer explicitly asked for a human agent "
                    "or a manager."
                ),
                severity="medium",
                queue=config.priority_queue,
            )

            actions.append("Hand the conversation to a human agent.")

        # ESC-07 -- unresolved beyond the limit
        if (
            unresolved >= config.unresolved_minutes
            and sentiment.label in NEGATIVE_LABELS
        ):

            add(
                "unresolved_too_long",
                reason=(
                    f"The conversation has been unresolved for "
                    f"{unresolved:.1f} minutes, at or beyond the "
                    f"{config.unresolved_minutes}-minute limit "
                    f"({config.unresolved_clock} clock)."
                ),
                severity="high",
                queue=config.priority_queue,
                priority="P2",
            )

            actions.append(
                "Auto-create a ticket: the conversation has been "
                "unresolved beyond the limit."
            )

        # ESC-08 / ESC-09 -- SLA state, when a ticket is supplied
        if ticket is not None:

            priority = getattr(ticket, "priority", None)

            breached = bool(getattr(ticket, "breached", False))

            warn_due = bool(getattr(ticket, "warning_due", False))

            if breached:

                add(
                    "sla_breached",
                    reason=(
                        f"The SLA target for ticket "
                        f"{getattr(ticket, 'ticket_id', 'unknown')} "
                        f"({priority or 'unprioritised'}) has been "
                        f"breached."
                    ),
                    severity="critical",
                    queue=config.on_call_queue,
                    priority=self._bump_priority(priority),
                    ticket_id=getattr(ticket, "ticket_id", None),
                )

                actions.append(
                    "Escalate the breached ticket and raise its "
                    "priority one band."
                )

            elif warn_due:

                add(
                    "sla_warning",
                    reason=(
                        f"Ticket "
                        f"{getattr(ticket, 'ticket_id', 'unknown')} "
                        f"has used at least "
                        f"{int(getattr(ticket, 'warn_fraction', 0.75) * 100)}% "
                        f"of its SLA budget without breaching it."
                    ),
                    severity="medium",
                    queue=config.priority_queue,
                    priority=priority,
                    ticket_id=getattr(ticket, "ticket_id", None),
                    is_warning=True,
                )

                actions.append(
                    "Warn the assigned agent: the SLA window is "
                    "nearly exhausted."
                )

        # Decide
        real = [record for record in records if not record.is_warning]

        queues: List[str] = []

        for record in records:

            if record.queue not in queues:

                queues.append(record.queue)

        severity = "none"

        for record in records:

            if (
                SEVERITY_ORDER.get(record.severity, 0)
                > SEVERITY_ORDER[severity]
            ):

                severity = record.severity

        return EscalationResult(
            escalate=bool(real),
            records=records,
            queues=queues,
            actions=actions,
            unresolved_minutes=unresolved,
            severity=severity,
        )

    @staticmethod
    def _bump_priority(priority: Optional[str]) -> Optional[str]:
        """
        Raise a priority one band: P3 -> P2. P1 is already the top.
        """

        if not priority:

            return "P1"

        order = ["P1", "P2", "P3", "P4"]

        if priority not in order:

            return priority

        index = order.index(priority)

        return order[max(0, index - 1)]


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    from datetime import timezone

    print("=" * 60)
    print("ESCALATION SELF-TEST")
    print("=" * 60)

    engine = EscalationEngine()

    # 2026-10-02 10:00 UTC is a Friday inside business hours,
    # given a UTC calendar.
    engine.config.calendar = BusinessCalendarConfig(
        timezone_name="UTC"
    )

    polite_risk = analyse_message(
        "I would appreciate it if you could look into what appears "
        "to be a duplicate charge on my card, whenever convenient."
    )

    print("\nPolite duplicate-charge message:")
    print("  sentiment:", polite_risk.label)
    print("  risk     :", polite_risk.risk_topics)

    moment = datetime(2026, 10, 2, 10, 0, tzinfo=timezone.utc)

    result = engine.evaluate(
        "sess-1", "duplicate charge", polite_risk, None, moment
    )

    print("  escalate :", result.escalate)
    print("  conditions:", result.condition_ids)

    assert result.escalate is True
    assert "ESC-03" in result.condition_ids

    # 2026-10-02 20:00 UTC on Friday is after hours.
    evening = datetime(2026, 10, 2, 20, 0, tzinfo=timezone.utc)

    calm = analyse_message("My order is late.")

    out = engine.evaluate(
        "sess-2", "My order is late.", calm, None, evening
    )

    print("\nCalm complaint on Friday evening:")
    print("  conditions:", out.condition_ids)
    print("  scheduled :", out.records[0].scheduled_for)

    assert "ESC-06" in out.condition_ids

    print("\nRESULT: PASS")
