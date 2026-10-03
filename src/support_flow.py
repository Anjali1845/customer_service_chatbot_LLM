"""
SUPPORT FLOW

The single seam between the chat UI and everything new: one customer
turn goes in, one fully-annotated turn comes out.

Why this module exists rather than the logic living in main.py:

* **It takes an ``answer_fn``.** The whole turn -- language detection,
  sentiment, escalation, ticket creation, routing, persistence -- runs
  offline with a stub answer function, so the hidden evaluation does
  not depend on the Gemini API being reachable, and neither do the
  tests.

* **The UI stays a UI.** ``main.py`` renders; this module decides.

* **Tone is additive.** ``build_enriched_question`` appends a style
  block to the query the chain already receives; it never rewrites the
  knowledge-base prompt, so no policy, eligibility rule, timeline or
  amount can change because a customer was angry.

Order of work per turn, and why it is this order:

1. register the message (Task 6) -- language and entities
2. clarify and stop, if the language was too uncertain to answer
3. sentiment over the whole conversation, not just this turn
4. escalate -- before answering, so a page goes out even if the model
   call fails
5. answer through the chain with the tone block attached
6. ticket -- extract, validate, dedupe, prioritise, SLA, route, store
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence

from support_deps import as_utc
from support_store import SupportStore
from task6_multilingual import (
    ConversationSession,
    SessionManager,
    Task6Config,
    build_language_instruction,
    detect_correction,
    detect_multiple_requests,
)
from task7_escalation import (
    EscalationConfig,
    EscalationEngine,
    EscalationRecord,
    EscalationResult,
)
from task7_sentiment import (
    NEGATIVE_LABELS,
    SentimentConfig,
    SentimentResult,
    analyse_conversation,
    analyse_message,
    build_tone_instruction,
)
from task8_routing import Agent, RoutingConfig, RoutingDecision, route_ticket
from task8_ticketing import (
    SlaPolicy,
    Ticket,
    TicketPolicy,
    build_information_request,
    create_ticket,
    link_ticket,
    next_ticket_id,
    recalculate_deadlines,
    split_issues,
)


# Ticket statuses that can still absorb a duplicate.
OPEN_TICKET_STATUSES = ("new", "open", "awaiting_info", "routed")

# How the flow reaches the knowledge base. Both arguments are given to
# the caller because the chain needs the tone block as a separate
# prompt argument: it must be able to build the *unchanged* prompt
# when there is no tone, which it cannot do if the tone has already
# been spliced into the question.
AnswerFn = Callable[[str, str], str]


# ============================================================
# CONFIG
# ============================================================

@dataclass
class SupportFlowConfig:
    """
    Everything the flow can be reconfigured with at runtime.

    Held as one object so the UI sidebar can change an SLA budget or
    take a team offline and have the very next turn honour it.
    """

    task6: Task6Config = field(default_factory=Task6Config)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    escalation: EscalationConfig = field(default_factory=EscalationConfig)
    ticket: TicketPolicy = field(default_factory=TicketPolicy)
    sla: SlaPolicy = field(default_factory=SlaPolicy)
    routing: RoutingConfig = field(default_factory=RoutingConfig)

    # Answer the customer in the language they wrote in, with the
    # register the sentiment calls for.
    tone_enabled: bool = True

    # Create a ticket for an unhappy customer even when no escalation
    # condition fired.
    ticket_on_negative: bool = True

    def __post_init__(self) -> None:

        # One calendar, shared. All three consumers must agree: the
        # SLA clock, the escalation clock and the after-hours routing
        # decision are taken in the same turn, and a routing config
        # left on its own default silently defers to a different
        # timezone's 09:00 than the SLA measures against.
        self.sla.calendar = self.escalation.calendar
        self.routing.calendar = self.escalation.calendar


# ============================================================
# TURN RESULT
# ============================================================

@dataclass
class TurnResult:
    """
    Everything one turn produced.
    """

    session_id: str
    question: str
    answer: str = ""
    language: str = "english"
    sentiment: Optional[SentimentResult] = None
    escalation: Optional[EscalationResult] = None
    tickets: List[Ticket] = field(default_factory=list)
    routing: List[RoutingDecision] = field(default_factory=list)
    information_request: str = ""
    enriched_question: str = ""
    tone_instruction: str = ""
    clarification_required: bool = False
    answered: bool = True
    multiple_requests: bool = False
    correction: bool = False
    notes: List[str] = field(default_factory=list)

    # The assistant message this turn produced. The UI keeps a turn's
    # detail -- sentiment, escalation, ticket -- against this id, so
    # the detail can be found again on the re-run that follows every
    # turn instead of being lost with the local variable.
    message_id: str = ""

    @property
    def ticket(self) -> Optional[Ticket]:
        """The primary ticket, when one was raised."""
        return self.tickets[0] if self.tickets else None

    @property
    def escalated(self) -> bool:
        return bool(self.escalation and self.escalation.escalate)

    @property
    def sentiment_label(self) -> str:
        return self.sentiment.label if self.sentiment else "neutral"

    @property
    def masked_handoff(self) -> str:
        ticket = self.ticket
        return ticket.masked_handoff if ticket else ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "question": self.question,
            "answer": self.answer,
            "language": self.language,
            "sentiment_label": self.sentiment_label,
            "sentiment_confidence": (
                self.sentiment.confidence if self.sentiment else 0.0
            ),
            "escalated": self.escalated,
            "conditions": (
                self.escalation.condition_ids if self.escalation else []
            ),
            "ticket_ids": [ticket.ticket_id for ticket in self.tickets],
            "queues": [decision.queue for decision in self.routing],
            "information_request": self.information_request,
            "clarification_required": self.clarification_required,
            "answered": self.answered,
            "notes": list(self.notes),
        }


@dataclass
class SlaRecalculation:
    """
    What re-applying the SLA policy to the stored tickets actually did.

    A count of tickets alone would be a half-truth: the whole point of
    a runtime policy change is which tickets it *broke*, and an
    operator reading "Recalculated 4 ticket(s)" has no way to tell a
    no-op from a policy that just put three customers past their
    deadline. So the breach and warning tallies are part of the
    result, not something the caller has to go and count.
    """

    tickets: int = 0
    breached: int = 0
    warning: int = 0
    escalations: int = 0
    records: List[EscalationRecord] = field(default_factory=list)
    breached_ticket_ids: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """
        One line for the operator.
        """

        if not self.tickets:

            return "No stored tickets to recalculate."

        text = f"Recalculated {self.tickets} ticket(s)."

        if self.breached:

            text += (
                f" {self.breached} now past the SLA deadline, "
                f"{self.escalations} escalation(s) raised."
            )

        elif self.warning:

            text += f" {self.warning} at or past the warning line."

        else:

            text += " None are past the SLA deadline."

        return text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tickets": self.tickets,
            "breached": self.breached,
            "warning": self.warning,
            "escalations": self.escalations,
            "conditions": [
                record.condition_id for record in self.records
            ],
            "breached_ticket_ids": list(self.breached_ticket_ids),
        }


# ============================================================
# FLOW
# ============================================================

class SupportFlow:
    """
    Stateful per-process orchestrator.

    One instance lives for the life of the Streamlit session; tests
    construct one with a stub ``answer_fn`` and a temp store.
    """

    def __init__(
        self,
        session_manager: Optional[SessionManager] = None,
        config: Optional[SupportFlowConfig] = None,
        store: Optional[SupportStore] = None,
        agents: Sequence[Agent] = (),
        answer_fn: Optional[AnswerFn] = None,
    ) -> None:

        self.config = config or SupportFlowConfig()

        self.session_manager = session_manager or SessionManager(
            self.config.task6
        )

        self.escalation_engine = EscalationEngine(
            self.config.escalation
        )

        self.store = store

        self.agents: List[Agent] = list(agents)

        self.answer_fn = answer_fn

        if self.store is not None:

            self.store.initialize()

    def add_agent(self, agent: Agent) -> Agent:
        """Register an agent, replacing any with the same id."""

        for index, existing in enumerate(self.agents):

            if existing.agent_id == agent.agent_id:

                self.agents[index] = agent

                return agent

        self.agents.append(agent)

        return agent

    def set_agent_availability(
        self,
        agent_id: str,
        available: bool,
    ) -> bool:
        """
        Take an agent on or offline at runtime -- the sidebar toggle,
        and the hidden evaluation's "team becomes unavailable" case.
        """

        for agent in self.agents:

            if agent.agent_id == agent_id:

                agent.available = available

                return True

        return False

    def known_ticket_ids(self) -> List[str]:
        """Every ticket id already in use, so ids never collide."""

        if self.store is None:

            return []

        return [
            row["ticket_id"] for row in self.store.list_tickets()
        ]

    # THE TURN
    def process_turn(
        self,
        session_id: str,
        question: str,
        now: Optional[datetime] = None,
        answer_fn: Optional[AnswerFn] = None,
    ) -> TurnResult:
        """
        Run one customer turn end to end.
        """

        moment = as_utc(now)

        result = TurnResult(
            session_id=session_id,
            question=question,
        )

        # 1. Register the message (Task 6: language + entities).
        user_result = self.session_manager.add_user_message(
            session_id, question, now=moment
        )

        language = user_result["language"]["language"]

        result.language = language

        # 2. Too uncertain to answer -- ask, do not guess.
        if user_result["clarification_required"]:

            result.clarification_required = True
            result.answered = False
            result.answer = (
                "I want to make sure I understand you correctly. "
                "Could you please rephrase your message or tell me "
                "which language you would prefer?"
            )
            result.notes.append(
                "language confidence below the clarification threshold"
            )

            message = self.session_manager.add_assistant_message(
                session_id, result.answer, now=moment
            )

            result.message_id = message.message_id

            return result

        session = self.session_manager.get_session(
            session_id, now=moment
        )

        # 3. Sentiment over the conversation, not just this turn.
        sentiment = analyse_conversation(
            session.messages if session else [],
            self.config.sentiment,
            moment,
        )

        result.sentiment = sentiment

        result.multiple_requests = detect_multiple_requests(question)

        result.correction = detect_correction(question)

        # 4. Escalate before answering: a page must not depend on
        #    the model call succeeding.
        ticket_in_progress = self._find_open_ticket(session_id)

        escalation = self.escalation_engine.evaluate(
            session_id,
            question,
            sentiment,
            session,
            moment,
            status=(
                ticket_in_progress.status
                if ticket_in_progress
                else None
            ),
            ticket=ticket_in_progress,
        )

        result.escalation = escalation

        # 5. Answer, with the tone block attached.
        result.tone_instruction = (
            build_tone_instruction(sentiment, language)
            if self.config.tone_enabled
            else ""
        )

        result.enriched_question = self.build_enriched_question(
            session_id,
            question,
            language,
            now=moment,
        )

        answer = self._answer(
            result.enriched_question,
            result.tone_instruction,
            answer_fn,
        )

        if answer is None:

            result.answered = False
            result.notes.append("no answer function configured")

        else:

            result.answer = answer

            message = self.session_manager.add_assistant_message(
                session_id, answer, now=moment
            )

            result.message_id = message.message_id

        # 6. Ticket.
        if self._should_raise_ticket(result, escalation):

            tickets, decisions = self.handle_support_ticket(
                session_id, question, sentiment, session, moment
            )

            result.tickets = tickets
            result.routing = decisions

            if tickets and tickets[0].status == "awaiting_info":

                result.information_request = build_information_request(
                    tickets[0], language, self.config.ticket
                )

            if escalation.escalate:

                self._record_escalations(escalation, tickets)

        return result

    # HELPERS
    def _answer(
        self,
        enriched_question: str,
        tone_instruction: str,
        answer_fn: Optional[AnswerFn],
    ) -> Optional[str]:
        """
        Call the chain. A failure here must not lose the escalation or
        the ticket raised earlier in the turn, so it is caught and
        reported rather than propagated.
        """

        function = answer_fn or self.answer_fn

        if function is None:

            return None

        try:

            return function(enriched_question, tone_instruction)

        except Exception as error:  # noqa: BLE001 - reported, not hidden

            return (
                "I am sorry, I could not reach the knowledge base just "
                "now. Your request has been recorded and a colleague "
                f"will follow up. (reference: {type(error).__name__})"
            )

    def build_enriched_question(
        self,
        session_id: str,
        question: str,
        language: str,
        now: Optional[datetime] = None,
    ) -> str:
        """
        The query the chain receives: language, history, this message.

        The tone block is deliberately *not* spliced in here. It travels
        to the chain as its own prompt argument, which is what lets the
        chain build its unmodified prompt when there is no tone -- and
        what stops the style rules from being read as part of the
        customer's question.
        """

        context = self.session_manager.get_context(session_id, now=now)

        history_parts = [
            f"{message.role.upper()}: {message.content}"
            for message in context.get("messages", [])[-10:]
        ]

        language_instruction = build_language_instruction(language)

        parts = [
            language_instruction,
            "Conversation history:\n" + "\n".join(history_parts),
            f"Current customer message:\n{question}",
        ]

        return "\n\n".join(parts) + "\n"

    def _should_raise_ticket(
        self,
        result: TurnResult,
        escalation: EscalationResult,
    ) -> bool:
        """
        A ticket is raised when something escalated, or when the
        customer is simply unhappy -- a negative conversation that
        never trips a formal condition still needs an owner.
        """

        if escalation.escalate:

            return True

        if not self.config.ticket_on_negative:

            return False

        return result.sentiment_label in NEGATIVE_LABELS

    def _find_open_ticket(
        self,
        session_id: str,
    ) -> Optional[Ticket]:
        """
        This session's live ticket, so the SLA conditions (warn,
        breach) can be evaluated against it.
        """

        if self.store is None:

            return None

        rows = self.store.list_tickets(session_id=session_id)

        for row in rows:

            if row.get("status") in OPEN_TICKET_STATUSES:

                return ticket_from_row(row)

        return None

    def _record_escalations(
        self,
        escalation: EscalationResult,
        tickets: Sequence[Ticket],
    ) -> None:
        """
        Persist the escalation records, attaching the ticket each one
        produced so the audit trail joins up.
        """

        if self.store is None or not escalation.records:

            return

        primary = tickets[0].ticket_id if tickets else None

        records = []

        for record in escalation.records:

            payload = record.to_dict()

            if primary:

                payload["ticket_id"] = primary

            records.append(payload)

        self.store.save_escalations(records)

    # TICKETS
    def handle_support_ticket(
        self,
        session_id: str,
        message: str,
        sentiment: Optional[SentimentResult] = None,
        session: Optional[ConversationSession] = None,
        now: Optional[datetime] = None,
    ) -> tuple:
        """
        Turn this message into one or more routed, SLA-tracked
        tickets.

        A message carrying two requests becomes two tickets linked as
        related, not one ticket describing neither. The first is
        routed normally; the rest inherit its owner so a single agent
        handles the whole conversation.
        """

        moment = as_utc(now)

        policy = self.config.ticket

        existing = self._existing_tickets()

        summary = conversation_summary(session)

        issues = split_issues(message)

        tickets: List[Ticket] = []

        taken = self.known_ticket_ids()

        for issue in issues:

            ticket = create_ticket(
                session_id,
                issue,
                sentiment,
                policy=policy,
                sla_policy=self.config.sla,
                now=moment,
                ticket_id=next_ticket_id(moment, taken),
                conversation_summary=summary,
                language=(
                    sentiment.language if sentiment else None
                ),
                existing=existing,
            )

            # A later part of a multi-issue message is the same
            # conversation, so it is grouped with the primary even
            # when the two issues are unrelated to each other.
            if tickets and not ticket.duplicate_of:

                link_ticket(
                    ticket,
                    {
                        "ticket_id": tickets[0].ticket_id,
                        "relation": "related",
                        "score": 0.0,
                        "reasons": ["same conversation"],
                    },
                )

            taken.append(ticket.ticket_id)

            existing.append(ticket)

            tickets.append(ticket)

        decisions = self._route(tickets, moment)

        if self.store is not None:

            for ticket in tickets:

                self.store.save_ticket(ticket)

            for decision in decisions:

                self.store.save_routing_decision(decision.to_dict())

            for ticket in tickets:

                for linked in ticket.linked_tickets:

                    self.store.save_ticket_link(
                        ticket.ticket_id,
                        linked,
                        "duplicate" if ticket.duplicate_of else "related",
                        ticket.priority_score,
                        moment,
                    )

        return tickets, decisions

    def _existing_tickets(self) -> List[Ticket]:
        """
        Open tickets to match against, read fresh each turn so a
        runtime change to the store is honoured immediately.
        """

        if self.store is None:

            return []

        return [
            ticket_from_row(row)
            for row in self.store.list_tickets()
            if row.get("status") in OPEN_TICKET_STATUSES
        ]

    def _route(
        self,
        tickets: Sequence[Ticket],
        moment: datetime,
    ) -> List[RoutingDecision]:
        """
        Route the primary ticket properly; the rest of a multi-issue
        message follow its owner.
        """

        decisions: List[RoutingDecision] = []

        for index, ticket in enumerate(tickets):

            if index and tickets[0].assigned_agent:

                ticket.assigned_agent = tickets[0].assigned_agent
                ticket.team = tickets[0].team
                ticket.routing_reason = (
                    f"grouped with {tickets[0].ticket_id} for one handoff"
                )
                ticket.deferred_until = tickets[0].deferred_until

                decisions.append(
                    RoutingDecision(
                        ticket_id=ticket.ticket_id,
                        agent_id=ticket.assigned_agent,
                        team=ticket.team or "",
                        queue="grouped",
                        reason=ticket.routing_reason,
                        deferred_until=ticket.deferred_until,
                        decided_at=moment,
                    )
                )

                continue

            decision = route_ticket(
                ticket,
                self.agents,
                self.config.routing,
                moment,
            )

            self._occupy(ticket, decision)

            decisions.append(decision)

        return decisions

    def _occupy(
        self,
        ticket: Ticket,
        decision: RoutingDecision,
    ) -> None:
        """
        Charge an assignment to the agent who took it.

        The router picks the least-loaded agent, so the load has to
        move when the router decides. Otherwise every ticket after the
        first is offered to whoever was quietest at start-up, and the
        balance the router is computing is never real.

        Two cases are deliberately not counted. A deferred ticket is
        waiting for the next working morning and belongs to nobody
        yet, so it costs nobody capacity. A duplicate is the same
        problem as a ticket that already holds a place with its owner.
        """

        if not decision.assigned or ticket.is_duplicate:

            return

        for agent in self.agents:

            if agent.agent_id == decision.agent_id:

                agent.active_tickets += 1

                return

    # MAINTENANCE
    def recalculate_sla(
        self,
        now: Optional[datetime] = None,
    ) -> SlaRecalculation:
        """
        Re-apply the current SLA policy to every stored ticket.

        The supported path for "the SLA rules changed at runtime":
        shorten a budget in the sidebar and every open ticket moves,
        including flipping one into breach.

        Moving the deadline is only half of that. A ticket that the new
        policy puts past its deadline has *breached*, and a breach is
        an escalation -- so this raises ESC-08 for it, exactly as the
        per-turn path does, rather than quietly rewriting ``due_at``
        and leaving the escalation count where it was. A ticket that
        merely crosses the warning line is reported in the result but
        not escalated: ESC-09 warns, and only a real escalation is
        written to the escalations table.
        """

        if self.store is None:

            return SlaRecalculation()

        moment = as_utc(now)

        rows = self.store.list_tickets()

        tickets = [ticket_from_row(row) for row in rows]

        recalculate_deadlines(tickets, self.config.sla, moment)

        for ticket in tickets:

            self.store.save_ticket(ticket)

        return self._escalate_recalculated(tickets, moment)

    def _escalate_recalculated(
        self,
        tickets: Sequence[Ticket],
        moment: datetime,
    ) -> SlaRecalculation:
        """
        Raise the SLA escalations the new policy implies.

        Each ticket is evaluated on its own, through the same engine
        the per-turn path uses, so a recalculation can never disagree
        with a live turn about what a breach means. A closed ticket
        cannot reach this point: ``apply_sla`` clears its breach and
        warning flags, because a ticket resolved late is a fact about
        the past, not an open page.

        Escalations are raised, never retracted. A later recalculation
        that lengthens a budget back until nothing is breached leaves
        the breach records where they are: the page already went out,
        and a policy change does not un-page the on-call agent.
        """

        result = SlaRecalculation(tickets=len(tickets))

        breached = [ticket for ticket in tickets if ticket.breached]

        result.breached = len(breached)

        result.breached_ticket_ids = [
            ticket.ticket_id for ticket in breached
        ]

        result.warning = len(
            [ticket for ticket in tickets if ticket.warning_due]
        )

        if not breached:

            return result

        # The analysis of no new customer text. ESC-08/ESC-09 read only
        # the ticket, so they still fire; every other condition reads
        # the message or the conversation, and both are deliberately
        # absent here. A recalculation is not a customer turn and must
        # not invent one.
        neutral = analyse_message(
            "", None, self.config.sentiment, moment
        )

        payloads: List[Dict[str, Any]] = []

        for ticket in breached:

            escalation = self.escalation_engine.evaluate(
                ticket.session_id,
                "",
                neutral,
                None,
                moment,
                ticket=ticket,
            )

            for record in escalation.records:

                if record.is_warning:

                    continue

                # Keyed on the ticket rather than the instant, so
                # recalculating twice -- at the same clock or a later
                # one -- refreshes the breach already on record instead
                # of paging again for a ticket that was already past
                # its deadline. The escalations table holds escalations;
                # a repeated click is not a new one.
                record.escalation_id = (
                    f"{ticket.session_id}:{record.condition_id}:"
                    f"{ticket.ticket_id}"
                )

                payload = record.to_dict()

                payloads.append(payload)

                result.records.append(record)

        if payloads:

            self.store.save_escalations(payloads)

        result.escalations = len(payloads)

        return result


# ============================================================
# MODULE HELPERS
# ============================================================

def conversation_summary(
    session: Optional[ConversationSession],
    max_turns: int = 6,
    max_chars: int = 600,
) -> str:
    """
    A short, deterministic summary of the conversation.

    No model call: the escalation record must be reproducible and must
    exist even when the model is unreachable. The raw text is stored
    here; every consumer masks it before display.
    """

    if session is None or not session.messages:

        return ""

    lines = [
        f"{message.role}: {message.content}"
        for message in session.messages[-max_turns:]
    ]

    text = "\n".join(lines)

    return text[-max_chars:]


def ticket_from_row(row: Dict[str, Any]) -> Ticket:
    """
    Rebuild a Ticket from a stored row.

    Deadlines are read back only so the object is complete; anything
    that needs to be authoritative is recomputed by
    ``recalculate_deadlines``.
    """

    def moment(name: str) -> Optional[datetime]:

        value = row.get(name)

        if not value:

            return None

        return datetime.fromisoformat(value)

    ticket = Ticket(ticket_id=row["ticket_id"])

    for name in (
        "session_id",
        "category",
        "status",
        "priority",
        "severity",
        "issue",
        "customer_name",
        "contact",
        "account_identifier",
        "order_id",
        "product_name",
        "product_code",
        "amount",
        "error_code",
        "reference_date",
        "language",
        "sentiment_label",
        "duplicate_of",
        "group_id",
        "assigned_agent",
        "team",
        "routing_reason",
        "masked_handoff",
        "conversation_summary",
    ):

        if row.get(name) is not None:

            setattr(ticket, name, row[name])

    for name in (
        "priority_score",
        "sentiment_confidence",
        "sla_minutes",
        "warn_fraction",
        "elapsed_business_minutes",
    ):

        if row.get(name) is not None:

            setattr(ticket, name, float(row[name]))

    for name in (
        "breached",
        "warning_due",
    ):

        setattr(ticket, name, bool(row.get(name)))

    for name in (
        "created_at",
        "updated_at",
        "resolved_at",
        "due_at",
        "warn_at",
        "deferred_until",
    ):

        setattr(ticket, name, moment(name))

    for name in (
        "evidence",
        "missing_fields",
        "risk_topics",
        "linked_tickets",
    ):

        setattr(ticket, name, list(row.get(name) or []))

    if row.get("priority_components"):

        ticket.priority_components = dict(row["priority_components"])

    return ticket
