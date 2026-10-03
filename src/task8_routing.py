"""
TASK 8.2 - ROUTING

Decides which team and which agent picks up a ticket, and when.

Design rules this module follows, and why:

* **A decision is data, not a side effect.** ``route_ticket`` returns
  a ``RoutingDecision`` and *also* stamps the ticket, so the same call
  can be logged, replayed, or asserted on without reading the ticket
  back.

* **Every decision explains itself.** ``reason`` records the rules
  that fired -- team match, language fallback, load, on-call, deferral
  -- because "why did this go to Ravi" is the first question asked of
  any router.

* **Deterministic.** Candidates are sorted by load then agent id, so
  two runs over the same state pick the same agent. No randomness, no
  dict ordering.

* **After hours is a queue decision, not a failure.** A P1 at 22:00
  goes to on-call; a polite P3 at 22:00 is scheduled for the next
  working morning rather than waking somebody up.

* **A duplicate does not get a second owner.** It is the same
  problem, so it is grouped with the original and routed to the
  original's agent when there is one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from business_calendar import (
    BusinessCalendarConfig,
    is_after_hours,
    next_business_start,
)
from support_deps import as_utc
from task8_ticketing import Ticket


# ============================================================
# CONSTANTS
# ============================================================

# Queue names. These are the hand-off targets a downstream worker
# subscribes to, so they are stable strings rather than free text.
QUEUE_PRIORITY = "priority_support"
QUEUE_ON_CALL = "on_call"
QUEUE_SCHEDULED = "scheduled_next_working_day"
QUEUE_TEAM = "team_queue"
QUEUE_FALLBACK = "general_fallback"
QUEUE_DUPLICATE = "duplicate_review"

UNASSIGNED = None


# ============================================================
# CONFIG
# ============================================================

@dataclass
class RoutingConfig:
    """
    Routing rules.

    Mutable by design: the evaluation may add a team, move a category,
    or take a team offline at runtime, and the next call must honour
    it without any cached state.
    """

    # Which team owns which ticket category.
    team_for_category: Dict[str, str] = field(
        default_factory=lambda: {
            "payment": "billing",
            "account": "security",
            "delivery": "logistics",
            "product": "product",
            "technical": "tech",
            "general": "general",
        }
    )

    on_call_team: str = "on_call"
    fallback_team: str = "general"

    # Priorities that must not wait for the next working day.
    immediate_priorities: Tuple[str, ...] = ("P1", "P2")

    # Severities that must not wait, whatever the priority band.
    immediate_severities: Tuple[str, ...] = ("critical",)

    # An agent may only be considered if they have spare capacity.
    respect_capacity: bool = True

    # Prefer an agent who speaks the customer's language. When nobody
    # does, fall back rather than leave the ticket unrouted.
    prefer_language_match: bool = True

    # A copy of a working agent's day, used to decide after-hours.
    calendar: BusinessCalendarConfig = field(
        default_factory=BusinessCalendarConfig
    )

    def team_for(self, ticket: Ticket) -> str:
        """
        The owning team for this ticket's category.
        """

        return self.team_for_category.get(
            ticket.category, self.fallback_team
        )

    def calendar_for_routing(self) -> BusinessCalendarConfig:
        """
        A copy of the calendar, so routing cannot mutate the config it
        was handed.
        """

        return BusinessCalendarConfig(
            **{
                name: getattr(self.calendar, name)
                for name in self.calendar.__dataclass_fields__
            }
        )


# ============================================================
# AGENT
# ============================================================

@dataclass
class Agent:
    """
    One support agent, and what they can currently take.
    """

    agent_id: str

    # team before name deliberately: `Agent("A1", "billing")` is how
    # every call site wants to write it, and a name in that slot
    # silently produces an agent belonging to no team at all.
    team: str = ""
    name: str = ""
    skills: Tuple[str, ...] = ()
    languages: Tuple[str, ...] = ()
    available: bool = True
    active_tickets: int = 0
    max_capacity: int = 5
    on_call: bool = False

    @property
    def load(self) -> float:
        """
        Fraction of capacity in use.
        """

        if self.max_capacity <= 0:

            return 1.0

        return self.active_tickets / self.max_capacity

    @property
    def has_capacity(self) -> bool:
        return self.active_tickets < self.max_capacity

    def speaks(self, language: Optional[str]) -> bool:
        """
        True when the agent covers this language (or the caller does
        not care).
        """

        if not language:

            return True

        return language.lower() in {
            item.lower() for item in self.languages
        }

    def can_work(self, category: str, respect_capacity: bool = True) -> bool:
        """
        Available, and either skilled for this category or a generalist
        on the owning team. Skills are advisory: a team member who has
        not listed the skill is still better than an unassigned
        ticket.
        """

        if not self.available:

            return False

        if respect_capacity and not self.has_capacity:

            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "team": self.team,
            "name": self.name,
            "skills": list(self.skills),
            "languages": list(self.languages),
            "available": self.available,
            "active_tickets": self.active_tickets,
            "max_capacity": self.max_capacity,
            "on_call": self.on_call,
        }


# ============================================================
# DECISION
# ============================================================

@dataclass
class RoutingDecision:
    """
    What routing decided, and why.
    """

    ticket_id: str
    agent_id: Optional[str] = None
    team: str = ""
    queue: str = QUEUE_TEAM
    reason: str = ""
    deferred_until: Optional[datetime] = None
    after_hours: bool = False
    on_call: bool = False
    language_fallback: bool = False
    candidates_considered: int = 0
    runner_up: Optional[str] = None
    decided_at: Optional[datetime] = None

    @property
    def assigned(self) -> bool:
        return self.agent_id is not None

    @property
    def is_deferred(self) -> bool:
        return self.deferred_until is not None

    def to_dict(self) -> Dict[str, Any]:
        def stamp(value: Optional[datetime]) -> Optional[str]:
            return as_utc(value).isoformat() if value else None

        return {
            "ticket_id": self.ticket_id,
            "agent_id": self.agent_id,
            "team": self.team,
            "queue": self.queue,
            "reason": self.reason,
            "deferred_until": stamp(self.deferred_until),
            "after_hours": self.after_hours,
            "on_call": self.on_call,
            "language_fallback": self.language_fallback,
            "candidates_considered": self.candidates_considered,
            "runner_up": self.runner_up,
            "decided_at": stamp(self.decided_at),
        }


# ============================================================
# ROUTING
# ============================================================

def needs_immediate_attention(
    ticket: Ticket,
    config: Optional[RoutingConfig] = None,
) -> bool:
    """
    True when this ticket cannot wait for the next working day.

    Priority band, severity and an actual SLA breach all qualify: a
    ticket that has already breached is late whatever its band said
    when it was created.
    """

    config = config or RoutingConfig()

    if ticket.priority in config.immediate_priorities:

        return True

    if ticket.severity in config.immediate_severities:

        return True

    return bool(ticket.breached)


def _rank(
    agents: Sequence[Agent],
    ticket: Ticket,
    config: RoutingConfig,
) -> Tuple[List[Agent], bool]:
    """
    Order candidates best-first and report whether the language
    preference had to be dropped.

    Sorted by load then agent id: the least-loaded agent wins, and a
    tie is broken deterministically rather than by dict order.
    """

    def by_load(agent: Agent) -> Tuple[float, str]:
        return (agent.load, agent.agent_id)

    if not config.prefer_language_match:

        return sorted(agents, key=by_load), False

    speaking = [agent for agent in agents if agent.speaks(ticket.language)]

    if speaking:

        return sorted(speaking, key=by_load), False

    return sorted(agents, key=by_load), True


def route_ticket(
    ticket: Ticket,
    agents: Sequence[Agent],
    config: Optional[RoutingConfig] = None,
    now: Optional[datetime] = None,
) -> RoutingDecision:
    """
    Choose a team, an agent and a queue for this ticket, and stamp the
    ticket with the outcome.
    """

    config = config or RoutingConfig()

    moment = as_utc(now)

    calendar = config.calendar_for_routing()

    decision = RoutingDecision(
        ticket_id=ticket.ticket_id,
        decided_at=moment,
    )

    after_hours = is_after_hours(moment, calendar)

    decision.after_hours = after_hours

    # A duplicate is the same problem as its original: group it and
    # send it to whoever already owns that problem.
    if ticket.is_duplicate:

        decision.agent_id = ticket.assigned_agent
        decision.team = config.team_for(ticket)
        decision.queue = QUEUE_DUPLICATE
        decision.reason = (
            f"duplicate of {ticket.duplicate_of}; grouped for one "
            f"handoff rather than routed separately"
        )

        _apply(ticket, decision, moment)

        return decision

    team = config.team_for(ticket)

    immediate = needs_immediate_attention(ticket, config)

    # After hours: either wake somebody, or schedule the morning.
    if after_hours:

        if immediate:

            decision.team = config.on_call_team
            decision.on_call = True
            decision.queue = QUEUE_ON_CALL
            decision.reason = (
                f"{ticket.priority}/{ticket.severity} outside business "
                f"hours; routed to the on-call queue"
            )

            on_call_agents = [
                agent
                for agent in agents
                if agent.team == config.on_call_team
                and agent.available
            ]

            decision.candidates_considered = len(on_call_agents)

            if on_call_agents:

                ranked, fallback = _rank(on_call_agents, ticket, config)

                decision.agent_id = ranked[0].agent_id

                decision.language_fallback = fallback

                if len(ranked) > 1:

                    decision.runner_up = ranked[1].agent_id

                decision.reason += f"; paged {ranked[0].agent_id}"

            else:

                decision.reason += (
                    "; no on-call agent available, left in the on-call "
                    "queue for the next pickup"
                )

            _apply(ticket, decision, moment)

            return decision

        # Not urgent: do not page anyone. Schedule it.
        decision.team = team
        decision.queue = QUEUE_SCHEDULED
        decision.deferred_until = next_business_start(moment, calendar)
        decision.reason = (
            f"{ticket.priority} outside business hours and not urgent; "
            f"deferred to the next business start"
        )

        # A deferred ticket is deliberately left unassigned: it is
        # waiting for the morning, so committing it to an agent's
        # capacity now would overstate that agent's load and could
        # defer somebody else's work in turn.
        _apply(ticket, decision, moment)

        return decision

    else:

        decision.team = team
        decision.queue = QUEUE_TEAM
        decision.reason = f"category {ticket.category} belongs to {team}"

    # Pick an agent from the owning team.
    candidates = [
        agent
        for agent in agents
        if agent.team == decision.team
        and agent.can_work(ticket.category, config.respect_capacity)
    ]

    decision.candidates_considered = len(candidates)

    if not candidates:

        decision.reason += f"; no available agent on {decision.team}"

        if immediate:

            decision.team = config.on_call_team
            decision.queue = QUEUE_ON_CALL
            decision.on_call = True
            decision.reason += "; escalated to the on-call queue"

        else:

            decision.queue = QUEUE_FALLBACK
            decision.reason += (
                f"; queued for {config.fallback_team} to pick up"
            )

        _apply(ticket, decision, moment)

        return decision

    ranked, language_fallback = _rank(candidates, ticket, config)

    decision.agent_id = ranked[0].agent_id
    decision.language_fallback = language_fallback

    if len(ranked) > 1:

        decision.runner_up = ranked[1].agent_id

    decision.reason += (
        f"; assigned {ranked[0].agent_id} at "
        f"{ranked[0].active_tickets}/{ranked[0].max_capacity} load"
    )

    if language_fallback:

        decision.reason += (
            f"; no {decision.team} agent speaks {ticket.language}, "
            f"routed without a language match"
        )

    _apply(ticket, decision, moment)

    return decision


def _apply(
    ticket: Ticket,
    decision: RoutingDecision,
    moment: datetime,
) -> None:
    """
    Stamp the decision onto the ticket, in the fields the rest of the
    system already reads.
    """

    ticket.assigned_agent = decision.agent_id
    ticket.team = decision.team
    ticket.routing_reason = decision.reason
    ticket.deferred_until = decision.deferred_until
    ticket.updated_at = moment


def assign_workload(
    ticket: Ticket,
    agents: Sequence[Agent],
    config: Optional[RoutingConfig] = None,
) -> Optional[Agent]:
    """
    The agent a ticket would go to right now, without side effects.

    Useful for capacity planning and for showing a dispatcher what the
    router is about to do.
    """

    config = config or RoutingConfig()

    team = config.team_for(ticket)

    candidates = [
        agent
        for agent in agents
        if agent.team == team
        and agent.can_work(ticket.category, config.respect_capacity)
    ]

    if not candidates:

        return None

    return _rank(candidates, ticket, config)[0][0]


def summarize_routing(
    decisions: Sequence[RoutingDecision],
) -> Dict[str, int]:
    """
    Counts by queue, for monitoring.
    """

    counts: Dict[str, int] = {}

    for decision in decisions:

        counts[decision.queue] = counts.get(decision.queue, 0) + 1

    return counts


def team_load(
    agents: Sequence[Agent],
) -> Dict[str, float]:
    """
    Total load per team, so an overloaded queue is visible before it
    starts breaching SLAs.
    """

    totals: Dict[str, float] = {}

    for agent in agents:

        totals[agent.team] = totals.get(agent.team, 0.0) + agent.load

    return totals
