"""
TASK 8.2 - ROUTING TEST SUITE

Covers:
- Category to team mapping, including an unknown category
- Skill/team matching and the capacity and availability rules
- Workload balance and the deterministic tie-break
- Language preference, and the recorded fallback when nobody matches
- After hours: urgent to on-call, ordinary deferred to the next
  business start
- An unavailable team: on-call for P1, fallback queue otherwise
- A duplicate getting no second owner
- An SLA breach forcing immediate attention whatever the band
- Determinism, serialisation, and routing obeying a runtime change
"""

from datetime import datetime, timedelta, timezone

from business_calendar import BusinessCalendarConfig
from task8_routing import (
    QUEUE_DUPLICATE,
    QUEUE_FALLBACK,
    QUEUE_ON_CALL,
    QUEUE_SCHEDULED,
    QUEUE_TEAM,
    Agent,
    RoutingConfig,
    assign_workload,
    needs_immediate_attention,
    route_ticket,
    summarize_routing,
    team_load,
)
from task8_ticketing import Ticket


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
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


def config(**overrides):
    overrides.setdefault(
        "calendar", BusinessCalendarConfig(timezone_name="UTC")
    )

    return RoutingConfig(**overrides)


print("=" * 75)
print("TASK 8.2 ROUTING TESTS")
print("=" * 75)

# Mon 2026-10-05 10:00 UTC is inside business hours; 22:00 is not.
IN_HOURS = utc(2026, 10, 5, 10, 0)
AFTER_HOURS = utc(2026, 10, 5, 22, 0)
SATURDAY = utc(2026, 10, 3, 11, 0)

CONFIG = config()


def agent(agent_id, team, **overrides):
    overrides.setdefault("languages", ("english",))
    overrides.setdefault("max_capacity", 5)

    return Agent(agent_id=agent_id, team=team, **overrides)


TEAM_OF_TWO = [
    agent("AGT-1", "billing", active_tickets=3),
    agent("AGT-2", "billing", active_tickets=1),
]


def ticket(**overrides):
    overrides.setdefault("ticket_id", "TCK-R")
    overrides.setdefault("category", "payment")
    overrides.setdefault("priority", "P3")
    overrides.setdefault("severity", "medium")
    overrides.setdefault("language", "english")

    return Ticket(**overrides)


# ============================================================
# 1. TEAM MAPPING
# ============================================================

print("\n[1] TEAM MAPPING")

for category, expected in (
    ("payment", "billing"),
    ("account", "security"),
    ("delivery", "logistics"),
    ("product", "product"),
    ("technical", "tech"),
    ("general", "general"),
):

    check(
        f"{category} belongs to {expected}",
        CONFIG.team_for(ticket(category=category)) == expected,
        CONFIG.team_for(ticket(category=category)),
    )

check(
    "an unknown category falls back",
    CONFIG.team_for(ticket(category="nonsense")) == CONFIG.fallback_team,
)

check(
    "routing does not mutate the config's calendar",
    config().calendar_for_routing() is not CONFIG.calendar,
)

custom = config(team_for_category={"payment": "revenue"})

check(
    "the team map is reconfigurable at runtime",
    custom.team_for(ticket(category="payment")) == "revenue",
)

check(
    "an unmapped category still falls back after reconfiguration",
    custom.team_for(ticket(category="account")) == "general",
)


# ============================================================
# 2. AGENT SELECTION
# ============================================================

print("\n[2] AGENT SELECTION")

decision = route_ticket(
    ticket(ticket_id="TCK-1"), TEAM_OF_TWO, CONFIG, IN_HOURS
)

check(
    "the owner team is chosen",
    decision.team == "billing",
    decision.team,
)

check(
    "the least-loaded agent wins",
    decision.agent_id == "AGT-2",
    f"got {decision.agent_id} ({decision.reason})",
)

check(
    "the queue is the team queue",
    decision.queue == QUEUE_TEAM,
)

check(
    "the decision is marked assigned",
    decision.assigned is True,
)

check(
    "the runner-up is recorded",
    decision.runner_up == "AGT-1",
    str(decision.runner_up),
)

check(
    "both candidates were considered",
    decision.candidates_considered == 2,
)

check(
    "the reason names the load",
    "1/5 load" in decision.reason,
    decision.reason,
)

routed_ticket = ticket(ticket_id="TCK-STAMP")

route_ticket(routed_ticket, TEAM_OF_TWO, CONFIG, IN_HOURS)

check(
    "the ticket records the agent, team and reason",
    routed_ticket.assigned_agent == "AGT-2"
    and routed_ticket.team == "billing"
    and routed_ticket.routing_reason,
    f"{routed_ticket.assigned_agent} / {routed_ticket.team}",
)

check(
    "the ticket's updated_at is the routed instant",
    routed_ticket.updated_at == IN_HOURS,
)

# A tie is broken by agent id, never by iteration order.
TIED = [
    agent("AGT-B", "billing", active_tickets=2),
    agent("AGT-A", "billing", active_tickets=2),
]

check(
    "a load tie breaks on agent id",
    route_ticket(
        ticket(ticket_id="TCK-2"), TIED, CONFIG, IN_HOURS
    ).agent_id
    == "AGT-A",
)

check(
    "a tie is stable when the input order is reversed",
    route_ticket(
        ticket(ticket_id="TCK-3"), list(reversed(TIED)), CONFIG, IN_HOURS
    ).agent_id
    == "AGT-A",
)

# Availability.
check(
    "an unavailable agent is skipped",
    route_ticket(
        ticket(ticket_id="TCK-4"),
        [
            agent("AGT-OFF", "billing", available=False),
            agent("AGT-ON", "billing", active_tickets=4),
        ],
        CONFIG,
        IN_HOURS,
    ).agent_id
    == "AGT-ON",
)

# Capacity.
check(
    "a full agent is skipped",
    route_ticket(
        ticket(ticket_id="TCK-5"),
        [
            agent("AGT-FULL", "billing", active_tickets=5, max_capacity=5),
            agent("AGT-FREE", "billing", active_tickets=4),
        ],
        CONFIG,
        IN_HOURS,
    ).agent_id
    == "AGT-FREE",
)

check(
    "capacity is ignored when the policy says so",
    route_ticket(
        ticket(ticket_id="TCK-6"),
        [
            agent("AGT-FULL", "billing", active_tickets=9, max_capacity=5),
            agent("AGT-FREE", "billing", active_tickets=1),
        ],
        config(respect_capacity=False),
        IN_HOURS,
    ).agent_id
    == "AGT-FREE",
)

check(
    "a zero-capacity agent reads as fully loaded",
    agent("AGT-Z", "billing", max_capacity=0).load == 1.0,
)

check(
    "another team's agent is never selected",
    route_ticket(
        ticket(ticket_id="TCK-7", category="account"),
        TEAM_OF_TWO,
        CONFIG,
        IN_HOURS,
    ).agent_id
    is None,
)


# ============================================================
# 3. LANGUAGE
# ============================================================

print("\n[3] LANGUAGE")

MIXED = [
    agent("AGT-EN", "billing", active_tickets=1, languages=("english",)),
    agent("AGT-HI", "billing", active_tickets=4, languages=("hindi",)),
]

check(
    "a language match beats a lower load",
    route_ticket(
        ticket(ticket_id="TCK-8", language="hindi"),
        MIXED,
        CONFIG,
        IN_HOURS,
    ).agent_id
    == "AGT-HI",
)

check(
    "a language match is not flagged as a fallback",
    route_ticket(
        ticket(ticket_id="TCK-9", language="hindi"),
        MIXED,
        CONFIG,
        IN_HOURS,
    ).language_fallback
    is False,
)

fallback = route_ticket(
    ticket(ticket_id="TCK-10", language="telugu"), MIXED, CONFIG, IN_HOURS
)

check(
    "nobody speaking the language still routes",
    fallback.agent_id is not None,
    fallback.reason,
)

check(
    "the language fallback is recorded",
    fallback.language_fallback is True,
)

check(
    "the fallback reason names the language",
    "telugu" in fallback.reason,
    fallback.reason,
)

check(
    "the fallback picks the least loaded agent",
    fallback.agent_id == "AGT-EN",
)

check(
    "the language preference can be switched off",
    route_ticket(
        ticket(ticket_id="TCK-11", language="telugu"),
        MIXED,
        config(prefer_language_match=False),
        IN_HOURS,
    ).language_fallback
    is False,
)


# ============================================================
# 4. AFTER HOURS
# ============================================================

print("\n[4] AFTER HOURS")

ON_CALL = [
    agent("AGT-OC", "on_call", on_call=True, active_tickets=0),
    *TEAM_OF_TWO,
]

urgent = route_ticket(
    ticket(ticket_id="TCK-12", priority="P1", severity="high"),
    ON_CALL,
    CONFIG,
    AFTER_HOURS,
)

check(
    "an urgent after-hours ticket goes to on-call",
    urgent.queue == QUEUE_ON_CALL and urgent.on_call is True,
    urgent.reason,
)

check(
    "the on-call agent is paged",
    urgent.agent_id == "AGT-OC",
    str(urgent.agent_id),
)

check(
    "the on-call decision records the hour",
    urgent.after_hours is True,
)

check(
    "an after-hours urgent ticket is not deferred",
    urgent.deferred_until is None,
)

ordinary = route_ticket(
    ticket(ticket_id="TCK-13", priority="P3", severity="low"),
    ON_CALL,
    CONFIG,
    AFTER_HOURS,
)

check(
    "an ordinary after-hours ticket is scheduled, not paged",
    ordinary.queue == QUEUE_SCHEDULED,
    ordinary.reason,
)

check(
    "the schedule is the next business start",
    ordinary.deferred_until == utc(2026, 10, 6, 9, 0),
    str(ordinary.deferred_until),
)

check(
    "a deferred ticket wakes nobody",
    ordinary.agent_id is None,
    str(ordinary.agent_id),
)

deferred_ticket = Ticket(
    ticket_id="TCK-D", category="payment", priority="P3"
)

route_ticket(deferred_ticket, ON_CALL, CONFIG, AFTER_HOURS)

check(
    "the deferral is stamped on the ticket",
    deferred_ticket.deferred_until == utc(2026, 10, 6, 9, 0),
    str(deferred_ticket.deferred_until),
)

# A Friday night defers over the weekend.
friday_night = route_ticket(
    ticket(ticket_id="TCK-14", priority="P3", severity="low"),
    ON_CALL,
    CONFIG,
    utc(2026, 10, 2, 22, 0),
)

check(
    "a Friday night deferral skips the weekend",
    friday_night.deferred_until == utc(2026, 10, 5, 9, 0),
    str(friday_night.deferred_until),
)

check(
    "a Saturday request is after hours",
    route_ticket(
        ticket(ticket_id="TCK-15", priority="P3"),
        ON_CALL,
        CONFIG,
        SATURDAY,
    ).after_hours
    is True,
)

check(
    "an urgent Saturday request pages on-call",
    route_ticket(
        ticket(ticket_id="TCK-16", priority="P1", severity="high"),
        ON_CALL,
        CONFIG,
        SATURDAY,
    ).queue
    == QUEUE_ON_CALL,
)

check(
    "with no on-call agent the ticket stays in the on-call queue",
    route_ticket(
        ticket(ticket_id="TCK-17", priority="P1", severity="high"),
        TEAM_OF_TWO,
        CONFIG,
        AFTER_HOURS,
    ).queue
    == QUEUE_ON_CALL,
)

check(
    "no on-call agent means no assignment, not a wrong one",
    route_ticket(
        ticket(ticket_id="TCK-18", priority="P1", severity="high"),
        TEAM_OF_TWO,
        CONFIG,
        AFTER_HOURS,
    ).agent_id
    is None,
)

check(
    "a critical severity is urgent even in a low band",
    needs_immediate_attention(
        ticket(priority="P4", severity="critical"), CONFIG
    )
    is True,
)

check(
    "an ordinary ticket is not urgent",
    needs_immediate_attention(
        ticket(priority="P3", severity="low"), CONFIG
    )
    is False,
)

check(
    "a breached ticket is urgent whatever its band",
    needs_immediate_attention(
        ticket(priority="P4", severity="low", breached=True), CONFIG
    )
    is True,
)


# ============================================================
# 5. UNAVAILABLE TEAMS AND DUPLICATES
# ============================================================

print("\n[5] UNAVAILABLE TEAMS AND DUPLICATES")

IDLE_TEAM = [agent("AGT-OTHER", "logistics")]

no_billing_urgent = route_ticket(
    ticket(ticket_id="TCK-19", priority="P1", severity="high"),
    IDLE_TEAM,
    CONFIG,
    IN_HOURS,
)

check(
    "an urgent ticket with no team available escalates to on-call",
    no_billing_urgent.queue == QUEUE_ON_CALL
    and no_billing_urgent.on_call is True,
    no_billing_urgent.reason,
)

check(
    "the escalation reason says the team was unavailable",
    "no available agent on billing" in no_billing_urgent.reason,
    no_billing_urgent.reason,
)

no_billing_ordinary = route_ticket(
    ticket(ticket_id="TCK-20", priority="P3", severity="low"),
    IDLE_TEAM,
    CONFIG,
    IN_HOURS,
)

check(
    "an ordinary ticket with no team available falls back",
    no_billing_ordinary.queue == QUEUE_FALLBACK,
    no_billing_ordinary.reason,
)

check(
    "a fallback is not paged to on-call",
    no_billing_ordinary.on_call is False,
)

check(
    "an empty agent list routes to the fallback queue",
    route_ticket(
        ticket(ticket_id="TCK-21"), [], CONFIG, IN_HOURS
    ).queue
    == QUEUE_FALLBACK,
)

duplicate = ticket(
    ticket_id="TCK-22", duplicate_of="TCK-ORIG", assigned_agent=None
)

duplicate_decision = route_ticket(
    duplicate, TEAM_OF_TWO, CONFIG, IN_HOURS
)

check(
    "a duplicate is not routed separately",
    duplicate_decision.queue == QUEUE_DUPLICATE,
    duplicate_decision.reason,
)

check(
    "the duplicate decision names the original",
    "TCK-ORIG" in duplicate_decision.reason,
    duplicate_decision.reason,
)

check(
    "a duplicate consumes no new agent's capacity",
    duplicate_decision.agent_id is None,
)

check(
    "a duplicate never goes to on-call",
    route_ticket(
        Ticket(
            ticket_id="TCK-23",
            category="payment",
            priority="P1",
            severity="critical",
            duplicate_of="TCK-ORIG",
        ),
        ON_CALL,
        CONFIG,
        AFTER_HOURS,
    ).queue
    == QUEUE_DUPLICATE,
)


# ============================================================
# 6. DETERMINISM AND REPORTING
# ============================================================

print("\n[6] DETERMINISM AND REPORTING")

first = route_ticket(
    ticket(ticket_id="TCK-24"), TEAM_OF_TWO, CONFIG, IN_HOURS
)

second = route_ticket(
    ticket(ticket_id="TCK-24"), TEAM_OF_TWO, CONFIG, IN_HOURS
)

check(
    "routing is deterministic",
    first.to_dict() == second.to_dict(),
)

check(
    "routing does not mutate the agent list",
    [item.active_tickets for item in TEAM_OF_TWO] == [3, 1],
)

serialised = first.to_dict()

check(
    "the decision serialises its simulated instant",
    serialised["decided_at"] == "2026-10-05T10:00:00+00:00",
    str(serialised["decided_at"]),
)

check(
    "an unassigned decision serialises a null agent",
    route_ticket(
        ticket(ticket_id="TCK-25", category="account"),
        TEAM_OF_TWO,
        CONFIG,
        IN_HOURS,
    ).to_dict()["agent_id"]
    is None,
)

check(
    "the deferred instant serialises as a string",
    isinstance(
        ordinary.to_dict()["deferred_until"], str
    ),
    str(ordinary.to_dict()["deferred_until"]),
)

check(
    "queues are summarised",
    summarize_routing([first, ordinary, urgent])["on_call"] == 1,
    str(summarize_routing([first, ordinary, urgent])),
)

check(
    "team load is reported per team",
    team_load(TEAM_OF_TWO)["billing"] == 0.8,
    str(team_load(TEAM_OF_TWO)),
)

check(
    "assign_workload previews the choice without side effects",
    assign_workload(ticket(ticket_id="TCK-26"), TEAM_OF_TWO, CONFIG).agent_id
    == "AGT-2",
)

check(
    "assign_workload returns None when nobody is eligible",
    assign_workload(
        ticket(ticket_id="TCK-27", category="account"), TEAM_OF_TWO, CONFIG
    )
    is None,
)

# Runtime change: take the preferred agent offline and re-run.
TEAM_OF_TWO[1].available = False

check(
    "a runtime availability change moves the next ticket",
    route_ticket(
        ticket(ticket_id="TCK-28"), TEAM_OF_TWO, CONFIG, IN_HOURS
    ).agent_id
    == "AGT-1",
)

TEAM_OF_TWO[1].available = True

check(
    "restoring availability restores the choice",
    route_ticket(
        ticket(ticket_id="TCK-29"), TEAM_OF_TWO, CONFIG, IN_HOURS
    ).agent_id
    == "AGT-2",
)

# A runtime calendar change moves the after-hours boundary.
LATE = utc(2026, 10, 5, 20, 0)

check(
    "20:00 is after hours on a 09:00-18:00 day",
    route_ticket(
        ticket(ticket_id="TCK-30", priority="P3"), ON_CALL, CONFIG, LATE
    ).after_hours
    is True,
)

late_config = config(
    calendar=BusinessCalendarConfig(
        timezone_name="UTC", business_hours_end="21:00"
    )
)

check(
    "extending the day at runtime makes 20:00 business hours",
    route_ticket(
        ticket(ticket_id="TCK-31", priority="P3"),
        ON_CALL,
        late_config,
        LATE,
    ).after_hours
    is False,
)

check(
    "the extended day routes to the owning team again",
    route_ticket(
        ticket(ticket_id="TCK-32", priority="P3"),
        ON_CALL,
        late_config,
        LATE,
    ).queue
    == QUEUE_TEAM,
)

check(
    "the config's own calendar is untouched by routing",
    CONFIG.calendar.business_hours_end == "18:00",
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 75)
print(f"PASSED: {PASS}")
print(f"FAILED: {FAIL}")
print("=" * 75)

raise SystemExit(0 if FAIL == 0 else 1)
