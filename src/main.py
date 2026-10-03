"""
CUSTOMER SERVICE CHATBOT -- STREAMLIT UI

The UI renders; ``support_flow.SupportFlow`` decides. One customer
turn goes into ``process_turn`` and comes back carrying the answer,
the sentiment, any escalation and any ticket that was raised.

The sidebar is the demo surface for the things the briefs say can
change at runtime: the clock (so after-hours and SLA behaviour can be
shown without waiting for the evening), the SLA budgets, and which
agents are on shift.
"""

import uuid
from datetime import date, datetime, time, timezone

import streamlit as st

from langchain_helper import create_vector_db, get_qa_chain

from business_calendar import BusinessCalendarConfig
from support_flow import SupportFlow, SupportFlowConfig
from support_store import SupportStore
from task6_multilingual import SessionManager, Task6Config
from task7_escalation import EscalationConfig
from task8_routing import Agent, RoutingConfig
from task8_ticketing import SlaPolicy


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Service Chatbot",
    page_icon="🤖",
)


# ============================================================
# SUPPORT FLOW SETUP
# ============================================================

if "task6_session_id" not in st.session_state:
    st.session_state.task6_session_id = str(uuid.uuid4())

if "support_flow" not in st.session_state:

    calendar = BusinessCalendarConfig(
        timezone_name="Asia/Kolkata",
        business_hours_start="09:00",
        business_hours_end="18:00",
    )

    flow = SupportFlow(
        session_manager=SessionManager(
            Task6Config(
                max_context_messages=10,
                inactivity_minutes=30,
                restore_window_hours=24,
            )
        ),
        config=SupportFlowConfig(
            escalation=EscalationConfig(calendar=calendar),
            sla=SlaPolicy(calendar=calendar),
            routing=RoutingConfig(calendar=calendar),
        ),
        store=SupportStore(),
        agents=[
            Agent(
                "AGT-101", "billing", name="Asha",
                languages=("english", "hindi"), active_tickets=1,
            ),
            Agent(
                "AGT-201", "security", name="Ravi",
                languages=("english",), active_tickets=0,
            ),
            Agent(
                "AGT-301", "logistics", name="Meera",
                languages=("english", "telugu", "spanish"),
                active_tickets=2,
            ),
            Agent(
                "AGT-401", "tech", name="Dev",
                languages=("english",), active_tickets=0,
            ),
            Agent("AGT-901", "on_call", name="On-call", on_call=True),
        ],
    )

    st.session_state.support_flow = flow

flow = st.session_state.support_flow

session_manager = flow.session_manager
session_id = st.session_state.task6_session_id

# Restore an inactive session when it is still within the
# configured 24-hour restoration window. If restoration has
# expired, start a fresh session.
existing_session = session_manager.get_session(session_id)

if existing_session is None:
    session_manager.create_session(session_id)
elif existing_session.closed:
    restoration = session_manager.restore_session(session_id)

    if not restoration.get("restored", False):
        session_manager.create_session(session_id)


# ============================================================
# SIDEBAR -- RUNTIME CONTROLS
# ============================================================

st.sidebar.header("🕒 Simulated clock")

st.sidebar.caption(
    "Move the clock to demonstrate after-hours escalation and SLA "
    "warnings without waiting for the evening."
)

simulate = st.sidebar.checkbox("Use a simulated time", value=False)

if simulate:

    default_moment = datetime.now(timezone.utc)

    chosen_date = st.sidebar.date_input(
        "Date (UTC)", value=default_moment.date()
    )

    chosen_time = st.sidebar.time_input(
        "Time (UTC)", value=default_moment.time().replace(
            second=0, microsecond=0
        )
    )

    now = datetime.combine(
        chosen_date, chosen_time, tzinfo=timezone.utc
    )

    st.sidebar.info(f"Now: {now.isoformat()}")

else:

    now = None


st.sidebar.divider()

st.sidebar.header("⏱ SLA policy")

st.sidebar.caption(
    "Change a budget and every open ticket is recalculated -- "
    "including flipping one into breach."
)

sla = flow.config.sla

for priority in ("P1", "P2", "P3", "P4"):

    sla.resolution_minutes_by_priority[priority] = st.sidebar.number_input(
        f"{priority} resolution (business minutes)",
        min_value=5,
        max_value=10080,
        step=15,
        value=int(sla.resolution_minutes_by_priority.get(priority, 480)),
        key=f"sla_{priority}",
    )

sla.warn_fraction = st.sidebar.slider(
    "Warn at this fraction of the budget",
    min_value=0.10,
    max_value=1.00,
    step=0.05,
    value=float(sla.warn_fraction),
)

if st.sidebar.button("🔄 Recalculate every open ticket"):

    outcome = flow.recalculate_sla(now)

    st.sidebar.success(outcome.summary())

    for record in outcome.records:

        st.sidebar.warning(
            f"🚨 {record.ticket_id or 'ticket'} · "
            f"{record.condition_id} · {record.priority or ''}\n\n"
            f"{record.reason}"
        )

st.sidebar.divider()

st.sidebar.header("👥 Agents on shift")

for agent in flow.agents:

    agent.available = st.sidebar.checkbox(
        f"{agent.agent_id} · {agent.name} ({agent.team})",
        value=agent.available,
        key=f"agent_{agent.agent_id}",
    )

    st.sidebar.caption(
        f"load {agent.active_tickets}/{agent.max_capacity}"
    )

st.sidebar.divider()

st.sidebar.caption(
    f"🎫 Stored tickets: {len(flow.store.list_tickets())}"
)

st.sidebar.caption(
    f"🚨 Recorded escalations: {flow.store.count_escalations()}"
)


# ============================================================
# UI
# ============================================================

st.title("🤖 CUSTOMER SERVICE CHATBOT")

st.caption(
    "Multilingual AI customer support with sentiment-aware "
    "escalation, ticketing and SLA tracking."
)


# ============================================================
# KNOWLEDGE BASE
# ============================================================

if st.button("📚 Create Knowledgebase"):
    with st.spinner("Creating knowledge base..."):
        create_vector_db()

    st.success(
        "Knowledge base created successfully! ✅"
    )


st.divider()


# ============================================================
# SESSION STATUS
# ============================================================

context_info = session_manager.get_context(session_id)

messages = context_info.get("messages", [])

if messages:
    st.caption(
        f"💬 Conversation context: {len(messages)}/10 messages"
    )


# ============================================================
# TURN DETAIL
# ============================================================

# The answer is stored on the session, so it survives the re-run at
# the end of every turn. The sentiment, the escalation and the ticket
# are not, and a re-run would wipe them before they could be read.
# They are kept here against the id of the assistant message they
# belong to, so the history loop can draw them back.

if "turn_details" not in st.session_state:
    st.session_state.turn_details = {}


def render_turn_detail(result) -> None:
    """
    Draw everything one turn produced, apart from the answer itself.
    """

    if result.information_request:
        st.warning(result.information_request)

    if result.multiple_requests:
        st.caption(
            "📋 Multiple requests detected — addressing them "
            "from the conversation context."
        )

    if result.correction:
        st.caption("✏️ Customer correction detected.")

    # --------------------------------------------------------
    # SENTIMENT
    # --------------------------------------------------------

    sentiment = result.sentiment

    if sentiment:

        st.caption(
            f"🧭 Sentiment: **{sentiment.label}** "
            f"({sentiment.confidence:.2f}) · "
            f"language: {result.language}"
        )

        if sentiment.risk_topics:
            st.caption(
                "⚠️ High-risk topic: "
                + ", ".join(sentiment.risk_topics)
            )

    # --------------------------------------------------------
    # ESCALATION
    # --------------------------------------------------------

    if result.escalated:

        with st.expander("🚨 Escalation", expanded=True):

            for record in result.escalation.records:

                st.markdown(
                    f"**{record.condition_id} — "
                    f"{record.condition}**"
                )
                st.write(f"Reason: {record.reason}")
                st.write(
                    f"Queue: `{record.queue}` · "
                    f"severity: {record.severity}"
                )

                if record.scheduled_for:
                    st.write(
                        "Scheduled for: "
                        f"{record.scheduled_for.isoformat()}"
                    )

                st.caption(record.masked_summary)

    # --------------------------------------------------------
    # TICKET
    # --------------------------------------------------------

    if result.ticket:

        ticket = result.ticket

        with st.expander("🎫 Ticket", expanded=True):

            st.markdown(
                f"**{ticket.ticket_id}** · "
                f"{ticket.category} · "
                f"**{ticket.priority}** "
                f"(score {ticket.priority_score:.2f}) · "
                f"`{ticket.status}`"
            )

            if ticket.duplicate_of:
                st.warning(
                    f"Duplicate of {ticket.duplicate_of} — it "
                    f"shares that ticket's SLA rather than "
                    f"starting a second clock."
                )

            st.write(
                f"SLA: {ticket.sla_minutes:.0f} business minutes · "
                f"due {ticket.due_at.isoformat() if ticket.due_at else '—'}"
            )

            if ticket.breached:
                st.error("SLA BREACHED")
            elif ticket.warning_due:
                st.warning("SLA warning — past the configured fraction")

            if ticket.assigned_agent:
                st.write(
                    f"Assigned: {ticket.assigned_agent} "
                    f"({ticket.team})"
                )
            elif ticket.deferred_until:
                st.write(
                    f"Deferred to {ticket.deferred_until.isoformat()} "
                    f"({ticket.team})"
                )
            else:
                st.write(f"Queue: {ticket.team}")

            if ticket.routing_reason:
                st.caption(ticket.routing_reason)

            with st.expander("Masked handoff"):
                st.code(ticket.masked_handoff or "—")


# ============================================================
# DISPLAY PREVIOUS CONVERSATION
# ============================================================

for message in messages:

    if message.role == "user":
        with st.chat_message("user"):
            st.write(message.content)

    elif message.role == "assistant":
        with st.chat_message("assistant"):
            st.write(message.content)

            detail = st.session_state.turn_details.get(
                message.message_id
            )

            if detail is not None:
                render_turn_detail(detail)

# Forget turns that have scrolled out of the context window, so the
# store cannot grow for the life of the browser tab.
visible = {
    message.message_id for message in messages
}

for stale in list(st.session_state.turn_details):

    if stale not in visible:
        del st.session_state.turn_details[stale]


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input("Ask your question...")


if question:

    with st.chat_message("user"):
        st.write(question)

    # --------------------------------------------------------
    # RUN THE TURN
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Generating answer..."):

            result = flow.process_turn(
                session_id,
                question,
                now=now,
                answer_fn=lambda enriched, tone: get_qa_chain(
                    tone_instruction=tone
                ).invoke({"query": enriched})["result"],
            )

        # ----------------------------------------------------
        # ANSWER
        # ----------------------------------------------------

        st.write(result.answer)

        render_turn_detail(result)

        # Keep it for the re-run below, which redraws this turn from
        # the session rather than from `result`.
        if result.message_id:
            st.session_state.turn_details[result.message_id] = result

    # --------------------------------------------------------
    # REFRESH CHAT
    # --------------------------------------------------------

    st.rerun()
