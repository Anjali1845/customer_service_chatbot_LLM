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

from langchain_helper import create_vector_db

from secure_answer import build_answer_fn

from document_intake import process_upload, run_retention_cleanup

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


# Uploaded documents are kept for a bounded time and then deleted.
# RetentionManager refuses to schedule its own sweep, so the app runs
# it here, once per browser session -- which is what makes retention
# actually happen rather than being a method nothing ever calls.
if "retention_sweep" not in st.session_state:
    st.session_state.retention_sweep = run_retention_cleanup()


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

st.sidebar.header("🔐 Knowledge-base access")

st.sidebar.caption(
    "Answers are retrieved through the authorised RAG layer, which "
    "filters documents by access level and effective date. Change "
    "the viewer to watch a restricted policy appear or vanish."
)

viewer_access_level = st.sidebar.selectbox(
    "Answer as",
    options=("public", "internal", "restricted"),
    index=0,
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

st.sidebar.header("📎 Document intake")

st.sidebar.caption(
    "Attach a PDF or image a customer sent. It is validated, "
    "security-scanned, read by OCR, compared against the message and "
    "masked before anything is shown."
)

uploaded_file = st.sidebar.file_uploader(
    "Attach a document",
    type=("pdf", "png", "jpg", "jpeg", "webp"),
)

upload_message = st.sidebar.text_input(
    "What did the customer say about it?",
    key="upload_message",
)

if "intake_result" not in st.session_state:
    st.session_state.intake_result = None

if uploaded_file is not None:

    # Re-process only when a different file arrives; Streamlit
    # re-runs the script on every interaction, and the pipeline does
    # real work (OCR, scanning), so it must not run on each rerun.
    signature = (uploaded_file.name, uploaded_file.size)

    if st.session_state.get("intake_signature") != signature:

        with st.sidebar.spinner("Processing the upload..."):

            st.session_state.intake_result = process_upload(
                uploaded_file.name,
                uploaded_file.getvalue(),
                message=upload_message,
            )

        st.session_state.intake_signature = signature

        # Sweep again after every upload, so a long-lived tab keeps
        # deleting expired documents rather than only doing it once.
        st.session_state.retention_sweep = run_retention_cleanup()

sweep = st.session_state.get("retention_sweep") or {}

st.sidebar.caption(
    f"🗑 Retention sweep: {sweep.get('deleted_count', 0)} expired "
    f"deleted · {sweep.get('skipped_count', 0)} still within "
    f"retention"
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


def _note_text(event) -> str:
    """
    A notification event as one readable line.

    The pipeline emits dicts carrying ``type``/``message``/``reason``,
    but the renderer must not depend on that exactly, so anything
    unexpected is stringified rather than dropped.
    """

    if isinstance(event, dict):

        for key in ("message", "reason", "type"):

            value = event.get(key)

            if value:

                return str(value)

        return str(event)

    return str(event)


def render_intake(result) -> None:
    """
    Draw the outcome of one document upload.

    Everything shown here is the masked copy: the raw extracted text
    stays in the result for evidence, but the panel is a handoff
    surface, so it renders ``masked_text`` and ``masked``.
    """

    if not result:
        return

    with st.expander("📎 Document intake", expanded=True):

        for event in result.get("notifications") or []:
            st.warning(_note_text(event))

        if not result.get("ok"):

            st.error(
                result.get("reason") or "The upload was rejected."
            )

            return

        elapsed = result.get("elapsed_seconds")

        st.success(
            "The document was accepted and processed"
            + (f" in {elapsed}s." if elapsed is not None else ".")
        )

        if result.get("clarification_required"):
            st.info(
                "The document was hard to read — a follow-up question "
                "is needed before it can be relied on."
            )

        structured = result.get("structured_data")

        if structured:

            st.markdown("**Extracted fields**")

            st.json(structured)

        comparison = result.get("comparison")

        if comparison:

            # Surface conflicts and clarifications as first-class
            # messages, not only inside the JSON. The top-level
            # clarification flag tracks OCR legibility, so a mismatch
            # between the message and the document has to be drawn
            # from the comparison itself.
            for conflict in comparison.get("conflicts") or []:
                st.error(f"Conflict: {_note_text(conflict)}")

            clarifications = comparison.get("clarifications") or []

            if clarifications:
                st.info(
                    "The message and the document disagree or one is "
                    "missing a value — ask about: "
                    + ", ".join(str(c) for c in clarifications)
                )

            st.markdown("**Message vs document**")

            st.json(comparison)

        masked_text = result.get("masked_text")

        if masked_text:

            st.markdown("**Document text (masked)**")

            st.code(masked_text)

        elif result.get("text"):

            st.caption(
                "No readable text was extracted from the document."
            )

        validation = result.get("validation")

        if validation:

            st.caption(f"Validation: {validation.get('reason') or 'passed'}")

        security = result.get("security")

        if security:

            st.caption(
                "Security scan: "
                + str(security.get("reason") or "no threats found")
            )


# Draw the last upload, if there was one. The result is kept on the
# session because the pipeline does real work and must not be re-run
# by the rerun at the end of every turn.
render_intake(st.session_state.get("intake_result"))


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
                answer_fn=build_answer_fn(
                    user_access_level=viewer_access_level
                ),
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
