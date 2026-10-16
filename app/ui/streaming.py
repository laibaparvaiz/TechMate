"""
File: TechMate/app/ui/streaming.py

Description:
Streamlit UI for TechMate.

Responsibilities:
- Display persistent LangGraph conversation history
- Display assistant responses
- Maintain PostgreSQL-backed LangGraph threads (long-term memory, e.g. on Neon)
- Handle chat sessions
- Handle file uploads / RAG ingestion
- Provide manual memory compression
- Stream only controlled final-answer tokens
- Hide CRAG / Self-RAG / routing internal output

Change in this version:
- When the last answer token arrives, the typing cursor is removed and
  later status events ("Verifying...", "Rewriting...") are shown as a
  caption directly under the answer, so it is clear the answer text is
  finished and the pipeline is only verifying it.
"""

import hashlib
import uuid

import streamlit as st

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from app.config import (
    BASE_DIR,
    POSTGRES_URL,
)

from app.agents.supervisor import build_graph

from app.database.models import init_db
from app.memory.memory import get_checkpointer as get_postgres_checkpointer

from app.sessions import (
    init_chat_db,
    save_chat_session,
    get_all_chat_sessions,
)

from app.rag.ingestion import ingest_document


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TechMate",
    layout="wide",
)


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

init_chat_db()

# Ensure user_skills / session_logs tables exist. Safe on every start.
init_db()


# ============================================================
# SINGLE LOCAL USER
# ============================================================
#
# No multi-user auth yet. Skill/weakness tracking needs a stable
# user_id that is NOT tied to thread_id (which changes per New Chat).

LOCAL_USER_ID = "local_user"


# ============================================================
# LANGGRAPH CHECKPOINTER (POSTGRES)
# ============================================================

@st.cache_resource
def get_checkpointer():
    """
    One PostgreSQL-backed checkpointer per Streamlit process.
    Requires a reachable PostgreSQL database (DATABASE_URL in .env).
    """

    checkpointer, _conn = get_postgres_checkpointer(POSTGRES_URL)

    return checkpointer


checkpointer = get_checkpointer()

graph = build_graph(checkpointer=checkpointer)


# ============================================================
# SESSION STATE
# ============================================================

if "thread_id" not in st.session_state:

    st.session_state.thread_id = str(uuid.uuid4())


if "chat_title" not in st.session_state:

    st.session_state.chat_title = "New Chat"


if "processed_upload" not in st.session_state:

    st.session_state.processed_upload = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("Conversations")

    # ---------------- NEW CHAT ----------------

    if st.button("New Chat", use_container_width=True):

        st.session_state.thread_id = str(uuid.uuid4())

        st.session_state.chat_title = "New Chat"

        st.rerun()

    st.divider()

    # ---------------- RECENT CHATS ----------------

    st.subheader("Recent Chats")

    sessions = get_all_chat_sessions()

    for session in sessions:

        is_active = session["thread_id"] == st.session_state.thread_id

        button_type = "primary" if is_active else "secondary"

        if st.button(
            session["title"],
            key=session["thread_id"],
            type=button_type,
            use_container_width=True,
        ):

            st.session_state.thread_id = session["thread_id"]

            st.session_state.chat_title = session["title"]

            st.rerun()

    st.divider()

    # ---------------- MEMORY & CONTEXT ----------------

    st.subheader("Memory & Context")

    if st.button("Force Compress Memory", use_container_width=True):

        config = {
            "configurable": {
                "thread_id": st.session_state.thread_id
            }
        }

        try:

            graph.invoke(
                {
                    "messages": [
                        HumanMessage(
                            content="[SYSTEM_TRIGGER: COMPRESS_MEMORY]"
                        )
                    ]
                },
                config=config,
            )

            st.success("Chat history summarized and trimmed!")

            st.rerun()

        except Exception as e:

            st.error("Failed to compress memory.")

            print("Memory compression error:", str(e))

    if st.button("Clear Thread History", use_container_width=True):

        st.session_state.thread_id = str(uuid.uuid4())

        st.session_state.chat_title = "New Chat"

        st.session_state.processed_upload = None

        st.success("Started a fresh thread!")

        st.rerun()

    st.divider()

    # ---------------- UPLOAD MATERIALS ----------------

    st.subheader("Upload Materials")

    uploaded_file = st.file_uploader(
        "Upload PDF, TXT, Code, or Images",
        type=["pdf", "txt", "py", "js", "md", "png", "jpg", "jpeg"],
    )

    if uploaded_file is not None:

        file_bytes = uploaded_file.getvalue()

        file_hash = hashlib.md5(file_bytes).hexdigest()

        upload_id = f"{uploaded_file.name}:{file_hash}"

        # Process only if this is a new upload.
        if st.session_state.processed_upload != upload_id:

            file_ext = uploaded_file.name.split(".")[-1].lower()

            temp_dir = BASE_DIR / "data" / "uploads"

            temp_dir.mkdir(parents=True, exist_ok=True)

            file_path = temp_dir / uploaded_file.name

            with open(file_path, "wb") as f:

                f.write(file_bytes)

            # Documents
            if file_ext in ["pdf", "txt", "md"]:

                with st.spinner("Ingesting document into RAG..."):

                    success = ingest_document(str(file_path))

                if success:

                    st.success(f"Indexed `{uploaded_file.name}` successfully!")

                else:

                    st.error("Failed to index document.")

            # Code
            elif file_ext in ["py", "js"]:

                st.success(
                    f"Code file `{uploaded_file.name}` saved to workspace."
                )

            # Images
            elif file_ext in ["png", "jpg", "jpeg"]:

                st.success(
                    f"Image `{uploaded_file.name}` uploaded successfully."
                )

            st.session_state.processed_upload = upload_id


# ============================================================
# MAIN CHAT INTERFACE
# ============================================================

st.header("TechMate")


# ============================================================
# CURRENT THREAD CONFIGURATION
# ============================================================

config = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}


# ============================================================
# CONTENT NORMALIZATION
# ============================================================

def content_to_text(content) -> str:
    """Convert LangChain/Gemini content into plain text."""

    if isinstance(content, str):

        return content

    if isinstance(content, (list, tuple)):

        parts = []

        for block in content:

            if isinstance(block, dict):

                text = block.get("text", "")

                if text:

                    parts.append(str(text))

            elif isinstance(block, str):

                parts.append(block)

        return "".join(parts)

    if isinstance(content, dict):

        return str(content.get("text", ""))

    if content is None:

        return ""

    return str(content)


# ============================================================
# LOAD PERSISTED CHAT HISTORY
# ============================================================

state_snapshot = graph.get_state(config)

if state_snapshot.values:

    messages = state_snapshot.values.get("messages", [])

else:

    messages = []


# ============================================================
# RENDER PERSISTED CHAT HISTORY
# ============================================================

for msg in messages:

    # Ignore internal system trigger messages.
    if (
        isinstance(msg, HumanMessage)
        and "[SYSTEM_TRIGGER" in content_to_text(msg.content)
    ):

        continue

    # Legacy safety check: memory summaries saved as AIMessage.
    if (
        isinstance(msg, AIMessage)
        and content_to_text(msg.content).startswith(
            "[TECHMATE_MEMORY_SUMMARY]"
        )
    ):

        continue

    # Memory summary (SystemMessage): show as a small collapsed note,
    # so compressed history looks summarized, not lost.
    if isinstance(msg, SystemMessage):

        text = content_to_text(msg.content)

        if text.startswith("[TECHMATE_MEMORY_SUMMARY]"):

            summary_body = text[len("[TECHMATE_MEMORY_SUMMARY]"):].strip()

            with st.expander(
                "Earlier conversation summarized (click to view)",
                expanded=False,
            ):

                st.markdown(summary_body or "_(empty summary)_")

        continue

    # User message
    if isinstance(msg, HumanMessage):

        text = content_to_text(msg.content)

        if not text.strip():

            continue

        with st.chat_message("user"):

            st.markdown(text)

    # Assistant message
    elif isinstance(msg, AIMessage):

        text = content_to_text(msg.content)

        if not text.strip():

            continue

        with st.chat_message("assistant"):

            st.markdown(text)


# ============================================================
# USER INPUT
# ============================================================

user_query = st.chat_input(
    "Ask a coding question, request a study quiz, or review code..."
)


# ============================================================
# PROCESS USER MESSAGE
# ============================================================

if user_query:

    # Display the user message immediately.
    with st.chat_message("user"):

        st.markdown(user_query)

    # Save chat title on the first message.
    if st.session_state.chat_title == "New Chat":

        st.session_state.chat_title = (
            user_query[:30] + ("..." if len(user_query) > 30 else "")
        )

        save_chat_session(
            st.session_state.thread_id,
            st.session_state.chat_title,
        )

    inputs = {
        "messages": [HumanMessage(content=user_query)],
        "user_id": LOCAL_USER_ID,
    }

    # ========================================================
    # STREAM ASSISTANT RESPONSE
    # ========================================================

    try:

        with st.chat_message("assistant"):

            processing_status = st.status(
                "Thinking & processing...",
                expanded=False,
            )

            # Answer text, then a small caption under it that shows
            # the current step once the text has finished streaming
            # ("Verifying...", "Rewriting...").
            response_placeholder = st.empty()

            footer_placeholder = st.empty()

            accumulated_response = ""

            generation_started = False

            # Use ONLY custom stream events, so internal CRAG,
            # Self-RAG, routing and structured-output messages are
            # never displayed.
            for event in graph.stream(
                inputs,
                config=config,
                stream_mode="custom",
            ):

                if not isinstance(event, dict):

                    continue

                event_type = event.get("type")

                # ---------------- STATUS ----------------

                if event_type == "status":

                    message = event.get(
                        "message",
                        "Thinking & processing...",
                    )

                    processing_status.update(
                        label=message,
                        state="running",
                        expanded=False,
                    )

                    # Status after the text started means the text is
                    # finished and the pipeline is verifying it: drop
                    # the typing cursor and say what is happening.
                    if generation_started:

                        response_placeholder.markdown(
                            accumulated_response
                        )

                        footer_placeholder.caption(message)

                # ---------------- TOKEN ----------------

                elif event_type == "token":

                    text = event.get("text", "")

                    if not text:

                        continue

                    generation_started = True

                    accumulated_response += text

                    processing_status.update(
                        label="Generating answer...",
                        state="running",
                        expanded=False,
                    )

                    response_placeholder.markdown(
                        accumulated_response + " ▌"
                    )

                # ---------------- VERIFICATION FAILED ----------------

                elif event_type == "verification_failed":

                    message = event.get(
                        "message",
                        "Answer verification failed.",
                    )

                    processing_status.update(
                        label="⚠️ Verification failed",
                        state="error",
                        expanded=False,
                    )

                    if generation_started:

                        footer_placeholder.caption(f"⚠️ {message}")

                # ---------------- COMPLETE ----------------

                elif event_type == "complete":

                    message = event.get(
                        "message",
                        "Answer verified.",
                    )

                    processing_status.update(
                        label=message,
                        state="complete",
                        expanded=False,
                    )

                    footer_placeholder.empty()

            # Final persisted state is the source of truth.
            final_state = graph.get_state(config)

            final_messages = (
                final_state.values.get("messages", [])
                if final_state.values
                else []
            )

            # Find the latest real AI response.
            final_response = ""

            for msg in reversed(final_messages):

                if not isinstance(msg, AIMessage):

                    continue

                text = content_to_text(msg.content)

                if not text.strip():

                    continue

                # Ignore internal memory summaries.
                if text.startswith("[TECHMATE_MEMORY_SUMMARY]"):

                    continue

                final_response = text

                break

            # Replace the streamed draft with the verified final result
            # when they differ (e.g. rewritten answer, PDF prompt added).
            if final_response:

                if final_response != accumulated_response:

                    response_placeholder.markdown(final_response)

                elif not generation_started:

                    response_placeholder.markdown(final_response)

                else:

                    # Same text: just remove the typing cursor.
                    response_placeholder.markdown(accumulated_response)

            else:

                response_placeholder.warning(
                    "The agent completed processing "
                    "but returned no visible response."
                )

            footer_placeholder.empty()

            processing_status.update(
                label="Processing complete",
                state="complete",
                expanded=False,
            )

        # Rerun once after successful completion; persisted LangGraph
        # state becomes the source of truth on the next render.
        st.rerun()

    except Exception as e:

        error_text = str(e).lower()

        # ---------------- RATE LIMIT ----------------

        if (
            "resource_exhausted" in error_text
            or "rate limit" in error_text
            or "quota exceeded" in error_text
            or "429" in error_text
        ):

            st.warning(
                "⚠️ **Rate limit reached.** "
                "The AI service has temporarily reached "
                "its request limit. Please try again later."
            )

        # ---------------- AUTHENTICATION ----------------

        elif (
            "api key" in error_text
            or "authentication" in error_text
            or "unauthenticated" in error_text
            or "401" in error_text
        ):

            st.error(
                "⚠️ **API authentication error.** "
                "Please check your API key configuration."
            )

        # ---------------- PERMISSION ----------------

        elif (
            "permission" in error_text
            or "forbidden" in error_text
            or "403" in error_text
        ):

            st.error(
                "⚠️ **API permission error.** "
                "The configured AI service does not have "
                "permission to perform this request."
            )

        # ---------------- TIMEOUT ----------------

        elif "timeout" in error_text or "timed out" in error_text:

            st.warning(
                "⚠️ **Request timed out.** Please try again."
            )

        # ---------------- GENERIC ----------------

        else:

            st.error(
                "⚠️ **Something went wrong while processing your request.**"
            )

            print("TechMate processing error:", str(e))