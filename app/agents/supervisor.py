"""
File: TechMate/app/agents/supervisor.py

Description:
LangGraph supervisor/router with:

- Persistent conversation memory (cost-efficient summarization)
- Deterministic routing first, LLM routing only when ambiguous
- RAG study pipeline (CRAG + web fallback + Self-RAG)
- HITL confirmation (yes/no) for write actions, incl. PDF notes
- Streaming for general chat, Mentor and Coding agents

Changes in this version (cleanup + fixes):
- Removed duplicated functions / blocks and fixed the mojibake text.
- confirmation_node: fixed the misplaced `or`, broader yes/no
  recognition, and "yes but ..." no longer silently cancels.
- Routing: "notes on / about / for" and "pdf notes" now count as a
  study request, so "Generate notes on X as a PDF" reaches the Study
  Agent deterministically.
- PDF title/filename: skips generic headings like "Summary".
- Flattened Markdown tables are repaired before the PDF is queued.
- The "all points come directly from..." footer is stripped.
- Study answers that used web sources get a clear note.
"""

import json
import re
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict

from pydantic import BaseModel, Field

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
)

from langchain_core.runnables import RunnableConfig

from langgraph.config import get_stream_writer

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from langgraph.graph.message import add_messages

from app.agents.coding_agent import get_coding_agent
from app.agents.mentor_agent import get_mentor_agent
from app.safety.hitl import pop_pending_action
from app.safety.privacy import redact_sensitive_data
from app.tools.code_tools import execute_edit_project_file
from app.tools.pdf_tools import (
    execute_generate_pdf_document,
    execute_merge_pdfs,
    execute_split_pdf,
    execute_add_watermark,
    execute_add_text_annotation,
    execute_fill_pdf_form,
)
from app.tools.study_tools import (
    execute_save_flashcards,
    execute_save_study_notes,
)
from app.memory.skills import (
    get_user_weaknesses,
    log_user_weakness,
)
from app.agents.study_agent import run_verified_study_agent

from app.agents.summarizer import (
    manage_conversation_memory,
    is_compression_trigger,
    find_memory_summary,
)

from app.utils.llm_router import (
    get_resilient_llm,
    get_fallback_llm,
    invoke_with_fallback,
    is_rate_limit_error,
)

from app.utils.context_budget import truncate_to_budget


# Shown at the top of a study answer that used web sources. Detected by
# exact text (never by emoji) when titles are derived and notes revised.
WEB_SOURCES_NOTE = (
    "Note: This wasn't fully covered in your uploaded materials, "
    "so I also used web sources."
)


# ============================================================
# 1. STATE
# ============================================================

class TechMateState(TypedDict, total=False):
    """Shared state for the TechMate supervisor graph."""

    messages: Annotated[Sequence[BaseMessage], add_messages]

    user_id: str

    retry_count: int

    current_intent: str

    final_response: str

    is_grounded: bool

    # Persistent compact memory (not stored as an AIMessage).
    conversation_summary: str

    # HITL: a proposed write action awaiting yes/no on the NEXT turn.
    pending_action: Optional[Dict[str, Any]]

    # Set fresh on every confirmation_node run: True only when THIS
    # turn's message resolved a pending_action.
    confirmation_handled: bool

    # Set when a pending action was dropped because the user moved on,
    # so a late "yes" gets a clear answer instead of going to chat.
    expired_action_hint: Optional[str]


# ============================================================
# 2. SCHEMAS
# ============================================================

class RouteDecision(BaseModel):
    """Structured fallback routing decision."""

    intent: str = Field(
        description="Must be exactly one of: 'study', 'mentor', 'coding', 'general'"
    )


class FlashcardItem(BaseModel):
    """One question/answer flashcard."""

    question: str = Field(description="The flashcard's question/prompt side.")

    answer: str = Field(description="The flashcard's answer side.")


class FlashcardList(BaseModel):
    """A structured set of flashcards."""

    cards: List[FlashcardItem] = Field(description="The generated flashcards.")


# ============================================================
# 3. CONTENT NORMALIZATION
# ============================================================

def _content_to_text(content) -> str:
    """Convert LangChain/LLM content of a COMPLETE message into clean text."""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, (list, tuple)):

        text_parts = []

        for block in content:

            if isinstance(block, dict):

                text = block.get("text", "")

                if text:
                    text_parts.append(str(text))

            elif isinstance(block, str):

                text_parts.append(block)

        return "\n".join(
            part.strip()
            for part in text_parts
            if part and part.strip()
        )

    if isinstance(content, dict):

        if "text" in content:
            return str(content["text"]).strip()

        return ""

    if content is None:
        return ""

    return str(content).strip()


def _stream_chunk_to_text(content) -> str:
    """
    Convert ONE streamed chunk into text WITHOUT stripping whitespace
    (stripping destroys word-boundary spaces in token streams).
    """

    if isinstance(content, str):
        return content

    if isinstance(content, (list, tuple)):

        text_parts = []

        for block in content:

            if isinstance(block, dict):

                text = block.get("text", "")

                if text:
                    text_parts.append(str(text))

            elif isinstance(block, str):

                text_parts.append(block)

        return "".join(text_parts)

    if isinstance(content, dict):

        if "text" in content:
            return str(content["text"])

        return ""

    if content is None:
        return ""

    return str(content)


# ============================================================
# 3B. STREAMING HELPERS
# ============================================================

def _get_stream_writer():
    """Safely obtain the LangGraph custom stream writer (None if unavailable)."""

    try:
        return get_stream_writer()
    except Exception:
        return None


def _emit_stream_event(writer, event_type: str, **data):
    """Send a controlled custom event to the UI."""

    if writer is None:
        print(f"Supervisor streaming writer unavailable: {event_type}")
        return

    writer({"type": event_type, **data})


def _stream_llm_response(llm, fallback_llm, messages, writer) -> str:
    """Stream a plain chat-model response token-by-token; return final text."""

    from app.utils.llm_router import stream_with_fallback

    accumulated = ""

    for chunk in stream_with_fallback(llm, fallback_llm, messages):

        piece = _stream_chunk_to_text(chunk.content)

        if not piece:
            continue

        accumulated += piece

        _emit_stream_event(writer, "token", text=piece)

    return accumulated.strip()


def _is_tool_call_chunk(message_chunk) -> bool:
    """True if a streamed chunk is an in-progress tool call (never shown to the user)."""

    if getattr(message_chunk, "tool_call_chunks", None):
        return True

    if getattr(message_chunk, "tool_calls", None):
        return True

    return False


def _stream_react_agent_response(agent, state_messages, writer) -> str:
    """
    Stream a create_react_agent()-style sub-agent's final answer.
    Tool-call chunks and tool results are NOT forwarded to the UI.
    """

    accumulated = ""

    for message_chunk, _metadata in agent.stream(
        {"messages": state_messages},
        stream_mode="messages",
    ):

        if not isinstance(message_chunk, (AIMessage, AIMessageChunk)):
            continue

        if _is_tool_call_chunk(message_chunk):
            continue

        piece = _stream_chunk_to_text(message_chunk.content)

        if not piece:
            continue

        accumulated += piece

        _emit_stream_event(writer, "token", text=piece)

    return accumulated.strip()


# ============================================================
# 4. USER-FRIENDLY ERRORS
# ============================================================

def _friendly_error_message(error: Exception) -> str:
    """Convert raw LLM/API exceptions into a short user-friendly message."""

    error_text = str(error).lower()

    if (
        "resource_exhausted" in error_text
        or "rate limit" in error_text
        or "quota exceeded" in error_text
        or "429" in error_text
    ):

        return (
            "⚠️ **Rate limit reached.** "
            "The AI service has temporarily reached its request limit. "
            "Please wait a little and try again."
        )

    if (
        "api key" in error_text
        or "authentication" in error_text
        or "unauthenticated" in error_text
        or "401" in error_text
    ):

        return (
            "⚠️ **API authentication error.** "
            "Please check the API key configuration."
        )

    if (
        "permission" in error_text
        or "forbidden" in error_text
        or "403" in error_text
    ):

        return (
            "⚠️ **API permission error.** "
            "The configured AI service does not have permission "
            "to perform this request."
        )

    if "timeout" in error_text or "timed out" in error_text:

        return "⚠️ **Request timed out.** Please try again."

    return (
        "⚠️ **Something went wrong while processing your request.** "
        "Please try again."
    )


# ============================================================
# 5. LATEST USER MESSAGE
# ============================================================

def _get_latest_user_message(messages: Sequence[BaseMessage]) -> str:
    """Return the latest real HumanMessage (internal triggers ignored)."""

    for message in reversed(list(messages)):

        if not isinstance(message, HumanMessage):
            continue

        content = _content_to_text(message.content)

        if not content:
            continue

        if "[SYSTEM_TRIGGER" in content:
            continue

        return content

    return ""


# ============================================================
# 6. NORMALIZE INTENT
# ============================================================

def _normalize_intent(intent: str) -> str:
    """Ensure routing always produces a supported intent."""

    if not intent:
        return "general"

    intent = intent.strip().lower()

    if intent in {"study", "mentor", "coding", "general"}:
        return intent

    return "general"


# ============================================================
# 7. DETERMINISTIC ROUTER
# ============================================================

# Action verb followed by a specific filename, e.g. "edit main.py".
_FILENAME_ACTION_RE = re.compile(
    r"\b(edit|modify|update|fix|delete|remove|rename|create|"
    r"write|refactor)\s+\S+\.\w{1,6}\b",
    re.IGNORECASE,
)

# Signals that the user wants a study ANSWER (explanation / notes).
# "notes on/about/for/of" and "pdf notes" were added so that
# "Generate notes on REST vs GraphQL as a PDF" routes to Study.
_STUDY_ANSWER_RE = re.compile(
    r"\b(explain|describe|summarize|summarise|summary|define|"
    r"discuss|teach|tell me about|what is|what are|how does|"
    r"how do|why does|why do|overview|answer|"
    r"notes\s+(?:on|about|for|of)|pdf\s+notes)\b",
)


# True: questions that START with "explain / what is / define / how does ..."
# go straight to the Study Agent (retrieval -> CRAG -> web fallback), which
# saves the one routing LLM call. Questions that look like code/debugging
# are left to the normal routing. Set to False to turn this off.
ROUTE_EXPLAIN_TO_STUDY = True

_EXPLAIN_START_RE = re.compile(
    r"^(?:please\s+|can you\s+|briefly\s+|quickly\s+)*"
    r"(?:explain|describe|define|what is|what are|how does|how do|"
    r"why does|why do|tell me about)\b"
)

_CODE_HINT_RE = re.compile(
    r"`|\bcode\b|\bbug\b|\berror\b|\bexception\b|\btraceback\b|"
    r"\bmy (?:project|app|script|function|file|program)\b|"
    r"\.py\b|\.js\b|[=(){};]"
)


def _deterministic_route(query: str) -> str | None:
    """
    Route obvious requests without an LLM call.

    Returns 'study' | 'mentor' | 'coding' | 'general', or None when
    the request is too ambiguous.
    """

    if not query:
        return "general"

    text = query.lower().strip()

    # ---------------- GENERAL / CASUAL ----------------

    general_phrases = {
        "hi", "hello", "hey", "hey there", "hi there",
        "good morning", "good afternoon", "good evening",
        "thanks", "thank you", "bye", "goodbye", "how are you",
    }

    if text in general_phrases:
        return "general"

    # "make pdf of this" / "save that as a pdf": save the PREVIOUS answer
    # (handled in study_node without any LLM call).
    if _is_save_previous_request(text):
        return "study"

    # ---------------- CODING: STRONG ACTION SIGNALS ----------------

    coding_action_terms = [
        "create a file", "create file", "make a file",
        "edit the file", "edit this file",
        "modify the file", "modify this file",
        "change the code", "update the code", "write the code",
        "implement this", "implement it",
        "build this", "build it",
        "create the project", "create a project",
        "run the tests", "run tests", "run pytest",
        "fix the file", "fix this file",
        "delete the file", "rename the file",
        "refactor the project", "modify my project", "change my project",
    ]

    if any(term in text for term in coding_action_terms):
        return "coding"

    # Explain/discuss a topic AND save it as a PDF -> Study pipeline
    # (so the PDF confirmation flow is used).
    if _looks_like_pdf_save_request(text) and _STUDY_ANSWER_RE.search(text):
        return "study"

    # ---------------- CODING: ACTION VERB + FILENAME ----------------

    if _FILENAME_ACTION_RE.search(text):
        return "coding"

    # ---------------- CODING: PDF-SPECIFIC ACTIONS ----------------

    pdf_action_terms = [
        "watermark",
        "merge pdf", "merge these pdf", "merge the pdf",
        "split pdf", "split the pdf",
        "extract pages", "extract page",
        "fill the form", "fill this form", "fill out the form",
        "pdf form",
        "annotate the pdf", "add a note to the pdf", "add annotation",
    ]

    if any(term in text for term in pdf_action_terms):
        return "coding"

    pdf_verbs = ["watermark", "merge", "split", "extract", "annotate", "fill"]

    if ".pdf" in text and any(verb in text for verb in pdf_verbs):
        return "coding"

    # ---------------- MENTOR: READ-ONLY PROGRAMMING HELP ----------------

    mentor_terms = [
        "code review", "review my code", "review this code",
        "explain this code", "explain my code",
        "debug this", "debug my code",
        "why does this code", "why is this code",
        "what is wrong with this code",
        "programming error", "python error", "javascript error",
        "syntax error", "runtime error", "exception", "stack trace",
        "programming question", "programming concept", "coding concept",
        "explain python", "explain javascript", "explain react",
        "how does this code work",
    ]

    if any(term in text for term in mentor_terms):
        return "mentor"

    # ---------------- STUDY: ACADEMIC / DOCUMENT SIGNALS ----------------

    study_terms = [
        "according to my uploaded material",
        "according to the uploaded material",
        "according to my notes", "according to the notes",
        "from my notes", "from the uploaded",
        "uploaded document", "uploaded documents", "uploaded material",
        "study material", "study notes", "lecture notes",
        "my notes", "my document", "my documents",
        "from the document", "from the documents",
        "in the document", "in the documents",
        "according to the pdf", "from the pdf", "in the pdf",
        "quiz me", "make flashcards", "create flashcards",
        "make a quiz", "create a quiz",
        "summarize my notes", "summarize the document", "summarize the notes",
        "explain this topic", "explain this concept",
        "what is an mvp", "define", "definition of",
    ]

    if any(term in text for term in study_terms):
        return "study"

    academic_terms = [
        "exam", "assignment", "homework", "lecture", "chapter",
        "course", "subject", "flashcard", "quiz", "revision",
        "revise", "study", "academic",
    ]

    if any(term in text for term in academic_terms):
        return "study"

    # ---------------- "ACCORDING TO THE ... PDF" -> STUDY ----------------

    if re.search(
        r"\b(?:according to|based on|from|in)\s+(?:the|my|this)\s+"
        r"(?:\w+\s+){0,3}(?:pdf|document|notes|material)s?\b",
        text,
    ):
        return "study"

    # ---------------- EXPLAIN-STYLE QUESTIONS -> STUDY ----------------

    if (
        ROUTE_EXPLAIN_TO_STUDY
        and _EXPLAIN_START_RE.match(text)
        and not _CODE_HINT_RE.search(text)
    ):
        return "study"

    # ---------------- AMBIGUOUS ----------------

    return None


# ============================================================
# 8. LLM FALLBACK ROUTER
# ============================================================

def _llm_route(state: TechMateState, query: str) -> str:
    """One LLM call, only when deterministic routing is unsure."""

    llm = get_resilient_llm(temperature=0.0)

    fallback_llm = get_fallback_llm(temperature=0.0)

    structured_llm = llm.with_structured_output(RouteDecision)

    structured_fallback = (
        fallback_llm.with_structured_output(RouteDecision)
        if fallback_llm is not None
        else None
    )

    system_prompt = """
You are the TechMate Supervisor.

Choose exactly one routing intent:

'study'
    Questions about uploaded documents, notes, quizzes,
    flashcards, academic material, learning, or studying.
    Also requests to explain a topic or make notes/PDF notes
    on a topic.

'mentor'
    Programming explanations, debugging guidance,
    code reviews, technical learning, and read-only
    programming assistance.

'coding'
    Requests to create, edit, modify, delete, run,
    test, or otherwise change the user's project/workspace.

'general'
    Casual conversation or requests that do not belong
    to the three specialist categories.

Important:

- Choose exactly one intent.
- Do not explain your decision.
- Prefer 'study' when the request is clearly academic or asks for notes.
- Prefer 'mentor' when the user wants programming help
  without asking you to modify files.
- Prefer 'coding' when the user asks you to actually
  modify or operate on the project.
"""

    conversation = list(state.get("messages", []))

    routing_messages = [
        {"role": "system", "content": system_prompt}
    ] + conversation

    try:

        decision = invoke_with_fallback(
            structured_llm,
            structured_fallback,
            routing_messages,
        )

        if decision is None:
            return "general"

        return _normalize_intent(decision.intent)

    except Exception as e:

        print("Supervisor fallback routing error:", str(e))

        raise


# ============================================================
# 9. MEMORY NODE
# ============================================================

def memory_node(state: TechMateState):
    """Cost-efficient conversation memory management."""

    return manage_conversation_memory(state)


def route_after_memory(state: TechMateState):
    """
    END after a manual "[SYSTEM_TRIGGER: COMPRESS_MEMORY]" turn (so the
    previous question is not re-answered); otherwise continue.
    """

    messages = list(state.get("messages", []))

    if not messages:
        return "router"

    if is_compression_trigger(messages[-1]):

        print(
            "Supervisor: manual compression trigger detected, "
            "skipping router (no re-answer will be generated)."
        )

        return END

    return "router"


# ============================================================
# 9C. HITL CONFIRMATION NODE
# ============================================================
#
# When a write tool (edit file / flashcards / notes / PDF) runs, it
# only REGISTERS a proposal and asks the user to confirm. This node
# checks whether the CURRENT message is a yes/no reply, executes or
# discards the proposal, and skips the router when it does.
# ============================================================

def _normalize_reply(text: str) -> str:
    """Lowercase and reduce to words only: "Don't!" -> "don t"."""

    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


_YES_EXACT_RE = re.compile(
    r"(?:y+e+s+|ys|y+e+a+h*|yep|yup|ok|okay|sure|confirm(?:ed)?|"
    r"approve(?:d)?|do it|go ahead|save it|please do|y)"
    r"(?: please)?"
)

_YES_PHRASES = {
    "yes save it", "yes confirm", "yes please", "save it",
}

_YES_START_RE = re.compile(r"^(?:y+e+s+|y+e+a+h*|yep|yup|ok|okay|sure)\b")

_NO_EXACT_RE = re.compile(
    r"(?:no+|n|nope|nah|cancel(?:l?ed)?|reject|stop|don t|dont|do not|"
    r"nevermind|never mind|no thanks|no thank you|not now|skip)"
)

_NO_START_RE = re.compile(
    r"^(?:no+|nope|nah|cancel|stop|dont|don t|do not)\b"
)


def _classify_reply(text: str) -> str:
    """
    Classify a reply to a pending confirmation:

        'yes'    - clear confirmation
        'no'     - clear rejection (also "no, but ...")
        'modify' - starts with yes but adds more ("yes but shorter")
        'other'  - unrelated message
    """

    normalized = _normalize_reply(text)

    if not normalized:
        return "other"

    if _YES_EXACT_RE.fullmatch(normalized) or normalized in _YES_PHRASES:
        return "yes"

    if _NO_EXACT_RE.fullmatch(normalized):
        return "no"

    if _NO_START_RE.match(normalized):
        return "no"

    if _YES_START_RE.match(normalized):
        return "modify"

    return "other"


def _looks_like_confirmation(text: str) -> bool:
    """True only for a clear "yes" (kept for older callers and tests)."""

    return _classify_reply(text) == "yes"


def _looks_like_rejection(text: str) -> bool:
    """True for a clear "no" (kept for older callers and tests)."""

    return _classify_reply(text) == "no"


def _looks_like_pending_action_confirmation(
    text: str,
    pending: Dict[str, Any],
) -> bool:
    """Accept clear, action-specific save wording as confirmation."""

    normalized = _normalize_reply(text)

    if re.match(r"^(?:please )?(?:do not|dont|don t|cancel|stop)\b", normalized):
        return False

    if pending.get("type") == "save_study_notes":

        is_save_command = bool(
            re.match(r"^(?:please )?(?:save|store)\b", normalized)
        )

        refers_to_notes = bool(
            re.search(r"\b(?:notes?|them|it|this|that)\b", normalized)
        )

        return is_save_command and refers_to_notes

    return False


def _execute_pending_action(pending: Dict[str, Any]) -> str:
    """
    Dispatch a confirmed pending action to its real write function.
    The ONLY place these write functions are called.
    """

    action_type = pending.get("type")

    if action_type == "save_flashcards":

        return execute_save_flashcards(
            pending.get("topic_name", "untitled"),
            pending.get("flashcards_json", "[]"),
        )

    if action_type == "save_study_notes":

        return execute_save_study_notes(
            pending.get("title", "untitled"),
            pending.get("markdown_content", ""),
        )

    if action_type == "edit_file":

        return execute_edit_project_file(
            pending.get("file_path", ""),
            pending.get("content", ""),
        )

    if action_type == "merge_pdfs":

        return execute_merge_pdfs(
            pending.get("file_paths", []),
            pending.get("output_filename", "merged.pdf"),
        )

    if action_type == "split_pdf":

        return execute_split_pdf(
            pending.get("file_path", ""),
            pending.get("page_ranges", ""),
            pending.get("output_filename", "split.pdf"),
        )

    if action_type == "add_watermark":

        return execute_add_watermark(
            pending.get("file_path", ""),
            pending.get("watermark_text", ""),
            pending.get("output_filename", "watermarked.pdf"),
        )

    if action_type == "add_text_annotation":

        return execute_add_text_annotation(
            pending.get("file_path", ""),
            pending.get("page_number", 1),
            pending.get("text", ""),
            pending.get("output_filename", "annotated.pdf"),
        )

    if action_type == "fill_pdf_form":

        return execute_fill_pdf_form(
            pending.get("file_path", ""),
            pending.get("field_values_json", "{}"),
            pending.get("output_filename", "filled.pdf"),
        )

    if action_type == "generate_pdf_document":

        return execute_generate_pdf_document(
            pending.get("title", "document"),
            _pdf_friendly_citations(
                pending.get("markdown_content", "")
            ),
            pending.get("output_filename", "document.pdf"),
        )

    return "I couldn't recognize the pending action to confirm."


def _handled(
    text: str,
    clear_pending: bool = True,
    new_pending: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a confirmation_node result that ends the turn with `text`."""

    update: Dict[str, Any] = {
        "confirmation_handled": True,
        "final_response": text,
        "messages": [AIMessage(content=text)],
        "is_grounded": True,
    }

    if new_pending is not None:
        update["pending_action"] = new_pending
    elif clear_pending:
        update["pending_action"] = None

    return update


_REVISABLE_TYPES = ("generate_pdf_document", "save_study_notes")


def _revise_pending_content(
    pending: Dict[str, Any],
    instruction: str,
) -> Optional[str]:
    """
    Rewrite the pending notes per the user's instruction (e.g. "make it
    shorter") with ONE LLM call. Uses only facts already in the notes.
    Returns the revised Markdown, or None on failure.
    """

    content = pending.get("markdown_content", "")

    if not content or not instruction:
        return None

    try:

        llm = get_resilient_llm(temperature=0.2)

        fallback_llm = get_fallback_llm(temperature=0.2)

        prompt = (
            "Revise the notes below according to the user's "
            "instruction. Use ONLY information already in the notes - "
            "do not add new facts. Keep Markdown headings and keep the "
            "inline source URLs for the points you keep. No Markdown "
            "tables. No commentary, no offers, and do not mention PDFs "
            "or files. Return ONLY the revised notes.\n\n"
            f"User instruction: {instruction}\n\n"
            f"Notes:\n{content}"
        )

        response = invoke_with_fallback(
            llm,
            fallback_llm,
            [HumanMessage(content=prompt)],
        )

        if response is None:
            return None

        revised = _content_to_text(response.content)

        if not revised:
            return None

        # Keep the "used web sources" note line if the rewrite dropped it.
        if content.startswith(WEB_SOURCES_NOTE) and not revised.startswith(
            WEB_SOURCES_NOTE
        ):
            revised = WEB_SOURCES_NOTE + "\n\n" + revised

        return revised

    except Exception as e:

        print("Revise pending content error (non-fatal):", str(e))

        return None


_BARE_YES_NO_RE = re.compile(
    r"(?:y+e+s+|ys|y+e+a+h*|yep|yup|no+|nope|nah)"
)


def _last_ai_invites_reply(messages: Sequence[BaseMessage]) -> bool:
    """
    True if the last assistant message asks/offers something, so a bare
    "yes" is a legitimate reply to it (normal chat) and not a stray one.
    """

    for message in reversed(list(messages)):

        if not isinstance(message, AIMessage):
            continue

        text = _content_to_text(message.content)

        if not text or text.startswith("[TECHMATE_MEMORY_SUMMARY]"):
            continue

        tail = text[-300:].lower()

        return bool(
            re.search(
                r"\?\s*$|want me to|would you like|do you want|let me know|"
                r"if you'd like|if you would like|shall i|should i",
                tail,
            )
        )

    return False


def _extract_instruction(user_query: str) -> str:
    """'yes but make it shorter' -> 'make it shorter'."""

    return re.sub(
        r"(?i)^\s*(?:y+e+s+|y+e+a+h*|yep|yup|ok|okay|sure)\b[\s,.!]*"
        r"(?:but|and|though|just)?\s*",
        "",
        user_query,
    ).strip()


def confirmation_node(state: TechMateState):
    """Check for a pending write action and whether this message confirms/rejects it."""

    pending = state.get("pending_action")

    messages = list(state.get("messages", []))

    user_query = _get_latest_user_message(messages)

    hint = state.get("expired_action_hint")

    if not pending or not user_query:

        # A late "yes"/"no" for an offer that expired: answer clearly
        # (and without an LLM call) instead of letting chat improvise.
        if (
            not pending
            and hint
            and user_query
            and _classify_reply(user_query) in ("yes", "no")
        ):

            if _classify_reply(user_query) == "yes":

                text = (
                    f"That request ({hint}) expired when you asked "
                    "another question, so nothing was saved. Ask again "
                    "and add 'save it as a PDF' if you still want it."
                )

            else:

                text = "Okay, nothing was saved."

            return {**_handled(text), "expired_action_hint": None}

        # A bare "yes"/"no" with nothing pending and nothing offered:
        # answer directly instead of letting the router re-run the
        # previous question (which wastes several LLM calls).
        if (
            not pending
            and not hint
            and user_query
            and _BARE_YES_NO_RE.fullmatch(_normalize_reply(user_query))
            and not _last_ai_invites_reply(messages)
        ):

            return {
                **_handled(
                    "There's nothing waiting for your confirmation right "
                    "now. What would you like me to do?"
                ),
                "expired_action_hint": None,
            }

        return {
            "confirmation_handled": False,
            "expired_action_hint": None,
        }

    reply = _classify_reply(user_query)

    # FIX: the second check used to sit INSIDE the first call's
    # argument (`f(a or b)`), so it never ran.
    if reply == "yes" or _looks_like_pending_action_confirmation(
        user_query,
        pending,
    ):

        return _handled(_execute_pending_action(pending))

    if reply == "no":

        if pending.get("type") == "generate_pdf_document":
            return _handled("Okay, I won't save the PDF.")

        return _handled("Okay, I won't make that change.")

    if reply == "modify":

        # "yes but make it shorter": revise the pending notes, then
        # ask again. Falls back to keeping the pending action as is.
        if pending.get("type") in _REVISABLE_TYPES:

            revised = _revise_pending_content(
                pending,
                _extract_instruction(user_query),
            )

            if revised:

                revised = _repair_flattened_tables(
                    _strip_save_chatter(revised)
                )

                new_pending = {**pending, "markdown_content": revised}

                if pending.get("type") == "generate_pdf_document":

                    ask = (
                        "\n\nPDF: I can save this version as "
                        f"'{pending.get('output_filename', 'document.pdf')}'. "
                        "Reply 'yes' to confirm, or 'no' to skip."
                    )

                else:

                    ask = (
                        "\n\nNotes: Save this version? "
                        "Reply 'yes' to confirm, or 'no' to skip."
                    )

                return _handled(
                    revised + ask,
                    new_pending=new_pending,
                )

        return _handled(
            "I can't change the content at this step. Reply 'yes' to "
            "save it as it is, or 'no' to skip. To get a different "
            "version, ask your question again with what you'd like "
            "changed.",
            clear_pending=False,
        )

    # Unrelated message: the user moved on. Drop the stale pending
    # action and route this as a normal new turn. Remember what was
    # dropped so a late "yes" can be answered clearly next turn.
    if pending.get("type") == "generate_pdf_document":
        dropped = f"PDF '{pending.get('output_filename', 'document.pdf')}'"
    else:
        dropped = "save request"

    return {
        "pending_action": None,
        "confirmation_handled": False,
        "expired_action_hint": dropped,
    }


def route_after_confirmation(state: TechMateState):
    """END if a pending action was resolved this turn, else continue to router."""

    if state.get("confirmation_handled"):
        return END

    return "router"


# ============================================================
# 10. ROUTER NODE
# ============================================================

def router_node(state: TechMateState):
    """Route the current request (deterministic first, one LLM call if needed)."""

    messages = list(state.get("messages", []))

    user_query = _get_latest_user_message(messages)

    if not user_query:

        response = "I couldn't determine what you want me to help with."

        return {
            "current_intent": "general",
            "final_response": response,
            "messages": [AIMessage(content=response)],
            "is_grounded": True,
        }

    intent = _deterministic_route(user_query)

    if intent is not None:

        print(f"Supervisor route: {intent} (deterministic, 0 routing API calls)")

    else:

        print("Supervisor route: ambiguous request; using one LLM routing call")

        try:

            intent = _llm_route(state, user_query)

        except Exception as e:

            print("Supervisor routing error:", str(e))

            error_message = _friendly_error_message(e)

            return {
                "current_intent": "general",
                "final_response": error_message,
                "messages": [AIMessage(content=error_message)],
                "is_grounded": False,
            }

    intent = _normalize_intent(intent)

    # ---------------- GENERAL CHAT ----------------

    if intent == "general":

        llm = get_resilient_llm(temperature=0.3)

        fallback_llm = get_fallback_llm(temperature=0.3)

        writer = _get_stream_writer()

        try:

            final_response = _stream_llm_response(
                llm,
                fallback_llm,
                messages,
                writer,
            )

            if not final_response:
                final_response = "I've processed your request."

        except Exception as e:

            print("General response error:", str(e))

            final_response = _friendly_error_message(e)

        final_response = redact_sensitive_data(final_response)

        return {
            "current_intent": "general",
            "final_response": final_response,
            "messages": [AIMessage(content=final_response)],
            "is_grounded": True,
        }

    # ---------------- SPECIALIST REQUEST ----------------

    return {
        "current_intent": intent,
        "final_response": "",
        "is_grounded": True,
    }


# ============================================================
# 11B. STUDY CONVERSATION CONTEXT
# ============================================================

STUDY_CONTEXT_MAX_ENTRIES = 4

STUDY_CONTEXT_PER_ENTRY_CHARS = 800


def _build_study_conversation_context(messages: Sequence[BaseMessage]) -> str:
    """
    Compact context (memory summary + last few entries) so the Study
    Agent can resolve follow-ups like "what about the second point?".
    The CURRENT question is excluded (it's passed separately).
    """

    summary = find_memory_summary(messages)

    conversation_only = []

    for message in messages:

        if isinstance(message, HumanMessage):

            text = _content_to_text(message.content)

            if not text or "[SYSTEM_TRIGGER" in text:
                continue

            conversation_only.append(("User", text))

        elif isinstance(message, AIMessage):

            text = _content_to_text(message.content)

            if not text or text.startswith("[TECHMATE_MEMORY_SUMMARY]"):
                continue

            # Skip the supervisor's own messages (save results, file
            # paths, confirmation replies): they are not study content
            # and the model started commenting on them.
            if re.match(
                r"(?i)^(?:successfully |okay, i won't|i can't change the "
                r"content|⚠️)",
                text,
            ):
                continue

            # Drop the trailing save prompt from earlier answers.
            text = re.split(r"\n\n(?:PDF:|Notes:|Flashcards:)", text)[0].strip()

            if not text:
                continue

            conversation_only.append(("Assistant", text))

    if conversation_only and conversation_only[-1][0] == "User":
        conversation_only = conversation_only[:-1]

    recent = conversation_only[-STUDY_CONTEXT_MAX_ENTRIES:]

    parts = []

    if summary:
        parts.append(f"Long-term memory summary:\n{summary}")

    if recent:

        recent_text = "\n".join(
            f"{role}: "
            + truncate_to_budget(
                text,
                STUDY_CONTEXT_PER_ENTRY_CHARS,
                label=f"{role.lower()} conversation entry",
            )
            for role, text in recent
        )

        parts.append(f"Recent conversation:\n{recent_text}")

    return "\n\n".join(parts)


# ============================================================
# 12B. STUDY SAVE-REQUEST DETECTION (HITL)
# ============================================================

FLASHCARD_REQUEST_TERMS = ["flashcard", "flash card"]

NOTES_SAVE_REQUEST_TERMS = [
    "save these notes",
    "save this as notes",
    "save my notes",
    "save study notes",
    "make a quiz",
    "create a quiz",
    "quiz me",
    "study guide",
]


def _looks_like_flashcard_request(query: str) -> bool:

    text = query.lower()

    return any(term in text for term in FLASHCARD_REQUEST_TERMS)


def _looks_like_notes_save_request(query: str) -> bool:

    text = query.lower()

    return any(term in text for term in NOTES_SAVE_REQUEST_TERMS)


def _looks_like_pdf_save_request(query: str) -> bool:
    """True if the user wants the answer saved/exported as a PDF."""

    text = query.lower()

    # Respect explicit cancellation language.
    if re.search(
        r"\b(?:don't|dont|do not|never)\s+"
        r"(?:save|store|export|download|create|make|generate|convert)\b",
        text,
    ):
        return False

    any_file_action = re.search(
        r"\b(?:save|store|export|download|create|make|generate|convert|turn)\b",
        text,
    )

    if not any_file_action:
        return False

    # An explicit PDF format always means PDF creation.
    if re.search(r"\bpdf\b", text):
        return True

    # Keep existing notes/quiz/flashcard saves on their own path.
    if _looks_like_notes_save_request(text):
        return False

    if _looks_like_flashcard_request(text):
        return False

    # Without an explicit format, only a save-like verb implies PDF.
    if not re.search(r"\b(?:save|store|export|download)\b", text):
        return False

    refers_to_answer = re.search(
        r"\b(?:it|this|that|the answer|my answer|your answer|"
        r"the explanation|my explanation)\b",
        text,
    )

    return bool(refers_to_answer or _STUDY_ANSWER_RE.search(text))


# ============================================================
# 12C. ANSWER CLEANUP HELPERS
# ============================================================

def _strip_pdf_save_disclaimer(response: str) -> str:
    """Remove model boilerplate that falsely says PDF saving is unavailable."""

    match = re.search(
        r"(?im)^\s*\**note\s+on\s+saving\s+as\s+pdf\**\\?\s*$",
        response,
    )

    if match:
        return response[:match.start()].rstrip()

    return response.rstrip()


def _strip_model_save_offer(response: str) -> str:
    """Remove a save prompt generated by the Study Agent (supervisor adds its own)."""

    match = re.search(
        r"(?im)^[^\r\n]*(?:"
        r"want me to save this\??"
        r"|i can save this answer as a pdf named\b"
        r"|i can save this as a pdf\b"
        r"|reply\s+['\"]?yes['\"]?\s+to confirm\b"
        r")[^\r\n]*",
        response,
    )

    if match:
        return response[:match.start()].rstrip()

    return response.rstrip()


def _strip_source_claim_footer(response: str) -> str:
    """
    Remove lines like "All points above come directly from the provided
    Chapter 10.pdf. No outside information was added." - an unverifiable
    claim that should not end up in saved notes.
    """

    cleaned = re.sub(
        r"(?im)^[ \t]*\**all\s+(?:the\s+)?(?:points|information|content)\b"
        r"[^\r\n]*(?:directly|only|solely|outside information)[^\r\n]*$",
        "",
        response,
    )

    return cleaned.rstrip()


def _strip_save_instructions(query: str) -> str:
    """
    Remove the "...and make notes as a pdf" part of a request so the
    Study pipeline (retrieval, web search, generation) only sees the
    actual question.

        "explain fastapi and make notes as a pdf" -> "explain fastapi"
        "Generate notes on X as a PDF"            -> "Explain X"
    """

    q = query.strip()

    text = re.sub(
        r"(?i)\s*(?:[,;]|\band\b|\bthen\b)\s*(?:please\s+)?(?:also\s+)?"
        r"(?:make|create|generate|save|export|turn|convert|put)\b"
        r"[^.?!]*\b(?:pdf|notes)\b[^.?!]*[.?!]*$",
        "",
        q,
    ).strip()

    if text and text != q:
        return text

    match = re.match(
        r"(?i)^(?:please\s+)?(?:can you\s+)?(?:generate|make|create|write|"
        r"give me)\s+(?:me\s+)?(?:a\s+)?(?:pdf\s+)?notes?\s+"
        r"(?:on|about|of|for)\s+(.+?)"
        r"(?:\s+(?:as|in|into)\s+(?:a\s+)?pdf)?[.?!]*$",
        q,
    )

    if match:
        return f"Explain {match.group(1).strip()}"

    return q


def _strip_save_chatter(response: str) -> str:
    """Remove model lines about exporting/printing/PDF files that it should not write."""

    return re.sub(
        r"(?im)^[^\r\n]*(?:export/print to pdf|copy the above notes|"
        r"pdf creation is handled|previous response indicated a pdf|"
        r"notes you can save as a pdf)[^\r\n]*\n?",
        "",
        response,
    ).rstrip()


def _pdf_friendly_citations(markdown: str) -> str:
    """
    For the PDF only: replace long inline URLs (which wrap badly and
    split hyphenated addresses) with numbered references [1], [2] and
    add a "Sources" list at the end. Chat text keeps its inline URLs.
    """

    # Some models wrap citations in 【 】; treat them as normal brackets
    # so the closing bracket is not read as part of the URL.
    markdown = markdown.replace("【", "[").replace("】", "]")

    urls: List[str] = []

    def number(match):

        url = match.group(0).rstrip(".;:")

        trailing = match.group(0)[len(url):]

        if url not in urls:
            urls.append(url)

        return f"[{urls.index(url) + 1}]{trailing}"

    text = re.sub(r"https?://[^\s,)\]>]+", number, markdown)

    if not urls:
        return markdown

    refs = r"(\[\d+\](?:\s*,\s*\[\d+\])*)"

    # "([1], [2])" -> "[1], [2]"   and   "[[1]]" -> "[1]"
    text = re.sub(r"\(\s*" + refs + r"\s*\)", r"\1", text)
    text = re.sub(r"\[\s*" + refs + r"\s*\]", r"\1", text)

    sources = "\n".join(f"{i}. {u}" for i, u in enumerate(urls, start=1))

    return text.rstrip() + "\n\n## Sources\n\n" + sources


def _repair_flattened_tables(text: str) -> str:
    """
    Split Markdown tables the model collapsed onto ONE line, e.g.
    "| A | B | |---|---| | 1 | 2 |", into proper rows so the PDF
    renderer (and the chat) show a real table.
    """

    fixed = []

    for line in text.splitlines():

        if line.count("|") >= 6 and re.search(r"\|\s*:?-{3,}:?\s*\|", line):
            line = re.sub(r"\|\s+\|", "|\n|", line)

        fixed.append(line)

    return "\n".join(fixed)


# ---------------- Titles / filenames ----------------

_GENERIC_HEADINGS = {
    "summary", "overview", "introduction", "notes",
    "structured notes", "key points", "answer", "conclusion",
}


def _derive_short_title(query: str, max_len: int = 60) -> str:

    text = query.strip()

    if len(text) > max_len:
        text = text[:max_len].rstrip() + "..."

    return text or "study_session"


def _topic_from_query(query: str) -> str:
    """Strip request boilerplate: 'Generate notes on X as a PDF' -> 'X'."""

    text = query.strip()

    text = re.sub(
        r"(?i)^(?:please\s+)?(?:can you\s+)?(?:generate|make|create|write|"
        r"give me|save|export)\s+(?:me\s+)?(?:a\s+|the\s+)?(?:pdf\s+)?"
        r"(?:notes?|summary|explanation)?\s*(?:on|about|of|for)?\s*",
        "",
        text,
    )

    text = re.sub(r"(?i)\s*(?:as|in|into|to)\s+(?:a\s+)?pdf\b.*$", "", text)

    text = re.sub(r"(?i)[,;]?\s*(?:and|then)?\s*save\b.*$", "", text)

    text = re.sub(
        r"(?i)^(?:briefly\s+)?(?:explain|describe|summari[sz]e|define|"
        r"teach me|tell me about)\s+(?:what\s+(?:is|are)\s+)?",
        "",
        text.strip(),
    )

    return text.strip(" .,:;") or query.strip()


def _derive_pdf_title(response: str, query: str) -> str:
    """Prefer a meaningful heading in the answer; skip generic ones like 'Summary'."""

    for line in response.splitlines():

        candidate = line.strip().rstrip("\\").strip()

        if not candidate:
            continue

        # Skip the "this used web sources" note added by study_node.
        if candidate.startswith(WEB_SOURCES_NOTE):
            continue

        # Only the FIRST content line may be the title. Later headings
        # are section headings ("In programming...") - never a title.
        # A real heading is "# Title" or a line that is ONLY "**Title**".
        heading = re.match(r"^#{1,6}\s+(.+?)\s*$", candidate) or re.match(
            r"^\*\*([^*]+)\*\*\s*$", candidate
        )

        if not heading:
            break

        title = heading.group(1).strip()

        if (
            len(title) > 70
            or title.endswith((".", "?", "!"))
            or "http" in title
            or "【" in title
            or title.lower().strip(" :") in _GENERIC_HEADINGS
        ):
            break

        return _derive_short_title(title)

    return _derive_short_title(
        _topic_from_query(_strip_save_instructions(query))
    )


def _make_pdf_filename(title: str) -> str:

    name = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("._-") or "study_notes"

    if not name.lower().endswith(".pdf"):
        name += ".pdf"

    return name


# ---------------- "Save the previous answer" requests ----------------

_SAVE_FILLER_WORDS = (
    r"please|can|could|you|make|create|generate|save|store|export|download|"
    r"turn|convert|put|a|an|the|as|into|in|to|of|this|that|it|answer|"
    r"response|explanation|above|previous|last|your|my|pdf|file|document|"
    r"notes?|me|for|and|then"
)


def _is_save_previous_request(query: str) -> bool:
    """
    True for "make pdf of this", "save that as a pdf", "convert the above
    to a pdf": a PDF request that names no topic of its own, so it refers
    to the previous answer.
    """

    text = query.lower().strip(" .!?")

    if not _looks_like_pdf_save_request(text):
        return False

    leftover = re.sub(rf"\b(?:{_SAVE_FILLER_WORDS})\b", " ", text)

    return re.sub(r"[^a-z0-9]+", " ", leftover).strip() == ""


_SUPERVISOR_REPLY_RE = re.compile(
    r"(?i)^(?:successfully |okay, i won't|i can't change the content|"
    r"there's nothing waiting|that request \(|pdf: i can save|notes: |"
    r"flashcards: |⚠️)"
)


def _find_previous_answer(messages: Sequence[BaseMessage]):
    """
    Return (answer, question) for the most recent real answer, skipping
    the supervisor's own messages (save results, prompts, errors).
    Returns (None, None) when there is nothing to save.
    """

    items = list(messages)

    for index in range(len(items) - 1, -1, -1):

        message = items[index]

        if not isinstance(message, AIMessage):
            continue

        text = _content_to_text(message.content)

        if (
            not text
            or text.startswith("[TECHMATE_MEMORY_SUMMARY]")
            or _SUPERVISOR_REPLY_RE.match(text)
        ):
            continue

        # Drop the trailing save prompt of that answer.
        text = re.split(r"\n\n(?:PDF:|Notes:|Flashcards:)", text)[0].strip()

        if not text:
            continue

        question = ""

        for earlier in range(index - 1, -1, -1):

            candidate = items[earlier]

            if isinstance(candidate, HumanMessage):

                asked = _content_to_text(candidate.content)

                if asked and "[SYSTEM_TRIGGER" not in asked:
                    question = asked
                    break

        return text, question

    return None, None


# ---------------- Flashcards ----------------

def _propose_flashcards_from_answer(query: str, answer: str):
    """
    Convert a verified answer into flashcards via ONE extra LLM call.
    Returns (pending_action, confirm_note) or (None, None) on failure.
    """

    try:

        llm = get_resilient_llm(temperature=0.0)

        fallback_llm = get_fallback_llm(temperature=0.0)

        structured_llm = llm.with_structured_output(FlashcardList)

        structured_fallback = (
            fallback_llm.with_structured_output(FlashcardList)
            if fallback_llm is not None
            else None
        )

        prompt = (
            "Convert the following study answer into a set of "
            "flashcards (question/answer pairs) that capture "
            "its key facts. Do not invent information beyond "
            "what's in the answer. Return only the structured "
            "flashcards.\n\n"
            f"Study Answer:\n{answer}"
        )

        result = invoke_with_fallback(
            structured_llm,
            structured_fallback,
            [HumanMessage(content=prompt)],
        )

        if not result or not result.cards:
            return None, None

        cards_json = json.dumps(
            [
                {"question": card.question, "answer": card.answer}
                for card in result.cards
            ],
            indent=2,
        )

        pending = {
            "type": "save_flashcards",
            "topic_name": _derive_short_title(query),
            "flashcards_json": cards_json,
        }

        confirm_note = (
            f"\n\nFlashcards: I've drafted {len(result.cards)} flashcards on "
            "this topic. Reply 'yes' to save them, or 'no' to skip."
        )

        return pending, confirm_note

    except Exception as e:

        print("Flashcard structuring error (non-fatal):", str(e))

        return None, None


# ============================================================
# 12. STUDY AGENT NODE
# ============================================================

def study_node(state: TechMateState, config: RunnableConfig):
    """
    Run the verified Study pipeline (Retrieve -> CRAG -> web fallback ->
    Generate -> Self-RAG) and, if the user asked, queue a flashcard /
    notes / PDF save that the user must confirm with yes/no.
    """

    try:

        messages = list(state.get("messages", []))

        user_query = _get_latest_user_message(messages)

        if not user_query:

            error_response = "I couldn't determine the study question."

            return {
                "messages": [AIMessage(content=error_response)],
                "final_response": error_response,
                "is_grounded": False,
            }

        # "make pdf of this": offer to save the previous answer. No LLM
        # call, no new study question.
        if _is_save_previous_request(user_query):

            previous, previous_question = _find_previous_answer(messages)

            if not previous:

                reply = (
                    "I don't have an answer to save yet. Ask a question "
                    "first, then ask me to save it as a PDF."
                )

                return {
                    "messages": [AIMessage(content=reply)],
                    "final_response": reply,
                    "is_grounded": True,
                    "pending_action": None,
                }

            previous = _repair_flattened_tables(
                _strip_save_chatter(_strip_source_claim_footer(previous))
            )

            pdf_title = _derive_pdf_title(previous, previous_question)

            output_filename = _make_pdf_filename(pdf_title)

            reply = (
                "PDF: I can save my previous answer as a PDF named "
                f"'{output_filename}'. Reply 'yes' to confirm, or 'no' "
                "to skip."
            )

            return {
                "messages": [AIMessage(content=reply)],
                "final_response": reply,
                "is_grounded": True,
                "pending_action": {
                    "type": "generate_pdf_document",
                    "title": pdf_title,
                    "markdown_content": previous,
                    "output_filename": output_filename,
                },
            }

        conversation_context = _build_study_conversation_context(messages)

        # The pipeline only sees the question itself, not the
        # "...and make notes as a pdf" part (that is handled below).
        verified_result = run_verified_study_agent(
            _strip_save_instructions(user_query),
            conversation_context=conversation_context,
        )

        final_response = verified_result.get("response", "")

        if not final_response:

            final_response = (
                "I couldn't produce a reliable answer from the "
                "available study material."
            )

        is_grounded = verified_result.get("is_grounded", False)

        source_mode = verified_result.get("source_mode", "kb")

        final_response = redact_sensitive_data(final_response)

        # ----------------------------------------------------
        # Clean the answer BEFORE it is captured for saving.
        # ----------------------------------------------------

        if is_grounded and final_response:

            final_response = _strip_source_claim_footer(final_response)

            final_response = _strip_save_chatter(final_response)

            final_response = final_response.replace("【", "[").replace("】", "]")

            final_response = _repair_flattened_tables(final_response)

            if source_mode in ("web", "mixed"):

                final_response = WEB_SOURCES_NOTE + "\n\n" + final_response

        # ----------------------------------------------------
        # HITL: propose a save (never saves immediately).
        # ----------------------------------------------------

        pending_action = None

        if is_grounded and final_response:

            wants_pdf = _looks_like_pdf_save_request(user_query)

            if wants_pdf or _looks_like_notes_save_request(user_query):
                final_response = _strip_model_save_offer(final_response)

            if wants_pdf:
                final_response = _strip_pdf_save_disclaimer(final_response)

            if _looks_like_flashcard_request(user_query):

                pending_action, confirm_note = _propose_flashcards_from_answer(
                    user_query,
                    final_response,
                )

                if pending_action and confirm_note:
                    final_response = final_response + confirm_note
                else:
                    final_response = (
                        final_response
                        + "\n\n⚠️ I couldn't prepare the flashcards for "
                        "saving just now (the AI service may be at its "
                        "limit). Ask again in a bit if you want them saved."
                    )

            elif wants_pdf:

                pdf_title = _derive_pdf_title(final_response, user_query)

                output_filename = _make_pdf_filename(pdf_title)

                pending_action = {
                    "type": "generate_pdf_document",
                    "title": pdf_title,
                    "markdown_content": final_response,
                    "output_filename": output_filename,
                }

                final_response = (
                    final_response
                    + "\n\nPDF: I can save this answer as a PDF named "
                    + f"'{output_filename}'. Reply 'yes' to confirm, "
                    + "or 'no' to skip."
                )

            elif _looks_like_notes_save_request(user_query):

                pending_action = {
                    "type": "save_study_notes",
                    "title": _derive_short_title(user_query),
                    "markdown_content": final_response,
                }

                final_response = (
                    final_response
                    + "\n\nNotes: Want me to save this? "
                    "Reply 'yes' to confirm, or 'no' to skip."
                )

        return {
            "messages": [AIMessage(content=final_response)],
            "final_response": final_response,
            "is_grounded": is_grounded,
            "pending_action": pending_action,
        }

    except Exception as e:

        print("Study agent error:", str(e))

        final_response = _friendly_error_message(e)

        return {
            "messages": [AIMessage(content=final_response)],
            "final_response": final_response,
            "is_grounded": False,
        }


# ============================================================
# 13B. WEAKNESS-TRACKING HEURISTICS (Mentor)
# ============================================================

ERROR_INDICATOR_TERMS = [
    "debug this", "debug my code",
    "why does this code", "why is this code",
    "why does this throw", "why does this raise", "why does this fail",
    "what is wrong with this code", "what causes this error",
    "getting this error", "seeing this error",
    "programming error", "python error", "javascript error",
    "runtime error", "exception", "stack trace", "traceback",
    "syntax error", "syntaxerror",
    "index error", "indexerror",
    "key error", "keyerror",
    "type error", "typeerror",
    "value error", "valueerror",
    "attribute error", "attributeerror",
    "name error", "nameerror",
    "zero division", "zerodivisionerror",
    "null pointer", "nullpointerexception",
    "reference error", "referenceerror",
    "undefined is not a function",
    "module not found", "modulenotfounderror",
    "import error", "importerror",
]


def _looks_like_error_report(user_query: str) -> bool:

    text = user_query.lower()

    return any(term in text for term in ERROR_INDICATOR_TERMS)


_KNOWN_EXCEPTION_PATTERNS = [
    (re.compile(r"\bindex\s*error\b", re.I), "IndexError"),
    (re.compile(r"\bkey\s*error\b", re.I), "KeyError"),
    (re.compile(r"\btype\s*error\b", re.I), "TypeError"),
    (re.compile(r"\bvalue\s*error\b", re.I), "ValueError"),
    (re.compile(r"\battribute\s*error\b", re.I), "AttributeError"),
    (re.compile(r"\bname\s*error\b", re.I), "NameError"),
    (re.compile(r"\bsyntax\s*error\b", re.I), "SyntaxError"),
    (re.compile(r"\bzero\s*division\s*error\b", re.I), "ZeroDivisionError"),
    (re.compile(r"\bmodule\s*not\s*found\s*error\b", re.I), "ModuleNotFoundError"),
    (re.compile(r"\bimport\s*error\b", re.I), "ImportError"),
    (re.compile(r"\breference\s*error\b", re.I), "ReferenceError"),
    (re.compile(r"\bnull\s*pointer\s*exception\b", re.I), "NullPointerException"),
    (re.compile(r"\bruntime\s*error\b", re.I), "RuntimeError"),
]


def _extract_weakness_topic(user_query: str, max_len: int = 80) -> str:
    """Fallback topic extraction: known exception name, else truncated question."""

    for pattern, canonical_name in _KNOWN_EXCEPTION_PATTERNS:

        if pattern.search(user_query):
            return canonical_name

    text = user_query.strip()

    if len(text) > max_len:
        text = text[:max_len].rstrip() + "..."

    return text or "unspecified issue"


_WEAKNESS_TOPIC_TAG_RE = re.compile(r"\[WEAKNESS_TOPIC:\s*([^\]]+)\]\s*$")

_WEAKNESS_META_COMMENTARY_RE = re.compile(
    r"\n*\[[^\]]*weakness[^\]]*\]\s*$",
    re.IGNORECASE,
)


def _extract_weakness_topic_from_response(response_text: str, user_query: str):
    """
    Parse the mentor's [WEAKNESS_TOPIC: ...] tag. Returns
    (display_response_without_tag, topic_or_none).
    """

    match = _WEAKNESS_TOPIC_TAG_RE.search(response_text)

    if not match:

        cleaned = _WEAKNESS_META_COMMENTARY_RE.sub("", response_text).rstrip()

        return cleaned, None

    topic = match.group(1).strip()

    display_response = _WEAKNESS_TOPIC_TAG_RE.sub("", response_text).rstrip()

    return display_response, (topic or None)


# ============================================================
# 13. MENTOR AGENT NODE
# ============================================================

def mentor_node(state: TechMateState):
    """
    Run the read-only Mentor Agent (streamed). Injects known weak areas
    into the prompt and logs new ones afterwards (both non-fatal).
    """

    writer = _get_stream_writer()

    state_messages = list(state.get("messages", []))

    user_id = state.get("user_id") or "local_user"

    user_query = _get_latest_user_message(state_messages)

    extra_context = ""

    try:

        weaknesses = get_user_weaknesses(user_id, limit=5)

        if weaknesses:

            weakness_lines = "; ".join(
                f"{w['topic']} (seen {w['error_count']}x)"
                for w in weaknesses
            )

            extra_context = (
                "Known recurring weak areas for this user: "
                f"{weakness_lines}. If relevant to the current "
                "question, reinforce these concepts naturally "
                "- don't be repetitive or condescending about "
                "it, and don't bring it up if it isn't relevant."
            )

    except Exception as weakness_err:

        print("Skill lookup error (non-fatal):", str(weakness_err))

    try:

        agent = get_mentor_agent(extra_context=extra_context)

        final_response = _stream_react_agent_response(
            agent,
            state_messages,
            writer,
        )

        if not final_response:

            final_response = (
                "I've processed your request, "
                "but no final response was returned."
            )

    except Exception as e:

        # Mentor tools are read-only, so retrying on the fallback
        # provider cannot duplicate side effects.
        if is_rate_limit_error(e):

            print(
                "Mentor agent failed on primary provider, retrying on "
                f"fallback: {str(e)[:200]}"
            )

            try:

                fallback_llm = get_fallback_llm(temperature=0.4)

                if fallback_llm is None:
                    raise e

                fallback_agent = get_mentor_agent(
                    llm=fallback_llm,
                    extra_context=extra_context,
                )

                final_response = _stream_react_agent_response(
                    fallback_agent,
                    state_messages,
                    writer,
                )

                if not final_response:

                    final_response = (
                        "I've processed your request, "
                        "but no final response was returned."
                    )

            except Exception as fallback_error:

                print("Mentor agent fallback also failed:", str(fallback_error))

                final_response = _friendly_error_message(fallback_error)

        else:

            print("Mentor agent error:", str(e))

            final_response = _friendly_error_message(e)

    final_response = redact_sensitive_data(final_response)

    display_response, tagged_topic = _extract_weakness_topic_from_response(
        final_response,
        user_query or "",
    )

    if tagged_topic:

        final_response = display_response

        try:

            log_user_weakness(
                user_id=user_id,
                topic=tagged_topic,
                error_details=final_response[:500],
            )

        except Exception as log_err:

            print("Skill logging error (non-fatal):", str(log_err))

    elif user_query and _looks_like_error_report(user_query):

        try:

            log_user_weakness(
                user_id=user_id,
                topic=_extract_weakness_topic(user_query),
                error_details=final_response[:500],
            )

        except Exception as log_err:

            print("Skill logging error (non-fatal):", str(log_err))

    return {
        "messages": [AIMessage(content=final_response)],
        "final_response": final_response,
        "is_grounded": True,
    }


# ============================================================
# 14. CODING AGENT NODE
# ============================================================

def coding_node(state: TechMateState, config: RunnableConfig):
    """
    Run the autonomous Coding Agent (streamed). Write tools only
    register a pending action; it is persisted into state here so
    confirmation_node can act on it next turn.
    """

    writer = _get_stream_writer()

    state_messages = list(state.get("messages", []))

    thread_id = (
        config
        .get("configurable", {})
        .get("thread_id", "unknown_thread")
    )

    try:

        agent = get_coding_agent()

        final_response = _stream_react_agent_response(
            agent,
            state_messages,
            writer,
        )

        if not final_response:

            final_response = (
                "I've processed your request, "
                "but no final response was returned."
            )

    except Exception as e:

        # CAUTION: Coding tools have real side effects. A retry is
        # only safe if the failure happened BEFORE any tool ran.
        if is_rate_limit_error(e):

            print(
                "Coding agent failed on primary provider, retrying on "
                f"fallback: {str(e)[:200]}"
            )

            try:

                fallback_llm = get_fallback_llm(temperature=0.2)

                if fallback_llm is None:
                    raise e

                fallback_agent = get_coding_agent(llm=fallback_llm)

                final_response = _stream_react_agent_response(
                    fallback_agent,
                    state_messages,
                    writer,
                )

                if not final_response:

                    final_response = (
                        "I've processed your request, "
                        "but no final response was returned."
                    )

            except Exception as fallback_error:

                print("Coding agent fallback also failed:", str(fallback_error))

                final_response = _friendly_error_message(fallback_error)

        else:

            print("Coding agent error:", str(e))

            final_response = _friendly_error_message(e)

    final_response = redact_sensitive_data(final_response)

    return {
        "messages": [AIMessage(content=final_response)],
        "final_response": final_response,
        "is_grounded": True,
        "pending_action": pop_pending_action(thread_id),
    }


# ============================================================
# 15. ROUTING EDGE
# ============================================================

def route_to_agent(state: TechMateState):
    """Determine which graph node should execute."""

    intent = state.get("current_intent", "general")

    if intent == "study":
        return "study_agent"

    if intent == "mentor":
        return "mentor_agent"

    if intent == "coding":
        return "coding_agent"

    return END


# ============================================================
# 16. BUILD GRAPH
# ============================================================

def build_graph(checkpointer=None):
    """Build and compile the TechMate LangGraph."""

    builder = StateGraph(TechMateState)

    builder.add_node("memory_manager", memory_node)
    builder.add_node("confirmation_node", confirmation_node)
    builder.add_node("router", router_node)
    builder.add_node("study_agent", study_node)
    builder.add_node("mentor_agent", mentor_node)
    builder.add_node("coding_agent", coding_node)

    builder.add_edge(START, "memory_manager")

    # memory -> confirmation (END on manual compression trigger)
    builder.add_conditional_edges(
        "memory_manager",
        route_after_memory,
        {
            "router": "confirmation_node",
            END: END,
        },
    )

    # confirmation -> router (END when a pending action was resolved)
    builder.add_conditional_edges(
        "confirmation_node",
        route_after_confirmation,
        {
            "router": "router",
            END: END,
        },
    )

    builder.add_conditional_edges(
        "router",
        route_to_agent,
        {
            "study_agent": "study_agent",
            "mentor_agent": "mentor_agent",
            "coding_agent": "coding_agent",
            END: END,
        },
    )

    builder.add_edge("study_agent", END)
    builder.add_edge("mentor_agent", END)
    builder.add_edge("coding_agent", END)

    return builder.compile(checkpointer=checkpointer)