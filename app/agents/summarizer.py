"""
File: TechMate/app/agents/summarizer.py

Description:
Cost-efficient conversation memory manager for TechMate.

Goals:
- Preserve persistent LangGraph conversation history.
- Avoid an LLM call on every request.
- Compress long conversations only when necessary.
- Keep recent messages intact.
- Store memory summaries inside LangGraph state.
- Support manual COMPRESS_MEMORY trigger.
- Never delete history if summarization fails.
- Work correctly with the add_messages reducer.

API behavior:

    Short conversation
        ↓
    0 Gemini summarization calls

    Long conversation
        ↓
    1 Gemini summarization call

    Manual compression
        ↓
    1 Gemini summarization call

Resilience:

    The summarization LLM call goes through
    invoke_with_fallback(), so if OpenRouter's free-tier
    quota is exhausted, it transparently retries on the
    configured fallback provider (see app/utils/llm_router.py).
"""

from typing import Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)

from app.utils.llm_router import (
    get_resilient_llm,
    get_fallback_llm,
    invoke_with_fallback,
)


# ============================================================
# CONFIGURATION
# ============================================================

# Automatic compression threshold.

MAX_MESSAGES = 20


# Number of newest conversation messages preserved.

KEEP_RECENT_MESSAGES = 8


# Maximum amount of conversation text sent to the
# summarization model.

MAX_SUMMARY_INPUT_CHARS = 24000


# ============================================================
# CONTENT NORMALIZATION
# ============================================================

def _content_to_text(content) -> str:
    """
    Convert LangChain structured content into plain text.
    """

    if isinstance(content, str):

        return content.strip()

    if isinstance(content, (list, tuple)):

        parts = []

        for block in content:

            if isinstance(block, dict):

                text = block.get(
                    "text",
                    "",
                )

                if text:

                    parts.append(
                        str(text)
                    )

            elif isinstance(block, str):

                parts.append(
                    block
                )

        return "\n".join(
            part.strip()
            for part in parts
            if part and part.strip()
        )

    if isinstance(content, dict):

        text = content.get(
            "text",
            "",
        )

        if text:

            return str(
                text
            ).strip()

        return ""

    if content is None:

        return ""

    return str(
        content
    ).strip()


# ============================================================
# SYSTEM TRIGGER DETECTION
# ============================================================

def _is_compression_trigger(
    message: BaseMessage,
) -> bool:
    """
    Detect the internal manual memory-compression trigger.
    """

    if not isinstance(
        message,
        HumanMessage,
    ):

        return False

    content = _content_to_text(
        message.content
    )

    return (
        "[SYSTEM_TRIGGER: COMPRESS_MEMORY]"
        in content
    )


def is_compression_trigger(
    message: BaseMessage,
) -> bool:
    """
    Public wrapper for _is_compression_trigger().

    Used by supervisor.py's graph routing (route_after_memory)
    to decide whether to skip the router after a manual
    "Force Compress Memory" trigger, preventing a duplicate
    re-answer of the previous real user question.
    """

    return _is_compression_trigger(
        message
    )


# ============================================================
# SUMMARY DETECTION
# ============================================================

def _is_memory_summary(
    message: BaseMessage,
) -> bool:
    """
    Detect a TechMate-generated memory summary.
    """

    content = _content_to_text(
        getattr(
            message,
            "content",
            "",
        )
    )

    return content.startswith(
        "[TECHMATE_MEMORY_SUMMARY]"
    )


# ============================================================
# GET REAL CONVERSATION
# ============================================================

def _get_conversation_messages(
    messages: Sequence[BaseMessage],
) -> list[BaseMessage]:
    """
    Return actual user/assistant conversation messages.

    Internal system triggers and memory summaries are excluded
    from the normal conversation count.
    """

    result = []

    for message in messages:

        # Ignore manual compression trigger.

        if _is_compression_trigger(
            message
        ):

            continue

        # Ignore existing memory summaries.

        if _is_memory_summary(
            message
        ):

            continue

        # Ignore other system instructions.

        if isinstance(
            message,
            SystemMessage,
        ):

            continue

        result.append(
            message
        )

    return result


# ============================================================
# FIND EXISTING SUMMARY
# ============================================================

def _find_existing_summary(
    messages: Sequence[BaseMessage],
) -> str:
    """
    Find the existing TechMate memory summary.

    Only the latest summary is needed.
    """

    for message in reversed(
        messages
    ):

        if not _is_memory_summary(
            message
        ):

            continue

        content = _content_to_text(
            getattr(
                message,
                "content",
                "",
            )
        )

        if content:

            return content

    return ""


def find_memory_summary(
    messages: Sequence[BaseMessage],
) -> str:
    """
    Public wrapper for _find_existing_summary().

    Used by supervisor.py to build conversation context for
    the Study Agent, so follow-up questions (e.g. "what about
    the second point?") can be answered with awareness of the
    persistent memory summary, not just the raw current query.
    """

    return _find_existing_summary(
        messages
    )


# ============================================================
# FORMAT SUMMARY INPUT
# ============================================================

def _format_messages_for_summary(
    messages: Sequence[BaseMessage],
) -> str:
    """
    Convert conversation messages into compact text.
    """

    parts = []

    for message in messages:

        content = _content_to_text(
            getattr(
                message,
                "content",
                "",
            )
        )

        if not content:

            continue

        if isinstance(
            message,
            HumanMessage,
        ):

            role = "User"

        elif isinstance(
            message,
            AIMessage,
        ):

            role = "Assistant"

        else:

            role = "Other"

        parts.append(
            f"{role}: {content}"
        )

    text = "\n\n".join(
        parts
    )

    # Keep only the most recent portion if the input is
    # unexpectedly large.

    if len(text) > MAX_SUMMARY_INPUT_CHARS:

        text = text[
            -MAX_SUMMARY_INPUT_CHARS:
        ]

    return text


# ============================================================
# BUILD SUMMARY PROMPT
# ============================================================

def _build_summary_prompt(
    existing_summary: str,
    older_messages: Sequence[BaseMessage],
) -> str:
    """
    Build the summarization prompt.
    """

    conversation_text = (
        _format_messages_for_summary(
            older_messages
        )
    )

    if existing_summary:

        previous_memory = existing_summary

    else:

        previous_memory = (
            "No previous memory summary exists."
        )

    return f"""
You are the TechMate persistent conversation-memory manager.

Create ONE concise factual memory summary.

The summary will be used in future conversation turns, so
preserve information that is genuinely useful for continuity.

Preserve:
- User goals
- Project names
- Important project context
- Important technical decisions
- Technologies and frameworks being used
- Relevant files
- Problems being solved
- Important study topics
- Important user-provided requirements
- Current task status
- Unresolved questions

Do NOT:
- Invent information
- Guess missing information
- Add outside knowledge
- Include greetings
- Include filler
- Repeat information unnecessarily
- Describe the summarization process

If an existing memory summary is provided, update it with
the new conversation information.

Existing memory:

{previous_memory}

New conversation information:

{conversation_text}

Return ONLY the updated factual memory summary.
""".strip()


# ============================================================
# SUMMARIZE
# ============================================================

def summarize_conversation(
    existing_summary: str,
    older_messages: Sequence[BaseMessage],
) -> str:
    """
    Perform exactly one LLM summarization call, with
    cross-provider fallback if OpenRouter's free-tier quota
    is exhausted.

    This function is called only when compression is required.
    """

    if not older_messages:

        return existing_summary


    prompt = _build_summary_prompt(
        existing_summary,
        older_messages,
    )


    llm = get_resilient_llm(
        temperature=0.0
    )

    fallback_llm = get_fallback_llm(
        temperature=0.0
    )


    response = invoke_with_fallback(
        llm,
        fallback_llm,
        [
            SystemMessage(
                content=prompt
            )
        ],
    )


    summary = _content_to_text(
        response.content
    )


    if not summary:

        return existing_summary


    return (
        "[TECHMATE_MEMORY_SUMMARY]\n"
        + summary
    )


# ============================================================
# COMPRESS CONVERSATION
# ============================================================

def compress_conversation(
    messages: Sequence[BaseMessage],
) -> dict:
    """
    Compress older messages while remaining compatible with
    LangGraph's add_messages reducer.

    IMPORTANT:

    We do NOT return a new complete message list.

    Instead we:

        1. Generate summary.
        2. Remove old messages using RemoveMessage.
        3. Remove the previous summary if one exists.
        4. Add the new summary.
        5. Keep the newest messages untouched.
    """

    conversation_messages = (
        _get_conversation_messages(
            messages
        )
    )


    # --------------------------------------------------------
    # Nothing useful to compress.
    # --------------------------------------------------------

    if len(conversation_messages) <= 1:

        return {}


    # --------------------------------------------------------
    # Preserve newest messages.
    # --------------------------------------------------------

    recent_messages = (
        conversation_messages[
            -KEEP_RECENT_MESSAGES:
        ]
    )


    older_messages = (
        conversation_messages[
            :-KEEP_RECENT_MESSAGES
        ]
    )


    if not older_messages:

        return {}


    # --------------------------------------------------------
    # Existing summary.
    # --------------------------------------------------------

    existing_summary = (
        _find_existing_summary(
            messages
        )
    )


    # --------------------------------------------------------
    # Generate updated summary.
    #
    # THIS IS THE ONLY LLM CALL DURING COMPRESSION
    # (with cross-provider fallback).
    # --------------------------------------------------------

    summary = summarize_conversation(
        existing_summary,
        older_messages,
    )


    # --------------------------------------------------------
    # If summarization failed, do not modify state.
    # --------------------------------------------------------

    if not summary:

        return {}


    # --------------------------------------------------------
    # Prepare removals.
    # --------------------------------------------------------

    updates = []


    # Remove all old conversation messages.

    for message in older_messages:

        message_id = getattr(
            message,
            "id",
            None,
        )

        if message_id:

            updates.append(
                RemoveMessage(
                    id=message_id
                )
            )


    # Remove the previous summary.

    for message in messages:

        if not _is_memory_summary(
            message
        ):

            continue

        message_id = getattr(
            message,
            "id",
            None,
        )

        if message_id:

            updates.append(
                RemoveMessage(
                    id=message_id
                )
            )


    # --------------------------------------------------------
    # Safety check.
    # --------------------------------------------------------

    if not updates:

        print(
            "Memory compression skipped: "
            "messages have no IDs."
        )

        return {}


    # --------------------------------------------------------
    # Add replacement summary.
    # --------------------------------------------------------

    updates.append(
        SystemMessage(
            content=summary
        )
    )


    return {
        "messages": updates
    }


# ============================================================
# MAIN MEMORY MANAGER
# ============================================================

def manage_conversation_memory(
    state,
) -> dict:
    """
    Cost-efficient memory manager.

    Normal short conversation:
        0 LLM calls.

    Normal long conversation:
        1 LLM call when threshold is exceeded.

    Manual compression:
        1 LLM call.

    Failed compression:
        0 state modifications.

    The function returns {} during normal operation so that
    add_messages does not duplicate existing messages.
    """

    messages = list(
        state.get(
            "messages",
            []
        )
    )


    if not messages:

        return {}


    # ========================================================
    # MANUAL COMPRESSION
    # ========================================================

    manual_trigger = any(
        _is_compression_trigger(
            message
        )
        for message in messages
    )


    # ========================================================
    # COUNT REAL CONVERSATION
    # ========================================================

    conversation_messages = (
        _get_conversation_messages(
            messages
        )
    )

    message_count = len(
        conversation_messages
    )


    # ========================================================
    # NORMAL OPERATION
    # ========================================================

    if (
        not manual_trigger
        and message_count <= MAX_MESSAGES
    ):

        return {}


    # ========================================================
    # COMPRESSION
    # ========================================================

    try:

        return compress_conversation(
            messages
        )

    except Exception as e:

        print(
            "Conversation compression error:",
            str(e)
        )

        # Never destroy conversation history if memory
        # compression fails.

        return {}