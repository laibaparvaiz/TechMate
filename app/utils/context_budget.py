"""
File: TechMate/app/utils/context_budget.py

Description:
Small shared helpers for keeping prompts within a safe context
window budget PROACTIVELY, rather than only reacting after an
API rejects an oversized request.

Why this matters here specifically: some of OpenRouter's free
":free" models have noticeably smaller context windows than
flagship paid models. Retrieved RAG context, conversation
history, and raw file reads can all grow large enough to
exceed that — especially as your uploaded documents and
conversations accumulate over time.

These helpers are character-based, not true token-based (that
would need a tokenizer per model to be exact). ~4 characters
per token is a reasonable English-text approximation, used
here as a conservative safety margin rather than a precise
measurement — the goal is "stay comfortably under the limit",
not "use every last token."
"""

# Conservative default: roughly 3,000 tokens' worth of
# characters, safe even for smaller free-tier context windows.
DEFAULT_MAX_CHARS = 12000


def truncate_to_budget(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    label: str = "text",
) -> str:
    """
    Truncates text to at most max_chars, keeping the START of
    the text (usually the most relevant part for retrieved
    context and instructions) and appending a clear marker so
    truncation is visible rather than a silent quality
    regression.

    Logs a warning when truncation actually occurs, so it's
    visible in your console during testing.
    """

    if not text or len(text) <= max_chars:

        return text

    print(
        f"⚠️ Context budget: truncating '{label}' from "
        f"{len(text)} to {max_chars} characters to stay "
        "within a safe context window."
    )

    marker = (
        "\n\n[...truncated to fit context window...]"
    )

    return (
        text[: max_chars - len(marker)] + marker
    )


def truncate_each(
    items,
    max_chars_each: int,
    label: str = "item",
):
    """
    Applies truncate_to_budget() to each string in a list, for
    cases where you want to cap PER-ITEM size (e.g. each
    retrieved document, each conversation entry) rather than
    only the combined total — this prevents one unusually long
    item from crowding out everything else.
    """

    return [
        truncate_to_budget(
            item,
            max_chars_each,
            label=label,
        )
        for item in items
    ]