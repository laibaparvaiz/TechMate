"""
File: TechMate/app/tools/study_tools.py
Description: Export and formatting tools for the Study Agent (saving notes, flashcards, quizzes).

HITL (Human-In-The-Loop):
These tools do NOT write anything immediately. They validate
the request, register a pending action for confirmation (see
app/safety/hitl.py), and return a description asking the user
to confirm before anything is actually saved to disk. The
actual write happens later via execute_save_flashcards() /
execute_save_study_notes(), called directly by
app/agents/supervisor.py's confirmation_node once the user
approves — never by the LLM/agent directly.

Bug fixes from the original version:
- topic_name/title are natural-language text, not file paths.
  The original code ran them through is_command_safe(), which
  attempts Path(...).resolve() — correct for actual paths, but
  liable to reject perfectly normal topic names containing
  characters that happen to be reserved in filenames on some
  platforms (e.g. "?" on Windows). Now uses
  contains_blocked_keywords(), which checks the same dangerous-
  keyword blocklist without attempting path resolution.
- Filenames are now properly sanitized (illegal filesystem
  characters stripped), not just space-to-underscore replaced.
"""

import json
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from app.safety.permissions import (
    contains_blocked_keywords,
    sanitize_filename_component,
)
from app.safety.hitl import register_pending_action
from app.config import BASE_DIR

STUDY_OUTPUT_DIR = BASE_DIR / "data" / "study_materials"


# ============================================================
# FILENAME SANITIZATION
# ============================================================
#
# Moved to app/safety/permissions.py as
# sanitize_filename_component() — shared with
# app/tools/pdf_tools.py, which needs the exact same behavior
# for its output filenames. Kept as a thin local alias so the
# rest of this file doesn't need to change.

_sanitize_filename_component = sanitize_filename_component


# ============================================================
# ACTUAL WRITE LOGIC — called ONLY after user confirmation
# ============================================================

def execute_save_flashcards(
    topic_name: str,
    flashcards_json: str,
) -> str:
    """
    Performs the real write. Called by supervisor.py's
    confirmation_node after the user approves — never called
    directly by the LLM/agent.
    """

    try:

        STUDY_OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        cards_data = json.loads(
            flashcards_json
        )

        safe_name = _sanitize_filename_component(
            topic_name
        )

        file_path = (
            STUDY_OUTPUT_DIR
            / f"{safe_name}_flashcards.json"
        )

        file_path.write_text(
            json.dumps(cards_data, indent=4),
            encoding="utf-8",
        )

        count = (
            len(cards_data)
            if isinstance(cards_data, list)
            else "some"
        )

        return (
            f"Successfully saved {count} flashcards "
            f"to {file_path}"
        )

    except json.JSONDecodeError:

        return (
            "Error: Provided flashcards data is not "
            "valid JSON."
        )

    except Exception as e:

        return f"Error saving flashcards: {str(e)}"


def execute_save_study_notes(
    title: str,
    markdown_content: str,
) -> str:
    """
    Performs the real write. Called by supervisor.py's
    confirmation_node after the user approves — never called
    directly by the LLM/agent.
    """

    try:

        STUDY_OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        safe_name = _sanitize_filename_component(
            title
        )

        file_path = (
            STUDY_OUTPUT_DIR / f"{safe_name}.md"
        )

        file_path.write_text(
            markdown_content,
            encoding="utf-8",
        )

        return (
            f"Successfully saved study notes to {file_path}"
        )

    except Exception as e:

        return f"Error saving notes: {str(e)}"


# ============================================================
# AGENT-FACING TOOLS — propose only, never write directly
# ============================================================

@tool
def save_flashcards(
    topic_name: str,
    flashcards_json: str,
    config: RunnableConfig,
) -> str:
    """Proposes saving a JSON string of generated flashcards to the local study folder.
    Use this when the user asks to generate and save flashcards for later review.
    This does NOT save immediately — it registers the proposal and asks the user
    to confirm before anything is written.
    """

    if contains_blocked_keywords(topic_name):
        return "Error: Invalid topic name."

    try:

        cards_data = json.loads(
            flashcards_json
        )

    except json.JSONDecodeError:

        return (
            "Error: Provided flashcards data is not "
            "valid JSON."
        )

    count = (
        len(cards_data)
        if isinstance(cards_data, list)
        else "some"
    )

    try:

        thread_id = (
            config["configurable"]["thread_id"]
        )

    except Exception:

        # Defensive fallback if config injection behaves
        # differently than expected in your installed
        # langchain/langgraph version — worth testing this
        # path specifically.
        thread_id = "unknown_thread"

    register_pending_action(
        thread_id,
        {
            "type": "save_flashcards",
            "topic_name": topic_name,
            "flashcards_json": flashcards_json,
        },
    )

    return (
        f"I've prepared {count} flashcards on "
        f"'{topic_name}', ready to save. Reply 'yes' to "
        "confirm saving them, or 'no' to cancel."
    )


@tool
def save_study_notes(
    title: str,
    markdown_content: str,
    config: RunnableConfig,
) -> str:
    """Proposes saving generated study notes, summaries, or quizzes as a Markdown file.
    Use this to persist study guides requested by the user.
    This does NOT save immediately — it registers the proposal and asks the user
    to confirm before anything is written.
    """

    if contains_blocked_keywords(title):
        return "Error: Invalid title format."

    try:

        thread_id = (
            config["configurable"]["thread_id"]
        )

    except Exception:

        thread_id = "unknown_thread"

    register_pending_action(
        thread_id,
        {
            "type": "save_study_notes",
            "title": title,
            "markdown_content": markdown_content,
        },
    )

    return (
        f"I've prepared study notes titled '{title}', "
        "ready to save. Reply 'yes' to confirm saving "
        "them, or 'no' to cancel."
    )