"""
File: TechMate/app/safety/permissions.py
Description: Path traversal and dangerous command filters.
"""

from pathlib import Path
import re
from app.config import BASE_DIR

BLOCKED_KEYWORDS = ["rm -rf", "drop database", "sudo", "shutdown", "../.."]


def is_command_safe(target_path_or_command: str) -> bool:
    """
    Checks if a file path or command string is safe to execute.

    Returns True only when:

      1. It contains none of the blocked dangerous keywords
         (defense-in-depth — easy to bypass on its own, e.g.
         via backslashes on Windows or absolute paths that
         need no ".." at all, so it is NOT the primary
         protection).

      2. If it resolves to a filesystem path, that path is
         contained within BASE_DIR after resolution.

    Path resolution notes:

      - Relative paths are resolved AGAINST BASE_DIR
        explicitly, not the process's current working
        directory. Path.resolve() on a relative path uses
        os.getcwd() by default, which is only guaranteed to
        equal BASE_DIR if the app happens to be launched from
        that exact directory — not a safe assumption.

      - .resolve() also dereferences symlinks. A symlink
        planted inside the project pointing outside it will
        resolve to its real (outside) target and correctly
        FAIL the containment check below. No separate
        symlink-specific check is needed as long as this
        containment check runs on the fully resolved path.
    """

    if not target_path_or_command:
        return True

    text_lower = target_path_or_command.lower()

    # --------------------------------------------------------
    # Blocked keywords
    # --------------------------------------------------------

    for keyword in BLOCKED_KEYWORDS:

        if keyword in text_lower:
            return False

    # --------------------------------------------------------
    # Path containment check
    # --------------------------------------------------------

    try:

        candidate = Path(
            target_path_or_command
        )

        if candidate.is_absolute():

            resolved_path = candidate.resolve()

        else:

            resolved_path = (
                BASE_DIR / candidate
            ).resolve()

        base_resolved = BASE_DIR.resolve()

        if not resolved_path.is_relative_to(
            base_resolved
        ):

            # Path escapes the project root — REJECT.
            return False

    except Exception:

        # If we can't safely resolve the path, fail closed
        # rather than silently allowing it through.
        return False

    return True


def contains_blocked_keywords(text: str) -> bool:
    """
    Checks arbitrary free text (e.g. a raw chat message) for
    blocked dangerous keywords ONLY — no filesystem path
    resolution is attempted.

    Use this for natural-language input like chat messages.
    Do NOT use is_command_safe() for that purpose: it calls
    Path(...).resolve() on its input, which is correct for
    actual file paths/commands but can raise on ordinary text
    containing characters that happen to be reserved in
    filenames on some platforms (e.g. "?" or "*" on Windows).
    Since is_command_safe() fails closed on exceptions, that
    misuse can cause completely ordinary chat messages to be
    rejected as "unsafe" for reasons that have nothing to do
    with actual safety.
    """

    if not text:
        return False

    text_lower = text.lower()

    for keyword in BLOCKED_KEYWORDS:

        if keyword in text_lower:
            return True

    return False


def resolve_within_base(file_path: str):
    """
    Resolve file_path, treating relative paths as relative to
    BASE_DIR rather than the process's current working
    directory.

    Shared by any tool that accepts a file path argument
    (code_tools.py, pdf_tools.py, etc.) so path resolution
    behavior is consistent and defined in exactly one place.
    Path.resolve() on a relative path defaults to using
    os.getcwd(), which is only guaranteed to equal BASE_DIR if
    the app happens to be launched from that exact directory —
    not a safe assumption.
    """

    candidate = Path(
        file_path
    )

    if candidate.is_absolute():

        return candidate.resolve()

    return (
        BASE_DIR / candidate
    ).resolve()


_ILLEGAL_FILENAME_CHARS = re.compile(
    r'[<>:"/\\|?*\x00-\x1f]'
)


def sanitize_filename_component(
    text: str,
    max_len: int = 100,
) -> str:
    """
    Produce a filesystem-safe filename component from arbitrary
    text (a topic name, title, or output filename).

    Shared by any tool that builds a filename from LLM/user-
    provided text (study_tools.py, pdf_tools.py, etc.). Strips
    characters illegal in Windows filenames (including path
    separators "/" and "\\", which also prevents this being
    used as a directory-traversal vector — the result can never
    contain a path separator to escape whatever directory it's
    later joined onto) and control characters, not just spaces.
    """

    cleaned = text.strip().replace(" ", "_").lower()
    cleaned = _ILLEGAL_FILENAME_CHARS.sub("", cleaned)
    cleaned = cleaned.strip("._")

    if not cleaned:
        cleaned = "untitled"

    return cleaned[:max_len]