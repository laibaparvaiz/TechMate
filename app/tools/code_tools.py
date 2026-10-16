"""
File: TechMate/app/tools/code_tools.py
Description: Local filesystem and testing tools for the Autonomous Coding Agent.
"""

import re
import subprocess
from pathlib import Path
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from app.safety.permissions import is_command_safe, resolve_within_base
from app.safety.hitl import register_pending_action
from app.config import BASE_DIR
from app.utils.context_budget import truncate_to_budget
from app.safety.privacy import redact_sensitive_data


# ============================================================
# PATH RESOLUTION HELPER
# ============================================================
#
# Moved to app/safety/permissions.py as resolve_within_base()
# — shared with app/tools/pdf_tools.py, which needs the exact
# same path-safety behavior. Kept as a thin local alias so the
# rest of this file (and anything importing
# _resolve_within_base from here) doesn't need to change.

_resolve_within_base = resolve_within_base


# ============================================================
# TEST TARGET VALIDATION
# ============================================================

# Only plain path-like targets are allowed: letters, numbers,
# underscore, hyphen, dot, and forward slash. This blocks
# argument-injection-style values (e.g. "--collect-only; rm -rf /")
# from ever reaching pytest's argv, on top of the path
# containment check in is_command_safe().

_SAFE_TEST_TARGET_RE = re.compile(
    r"^[A-Za-z0-9_\-./]+$"
)


@tool
def read_project_file(file_path: str, chunk: int = 1) -> str:
    """Reads and returns the content of a project file, in chunks of ~20,000 characters.
    Use this to inspect source code, configuration files, or logs.
    chunk is 1-based: use chunk=1 for the beginning. If the file is larger than
    one chunk, the result tells you the total chunk count — call this again
    with the next chunk number to continue reading, rather than re-reading
    chunk 1 again (which just returns the identical content).
    """
    try:
        path = _resolve_within_base(file_path)

        # Check safety guard
        if not is_command_safe(str(path)):
            return "Error: Permission denied. Access restricted by safety policy."

        if not path.exists():
            return f"Error: File '{file_path}' does not exist."

        # Redact secrets BEFORE the content ever becomes part of
        # the LLM's context — not just before the final displayed
        # message. This tool can read .env/config files, and
        # responses stream token-by-token live to the screen
        # before a complete final message even exists to redact;
        # scrubbing here means the model never sees the raw
        # secret at all, so it can't leak it either way.
        content = path.read_text(encoding="utf-8")
        content = redact_sensitive_data(content)

        # Chunked reading, replacing blind single-chunk
        # truncation. Previously, a file over 20,000 characters
        # always returned the SAME first 20,000 characters no
        # matter how many times the tool was called — observed
        # in practice causing the agent to re-request the
        # identical content multiple times ("reading it in
        # sections") without actually getting new information,
        # wasting tool/LLM calls. Now each call can request a
        # specific chunk, and the response tells the agent how
        # many chunks exist so it can request the next one.
        CHUNK_SIZE = 20000

        total_chars = len(content)
        total_chunks = max(
            1,
            -(-total_chars // CHUNK_SIZE),  # ceiling division
        )

        if chunk < 1 or chunk > total_chunks:
            return (
                f"Error: chunk {chunk} is out of range. "
                f"'{file_path}' has {total_chunks} chunk(s) "
                f"of ~{CHUNK_SIZE} characters each "
                f"({total_chars} characters total)."
            )

        start = (chunk - 1) * CHUNK_SIZE
        end = start + CHUNK_SIZE
        section = content[start:end]

        if total_chunks > 1:

            header = (
                f"[Chunk {chunk} of {total_chunks} — "
                f"'{file_path}' is {total_chars} characters "
                "total. If you need more of this file, call "
                f"read_project_file again with chunk={chunk + 1} "
                "rather than re-reading this same chunk.]\n\n"
            )

            return header + section

        return section

    except Exception as e:
        return f"Error reading file '{file_path}': {str(e)}"


@tool
def edit_project_file(file_path: str, content: str, config: RunnableConfig) -> str:
    """Proposes overwriting or creating a file at file_path with the specified content.
    Use this to apply code changes, write tests, or update configurations.
    This does NOT write immediately — it registers the proposal and asks the user
    to confirm before anything is written to disk.
    """
    try:
        path = _resolve_within_base(file_path)

        # Check safety guard
        if not is_command_safe(str(path)):
            return "Error: Permission denied. Write operation restricted by safety policy."

        try:
            thread_id = config["configurable"]["thread_id"]
        except Exception:
            # Defensive fallback if config injection behaves
            # differently than expected in your installed
            # langchain/langgraph version.
            thread_id = "unknown_thread"

        register_pending_action(
            thread_id,
            {
                "type": "edit_file",
                "file_path": file_path,
                "content": content,
            },
        )

        preview = (
            content
            if len(content) <= 300
            else content[:300] + "... (truncated)"
        )

        return (
            f"I'm ready to write to '{file_path}'. Preview:\n\n"
            f"{preview}\n\n"
            "Reply 'yes' to confirm this change, or 'no' to cancel."
        )
    except Exception as e:
        return f"Error preparing file write for '{file_path}': {str(e)}"


def execute_edit_project_file(file_path: str, content: str) -> str:
    """
    Performs the real write. Called by supervisor.py's
    confirmation_node after the user approves — never called
    directly by the LLM/agent.
    """
    try:
        path = _resolve_within_base(file_path)

        if not is_command_safe(str(path)):
            return "Error: Permission denied. Write operation restricted by safety policy."

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Successfully updated '{file_path}'."
    except Exception as e:
        return f"Error writing to file '{file_path}': {str(e)}"


@tool
def run_pytest_suite(test_target: str = "tests/") -> str:
    """Runs pytest on the specified directory or test file and returns output logs.
    Use this to check whether implementation changes passed or broke existing tests.
    """
    try:
        # Reject anything that isn't a plain path-like string
        # BEFORE it ever reaches subprocess argv.
        if not _SAFE_TEST_TARGET_RE.match(test_target):
            return "Error: Invalid test target format."

        # Check safety guard on execution directory
        if not is_command_safe(test_target):
            return "Error: Execution blocked by safety permissions."

        result = subprocess.run(
            ["pytest", test_target],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            return f"✅ Tests Passed:\n{result.stdout}"
        else:
            return f"❌ Tests Failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Test execution timed out (limit: 30 seconds)."
    except Exception as e:
        return f"Error running test suite: {str(e)}"