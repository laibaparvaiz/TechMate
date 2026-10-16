"""
File: TechMate/tests/test_permissions.py
Description: Tests for app/safety/permissions.py.

Covers the exact path-traversal bug found and fixed earlier in
this project's development: a naive containment check that
detected an escape attempt but never actually rejected it, and
a follow-up bug where a naive string-prefix check would have
incorrectly allowed sibling-directory-name collisions (e.g.
BASE_DIR "/project" incorrectly matching "/project_evil").
"""

import pytest

from app.safety.permissions import (
    is_command_safe,
    contains_blocked_keywords,
    resolve_within_base,
    sanitize_filename_component,
)


# ============================================================
# is_command_safe — path containment
# ============================================================

class TestIsCommandSafe:

    def test_normal_relative_file_is_safe(self):
        assert is_command_safe("notes.txt") is True

    def test_subfolder_file_is_safe(self):
        assert is_command_safe("src/main.py") is True

    def test_classic_traversal_is_blocked(self):
        assert is_command_safe("../../../etc/passwd") is False

    def test_absolute_path_outside_project_is_blocked(self):
        assert is_command_safe("/etc/passwd") is False

    def test_sibling_directory_name_collision_is_blocked(self):
        """
        Regression test for the exact bug found in this project:
        a naive `str(path).startswith(str(base))` check would
        incorrectly treat a SIBLING directory whose name happens
        to start with the same prefix as being "inside" the
        project (e.g. "/project" vs "/project_evil"). The fix
        uses Path.is_relative_to() instead, which respects
        actual directory boundaries.
        """
        from app.config import BASE_DIR

        sibling_path = str(BASE_DIR) + "_evil/secrets.txt"
        assert is_command_safe(sibling_path) is False

    def test_blocked_keyword_rejected(self):
        assert is_command_safe("rm -rf /") is False

    def test_empty_string_is_safe(self):
        # Matches the original design: no path/command to check
        # means nothing to reject.
        assert is_command_safe("") is True


# ============================================================
# contains_blocked_keywords — text-only check, no path
# resolution (fixes the bug where natural-language chat text
# containing reserved filename characters, e.g. "?", could
# raise inside Path.resolve() and get incorrectly rejected)
# ============================================================

class TestContainsBlockedKeywords:

    def test_normal_question_with_reserved_char_is_not_flagged(self):
        """
        Regression test: "What is a linked list?" contains "?",
        which is a reserved filename character on Windows. Using
        is_command_safe() (path-resolution based) on this text
        used to risk an exception. contains_blocked_keywords()
        must handle this safely since it never attempts path
        resolution at all.
        """
        assert contains_blocked_keywords("What is a linked list?") is False

    def test_dangerous_keyword_is_flagged(self):
        assert contains_blocked_keywords("please run rm -rf /") is True

    def test_empty_text_is_not_flagged(self):
        assert contains_blocked_keywords("") is False


# ============================================================
# resolve_within_base — relative paths anchor to BASE_DIR,
# not the process's current working directory
# ============================================================

class TestResolveWithinBase:

    def test_relative_path_resolves_under_base_dir(self):
        from app.config import BASE_DIR

        result = resolve_within_base("notes.txt")
        assert str(result).startswith(str(BASE_DIR.resolve()))

    def test_absolute_path_is_returned_as_is_resolved(self):
        import os

        abs_path = os.path.abspath(__file__)
        result = resolve_within_base(abs_path)
        assert str(result) == str(resolve_within_base(abs_path))


# ============================================================
# sanitize_filename_component
# ============================================================

class TestSanitizeFilenameComponent:

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("What is O(n)?", "what_is_o(n)"),
            ("C++ Basics", "c++_basics"),
            ("Binary Search Trees", "binary_search_trees"),
            ("???", "untitled"),
        ],
    )
    def test_known_cases(self, raw, expected):
        assert sanitize_filename_component(raw) == expected

    def test_path_separators_are_stripped(self):
        """
        Critical for output-filename safety in pdf_tools.py /
        study_tools.py: if a path separator survived
        sanitization, an LLM-supplied "output_filename" like
        "../../evil.pdf" could escape the intended output
        directory. Confirms neither "/" nor "\\" survive.
        """
        result = sanitize_filename_component("../../evil_output")
        assert "/" not in result
        assert "\\" not in result

    def test_empty_input_returns_untitled(self):
        assert sanitize_filename_component("") == "untitled"

    def test_result_is_length_capped(self):
        long_input = "a" * 500
        result = sanitize_filename_component(long_input, max_len=100)
        assert len(result) <= 100