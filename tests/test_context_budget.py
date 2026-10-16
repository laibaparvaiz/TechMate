"""
File: TechMate/tests/test_context_budget.py
Description: Tests for app/utils/context_budget.py.
"""

from app.utils.context_budget import (
    truncate_to_budget,
    truncate_each,
    DEFAULT_MAX_CHARS,
)


class TestTruncateToBudget:

    def test_short_text_is_unaffected(self):
        text = "This is a short answer."
        assert truncate_to_budget(text, max_chars=1000) == text

    def test_long_text_is_truncated_to_exact_budget(self):
        text = "A" * 5000
        result = truncate_to_budget(text, max_chars=1000)
        assert len(result) == 1000

    def test_truncated_text_has_visible_marker(self):
        text = "A" * 5000
        result = truncate_to_budget(text, max_chars=1000)
        assert "truncated" in result.lower()

    def test_empty_text_returns_empty(self):
        assert truncate_to_budget("", max_chars=1000) == ""

    def test_default_budget_is_reasonable(self):
        # Sanity check that the constant hasn't been accidentally
        # set to something absurd (too small to be useful, or so
        # large it defeats the purpose).
        assert 2000 <= DEFAULT_MAX_CHARS <= 50000


class TestTruncateEach:

    def test_each_item_capped_independently(self):
        items = ["A" * 3000, "short", "B" * 3000]
        result = truncate_each(items, max_chars_each=500)

        assert len(result) == 3
        assert len(result[0]) == 500
        assert result[1] == "short"
        assert len(result[2]) == 500

    def test_empty_list_returns_empty_list(self):
        assert truncate_each([], max_chars_each=500) == []