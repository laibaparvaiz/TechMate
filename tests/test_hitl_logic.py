"""
File: TechMate/tests/test_hitl_logic.py
Description: Tests for the HITL confirmation/rejection phrase
matching in app/agents/supervisor.py, and the pending-action
registry in app/safety/hitl.py.
"""

import pytest

from app.agents.supervisor import (
    _looks_like_confirmation,
    _looks_like_rejection,
)
from app.safety.hitl import (
    register_pending_action,
    pop_pending_action,
)


class TestConfirmationPhrases:

    @pytest.mark.parametrize(
        "text",
        ["yes", "Yes!", "  Okay  ", "confirm", "sure", "go ahead", "save it"],
    )
    def test_confirmation_phrases_recognized(self, text):
        assert _looks_like_confirmation(text) is True

    @pytest.mark.parametrize(
        "text",
        ["no", "cancel", "what is a linked list?", "edit main.py"],
    )
    def test_non_confirmation_text_not_recognized(self, text):
        assert _looks_like_confirmation(text) is False


class TestRejectionPhrases:

    @pytest.mark.parametrize(
        "text",
        ["no", "No.", "cancel", "don't", "nevermind", "no thanks"],
    )
    def test_rejection_phrases_recognized(self, text):
        assert _looks_like_rejection(text) is True

    @pytest.mark.parametrize(
        "text",
        ["yes", "what is a linked list?", "edit main.py"],
    )
    def test_non_rejection_text_not_recognized(self, text):
        assert _looks_like_rejection(text) is False

    def test_confirmation_and_rejection_are_mutually_exclusive(self):
        """
        No text should ever be classified as both — that would
        make confirmation_node's routing logic ambiguous.
        """
        all_confirm_words = [
            "yes", "y", "confirm", "confirmed", "do it", "go ahead",
            "approve", "approved", "save it", "ok", "okay", "sure",
        ]
        for word in all_confirm_words:
            assert not (
                _looks_like_confirmation(word)
                and _looks_like_rejection(word)
            )


class TestPendingActionRegistry:
    """
    app/safety/hitl.py is an in-process registry — tests here
    exercise it directly rather than through the full graph.
    """

    def test_register_and_pop_returns_the_action(self):
        register_pending_action(
            "test_thread_1",
            {"type": "edit_file", "file_path": "x.py", "content": "y"},
        )

        result = pop_pending_action("test_thread_1")

        assert result is not None
        assert result["type"] == "edit_file"
        assert result["file_path"] == "x.py"

    def test_pop_removes_the_action_not_just_reads_it(self):
        register_pending_action(
            "test_thread_2",
            {"type": "save_study_notes", "title": "t", "markdown_content": "c"},
        )

        first_pop = pop_pending_action("test_thread_2")
        second_pop = pop_pending_action("test_thread_2")

        assert first_pop is not None
        assert second_pop is None

    def test_pop_on_unregistered_thread_returns_none(self):
        assert pop_pending_action("thread_that_never_registered_anything") is None

    def test_different_threads_dont_interfere(self):
        register_pending_action("thread_a", {"type": "edit_file", "file_path": "a.py"})
        register_pending_action("thread_b", {"type": "edit_file", "file_path": "b.py"})

        result_a = pop_pending_action("thread_a")
        result_b = pop_pending_action("thread_b")

        assert result_a["file_path"] == "a.py"
        assert result_b["file_path"] == "b.py"