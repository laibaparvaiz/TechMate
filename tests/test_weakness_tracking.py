"""
File: TechMate/tests/test_weakness_tracking.py
Description: Tests for the Mentor Agent's weakness-tracking
topic extraction in app/agents/supervisor.py.

Covers two real bugs found during live testing:
1. The [WEAKNESS_TOPIC: ...] tag not always firing (model
   inconsistency), and the raw-truncation fallback fragmenting
   the SAME exception type across differently-worded questions
   instead of consolidating them.
2. The model narrating its own decision NOT to include the tag
   (e.g. "[No weakness topic to log...]") instead of silently
   omitting it, leaking internal mechanism text into the
   visible response.
"""

import pytest

from app.agents.supervisor import (
    _looks_like_error_report,
    _extract_weakness_topic,
    _extract_weakness_topic_from_response,
)


class TestLooksLikeErrorReport:

    @pytest.mark.parametrize(
        "query",
        [
            "debug this error",
            "why does this throw an IndexError: arr[len(arr)]",
            "getting this error: TypeError: unsupported operand",
            "ZeroDivisionError when I run this",
            "ModuleNotFoundError: No module named 'requests'",
        ],
    )
    def test_error_reports_recognized(self, query):
        """
        The "why does this throw an IndexError" case is a
        regression test for a real bug: the original keyword
        list only had "why does this code", which didn't match
        "why does this throw", silently failing to log a
        legitimate error report.
        """
        assert _looks_like_error_report(query) is True

    @pytest.mark.parametrize(
        "query",
        [
            "explain what this code does",
            "review my code",
            "max_tokens = 100000",
        ],
    )
    def test_non_error_questions_not_flagged(self, query):
        assert _looks_like_error_report(query) is False


class TestExtractWeaknessTopicFallback:

    def test_known_exception_name_is_normalized(self):
        topic = _extract_weakness_topic(
            "Why does this throw an IndexError: arr[len(arr)]"
        )
        assert topic == "IndexError"

    def test_differently_worded_same_exception_consolidates(self):
        """
        Regression test for the exact bug found in production:
        the tag-based path missed a turn, and the OLD raw-
        truncation fallback logged two differently-worded
        IndexError questions as two unrelated topics instead of
        one consolidated "IndexError" entry.
        """
        topic_a = _extract_weakness_topic(
            "Why does this throw an IndexError: arr[len(arr)]"
        )
        topic_b = _extract_weakness_topic(
            "my_list[5] gives IndexError, what's wrong"
        )
        assert topic_a == topic_b == "IndexError"

    def test_benign_llm_parameter_name_not_falsely_matched(self):
        """
        Regression test: an earlier draft of the exception-name
        matching would have matched "TOKEN" as a substring of
        "TOKENS" in "max_tokens", incorrectly treating a
        completely benign LLM parameter as a weakness topic.
        """
        topic = _extract_weakness_topic("max_tokens = 100000")
        assert topic != "IndexError"
        assert "token" not in topic.lower() or topic == "max_tokens = 100000"

    def test_no_known_exception_falls_back_to_truncated_text(self):
        topic = _extract_weakness_topic("my code isn't working as expected")
        assert topic == "my code isn't working as expected"

    def test_long_query_is_truncated(self):
        long_query = "a" * 200
        topic = _extract_weakness_topic(long_query, max_len=80)
        assert len(topic) <= 83  # 80 + "..."


class TestExtractWeaknessTopicFromResponse:

    def test_real_tag_is_extracted_and_stripped(self):
        response = "Classic off-by-one bug here.\n\n[WEAKNESS_TOPIC: Off-by-one errors]"
        display, topic = _extract_weakness_topic_from_response(response, "why does this fail")

        assert topic == "Off-by-one errors"
        assert "[WEAKNESS_TOPIC" not in display

    def test_two_phrasings_produce_same_tag_name(self):
        response_a = "Loop boundary issue.\n\n[WEAKNESS_TOPIC: Off-by-one errors]"
        response_b = "You're one index short.\n\n[WEAKNESS_TOPIC: Off-by-one errors]"

        _, topic_a = _extract_weakness_topic_from_response(response_a, "q1")
        _, topic_b = _extract_weakness_topic_from_response(response_b, "q2")

        assert topic_a == topic_b == "Off-by-one errors"

    def test_leaked_meta_commentary_is_stripped_regression(self):
        """
        Regression test for the exact leak observed in
        production: the model, instead of silently omitting the
        tag as instructed, narrated its own decision not to
        include it. This must be caught by the defensive
        cleanup regex even without an actual [WEAKNESS_TOPIC:]
        tag present.
        """
        response = (
            "A linked list is a data structure...\n"
            "[No weakness topic to log—this is a conceptual explanation request.]"
        )

        display, topic = _extract_weakness_topic_from_response(response, "what is a linked list?")

        assert topic is None
        assert "weakness" not in display.lower()
        assert "[No weakness" not in display

    def test_normal_response_with_no_tag_or_leak_is_unchanged(self):
        response = "This code just prints hello world, nothing special here."
        display, topic = _extract_weakness_topic_from_response(response, "explain this")

        assert display == response
        assert topic is None