"""
File: TechMate/tests/test_routing.py
Description: Tests for app/agents/supervisor.py's deterministic router.

Every parametrized case below marked "regression" corresponds
to an actual bug found and fixed during this project's testing,
not a hypothetical — see the inline notes on each.
"""

import pytest

from app.agents.supervisor import _deterministic_route


class TestGeneralRouting:

    @pytest.mark.parametrize(
        "query",
        ["hi", "hello", "hey", "thanks", "thank you", "bye", "how are you"],
    )
    def test_casual_phrases_route_general(self, query):
        assert _deterministic_route(query) == "general"

    def test_empty_query_routes_general(self):
        assert _deterministic_route("") == "general"


class TestCodingRouting:

    @pytest.mark.parametrize(
        "query",
        [
            "create a file called hello.py",
            "edit the file to fix the bug",
            "run the tests",
            "run pytest",
            "delete the file",
        ],
    )
    def test_explicit_coding_action_phrases(self, query):
        assert _deterministic_route(query) == "coding"

    @pytest.mark.parametrize(
        "query",
        [
            "edit main.py to add a comment",
            "fix app.js",
            "update config.yaml please",
            "delete old_notes.txt",
        ],
    )
    def test_action_verb_plus_specific_filename_regression(self, query):
        """
        Regression test: these phrasings name a specific file
        instead of saying "the file"/"this file", and originally
        fell through to the LLM fallback router unnecessarily
        (extra cost, and exposed to free-model flakiness) before
        the filename-action regex was added.
        """
        assert _deterministic_route(query) == "coding"

    @pytest.mark.parametrize(
        "query",
        [
            'add a watermark saying "DRAFT" to chapter.10, save as draft_test.pdf',
            "merge chapter1.pdf and chapter2.pdf into combined.pdf",
            "split the pdf notes.pdf, extract pages 1-3",
            "fill out the form application.pdf with my details",
        ],
    )
    def test_pdf_actions_regression(self, query):
        """
        Regression test for the exact bug found in production
        testing: a PDF watermark request containing the word
        "chapter" (as in "chapter.10") was matching the loose
        academic_terms keyword list and getting misrouted to the
        Study Agent instead of the Coding Agent, where the PDF
        tools actually live. PDF-specific action detection must
        take priority over the academic_terms check.
        """
        assert _deterministic_route(query) == "coding"


class TestMentorRouting:

    @pytest.mark.parametrize(
        "query",
        [
            "review my code",
            "explain this code",
            "debug this",
            "why does this code fail",
            "what is wrong with this code",
            "python error",
            "stack trace",
        ],
    )
    def test_mentor_phrases(self, query):
        assert _deterministic_route(query) == "mentor"


class TestStudyRouting:

    @pytest.mark.parametrize(
        "query",
        [
            "quiz me on chapter 2",
            "make flashcards on this topic",
            "summarize my notes",
            "according to my uploaded material",
        ],
    )
    def test_study_phrases(self, query):
        assert _deterministic_route(query) == "study"


class TestAmbiguousFallsThroughToNone:

    @pytest.mark.parametrize(
        "query",
        [
            "can you help me out",
            "what do you think about this",
            "I have a question",
        ],
    )
    def test_ambiguous_queries_return_none(self, query):
        """
        None signals "not deterministically routable" — the
        caller (router_node) falls back to a single LLM routing
        call for these, which is correct/expected. This is NOT
        a bug — it's the deliberate design for genuinely
        ambiguous requests.
        """
        assert _deterministic_route(query) is None