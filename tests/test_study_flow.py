"""
File: TechMate/tests/test_study_flow.py

Tests for the Study / PDF-notes / confirmation / fallback features.

All tests are offline: no API calls, no files written, no network.
LLM calls are replaced with small fakes via monkeypatch.

Run:
    pytest tests/test_study_flow.py -q
"""

from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

import app.agents.study_agent as study
import app.agents.supervisor as sup
import app.utils.llm_router as router


# ============================================================
# 1. YES / NO CLASSIFICATION
# ============================================================

@pytest.mark.parametrize(
    "text,expected",
    [
        ("yes", "yes"),
        ("Yess", "yes"),
        ("yeah", "yes"),
        ("yep", "yes"),
        ("ys", "yes"),
        ("ok", "yes"),
        ("yes please", "yes"),
        ("no", "no"),
        ("nope", "no"),
        ("Don't", "no"),
        ("cancel", "no"),
        ("no, but show it here", "no"),
        ("yes but make it shorter", "modify"),
        ("yes, make it 5 bullets", "modify"),
        ("what is TCP", "other"),
        ("maybe", "other"),
        ("", "other"),
    ],
)
def test_classify_reply(text, expected):
    assert sup._classify_reply(text) == expected


@pytest.mark.parametrize(
    "text",
    ["yes", "y", "confirm", "confirmed", "do it", "go ahead", "approve",
     "yes please", "save it", "ok", "okay", "sure", "Yes!", "YES."],
)
def test_old_confirmation_helper_still_accepts(text):
    assert sup._looks_like_confirmation(text) is True
    assert sup._looks_like_rejection(text) is False


@pytest.mark.parametrize(
    "text",
    ["no", "n", "cancel", "cancelled", "reject", "don't", "do not", "stop",
     "nevermind", "never mind", "no thanks", "no thank you"],
)
def test_old_rejection_helper_still_accepts(text):
    assert sup._looks_like_rejection(text) is True
    assert sup._looks_like_confirmation(text) is False


@pytest.mark.parametrize(
    "text", ["yes but make it shorter", "what is TCP", "maybe", ""]
)
def test_old_helpers_reject_unclear_replies(text):
    assert sup._looks_like_confirmation(text) is False


def test_extract_instruction():
    assert sup._extract_instruction("yes but make it shorter") == "make it shorter"
    assert sup._extract_instruction("yeah, shorter please") == "shorter please"


# ============================================================
# 2. CLEANING THE QUESTION / TITLES / FILENAMES
# ============================================================

@pytest.mark.parametrize(
    "query,expected",
    [
        ("briefly explain fastapi and make notes as a pdf", "briefly explain fastapi"),
        (
            "briefly explain what is asynchronous and save it as a pdf",
            "briefly explain what is asynchronous",
        ),
        ("Generate notes on REST vs GraphQL as a PDF", "Explain REST vs GraphQL"),
        ("Summarize the startup PDF and make notes.", "Summarize the startup PDF"),
        ("Explain TCP", "Explain TCP"),
    ],
)
def test_strip_save_instructions(query, expected):
    assert sup._strip_save_instructions(query) == expected


def test_pdf_title_from_real_heading():
    answer = "## REST vs GraphQL — Study Notes\nbody"
    assert sup._derive_pdf_title(answer, "x") == "REST vs GraphQL — Study Notes"


def test_pdf_title_skips_web_note_line():
    answer = sup.WEB_SOURCES_NOTE + "\n\n## Real Title\nbody"
    assert sup._derive_pdf_title(answer, "x") == "Real Title"


def test_pdf_title_generic_heading_falls_back_to_topic():
    title = sup._derive_pdf_title(
        "**Summary**\ntext",
        "explain asynchronous and save it as a pdf",
    )
    assert title == "asynchronous"


def test_pdf_title_bold_first_sentence_is_not_a_heading():
    answer = "**Asynchronous means not at the same time.**\n\nmore"
    title = sup._derive_pdf_title(answer, "explain asynchronous and save it as a pdf")
    assert title == "asynchronous"


def test_pdf_title_section_heading_is_not_the_title():
    answer = "Intro sentence.\n\n## In computing and programming\n- a"
    title = sup._derive_pdf_title(answer, "explain fastapi and make notes as a pdf")
    assert title == "fastapi"


def test_make_pdf_filename():
    assert sup._make_pdf_filename("REST vs GraphQL") == "REST_vs_GraphQL.pdf"
    assert sup._make_pdf_filename("???") == "study_notes.pdf"


# ============================================================
# 3. ANSWER CLEANUP HELPERS
# ============================================================

def test_repair_flattened_table():
    flat = "| A | B | |---|---| | 1 | 2 |"
    fixed = sup._repair_flattened_tables(flat)
    assert fixed.splitlines() == ["| A | B |", "|---|---|", "| 1 | 2 |"]


def test_repair_leaves_normal_text_alone():
    text = "Normal line | with one pipe\n\n- bullet"
    assert sup._repair_flattened_tables(text) == text


def test_strip_save_chatter():
    text = "Body\nYou can copy the above notes into a text editor and export/print to PDF.\nEnd"
    cleaned = sup._strip_save_chatter(text)
    assert "export/print" not in cleaned
    assert "Body" in cleaned and "End" in cleaned


def test_strip_source_claim_footer():
    text = "Body\n\nAll points above come directly from the provided Chapter 10.pdf. No outside information was added."
    assert sup._strip_source_claim_footer(text) == "Body"


def test_pdf_friendly_citations_numbers_and_dedupes():
    text = (
        "A (https://a.com/x) and B (https://b.com/y, https://a.com/x)."
    )
    out = sup._pdf_friendly_citations(text)
    assert "https://a.com/x)" not in out.split("## Sources")[0]
    assert "[1]" in out and "[2]" in out
    assert out.count("https://a.com/x") == 1
    assert "## Sources" in out
    assert "1. https://a.com/x" in out
    assert "2. https://b.com/y" in out


def test_pdf_friendly_citations_fullwidth_brackets():
    text = "Fact【https://a.com/x】."
    out = sup._pdf_friendly_citations(text)
    assert "【" not in out and "】" not in out
    assert "1. https://a.com/x" in out


def test_pdf_friendly_citations_no_urls_unchanged():
    text = "No links here [Chapter 10.pdf, page 5]."
    assert sup._pdf_friendly_citations(text) == text


def test_pdf_keeps_web_note_as_plain_text():
    text = sup.WEB_SOURCES_NOTE + "\n\nFact (https://a.com/x)."
    out = sup._pdf_friendly_citations(text)
    assert out.startswith("Note: This wasn't fully covered")


def test_revision_keeps_the_web_note(monkeypatch):
    pending = {"markdown_content": sup.WEB_SOURCES_NOTE + "\n\nLong body here."}
    monkeypatch.setattr(sup, "get_resilient_llm", lambda temperature=0.2: object())
    monkeypatch.setattr(sup, "get_fallback_llm", lambda temperature=0.2: None)
    monkeypatch.setattr(
        sup, "invoke_with_fallback",
        lambda *a, **k: SimpleNamespace(content="Short body."),
    )
    revised = sup._revise_pending_content(pending, "make it shorter")
    assert revised.startswith(sup.WEB_SOURCES_NOTE)
    assert revised.endswith("Short body.")


# ============================================================
# 4. ROUTING
# ============================================================

@pytest.mark.parametrize(
    "query,expected",
    [
        ("hi", "general"),
        ("explain fastapi", "study"),
        ("Explain Git branching", "study"),
        ("what is a linked list?", "study"),
        ("Generate notes on REST vs GraphQL as a PDF", "study"),
        ("explain asynchronous and save it as a pdf", "study"),
        ("According to the startup PDF, what is the funding amount?", "study"),
        ("explain python decorators", "mentor"),
        ("explain this code: x=1", "mentor"),
        ("debug this: x=[1,2]; print(x[5])", "mentor"),
        ("create a file hello.py that prints hi", "coding"),
        ("add a watermark DRAFT to Chapter 10.pdf, save as draft_test.pdf", "coding"),
        ("how do I fix this error in my code", None),
        ("tell me a joke", None),
    ],
)
def test_deterministic_route(query, expected):
    assert sup._deterministic_route(query) == expected


def test_explain_routing_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(sup, "ROUTE_EXPLAIN_TO_STUDY", False)
    assert sup._deterministic_route("explain fastapi") is None


@pytest.mark.parametrize(
    "query,expected",
    [
        ("Generate notes on REST vs GraphQL as a PDF", True),
        ("explain fastapi and make notes as a pdf", True),
        ("explain asynchronous and save it as a pdf", True),
        ("explain fastapi", False),
        ("make flashcards on the Build-Measure-Learn loop", False),
        ("don't save this as a pdf", False),
    ],
)
def test_pdf_save_request_detection(query, expected):
    assert sup._looks_like_pdf_save_request(query) is expected


@pytest.mark.parametrize(
    "query,expected",
    [
        ("make pdf of this", True),
        ("Make PDF of that.", True),
        ("save this as a pdf", True),
        ("save it as pdf", True),
        ("save this answer as a PDF", True),
        ("convert the above to a pdf", True),
        ("explain fastapi and make notes as a pdf", False),
        ("Generate notes on REST vs GraphQL as a PDF", False),
        ("explain asynchronous and save it as a pdf", False),
        ("make a pdf of the startup chapter", False),
        ("explain fastapi", False),
        ("make flashcards on the loop", False),
    ],
)
def test_is_save_previous_request(query, expected):
    assert sup._is_save_previous_request(query) is expected


def test_save_previous_request_routes_to_study_without_llm():
    assert sup._deterministic_route("make pdf of this") == "study"


def _previous_answer_thread():
    return [
        HumanMessage(content="explain business startup"),
        AIMessage(content="A startup is X [Chapter 10.pdf, Page: 4].\n\nIt grows fast."),
    ]


def test_save_previous_answer_builds_pending_pdf_without_pipeline(monkeypatch):
    def must_not_run(*args, **kwargs):
        raise AssertionError("the study pipeline must not run for this request")

    monkeypatch.setattr(sup, "run_verified_study_agent", must_not_run)
    messages = _previous_answer_thread() + [HumanMessage(content="make pdf of this")]
    result = sup.study_node({"messages": messages}, {})
    pending = result["pending_action"]
    assert pending["type"] == "generate_pdf_document"
    assert pending["output_filename"] == "business_startup.pdf"
    assert "A startup is X" in pending["markdown_content"]
    assert "Reply 'yes'" in result["final_response"]


def test_save_previous_answer_skips_supervisor_messages(monkeypatch):
    monkeypatch.setattr(
        sup, "run_verified_study_agent",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("pipeline ran")),
    )
    messages = _previous_answer_thread() + [
        HumanMessage(content="make pdf of this"),
        AIMessage(content="PDF: I can save my previous answer as a PDF named 'x.pdf'. Reply 'yes' to confirm, or 'no' to skip."),
        HumanMessage(content="no"),
        AIMessage(content="Okay, I won't save the PDF."),
        HumanMessage(content="make pdf of this"),
    ]
    result = sup.study_node({"messages": messages}, {})
    assert "A startup is X" in result["pending_action"]["markdown_content"]


def test_save_previous_answer_with_nothing_to_save():
    result = sup.study_node({"messages": [HumanMessage(content="make pdf of this")]}, {})
    assert result["pending_action"] is None
    assert "don't have an answer" in result["final_response"]


# ============================================================
# 5. CONFIRMATION NODE (HITL)
# ============================================================

PENDING_PDF = {
    "type": "generate_pdf_document",
    "title": "t",
    "markdown_content": "# T\n\nbody",
    "output_filename": "t.pdf",
}


def run_confirmation(messages, **state):
    return sup.confirmation_node({"messages": messages, **state})


def test_yes_executes_pending_action(monkeypatch):
    monkeypatch.setattr(sup, "_execute_pending_action", lambda pending: "DONE")
    result = run_confirmation([HumanMessage(content="yes")], pending_action=PENDING_PDF)
    assert result["confirmation_handled"] is True
    assert result["final_response"] == "DONE"
    assert result["pending_action"] is None


def test_no_cancels_pdf():
    result = run_confirmation([HumanMessage(content="no")], pending_action=PENDING_PDF)
    assert result["confirmation_handled"] is True
    assert "won't save the PDF" in result["final_response"]
    assert result["pending_action"] is None


def test_unrelated_message_drops_pending_and_sets_hint():
    result = run_confirmation(
        [HumanMessage(content="what is an array?")],
        pending_action=PENDING_PDF,
    )
    assert result["confirmation_handled"] is False
    assert result["pending_action"] is None
    assert "t.pdf" in result["expired_action_hint"]


def test_late_yes_after_expiry_gets_clear_message():
    result = run_confirmation(
        [HumanMessage(content="yes")],
        pending_action=None,
        expired_action_hint="PDF 't.pdf'",
    )
    assert result["confirmation_handled"] is True
    assert "expired" in result["final_response"]


def test_stray_yes_with_nothing_offered_is_answered_directly():
    messages = [
        HumanMessage(content="make flashcards"),
        AIMessage(content="- Q: a\nA: b"),
        HumanMessage(content="yes"),
    ]
    result = run_confirmation(messages, pending_action=None)
    assert result["confirmation_handled"] is True
    assert "nothing waiting" in result["final_response"].lower()


def test_yes_after_an_assistant_offer_goes_to_normal_chat():
    messages = [
        HumanMessage(content="explain x"),
        AIMessage(content="Here you go. Let me know if you'd like more!"),
        HumanMessage(content="yes"),
    ]
    result = run_confirmation(messages, pending_action=None)
    assert result["confirmation_handled"] is False


def test_ok_thanks_is_not_treated_as_confirmation():
    messages = [
        HumanMessage(content="q"),
        AIMessage(content="- Q: a\nA: b"),
        HumanMessage(content="ok thanks"),
    ]
    result = run_confirmation(messages, pending_action=None)
    assert result["confirmation_handled"] is False


def test_yes_but_shorter_revises_and_asks_again(monkeypatch):
    monkeypatch.setattr(
        sup, "_revise_pending_content", lambda pending, instruction: "# Short\n\nshort body"
    )
    result = run_confirmation(
        [HumanMessage(content="yes but make it shorter")],
        pending_action=PENDING_PDF,
    )
    assert result["confirmation_handled"] is True
    assert result["pending_action"]["markdown_content"].startswith("# Short")
    assert "Reply 'yes'" in result["final_response"]


def test_yes_but_shorter_keeps_pending_if_revision_fails(monkeypatch):
    monkeypatch.setattr(sup, "_revise_pending_content", lambda pending, instruction: None)
    result = run_confirmation(
        [HumanMessage(content="yes but make it shorter")],
        pending_action=PENDING_PDF,
    )
    assert result["confirmation_handled"] is True
    assert "pending_action" not in result  # left untouched in state


def test_nothing_pending_and_normal_question_passes_through():
    result = run_confirmation(
        [HumanMessage(content="explain fastapi")],
        pending_action=None,
    )
    assert result["confirmation_handled"] is False


# ============================================================
# 6. CRAG LABELS (study_agent.check_crag_relevance)
# ============================================================

class FakeLLM:
    def with_structured_output(self, schema):
        return self


def make_docs(count):
    return [
        Document(
            page_content=f"chunk {index}",
            metadata={"source": "Chapter 10.pdf", "page": 5},
        )
        for index in range(count)
    ]


def patch_crag(monkeypatch, relevant_flags):
    decision = SimpleNamespace(
        documents=[
            SimpleNamespace(document_index=i + 1, relevant=flag, reason="r")
            for i, flag in enumerate(relevant_flags)
        ]
    )
    monkeypatch.setattr(study, "get_resilient_llm", lambda temperature=0.0: FakeLLM())
    monkeypatch.setattr(study, "get_fallback_llm", lambda temperature=0.0: None)
    monkeypatch.setattr(study, "invoke_with_fallback", lambda *a, **k: decision)


def test_crag_correct_with_two_relevant(monkeypatch):
    patch_crag(monkeypatch, [True, True, False, False, False])
    status, relevant, _ = study.check_crag_relevance("q", make_docs(5))
    assert status == "correct" and len(relevant) == 2


def test_crag_ambiguous_with_one_relevant(monkeypatch):
    patch_crag(monkeypatch, [True, False, False, False, False])
    status, relevant, _ = study.check_crag_relevance("q", make_docs(5))
    assert status == "ambiguous" and len(relevant) == 1


def test_crag_incorrect_with_none_relevant(monkeypatch):
    patch_crag(monkeypatch, [False] * 5)
    status, relevant, _ = study.check_crag_relevance("q", make_docs(5))
    assert status == "incorrect" and relevant == []


def test_crag_no_documents_is_incorrect_without_llm():
    status, relevant, _ = study.check_crag_relevance("q", [])
    assert status == "incorrect" and relevant == []


def test_crag_none_result_is_error(monkeypatch):
    monkeypatch.setattr(study, "get_resilient_llm", lambda temperature=0.0: FakeLLM())
    monkeypatch.setattr(study, "get_fallback_llm", lambda temperature=0.0: None)
    monkeypatch.setattr(study, "invoke_with_fallback", lambda *a, **k: None)
    status, relevant, _ = study.check_crag_relevance("q", make_docs(3))
    assert status == "error" and relevant == []


def test_crag_exception_is_error_not_incorrect(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("RESOURCE_EXHAUSTED 429")

    monkeypatch.setattr(study, "get_resilient_llm", lambda temperature=0.0: FakeLLM())
    monkeypatch.setattr(study, "get_fallback_llm", lambda temperature=0.0: None)
    monkeypatch.setattr(study, "invoke_with_fallback", boom)
    status, _, reason = study.check_crag_relevance("q", make_docs(3))
    assert status == "error"
    assert "429" in reason  # real error kept for the quota message


# ============================================================
# 7. SELF-RAG STATUSES
# ============================================================

def patch_self_rag(monkeypatch, result):
    monkeypatch.setattr(study, "get_resilient_llm", lambda temperature=0.0: FakeLLM())
    monkeypatch.setattr(study, "get_fallback_llm", lambda temperature=0.0: None)
    monkeypatch.setattr(study, "invoke_with_fallback", lambda *a, **k: result)


def test_self_rag_pass(monkeypatch):
    patch_self_rag(
        monkeypatch,
        SimpleNamespace(is_grounded=True, hallucination_detected=False, notes="ok"),
    )
    assert study.verify_study_response("q", "ctx", "answer")[0] == "pass"


def test_self_rag_fail_on_hallucination(monkeypatch):
    patch_self_rag(
        monkeypatch,
        SimpleNamespace(is_grounded=True, hallucination_detected=True, notes="bad"),
    )
    assert study.verify_study_response("q", "ctx", "answer")[0] == "fail"


def test_self_rag_none_result_is_error(monkeypatch):
    patch_self_rag(monkeypatch, None)
    assert study.verify_study_response("q", "ctx", "answer")[0] == "error"


def test_self_rag_no_context_fails_without_llm():
    assert study.verify_study_response("q", "", "answer")[0] == "fail"


# ============================================================
# 8. CITATIONS AND ERROR MESSAGES (study_agent)
# ============================================================

def test_resolve_citations_web_and_kb():
    web = [
        Document(page_content="w1", metadata={"source": "https://a.com/x"}),
        Document(page_content="w2", metadata={"source": "https://b.com/y"}),
    ]
    kb = [
        Document(
            page_content="k1",
            metadata={"source": "C:\\data\\Chapter 10.pdf", "page": 5},
        )
    ]
    text = "One Web source 2. Two (Web source 1). Three Document 1 says so."
    out = study._resolve_citations(text, kb, web)
    assert "[https://b.com/y]" in out
    assert "[https://a.com/x]" in out
    assert "[Chapter 10.pdf, page 5] says so" in out
    assert "Web source" not in out and "Document 1" not in out


def test_resolve_citations_unknown_number_is_dropped():
    out = study._resolve_citations("Fact Document 9 here.", [], [])
    assert "Document 9" not in out


def test_busy_message_quota_vs_temporary():
    quota = study._busy_message("Error: RESOURCE_EXHAUSTED 429 PerDay")
    assert quota == study.QUOTA_MESSAGE
    temporary = study._busy_message("CRAG returned no document evaluations.")
    assert temporary == study.SERVICE_BUSY_MESSAGE


# ============================================================
# 9. LLM ROUTER: FALLBACK, NONE RESULT, CIRCUIT BREAKER
# ============================================================

class Boom:
    def __init__(self, message):
        self.message = message
        self.calls = 0

    def invoke(self, *args, **kwargs):
        self.calls += 1
        raise Exception(self.message)

    def stream(self, *args, **kwargs):
        self.calls += 1
        raise Exception(self.message)
        yield  # pragma: no cover


class Fine:
    def __init__(self, value="ok"):
        self.value = value
        self.calls = 0

    def invoke(self, *args, **kwargs):
        self.calls += 1
        return self.value

    def stream(self, *args, **kwargs):
        self.calls += 1
        yield self.value


@pytest.fixture(autouse=True)
def reset_breaker(monkeypatch):
    """The circuit breaker is module-level state; reset it for every test."""
    monkeypatch.setattr(router, "_PRIMARY_BLOCKED_UNTIL", 0.0)


@pytest.mark.parametrize(
    "message,expected",
    [
        ("Error code: 429 - Rate limit exceeded", True),
        ("Rate limit exceeded: free-models-per-day", True),
        ("Upstream error from Nvidia: Service temporarily overloaded", True),
        ("Structured Output response does not have a 'parsed' field", True),
        ("Invalid API key", False),
        ("used 14290 tokens", False),
    ],
)
def test_is_rate_limit_error(message, expected):
    assert router.is_rate_limit_error(Exception(message)) is expected


def test_fallback_used_on_retryable_error():
    primary, fallback = Boom("Error code: 429 rate limit"), Fine("from fallback")
    assert router.invoke_with_fallback(primary, fallback, "x") == "from fallback"


def test_non_retryable_error_is_raised():
    with pytest.raises(Exception):
        router.invoke_with_fallback(Boom("Invalid API key"), Fine(), "x")


def test_none_result_triggers_fallback():
    class ReturnsNone:
        def invoke(self, *args, **kwargs):
            return None

    assert router.invoke_with_fallback(ReturnsNone(), Fine("rescued"), "x") == "rescued"


def test_none_result_without_fallback_returns_none():
    class ReturnsNone:
        def invoke(self, *args, **kwargs):
            return None

    assert router.invoke_with_fallback(ReturnsNone(), None, "x") is None


def test_circuit_breaker_skips_primary_after_daily_quota():
    primary = Boom("Rate limit exceeded: free-models-per-day. 429")
    fallback = Fine()
    for _ in range(3):
        assert router.invoke_with_fallback(primary, fallback, "x") == "ok"
    assert primary.calls == 1  # tried once, then skipped
    assert fallback.calls == 3
    assert router._primary_is_blocked() is True


def test_circuit_breaker_not_tripped_by_overload():
    primary = Boom("Upstream error: Service temporarily overloaded")
    router.invoke_with_fallback(primary, Fine(), "x")
    assert router._primary_is_blocked() is False


def test_stream_falls_back_before_first_chunk():
    chunks = list(router.stream_with_fallback(Boom("429 rate limit"), Fine("chunk"), "x"))
    assert chunks == ["chunk"]