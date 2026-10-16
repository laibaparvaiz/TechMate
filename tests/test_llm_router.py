"""
File: TechMate/tests/test_llm_router.py
Description: Tests for app/utils/llm_router.py's fallback
detection logic.

Covers two real failure modes discovered during live testing
that were NOT originally caught by is_rate_limit_error(), each
of which silently fell through to a generic "something went
wrong" error instead of triggering the Gemini fallback:

1. A technically-200-OK but EMPTY OpenRouter response with
   finish_reason='error' when asked for structured output.
2. The upstream model provider (Nvidia, hosting the free
   nemotron model) being temporarily overloaded and rejecting
   the request outright.
"""

import pytest

from app.utils.llm_router import is_rate_limit_error


class FakeError(Exception):
    """Lets us construct an exception with arbitrary text,
    since is_rate_limit_error() only inspects str(error)."""
    pass


class TestOriginalRateLimitDetection:

    def test_429_error_code_detected(self):
        err = FakeError("Error code: 429 - Rate limit exceeded")
        assert is_rate_limit_error(err) is True

    def test_free_models_per_day_detected(self):
        err = FakeError(
            "Rate limit exceeded: free-models-per-day. "
            "Add 10 credits to unlock 1000 free model requests per day"
        )
        assert is_rate_limit_error(err) is True

    def test_resource_exhausted_detected(self):
        err = FakeError("google.api_core.exceptions.ResourceExhausted: quota exceeded")
        assert is_rate_limit_error(err) is True


class TestMalformedStructuredOutputRegression:

    def test_missing_parsed_field_error_detected(self):
        """
        Regression test for the exact error text observed in
        production: OpenRouter's free nemotron model returned an
        empty response with finish_reason='error', which
        LangChain's with_structured_output() then failed to
        parse. This text contains no rate-limit keyword at all,
        so the ORIGINAL is_rate_limit_error() missed it
        entirely.
        """
        err = FakeError(
            "Structured Output response does not have a 'parsed' "
            "field nor a 'refusal' field. Received message: "
            "content='' additional_kwargs={'parsed': None, "
            "'refusal': None} response_metadata={'finish_reason': 'error'}"
        )
        assert is_rate_limit_error(err) is True


class TestUpstreamOverloadRegression:

    def test_nvidia_overload_error_detected(self):
        """
        Regression test for the exact error text observed in
        production: "Upstream error from Nvidia: Service
        temporarily overloaded" — also contained no rate-limit
        keyword and was missed before this pattern was added.
        """
        err = FakeError("Upstream error from Nvidia: Service temporarily overloaded")
        assert is_rate_limit_error(err) is True

    @pytest.mark.parametrize(
        "error_text",
        [
            "503 Service Unavailable",
            "502 Bad Gateway",
            "The model is currently overloaded, please try again later",
        ],
    )
    def test_generic_overload_variants_detected(self, error_text):
        assert is_rate_limit_error(FakeError(error_text)) is True


class TestFalsePositivePrevention:

    @pytest.mark.parametrize(
        "error_text",
        [
            "ValueError: invalid literal for int() with base 10: 'abc'",
            "FileNotFoundError: [Errno 2] No such file or directory",
            "KeyError: 'missing_key'",
        ],
    )
    def test_unrelated_errors_not_falsely_flagged(self, error_text):
        """
        Broadening the detection patterns must not turn this
        into a catch-all for every exception — only genuine
        rate-limit/quota/overload signals should trigger a
        fallback retry. An unrelated bug retrying on a second
        provider would just waste that provider's quota too.
        """
        assert is_rate_limit_error(FakeError(error_text)) is False