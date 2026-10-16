"""
File: TechMate/app/utils/llm_router.py

Description:
Resilient LLM factory for TechMate agents.

Two layers of resilience:

    1. OpenRouter model list fallback (extra_body["models"])
       -> protects against a SPECIFIC model being down/broken.
       -> does NOT protect against the free-tier daily quota
          being exhausted (account-wide across ":free" models).

    2. Cross-provider fallback (OpenRouter -> Gemini)
       -> protects against the free-tier quota itself being
          exhausted, by switching to a different provider.

Changes in this version:
    - max_retries lowered so the fallback is reached faster.
    - invoke_with_fallback also falls back when the primary returns
      None (structured output sometimes returns None without raising).
    - HTTP status codes matched with word boundaries (no false
      positives from numbers inside other text).
    - Logs when the fallback provider is actually used.
"""

import re
import time

from langchain_openai import ChatOpenAI

from app.config import (
    OPENROUTER_API_KEY,
    GEMINI_API_KEY,
    PRIMARY_MODEL,
)


# ============================================================
# PRIMARY LLM FACTORY (OPENROUTER)
# ============================================================

def get_resilient_llm(temperature: float = 0.2):
    """
    Primary LLM on OpenRouter free models. The "models" list in
    extra_body is OpenRouter's own model-level fallback.
    """

    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY must be set in the environment variables. "
            "Get a free key at https://openrouter.ai"
        )

    return ChatOpenAI(
        model="nvidia/nemotron-3-ultra-550b-a55b:free",
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=temperature,
        # Was 2. Each internal retry adds backoff delay before the
        # Gemini fallback is even tried.
        max_retries=1,
        extra_body={
            "models": [
                "nvidia/nemotron-3-super-120b-a12b:free",
                "openai/gpt-oss-120b:free",
                "poolside/laguna-s-2.1:free",
            ]
        },
    )


# ============================================================
# FALLBACK LLM FACTORY (GEMINI)
# ============================================================

def get_fallback_llm(temperature: float = 0.2):
    """
    Fallback LLM on a different provider (Gemini).

    Returns None if no Gemini key is configured, so callers must
    always be prepared for a None fallback.
    """

    if not GEMINI_API_KEY:

        print(
            "GEMINI_API_KEY not configured - no cross-provider "
            "fallback available."
        )

        return None

    try:

        from langchain_google_genai import ChatGoogleGenerativeAI

    except ImportError:

        print(
            "langchain-google-genai is not installed. "
            "Run: pip install langchain-google-genai"
        )

        return None

    return ChatGoogleGenerativeAI(
        model=PRIMARY_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=temperature,
    )


# ============================================================
# RETRYABLE ERROR DETECTION
# ============================================================

_STATUS_CODE_RE = re.compile(r"\b(429|502|503)\b")


def is_rate_limit_error(error: Exception) -> bool:
    """
    True if the error is worth retrying on the fallback provider:
    rate limits, quota, overloaded upstream, empty structured output.

    Name kept for backward compatibility with existing call sites.
    """

    text = str(error).lower()

    return (
        bool(_STATUS_CODE_RE.search(text))
        or "rate limit" in text
        or "rate_limit" in text
        or "resource_exhausted" in text
        or "quota" in text
        or "free-models-per-day" in text
        or "does not have a 'parsed' field" in text
        or "finish_reason': 'error'" in text
        or 'finish_reason": "error"' in text
        or "overloaded" in text
        or "upstream error" in text
        or "service unavailable" in text
    )


is_retryable_provider_error = is_rate_limit_error


# ============================================================
# CIRCUIT BREAKER (skip a quota-exhausted primary for a while)
# ============================================================
#
# When OpenRouter's free daily quota is exhausted, EVERY call would
# otherwise try the primary, fail with 429, then go to the fallback.
# A single answer makes ~5 calls, so that wastes ~5 failed requests.
# After a quota error we send calls straight to the fallback for a
# while. Only applies when a fallback exists.

_PRIMARY_BLOCKED_UNTIL = 0.0


def _is_quota_error(error: Exception) -> bool:
    """429 / quota style errors (NOT per-request glitches like empty output)."""

    text = str(error).lower()

    return (
        bool(re.search(r"\b429\b", text))
        or "rate limit" in text
        or "rate_limit" in text
        or "quota" in text
        or "free-models-per-day" in text
        or "resource_exhausted" in text
    )


def _mark_primary_blocked(error: Exception) -> None:

    global _PRIMARY_BLOCKED_UNTIL

    text = str(error).lower()

    # Daily quota: stay on the fallback for 30 minutes.
    # Short-term rate limit: 60 seconds.
    seconds = 1800 if "per-day" in text or "daily" in text else 60

    _PRIMARY_BLOCKED_UNTIL = time.time() + seconds

    print(
        f"Primary LLM paused for {seconds}s "
        "(quota/rate limit); using fallback provider directly."
    )


def _primary_is_blocked() -> bool:

    return time.time() < _PRIMARY_BLOCKED_UNTIL


# ============================================================
# INVOKE WITH CROSS-PROVIDER FALLBACK
# ============================================================

def invoke_with_fallback(
    primary,
    fallback,
    *args,
    **kwargs,
):
    """
    Call primary.invoke(). Falls back once to `fallback` when:
      - the primary raises a retryable error, or
      - the primary returns None (empty structured output).

    Any non-retryable error is raised normally. If the primary
    returned None and there is no fallback, None is returned.
    """

    if fallback is not None and _primary_is_blocked():

        return fallback.invoke(*args, **kwargs)

    try:

        result = primary.invoke(*args, **kwargs)

        if result is not None:
            return result

        print("Primary LLM returned None; trying fallback provider.")

    except Exception as e:

        if fallback is None or not is_rate_limit_error(e):
            raise

        if _is_quota_error(e):
            _mark_primary_blocked(e)

        print(
            "Primary LLM failed, falling back to secondary "
            f"provider: {str(e)[:200]}"
        )

    if fallback is None:
        return None

    result = fallback.invoke(*args, **kwargs)

    print("Answered by fallback provider (Gemini).")

    return result


# ============================================================
# STREAM WITH CROSS-PROVIDER FALLBACK
# ============================================================

def stream_with_fallback(
    primary,
    fallback,
    *args,
    **kwargs,
):
    """
    Stream from primary. If it fails with a retryable error BEFORE
    any chunk was yielded and a fallback exists, restart the stream
    on the fallback. Mid-stream failures are raised as-is.
    """

    if fallback is not None and _primary_is_blocked():

        for chunk in fallback.stream(*args, **kwargs):

            yield chunk

        return

    yielded_any = False

    try:

        for chunk in primary.stream(*args, **kwargs):

            yielded_any = True

            yield chunk

    except Exception as e:

        if (
            not yielded_any
            and fallback is not None
            and is_rate_limit_error(e)
        ):

            if _is_quota_error(e):
                _mark_primary_blocked(e)

            print(
                "Primary LLM failed before generation started, "
                f"falling back to secondary provider: {str(e)[:200]}"
            )

            for chunk in fallback.stream(*args, **kwargs):

                yield chunk

        else:

            raise