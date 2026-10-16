"""
File: TechMate/app/safety/limits.py
Description: Last-resort error boundary for the FastAPI /chat endpoint.

History: originally caught only google.api_core.exceptions.
ResourceExhausted (a Gemini-specific rate-limit exception).
That never actually fired in practice once OpenRouter became
the primary provider — OpenRouter's 429s don't raise that
exception type, and every internal call site (routing, CRAG,
Self-RAG, generation — see app/agents/supervisor.py and
app/agents/study_agent.py) already catches its own errors and
converts them to friendly responses before anything reaches
this decorator. The real rate-limit/fallback protection lives
in app/utils/llm_router.py's is_rate_limit_error() /
invoke_with_fallback().

Repurposed as a genuine last-resort catch-all instead: if
something truly unexpected slips through every inner layer (a
bug, a database connection issue, an unanticipated error type),
this logs the FULL exception server-side for debugging, but
returns a generic, safe message rather than leaking raw
exception text to the API consumer — main.py's chat_endpoint
previously did `detail=f"Internal Agent Error: {str(err)}"` on
any exception, which exposes internal details (file paths,
connection info, etc.) to whoever calls the API.
"""

import traceback
from functools import wraps


def handle_gemini_rate_limit(func):
    """
    Last-resort error boundary. Wrap the top-level
    execute_agent_safely() call with this so that ANY
    unexpected exception that slipped past every inner
    try/except still returns a clean, safe response instead of
    an unhandled exception (which main.py would otherwise turn
    into an HTTPException containing the raw exception text).
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Full details server-side only — never sent to the
            # API consumer.
            print("Unhandled error in execute_agent_safely:")
            traceback.print_exc()

            return {
                "response": (
                    "⚠️ Something went wrong processing your "
                    "request. Please try again in a moment."
                ),
                "intent": "error",
                "is_grounded": True,
            }
    return wrapper