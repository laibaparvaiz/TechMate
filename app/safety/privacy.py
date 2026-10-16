"""
File: TechMate/app/safety/privacy.py
Description: PII and API key redaction utilities.
"""

import re

# Compound secret-key identifier names, matched with optional
# underscore-joined prefix/suffix segments (e.g. "OPENROUTER_"
# + "API_KEY", or "POSTGRES_" + "PASSWORD"). Deliberately does
# NOT include bare "TOKEN" or "SECRET" alone — those would
# false-positive on entirely benign names like "max_tokens" in
# ordinary LLM API call examples the Coding/Mentor agents might
# legitimately show. Requires an assignment (= or :) followed
# by a reasonably long value to avoid matching bare mentions.
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b((?:[A-Z0-9]+_)*(?:API_KEY|APIKEY|SECRET_KEY|"
    r"SECRETKEY|CLIENT_SECRET|PASSWORD|PASSWD|ACCESS_TOKEN|"
    r"AUTH_TOKEN|REFRESH_TOKEN|PRIVATE_KEY)"
    r"(?:_[A-Z0-9]+)*)\s*[:=]\s*[\"']?[\w\-./+]{6,}[\"']?"
)


def redact_sensitive_data(text: str) -> str:
    """Scans text for common API keys, passwords, or tokens and redacts them."""
    if not text:
        return ""

    # Redact Google/Gemini API keys (AIza...)
    text = re.sub(
        r'AIza[0-9A-Za-z\-_]{35}',
        '[REDACTED_GEMINI_KEY]',
        text,
    )

    # Redact OpenRouter API keys (sk-or-v1-...)
    text = re.sub(
        r'sk-or-v1-[A-Za-z0-9]{20,}',
        '[REDACTED_OPENROUTER_KEY]',
        text,
    )

    # Redact generic "sk-"-prefixed secret keys — a broad net
    # for any other provider using this common convention
    # (OpenAI, OpenAI-compatible proxies, etc.). Checked AFTER
    # the more specific OpenRouter pattern above so that one
    # takes precedence for its exact format.
    text = re.sub(
        r'\bsk-[A-Za-z0-9]{20,}\b',
        '[REDACTED_API_KEY]',
        text,
    )

    # Redact standard bearer/authorization tokens
    text = re.sub(
        r'Bearer\s+[A-Za-z0-9\-._~+/]+=*',
        'Bearer [REDACTED_TOKEN]',
        text,
    )

    # Redact database connection URIs with passwords — both the
    # "postgresql://" and short "postgres://" scheme forms
    # (different providers/tools use either).
    text = re.sub(
        r'postgres(?:ql)?://([^:]+):([^@]+)@',
        r'postgresql://\1:****@',
        text,
    )

    # Redact generic KEY=VALUE style secrets — catches raw .env
    # or config file content if it ever gets echoed back (e.g.
    # via the Coding Agent's read_project_file tool reading
    # .env or config.py). Keeps the key NAME visible for
    # debugging context, redacts only the value.
    text = _SECRET_ASSIGNMENT_RE.sub(
        r'\1=[REDACTED]',
        text,
    )

    return text