"""
File: TechMate/tests/test_privacy.py
Description: Tests for app/safety/privacy.py's redact_sensitive_data().
"""

import pytest

from app.safety.privacy import redact_sensitive_data


class TestRealSecretsAreCaught:

    def test_gemini_api_key_redacted(self):
        text = "my key is AIzaSyD1234567890abcdefghijklmnopqrstuv12"
        result = redact_sensitive_data(text)
        assert "AIzaSyD" not in result
        assert "REDACTED" in result

    def test_openrouter_key_redacted(self):
        text = "OPENROUTER_API_KEY=sk-or-v1-abcdefghijklmnopqrstuvwxyz1234567890"
        result = redact_sensitive_data(text)
        assert "sk-or-v1-abcdefghijklmnopqrstuvwxyz1234567890" not in result

    def test_generic_sk_prefixed_key_redacted(self):
        text = "the key is sk-abcdefghijklmnopqrstuvwxyz1234567890"
        result = redact_sensitive_data(text)
        assert "sk-abcdefghijklmnopqrstuvwxyz1234567890" not in result

    def test_bearer_token_redacted(self):
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9abcdefgh"
        result = redact_sensitive_data(text)
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9abcdefgh" not in result

    def test_postgres_password_in_connection_string_redacted(self):
        text = "DATABASE_URL=postgresql://neondb_owner:realpassword123@ep-test.neon.tech/neondb"
        result = redact_sensitive_data(text)
        assert "realpassword123" not in result
        assert "neondb_owner" in result  # username can stay visible

    def test_postgres_short_scheme_also_redacted(self):
        text = "postgres://user:secretpass@host/db"
        result = redact_sensitive_data(text)
        assert "secretpass" not in result

    def test_generic_env_style_password_redacted(self):
        text = "POSTGRES_PASSWORD=mySecretPass123"
        result = redact_sensitive_data(text)
        assert "mySecretPass123" not in result
        assert "POSTGRES_PASSWORD" in result  # key name can stay visible


class TestFalsePositivePrevention:

    def test_max_tokens_parameter_not_redacted(self):
        """
        Regression guard: an LLM API parameter like max_tokens
        should never be mistaken for a secret just because it
        contains the substring "token".
        """
        text = "max_tokens = 100000"
        result = redact_sensitive_data(text)
        assert result == text

    def test_token_count_variable_not_redacted(self):
        text = "token_count = 5000000"
        result = redact_sensitive_data(text)
        assert result == text


class TestEdgeCases:

    def test_empty_string_returns_empty(self):
        assert redact_sensitive_data("") == ""

    def test_none_input_returns_empty_string(self):
        assert redact_sensitive_data(None) == ""

    def test_text_with_no_secrets_is_unchanged(self):
        text = "This is a perfectly normal sentence about linked lists."
        assert redact_sensitive_data(text) == text