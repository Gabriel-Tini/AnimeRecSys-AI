"""
tests/unit/test_config.py
Comprehensive unit test suite for application Settings, SecretStr isolation,
CORS security validation, strict path resolution, and SensitiveDataFilter logging redaction.
"""

import io
import json
import logging
import os
from pathlib import Path
import sys
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr, ValidationError

from app.core.config import (
    DEFAULT_WORKSPACE_ROOT,
    SensitiveDataFilter,
    Settings,
    get_settings,
    setup_logging,
)


# -----------------------------------------------------------------------------
# TC-CFG-01: Default Instantiation & Values
# -----------------------------------------------------------------------------
def test_default_settings_instantiation():
    """Verify that default settings populate all expected defaults correctly."""
    settings = Settings()
    assert settings.app_name == "AnimeRecSys AI"
    assert settings.environment in ("development", "production", "test")
    assert settings.debug is False
    assert settings.host == "127.0.0.1"
    assert settings.port == 8000
    assert settings.log_level == "INFO"

    # Gemini defaults
    assert settings.gemini_model == "gemini-2.5-flash"
    assert settings.gemini_temperature == 0.2
    assert settings.gemini_max_feedback_length == 300
    assert settings.is_gemini_configured is False

    # AniList defaults
    assert settings.anilist_api_url == "https://graphql.anilist.co"
    assert settings.anilist_rate_limit_per_minute == 60
    assert settings.anilist_request_timeout_seconds == 10.0
    assert settings.anilist_retry_after_default == 60.0

    # Hyperparameters
    assert settings.rec_eta == 0.15
    assert settings.rec_alpha == 0.25
    assert settings.rec_gamma == 0.98
    assert settings.rec_lambda_mmr == 0.50

    # Concurrency
    assert settings.filelock_timeout_seconds == 10.0
    assert settings.catalog_cache_ttl_seconds == 3600


# -----------------------------------------------------------------------------
# TC-CFG-02: SecretStr Masking in Representations
# -----------------------------------------------------------------------------
def test_gemini_api_key_secret_masking():
    """Verify that GEMINI_API_KEY is masked in repr, str, and serialization."""
    raw_key = "AIzaSyA_TestKey_1234567890abcdefghijk"
    settings = Settings(GEMINI_API_KEY=raw_key)

    assert settings.gemini_api_key is not None
    assert isinstance(settings.gemini_api_key, SecretStr)

    # __str__ and __repr__ must NEVER leak the raw key
    assert str(settings.gemini_api_key) == "**********"
    assert repr(settings.gemini_api_key) == "SecretStr('**********')"
    assert raw_key not in repr(settings)
    assert raw_key not in str(settings)

    # model_dump() must also mask or not expose raw key
    dumped = settings.model_dump()
    assert dumped["gemini_api_key"] != raw_key

    # Unwrapping via get_secret_value()
    assert settings.gemini_api_key.get_secret_value() == raw_key
    assert settings.is_gemini_configured is True


def test_is_gemini_configured_with_placeholders():
    """Verify is_gemini_configured returns False for absent, empty, or placeholder keys."""
    assert Settings(GEMINI_API_KEY=None).is_gemini_configured is False
    assert Settings(GEMINI_API_KEY="").is_gemini_configured is False
    assert Settings(GEMINI_API_KEY="   ").is_gemini_configured is False
    assert Settings(GEMINI_API_KEY="your_gemini_api_key_here").is_gemini_configured is False


# -----------------------------------------------------------------------------
# TC-CFG-03: CORS Security & Validation
# -----------------------------------------------------------------------------
def test_cors_origins_default():
    """Verify default CORS origins allow only local development hosts."""
    settings = Settings()
    assert settings.cors_origins == ["http://localhost:5173", "http://127.0.0.1:5173"]


def test_cors_origins_wildcard_rejection():
    """Verify that wildcard origins ('*') are strictly rejected per SECURITY.md."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(CORS_ORIGINS=["*"])
    assert "wildcard" in str(exc_info.value).lower()

    with pytest.raises(ValidationError) as exc_info:
        Settings(CORS_ORIGINS=["http://localhost:5173", "*"])
    assert "wildcard" in str(exc_info.value).lower()

    with pytest.raises(ValidationError) as exc_info:
        Settings(CORS_ORIGINS="*")
    assert "wildcard" in str(exc_info.value).lower()

    with pytest.raises(ValidationError) as exc_info:
        Settings(CORS_ORIGINS='["http://localhost:5173", "*"]')
    assert "wildcard" in str(exc_info.value).lower()

    with pytest.raises(ValidationError) as exc_info:
        Settings(CORS_ORIGINS="http://*.example.com")
    assert "wildcard" in str(exc_info.value).lower()


def test_cors_origins_parsing_formats():
    """Verify comma-separated string, JSON list string, and trailing slash removal."""
    # Comma-separated string with spaces and trailing slashes
    s1 = Settings(CORS_ORIGINS="http://localhost:5173/, http://127.0.0.1:5173")
    assert s1.cors_origins == ["http://localhost:5173", "http://127.0.0.1:5173"]

    # JSON array string
    s2 = Settings(CORS_ORIGINS='["http://localhost:3000/", "http://127.0.0.1:3000"]')
    assert s2.cors_origins == ["http://localhost:3000", "http://127.0.0.1:3000"]

    # Empty origins rejection
    with pytest.raises(ValidationError):
        Settings(CORS_ORIGINS=[])


# -----------------------------------------------------------------------------
# TC-CFG-04: Strict Path Resolution
# -----------------------------------------------------------------------------
def test_path_anchors_and_resolution(tmp_path: Path):
    """Verify paths resolve deterministically against workspace root."""
    custom_root = tmp_path / "custom_workspace"
    settings = Settings(WORKSPACE_ROOT=str(custom_root))

    assert settings.workspace_root == custom_root.resolve()
    assert settings.data_dir == (custom_root / "data").resolve()
    assert settings.locks_dir == (custom_root / "data" / ".locks").resolve()
    assert settings.catalog_cache_path == (custom_root / "data" / "cache_catalog.json").resolve()
    assert settings.user_profile_path == (custom_root / "data" / "user_profile.json").resolve()
    assert settings.watch_history_path == (custom_root / "data" / "watch_history.json").resolve()
    assert settings.fixtures_path == (custom_root / ".antigravity" / "FIXTURES.json").resolve()

    # resolve_path with relative path
    resolved_rel = settings.resolve_path("data/test_file.json")
    assert resolved_rel == (custom_root / "data" / "test_file.json").resolve()

    # resolve_path with absolute path
    abs_path = tmp_path / "absolute_file.json"
    assert settings.resolve_path(abs_path) == abs_path.resolve()

    # ensure_directories creates data and locks directories
    settings.ensure_directories()
    assert settings.data_dir.exists()
    assert settings.locks_dir.exists()


# -----------------------------------------------------------------------------
# TC-CFG-05: Hyperparameter Boundary Enforcement
# -----------------------------------------------------------------------------
def test_hyperparameter_boundaries():
    """Verify that recommendation hyperparameters enforce mathematical bounds."""
    # rec_eta bounds [0.01, 1.0]
    with pytest.raises(ValidationError):
        Settings(REC_ETA=0.0)
    with pytest.raises(ValidationError):
        Settings(REC_ETA=1.5)
    assert Settings(REC_ETA=0.20).rec_eta == 0.20

    # rec_gamma bounds [0.50, 0.999]
    with pytest.raises(ValidationError):
        Settings(REC_GAMMA=0.49)
    with pytest.raises(ValidationError):
        Settings(REC_GAMMA=1.0)  # Gamma must be < 1.0 for decay stability
    assert Settings(REC_GAMMA=0.95).rec_gamma == 0.95

    # rec_alpha bounds [0.01, 1.0]
    with pytest.raises(ValidationError):
        Settings(REC_ALPHA=0.0)
    with pytest.raises(ValidationError):
        Settings(REC_ALPHA=1.2)

    # rec_lambda_mmr bounds [0.0, 1.0]
    with pytest.raises(ValidationError):
        Settings(REC_LAMBDA_MMR=-0.1)
    with pytest.raises(ValidationError):
        Settings(REC_LAMBDA_MMR=1.1)


# -----------------------------------------------------------------------------
# TC-CFG-06: SensitiveDataFilter Logging Redaction
# -----------------------------------------------------------------------------
def test_sensitive_data_filter_patterns():
    """Verify SensitiveDataFilter scrubs API keys, Bearer tokens, query params, and headers."""
    secret = "CustomSecretKey998877"
    filter_obj = SensitiveDataFilter(exact_secrets=[secret])

    # 1. Exact secret redaction
    assert filter_obj.redact_text(f"Token is {secret}") == "Token is [REDACTED_SECRET]"

    # 2. Google API key regex (AIza...)
    google_key = "AIzaSyD_TestDummyKey1234567890abcdef"
    assert filter_obj.redact_text(f"Calling Gemini with {google_key}") == "Calling Gemini with [REDACTED_GEMINI_KEY]"

    # 3. Bearer token
    bearer_log = "Header: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    assert "Bearer [REDACTED_TOKEN]" in filter_obj.redact_text(bearer_log)

    # 4. URL query parameters (key=..., apikey=..., secret=...)
    url_log = "GET /v1/models?key=secret_param_value&other=1"
    assert "key=[REDACTED]&other=1" in filter_obj.redact_text(url_log)

    # 5. Header patterns
    header_log = "x-goog-api-key: my-secret-header-value"
    assert "x-goog-api-key: [REDACTED]" in filter_obj.redact_text(header_log)


def test_sensitive_data_filter_record_processing():
    """Verify filter modifies LogRecord msg, args, and exc_text."""
    filter_obj = SensitiveDataFilter(exact_secrets=["SuperSecretPass"])

    # Test msg
    record1 = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="Error connecting with SuperSecretPass", args=(), exc_info=None
    )
    filter_obj.filter(record1)
    assert "[REDACTED_SECRET]" in record1.msg
    assert "SuperSecretPass" not in record1.msg

    # Test args tuple
    record2 = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="User authenticated: %s", args=("SuperSecretPass",), exc_info=None
    )
    filter_obj.filter(record2)
    assert record2.args == ("[REDACTED_SECRET]",)

    # Test args dict (LogRecord extracts single mapping from tuple to self.args)
    record3 = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="Payload: %(secret)s", args=({"secret": "SuperSecretPass"},), exc_info=None
    )
    filter_obj.filter(record3)
    assert record3.args["secret"] == "[REDACTED_SECRET]"

    # Test exc_text
    record4 = logging.LogRecord(
        name="test", level=logging.ERROR, pathname=__file__, lineno=1,
        msg="Traceback occurred", args=(), exc_info=None
    )
    record4.exc_text = "Traceback: Key AIzaSyA1234567890abcdefghijklmnopqrstuv in frame"
    filter_obj.filter(record4)
    assert "[REDACTED_GEMINI_KEY]" in record4.exc_text


# -----------------------------------------------------------------------------
# TC-CFG-07: Singleton Accessor
# -----------------------------------------------------------------------------
def test_get_settings_singleton():
    """Verify get_settings() returns a cached singleton instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


# -----------------------------------------------------------------------------
# TC-CFG-08: Exception Traceback & Secret Scrubbing (Adversarial M1-It2)
# -----------------------------------------------------------------------------
def test_sensitive_data_filter_logger_exception_traceback_scrubbing():
    """Verify logger.exception() redacts exact secrets from tracebacks and stream output."""
    secret = "AIzaSyD_TopSecret1234567890abcdef"
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    filter_obj = SensitiveDataFilter(exact_secrets=[secret])
    handler.addFilter(filter_obj)

    logger = logging.getLogger("test_traceback_scrub")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    try:
        raise ValueError(f"Failed to query model with key: {secret}")
    except Exception:
        logger.exception("An unhandled exception occurred")

    output = stream.getvalue()
    assert secret not in output
    assert "[REDACTED_GEMINI_KEY]" in output
    assert "ValueError: Failed to query model with key: [REDACTED_GEMINI_KEY]" in output


def test_sensitive_data_filter_error_exc_info_true():
    """Verify logger.error(..., exc_info=True) redacts configured exact secrets."""
    secret = "CustomSecretKey998877"
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    filter_obj = SensitiveDataFilter(exact_secrets=[secret])
    handler.addFilter(filter_obj)

    logger = logging.getLogger("test_exc_info_true")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    try:
        raise RuntimeError(f"Database auth failed with password={secret}")
    except Exception:
        logger.error("Database connection dropped", exc_info=True)

    output = stream.getvalue()
    assert secret not in output
    assert "[REDACTED_SECRET]" in output


def test_sensitive_data_filter_exc_info_attribute_and_reformatting_sanitized():
    """Verify record.exc_info[1].args and reformatted tracebacks do not retain secrets."""
    secret = "SuperSecretDbPassword"
    filter_obj = SensitiveDataFilter(exact_secrets=[secret])

    try:
        raise ValueError(f"Connection aborted: {secret}")
    except Exception:
        rec = logging.LogRecord("test", logging.ERROR, "test.py", 1, "Error", (), sys.exc_info())

    filter_obj.filter(rec)

    # 1. Check exc_text
    assert secret not in rec.exc_text
    assert "[REDACTED_SECRET]" in rec.exc_text

    # 2. Check exception instance args
    assert secret not in str(rec.exc_info[1])
    assert "[REDACTED_SECRET]" in str(rec.exc_info[1])

    # 3. Check downstream formatter evaluation of exc_info
    downstream_formatter = logging.Formatter()
    reformatted = downstream_formatter.formatException(rec.exc_info)
    assert secret not in reformatted
    assert "[REDACTED_SECRET]" in reformatted


def test_sensitive_data_filter_non_string_msg_sanitization():
    """Verify logging an exception object directly as msg is sanitized."""
    gemini_key = "AIzaSyA_GeminiApiKey1234567890abcdef"
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    filter_obj = SensitiveDataFilter()
    handler.addFilter(filter_obj)

    logger = logging.getLogger("test_non_string_msg")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Direct exception object logged as msg
    logger.error(ValueError(f"Bad API call: {gemini_key}"))

    output = stream.getvalue()
    assert gemini_key not in output
    assert "[REDACTED_GEMINI_KEY]" in output


def test_sensitive_data_filter_chained_exception_sanitization():
    """Verify chained exceptions (__cause__) have tracebacks and attributes scrubbed."""
    secret = "AIzaSyB_InnerSecretKey1234567890abcde"
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    filter_obj = SensitiveDataFilter()
    handler.addFilter(filter_obj)

    logger = logging.getLogger("test_chained")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    try:
        try:
            raise ValueError(f"Inner service failed: {secret}")
        except Exception as inner_err:
            raise RuntimeError("Outer gateway failed") from inner_err
    except Exception:
        logger.exception("Chained failure caught")

    output = stream.getvalue()
    assert secret not in output
    assert "[REDACTED_GEMINI_KEY]" in output


def test_sensitive_data_filter_stack_info_sanitization():
    """Verify stack_info=True tracebacks are sanitized."""
    secret = "SecretInStackToken123"
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    filter_obj = SensitiveDataFilter(exact_secrets=[secret])
    handler.addFilter(filter_obj)

    logger = logging.getLogger("test_stack_info")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.error(f"Stack checkpoint with {secret}", stack_info=True)

    output = stream.getvalue()
    assert secret not in output
    assert "[REDACTED_SECRET]" in output


# -----------------------------------------------------------------------------
# TC-CFG-09: Milestone 1 Iteration 3 Hardening Tests
# -----------------------------------------------------------------------------
def test_sensitive_data_filter_dict_key_redaction_in_args():
    """Verify dictionary keys in record.args (direct dict and inside tuple/list) are sanitized."""
    gemini_key = "AIzaSyTestKeyInDictKey1234567890123"
    exact_secret = "ExactSecretPasswordInKey99"
    filter_obj = SensitiveDataFilter(exact_secrets=[exact_secret])

    # Direct dict args
    rec1 = logging.LogRecord("test", logging.INFO, "t.py", 1, "Payload: %(k)s", (), None)
    rec1.args = {gemini_key: "val1", exact_secret: "val2"}
    filter_obj.filter(rec1)
    assert gemini_key not in rec1.args
    assert exact_secret not in rec1.args
    assert "[REDACTED_GEMINI_KEY]" in rec1.args
    assert "[REDACTED_SECRET]" in rec1.args

    # Dict inside list/tuple args
    rec2 = logging.LogRecord("test", logging.INFO, "t.py", 1, "Dump: %s", (), None)
    rec2.args = ({gemini_key: "nested_val"},)
    filter_obj.filter(rec2)
    assert gemini_key not in str(rec2.args)
    assert "[REDACTED_GEMINI_KEY]" in str(rec2.args)


def test_sensitive_data_filter_ansi_escape_gemini_key_redaction():
    """Verify Gemini keys preceded by ANSI color escape codes (e.g. \\x1b[31m) are redacted."""
    gemini_key = "AIzaSyAnsiTestKey123456789012345678"
    filter_obj = SensitiveDataFilter()
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.addFilter(filter_obj)

    logger = logging.getLogger("test_ansi_redaction")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.error(f"Status: \x1b[31m{gemini_key}\x1b[0m")
    output = stream.getvalue()
    assert gemini_key not in output
    assert "[REDACTED_GEMINI_KEY]" in output


def test_sensitive_data_filter_extra_dict_sanitization():
    """Verify custom attributes attached via extra={...} on LogRecord are sanitized."""
    secret = "AIzaSyExtraSecretToken123456789012"
    filter_obj = SensitiveDataFilter()
    formatter = logging.Formatter("%(message)s | token=%(token)s | dict_key=%(meta)s")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)
    handler.addFilter(filter_obj)

    logger = logging.getLogger("test_extra_sanitization")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info("Request processed", extra={"token": secret, "meta": {secret: "active"}})
    output = stream.getvalue()
    assert secret not in output
    assert "[REDACTED_GEMINI_KEY]" in output


def test_sensitive_data_filter_exception_group_recursive_sanitization():
    """Verify ExceptionGroup sub-exception attributes are recursively sanitized."""
    secret = "AIzaSySubExcSecretKey1234567890123"
    filter_obj = SensitiveDataFilter()

    child = ValueError(f"Nested failure with key: {secret}")
    eg = ExceptionGroup("Top group", [child])
    rec = logging.LogRecord("test", logging.ERROR, "t.py", 1, "Group err", (), (ExceptionGroup, eg, None))

    filter_obj.filter(rec)

    # Both formatted traceback and sub-exception args must be scrubbed
    assert secret not in (rec.exc_text or "")
    assert secret not in str(rec.exc_info[1].exceptions[0].args)
    assert "[REDACTED_GEMINI_KEY]" in str(rec.exc_info[1].exceptions[0].args)


# -----------------------------------------------------------------------------
# TC-CFG-10: Milestone 1 Iteration 4 Hardening & Regression Prevention
# -----------------------------------------------------------------------------
def test_sensitive_data_filter_modulo_formatting_numeric_and_float():
    """
    Verify modulo formatting with numeric specifiers (%d, %.2f) executes cleanly
    without TypeError and without dropping logs.
    """
    filter_obj = SensitiveDataFilter()
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.addFilter(filter_obj)

    logger = logging.getLogger("test_modulo_numeric")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Standard numeric specifiers
    logger.info("Count: %d, Rate: %.2f", 42, 3.14)

    output = stream.getvalue()
    assert "Count: 42, Rate: 3.14" in output
    assert len(output.strip()) > 0, "Log was dropped!"


def test_sensitive_data_filter_modulo_formatting_hex_and_dict():
    """
    Verify modulo formatting with hex (%x) and dictionary mapping (%(port)d)
    executes cleanly without TypeError.
    """
    filter_obj = SensitiveDataFilter()
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.addFilter(filter_obj)

    logger = logging.getLogger("test_modulo_hex_dict")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Hex specifier
    logger.info("Hex: %x", 255)
    # Mapping specifier
    logger.info("Port: %(port)d", {"port": 8000})

    output = stream.getvalue()
    assert "Hex: ff" in output
    assert "Port: 8000" in output


def test_sensitive_data_filter_modulo_formatting_with_embedded_secrets():
    """
    Verify that secret redaction functions seamlessly alongside numeric specifiers
    without corrupting numeric types or leaking secrets.
    """
    gemini_key = "AIzaSyTestKeyForModuloFormatting1234"
    filter_obj = SensitiveDataFilter()
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.addFilter(filter_obj)

    logger = logging.getLogger("test_modulo_secrets")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Tuple arguments with mixed types
    logger.info("ID: %d, Key: %s, Ratio: %.3f", 101, gemini_key, 0.875)
    # Dict arguments with mixed types
    logger.info("Host: %(host)s:%(port)d, Key: %(key)s", {"host": "localhost", "port": 8080, "key": gemini_key})

    output = stream.getvalue()
    assert "ID: 101, Key: [REDACTED_GEMINI_KEY], Ratio: 0.875" in output
    assert "Host: localhost:8080, Key: [REDACTED_GEMINI_KEY]" in output
    assert gemini_key not in output


def test_sensitive_data_filter_primitive_types_preserved_in_record_args():
    """
    Verify that primitive types (int, float, bool, bytes, None) are strictly preserved
    in record.args rather than being converted to strings.
    """
    gemini_key = "AIzaSyArgTypePreserveKey123456789012"
    filter_obj = SensitiveDataFilter()

    # Tuple args
    rec1 = logging.LogRecord("test", logging.INFO, "t.py", 1, "test", (42, 3.14, True, False, None, b"raw", gemini_key), None)
    filter_obj.filter(rec1)

    assert rec1.args[0] == 42
    assert type(rec1.args[0]) is int
    assert rec1.args[1] == 3.14
    assert type(rec1.args[1]) is float
    assert rec1.args[2] is True
    assert type(rec1.args[2]) is bool
    assert rec1.args[3] is False
    assert type(rec1.args[3]) is bool
    assert rec1.args[4] is None
    assert rec1.args[5] == b"raw"
    assert rec1.args[6] == "[REDACTED_GEMINI_KEY]"

    # Dict args
    rec2 = logging.LogRecord("test", logging.INFO, "t.py", 1, "test", (), None)
    rec2.args = {"port": 8000, "rate": 1.25, "active": True, "token": gemini_key}
    filter_obj.filter(rec2)

    assert rec2.args["port"] == 8000
    assert type(rec2.args["port"]) is int
    assert rec2.args["rate"] == 1.25
    assert type(rec2.args["rate"]) is float
    assert rec2.args["active"] is True
    assert type(rec2.args["active"]) is bool
    assert rec2.args["token"] == "[REDACTED_GEMINI_KEY]"


def test_sensitive_data_filter_simultaneous_cause_and_context_scrubbing():
    """
    Verify that an exception with BOTH __cause__ and __context__ has secrets
    in both __cause__.args and __context__.args scrubbed to [REDACTED_GEMINI_KEY].
    """
    key_ctx = "AIzaSyContextKeySimultaneous1234567"
    key_cause = "AIzaSyCauseKeySimultaneous12345678"
    key_out = "AIzaSyOuterKeySimultaneous12345678"

    e_ctx = ValueError(f"Context error with secret: {key_ctx}")
    e_cause = KeyError(f"Cause error with secret: {key_cause}")
    e_out = RuntimeError(f"Outer error with secret: {key_out}")
    e_out.__context__ = e_ctx
    e_out.__cause__ = e_cause

    filter_obj = SensitiveDataFilter()
    filter_obj._sanitize_exception_attributes(e_out)

    # Verify cause args scrubbed
    assert key_cause not in str(e_cause.args)
    assert "[REDACTED_GEMINI_KEY]" in str(e_cause.args)

    # Verify context args scrubbed (proves non-short-circuiting)
    assert key_ctx not in str(e_ctx.args)
    assert "[REDACTED_GEMINI_KEY]" in str(e_ctx.args)

    # Verify outer args scrubbed
    assert key_out not in str(e_out.args)
    assert "[REDACTED_GEMINI_KEY]" in str(e_out.args)


def test_sensitive_data_filter_chained_cause_and_context_exc_info_and_downstream_format():
    """
    Verify that when an exception with BOTH __cause__ and __context__ is attached
    to record.exc_info, filter() scrubs secrets from both __cause__.args and __context__.args,
    and downstream Formatter.formatException() on either yields redacted text.
    """
    key_ctx = "AIzaSyContextKeyDownstream1234567"
    key_cause = "AIzaSyCauseKeyDownstream12345678"
    key_out = "AIzaSyOuterKeyDownstream12345678"

    e_ctx = ValueError(f"Context error: {key_ctx}")
    e_cause = KeyError(f"Cause error: {key_cause}")
    e_out = RuntimeError(f"Outer error: {key_out}")
    e_out.__context__ = e_ctx
    e_out.__cause__ = e_cause

    filter_obj = SensitiveDataFilter()
    rec = logging.LogRecord(
        "test", logging.ERROR, "test.py", 1, "Chained exc", (), (type(e_out), e_out, None)
    )
    filter_obj.filter(rec)

    # 1. Direct inspection of cause and context args
    assert key_cause not in str(rec.exc_info[1].__cause__.args)
    assert "[REDACTED_GEMINI_KEY]" in str(rec.exc_info[1].__cause__.args)

    assert key_ctx not in str(rec.exc_info[1].__context__.args)
    assert "[REDACTED_GEMINI_KEY]" in str(rec.exc_info[1].__context__.args)

    # 2. Downstream Formatter evaluation on context
    formatter = logging.Formatter()
    formatted_ctx = formatter.formatException(
        (type(rec.exc_info[1].__context__), rec.exc_info[1].__context__, None)
    )
    assert key_ctx not in formatted_ctx
    assert "[REDACTED_GEMINI_KEY]" in formatted_ctx

    # 3. Downstream Formatter evaluation on outer / cause chain
    formatted_chain = formatter.formatException(rec.exc_info)
    assert key_cause not in formatted_chain
    assert key_out not in formatted_chain
    assert "[REDACTED_GEMINI_KEY]" in formatted_chain


def test_sensitive_data_filter_nested_collections_extra_sanitization():
    """
    Verify that nested collections (e.g. dict inside list) passed via extra={...}
    have string secrets scrubbed while retaining native types for numbers and booleans.
    """
    gemini_key = "AIzaSyNestedExtraKey123456789012345"
    exact_secret = "TopSecretCustomToken999888"
    filter_obj = SensitiveDataFilter(exact_secrets=[exact_secret])

    rec = logging.LogRecord("test", logging.INFO, "t.py", 1, "Nested payload check", (), None)
    rec.metadata_list = [
        {
            "api_key": gemini_key,
            "secret": exact_secret,
            "count": 42,
            "rate": 3.14,
            "enabled": True,
            "inner_tags": [gemini_key, "safe_tag", 100],
        }
    ]

    filter_obj.filter(rec)

    item = rec.metadata_list[0]
    # Secrets must be scrubbed
    assert item["api_key"] == "[REDACTED_GEMINI_KEY]"
    assert item["secret"] == "[REDACTED_SECRET]"
    assert item["inner_tags"][0] == "[REDACTED_GEMINI_KEY]"
    assert item["inner_tags"][1] == "safe_tag"

    # Numeric and boolean types must be strictly preserved
    assert item["count"] == 42
    assert type(item["count"]) is int
    assert item["rate"] == 3.14
    assert type(item["rate"]) is float
    assert item["enabled"] is True
    assert type(item["enabled"]) is bool
    assert item["inner_tags"][2] == 100
    assert type(item["inner_tags"][2]) is int


def test_sensitive_data_filter_cyclic_chained_exceptions_safety():
    """
    Verify that cyclic chained exception references (e.g. e1.__cause__ = e2; e2.__context__ = e1)
    terminate safely without RecursionError or infinite loop, while redacting secrets.
    """
    key_e1 = "AIzaSyCyclicKey1111111111111111111"
    key_e2 = "AIzaSyCyclicKey2222222222222222222"

    e1 = ValueError(f"Cycle node 1: {key_e1}")
    e2 = RuntimeError(f"Cycle node 2: {key_e2}")
    e1.__cause__ = e2
    e2.__context__ = e1  # Mutual cyclic reference

    filter_obj = SensitiveDataFilter()
    # Must not raise RecursionError
    filter_obj._sanitize_exception_attributes(e1)

    assert key_e1 not in str(e1.args)
    assert "[REDACTED_GEMINI_KEY]" in str(e1.args)
    assert key_e2 not in str(e2.args)
    assert "[REDACTED_GEMINI_KEY]" in str(e2.args)



