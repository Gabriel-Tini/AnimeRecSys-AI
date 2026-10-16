"""
tests/test_challenger_logging_m1_it2.py
Empirical Challenger Adversarial Stress Test Suite for Logging Secret Scrubbing Filter.
Milestone 1 Iteration 2 - AnimeRecSys AI
"""

from __future__ import annotations

import io
import logging
import sys
from typing import Any, List

import pytest
from pydantic import SecretStr

from backend.app.core.config import SensitiveDataFilter, Settings, setup_logging


# =============================================================================
# Helper Utilities
# =============================================================================
def make_logger(
    name: str,
    filter_obj: SensitiveDataFilter,
    formatter: logging.Formatter | None = None,
) -> tuple[logging.Logger, io.StringIO, logging.StreamHandler]:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    fmt = formatter or logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(fmt)
    handler.addFilter(filter_obj)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    return logger, stream, handler


# =============================================================================
# 1. Chained Exceptions (__cause__, __context__)
# =============================================================================
class TestChainedExceptionsScrubbing:
    """Stress tests on chained exceptions with explicit cause and implicit context."""

    def test_explicit_cause_chain_scrubbing(self):
        """Verify explicit __cause__ chain (raise ... from ...) scrubs secrets from all frames."""
        sec_inner = "AIzaSyInnerCauseKey1234567890123456"
        sec_outer = "AIzaSyOuterCauseKey1234567890123456"
        filter_obj = SensitiveDataFilter()
        logger, stream, _ = make_logger("test_chain_cause", filter_obj)

        try:
            try:
                raise ValueError(f"Inner failure: {sec_inner}")
            except Exception as e_inner:
                raise RuntimeError(f"Outer gateway failure: {sec_outer}") from e_inner
        except Exception:
            logger.exception("Chained failure occurred")

        output = stream.getvalue()
        assert sec_inner not in output, f"Inner secret leaked in traceback output: {output}"
        assert sec_outer not in output, f"Outer secret leaked in traceback output: {output}"
        assert output.count("[REDACTED_GEMINI_KEY]") >= 2

    def test_implicit_context_chain_scrubbing(self):
        """Verify implicit __context__ chain (raise without from inside except) scrubs secrets."""
        sec_ctx = "AIzaSyContextKey1234567890123456789"
        sec_active = "AIzaSyActiveKey12345678901234567890"
        filter_obj = SensitiveDataFilter()
        logger, stream, _ = make_logger("test_chain_context", filter_obj)

        try:
            try:
                raise KeyError(f"Missing config param: {sec_ctx}")
            except Exception:
                raise ConnectionError(f"Failed to connect using token: {sec_active}")
        except Exception:
            logger.exception("Context chain caught")

        output = stream.getvalue()
        assert sec_ctx not in output, f"Context secret leaked: {output}"
        assert sec_active not in output, f"Active secret leaked: {output}"
        assert output.count("[REDACTED_GEMINI_KEY]") >= 2

    def test_multi_tier_deep_chain_scrubbing(self):
        """Verify 4-tier chained exception mixing __cause__ and __context__ scrubs all secrets."""
        secrets = [f"AIzaSyTier{i}Key123456789012345678901" for i in range(4)]
        filter_obj = SensitiveDataFilter()
        logger, stream, _ = make_logger("test_multi_chain", filter_obj)

        try:
            try:
                try:
                    try:
                        raise ValueError(f"Tier 0 failure: {secrets[0]}")
                    except Exception as e0:
                        raise KeyError(f"Tier 1 lookup failed: {secrets[1]}") from e0
                except Exception as e1:
                    # implicit context
                    raise TimeoutError(f"Tier 2 timeout: {secrets[2]}")
            except Exception as e2:
                raise RuntimeError(f"Tier 3 top-level: {secrets[3]}") from e2
        except Exception:
            logger.exception("Deep chain caught")

        output = stream.getvalue()
        for i, s in enumerate(secrets):
            assert s not in output, f"Tier {i} secret leaked in output: {output}"

    def test_suppressed_context_from_none(self):
        """Verify that raise ... from None properly hides context and scrubs active secret."""
        sec_suppressed = "AIzaSySuppressedKey1234567890123456"
        sec_active = "AIzaSyActiveFromNone1234567890123456"
        filter_obj = SensitiveDataFilter()
        logger, stream, _ = make_logger("test_from_none", filter_obj)

        try:
            try:
                raise ValueError(f"Secret to suppress: {sec_suppressed}")
            except Exception:
                raise RuntimeError(f"Active sanitized: {sec_active}") from None
        except Exception:
            logger.exception("Caught from None")

        output = stream.getvalue()
        assert sec_suppressed not in output
        assert sec_active not in output
        assert "[REDACTED_GEMINI_KEY]" in output


# =============================================================================
# 2. Nested Exception Groups (Python 3.11+)
# =============================================================================
class TestExceptionGroupsScrubbing:
    """Stress tests on Python 3.11+ ExceptionGroup and BaseExceptionGroup."""

    def test_single_level_exception_group_stream_scrubbing(self):
        """Verify ExceptionGroup with multiple children scrubs all secrets in output stream."""
        sec_gemini = "AIzaSyGroupChildKey1234567890123456"
        sec_exact = "ExactSecretTokenForGroupTesting99"
        filter_obj = SensitiveDataFilter(exact_secrets=[sec_exact])
        logger, stream, _ = make_logger("test_eg_single", filter_obj)

        try:
            children = [
                ValueError(f"Child 1: {sec_gemini}"),
                TypeError(f"Child 2: Bearer {sec_exact}"),
                RuntimeError("Child 3: key=secret_val_12345"),
            ]
            raise ExceptionGroup("Batch tasks failed", children)
        except ExceptionGroup:
            logger.exception("ExceptionGroup caught")

        output = stream.getvalue()
        assert sec_gemini not in output, f"Gemini key leaked in EG stream: {output}"
        assert sec_exact not in output, f"Exact secret leaked in EG stream: {output}"
        assert "secret_val_12345" not in output
        assert "[REDACTED_GEMINI_KEY]" in output
        assert "[REDACTED_SECRET]" in output or "[REDACTED_TOKEN]" in output
        assert "[REDACTED]" in output

    def test_nested_exception_group_stream_scrubbing(self):
        """Verify deeply nested ExceptionGroups (tree hierarchy) scrub all secrets in stream."""
        sec_leaf1 = "AIzaSyLeafKey1_12345678901234567890"
        sec_leaf2 = "AIzaSyLeafKey2_12345678901234567890"
        sec_leaf3 = "AIzaSyLeafKey3_12345678901234567890"
        filter_obj = SensitiveDataFilter()
        logger, stream, _ = make_logger("test_eg_nested", filter_obj)

        try:
            inner_group1 = ExceptionGroup(
                "SubGroup 1",
                [ValueError(f"Leaf 1: {sec_leaf1}"), KeyError(f"Leaf 2: {sec_leaf2}")],
            )
            inner_group2 = ExceptionGroup(
                "SubGroup 2",
                [RuntimeError(f"Leaf 3: {sec_leaf3}")],
            )
            raise ExceptionGroup("Root Group", [inner_group1, inner_group2])
        except ExceptionGroup:
            logger.exception("Root ExceptionGroup caught")

        output = stream.getvalue()
        assert sec_leaf1 not in output, f"Leaf 1 leaked: {output}"
        assert sec_leaf2 not in output, f"Leaf 2 leaked: {output}"
        assert sec_leaf3 not in output, f"Leaf 3 leaked: {output}"

    def test_exception_group_direct_message_logging(self):
        """Verify logger.error(eg) where msg is an ExceptionGroup instance is sanitized."""
        sec = "AIzaSyDirectEGMsgKey123456789012345"
        filter_obj = SensitiveDataFilter()
        logger, stream, _ = make_logger("test_eg_msg", filter_obj)

        eg = ExceptionGroup(f"Group failed with {sec}", [ValueError("err")])
        logger.error(eg)

        output = stream.getvalue()
        assert sec not in output, f"Secret leaked in logger.error(eg): {output}"
        assert "[REDACTED_GEMINI_KEY]" in output

    def test_exception_group_sub_exception_attribute_sanitization(self):
        """
        Adversarial test on structured / downstream logging:
        Verifies whether sub-exception attributes within an ExceptionGroup are sanitized,
        preventing leakage if downstream systems inspect exc_info[1].exceptions or reformat it.
        """
        sec = "AIzaSySubExcAttributeKey12345678901"
        filter_obj = SensitiveDataFilter()

        child = ValueError(f"Unraised sub-exception: {sec}")
        eg = ExceptionGroup("Direct group", [child])
        rec = logging.LogRecord("test", logging.ERROR, "t.py", 1, "Err", (), (ExceptionGroup, eg, None))

        filter_obj.filter(rec)

        # 1. Check formatted exc_text
        assert sec not in (rec.exc_text or "")

        # 2. Check child args for structured loggers (e.g. Sentry/OpenTelemetry)
        # BUG CHECK: _sanitize_exception_attributes does not traverse BaseExceptionGroup.exceptions!
        child_args_str = str(rec.exc_info[1].exceptions[0].args)
        assert sec not in child_args_str, (
            f"VULNERABILITY DETECTED: ExceptionGroup sub-exception args were NOT sanitized! "
            f"Child args: {child_args_str}"
        )

        # 3. Check downstream formatter reformatting
        formatter = logging.Formatter()
        reformatted = formatter.formatException(rec.exc_info)
        assert sec not in reformatted, (
            f"VULNERABILITY DETECTED: Downstream formatException on ExceptionGroup disclosed secret! "
            f"Output: {reformatted}"
        )


# =============================================================================
# 3. Non-String Messages & Args Scrubbing
# =============================================================================
class TestNonStringMessagesScrubbing:
    """Stress tests on arbitrary non-string message types and record.args structures."""

    def test_exception_object_direct_msg(self):
        """Verify logger.error(exc) cleans secret when message is an Exception object."""
        sec = "AIzaSyExcDirectMsgKey12345678901234"
        filter_obj = SensitiveDataFilter()
        logger, stream, _ = make_logger("test_nonstr_exc", filter_obj)

        logger.error(ValueError(f"Direct error object with key {sec}"))

        output = stream.getvalue()
        assert sec not in output
        assert "[REDACTED_GEMINI_KEY]" in output

    def test_dict_direct_msg(self):
        """Verify logging a dict directly as msg is converted and redacted."""
        sec = "AIzaSyDictDirectMsgKey1234567890123"
        filter_obj = SensitiveDataFilter()
        logger, stream, _ = make_logger("test_nonstr_dict", filter_obj)

        logger.info({"status": "failed", "api_key": sec})

        output = stream.getvalue()
        assert sec not in output
        assert "[REDACTED_GEMINI_KEY]" in output

    def test_list_tuple_direct_msg(self):
        """Verify logging list or tuple directly as msg is converted and redacted."""
        sec = "AIzaSyListDirectMsgKey1234567890123"
        filter_obj = SensitiveDataFilter()
        logger, stream, _ = make_logger("test_nonstr_list", filter_obj)

        logger.info([sec, {"nested": sec}])

        output = stream.getvalue()
        assert sec not in output

    def test_custom_object_with_str_and_repr(self):
        """Verify custom objects with secrets in __str__ or __repr__ are scrubbed."""
        sec = "AIzaSyCustomObjKey12345678901234567"
        filter_obj = SensitiveDataFilter()
        logger, stream, _ = make_logger("test_custom_obj", filter_obj)

        class Payload:
            def __str__(self):
                return f"Payload(str={sec})"

            def __repr__(self):
                return f"Payload(repr={sec})"

        # Tested as direct msg
        logger.info(Payload())
        # Tested in args with %s
        logger.info("Using percent s: %s", Payload())
        # Tested in args with %r
        logger.info("Using percent r: %r", Payload())

        output = stream.getvalue()
        assert sec not in output
        assert "[REDACTED_GEMINI_KEY]" in output

    def test_primitive_types_no_crash(self):
        """Verify filter handles None, int, float, bool without crashing."""
        filter_obj = SensitiveDataFilter()
        logger, stream, _ = make_logger("test_primitives", filter_obj)

        logger.info(None)
        logger.info(12345)
        logger.info(3.14159)
        logger.info(True)
        logger.info("Values: %s, %s, %s", None, 0, False)

        output = stream.getvalue()
        assert "None" in output
        assert "12345" in output

    def test_dict_key_in_record_args(self):
        """
        Adversarial test: Verify whether secrets residing in dictionary KEYS inside record.args
        are scrubbed or leaked to log streams.
        """
        sec = "AIzaSySecretInDictKey12345678901234"
        filter_obj = SensitiveDataFilter()
        logger, stream, _ = make_logger("test_dict_key_adv", filter_obj)

        # Passing a dictionary where the secret is in the dictionary key
        logger.info("Dumping dict: %s", {sec: "secret_value"})

        output = stream.getvalue()
        assert sec not in output, (
            f"VULNERABILITY DETECTED: Secret in dictionary KEY inside record.args leaked to stream! "
            f"Output: {output}"
        )


# =============================================================================
# 4. Custom Exception Classes
# =============================================================================
class TestCustomExceptionClassesScrubbing:
    """Stress tests on custom exception classes, overrides, and built-in error attributes."""

    def test_custom_exception_str_override(self):
        """Verify custom exception with __str__ containing secret is scrubbed in traceback."""
        sec = "AIzaSyCustomStrKey12345678901234567"
        filter_obj = SensitiveDataFilter()
        logger, stream, _ = make_logger("test_custom_str_exc", filter_obj)

        class CustomServiceError(Exception):
            def __init__(self, key: str):
                self.key = key
                super().__init__("Base message")

            def __str__(self):
                return f"CustomServiceError(secret_key={self.key})"

        try:
            raise CustomServiceError(sec)
        except CustomServiceError:
            logger.exception("CustomServiceError caught")

        output = stream.getvalue()
        assert sec not in output
        assert "[REDACTED_GEMINI_KEY]" in output or "[REDACTED]" in output

    def test_custom_exception_repr_override(self):
        """Verify custom exception with __repr__ containing secret is scrubbed."""
        sec = "AIzaSyCustomReprKey1234567890123456"
        filter_obj = SensitiveDataFilter()
        logger, stream, _ = make_logger("test_custom_repr_exc", filter_obj)

        class CustomReprError(Exception):
            def __init__(self, key: str):
                self.key = key

            def __repr__(self):
                return f"CustomReprError({self.key})"

        logger.error("Logged with %r: %r", "arg", CustomReprError(sec))

        output = stream.getvalue()
        assert sec not in output

    def test_os_error_strerror_filename_scrubbing(self):
        """Verify OSError with secrets in strerror and filename are scrubbed in exc_info[1]."""
        sec = "AIzaSyOsErrorKey123456789012345678"
        filter_obj = SensitiveDataFilter()

        err = OSError(2, f"No such file with key {sec}", f"/tmp/{sec}.txt")
        rec = logging.LogRecord("test", logging.ERROR, "t.py", 1, "Err", (), (OSError, err, None))

        filter_obj.filter(rec)

        assert sec not in (rec.exc_text or "")
        assert sec not in rec.exc_info[1].strerror
        assert sec not in rec.exc_info[1].filename
        assert "[REDACTED_GEMINI_KEY]" in rec.exc_info[1].strerror
        assert "[REDACTED_GEMINI_KEY]" in rec.exc_info[1].filename


# =============================================================================
# 5. Large Stack Traces & Extreme Payloads
# =============================================================================
class TestLargeStackTracesAndStress:
    """Stress tests on deep call stacks, huge multiline payloads, and secret volume."""

    def test_deep_recursion_stack_trace_250_frames(self):
        """Verify 250-frame recursive traceback scrubs secret without crashing."""
        sec = "AIzaSyDeepRecursionSecretKey1234567"
        filter_obj = SensitiveDataFilter()
        logger, stream, _ = make_logger("test_recursion_deep", filter_obj)

        def recurse(depth: int) -> None:
            if depth <= 0:
                raise RuntimeError(f"Deepest frame reached with secret: {sec}")
            recurse(depth - 1)

        try:
            recurse(250)
        except RuntimeError:
            logger.exception("Deep recursion failed")

        output = stream.getvalue()
        assert sec not in output
        assert "[REDACTED_GEMINI_KEY]" in output
        assert "RuntimeError" in output

    def test_very_large_multiline_exception_message(self):
        """Verify multiline exception with 500 lines and secrets on every line is scrubbed."""
        sec = "AIzaSyMultiLineSecretKey12345678901"
        filter_obj = SensitiveDataFilter()
        logger, stream, _ = make_logger("test_multiline", filter_obj)

        lines = [f"Trace dump line {i}: key={sec} status=FAIL" for i in range(500)]
        big_msg = "\n".join(lines)

        try:
            raise ValueError(big_msg)
        except ValueError:
            logger.exception("Massive exception dumped")

        output = stream.getvalue()
        assert sec not in output, f"Secret leaked in multiline message: {output[:500]}"
        assert "key=[REDACTED]" in output

    def test_100_distinct_secrets_batch_scrubbing(self):
        """Verify filter with 100 configured exact secrets redacts all of them correctly."""
        exact_secrets = [f"ExactSecretNumber_{i:03d}_KeyXYZ9988" for i in range(100)]
        filter_obj = SensitiveDataFilter(exact_secrets=exact_secrets)
        logger, stream, _ = make_logger("test_100_secrets", filter_obj)

        # Log records referencing all 100 secrets
        for i, s in enumerate(exact_secrets):
            logger.warning(f"Warning {i}: Secret leaked in msg: {s}")

        output = stream.getvalue()
        for i, s in enumerate(exact_secrets):
            assert s not in output, f"Secret {i} ({s}) leaked!"
        assert output.count("[REDACTED_SECRET]") == 100

    def test_unicode_null_bytes_and_surrogates_with_secrets(self):
        """Verify payloads with null bytes, emojis, and non-ASCII chars do not bypass filter."""
        sec = "AIzaSyUnicodeSecretKey1234567890123"
        filter_obj = SensitiveDataFilter()
        logger, stream, _ = make_logger("test_unicode", filter_obj)

        adversarial_payloads = [
            f"Zero width:\u200b{sec}\u200b",
            f"Null byte:\x00{sec}\x00",
            f"Emoji string: 💥🚀🔑 {sec} 🛡️",
            f"Japanese text: 認証エラー {sec} トークン無効",
        ]

        for p in adversarial_payloads:
            logger.error("Payload test: %s", p)

        output = stream.getvalue()
        assert sec not in output, f"Secret leaked with unicode payload! Output: {output}"

    def test_ansi_color_escape_bypass(self):
        """
        Adversarial test: ANSI escape codes (e.g. \\x1b[31m) ending in alphanumeric 'm'
        break word boundaries (\\b) in GEMINI_KEY_PATTERN regex.
        """
        sec = "AIzaSyAnsiColorKey1234567890123456"
        filter_obj = SensitiveDataFilter()
        logger, stream, _ = make_logger("test_ansi_bypass", filter_obj)

        # In colored terminal loggers, ANSI red code precedes the key: \x1b[31mAIzaSy...
        logger.error(f"Colored error: \x1b[31m{sec}\x1b[0m")

        output = stream.getvalue()
        assert sec not in output, (
            f"VULNERABILITY DETECTED: ANSI color code preceding key defeated regex word boundary \\b! "
            f"Output: {output}"
        )


# =============================================================================
# 6. End-to-End Streams & Loggers
# =============================================================================
class TestEndToEndStreamsAndLoggers:
    """Stress tests on setup_logging configuration, log propagation, and extra dicts."""

    def test_setup_logging_stream_redaction(self, monkeypatch):
        """Verify setup_logging attaches filter to root and custom loggers."""
        sec = "AIzaSySetupLoggingKey12345678901234"
        monkeypatch.setenv("GEMINI_API_KEY", sec)
        settings = Settings(GEMINI_API_KEY=SecretStr(sec))
        setup_logging(settings)

        root = logging.getLogger()
        stream = io.StringIO()
        test_handler = logging.StreamHandler(stream)
        test_handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(test_handler)

        try:
            logger = logging.getLogger("animerecsys.custom")
            logger.error(f"Root test with {sec}")
            output = stream.getvalue()
            assert sec not in output
            assert "[REDACTED_SECRET]" in output or "[REDACTED_GEMINI_KEY]" in output
        finally:
            root.removeHandler(test_handler)

    def test_stack_info_redaction(self):
        """Verify stack_info=True produces sanitized stack traces."""
        sec = "AIzaSyStackInfoKey12345678901234567"
        filter_obj = SensitiveDataFilter()
        logger, stream, _ = make_logger("test_stack_info_adv", filter_obj)

        logger.info(f"Log with stack and key: {sec}", stack_info=True)

        output = stream.getvalue()
        assert sec not in output
        assert "[REDACTED_GEMINI_KEY]" in output

    def test_extra_dict_secrets_redaction(self):
        """
        Adversarial test: Verify whether secrets passed via logger.info(..., extra={...})
        are sanitized when logged through formatters that interpolate extra attributes.
        """
        sec = "AIzaSySecretExtraToken1234567890123"
        filter_obj = SensitiveDataFilter()
        formatter = logging.Formatter("%(message)s | extra_token=%(token)s")
        logger, stream, _ = make_logger("test_extra_adv", filter_obj, formatter=formatter)

        logger.info("User request processed", extra={"token": sec})

        output = stream.getvalue()
        assert sec not in output, (
            f"VULNERABILITY DETECTED: Secret passed via extra dict leaked to stream! "
            f"Output: {output}"
        )
