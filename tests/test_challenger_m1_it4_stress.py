"""
tests/test_challenger_m1_it4_stress.py
Empirical Challenger Stress & Adversarial Test Suite for Milestone 1 Iteration 4.
Validates:
1. Printf formatting (%d, %i, %u, %f, %.2f, %x, %o, %(port)d) across diverse primitive types.
2. Complex mixed payloads ensuring numeric formatting and secret scrubbing.
3. Chained exceptions (__cause__, __context__, cycles of 2 & 3, self-cycles, deep chains > 1000 nodes).
4. Nested collections in extra={...} (dicts in lists, lists in dicts, circular references, diamond DAGs, unprintable objects).
"""

from __future__ import annotations

import io
import logging
import sys
from typing import Any, List

import pytest
from pydantic import SecretStr

from backend.app.core.config import SensitiveDataFilter, Settings, setup_logging


def make_logger(
    name: str,
    filter_obj: SensitiveDataFilter,
    formatter: logging.Formatter | None = None,
) -> tuple[logging.Logger, io.StringIO, logging.StreamHandler]:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    fmt = formatter or logging.Formatter("%(message)s")
    handler.setFormatter(fmt)
    handler.addFilter(filter_obj)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    return logger, stream, handler


# =============================================================================
# 1. Printf Formatting Specifiers Stress Tests
# =============================================================================
class TestPrintfFormattingStress:
    """Empirical verification of standard printf format specifiers."""

    def test_printf_integer_specifiers_d_i_u(self):
        """Verify %d, %i, and %u with positive, negative, zero, booleans, and large ints."""
        filter_obj = SensitiveDataFilter()
        logger, stream, _ = make_logger("test_printf_int", filter_obj)

        # %d with positive, zero, negative, large int, and booleans
        logger.info("Pos: %d, Zero: %d, Neg: %d", 42, 0, -100)
        logger.info("BoolTrue: %d, BoolFalse: %d", True, False)
        large_int = 2**62 - 1
        logger.info("LargeInt: %d", large_int)

        # %i with positive, zero, negative
        logger.info("SpecI: %i, Zero: %i, Neg: %i", 99, 0, -7)

        # %u (unsigned in C, integer format in Python)
        logger.info("SpecU: %u, NegU: %u", 1234, -1)

        output = stream.getvalue().splitlines()
        assert output[0] == "Pos: 42, Zero: 0, Neg: -100"
        assert output[1] == "BoolTrue: 1, BoolFalse: 0"
        assert output[2] == f"LargeInt: {large_int}"
        assert output[3] == "SpecI: 99, Zero: 0, Neg: -7"
        assert output[4] == "SpecU: 1234, NegU: -1"

    def test_printf_float_specifiers_f_and_precision(self):
        """Verify %f and precision specifiers (%.2f, %.0f, %.4f) with floats and ints."""
        filter_obj = SensitiveDataFilter()
        logger, stream, _ = make_logger("test_printf_float", filter_obj)

        logger.info("StandardFloat: %f", 3.14159)
        logger.info("Precision2: %.2f", 19.999)
        logger.info("Precision0: %.0f", 42.7)
        logger.info("NegFloat: %.3f", -0.005)
        logger.info("ZeroFloat: %.2f", 0.0)
        logger.info("IntAsFloat: %.2f", 50)

        output = stream.getvalue().splitlines()
        assert output[0].startswith("StandardFloat: 3.14159")
        assert output[1] == "Precision2: 20.00"
        assert output[2] == "Precision0: 43"
        assert output[3] == "NegFloat: -0.005"
        assert output[4] == "ZeroFloat: 0.00"
        assert output[5] == "IntAsFloat: 50.00"

    def test_printf_hex_and_octal_specifiers(self):
        """Verify %x, %X, %o, %#x, %#o with various integers."""
        filter_obj = SensitiveDataFilter()
        logger, stream, _ = make_logger("test_printf_hex_oct", filter_obj)

        logger.info("HexLower: %x, HexUpper: %X", 255, 255)
        logger.info("HexZero: %x, HexPrefix: %#x", 0, 0xCAFE)
        logger.info("Octal: %o, OctalPrefix: %#o", 64, 64)

        output = stream.getvalue().splitlines()
        assert output[0] == "HexLower: ff, HexUpper: FF"
        assert output[1] == "HexZero: 0, HexPrefix: 0xcafe"
        assert output[2] == "Octal: 100, OctalPrefix: 0o100"

    def test_printf_mapping_dict_port_d(self):
        """Verify dictionary mapping formatting %(port)d with various types and numbers."""
        filter_obj = SensitiveDataFilter()
        logger, stream, _ = make_logger("test_printf_dict_mapping", filter_obj)

        logger.info("Port: %(port)d, Host: %(host)s", {"port": 8000, "host": "127.0.0.1"})
        logger.info("ZeroPort: %(port)d, NegPort: %(neg)d", {"port": 0, "neg": -80})
        logger.info("HexVal: %(h)x, Rate: %(r).1f", {"h": 16, "r": 2.5})

        output = stream.getvalue().splitlines()
        assert output[0] == "Port: 8000, Host: 127.0.0.1"
        assert output[1] == "ZeroPort: 0, NegPort: -80"
        assert output[2] == "HexVal: 10, Rate: 2.5"

    def test_printf_bytes_argument(self):
        """Verify bytes argument preservation with %s and %r."""
        filter_obj = SensitiveDataFilter()
        logger, stream, _ = make_logger("test_printf_bytes", filter_obj)

        logger.info("RawBytes: %s, ReprBytes: %r", b"hello", b"world")

        output = stream.getvalue()
        assert "RawBytes: b'hello'" in output
        assert "ReprBytes: b'world'" in output

    def test_no_logs_dropped_on_all_printf_specifiers(self, capsys):
        """
        Verify that a suite of 50 consecutive varied printf format calls succeed
        without any dropped logs and zero warnings on sys.stderr.
        """
        filter_obj = SensitiveDataFilter()
        logger, stream, _ = make_logger("test_no_dropped_logs", filter_obj)

        specifiers = [
            ("Count: %d", 1),
            ("Int: %i", -2),
            ("Unsigned: %u", 3),
            ("Float: %.2f", 4.56),
            ("Hex: %x", 0xABC),
            ("Octal: %o", 0o77),
            ("Dict: %(port)d", {"port": 9000}),
            ("Bool: %d", True),
            ("Zero: %d", 0),
            ("Bytes: %s", b"raw"),
        ]

        for i in range(5):
            for fmt, val in specifiers:
                logger.info(fmt, val)

        lines = [line for line in stream.getvalue().splitlines() if line.strip()]
        assert len(lines) == 50, f"Expected 50 log lines, but got {len(lines)}. Logs were dropped!"

        captured = capsys.readouterr()
        assert "TypeError" not in captured.err, f"TypeError emitted to stderr: {captured.err}"


# =============================================================================
# 2. Complex Mixed Payloads & Redaction
# =============================================================================
class TestMixedPayloadsAndRedaction:
    """Stress tests for mixed payloads containing numbers, format specifiers, and secrets."""

    def test_stats_mixed_format_with_gemini_key(self):
        """
        Target verification:
        logger.info('Stats: %d items, cost: $%.2f, key: %s', 100, 19.99, gemini_key)
        Verify numbers format properly and key is redacted to [REDACTED_GEMINI_KEY].
        """
        gemini_key = "AIzaSyMixedStatsKey123456789012345"
        filter_obj = SensitiveDataFilter()
        logger, stream, _ = make_logger("test_mixed_stats", filter_obj)

        logger.info("Stats: %d items, cost: $%.2f, key: %s", 100, 19.99, gemini_key)

        output = stream.getvalue().strip()
        expected = "Stats: 100 items, cost: $19.99, key: [REDACTED_GEMINI_KEY]"
        assert output == expected
        assert gemini_key not in output

    def test_complex_mapping_dict_with_mixed_secrets_and_numbers(self):
        """Verify dictionary mapping with multiple numbers and regex / exact secrets."""
        gemini_key = "AIzaSyDictMixedKey1234567890123456"
        exact_sec = "DatabaseSecretPassword123"
        filter_obj = SensitiveDataFilter(exact_secrets=[exact_sec])
        logger, stream, _ = make_logger("test_dict_mixed", filter_obj)

        logger.info(
            "Worker %(worker_id)d running on port %(port)d: status=%(status)s, "
            "cost=$%(cost).2f, gemini=%(gemini)s, db=%(db)s",
            {
                "worker_id": 7,
                "port": 5432,
                "status": "active",
                "cost": 12.345,
                "gemini": gemini_key,
                "db": exact_sec,
            },
        )

        output = stream.getvalue().strip()
        assert "Worker 7 running on port 5432: status=active, cost=$12.35" in output
        assert "gemini=[REDACTED_GEMINI_KEY]" in output
        assert "db=[REDACTED_SECRET]" in output
        assert gemini_key not in output
        assert exact_sec not in output

    def test_multiple_keys_in_mixed_tuple_and_nested_args(self):
        """Verify multiple distinct secrets in mixed tuple arguments are all redacted."""
        k1 = "AIzaSyFirstKey123456789012345678901"
        k2 = "AIzaSySecondKey12345678901234567890"
        filter_obj = SensitiveDataFilter()
        logger, stream, _ = make_logger("test_multi_keys_mixed", filter_obj)

        logger.info(
            "Count: %d, Key1: %s, Percent: %.1f%%, Key2: %s, Flag: %d",
            42,
            k1,
            99.9,
            k2,
            False,
        )

        output = stream.getvalue().strip()
        assert "Count: 42" in output
        assert "Percent: 99.9%" in output
        assert "Flag: 0" in output
        assert output.count("[REDACTED_GEMINI_KEY]") == 2
        assert k1 not in output
        assert k2 not in output


# =============================================================================
# 3. Chained Exceptions & Graph Traversal Stress Tests
# =============================================================================
class TestChainedExceptionsAdversarial:
    """Stress tests for exception chains, circular references, and deep recursion trees."""

    def test_simultaneous_cause_and_context_scrubbed_with_formatting(self):
        """
        Verify exception with BOTH __cause__ and __context__ set has secrets
        in all 3 exception frames scrubbed and downstream formatting redacted.
        """
        k_outer = "AIzaSyOuterSimulKey123456789012345"
        k_cause = "AIzaSyCauseSimulKey123456789012345"
        k_ctx = "AIzaSyContextSimulKey1234567890123"

        e_ctx = ValueError(f"Context exc: {k_ctx}")
        e_cause = KeyError(f"Cause exc: {k_cause}")
        e_outer = RuntimeError(f"Outer exc: {k_outer}")
        e_outer.__cause__ = e_cause
        e_outer.__context__ = e_ctx

        filter_obj = SensitiveDataFilter()
        rec = logging.LogRecord(
            "test_exc", logging.ERROR, "t.py", 1, "Simultaneous failure", (), (RuntimeError, e_outer, None)
        )
        filter_obj.filter(rec)

        # 1. Check raw args
        assert k_outer not in str(e_outer.args)
        assert "[REDACTED_GEMINI_KEY]" in str(e_outer.args)
        assert k_cause not in str(e_cause.args)
        assert "[REDACTED_GEMINI_KEY]" in str(e_cause.args)
        assert k_ctx not in str(e_ctx.args)
        assert "[REDACTED_GEMINI_KEY]" in str(e_ctx.args)

        # 2. Check record.exc_text
        assert k_outer not in rec.exc_text
        assert k_cause not in rec.exc_text
        assert k_ctx not in rec.exc_text

        # 3. Downstream formatter on exc_info
        formatter = logging.Formatter()
        reformatted = formatter.formatException(rec.exc_info)
        assert k_outer not in reformatted
        assert k_cause not in reformatted
        assert k_ctx not in reformatted

    def test_cycle_of_2_exceptions(self):
        """Verify circular reference e1.__cause__ = e2; e2.__cause__ = e1 terminates cleanly."""
        k1 = "AIzaSyCycle2Node1Key12345678901234"
        k2 = "AIzaSyCycle2Node2Key12345678901234"

        e1 = ValueError(f"Node 1: {k1}")
        e2 = RuntimeError(f"Node 2: {k2}")
        e1.__cause__ = e2
        e2.__cause__ = e1

        filter_obj = SensitiveDataFilter()
        # Must not raise RecursionError
        filter_obj._sanitize_exception_attributes(e1)

        assert k1 not in str(e1.args)
        assert "[REDACTED_GEMINI_KEY]" in str(e1.args)
        assert k2 not in str(e2.args)
        assert "[REDACTED_GEMINI_KEY]" in str(e2.args)

    def test_cycle_of_3_exceptions(self):
        """Verify cycle of 3 exceptions: e1 -> e2 -> e3 -> e1 terminates cleanly."""
        k1 = "AIzaSyCycle3Node1Key12345678901234"
        k2 = "AIzaSyCycle3Node2Key12345678901234"
        k3 = "AIzaSyCycle3Node3Key12345678901234"

        e1 = ValueError(f"Cycle 3 N1: {k1}")
        e2 = TypeError(f"Cycle 3 N2: {k2}")
        e3 = KeyError(f"Cycle 3 N3: {k3}")
        e1.__cause__ = e2
        e2.__cause__ = e3
        e3.__cause__ = e1

        filter_obj = SensitiveDataFilter()
        filter_obj._sanitize_exception_attributes(e1)

        assert k1 not in str(e1.args) and "[REDACTED_GEMINI_KEY]" in str(e1.args)
        assert k2 not in str(e2.args) and "[REDACTED_GEMINI_KEY]" in str(e2.args)
        assert k3 not in str(e3.args) and "[REDACTED_GEMINI_KEY]" in str(e3.args)

    def test_self_referencing_exception(self):
        """Verify self-referencing exception e.__cause__ = e terminates cleanly."""
        k = "AIzaSySelfCycleKey1234567890123456"
        e = RuntimeError(f"Self-loop: {k}")
        e.__cause__ = e

        filter_obj = SensitiveDataFilter()
        filter_obj._sanitize_exception_attributes(e)

        assert k not in str(e.args)
        assert "[REDACTED_GEMINI_KEY]" in str(e.args)

    def test_tangled_cause_and_context_graph(self):
        """
        Verify tangled directed graph of exceptions with multiple cycles:
        e1.__cause__ = e2; e1.__context__ = e3;
        e2.__context__ = e1; e3.__cause__ = e2; e3.__context__ = e1
        """
        keys = [f"AIzaSyTangledKey{i}12345678901234567" for i in range(4)]
        e1 = ValueError(f"Tangled 1: {keys[0]}")
        e2 = TypeError(f"Tangled 2: {keys[1]}")
        e3 = KeyError(f"Tangled 3: {keys[2]}")
        e4 = RuntimeError(f"Tangled 4: {keys[3]}")

        e1.__cause__ = e2
        e1.__context__ = e3
        e2.__context__ = e1
        e2.__cause__ = e4
        e3.__cause__ = e2
        e3.__context__ = e1
        e4.__cause__ = e3

        filter_obj = SensitiveDataFilter()
        filter_obj._sanitize_exception_attributes(e1)

        for i, (k, e) in enumerate(zip(keys, [e1, e2, e3, e4])):
            assert k not in str(e.args), f"Node {i} secret leaked: {k}"
            assert "[REDACTED_GEMINI_KEY]" in str(e.args)

    def test_deep_cause_chain_1500_nodes_iterative_safety(self):
        """
        EMPIRICAL CHALLENGE: Verify chain of 1500 exceptions via __cause__.
        In standard Python, recursion limit is 1000.
        A recursive implementation would fail with RecursionError.
        The worker's iterative DFS stack must handle 1500 nodes cleanly without crashing.
        """
        filter_obj = SensitiveDataFilter()
        secret_tail = "AIzaSyDeepTailKey12345678901234567"

        # Build 1500-node chain
        head = ValueError("Head of chain")
        curr = head
        for i in range(1, 1500):
            msg = f"Chain node {i}: {secret_tail}" if i == 1499 else f"Chain node {i}"
            nxt = RuntimeError(msg)
            curr.__cause__ = nxt
            curr = nxt

        # Sanitize head - must complete without RecursionError
        filter_obj._sanitize_exception_attributes(head)

        # Check that the 1500th node secret was sanitized
        assert secret_tail not in str(curr.args)
        assert "[REDACTED_GEMINI_KEY]" in str(curr.args)

    def test_deep_context_chain_1500_nodes_iterative_safety(self):
        """
        EMPIRICAL CHALLENGE: Verify chain of 1500 exceptions via __context__.
        Must execute without RecursionError and redact secrets at the leaf.
        """
        filter_obj = SensitiveDataFilter()
        secret_tail = "AIzaSyContextTailKey1234567890123"

        head = ValueError("Head of context chain")
        curr = head
        for i in range(1, 1500):
            msg = f"Context node {i}: {secret_tail}" if i == 1499 else f"Context node {i}"
            nxt = KeyError(msg)
            curr.__context__ = nxt
            curr = nxt

        filter_obj._sanitize_exception_attributes(head)

        assert secret_tail not in str(curr.args)
        assert "[REDACTED_GEMINI_KEY]" in str(curr.args)

    def test_deep_mixed_alternating_chain_1500_nodes(self):
        """Verify 1500-node chain alternating __cause__ and __context__."""
        filter_obj = SensitiveDataFilter()
        secret_tail = "AIzaSyMixedTailKey123456789012345"

        head = ValueError("Head")
        curr = head
        for i in range(1, 1500):
            msg = f"Node {i}: {secret_tail}" if i == 1499 else f"Node {i}"
            nxt = RuntimeError(msg)
            if i % 2 == 0:
                curr.__cause__ = nxt
            else:
                curr.__context__ = nxt
            curr = nxt

        filter_obj._sanitize_exception_attributes(head)

        assert secret_tail not in str(curr.args)
        assert "[REDACTED_GEMINI_KEY]" in str(curr.args)


# =============================================================================
# 4. Nested Collections in Extra & Record Args Stress Tests
# =============================================================================
class TestNestedCollectionsExtraStress:
    """Stress tests on nested collections in extra={...} and record.args."""

    def test_deeply_nested_mixed_collections_in_extra(self):
        """
        Verify deeply nested alternating dict and list structures preserve primitives
        (int, float, bool, bytes) while scrubbing sensitive strings.
        """
        gemini_key = "AIzaSyDeepNestedKey12345678901234"
        exact_sec = "ExactSecretToken12345678"
        filter_obj = SensitiveDataFilter(exact_secrets=[exact_sec])

        payload = {
            "level1": [
                {
                    "level2": {
                        "level3": [
                            {
                                "gemini": gemini_key,
                                "secret": exact_sec,
                                "int_val": 42,
                                "float_val": 3.1415,
                                "bool_true": True,
                                "bool_false": False,
                                "none_val": None,
                                "bytes_val": b"preserved_bytes",
                                "sub_list": [gemini_key, 100, 200.5, False],
                                "sub_tuple": (exact_sec, 999),
                                "sub_set": {gemini_key, 123},
                            }
                        ]
                    }
                }
            ]
        }

        rec = logging.LogRecord("test_extra", logging.INFO, "t.py", 1, "Deep extra", (), None)
        rec.nested_payload = payload

        filter_obj.filter(rec)

        target = rec.nested_payload["level1"][0]["level2"]["level3"][0]
        assert target["gemini"] == "[REDACTED_GEMINI_KEY]"
        assert target["secret"] == "[REDACTED_SECRET]"
        assert target["int_val"] == 42 and type(target["int_val"]) is int
        assert target["float_val"] == 3.1415 and type(target["float_val"]) is float
        assert target["bool_true"] is True and type(target["bool_true"]) is bool
        assert target["bool_false"] is False and type(target["bool_false"]) is bool
        assert target["none_val"] is None
        assert target["bytes_val"] == b"preserved_bytes"

        assert target["sub_list"] == ["[REDACTED_GEMINI_KEY]", 100, 200.5, False]
        assert type(target["sub_list"]) is list
        assert target["sub_tuple"] == ("[REDACTED_SECRET]", 999)
        assert type(target["sub_tuple"]) is tuple
        assert "[REDACTED_GEMINI_KEY]" in target["sub_set"]
        assert 123 in target["sub_set"]
        assert type(target["sub_set"]) is set

    def test_circular_dictionary_in_extra(self):
        """Verify self-referencing dictionary in extra={...} terminates cleanly."""
        gemini_key = "AIzaSyCircDictKey1234567890123456"
        filter_obj = SensitiveDataFilter()

        d: dict[str, Any] = {"key": gemini_key, "count": 100, "rate": 2.5}
        d["self"] = d  # Direct cycle

        rec = logging.LogRecord("test_extra", logging.INFO, "t.py", 1, "Circ dict", (), None)
        rec.data = d

        # Must not raise RecursionError
        filter_obj.filter(rec)

        assert rec.data["key"] == "[REDACTED_GEMINI_KEY]"
        assert rec.data["count"] == 100
        assert rec.data["rate"] == 2.5
        assert rec.data["self"] == "<circular_ref>"

    def test_circular_list_in_extra(self):
        """Verify self-referencing list in extra={...} terminates cleanly."""
        gemini_key = "AIzaSyCircListKey1234567890123456"
        filter_obj = SensitiveDataFilter()

        lst: list[Any] = [gemini_key, 42, 99.9]
        lst.append(lst)  # Direct cycle

        rec = logging.LogRecord("test_extra", logging.INFO, "t.py", 1, "Circ list", (), None)
        rec.items = lst

        filter_obj.filter(rec)

        assert rec.items[0] == "[REDACTED_GEMINI_KEY]"
        assert rec.items[1] == 42
        assert rec.items[2] == 99.9
        assert rec.items[3] == "<circular_ref>"

    def test_mutual_circular_dict_in_extra(self):
        """Verify mutual circular dictionary references (d1 -> d2 -> d1)."""
        gemini_key = "AIzaSyMutualCircKey12345678901234"
        filter_obj = SensitiveDataFilter()

        d1: dict[str, Any] = {"secret": gemini_key, "num": 1}
        d2: dict[str, Any] = {"status": "ok", "d1": d1}
        d1["d2"] = d2  # Mutual cycle

        rec = logging.LogRecord("test_extra", logging.INFO, "t.py", 1, "Mutual circ", (), None)
        rec.root = d1

        filter_obj.filter(rec)

        assert rec.root["secret"] == "[REDACTED_GEMINI_KEY]"
        assert rec.root["num"] == 1
        assert rec.root["d2"]["status"] == "ok"
        assert rec.root["d2"]["d1"] == "<circular_ref>"

    def test_diamond_dag_collections_no_false_positive_cycle(self):
        """
        Verify that a diamond Directed Acyclic Graph (DAG) sharing a node does NOT
        falsely report '<circular_ref>', confirming visited IDs are correctly removed on return.
        """
        gemini_key = "AIzaSyDiamondDagKey12345678901234"
        filter_obj = SensitiveDataFilter()

        shared_child = {"key": gemini_key, "val": 42}
        diamond = {
            "left": {"child": shared_child},
            "right": {"child": shared_child},
        }

        rec = logging.LogRecord("test_extra", logging.INFO, "t.py", 1, "Diamond DAG", (), None)
        rec.dag = diamond

        filter_obj.filter(rec)

        # Both paths must have their secret cleaned without either being falsely flagged as circular
        assert rec.dag["left"]["child"]["key"] == "[REDACTED_GEMINI_KEY]"
        assert rec.dag["left"]["child"]["val"] == 42
        assert rec.dag["right"]["child"]["key"] == "[REDACTED_GEMINI_KEY]"
        assert rec.dag["right"]["child"]["val"] == 42
        assert rec.dag["left"]["child"] != "<circular_ref>"
        assert rec.dag["right"]["child"] != "<circular_ref>"

    def test_unprintable_custom_object_resilience(self):
        """Verify custom objects whose __str__ raises an exception are handled safely."""
        filter_obj = SensitiveDataFilter()

        class ExplodingObject:
            def __str__(self):
                raise RuntimeError("Exploding __str__ method!")

        rec = logging.LogRecord("test_extra", logging.INFO, "t.py", 1, "Exploding obj", (), None)
        rec.bad_obj = ExplodingObject()

        filter_obj.filter(rec)
        assert rec.bad_obj == "<unprintable_object>"

    def test_circular_collections_in_record_args_tuple(self):
        """Verify circular dictionary and circular list passed within record.args tuple."""
        gemini_key = "AIzaSyArgsCircKey1234567890123456"
        filter_obj = SensitiveDataFilter()
        logger, stream, _ = make_logger("test_args_circ", filter_obj)

        d: dict[str, Any] = {"key": gemini_key, "count": 10}
        d["self"] = d
        logger.info("Circular dict: %s", d)

        l: list[Any] = [gemini_key, 99]
        l.append(l)
        logger.info("Circular list: %s", l)

        output = stream.getvalue()
        assert gemini_key not in output
        assert "[REDACTED_GEMINI_KEY]" in output
        assert "<circular_ref>" in output

    def test_custom_object_with_secrets_and_exploding_object_in_printf_args(self):
        """Verify printf args formatting with custom objects and exploding objects."""
        gemini_key = "AIzaSyCustomObjArgsKey1234567890"
        filter_obj = SensitiveDataFilter()
        logger, stream, _ = make_logger("test_custom_args", filter_obj)

        class CustomObj:
            def __str__(self):
                return f"MySecretObj({gemini_key})"

        class BadObj:
            def __str__(self):
                raise ValueError("Cannot stringify BadObj")

        logger.info("Int: %d, Custom: %s, Bad: %s", 100, CustomObj(), BadObj())

        output = stream.getvalue().strip()
        assert output == "Int: 100, Custom: MySecretObj([REDACTED_GEMINI_KEY]), Bad: <unprintable_object>"
        assert gemini_key not in output

    def test_exception_wrapping_another_exception_in_args_deeply_sanitized(self):
        """Verify that an exception wrapping another exception instance in .args is traversed and sanitized."""
        gemini_key = "AIzaSyWrappedExcKey12345678901234"
        filter_obj = SensitiveDataFilter()

        inner_exc = ValueError(f"Inner error with key: {gemini_key}")
        outer_exc = RuntimeError(inner_exc)

        rec = logging.LogRecord("test_wrapped", logging.ERROR, "t.py", 1, "Wrapped error", (), (RuntimeError, outer_exc, None))
        filter_obj.filter(rec)

        assert gemini_key not in str(inner_exc.args)
        assert "[REDACTED_GEMINI_KEY]" in str(inner_exc.args)
        assert gemini_key not in (rec.exc_text or "")
        assert "[REDACTED_GEMINI_KEY]" in (rec.exc_text or "")

    def test_deep_exception_group_hierarchy_with_chained_exceptions(self):
        """Verify nested ExceptionGroups containing chained exceptions with secrets."""
        gemini_key = "AIzaSyEgDeepChainKey123456789012"
        filter_obj = SensitiveDataFilter()

        child_cause = KeyError(f"Root cause in EG: {gemini_key}")
        child_exc = ValueError("Intermediate child in EG")
        child_exc.__cause__ = child_cause

        eg_inner = ExceptionGroup("Inner EG", [child_exc])
        eg_root = ExceptionGroup("Root EG", [eg_inner])

        rec = logging.LogRecord("test_eg_deep", logging.ERROR, "t.py", 1, "EG Deep", (), (ExceptionGroup, eg_root, None))
        filter_obj.filter(rec)

        assert gemini_key not in str(child_cause.args)
        assert "[REDACTED_GEMINI_KEY]" in str(child_cause.args)
        assert gemini_key not in (rec.exc_text or "")
        assert "[REDACTED_GEMINI_KEY]" in (rec.exc_text or "")

