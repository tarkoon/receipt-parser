"""Structural contracts for immediate receipt postprocess trace attribution."""

import ast
from pathlib import Path

import pytest

from receipt_parser.receipt_phase_trace import (
    POSTPROCESS_PHASE_BY_NAME,
    _record_receipt_phase_mutation,
    _snapshot_receipt_mutation_fields,
)
from receipt_parser.receipt_output import _record_receipt_output_repair


POSTPROCESS_PATH = (
    Path(__file__).parents[1] / "src" / "receipt_parser" / "receipt_postprocess.py"
)


def _call_name(statement: ast.stmt) -> str | None:
    value = statement.value if isinstance(statement, (ast.Expr, ast.Assign)) else None
    call = value if isinstance(value, ast.Call) else None
    return call.func.id if call and isinstance(call.func, ast.Name) else None


def _recorded_stage(statement: ast.stmt) -> str | None:
    if _call_name(statement) != "_record_receipt_phase_mutation":
        return None
    call = statement.value
    return call.args[1].value if isinstance(call.args[1], ast.Constant) else None


def test_direct_semantic_mutations_are_immediately_traced_by_field_owner():
    tree = ast.parse(POSTPROCESS_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "postprocess_receipt"
    )

    merchant = next(
        index
        for index, statement in enumerate(function.body)
        if _call_name(statement) == "_run_merchant_identity_repair_phase"
    )
    usage = next(
        index
        for index, statement in enumerate(function.body)
        if _call_name(statement) == "_extract_fuel_usage"
    )
    masked_account = next(
        index
        for index, statement in enumerate(function.body)
        if isinstance(statement, ast.If)
        and any(
            isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == "account_number"
            and isinstance(node.ctx, ast.Store)
            for node in ast.walk(statement)
        )
    )

    expected = {
        merchant: ("header_identity_repair", "merchant"),
        masked_account: ("payment_method_repair", "account_number"),
        usage: ("service_receipt_recovery", "usage"),
    }
    for index, (stage, field) in expected.items():
        assert _recorded_stage(function.body[index + 1]) == stage
        assert field in POSTPROCESS_PHASE_BY_NAME[stage]["writes"]


def test_phase_trace_rejects_mutations_outside_declared_writes():
    receipt = {"merchant": "before", "total": 100}
    before = _snapshot_receipt_mutation_fields(receipt)
    receipt["total"] = 200

    with pytest.raises(AssertionError, match="undeclared fields.*total"):
        _record_receipt_phase_mutation(
            [],
            "header_identity_repair",
            before,
            receipt,
        )


def test_output_trace_rejects_mutations_outside_owner_writes():
    receipt = {"merchant": "before", "total": 100}

    with pytest.raises(AssertionError, match="undeclared fields.*total"):
        _record_receipt_output_repair(
            "receipt_output_merchant_identity",
            receipt,
            [],
            lambda: receipt.update(total=200),
        )


def test_top_level_postprocess_phases_record_before_the_next_phase_runs():
    tree = ast.parse(POSTPROCESS_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "postprocess_receipt"
    )
    pending_phase = None
    gaps = []
    for statement in function.body:
        call = _call_name(statement)
        if call and call.startswith("_run_"):
            if pending_phase is not None:
                gaps.append((pending_phase, call))
            pending_phase = call
        elif call == "_record_receipt_phase_mutation":
            pending_phase = None

    if pending_phase is not None:
        gaps.append((pending_phase, "function return"))
    assert not gaps, f"Missing top-level trace boundaries: {gaps}"
