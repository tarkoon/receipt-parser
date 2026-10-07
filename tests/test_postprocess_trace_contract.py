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
from receipt_parser.receipt_output import _record_final_receipt_output_repair


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
    account = next(
        index
        for index, statement in enumerate(function.body)
        if _call_name(statement) == "_run_payment_method_repair_phase"
    )

    expected = {
        merchant: ("header_identity_repair", "merchant"),
        account: ("payment_method_repair", "account_number"),
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


@pytest.mark.parametrize(
    "stage,owner,text,helper_name",
    [
        (
            "single_rate_inclusive_tax_block",
            "single_rate_inclusive_tax_restoration",
            "10%対象 110 内消費税 10",
            "_restore_single_rate_inclusive_tax_block",
        ),
        (
            "stacked_inclusive_tax_block",
            "stacked_inclusive_tax_restoration",
            "10%対象\n内消費税\n¥110)\n¥10)",
            "_restore_stacked_inclusive_tax_block",
        ),
    ],
)
def test_final_printed_inclusive_tax_restoration_traces_total_minus_tax_subtotal(
    stage, owner, text, helper_name
):
    from receipt_parser import receipt_late_repairs

    receipt = {"total": 110, "subtotal": 110, "taxes": []}
    trace = []
    helper = getattr(receipt_late_repairs, helper_name)

    _record_final_receipt_output_repair(
        stage, receipt, trace, lambda: helper(receipt, text)
    )

    assert receipt["subtotal"] == 100
    assert receipt["taxes"] == [{"rate": "10%", "label": "内税", "amount": 10}]
    assert trace[-1]["owner_phase"] == owner
    assert set(trace[-1]["changes"]) == {"subtotal", "taxes"}


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
