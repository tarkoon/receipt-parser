"""Validation edge cases — tax-inclusive, tolerance boundaries, empty receipts."""

from copy import deepcopy

from receipt_parser.schema import Receipt
from receipt_parser.validation import validate_receipt


def test_empty_receipt_no_warnings():
    """Empty receipt should produce no false warnings."""
    receipt = Receipt()
    assert validate_receipt(receipt) == []


def test_tax_exclusive_correct():
    """外税 (tax-exclusive): subtotal + tax = total."""
    receipt = Receipt(
        total=330, subtotal=300,
        taxes=[{"rate": "10%", "amount": 30}],
    )
    assert validate_receipt(receipt) == []


def test_tax_inclusive_correct():
    """内税 (tax-inclusive): subtotal is the pre-tax base (total - tax)."""
    receipt = Receipt(
        total=324, subtotal=300,
        taxes=[{"rate": "8%", "label": "内税", "amount": 24}],
    )
    assert validate_receipt(receipt) == []


def test_tax_rate_unusual_warns():
    """Non-standard JP tax rate should produce a warning."""
    receipt = Receipt(
        taxes=[{"rate": "15%", "amount": 45}],
    )
    warnings = validate_receipt(receipt)
    assert any("15%" in w or "Unusual" in w for w in warnings)


def test_tolerance_line_item_within_1():
    """Line item math within ±1 tolerance should not warn."""
    receipt = Receipt(
        line_items=[{"description": "item", "qty": 3, "unit_price": 33.33, "total": 100}],
    )
    # 3 * 33.33 = 99.99, total = 100, diff = 0.01 (within ±1)
    warnings = validate_receipt(receipt)
    line_warnings = [w for w in warnings if "Line" in w]
    assert len(line_warnings) == 0


def test_tolerance_subtotal_within_2():
    """Subtotal sum within ±2 tolerance should not warn."""
    receipt = Receipt(
        subtotal=301,
        line_items=[
            {"description": "a", "qty": 1, "unit_price": 150, "total": 150},
            {"description": "b", "qty": 1, "unit_price": 150, "total": 150},
        ],
    )
    # sum = 300, subtotal = 301, diff = 1 (within ±2)
    warnings = validate_receipt(receipt)
    subtotal_warnings = [w for w in warnings if "subtotal" in w.lower()]
    assert len(subtotal_warnings) == 0


def test_standard_tax_rates_no_warning():
    """Standard JP tax rates (0%, 8%, 10%) should not warn."""
    receipt = Receipt(
        taxes=[
            {"rate": "8%", "amount": 24},
            {"rate": "10%", "amount": 50},
        ],
    )
    warnings = validate_receipt(receipt)
    tax_warnings = [w for w in warnings if "tax rate" in w.lower() or "Unusual" in w]
    assert len(tax_warnings) == 0


# --- Tax ratio cross-check tests ---

def test_tax_ratio_8pct_correct():
    receipt = Receipt(total=1080, subtotal=1000, taxes=[{"rate": "8%", "amount": 80}])
    assert not any("Tax ratio" in w for w in validate_receipt(receipt))


def test_tax_ratio_10pct_correct():
    receipt = Receipt(total=1100, subtotal=1000, taxes=[{"rate": "10%", "amount": 100}])
    assert not any("Tax ratio" in w for w in validate_receipt(receipt))


def test_tax_ratio_mismatch_warns():
    receipt = Receipt(total=1500, subtotal=1000, taxes=[{"rate": "10%", "amount": 500}])
    assert any("Tax ratio" in w for w in validate_receipt(receipt))


def test_tax_ratio_with_existing_taxes():
    receipt = Receipt(total=324, subtotal=300, taxes=[{"rate": "8%", "amount": 24}])
    assert not any("Tax ratio" in w for w in validate_receipt(receipt))


def test_tax_ratio_skip_when_missing():
    r1 = Receipt(total=100, taxes=[{"rate": "8%", "amount": 8}])
    r2 = Receipt(subtotal=100, taxes=[{"rate": "8%", "amount": 8}])
    r3 = Receipt(total=100, subtotal=100)
    assert not any("Tax ratio" in w for w in validate_receipt(r1))
    assert not any("Tax ratio" in w for w in validate_receipt(r2))
    assert not any("Tax ratio" in w for w in validate_receipt(r3))


# --- Discount rate consistency tests ---

def test_discount_rate_consistency():
    receipt = Receipt(line_items=[{
        "description": "item", "qty": 1, "unit_price": 467,
        "total": 373, "discount": 94, "discount_rate": "20%",
    }])
    assert not any("discount_rate" in w for w in validate_receipt(receipt))


def test_discount_rate_inconsistent_warns():
    receipt = Receipt(line_items=[{
        "description": "item", "qty": 1, "unit_price": 200,
        "total": 100, "discount": 100, "discount_rate": "20%",
    }])
    assert any("discount_rate" in w for w in validate_receipt(receipt))


def test_multiple_issues():
    """Receipt with multiple issues should return multiple warnings."""
    receipt = Receipt(
        total=999, subtotal=100,
        line_items=[{"description": "x", "qty": 5, "unit_price": 10, "total": 100}],
        taxes=[{"rate": "8%", "amount": 8}],
    )
    # Line item: 5*10=50, not 100. Subtotal: 100 matches. Total: 100+8=108, not 999.
    warnings = validate_receipt(receipt)
    assert len(warnings) >= 2


def test_final_payload_validation_replaces_stale_warnings_after_late_mutation():
    from receipt_parser.pipeline import _validate_and_serialize_final_receipt_payload

    valid = {
        "document_type": "receipt",
        "subtotal": 100,
        "total": 108,
        "taxes": [{"rate": "8%", "amount": 8}],
    }
    invalid = {**valid, "total": 999}

    _, warnings = _validate_and_serialize_final_receipt_payload(
        invalid,
        prior_warnings=[],
    )
    assert warnings
    _, valid_warnings = _validate_and_serialize_final_receipt_payload(
        valid,
        prior_warnings=warnings,
    )
    assert valid_warnings == []


def test_user_rules_run_before_final_validation_metadata_refresh(monkeypatch):
    from receipt_parser import pipeline

    location_warning = "Location resolution failed: no supported match"
    result = {
        "document_type": "receipt",
        "location": None,
        "subtotal": 100,
        "total": 100,
        "line_items": [{
            "description": "item",
            "qty": 2,
            "unit_price": 50,
            "total": 100,
        }],
        "taxes": [],
        "_warnings": ["stale warning", location_warning],
        "_line_items_reliable": True,
        "_model": "test-model",
        "_pass_count": 1,
    }

    def mutate_after_validation(payload):
        payload["line_items"][0]["total"] = 80
        payload["_category"] = "test-category"
        return payload

    monkeypatch.setattr(pipeline, "_apply_user_rules", mutate_after_validation)
    finalized = pipeline._finalize_receipt_result(result, apply_user_rules=True)

    assert any(warning.startswith("Line 1") for warning in finalized["_warnings"])
    assert "stale warning" not in finalized["_warnings"]
    assert location_warning in finalized["_warnings"]
    assert finalized["_line_items_reliable"] is False
    assert finalized["_model"] == "test-model"
    assert finalized["_pass_count"] == 1
    assert finalized["_category"] == "test-category"


def test_finalization_returns_canonical_payload_instead_of_shadow_input():
    from receipt_parser.pipeline import _finalize_receipt_result

    location_warning = "Location resolution failed: no supported match"
    result = {
        "document_type": "receipt",
        "location": None,
        "total": 100,
        "amount_paid": 100,
        "line_items": [{
            "description": "invalid negative row",
            "qty": 1,
            "unit_price": 5,
            "total": -5,
        }],
        "taxes": [],
        "_warnings": [location_warning],
        "_model": "test-model",
        "_debug_dir": "debug/test",
    }

    finalized = _finalize_receipt_result(result, apply_user_rules=False)

    assert finalized["line_items"] == []
    assert location_warning in finalized["_warnings"]
    assert finalized["_line_items_reliable"] is True
    assert finalized["_model"] == "test-model"
    assert finalized["_debug_dir"] == "debug/test"


def test_finalization_marks_balanced_rows_unreliable_when_basket_sum_mismatches():
    from receipt_parser.pipeline import _finalize_receipt_result

    result = {
        "document_type": "receipt",
        "subtotal": 100,
        "total": 100,
        "amount_paid": 100,
        "line_items": [{
            "description": "item",
            "qty": 1,
            "unit_price": 50,
            "total": 50,
        }],
        "taxes": [],
        "_warnings": [],
    }

    finalized = _finalize_receipt_result(result, apply_user_rules=False)

    assert any(
        warning.startswith("Sum of line items")
        for warning in finalized["_warnings"]
    )
    assert finalized["_line_items_reliable"] is False


def test_injected_ocr_done_event_fires_after_finalization(monkeypatch):
    from receipt_parser import pipeline

    events = []
    finalized = False
    real_finalize = pipeline._finalize_receipt_result

    monkeypatch.setattr(pipeline, "check_model_available", lambda _model: None)
    monkeypatch.setattr(
        pipeline,
        "extract_with_verification",
        lambda *_args, **_kwargs: (
            {
                "document_type": "receipt",
                "merchant": "TEST",
                "currency": "JPY",
                "total": 100,
                "amount_paid": 100,
                "line_items": [],
                "taxes": [],
            },
            [{"pass": 1, "extraction": {}, "warnings": []}],
        ),
    )

    def finalize(*args, **kwargs):
        nonlocal finalized
        assert "done" not in events
        result = real_finalize(*args, **kwargs)
        finalized = True
        return result

    def on_stage(stage, _detail, _progress, payload=None):
        del payload
        if stage in {"validate", "done"}:
            assert finalized is True
        events.append(stage)

    monkeypatch.setattr(pipeline, "_finalize_receipt_result", finalize)

    pipeline.process_ocr_text("TEST\n合計 ¥100", on_stage=on_stage)

    assert events[-2:] == ["validate", "done"]


def test_invalid_user_rule_returns_last_valid_canonical_payload(monkeypatch):
    from receipt_parser import pipeline

    result = {
        "document_type": "receipt",
        "merchant": "TEST",
        "currency": "JPY",
        "total": 100,
        "amount_paid": 100,
        "line_items": [],
        "taxes": [],
        "_model": "test-model",
    }

    def invalidate(payload):
        payload["document_type"] = "unsupported"
        return payload

    monkeypatch.setattr(pipeline, "_apply_user_rules", invalidate)

    finalized = pipeline._finalize_receipt_result(result, apply_user_rules=True)

    assert finalized["document_type"] == "receipt"
    assert finalized["total"] == 100
    assert finalized["_model"] == "test-model"
    assert any(
        warning.startswith("Final payload validation failed")
        for warning in finalized["_warnings"]
    )


def test_final_receipt_output_repairs_are_semantically_idempotent():
    from receipt_parser.pipeline import _apply_final_receipt_output_repairs

    result = {
        "document_type": "receipt",
        "total": 2200,
        "amount_paid": 2200,
        "payment_method": "credit",
        "line_items": [],
        "taxes": [],
    }
    ocr_text = "\n".join([
        "小計",
        "¥2,018",
        "8%内税対象",
        "¥2,175",
        "税合計",
        "¥161",
        "合計",
        "¥2,179",
        "¥2,200",
        "お釣り",
        "¥21",
        "お買上点数",
        "7点",
    ])

    _apply_final_receipt_output_repairs(result, ocr_text)
    once = deepcopy(result)
    second_pass_changes = []
    _apply_final_receipt_output_repairs(
        result,
        ocr_text,
        mutation_trace=second_pass_changes,
    )

    assert result == once
    assert second_pass_changes == []
