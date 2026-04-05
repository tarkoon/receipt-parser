"""Validation edge cases — tax-inclusive, tolerance boundaries, empty receipts."""

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
    """内税 (tax-inclusive): subtotal = total (tax is included)."""
    receipt = Receipt(
        total=324, subtotal=324,
        taxes=[{"rate": "8%", "amount": 24}],
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
