"""validation.py — Schema-driven arithmetic & consistency checks."""

from schema import Receipt


def validate_receipt(receipt: Receipt) -> list[str]:
    """Run all arithmetic cross-checks, return list of warning strings."""
    warnings = []

    # Check line item math: qty x unit_price - discount ~ total (+-1 tolerance)
    for i, item in enumerate(receipt.line_items):
        if item.unit_price is not None and item.qty:
            expected = item.qty * item.unit_price - item.discount
            if abs(expected - item.total) > 1:
                warnings.append(
                    f"Line {i+1} ({item.description}): qty ({item.qty}) × "
                    f"unit_price ({item.unit_price}) - discount ({item.discount}) "
                    f"= {expected}, but total is {item.total}"
                )

    # Check sum of line item totals ~ subtotal (+-2 tolerance)
    if receipt.subtotal is not None and receipt.line_items:
        items_sum = sum(item.total for item in receipt.line_items)
        if abs(items_sum - receipt.subtotal) > 2:
            warnings.append(
                f"Sum of line items ({items_sum}) does not match "
                f"subtotal ({receipt.subtotal}). Review all item quantities "
                f"and unit prices — numbers in product names (e.g., size, "
                f"pack count) may have been mistaken for quantities."
            )

    # Check total vs subtotal + taxes (both inclusive and exclusive)
    if receipt.total is not None and receipt.subtotal is not None and receipt.taxes:
        tax_sum = sum(t.amount for t in receipt.taxes)

        # Scenario 1: Tax-exclusive — subtotal + tax = total
        is_exclusive = abs((receipt.subtotal + tax_sum) - receipt.total) <= 2

        # Scenario 2: Tax-inclusive — subtotal already includes tax, subtotal = total
        is_inclusive = abs(receipt.subtotal - receipt.total) <= 2

        if not (is_exclusive or is_inclusive):
            warnings.append(
                f"Total ({receipt.total}) does not match subtotal ({receipt.subtotal}) "
                f"+/- taxes ({tax_sum}) under either inclusive or exclusive tax model."
            )

    # Check JP tax rates are plausible (0%, 8%, or 10%)
    for tax in receipt.taxes:
        rate_str = tax.rate.replace('%', '').strip()
        try:
            rate_val = float(rate_str)
            if rate_val not in (0, 8, 10):
                warnings.append(
                    f"Unusual tax rate: {tax.rate} (expected 0%, 8%, or 10% for JP receipts)"
                )
        except ValueError:
            pass  # Non-numeric rate string, skip check

    return warnings
