"""validation.py — Schema-driven arithmetic & consistency checks."""

import re

from .schema import Receipt, VALID_TAX_RATES


def validate_receipt(receipt: Receipt) -> list[str]:
    """Run type-aware validation, return list of warning strings."""
    warnings = []

    doc_type = receipt.document_type

    # Common: points/amount_paid consistency
    if receipt.points_used and receipt.total:
        expected_paid = receipt.total - receipt.points_used
        if receipt.amount_paid and abs(receipt.amount_paid - expected_paid) > 1:
            warnings.append(
                f"amount_paid ({receipt.amount_paid}) != total ({receipt.total}) "
                f"- points_used ({receipt.points_used})"
            )

    if doc_type == "receipt":
        warnings.extend(_validate_receipt_fields(receipt))
    elif doc_type == "utility_bill":
        warnings.extend(_validate_utility_bill(receipt))
    elif doc_type == "payment_slip":
        warnings.extend(_validate_payment_slip(receipt))

    return warnings


def _validate_receipt_fields(receipt: Receipt) -> list[str]:
    """Receipt-specific arithmetic cross-checks."""
    warnings = []

    # Check line item math: qty x unit_price - discount ~ total (+-1 tolerance)
    for i, item in enumerate(receipt.line_items):
        if item.unit_price is not None and item.qty:
            expected = item.qty * item.unit_price - item.discount
            if abs(expected - item.total) > 1:
                warnings.append(
                    f"Line {i+1} ({item.description}): qty ({item.qty}) × "
                    f"unit_price ({item.unit_price}) - discount ({item.discount}) "
                    f"= {expected}, but total is {item.total}. "
                    f"Suggested: set total to {expected} or adjust qty/unit_price."
                )

        # Discount rate consistency: if discount_rate is set, verify discount matches
        if item.discount_rate and item.discount > 0 and item.unit_price is not None and item.qty:
            rate_match = re.match(r'(\d+(?:\.\d+)?)', item.discount_rate)
            if rate_match:
                rate_pct = float(rate_match.group(1)) / 100.0
                expected_discount = round(item.unit_price * item.qty * rate_pct)
                if abs(expected_discount - item.discount) > 2:
                    warnings.append(
                        f"Line {i+1} ({item.description}): discount_rate "
                        f"{item.discount_rate} implies discount ~{expected_discount}, "
                        f"but discount is {item.discount}."
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

        is_exclusive = abs((receipt.subtotal + tax_sum) - receipt.total) <= 2
        is_inclusive = abs(receipt.subtotal - receipt.total) <= 2

        if not (is_exclusive or is_inclusive):
            warnings.append(
                f"Total ({receipt.total}) does not match subtotal ({receipt.subtotal}) "
                f"+/- taxes ({tax_sum}) under either inclusive or exclusive tax model."
            )

    # Tax ratio cross-check: does subtotal * (1 + rate) ≈ total?
    warnings.extend(_check_tax_ratio(receipt))

    # Check JP tax rates are plausible
    valid_rate_values = set()
    for r in VALID_TAX_RATES:
        try:
            valid_rate_values.add(float(r.replace('%', '')))
        except ValueError:
            pass
    for tax in receipt.taxes:
        rate_str = tax.rate.replace('%', '').strip()
        try:
            rate_val = float(rate_str)
            if rate_val not in valid_rate_values:
                warnings.append(
                    f"Unusual tax rate: {tax.rate} (expected one of {VALID_TAX_RATES} for JP receipts)"
                )
        except ValueError:
            pass

    return warnings


def _check_tax_ratio(receipt: "Receipt") -> list[str]:
    """Check if subtotal * known tax rate ≈ total (within ±2 yen tolerance).

    Catches gross total/subtotal extraction errors before subset-sum matching.
    Skips if subtotal, total, or taxes are missing.
    Handles multi-rate receipts by computing the blended effective rate.
    """
    warnings = []
    if receipt.subtotal is None or receipt.total is None or not receipt.taxes:
        return warnings

    # Also check tax-inclusive model (subtotal == total)
    if abs(receipt.subtotal - receipt.total) <= 2:
        return warnings

    known_rates = [0.08, 0.10]
    for rate in known_rates:
        if abs(receipt.subtotal * (1 + rate) - receipt.total) <= 2:
            return warnings  # Matches a known rate, no warning

    # Multi-rate check: if multiple tax entries exist, compute blended rate
    if len(receipt.taxes) > 1 and receipt.subtotal > 0:
        tax_sum = sum(t.amount for t in receipt.taxes)
        if abs(receipt.subtotal + tax_sum - receipt.total) <= 2:
            return warnings  # Subtotal + sum of per-rate taxes ≈ total
        effective_rate = tax_sum / receipt.subtotal
        if abs(receipt.subtotal * (1 + effective_rate) - receipt.total) <= 2:
            return warnings  # Blended rate matches

    warnings.append(
        f"Tax ratio check: subtotal ({receipt.subtotal}) × known rate "
        f"does not produce total ({receipt.total}). Verify subtotal and total."
    )
    return warnings


def _validate_utility_bill(receipt: Receipt) -> list[str]:
    """Utility bill-specific checks."""
    warnings = []

    if receipt.usage:
        if receipt.usage.amount is not None and receipt.usage.amount <= 0:
            warnings.append(f"usage.amount should be positive, got {receipt.usage.amount}")
        if (receipt.usage.meter_current is not None
                and receipt.usage.meter_previous is not None
                and receipt.usage.meter_current <= receipt.usage.meter_previous):
            warnings.append(
                f"usage.meter_current ({receipt.usage.meter_current}) <= "
                f"meter_previous ({receipt.usage.meter_previous})"
            )

    if receipt.billing_period:
        if (receipt.billing_period.start and receipt.billing_period.end
                and receipt.billing_period.start >= receipt.billing_period.end):
            warnings.append(
                f"billing_period.start ({receipt.billing_period.start}) >= "
                f"end ({receipt.billing_period.end})"
            )

    if receipt.total is not None and receipt.total <= 0:
        warnings.append(f"Total should be positive, got {receipt.total}")

    return warnings


def _validate_payment_slip(receipt: Receipt) -> list[str]:
    """Payment slip-specific checks."""
    warnings = []

    if receipt.total is not None and receipt.total <= 0:
        warnings.append(f"Total should be positive, got {receipt.total}")

    if not receipt.merchant:
        warnings.append("Payment slip has no merchant (who receives the money?)")

    return warnings
