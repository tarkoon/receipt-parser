"""validation.py — Schema-driven arithmetic & consistency checks."""

import re
from datetime import date, timedelta

from .schema import Receipt, VALID_TAX_RATES


def validate_receipt(receipt: Receipt) -> list[str]:
    """Run type-aware validation, return list of warning strings."""
    warnings = []

    doc_type = receipt.document_type

    # Common: date reasonableness check
    if receipt.date:
        warnings.extend(_check_date_reasonableness(receipt.date))

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
        # Negative total after discount is always wrong
        if item.total < 0:
            warnings.append(
                f"Line {i+1} ({item.description}): total is negative ({item.total}). "
                f"Likely OCR error in discount ({item.discount}) or unit_price."
            )

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

    # Duplicate-description warning: when 2+ items share the same normalized
    # description AND total. Real same-item-twice purchases usually print as
    # qty>1 with a single line; same desc + same total is more likely a
    # copy-paste error where the LLM duplicated a nearby item over a distinct
    # one. Normalize: strip trailing whitespace+digits (OCR sometimes leaves
    # the price embedded in the description).
    if len(receipt.line_items) >= 2:
        def _norm_desc(d: str) -> str:
            d = (d or "").strip()
            d = re.sub(r'\s+[\d,]{1,6}\s*[\*※]?\s*$', '', d)
            return d.strip()
        seen_pairs: dict[tuple, list[str]] = {}
        for item in receipt.line_items:
            norm = _norm_desc(item.description)
            key = (norm, item.total)
            seen_pairs.setdefault(key, []).append(item.description or "")
        for (desc, total), descs in seen_pairs.items():
            if len(descs) >= 2 and desc and total > 0:
                shown = list(dict.fromkeys(descs))[:3]
                warnings.append(
                    f"Duplicate item: '{desc}' appears {len(descs)} times "
                    f"with total {total} (descriptions: {shown}). If two "
                    f"adjacent items share a price, the second item likely "
                    f"has a DIFFERENT description in the OCR text — re-read "
                    f"each row carefully and use the correct OCR description."
                )

    # Check sum of line item totals matches either subtotal (pre-tax items)
    # or total (post-tax items, common on 内税 receipts where printed prices
    # already include tax). Tolerate ±2 yen rounding.
    if receipt.subtotal is not None and receipt.line_items:
        items_sum = sum(item.total for item in receipt.line_items)
        items_match_subtotal = abs(items_sum - receipt.subtotal) <= 2
        items_match_total = (receipt.total is not None
                             and abs(items_sum - receipt.total) <= 2)
        if not (items_match_subtotal or items_match_total):
            warnings.append(
                f"Sum of line items ({items_sum}) does not match "
                f"subtotal ({receipt.subtotal}) or total ({receipt.total}). "
                f"Review all item quantities and unit prices — numbers in "
                f"product names (e.g., size, pack count) may have been "
                f"mistaken for quantities."
            )

    # Universal rule: subtotal + tax_sum = total
    if receipt.total is not None and receipt.subtotal is not None and receipt.taxes:
        tax_sum = sum(t.amount for t in receipt.taxes)
        if abs((receipt.subtotal + tax_sum) - receipt.total) > 2:
            warnings.append(
                f"Total ({receipt.total}) does not match subtotal ({receipt.subtotal}) "
                f"+ taxes ({tax_sum}). Subtotal must be the pre-tax base."
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

    # Skip when there's effectively no tax (0% / non-taxable). subtotal == total
    # is then expected and there's no rate to check.
    tax_sum = sum(t.amount for t in receipt.taxes)
    if tax_sum == 0:
        return warnings

    known_rates = [0.08, 0.10]
    for rate in known_rates:
        if abs(receipt.subtotal * (1 + rate) - receipt.total) <= 2:
            return warnings  # Matches a known single rate

    # Multi-rate check: blended effective rate
    if len(receipt.taxes) > 1 and receipt.subtotal > 0:
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


def _check_date_reasonableness(date_str: str) -> list[str]:
    """Warn if the extracted date is implausibly far in the future or past."""
    warnings = []
    try:
        parsed = date.fromisoformat(date_str)
        today = date.today()
        if parsed > today + timedelta(days=365):
            warnings.append(
                f"Date ({date_str}) is more than 1 year in the future. "
                f"Possible era conversion or OCR error."
            )
        elif parsed < today - timedelta(days=5 * 365):
            warnings.append(
                f"Date ({date_str}) is more than 5 years in the past. "
                f"Possible era conversion or OCR error."
            )
    except (ValueError, TypeError):
        pass  # Malformed date strings are handled elsewhere
    return warnings
