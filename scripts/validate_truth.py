"""validate_truth.py — Validate truth JSON files against schema constraints.

Run after creating or editing truth files to catch errors before they reach tests.

Usage:
    python validate_truth.py                          # fixtures dir
    python validate_truth.py receipt_14_truth.json     # specific files
    python validate_truth.py --all                     # entire project
"""

import json
import re
import sys
from pathlib import Path

# ── Allowed values (mirrors Enum Reference in new_schema.md) ─────────

DOCUMENT_TYPES = {"receipt", "utility_bill", "payment_slip"}
PAYMENT_METHODS = {"cash", "credit", "debit", "bank_payment", "WAON", None}
CURRENCIES = {"JPY", "USD"}
TAX_CATEGORIES = {"8%", "10%", "0%"}
SERVICE_TYPES = {"gas", "water", "electric", "sewage", "internet", "phone", None}
USAGE_UNITS = {"m3", "kWh", "L", None}

COMMON_FIELDS = {
    "document_type", "merchant", "date", "location", "currency",
    "total", "payment_method", "invoice_number", "account_number",
    "points_used", "amount_paid",
    "line_items", "subtotal", "taxes",
    "service_type", "billing_period", "usage",
    "payer", "payment_reference",
}

USAGE_FIELDS = {"amount", "unit", "cost_per", "meter_previous", "meter_current"}

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ── Validation logic ─────────────────────────────────────────────────

def validate_file(path: Path) -> tuple[list[str], list[str]]:
    """Validate a single truth file. Returns (errors, warnings)."""
    errors = []
    warnings = []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [f"Invalid JSON: {e}"], []

    if not isinstance(data, dict):
        return ["Root is not a JSON object"], []

    # Skip template file
    if "_llm_prompt" in data:
        return [], []

    # Unknown keys
    unknown = set(data.keys()) - COMMON_FIELDS
    for key in sorted(unknown):
        if not key.startswith("_"):
            errors.append(f'Unknown field "{key}" (typo?)')

    # document_type
    doc_type = data.get("document_type")
    if doc_type is None:
        errors.append("document_type is missing")
    elif doc_type not in DOCUMENT_TYPES:
        errors.append(f'document_type "{doc_type}" is not valid (expected: {", ".join(sorted(DOCUMENT_TYPES))})')

    # currency
    currency = data.get("currency")
    if currency is not None and currency not in CURRENCIES:
        errors.append(f'currency "{currency}" is not valid (expected: {", ".join(sorted(CURRENCIES))})')

    # total
    total = data.get("total")
    if total is None:
        warnings.append("total is null")
    elif not isinstance(total, (int, float)):
        errors.append(f"total must be a number, got {type(total).__name__}")
    elif total < 0:
        errors.append(f"total is negative: {total}")

    # date format
    date = data.get("date")
    if date is not None and not DATE_RE.match(str(date)):
        errors.append(f'date "{date}" does not match YYYY-MM-DD format')

    # payment_method
    pm = data.get("payment_method")
    if pm not in PAYMENT_METHODS:
        errors.append(f'payment_method "{pm}" is not valid (expected: {", ".join(str(v) for v in sorted(PAYMENT_METHODS, key=lambda x: str(x)))})')

    # points_used / amount_paid
    points = data.get("points_used")
    paid = data.get("amount_paid")
    if points is not None and isinstance(points, (int, float)) and points < 0:
        errors.append(f"points_used is negative: {points}")
    if paid is not None and isinstance(paid, (int, float)) and isinstance(total, (int, float)):
        if points is not None and isinstance(points, (int, float)):
            expected = total - points
            if abs(paid - expected) > 1:
                errors.append(f"amount_paid ({paid}) != total ({total}) - points_used ({points}) = {expected}")
        elif abs(paid - total) > 1:
            errors.append(f"amount_paid ({paid}) != total ({total}) but points_used is null")
    if paid is None and total is not None and points is None:
        warnings.append("amount_paid is null (should it equal total?)")

    # ── Type-specific checks ──

    if doc_type == "receipt":
        _validate_receipt(data, errors, warnings)
    elif doc_type == "utility_bill":
        _validate_utility_bill(data, errors, warnings)
    elif doc_type == "payment_slip":
        _validate_payment_slip(data, errors, warnings)

    return errors, warnings


def _validate_receipt(data: dict, errors: list, warnings: list):
    """Receipt-specific checks."""
    items = data.get("line_items", [])
    subtotal = data.get("subtotal")

    for i, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(f"line_items[{i}] is not an object")
            continue

        desc = item.get("description")
        if not desc or not isinstance(desc, str):
            errors.append(f"line_items[{i}].description is missing or empty")

        item_total = item.get("total")
        if item_total is None:
            errors.append(f"line_items[{i}].total is missing")
        elif isinstance(item_total, (int, float)) and item_total < 0:
            errors.append(f"line_items[{i}].total is negative: {item_total}")

        tc = item.get("tax_category", "0%")
        if tc not in TAX_CATEGORIES:
            errors.append(f'line_items[{i}].tax_category "{tc}" is not valid (expected: {", ".join(sorted(TAX_CATEGORIES))})')

        qty = item.get("qty", 1)
        if isinstance(qty, (int, float)) and qty <= 0:
            errors.append(f"line_items[{i}].qty must be > 0, got {qty}")

        discount = item.get("discount") or 0
        if isinstance(discount, (int, float)) and discount < 0:
            errors.append(f"line_items[{i}].discount is negative: {discount}")

        # Arithmetic: qty * unit_price - discount ≈ total
        up = item.get("unit_price")
        if (up is not None and isinstance(up, (int, float))
                and isinstance(item_total, (int, float))
                and isinstance(qty, (int, float))):
            expected = qty * up - (discount or 0)
            if abs(expected - item_total) > 1:
                errors.append(
                    f"line_items[{i}] arithmetic: {qty} × {up} - {discount} = {expected}, "
                    f"but total is {item_total}"
                )

    # subtotal ≈ sum(totals)
    if subtotal is not None and items:
        items_sum = sum(it.get("total", 0) for it in items if isinstance(it, dict))
        if isinstance(subtotal, (int, float)) and abs(items_sum - subtotal) > 2:
            errors.append(f"subtotal ({subtotal}) != sum of line item totals ({items_sum})")

    # Tax entries
    for i, tax in enumerate(data.get("taxes", [])):
        if not isinstance(tax, dict):
            errors.append(f"taxes[{i}] is not an object")
            continue
        rate = tax.get("rate", "")
        if not isinstance(rate, str):
            errors.append(f"taxes[{i}].rate must be a string")
        amount = tax.get("amount")
        if amount is not None and isinstance(amount, (int, float)) and amount < 0:
            errors.append(f"taxes[{i}].amount is negative: {amount}")

    # Cross-contamination: utility/payment fields should be null/empty
    if data.get("service_type") is not None:
        warnings.append("receipt has service_type set (should be null)")
    if data.get("billing_period") is not None:
        bp = data["billing_period"]
        if isinstance(bp, dict) and (bp.get("start") or bp.get("end")):
            warnings.append("receipt has billing_period set (should be null)")
    if data.get("payer") is not None:
        warnings.append("receipt has payer set (should be null)")


def _validate_utility_bill(data: dict, errors: list, warnings: list):
    """Utility bill-specific checks."""
    st = data.get("service_type")
    if st not in SERVICE_TYPES:
        errors.append(f'service_type "{st}" is not valid (expected: {", ".join(str(v) for v in sorted(SERVICE_TYPES, key=lambda x: str(x)))})')

    # Usage
    usage = data.get("usage")
    if usage and isinstance(usage, dict):
        unknown_usage = set(usage.keys()) - USAGE_FIELDS
        for key in sorted(unknown_usage):
            errors.append(f'Unknown usage field "{key}"')

        unit = usage.get("unit")
        if unit not in USAGE_UNITS:
            errors.append(f'usage.unit "{unit}" is not valid (expected: {", ".join(str(v) for v in sorted(USAGE_UNITS, key=lambda x: str(x)))})')

        amount = usage.get("amount")
        if amount is not None and isinstance(amount, (int, float)) and amount <= 0:
            errors.append(f"usage.amount must be > 0, got {amount}")

        prev = usage.get("meter_previous")
        curr = usage.get("meter_current")
        if (prev is not None and curr is not None
                and isinstance(prev, (int, float)) and isinstance(curr, (int, float))):
            if curr <= prev:
                errors.append(f"usage.meter_current ({curr}) <= meter_previous ({prev})")

    # Billing period
    bp = data.get("billing_period")
    if bp and isinstance(bp, dict):
        start = bp.get("start")
        end = bp.get("end")
        if start and end and start >= end:
            errors.append(f'billing_period.start ({start}) >= end ({end})')

    # line_items should be empty
    if data.get("line_items"):
        warnings.append("utility_bill has line_items (should be empty)")


def _validate_payment_slip(data: dict, errors: list, warnings: list):
    """Payment slip-specific checks."""
    if not data.get("merchant"):
        warnings.append("payment_slip has no merchant")

    if data.get("line_items"):
        warnings.append("payment_slip has line_items (should be empty)")
    if data.get("subtotal") is not None:
        warnings.append("payment_slip has subtotal (should be null)")


# ── Discovery and CLI ────────────────────────────────────────────────

def discover_truth_files(search_all: bool = False) -> list[Path]:
    """Find all truth JSON files."""
    project_root = Path(__file__).resolve().parent.parent
    files = []

    # Fixtures directory
    fixtures = project_root / "tests" / "fixtures"
    if fixtures.exists():
        files.extend(sorted(fixtures.glob("*_truth.json")))

    # Root directory truth files
    if search_all:
        # Search project root and parent
        for search_dir in [project_root, project_root.parent]:
            files.extend(sorted(search_dir.glob("receipt_*_truth.json")))
    else:
        parent = project_root.parent
        files.extend(sorted(parent.glob("receipt_*_truth.json")))

    # Deduplicate
    seen = set()
    unique = []
    for f in files:
        resolved = f.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(f)

    return unique


def main():
    args = sys.argv[1:]
    search_all = "--all" in args
    args = [a for a in args if a != "--all"]

    if args:
        files = [Path(a) for a in args]
    else:
        files = discover_truth_files(search_all=search_all)

    if not files:
        print("No truth files found.")
        sys.exit(0)

    print(f"Validating {len(files)} truth files...\n")

    total_ok = 0
    total_fail = 0
    total_warn = 0

    for path in files:
        if not path.exists():
            print(f"  {path.name} {'.' * max(1, 40 - len(path.name))} NOT FOUND")
            total_fail += 1
            continue

        file_errors, file_warnings = validate_file(path)

        name = path.name
        dots = "." * max(1, 40 - len(name))

        if file_errors:
            print(f"  {name} {dots} FAIL")
            for err in file_errors:
                print(f"    - {err}")
            total_fail += 1
        elif file_warnings:
            print(f"  {name} {dots} WARN")
            for warn in file_warnings:
                print(f"    - {warn}")
            total_warn += 1
        else:
            print(f"  {name} {dots} OK")
            total_ok += 1

    print(f"\nResults: {total_ok} OK, {total_fail} FAIL, {total_warn} WARN")
    sys.exit(1 if total_fail > 0 else 0)


if __name__ == "__main__":
    main()
