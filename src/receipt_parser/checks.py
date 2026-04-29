"""checks.py — Field accuracy checks shared by tests and benchmarks.

Single source of truth for all field validation logic. Used by:
- tests/test_accuracy.py (pytest integration/regression tests)
- tests/benchmark.py (robustness benchmark)
- scripts/benchmark_models.py (model comparison)
"""

from difflib import SequenceMatcher


# ---------------------------------------------------------------------------
# Katakana-to-romaji for cross-script merchant comparison
# ---------------------------------------------------------------------------

_KATAKANA_MAP = {
    'ア': 'a', 'イ': 'i', 'ウ': 'u', 'エ': 'e', 'オ': 'o',
    'カ': 'ka', 'キ': 'ki', 'ク': 'ku', 'ケ': 'ke', 'コ': 'ko',
    'サ': 'sa', 'シ': 'shi', 'ス': 'su', 'セ': 'se', 'ソ': 'so',
    'タ': 'ta', 'チ': 'chi', 'ツ': 'tsu', 'テ': 'te', 'ト': 'to',
    'ナ': 'na', 'ニ': 'ni', 'ヌ': 'nu', 'ネ': 'ne', 'ノ': 'no',
    'ハ': 'ha', 'ヒ': 'hi', 'フ': 'fu', 'ヘ': 'he', 'ホ': 'ho',
    'マ': 'ma', 'ミ': 'mi', 'ム': 'mu', 'メ': 'me', 'モ': 'mo',
    'ヤ': 'ya', 'ユ': 'yu', 'ヨ': 'yo',
    'ラ': 'ra', 'リ': 'ri', 'ル': 'ru', 'レ': 're', 'ロ': 'ro',
    'ワ': 'wa', 'ヲ': 'wo', 'ン': 'n',
    'ガ': 'ga', 'ギ': 'gi', 'グ': 'gu', 'ゲ': 'ge', 'ゴ': 'go',
    'ザ': 'za', 'ジ': 'ji', 'ズ': 'zu', 'ゼ': 'ze', 'ゾ': 'zo',
    'ダ': 'da', 'ヂ': 'di', 'ヅ': 'du', 'デ': 'de', 'ド': 'do',
    'バ': 'ba', 'ビ': 'bi', 'ブ': 'bu', 'ベ': 'be', 'ボ': 'bo',
    'パ': 'pa', 'ピ': 'pi', 'プ': 'pu', 'ペ': 'pe', 'ポ': 'po',
    'ッ': '', 'ー': '', 'ャ': 'ya', 'ュ': 'yu', 'ョ': 'yo',
    'ァ': 'a', 'ィ': 'i', 'ゥ': 'u', 'ェ': 'e', 'ォ': 'o',
}


def fuzzy_similarity(a: str, b: str) -> float:
    """Compare strings with cross-script katakana fallback."""
    ratio = SequenceMatcher(None, a, b).ratio()
    if ratio >= 0.4:
        return ratio
    a_r = ''.join(_KATAKANA_MAP.get(c, c) for c in a).lower()
    b_r = ''.join(_KATAKANA_MAP.get(c, c) for c in b).lower()
    return max(ratio, SequenceMatcher(None, a_r, b_r).ratio())


# ---------------------------------------------------------------------------
# Common checks (all document types)
# ---------------------------------------------------------------------------

def check_total(result: dict, truth: dict) -> dict:
    got, exp = result.get("total"), truth.get("total")
    ok = got == exp
    return {"pass": ok, "expected": exp, "got": got,
            "detail": f"got {got}, expected {exp}"}


def check_date(result: dict, truth: dict) -> dict:
    got, exp = result.get("date"), truth.get("date")
    ok = got == exp
    return {"pass": ok, "expected": exp, "got": got,
            "detail": f"got {got}, expected {exp}"}


def check_time(result: dict, truth: dict) -> dict:
    exp = truth.get("time")
    if exp is None:
        return {"pass": True, "detail": "no time in truth, skipped"}
    got = result.get("time")
    ok = got == exp
    return {"pass": ok, "expected": exp, "got": got,
            "detail": f"got {got}, expected {exp}"}


def check_currency(result: dict, truth: dict) -> dict:
    got, exp = result.get("currency"), truth.get("currency")
    ok = got == exp
    return {"pass": ok, "expected": exp, "got": got,
            "detail": f"got {got}, expected {exp}"}


def check_payment_method(result: dict, truth: dict) -> dict:
    got, exp = result.get("payment_method"), truth.get("payment_method")
    ok = got == exp
    return {"pass": ok, "expected": exp, "got": got,
            "detail": f"got {got}, expected {exp}"}


def check_document_type(result: dict, truth: dict) -> dict:
    got = result.get("document_type")
    exp = truth.get("document_type", "receipt")
    ok = got == exp
    return {"pass": ok, "expected": exp, "got": got,
            "detail": f"got {got}, expected {exp}"}


def check_amount_paid(result: dict, truth: dict) -> dict:
    exp = truth.get("amount_paid")
    if exp is None:
        return {"pass": True, "detail": "no amount_paid in truth, skipped"}
    got = result.get("amount_paid")
    ok = got is not None and abs(got - exp) < 5
    return {"pass": ok, "expected": exp, "got": got,
            "detail": f"got {got}, expected {exp} (tol +-5)"}


def check_merchant_similarity(result: dict, truth: dict) -> dict:
    got = result.get("merchant") or ""
    exp = truth.get("merchant") or ""
    if not exp:
        return {"pass": True, "detail": "no merchant in truth, skipped"}
    ratio = fuzzy_similarity(got, exp)
    ok = ratio >= 0.4
    return {"pass": ok, "expected": exp, "got": got,
            "detail": f"'{got}' vs '{exp}' ({ratio:.0%})"}


def check_location(result: dict, truth: dict) -> dict:
    exp = truth.get("location")
    if exp is None:
        got = result.get("location")
        ok = got is None
        return {"pass": ok, "expected": None, "got": got,
                "detail": f"got {got}, expected null"}
    got = result.get("location") or ""
    ratio = fuzzy_similarity(got, exp)
    ok = ratio >= 0.5
    return {"pass": ok, "expected": exp, "got": got,
            "detail": f"'{got}' vs '{exp}' ({ratio:.0%})"}


COMMON_CHECKS = {
    "total": check_total,
    "date": check_date,
    "time": check_time,
    "currency": check_currency,
    "payment_method": check_payment_method,
    "document_type": check_document_type,
    "amount_paid": check_amount_paid,
    "merchant_similarity": check_merchant_similarity,
    "location": check_location,
}


# ---------------------------------------------------------------------------
# Receipt-specific checks
# ---------------------------------------------------------------------------

def check_subtotal(result: dict, truth: dict) -> dict:
    got = result.get("subtotal")
    exp = truth.get("subtotal")
    if exp is None:
        ok = got is None or got == result.get("total")
        return {"pass": ok, "expected": None, "got": got,
                "detail": f"got {got}, expected None or {result.get('total')}"}
    ok = got == exp
    return {"pass": ok, "expected": exp, "got": got,
            "detail": f"got {got}, expected {exp}"}


def check_line_items_count(result: dict, truth: dict) -> dict:
    got = len(result.get("line_items", []))
    exp = len(truth.get("line_items", []))
    ok = got == exp
    return {"pass": ok, "expected": exp, "got": got,
            "detail": f"got {got}, expected {exp}"}


def check_line_items_totals(result: dict, truth: dict) -> dict:
    got = sorted(i.get("total", 0) for i in result.get("line_items", []))
    exp = sorted(i.get("total", 0) for i in truth.get("line_items", []))
    ok = got == exp
    return {"pass": ok, "expected": exp, "got": got,
            "detail": f"got {got}, expected {exp}"}


def check_tax_amount(result: dict, truth: dict) -> dict:
    got = sum(t.get("amount", 0) for t in result.get("taxes", []))
    exp = sum(t.get("amount", 0) for t in truth.get("taxes", []))
    ok = abs(got - exp) < 5
    return {"pass": ok, "expected": exp, "got": got,
            "detail": f"got {got}, expected {exp} (tol +-5)"}


def check_tax_rates(result: dict, truth: dict) -> dict:
    exp_taxes = truth.get("taxes", [])
    if not exp_taxes:
        return {"pass": True, "detail": "no taxes in truth, skipped"}
    exp = sorted(t.get("rate", "unknown") for t in exp_taxes)
    got = sorted(t.get("rate", "unknown") for t in result.get("taxes", []))
    ok = got == exp
    return {"pass": ok, "expected": exp, "got": got,
            "detail": f"got {got}, expected {exp}"}


def check_tax_labels(result: dict, truth: dict) -> dict:
    exp_taxes = truth.get("taxes", [])
    if not exp_taxes:
        return {"pass": True, "detail": "no taxes in truth, skipped"}
    exp = sorted(t.get("label", "") or "" for t in exp_taxes)
    got = sorted(t.get("label", "") or "" for t in result.get("taxes", []))
    ok = got == exp
    return {"pass": ok, "expected": exp, "got": got,
            "detail": f"got {got}, expected {exp}"}


def check_points_used(result: dict, truth: dict) -> dict:
    exp = truth.get("points_used")
    if exp is None:
        return {"pass": True, "detail": "no points_used in truth, skipped"}
    got = result.get("points_used")
    ok = got is not None and abs(got - exp) < 2
    return {"pass": ok, "expected": exp, "got": got,
            "detail": f"got {got}, expected {exp}"}


def check_line_items_qty(result: dict, truth: dict) -> dict:
    true_items = truth.get("line_items", [])
    if not true_items:
        return {"pass": True, "detail": "no line items in truth, skipped"}
    exp = sorted((i.get("qty") or 1) for i in true_items)
    got = sorted((i.get("qty") or 1) for i in result.get("line_items", []))
    ok = got == exp
    return {"pass": ok, "expected": exp, "got": got,
            "detail": f"got {got}, expected {exp}"}


def check_line_items_unit_price(result: dict, truth: dict) -> dict:
    true_items = truth.get("line_items", [])
    if not true_items:
        return {"pass": True, "detail": "no line items in truth, skipped"}
    exp = sorted((i.get("unit_price") or 0) for i in true_items)
    got = sorted((i.get("unit_price") or 0) for i in result.get("line_items", []))
    ok = got == exp
    return {"pass": ok, "expected": exp, "got": got,
            "detail": f"got {got}, expected {exp}"}


def check_tax_categories(result: dict, truth: dict) -> dict:
    true_cats = sorted(
        i.get("tax_category", "0%") for i in truth.get("line_items", [])
    )
    if not true_cats:
        return {"pass": True, "detail": "no line items in truth, skipped"}
    pred_cats = sorted(
        i.get("tax_category", "0%") for i in result.get("line_items", [])
    )
    ok = pred_cats == true_cats
    return {"pass": ok, "expected": true_cats, "got": pred_cats,
            "detail": f"got {pred_cats}, expected {true_cats}"}


def check_item_descriptions(result: dict, truth: dict) -> dict:
    true_items = truth.get("line_items", [])
    pred_items = result.get("line_items", [])
    if not true_items:
        return {"pass": True, "detail": "no line items in truth, skipped"}
    true_descs = [i.get("description", "") for i in true_items]
    pred_descs = [i.get("description", "") for i in pred_items]
    matched = 0
    mismatches = []
    for td in true_descs:
        best_ratio = 0
        best_match = ""
        for pd in pred_descs:
            ratio = fuzzy_similarity(td, pd)
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = pd
        if best_ratio >= 0.5:
            matched += 1
        else:
            mismatches.append(f"'{td}' (best: '{best_match}' {best_ratio:.0%})")
    ok = matched == len(true_descs)
    detail = f"{matched}/{len(true_descs)} matched"
    if mismatches:
        detail += f"; unmatched: {', '.join(mismatches[:3])}"
    return {"pass": ok, "detail": detail}


RECEIPT_CHECKS = {
    "subtotal": check_subtotal,
    "line_items_count": check_line_items_count,
    "line_items_totals": check_line_items_totals,
    "line_items_qty": check_line_items_qty,
    "line_items_unit_price": check_line_items_unit_price,
    "tax_amount": check_tax_amount,
    "tax_rates": check_tax_rates,
    "tax_labels": check_tax_labels,
    "tax_categories": check_tax_categories,
    "item_descriptions": check_item_descriptions,
    "points_used": check_points_used,
}


# ---------------------------------------------------------------------------
# Utility bill-specific checks
# ---------------------------------------------------------------------------

def check_service_type(result: dict, truth: dict) -> dict:
    got = result.get("service_type")
    exp = truth.get("service_type")
    ok = got == exp
    return {"pass": ok, "expected": exp, "got": got,
            "detail": f"got {got}, expected {exp}"}


def check_usage_amount(result: dict, truth: dict) -> dict:
    true_usage = truth.get("usage") or {}
    exp = true_usage.get("amount")
    if exp is None:
        return {"pass": True, "detail": "no usage.amount in truth, skipped"}
    pred_usage = result.get("usage") or {}
    got = pred_usage.get("amount") if isinstance(pred_usage, dict) else None
    ok = got is not None and abs(got - exp) < 1
    return {"pass": ok, "expected": exp, "got": got,
            "detail": f"got {got}, expected {exp} (tol +-1)"}


UTILITY_CHECKS = {
    "service_type": check_service_type,
    "usage_amount": check_usage_amount,
}

# usage_amount is also relevant for fuel receipts (volume/cost_per data)
RECEIPT_CHECKS["usage_amount"] = check_usage_amount


# ---------------------------------------------------------------------------
# Payment slip-specific checks
# ---------------------------------------------------------------------------

def check_payer(result: dict, truth: dict) -> dict:
    exp = truth.get("payer")
    if exp is None:
        return {"pass": True, "detail": "no payer in truth, skipped"}
    got = result.get("payer") or ""
    ratio = fuzzy_similarity(got, exp)
    ok = ratio >= 0.4
    return {"pass": ok, "expected": exp, "got": got,
            "detail": f"'{got}' vs '{exp}' ({ratio:.0%})"}


def check_account_number(result: dict, truth: dict) -> dict:
    exp = truth.get("account_number")
    if exp is None:
        return {"pass": True, "detail": "no account_number in truth, skipped"}
    got = result.get("account_number")
    ok = got == exp
    return {"pass": ok, "expected": exp, "got": got,
            "detail": f"got {got}, expected {exp}"}


def check_payment_reference(result: dict, truth: dict) -> dict:
    exp = truth.get("payment_reference")
    if exp is None:
        return {"pass": True, "detail": "no payment_reference in truth, skipped"}
    got = result.get("payment_reference")
    ok = got == exp
    return {"pass": ok, "expected": exp, "got": got,
            "detail": f"got {got}, expected {exp}"}


SLIP_CHECKS = {
    "payer": check_payer,
    "account_number": check_account_number,
    "payment_reference": check_payment_reference,
}


# ---------------------------------------------------------------------------
# Tree Edit Distance metric
# ---------------------------------------------------------------------------

def _flatten_dict(d: dict, prefix: str = "") -> list[tuple[str, object]]:
    """Flatten a dict into a list of (key_path, value) tuples."""
    items = []
    for key, val in d.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(val, dict):
            items.extend(_flatten_dict(val, path))
        elif isinstance(val, list):
            for i, item in enumerate(val):
                if isinstance(item, dict):
                    items.extend(_flatten_dict(item, f"{path}[{i}]"))
                else:
                    items.append((f"{path}[{i}]", item))
        else:
            items.append((path, val))
    return items


def check_tree_edit_distance(result: dict, truth: dict) -> dict:
    """Compute normalized tree edit distance between result and truth.

    Measures structural correctness of the entire JSON output.
    Returns a score from 0.0 (completely different) to 1.0 (identical).
    Only compares keys present in the truth file (ignores pipeline metadata).
    """
    truth_keys_top = {k for k in truth if not k.startswith("_")}

    result_clean = {k: result.get(k) for k in truth_keys_top}
    truth_clean = {k: truth[k] for k in truth_keys_top}

    result_flat = _flatten_dict(result_clean)
    truth_flat = _flatten_dict(truth_clean)

    truth_paths = {k for k, _ in truth_flat}
    result_paths = {k for k, _ in result_flat}
    truth_map = dict(truth_flat)
    result_map = dict(result_flat)

    insertions = len(result_paths - truth_paths)
    deletions = len(truth_paths - result_paths)
    substitutions = 0
    for key in truth_paths & result_paths:
        tv, rv = truth_map[key], result_map[key]
        if isinstance(tv, (int, float)) and isinstance(rv, (int, float)):
            if abs(tv - rv) > 1:
                substitutions += 1
        elif tv != rv:
            substitutions += 1

    total_edits = insertions + deletions + substitutions
    truth_size = max(len(truth_flat), 1)
    score = max(0.0, 1.0 - total_edits / truth_size)

    ok = score >= 0.5
    return {
        "pass": ok,
        "score": round(score, 4),
        "detail": f"score={score:.2%} (edits={total_edits}: +{insertions} -{deletions} ~{substitutions}, truth_size={truth_size})",
    }


# Add tree_edit_distance to common checks
COMMON_CHECKS["tree_edit_distance"] = check_tree_edit_distance


# ---------------------------------------------------------------------------
# Unified accessor
# ---------------------------------------------------------------------------

ALL_CHECKS = {**COMMON_CHECKS, **RECEIPT_CHECKS, **UTILITY_CHECKS, **SLIP_CHECKS}


def get_checks_for(truth: dict) -> dict:
    """Return the right set of checks based on document_type in truth."""
    doc_type = truth.get("document_type", "receipt")
    checks = dict(COMMON_CHECKS)
    if doc_type == "receipt":
        checks.update(RECEIPT_CHECKS)
    elif doc_type == "utility_bill":
        checks.update(UTILITY_CHECKS)
    elif doc_type == "payment_slip":
        checks.update(SLIP_CHECKS)
    return checks
