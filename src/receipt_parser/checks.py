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


COMMON_CHECKS = {
    "total": check_total,
    "date": check_date,
    "currency": check_currency,
    "payment_method": check_payment_method,
    "document_type": check_document_type,
    "amount_paid": check_amount_paid,
    "merchant_similarity": check_merchant_similarity,
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
    "tax_amount": check_tax_amount,
    "tax_categories": check_tax_categories,
    "item_descriptions": check_item_descriptions,
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


SLIP_CHECKS = {
    "payer": check_payer,
}


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
