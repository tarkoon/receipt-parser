"""checks.py — Field accuracy checks shared by tests and benchmarks.

Single source of truth for all field validation logic. Used by:
- tests/test_accuracy.py (pytest integration/regression tests)
- tests/benchmark.py (robustness benchmark)
- scripts/benchmark_models.py (model comparison)
"""

from collections import Counter
from difflib import SequenceMatcher
import re
import unicodedata


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


def _numbers_close(got: object, expected: object, tolerance: float,
                   *, inclusive: bool = False) -> bool:
    """Compare extracted numeric values without accepting missing values."""
    if got is None or expected is None:
        return got is expected
    try:
        difference = abs(float(got) - float(expected))
        return difference <= tolerance if inclusive else difference < tolerance
    except (TypeError, ValueError):
        return got == expected


# ---------------------------------------------------------------------------
# Common checks (all document types)
# ---------------------------------------------------------------------------

def check_canonical_keys(result: dict, truth: dict) -> dict:
    expected = sorted(key for key in truth if not key.startswith("_"))
    missing = [key for key in expected if key not in result]
    return {"pass": not missing, "expected": expected,
            "got": sorted(key for key in expected if key in result),
            "detail": "all canonical keys present"
                      if not missing else f"missing keys: {missing}"}

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
    got = result.get("time")
    if exp is None:
        ok = got is None
        return {"pass": ok, "expected": None, "got": got,
                "detail": f"got {got}, expected null"}
    ok = _normalize_time_for_compare(got) == _normalize_time_for_compare(exp)
    return {"pass": ok, "expected": exp, "got": got,
            "detail": f"got {got}, expected {exp}"}


def _normalize_time_for_compare(value: object) -> str | None:
    if value is None:
        return None
    m = re.match(r'^\s*(\d{1,2})\s*[:：]\s*(\d{1,2})(?:\s*[:：]\s*\d{1,2})?\s*$', str(value))
    if not m:
        return str(value).strip()
    hh, mm = int(m.group(1)), int(m.group(2))
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        return str(value).strip()
    return f"{hh:02d}:{mm:02d}"


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
    got = result.get("amount_paid")
    if exp is None:
        ok = got is None
        return {"pass": ok, "expected": None, "got": got,
                "detail": f"got {got}, expected null"}
    ok = _numbers_close(got, exp, 5, inclusive=True)
    return {"pass": ok, "expected": exp, "got": got,
            "detail": f"got {got}, expected {exp} (tol +-5)"}


_MERCHANT_SIMILARITY_THRESHOLD = 0.4
_GENERIC_MERCHANT_TOKENS = frozenset({
    "cafe", "co", "coffee", "com", "company", "corp", "corporation",
    "court", "farm", "food", "inc", "jp", "limited", "ltd", "market",
    "mart", "restaurant", "shop", "store", "the", "wholesale", "www",
})


def _normalize_merchant(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(
        "".join(char if char.isalnum() else " " for char in normalized).split()
    )


def _merchant_similarity(got: str, expected: str) -> float:
    got_normalized = _normalize_merchant(got)
    expected_normalized = _normalize_merchant(expected)
    if got_normalized == expected_normalized:
        return 1.0

    got_tokens = [
        token for token in got_normalized.split()
        if token not in _GENERIC_MERCHANT_TOKENS
    ]
    expected_tokens = [
        token for token in expected_normalized.split()
        if token not in _GENERIC_MERCHANT_TOKENS
    ]
    if not got_tokens or not expected_tokens:
        return 0.0
    return max(
        fuzzy_similarity(" ".join(got_tokens), " ".join(expected_tokens)),
        fuzzy_similarity(got_tokens[0], expected_tokens[0]),
    )


def check_merchant_similarity(result: dict, truth: dict) -> dict:
    """Compare printed merchant identity at the documented 40% threshold."""
    exp = truth.get("merchant")
    if exp is None:
        got = result.get("merchant")
        ok = got is None
        return {"pass": ok, "expected": None, "got": got,
                "detail": f"got {got}, expected null"}
    got = result.get("merchant") or ""
    ratio = _merchant_similarity(got, exp)
    ok = ratio >= _MERCHANT_SIMILARITY_THRESHOLD
    return {"pass": ok, "expected": exp, "got": got,
            "similarity": round(ratio, 4),
            "detail": f"'{got}' vs '{exp}' ({ratio:.0%})"}


_LOCATION_CUE_RE = re.compile(
    r"(?:都|道|府|県|市|区|町|村|郡|倉庫店|支店|本店|店舗|ショップ|モール|センター|館|駅|空港|営業所|料金所|店|(?i:IC)(?=$))"
)
_TERMINAL_LOCATION_CUE_RE = re.compile(
    r"(?:倉庫店|支店|本店|店舗|ショップ|モール|センター|館|駅|空港|営業所|料金所|店|(?i:IC))$"
)


def _location_core(value: str) -> str:
    if not _LOCATION_CUE_RE.search(value):
        return ""
    return _LOCATION_CUE_RE.sub("", re.sub(r"[\W_]+", "", value))


def _location_core_coverage(got_core: str, expected_core: str) -> float:
    """Measure directional locality coverage after removing structural cues."""
    if len(expected_core) < 2:
        return 0.0
    return sum((Counter(got_core) & Counter(expected_core)).values()) / len(expected_core)


def _terminal_location_similarity(got: str, expected: str) -> float:
    def terminal(value: str) -> str:
        return next(
            (
                token for token in reversed(re.split(r"[\s,/|]+", value))
                if token and _TERMINAL_LOCATION_CUE_RE.search(token)
            ),
            "",
        )

    got_terminal = terminal(got)
    if not got_terminal:
        return 0.0
    return fuzzy_similarity(got_terminal, terminal(expected) or expected)


def check_location(result: dict, truth: dict) -> dict:
    exp = truth.get("location")
    if exp is None:
        got = result.get("location")
        ok = got is None
        return {"pass": ok, "expected": None, "got": got,
                "detail": f"got {got}, expected null"}
    got = result.get("location") or ""
    got_compare = unicodedata.normalize("NFKC", got).strip()
    expected_compare = unicodedata.normalize("NFKC", exp).strip()
    ratio = fuzzy_similarity(got_compare, expected_compare)
    terminal_ratio = _terminal_location_similarity(got_compare, expected_compare)
    got_core = _location_core(got_compare)
    expected_core = _location_core(expected_compare)
    core_coverage = _location_core_coverage(got_core, expected_core)
    both_structured = bool(
        _LOCATION_CUE_RE.search(got_compare)
        and _LOCATION_CUE_RE.search(expected_compare)
    )
    ok = (core_coverage >= 0.75 if both_structured
          else max(ratio, terminal_ratio) >= 0.5)
    return {"pass": ok, "expected": exp, "got": got,
            "similarity": round(ratio, 4),
            "terminal_similarity": round(terminal_ratio, 4),
            "core_coverage": round(core_coverage, 4),
            "detail": f"'{got}' vs '{exp}' ({ratio:.0%}, terminal {terminal_ratio:.0%}, core {core_coverage:.0%})"}


COMMON_CHECKS = {
    "canonical_keys": check_canonical_keys,
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
        ok = got is None
        return {"pass": ok, "expected": None, "got": got,
                "detail": f"got {got}, expected None"}
    # ±5 tolerance, matching check_tax_amount: subtotal = total − tax_sum
    # propagates 1:1 the rounding noise in tax extraction.
    try:
        ok = abs(float(got) - float(exp)) <= 5
    except (TypeError, ValueError):
        ok = got == exp
    return {"pass": ok, "expected": exp, "got": got,
            "detail": f"got {got}, expected {exp} (tol ±5)"}


def check_line_items_count(result: dict, truth: dict) -> dict:
    got = len(result.get("line_items", []))
    exp = len(truth.get("line_items", []))
    ok = got == exp
    return {"pass": ok, "expected": exp, "got": got,
            "detail": f"got {got}, expected {exp}"}


_LINE_ITEM_DEFAULTS = {
    "qty": 1,
    "unit_price": None,
    "total": None,
    "tax_category": "0%",
    "discount": 0,
    "discount_rate": "",
}


def _line_item_value(item: dict, field: str):
    value = item.get(field)
    if value is None and field in _LINE_ITEM_DEFAULTS:
        return _LINE_ITEM_DEFAULTS[field]
    return value


def _maximum_weight_assignment(weights: list[list[int]]) -> list[int]:
    """Return a deterministic maximum-weight square assignment (Hungarian)."""
    size = len(weights)
    row_potential = [0] * (size + 1)
    col_potential = [0] * (size + 1)
    col_row = [0] * (size + 1)
    previous_col = [0] * (size + 1)

    for row in range(1, size + 1):
        col_row[0] = row
        min_cost = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        col = 0
        while True:
            used[col] = True
            active_row = col_row[col]
            delta, next_col = float("inf"), 0
            for candidate_col in range(1, size + 1):
                if used[candidate_col]:
                    continue
                cost = (-weights[active_row - 1][candidate_col - 1]
                        - row_potential[active_row] - col_potential[candidate_col])
                if cost < min_cost[candidate_col]:
                    min_cost[candidate_col] = cost
                    previous_col[candidate_col] = col
                if min_cost[candidate_col] < delta:
                    delta, next_col = min_cost[candidate_col], candidate_col
            for candidate_col in range(size + 1):
                if used[candidate_col]:
                    row_potential[col_row[candidate_col]] += delta
                    col_potential[candidate_col] -= delta
                else:
                    min_cost[candidate_col] -= delta
            col = next_col
            if col_row[col] == 0:
                break
        while col:
            previous = previous_col[col]
            col_row[col] = col_row[previous]
            col = previous

    assignment = [0] * size
    for col in range(1, size + 1):
        assignment[col_row[col] - 1] = col - 1
    return assignment


def _match_line_items(result: dict, truth: dict):
    """Match rows one-to-one by description so fields keep their item owner."""
    true_items = truth.get("line_items") or []
    pred_items = result.get("line_items") or []
    ratios = [
        [
            fuzzy_similarity(
                str(true_item.get("description") or ""),
                str(pred_item.get("description") or ""),
            )
            for pred_item in pred_items
        ]
        for true_item in true_items
    ]
    size = max(len(true_items), len(pred_items))
    weights = [[0] * size for _ in range(size)]
    detail_scale = len(_LINE_ITEM_DEFAULTS) + 1
    max_detail = 1_000_000_000 * detail_scale + len(_LINE_ITEM_DEFAULTS)
    match_weight = size * max_detail + 1
    for true_idx, row in enumerate(ratios):
        for pred_idx, ratio in enumerate(row):
            if ratio < 0.5:
                continue
            field_matches = sum(
                _line_item_value(true_items[true_idx], field)
                == _line_item_value(pred_items[pred_idx], field)
                for field in _LINE_ITEM_DEFAULTS
            )
            weights[true_idx][pred_idx] = (
                match_weight + round(ratio * 1_000_000_000) * detail_scale
                + field_matches
            )
    assignment = _maximum_weight_assignment(weights) if size else []
    true_to_pred = {
        true_idx: pred_idx
        for true_idx, pred_idx in enumerate(assignment[:len(true_items)])
        if pred_idx < len(pred_items) and ratios[true_idx][pred_idx] >= 0.5
    }
    pairs = [
        (true_idx, pred_idx, ratios[true_idx][pred_idx])
        for true_idx, pred_idx in sorted(true_to_pred.items())
    ]
    unmatched_true = [idx for idx in range(len(true_items)) if idx not in true_to_pred]
    matched_pred = set(true_to_pred.values())
    unmatched_pred = [idx for idx in range(len(pred_items)) if idx not in matched_pred]
    return true_items, pred_items, pairs, unmatched_true, unmatched_pred


def _check_line_item_field(result: dict, truth: dict, field: str):
    true_items = truth.get("line_items") or []
    if not true_items:
        return {"pass": True, "detail": "no line items in truth, skipped"}

    true_items, pred_items, pairs, unmatched_true, unmatched_pred = _match_line_items(
        result, truth
    )
    true_to_pred = {true_idx: pred_idx for true_idx, pred_idx, _ in pairs}
    expected = [_line_item_value(item, field) for item in true_items]
    got = [
        _line_item_value(pred_items[true_to_pred[idx]], field)
        if idx in true_to_pred else None
        for idx in range(len(true_items))
    ]
    mismatches = [
        f"'{true_items[idx].get('description', '')}': got {got[idx]}, expected {expected[idx]}"
        for idx in range(len(true_items))
        if got[idx] != expected[idx]
    ]
    if unmatched_pred:
        extras = [pred_items[idx].get("description", "") for idx in unmatched_pred]
        mismatches.append(f"unmatched predicted rows: {extras}")
    ok = not unmatched_true and not unmatched_pred and not mismatches
    return {"pass": ok, "expected": expected, "got": got,
            "detail": "rows matched by description"
                      if ok else "; ".join(mismatches[:3])}


def check_line_items_totals(result: dict, truth: dict) -> dict:
    return _check_line_item_field(result, truth, "total")


def check_tax_amount(result: dict, truth: dict) -> dict:
    def rows(document: dict):
        values = [(t.get("rate", "unknown"), t.get("label", "") or "",
                   t.get("amount", 0))
                  for t in (document.get("taxes") or [])]

        def sort_key(row):
            try:
                return str(row[0]), str(row[1]), 0, float(row[2])
            except (TypeError, ValueError):
                return str(row[0]), str(row[1]), 1, str(row[2])

        return sorted(values, key=sort_key)

    got, exp = rows(result), rows(truth)
    ok = len(got) == len(exp) and all(
        got_row[:2] == exp_row[:2]
        and _numbers_close(got_row[2], exp_row[2], 5)
        for got_row, exp_row in zip(got, exp)
    )
    return {"pass": ok, "expected": exp, "got": got,
            "detail": "rate/label-attached amounts match (tol +-5)"
                      if ok else f"got {got}, expected {exp} by row (tol +-5)"}


def check_tax_rates(result: dict, truth: dict) -> dict:
    exp_taxes = truth.get("taxes") or []
    exp = sorted(t.get("rate", "unknown") for t in exp_taxes)
    got = sorted(t.get("rate", "unknown")
                 for t in (result.get("taxes") or []))
    ok = got == exp
    return {"pass": ok, "expected": exp, "got": got,
            "detail": f"got {got}, expected {exp}"}


def check_tax_labels(result: dict, truth: dict) -> dict:
    exp_taxes = truth.get("taxes") or []
    exp = sorted((t.get("rate", "unknown"), t.get("label", "") or "")
                 for t in exp_taxes)
    got = sorted((t.get("rate", "unknown"), t.get("label", "") or "")
                 for t in (result.get("taxes") or []))
    ok = got == exp
    return {"pass": ok, "expected": exp, "got": got,
            "detail": f"got {got}, expected {exp}"}


def check_points_used(result: dict, truth: dict) -> dict:
    exp = truth.get("points_used")
    got = result.get("points_used")
    if exp is None:
        ok = got is None
        return {"pass": ok, "expected": None, "got": got,
                "detail": f"got {got}, expected null"}
    ok = _numbers_close(got, exp, 2)
    return {"pass": ok, "expected": exp, "got": got,
            "detail": f"got {got}, expected {exp}"}


def check_line_items_qty(result: dict, truth: dict) -> dict:
    return _check_line_item_field(result, truth, "qty")


def check_line_items_unit_price(result: dict, truth: dict) -> dict:
    return _check_line_item_field(result, truth, "unit_price")


def check_tax_categories(result: dict, truth: dict) -> dict:
    return _check_line_item_field(result, truth, "tax_category")


def check_line_items_discounts(result: dict, truth: dict) -> dict:
    return _check_line_item_field(result, truth, "discount")


def check_line_items_discount_rates(result: dict, truth: dict) -> dict:
    return _check_line_item_field(result, truth, "discount_rate")


def check_item_descriptions(result: dict, truth: dict) -> dict:
    true_items = truth.get("line_items") or []
    if not true_items:
        return {"pass": True, "detail": "no line items in truth, skipped"}
    true_items, pred_items, pairs, unmatched_true, unmatched_pred = _match_line_items(
        result, truth
    )
    matched = len(pairs)
    ok = not unmatched_true and not unmatched_pred
    detail = f"{matched}/{len(true_items)} matched one-to-one"
    if unmatched_true:
        missing = [true_items[idx].get("description", "") for idx in unmatched_true]
        detail += f"; unmatched truth: {missing[:3]}"
    if unmatched_pred:
        extras = [pred_items[idx].get("description", "") for idx in unmatched_pred]
        detail += f"; unmatched predictions: {extras[:3]}"
    return {"pass": ok, "detail": detail}


RECEIPT_CHECKS = {
    "subtotal": check_subtotal,
    "line_items_count": check_line_items_count,
    "line_items_totals": check_line_items_totals,
    "line_items_qty": check_line_items_qty,
    "line_items_unit_price": check_line_items_unit_price,
    "line_items_discounts": check_line_items_discounts,
    "line_items_discount_rates": check_line_items_discount_rates,
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


def check_billing_period(result: dict, truth: dict) -> dict:
    def values(document: dict):
        period = document.get("billing_period")
        if period is None:
            return {"start": None, "end": None}
        if not isinstance(period, dict):
            return period
        return {"start": period.get("start"), "end": period.get("end")}

    got, exp = values(result), values(truth)
    ok = got == exp
    return {"pass": ok, "expected": exp, "got": got,
            "detail": f"got {got}, expected {exp}"}


def _check_usage_field(result: dict, truth: dict, field: str,
                       tolerance: float | None = None) -> dict:
    def value(document: dict):
        usage = document.get("usage")
        if usage is None:
            return None
        return usage.get(field) if isinstance(usage, dict) else usage

    got, exp = value(result), value(truth)
    ok = got == exp if tolerance is None else _numbers_close(got, exp, tolerance)
    suffix = "" if tolerance is None else f" (tol +-{tolerance:g})"
    return {"pass": ok, "expected": exp, "got": got,
            "detail": f"got {got}, expected {exp}{suffix}"}


def check_usage_amount(result: dict, truth: dict) -> dict:
    return _check_usage_field(result, truth, "amount", 1)


def check_usage_unit(result: dict, truth: dict) -> dict:
    return _check_usage_field(result, truth, "unit")


def check_usage_cost_per(result: dict, truth: dict) -> dict:
    return _check_usage_field(result, truth, "cost_per", 1)


def check_usage_meter_previous(result: dict, truth: dict) -> dict:
    return _check_usage_field(result, truth, "meter_previous", 1)


def check_usage_meter_current(result: dict, truth: dict) -> dict:
    return _check_usage_field(result, truth, "meter_current", 1)


UTILITY_CHECKS = {
    "service_type": check_service_type,
    "billing_period": check_billing_period,
    "usage_amount": check_usage_amount,
    "usage_unit": check_usage_unit,
    "usage_cost_per": check_usage_cost_per,
    "usage_meter_previous": check_usage_meter_previous,
    "usage_meter_current": check_usage_meter_current,
}

# usage_amount is also relevant for fuel receipts (volume/cost_per data)
RECEIPT_CHECKS["usage_amount"] = check_usage_amount


# ---------------------------------------------------------------------------
# Payment slip-specific checks
# ---------------------------------------------------------------------------

def check_payer(result: dict, truth: dict) -> dict:
    exp = truth.get("payer")
    got = result.get("payer")
    if exp is None:
        ok = got is None
        return {"pass": ok, "expected": None, "got": got,
                "detail": f"got {got}, expected null"}
    got = got or ""
    ratio = fuzzy_similarity(got, exp)
    ok = ratio >= 0.6
    return {"pass": ok, "expected": exp, "got": got,
            "similarity": round(ratio, 4),
            "detail": f"'{got}' vs '{exp}' ({ratio:.0%})"}


def check_account_number(result: dict, truth: dict) -> dict:
    exp = truth.get("account_number")
    got = result.get("account_number")
    if exp is None:
        ok = got is None
        return {"pass": ok, "expected": None, "got": got,
                "detail": f"got {got}, expected null"}
    ok = got == exp
    return {"pass": ok, "expected": exp, "got": got,
            "detail": f"got {got}, expected {exp}"}


def check_payment_reference(result: dict, truth: dict) -> dict:
    exp = truth.get("payment_reference")
    got = result.get("payment_reference")
    if exp is None:
        ok = got is None
        return {"pass": ok, "expected": None, "got": got,
                "detail": f"got {got}, expected null"}
    ok = got == exp
    return {"pass": ok, "expected": exp, "got": got,
            "detail": f"got {got}, expected {exp}"}


SLIP_CHECKS = {
    "payer": check_payer,
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
    truth_keys_top = {
        k for k in truth
        if not k.startswith("_") and k != "account_number"
    }

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

    return {
        "pass": True,
        "score": round(score, 4),
        "detail": f"report-only score={score:.2%} (edits={total_edits}: +{insertions} -{deletions} ~{substitutions}, truth_size={truth_size})",
    }


# Add tree_edit_distance to common checks
COMMON_CHECKS["tree_edit_distance"] = check_tree_edit_distance


# ---------------------------------------------------------------------------
# Unified accessor
# ---------------------------------------------------------------------------

ALL_CHECKS = {**COMMON_CHECKS, **RECEIPT_CHECKS, **UTILITY_CHECKS, **SLIP_CHECKS}


def get_checks_for(truth: dict) -> dict:
    """Return checks for the complete canonical document contract.

    Type-specific fields still matter when they are irrelevant: their expected
    null/empty values prevent hallucinated receipt, utility, or slip data.
    """
    checks = dict(COMMON_CHECKS)
    checks.update(RECEIPT_CHECKS)
    checks.update(UTILITY_CHECKS)
    checks.update(SLIP_CHECKS)
    return checks
