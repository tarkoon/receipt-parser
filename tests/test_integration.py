"""Integration tests — requires Cloud Vision API + Ollama running.

Auto-discovers fixtures: drop a receipt image + matching *_truth.json in
tests/fixtures/ and the tests run automatically. No code changes needed.

Run with:
    conda run -n financial-aid python -m pytest tests/test_integration.py -v

These tests make real API calls (~2 Cloud Vision calls per receipt).
Check usage with: python cli.py usage
"""

import json
from pathlib import Path
from difflib import SequenceMatcher

import pytest

# Skip all tests if Cloud Vision is not configured
try:
    import os
    if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
        pytest.skip("GOOGLE_CLOUD_PROJECT not set", allow_module_level=True)
    from receipt_parser.ocr import init_cloud_vision
    init_cloud_vision()
except Exception as e:
    pytest.skip(f"Cloud Vision not available: {e}", allow_module_level=True)

from receipt_parser.pipeline import process_document

FIXTURES = Path(__file__).parent / "fixtures"


# ── Auto-discovery ────────────────────────────────────────────────────

def _discover_fixtures() -> list[tuple[str, Path, dict]]:
    """Find all receipt image + truth.json pairs in fixtures/."""
    fixtures = []
    for truth_file in sorted(FIXTURES.glob("*_truth.json")):
        base = truth_file.stem.replace("_truth", "")
        image = None
        for ext in (".jpg", ".jpeg", ".png", ".pdf", ".tiff", ".bmp"):
            candidate = FIXTURES / f"{base}{ext}"
            if candidate.exists():
                image = candidate
                break
        if image is None:
            continue
        truth = json.loads(truth_file.read_text(encoding="utf-8"))
        fixtures.append((base, image, truth))
    return fixtures


_FIXTURES = _discover_fixtures()
_FIXTURE_IDS = [f[0] for f in _FIXTURES]

_RESULTS_CACHE: dict[str, dict] = {}


def _get_result(name: str, image: Path) -> dict:
    if name not in _RESULTS_CACHE:
        _RESULTS_CACHE[name] = process_document(image, passes=3, apply_user_rules=False)
    return _RESULTS_CACHE[name]


def _get_doc_type(truth: dict) -> str:
    return truth.get("document_type", "receipt")


# ── Common field tests (all document types) ───────────────────────────

@pytest.mark.parametrize("name,image,truth", _FIXTURES, ids=_FIXTURE_IDS)
def test_total(name, image, truth):
    result = _get_result(name, image)
    assert result.get("total") == truth.get("total"), \
        f"total: got {result.get('total')}, expected {truth.get('total')}"


@pytest.mark.parametrize("name,image,truth", _FIXTURES, ids=_FIXTURE_IDS)
def test_date(name, image, truth):
    result = _get_result(name, image)
    assert result.get("date") == truth.get("date"), \
        f"date: got {result.get('date')}, expected {truth.get('date')}"


@pytest.mark.parametrize("name,image,truth", _FIXTURES, ids=_FIXTURE_IDS)
def test_currency(name, image, truth):
    result = _get_result(name, image)
    assert result.get("currency") == truth.get("currency")


@pytest.mark.parametrize("name,image,truth", _FIXTURES, ids=_FIXTURE_IDS)
def test_payment_method(name, image, truth):
    result = _get_result(name, image)
    assert result.get("payment_method") == truth.get("payment_method"), \
        f"payment: got {result.get('payment_method')}, expected {truth.get('payment_method')}"


def _katakana_to_romaji(text: str) -> str:
    """Convert katakana to approximate romaji for cross-script comparison."""
    _MAP = {
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
    return ''.join(_MAP.get(c, c) for c in text).lower()


def _merchant_similarity(pred: str, truth: str) -> float:
    """Compare merchant names with cross-script fallback."""
    ratio = SequenceMatcher(None, pred, truth).ratio()
    if ratio >= 0.4:
        return ratio
    # Cross-script fallback: convert both to romaji and compare
    pred_r = _katakana_to_romaji(pred)
    truth_r = _katakana_to_romaji(truth)
    return max(ratio, SequenceMatcher(None, pred_r, truth_r).ratio())


@pytest.mark.parametrize("name,image,truth", _FIXTURES, ids=_FIXTURE_IDS)
def test_merchant_similarity(name, image, truth):
    result = _get_result(name, image)
    pred_m = result.get("merchant") or ""
    true_m = truth.get("merchant") or ""
    if not true_m:
        return
    ratio = _merchant_similarity(pred_m, true_m)
    assert ratio >= 0.4, \
        f"merchant: got '{pred_m}', expected '{true_m}' (similarity {ratio:.0%})"


@pytest.mark.parametrize("name,image,truth", _FIXTURES, ids=_FIXTURE_IDS)
def test_document_type(name, image, truth):
    result = _get_result(name, image)
    expected = truth.get("document_type", "receipt")
    assert result.get("document_type") == expected, \
        f"document_type: got {result.get('document_type')}, expected {expected}"


@pytest.mark.parametrize("name,image,truth", _FIXTURES, ids=_FIXTURE_IDS)
def test_amount_paid(name, image, truth):
    result = _get_result(name, image)
    expected = truth.get("amount_paid")
    if expected is None:
        return
    pred = result.get("amount_paid")
    assert pred is not None and abs(pred - expected) < 5, \
        f"amount_paid: got {pred}, expected {expected}"


# ── Receipt-specific tests ────────────────────────────────────────────

def _receipt_fixtures():
    return [(n, i, t) for n, i, t in _FIXTURES if _get_doc_type(t) == "receipt"]

_RECEIPT_FIXTURES = _receipt_fixtures()
_RECEIPT_IDS = [f[0] for f in _RECEIPT_FIXTURES]


@pytest.mark.parametrize("name,image,truth", _RECEIPT_FIXTURES, ids=_RECEIPT_IDS)
def test_subtotal(name, image, truth):
    result = _get_result(name, image)
    pred = result.get("subtotal")
    expected = truth.get("subtotal")
    if expected is None:
        assert pred is None or pred == result.get("total"), \
            f"subtotal: got {pred}, expected None or {result.get('total')}"
    else:
        assert pred == expected, f"subtotal: got {pred}, expected {expected}"


@pytest.mark.parametrize("name,image,truth", _RECEIPT_FIXTURES, ids=_RECEIPT_IDS)
def test_line_items_count(name, image, truth):
    result = _get_result(name, image)
    pred_count = len(result.get("line_items", []))
    true_count = len(truth.get("line_items", []))
    assert pred_count == true_count, \
        f"line_items: got {pred_count}, expected {true_count}"


@pytest.mark.parametrize("name,image,truth", _RECEIPT_FIXTURES, ids=_RECEIPT_IDS)
def test_line_items_totals(name, image, truth):
    result = _get_result(name, image)
    pred_totals = sorted(i.get("total", 0) for i in result.get("line_items", []))
    true_totals = sorted(i.get("total", 0) for i in truth.get("line_items", []))
    assert pred_totals == true_totals, \
        f"item totals: got {pred_totals}, expected {true_totals}"


@pytest.mark.parametrize("name,image,truth", _RECEIPT_FIXTURES, ids=_RECEIPT_IDS)
def test_tax_amount(name, image, truth):
    result = _get_result(name, image)
    pred_tax = sum(t.get("amount", 0) for t in result.get("taxes", []))
    true_tax = sum(t.get("amount", 0) for t in truth.get("taxes", []))
    assert abs(pred_tax - true_tax) < 5, \
        f"tax: got {pred_tax}, expected {true_tax}"


@pytest.mark.parametrize("name,image,truth", _RECEIPT_FIXTURES, ids=_RECEIPT_IDS)
def test_tax_categories(name, image, truth):
    result = _get_result(name, image)
    true_cats = sorted(
        i.get("tax_category", "0%") for i in truth.get("line_items", [])
    )
    if not true_cats:
        return
    pred_cats = sorted(
        i.get("tax_category", "0%") for i in result.get("line_items", [])
    )
    assert pred_cats == true_cats, \
        f"tax_categories: got {pred_cats}, expected {true_cats}"


# ── Utility bill-specific tests ───────────────────────────────────────

def _utility_fixtures():
    return [(n, i, t) for n, i, t in _FIXTURES if _get_doc_type(t) == "utility_bill"]

_UTILITY_FIXTURES = _utility_fixtures()
_UTILITY_IDS = [f[0] for f in _UTILITY_FIXTURES]


@pytest.mark.parametrize("name,image,truth", _UTILITY_FIXTURES, ids=_UTILITY_IDS)
def test_service_type(name, image, truth):
    result = _get_result(name, image)
    assert result.get("service_type") == truth.get("service_type"), \
        f"service_type: got {result.get('service_type')}, expected {truth.get('service_type')}"


@pytest.mark.parametrize("name,image,truth", _UTILITY_FIXTURES, ids=_UTILITY_IDS)
def test_usage_amount(name, image, truth):
    result = _get_result(name, image)
    true_usage = truth.get("usage") or {}
    pred_usage = result.get("usage") or {}
    expected = true_usage.get("amount")
    if expected is None:
        return
    pred = pred_usage.get("amount") if isinstance(pred_usage, dict) else None
    assert pred is not None and abs(pred - expected) < 1, \
        f"usage.amount: got {pred}, expected {expected}"


# ── Payment slip-specific tests ───────────────────────────────────────

def _slip_fixtures():
    return [(n, i, t) for n, i, t in _FIXTURES if _get_doc_type(t) == "payment_slip"]

_SLIP_FIXTURES = _slip_fixtures()
_SLIP_IDS = [f[0] for f in _SLIP_FIXTURES]


@pytest.mark.parametrize("name,image,truth", _SLIP_FIXTURES, ids=_SLIP_IDS)
def test_payer(name, image, truth):
    result = _get_result(name, image)
    expected = truth.get("payer")
    if expected is None:
        return
    pred = result.get("payer") or ""
    ratio = SequenceMatcher(None, pred, expected).ratio()
    assert ratio >= 0.4, \
        f"payer: got '{pred}', expected '{expected}' (similarity {ratio:.0%})"
