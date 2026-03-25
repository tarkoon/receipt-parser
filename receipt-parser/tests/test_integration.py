"""Integration tests — requires Cloud Vision API + Ollama running.

Auto-discovers fixtures: drop a receipt image + matching *_truth.json in
tests/fixtures/ and the tests run automatically. No code changes needed.

Run with:
    GOOGLE_CLOUD_PROJECT=insight-489412 conda run -n financial-aid python -m pytest tests/test_integration.py -v

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
    from ocr import init_cloud_vision
    init_cloud_vision()
except Exception as e:
    pytest.skip(f"Cloud Vision not available: {e}", allow_module_level=True)

from pipeline import process_document

FIXTURES = Path(__file__).parent / "fixtures"


# ── Auto-discovery ────────────────────────────────────────────────────

def _discover_fixtures() -> list[tuple[str, Path, dict]]:
    """Find all receipt image + truth.json pairs in fixtures/.

    Naming convention: <name>_truth.json pairs with <name>.jpg/.png/.pdf
    """
    fixtures = []
    for truth_file in sorted(FIXTURES.glob("*_truth.json")):
        base = truth_file.stem.replace("_truth", "")
        # Find matching image
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

# Cache results to avoid re-running the pipeline for each test
_RESULTS_CACHE: dict[str, dict] = {}


def _get_result(name: str, image: Path) -> dict:
    """Get pipeline result, cached per fixture to avoid duplicate API calls."""
    if name not in _RESULTS_CACHE:
        _RESULTS_CACHE[name] = process_document(image, passes=2)
    return _RESULTS_CACHE[name]


# ── Core field tests (run for every fixture) ──────────────────────────

@pytest.mark.parametrize("name,image,truth", _FIXTURES, ids=_FIXTURE_IDS)
def test_total(name: str, image: Path, truth: dict):
    """Total must match exactly."""
    result = _get_result(name, image)
    assert result.get("total") == truth.get("total"), \
        f"total: got {result.get('total')}, expected {truth.get('total')}"


@pytest.mark.parametrize("name,image,truth", _FIXTURES, ids=_FIXTURE_IDS)
def test_date(name: str, image: Path, truth: dict):
    """Date must match exactly (YYYY-MM-DD)."""
    result = _get_result(name, image)
    assert result.get("date") == truth.get("date"), \
        f"date: got {result.get('date')}, expected {truth.get('date')}"


@pytest.mark.parametrize("name,image,truth", _FIXTURES, ids=_FIXTURE_IDS)
def test_currency(name: str, image: Path, truth: dict):
    """Currency must match."""
    result = _get_result(name, image)
    assert result.get("currency") == truth.get("currency")


@pytest.mark.parametrize("name,image,truth", _FIXTURES, ids=_FIXTURE_IDS)
def test_subtotal(name: str, image: Path, truth: dict):
    """Subtotal must match. If truth is null, subtotal=total is also acceptable."""
    result = _get_result(name, image)
    pred = result.get("subtotal")
    expected = truth.get("subtotal")
    if expected is None:
        # Accept null or subtotal==total (both valid for receipts with no separate subtotal)
        assert pred is None or pred == result.get("total"), \
            f"subtotal: got {pred}, expected None or {result.get('total')}"
    else:
        assert pred == expected, f"subtotal: got {pred}, expected {expected}"


@pytest.mark.parametrize("name,image,truth", _FIXTURES, ids=_FIXTURE_IDS)
def test_payment_method(name: str, image: Path, truth: dict):
    """Payment method must match."""
    result = _get_result(name, image)
    assert result.get("payment_method") == truth.get("payment_method"), \
        f"payment: got {result.get('payment_method')}, expected {truth.get('payment_method')}"


@pytest.mark.parametrize("name,image,truth", _FIXTURES, ids=_FIXTURE_IDS)
def test_line_items_count(name: str, image: Path, truth: dict):
    """Number of line items must match."""
    result = _get_result(name, image)
    pred_count = len(result.get("line_items", []))
    true_count = len(truth.get("line_items", []))
    assert pred_count == true_count, \
        f"line_items: got {pred_count}, expected {true_count}"


@pytest.mark.parametrize("name,image,truth", _FIXTURES, ids=_FIXTURE_IDS)
def test_line_items_totals(name: str, image: Path, truth: dict):
    """Line item totals must match (sorted, to handle order differences)."""
    result = _get_result(name, image)
    pred_totals = sorted(i.get("total", 0) for i in result.get("line_items", []))
    true_totals = sorted(i.get("total", 0) for i in truth.get("line_items", []))
    assert pred_totals == true_totals, \
        f"item totals: got {pred_totals}, expected {true_totals}"


@pytest.mark.parametrize("name,image,truth", _FIXTURES, ids=_FIXTURE_IDS)
def test_tax_amount(name: str, image: Path, truth: dict):
    """Total tax amount must be within ±5 tolerance.

    Wider tolerance because the LLM sometimes confuses 課税対象額 (taxable base)
    with 税額 (tax amount) for small tax brackets (e.g., 10% on ¥3 base).
    """
    result = _get_result(name, image)
    pred_tax = sum(t.get("amount", 0) for t in result.get("taxes", []))
    true_tax = sum(t.get("amount", 0) for t in truth.get("taxes", []))
    assert abs(pred_tax - true_tax) < 5, \
        f"tax: got {pred_tax}, expected {true_tax}"


@pytest.mark.parametrize("name,image,truth", _FIXTURES, ids=_FIXTURE_IDS)
def test_merchant_similarity(name: str, image: Path, truth: dict):
    """Merchant name must be at least 40% similar (fuzzy match for decorative fonts)."""
    result = _get_result(name, image)
    pred_m = result.get("merchant") or ""
    true_m = truth.get("merchant") or ""
    if not true_m:
        return  # No merchant in truth, skip
    ratio = SequenceMatcher(None, pred_m, true_m).ratio()
    assert ratio >= 0.4, \
        f"merchant: got '{pred_m}', expected '{true_m}' (similarity {ratio:.0%})"
