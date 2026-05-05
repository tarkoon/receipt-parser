"""Fast unit tests — no cloud APIs or Ollama needed."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import cv2
import numpy as np

from receipt_parser.schema import Receipt, generate_extraction_prompt, get_debug_color_map
from receipt_parser.llm import get_ollama_schema, _extract_confidence
from receipt_parser.validation import validate_receipt
from receipt_parser.normalize import normalize_fullwidth, clean_handwritten_ocr
from receipt_parser.ocr import compute_ocr_confidence, OCRResult
from receipt_parser.checks import check_tree_edit_distance


# --- Tree Edit Distance tests ---

def test_tree_ed_identical():
    d = {"total": 100, "merchant": "Test", "line_items": [{"description": "a", "total": 100}]}
    result = check_tree_edit_distance(d, d)
    assert result["score"] == 1.0
    assert result["pass"]


def test_tree_ed_wrong_value():
    truth = {"total": 100, "merchant": "Test"}
    result_d = {"total": 100, "merchant": "Wrong", "extra": "value"}
    result = check_tree_edit_distance(result_d, truth)
    assert result["score"] < 1.0


def test_tree_ed_missing_line_item():
    truth = {"line_items": [{"description": "a", "total": 50}, {"description": "b", "total": 50}]}
    result_d = {"line_items": [{"description": "a", "total": 50}]}
    result = check_tree_edit_distance(result_d, truth)
    assert result["score"] < 1.0


def test_tree_ed_completely_wrong():
    truth = {"total": 100, "merchant": "Store", "date": "2026-01-01"}
    result_d = {"total": 999, "merchant": "Wrong", "date": "1999-01-01"}
    result = check_tree_edit_distance(result_d, truth)
    assert result["score"] < 0.5


# --- Normalization tests ---

def test_fullwidth_digits():
    assert normalize_fullwidth("￥１，５００") == "¥1,500"


def test_fullwidth_mixed():
    assert normalize_fullwidth("２０２６年") == "2026年"


def test_yen_preserved():
    result = normalize_fullwidth("¥100")
    assert "¥" in result
    assert "100" in result


def test_handwritten_cleanup_fixes_yen_as_1():
    """The ¥ sign misread as 1: ¥3000 → 13000 should be fixed to 金額:3000."""
    text = "金額\n13000\n宗像市産後ケア"
    result = clean_handwritten_ocr(text)
    assert "金額:3000" in result
    assert "13000" not in result


def test_handwritten_cleanup_skips_printed():
    """Printed receipts (with 小計/合計) should not be cleaned."""
    text = "小計 ¥837\n合計 ¥903\n¥3000"
    result = clean_handwritten_ocr(text)
    assert result == text


# --- Validation tests ---

def test_valid_receipt_no_warnings():
    receipt = Receipt(
        total=324, subtotal=300,
        line_items=[{"description": "おにぎり", "qty": 2, "unit_price": 150, "total": 300}],
        taxes=[{"rate": "8%", "amount": 24}],
    )
    assert validate_receipt(receipt) == []


def test_bad_line_item_math():
    receipt = Receipt(
        total=300,
        line_items=[{"description": "item", "qty": 2, "unit_price": 100, "total": 300}],
    )
    warnings = validate_receipt(receipt)
    assert any("200" in w for w in warnings)


def test_bad_subtotal_plus_tax():
    receipt = Receipt(
        total=3240, subtotal=300,
        taxes=[{"rate": "8%", "amount": 24}],
    )
    warnings = validate_receipt(receipt)
    assert len(warnings) > 0


def test_tax_inclusive_no_false_warning():
    """内税 (tax-inclusive): subtotal is pre-tax base (total - tax)."""
    receipt = Receipt(
        total=324, subtotal=300,
        taxes=[{"rate": "8%", "label": "内税", "amount": 24}],
    )
    assert validate_receipt(receipt) == []


# --- OCR confidence tests ---

def test_ocr_confidence_empty():
    assert compute_ocr_confidence([]) == 0.0


def test_ocr_confidence_uniform():
    blocks = [
        {"text": "hello", "confidence": 0.95},
        {"text": "world", "confidence": 0.95},
    ]
    assert abs(compute_ocr_confidence(blocks) - 0.95) < 0.001


def test_ocr_confidence_weighted():
    blocks = [
        {"text": "a", "confidence": 0.5},      # 1 char, weight 1
        {"text": "bbbbb", "confidence": 1.0},   # 5 chars, weight 5
    ]
    expected = (0.5 * 1 + 1.0 * 5) / 6  # 0.9167
    assert abs(compute_ocr_confidence(blocks) - expected) < 0.001


# --- LLM confidence extraction tests ---

def test_extract_confidence_valid():
    data = {"merchant": "test", "_confidence": {"merchant": 0.9, "total": 0.8}}
    conf = _extract_confidence(data)
    assert conf == {"merchant": 0.9, "total": 0.8}
    assert "_confidence" not in data  # should be popped


def test_extract_confidence_invalid_values():
    data = {"_confidence": {"merchant": 1.5, "total": "bad", "date": -0.1, "subtotal": 0.7}}
    conf = _extract_confidence(data)
    assert conf == {"subtotal": 0.7}


def test_extract_confidence_missing():
    data = {"merchant": "test"}
    conf = _extract_confidence(data)
    assert conf is None


# --- Era date conversion tests ---

def test_era_reiwa():
    from receipt_parser.patterns import era_to_western_year as _era_to_western_year
    assert _era_to_western_year(8, "令和") == 2026
    assert _era_to_western_year(1, "令和") == 2019


def test_era_heisei():
    from receipt_parser.patterns import era_to_western_year as _era_to_western_year
    assert _era_to_western_year(31, "平成") == 2019
    assert _era_to_western_year(1, "平成") == 1989


def test_era_default_assumes_reiwa():
    from receipt_parser.patterns import era_to_western_year as _era_to_western_year
    assert _era_to_western_year(8) == 2026


def test_era_invalid():
    from receipt_parser.patterns import era_to_western_year as _era_to_western_year
    assert _era_to_western_year(0) is None
    assert _era_to_western_year(100) is None


# --- Pydantic coercion tests ---

def test_pydantic_coerces_quantity_alias():
    r = Receipt(**{"total": 100, "line_items": [
        {"name": "item1", "quantity": 2, "unit_price": 50, "total": 100}
    ]})
    assert r.line_items[0].description == "item1"
    assert r.line_items[0].qty == 2


def test_pydantic_coerces_tax_category():
    r = Receipt(**{"total": 100, "line_items": [
        {"description": "item", "total": 100, "tax_category": "8percent"}
    ]})
    assert r.line_items[0].tax_category == "8%"


def test_pydantic_coerces_taxes_from_number():
    r = Receipt(**{"total": 100, "taxes": 24})
    assert len(r.taxes) == 1
    assert r.taxes[0].amount == 24


def test_pydantic_coerces_string_amounts():
    r = Receipt(**{"total": "1,500", "subtotal": "1,200"})
    assert r.total == 1500.0
    assert r.subtotal == 1200.0


# --- Configurable tax rates tests ---

def test_valid_tax_rates_constant():
    from receipt_parser.schema import VALID_TAX_RATES, REDUCED_RATE, STANDARD_RATE, EXEMPT_RATE
    assert REDUCED_RATE in VALID_TAX_RATES
    assert STANDARD_RATE in VALID_TAX_RATES
    assert EXEMPT_RATE in VALID_TAX_RATES


# --- Schema tests ---

def test_receipt_model_json_schema():
    schema = Receipt.model_json_schema()
    assert "total" in schema.get("properties", {})
    assert "line_items" in schema.get("properties", {})


def test_ollama_schema_no_refs():
    schema_str = json.dumps(get_ollama_schema())
    assert "$ref" not in schema_str
    assert "$defs" not in schema_str


def test_prompt_includes_hints_and_aliases():
    system_prompt, user_prompt = generate_extraction_prompt("test OCR text")
    prompt = system_prompt + user_prompt
    assert "merchant" in prompt
    assert "PAGE" in prompt or "OCR TEXT" in prompt
    assert "店名" in prompt
    assert "合計" in prompt
    assert "Look for labels:" in prompt


def test_debug_color_map_has_all_fields():
    color_map = get_debug_color_map()
    expected_fields = ["merchant", "date", "location", "currency", "line_items",
                       "subtotal", "taxes", "total", "payment_method"]
    for field in expected_fields:
        assert field in color_map, f"Missing field: {field}"



# --- Pipeline tests with mocked Cloud Vision + LLM ---

@patch("receipt_parser.pipeline.extract_with_verification")
@patch("receipt_parser.pipeline.run_cloud_vision")
@patch("receipt_parser.pipeline.init_cloud_vision")
@patch("receipt_parser.pipeline.check_model_available")
def test_pipeline_with_mocked_inference(mock_check, mock_init_cv, mock_ocr, mock_extract):
    mock_check.return_value = None
    mock_init_cv.return_value = MagicMock()
    mock_blocks = [
        {"text": "セブンイレブン", "confidence": 0.95, "x": 100, "y": 10,
         "bbox": [[0, 0], [200, 0], [200, 20], [0, 20]]},
        {"text": "合計 ¥150", "confidence": 0.92, "x": 100, "y": 50,
         "bbox": [[0, 40], [200, 40], [200, 60], [0, 60]]},
    ]
    mock_ocr.return_value = OCRResult(
        blocks=mock_blocks, confidence=0.93, source="fresh",
        chosen_text="セブンイレブン\n合計 ¥150",
    )
    mock_extract.return_value = (
        {"merchant": "セブンイレブン", "total": 150, "line_items": [],
         "taxes": [], "subtotal": None, "date": None, "currency": "JPY",
         "payment_method": None, "location": None,
         "raw_text_summary": "convenience store receipt"},
        [{"pass": 1, "extraction": {}, "warnings": []}]
    )

    img = np.zeros((100, 200, 3), dtype=np.uint8)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        cv2.imwrite(tmp.name, img)
        tmp.close()
        from receipt_parser.pipeline import process_document
        result = process_document(Path(tmp.name), passes=1)
    finally:
        os.unlink(tmp.name)

    assert result["merchant"] == "セブンイレブン"
    assert result.get("_pass_count", 0) >= 1
    assert "_model" in result
    assert "_line_items_reliable" in result
    assert "_pipeline_version" in result


@patch("receipt_parser.pipeline.run_cloud_vision")
@patch("receipt_parser.pipeline.init_cloud_vision")
@patch("receipt_parser.pipeline.check_model_available")
def test_pipeline_blank_image_returns_error(mock_check, mock_init_cv, mock_ocr):
    mock_check.return_value = None
    mock_init_cv.return_value = MagicMock()
    mock_ocr.return_value = OCRResult(source="fresh")

    img = np.zeros((100, 200, 3), dtype=np.uint8)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        cv2.imwrite(tmp.name, img)
        tmp.close()
        from receipt_parser.pipeline import process_document
        result = process_document(Path(tmp.name), passes=1)
    finally:
        os.unlink(tmp.name)

    assert "_error" in result


# --- API usage tracking tests ---

def test_api_usage_tracking():
    from receipt_parser.usage import (
        _save, _empty_month, get_usage, sync_usage, _billing_period_month_key,
        _USAGE_FILE, _HISTORY_FILE, _SETTINGS_FILE,
    )
    from receipt_parser.ocr import get_api_usage
    month_key = _billing_period_month_key()
    _save(_empty_month(month_key))
    # Unified tracker
    stats = get_usage()
    assert stats["cloud_vision"]["calls"] == 0
    assert stats["cloud_vision"]["remaining_free"] == 1000
    assert stats["deepseek"]["calls"] == 0
    assert stats["deepseek"]["cache_hit_tokens"] == 0
    assert stats["deepseek"]["cache_miss_tokens"] == 0
    assert stats["documents"]["total_processed"] == 0
    assert stats["documents"]["unique_processed"] == 0
    # Legacy interface
    legacy = get_api_usage()
    assert legacy["calls"] == 0
    assert legacy["remaining"] == 1000
    # Sync with cache hit/miss breakdown
    sync_usage(cv_calls=42, ds_cache_hit=18_507_008, ds_cache_miss=1_019_604,
               ds_output=2_662_239, ds_calls=6959)
    stats = get_usage()
    assert stats["cloud_vision"]["calls"] == 42
    assert stats["deepseek"]["calls"] == 6959
    assert stats["deepseek"]["cache_hit_tokens"] == 18_507_008
    assert stats["deepseek"]["cache_miss_tokens"] == 1_019_604
    assert stats["deepseek"]["output_tokens"] == 2_662_239
    assert stats["deepseek"]["est_cost_usd"] > 0
    # Cleanup
    if _USAGE_FILE.exists():
        _USAGE_FILE.unlink()
    if _HISTORY_FILE.exists():
        _HISTORY_FILE.unlink()
    if _SETTINGS_FILE.exists():
        _SETTINGS_FILE.unlink()


# --- Stage callback tests ---

def test_stage_callback_fires_in_order():
    """on_stage callback should fire with monotonically increasing progress."""
    from receipt_parser.pipeline import process_ocr_text

    calls = []

    def on_stage(stage, detail, progress):
        calls.append((stage, detail, progress))

    # Use a minimal OCR text that will produce a result
    ocr_text = "マクドナルド\n2025-01-15\n合計 ¥500\nビッグマック ¥500"

    with patch("receipt_parser.pipeline.extract_with_verification") as mock_extract:
        mock_extract.return_value = (
            {"merchant": "マクドナルド", "date": "2025-01-15", "total": 500,
             "line_items": [{"description": "ビッグマック", "total": 500}],
             "currency": "JPY", "_confidence": {"overall": "high"}},
            [{"pass": 1, "extraction": {}, "warnings": []}],
        )
        process_ocr_text(ocr_text, on_stage=on_stage)

    # Should have received calls
    assert len(calls) >= 3, f"Expected at least 3 stage calls, got {len(calls)}"

    # All values should be correct types
    for stage, detail, progress in calls:
        assert isinstance(stage, str)
        assert isinstance(detail, str)
        assert isinstance(progress, float) or isinstance(progress, int)

    # Progress should be monotonically non-decreasing
    progress_values = [c[2] for c in calls]
    for i in range(1, len(progress_values)):
        assert progress_values[i] >= progress_values[i - 1], \
            f"Progress not monotonic: {progress_values}"

    # Should end with "done" at 1.0
    assert calls[-1][0] == "done"
    assert calls[-1][2] == 1.0


def test_stage_callback_none_default():
    """on_stage=None (default) should not break anything."""
    from receipt_parser.pipeline import process_ocr_text

    ocr_text = "マクドナルド\n2025-01-15\n合計 ¥500"

    with patch("receipt_parser.pipeline.extract_with_verification") as mock_extract:
        mock_extract.return_value = (
            {"merchant": "マクドナルド", "date": "2025-01-15", "total": 500,
             "currency": "JPY", "_confidence": {"overall": "high"}},
            [{"pass": 1, "extraction": {}, "warnings": []}],
        )
        # Should not raise — on_stage defaults to None
        result = process_ocr_text(ocr_text)
        assert "merchant" in result


def test_stage_callback_emits_classify_and_plan():
    """Phase 2 contract: a 4-arg callback (param named `payload`) receives the
    structured plan and classify payloads."""
    from receipt_parser.pipeline import process_ocr_text

    events: list[tuple[str, str, float, dict | None]] = []

    def on_stage(stage, detail, progress, payload=None):
        events.append((stage, detail, progress, payload))

    with patch("receipt_parser.pipeline.extract_with_verification") as mock_extract:
        mock_extract.return_value = (
            {"merchant": "マクドナルド", "date": "2025-01-15", "total": 500,
             "currency": "JPY", "line_items": [
                 {"description": "ビッグマック", "total": 500}]},
            [{"pass": 1, "extraction": {}, "warnings": []}],
        )
        process_ocr_text(
            "マクドナルド\n2025-01-15\n合計 ¥500\nビッグマック ¥500",
            on_stage=on_stage,
        )

    stages = [e[0] for e in events]
    assert "plan" in stages
    assert "classify" in stages
    assert "validate" in stages
    assert stages[-1] == "done"

    plan = next(e for e in events if e[0] == "plan")
    assert plan[3] is not None and plan[3]["page_count"] == 1
    assert plan[3]["pass_budget"] >= 1
    assert plan[3]["path"] in ("receipt", "tbd", "utility_bill", "payment_slip")

    classify = next(e for e in events if e[0] == "classify")
    assert classify[3] is not None
    assert classify[3].get("document_type") in ("receipt", "utility_bill", "payment_slip")
    # Classify payload must let the consumer finalize the step list without
    # memorizing the path table.
    assert "expected_stages" in classify[3]
    assert classify[3]["expected_stages"][-1] == "done"
    assert "will_resolve_location" in classify[3]
    assert isinstance(classify[3]["will_resolve_location"], bool)

    # Validate-stage detail should include the result preview ("N items · ¥M").
    validate_event = next(e for e in events if e[0] == "validate")
    assert "item" in validate_event[1]
    assert "500" in validate_event[1]


def test_stage_callback_legacy_three_arg_still_works():
    """3-arg callbacks must keep working; payload must NOT be passed positionally."""
    from receipt_parser.pipeline import process_ocr_text

    events: list[tuple] = []

    def on_stage(stage, detail, progress):
        events.append((stage, detail, progress))

    with patch("receipt_parser.pipeline.extract_with_verification") as mock_extract:
        mock_extract.return_value = (
            {"merchant": "マクドナルド", "total": 500, "currency": "JPY",
             "line_items": []},
            [{"pass": 1, "extraction": {}, "warnings": []}],
        )
        process_ocr_text("マクドナルド\n合計 ¥500", on_stage=on_stage)

    assert events[-1][0] == "done"
    # Legacy callback must never receive a 4th element.
    assert all(len(e) == 3 for e in events)


def test_stage_callback_default_arg_closure_pattern_preserved():
    """The CLI-style closure-capture-via-default trick (`def cb(s,d,p,_task=t)`)
    must not be misclassified as a 4-arg payload-receiving callback. The
    default-bound 4th arg is closure state, not an output channel."""
    from receipt_parser.pipeline import _callback_accepts_payload

    sentinel = object()

    def cb(stage, detail, pct, _task=sentinel):
        return _task  # would expose accidental override

    # Conservative: only the explicit `payload` name (or **kwargs) opts in.
    assert _callback_accepts_payload(cb) is False

    def cb_payload(stage, detail, progress, payload=None):
        return payload

    assert _callback_accepts_payload(cb_payload) is True

    def cb_kwargs(*args, **kwargs):
        return kwargs

    assert _callback_accepts_payload(cb_kwargs) is True


def test_stage_callback_cancellation_raises():
    """A callback returning False must abort the pipeline with PipelineCancelled."""
    from receipt_parser.pipeline import process_ocr_text, PipelineCancelled

    def on_stage(stage, detail, progress):
        if stage == "extract":
            return False  # request cancellation
        return None

    with patch("receipt_parser.pipeline.extract_with_verification") as mock_extract:
        mock_extract.return_value = (
            {"merchant": "X", "total": 1, "currency": "JPY", "line_items": []},
            [{"pass": 1, "extraction": {}, "warnings": []}],
        )
        try:
            process_ocr_text("X\n合計 ¥1", on_stage=on_stage)
        except PipelineCancelled as e:
            assert e.stage == "extract"
        else:
            raise AssertionError("Expected PipelineCancelled to be raised")


def test_stage_callback_warn_fires_on_low_block_retry():
    """The warn stage should be emitted when OCR returns <3 blocks and rotation
    retry kicks in. Verifies the engineer's example: 'OCR returned <3 blocks,
    retrying'."""
    from receipt_parser.pipeline import process_document
    from receipt_parser.ocr import OCRResult

    events: list[tuple] = []

    def on_stage(stage, detail, progress, payload=None):
        events.append((stage, detail, progress, payload))

    # Two blocks → triggers retry. Subsequent rotations also return weak results
    # (so the loop runs but doesn't early-stop).
    weak_blocks = [
        {"text": "a", "confidence": 0.5, "x": 0, "y": 0,
         "bbox": [[0, 0], [10, 0], [10, 10], [0, 10]]},
        {"text": "b", "confidence": 0.5, "x": 0, "y": 12,
         "bbox": [[0, 12], [10, 12], [10, 22], [0, 22]]},
    ]

    def fake_ocr(*_args, **_kwargs):
        return OCRResult(blocks=list(weak_blocks),
                         source="cloud_vision", retried=False)

    with tempfile.TemporaryDirectory() as tmp:
        img_path = Path(tmp) / "tiny.png"
        img = np.full((50, 50, 3), 255, dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        with patch("receipt_parser.pipeline.run_cloud_vision",
                   side_effect=fake_ocr), \
             patch("receipt_parser.pipeline.init_cloud_vision",
                   return_value=MagicMock()), \
             patch("receipt_parser.pipeline.extract_with_verification") as mock_extract:
            mock_extract.return_value = (
                {"merchant": "X", "total": 1, "currency": "JPY",
                 "line_items": []},
                [{"pass": 1, "extraction": {}, "warnings": []}],
            )
            process_document(img_path, on_stage=on_stage)

    warn_events = [e for e in events if e[0] == "warn"]
    assert warn_events, f"Expected at least one 'warn' event; got {[e[0] for e in events]}"
    # Payload should describe the reason and page index.
    payload = warn_events[0][3]
    assert payload is not None
    assert payload.get("reason") == "low_block_count"
    assert payload.get("page") == 1
