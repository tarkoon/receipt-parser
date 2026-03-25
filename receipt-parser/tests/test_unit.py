"""Fast unit tests — no cloud APIs or Ollama needed."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import cv2
import numpy as np

from schema import Receipt, generate_extraction_prompt, get_debug_color_map
from extraction import get_ollama_schema
from validation import validate_receipt
from normalization import normalize_fullwidth, clean_handwritten_ocr


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
    receipt = Receipt(
        total=324, subtotal=324,
        taxes=[{"rate": "8%", "amount": 24}],
    )
    assert validate_receipt(receipt) == []


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
    prompt = generate_extraction_prompt("test OCR text")
    assert "merchant" in prompt
    assert "PAGE" in prompt
    assert "店名" in prompt
    assert "合計" in prompt
    assert "Look for labels:" in prompt


def test_debug_color_map_has_all_fields():
    color_map = get_debug_color_map()
    expected_fields = ["merchant", "date", "location", "currency", "line_items",
                       "subtotal", "taxes", "total", "payment_method", "invoice_number"]
    for field in expected_fields:
        assert field in color_map, f"Missing field: {field}"


def test_receipt_instantiation():
    r = Receipt(total=100)
    assert r.total == 100
    assert r.merchant is None
    assert r.line_items == []


# --- Pipeline tests with mocked Cloud Vision + LLM ---

@patch("pipeline.extract_with_verification")
@patch("pipeline.run_cloud_vision")
@patch("pipeline.init_cloud_vision")
@patch("pipeline.check_ollama_available")
def test_pipeline_with_mocked_inference(mock_check, mock_init_cv, mock_ocr, mock_extract):
    mock_check.return_value = None
    mock_init_cv.return_value = MagicMock()
    mock_ocr.return_value = [
        {"text": "セブンイレブン", "confidence": 0.95, "x": 100, "y": 10,
         "bbox": [[0, 0], [200, 0], [200, 20], [0, 20]]},
        {"text": "合計 ¥150", "confidence": 0.92, "x": 100, "y": 50,
         "bbox": [[0, 40], [200, 40], [200, 60], [0, 60]]},
    ]
    mock_extract.return_value = (
        {"merchant": "セブンイレブン", "total": 150, "line_items": [],
         "taxes": [], "subtotal": None, "date": None, "currency": "JPY",
         "payment_method": None, "invoice_number": None, "location": None,
         "raw_text_summary": "convenience store receipt"},
        [{"pass": 1, "extraction": {}, "warnings": []}]
    )

    img = np.zeros((100, 200, 3), dtype=np.uint8)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        cv2.imwrite(tmp.name, img)
        tmp.close()
        from pipeline import process_document
        result = process_document(Path(tmp.name), passes=1)
    finally:
        os.unlink(tmp.name)

    assert result["merchant"] == "セブンイレブン"
    assert result.get("_pass_count", 0) >= 1
    assert "_model" in result
    assert "_line_items_reliable" in result
    assert "_pipeline_version" in result


@patch("pipeline.run_cloud_vision")
@patch("pipeline.init_cloud_vision")
@patch("pipeline.check_ollama_available")
def test_pipeline_blank_image_returns_error(mock_check, mock_init_cv, mock_ocr):
    mock_check.return_value = None
    mock_init_cv.return_value = MagicMock()
    mock_ocr.return_value = []

    img = np.zeros((100, 200, 3), dtype=np.uint8)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        cv2.imwrite(tmp.name, img)
        tmp.close()
        from pipeline import process_document
        result = process_document(Path(tmp.name), passes=1)
    finally:
        os.unlink(tmp.name)

    assert "_error" in result


# --- API usage tracking tests ---

def test_api_usage_tracking():
    from ocr import _load_usage, _save_usage, get_api_usage, _USAGE_FILE
    _save_usage({"month": "2099-01", "calls": 0})
    stats = get_api_usage()
    assert stats["calls"] == 0
    assert stats["remaining"] == 1000
    if _USAGE_FILE.exists():
        _USAGE_FILE.unlink()
